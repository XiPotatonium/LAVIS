import copy
import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import warnings
from torch import Tensor, nn

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Blip2VisionModel,
    Blip2QFormerModel,
    Blip2Model,
    Blip2PreTrainedModel,
    Blip2ForConditionalGeneration,
    GenerationConfig,
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
)
from transformers.utils import logging
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList

from .modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    InvalidScoreLogitsProcessor,
)
from .configuration_blip2chatglm import Blip2ChatGLMConfig


logger = logging.get_logger(__name__)


class Blip2ChatGLMForConditionalGeneration(Blip2ForConditionalGeneration):
    config_class = Blip2ChatGLMConfig

    def __init__(self, config: Blip2ChatGLMConfig):
        Blip2PreTrainedModel.__init__(self, config)
        # NOTE: we only initialize Blip2PreTrainedModel
        # directly call super().__init__() will cause error since ChatGLM cannot be found by AutoModel

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        self.language_model = ChatGLMForConditionalGeneration(config.text_config)

        # Initialize weights and apply final processing
        # self.post_init()

    def setup_dtype(self, vision_encoder_dtype: str = "fp32", lm_dtype: str = "fp16"):
        if vision_encoder_dtype == "fp32":
            self.vision_model = self.vision_model.float()
        elif vision_encoder_dtype == "fp16":
            self.vision_model = self.vision_model.half()
        else:
            raise NotImplementedError(
                f"Unsupported vision_encoder_dtype: {vision_encoder_dtype}"
            )

        if lm_dtype == "fp32":
            self.language_model = self.language_model.float()
        elif lm_dtype == "fp16":
            self.language_model = self.language_model.half()
        elif lm_dtype == "int4":
            self.language_model = self.language_model.half().quantize(4)
        elif lm_dtype == "int8":
            self.language_model = self.language_model.half().quantize(8)
        else:
            raise NotImplementedError(f"Unsupported lm_dtype: {lm_dtype}")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        image_slot_offset: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        """_summary_

        Args:
            pixel_values (torch.FloatTensor): _description_
            input_ids (torch.FloatTensor): input_ids[:, :num_query_tokens] should be filled with tokenizer.unk_token_id
            image_slot_offset (Optional[torch.LongTensor], optional): if not set, all vtokens are placed as prefix (image_slot_offset = torch.zeros(bsz)). Defaults to None.
            attention_mask (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            output_attentions (Optional[bool], optional): _description_. Defaults to None.
            output_hidden_states (Optional[bool], optional): _description_. Defaults to None.
            labels (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            return_dict (Optional[bool], optional): _description_. Defaults to None.

        Returns:
            Union[Tuple, Blip2ForConditionalGenerationModelOutput]: _description_
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if image_slot_offset is None:
            # image as prefix
            # update data to avoid inplace operation of leaf Variable
            inputs_embeds.data[
                :, : self.config.num_query_tokens, :
            ] = language_model_inputs
        else:
            for i, offset in enumerate(image_slot_offset):
                inputs_embeds.data[
                    i, offset : offset + self.config.num_query_tokens, :
                ] = language_model_inputs[i]

        outputs = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.view(-1, self.config.text_config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    def prepare_inputs_for_chat(
        self,
        tokenizer: PreTrainedTokenizer,
        queries: List[Union[str, Tuple[str, torch.Tensor]]],
        histories: List[List[Tuple[Union[str, Tuple[str, torch.Tensor]], str]]],
        max_length: int,
    ):
        device = self.device
        nvtokens = self.config.num_query_tokens
        # 1. Prepare token ids
        all_images = []
        all_image_slots = []
        all_input_ids = []
        for query, history in zip(queries, histories):
            image_slots = []

            if history:
                input_ids = tokenizer(
                    f"[Round {len(history)}]\n问：", add_special_tokens=False
                ).input_ids
                slot_offset = len(input_ids)
                if isinstance(query, tuple):
                    qtext, qimg = query
                    # image slot, embedding will be replaced by image embeddings
                    input_ids.extend([tokenizer.unk_token_id] * nvtokens)
                else:
                    qtext = query
                    qimg = None
                input_ids += tokenizer(qtext + f"\n答：").input_ids
                if qimg is not None:
                    all_images.append(qimg)
                    image_slots.append(
                        len(input_ids) - slot_offset
                    )  # count from backward

                for ri, (q, r) in enumerate(reversed(history)):
                    if len(input_ids) >= max_length:
                        break
                    i = len(history) - ri - 1
                    cur_input_ids: List[int] = tokenizer(
                        f"[Round {i}]\n问：", add_special_tokens=False
                    ).input_ids
                    slot_offset = len(cur_input_ids)
                    if isinstance(q, tuple):
                        qtext, qimg = q
                        # image slot, embedding will be replaced by image embeddings
                        cur_input_ids.extend([tokenizer.unk_token_id] * nvtokens)
                    else:
                        qtext = q
                        qimg = None
                    cur_input_ids += tokenizer(
                        qtext + f"\n答：{r}\n", add_special_tokens=False
                    ).input_ids
                    input_ids = cur_input_ids + input_ids
                    if qimg is not None:
                        all_images.append(qimg)
                        image_slots.append(
                            len(input_ids) - slot_offset
                        )  # count from backward
            else:
                input_ids = []
                if isinstance(query, tuple):
                    qtext, qimg = query
                    # image slot, embedding will be replaced by image embeddings
                    input_ids.extend([tokenizer.unk_token_id] * nvtokens)
                else:
                    qtext = query
                    qimg = None
                input_ids += tokenizer(qtext).input_ids
                if qimg is not None:
                    all_images.append(qimg)
                    image_slots.append(len(input_ids))  # count from backward

            if len(input_ids) >= max_length:
                # truncate
                if (
                    image_slots[-1] > max_length
                    and image_slots[-1] - nvtokens < max_length
                ):
                    # A non-intact image slot is not allowed
                    input_ids = input_ids[-(image_slots[-1] - nvtokens) :]
                else:
                    input_ids = input_ids[-max_length:]
                if image_slots[-1] > max_length:
                    image_slots.pop()
                    all_images.pop()

            all_image_slots.append(image_slots)
            all_input_ids.append(input_ids)

        # 2. Prepare image embeddings
        if len(all_images) != 0:
            vision_outputs = self.vision_model.forward(torch.cat(all_images, dim=0))
            all_image_embeds = vision_outputs[0]
            indices_or_sections = [len(chunk) for chunk in all_image_slots]
            indices_or_sections = np.cumsum(indices_or_sections)
            all_vtokens = []
            # TODO: qformer not batched
            for image_embeds in torch.tensor_split(
                all_image_embeds, tuple(indices_or_sections)
            ):
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    device
                )

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.qformer.forward(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                )
                query_output = query_outputs[0]

                all_vtokens.append(self.language_projection(query_output))
        else:
            all_vtokens = None

        # 3. Place image embeddings into slots
        input_ids = (
            torch.ones(
                (len(all_input_ids), max(len(ids) for ids in all_input_ids)),
                dtype=torch.long,
            )
            * tokenizer.pad_token_id
        )
        for i, ids in enumerate(all_input_ids):
            # pad left
            input_ids[i][-len(ids) :] = torch.as_tensor(ids, dtype=torch.long)
        input_ids = input_ids.to(device)
        inputs_embeds = self.language_model.transformer.word_embeddings(input_ids)
        for i, (image_slots, vtokens) in enumerate(zip(all_image_slots, all_vtokens)):
            for slot, vimg in zip(image_slots, vtokens):
                inputs_embeds[i][-slot : -slot + nvtokens, :] = vimg

        return input_ids, inputs_embeds

    @torch.no_grad()
    def batch_chat(
        self,
        tokenizer: PreTrainedTokenizer,
        queries: List[Union[str, Tuple[str, torch.Tensor]]],
        histories: List[List[Tuple[Union[str, Tuple[str, torch.Tensor]], str]]],
        max_length: int = 2048,
        num_beams=1,
        do_sample=True,
        top_p=0.7,
        temperature=0.95,
        logits_processor=None,
        **kwargs,
    ):
        input_ids, inputs_embeds = self.prepare_inputs_for_chat(
            tokenizer, queries, histories, max_length
        )

        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }

        outputs = self.language_model.generate(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **gen_kwargs
        )
        responses = []
        for i, output in enumerate(outputs.tolist()):
            output = output[len(input_ids[i]) :]
            response = tokenizer.decode(output)
            responses.append(self.language_model.process_response(response))
        return responses

    @torch.no_grad()
    def stream_chat(
        self,
        tokenizer: PreTrainedTokenizer,
        query: Union[str, Tuple[str, torch.Tensor]],
        history: List[Tuple[Union[str, Tuple[str, torch.Tensor]], str]],
        num_beams=5,
        max_length=128,
        top_p=0.9,
        do_sample=True,
        temperature=1,
        **kwargs,
    ):
        input_ids, inputs_embeds = self.prepare_inputs_for_chat(
            tokenizer, [query], [history], max_length
        )

        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }

        for outputs in self.language_model.stream_generate(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **gen_kwargs
        ):
            outputs = outputs.tolist()[0][len(input_ids[0]) :]
            response = tokenizer.decode(outputs)
            response = self.language_model.process_response(response)
            yield response
