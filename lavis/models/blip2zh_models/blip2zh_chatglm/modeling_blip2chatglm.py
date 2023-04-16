import copy
import os
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss
import warnings
from torch import Tensor, nn

from transformers import (
    PreTrainedModel,
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
            inputs_embeds.data[:, : self.config.num_query_tokens, :] = language_model_inputs
        else:
            for i, offset in enumerate(image_slot_offset):
                inputs_embeds.data[i, offset : offset + self.config.num_query_tokens, :] = (
                    language_model_inputs[i]
                )

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

    @torch.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: Union[str, Tuple[str, torch.Tensor]],
        history: List[Tuple[Union[str, Tuple[str, torch.Tensor]], str]] = [],
        num_beams=5,
        max_length=128,
        top_p=0.9,
        do_sample=True,
        temperature=1,
    ):
        device = self.device
        # 1. Prepare token ids
        images = []
        image_slots = []

        nvtokens = self.config.num_query_tokens
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
                images.append(qimg)
                image_slots.append(len(input_ids) - slot_offset)  # count from backward

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
                    images.append(qimg)
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
                images.append(qimg)
                image_slots.append(len(input_ids))  # count from backward

        if len(input_ids) >= max_length:
            # truncate
            if image_slots[-1] > max_length and image_slots[-1] - nvtokens < max_length:
                # A non-intact image slot is not allowed
                input_ids = input_ids[-(image_slots[-1] - nvtokens) :]
            else:
                input_ids = input_ids[-max_length:]
            if image_slots[-1] > max_length:
                image_slots.pop()
                images.pop()

        # 2. Prepare image embeddings
        if len(images) != 0:
            image = torch.cat(list(images), dim=0)
            vision_outputs = self.vision_model.forward(image)
            image_embeds = vision_outputs[0]
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

            vtokens = self.language_projection(query_output)
        else:
            vtokens = []

        # 3. Place image embeddings into slots
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        inputs_embeds = self.language_model.transformer.word_embeddings(input_ids)
        for slot, vimg in zip(image_slots, vtokens):
            inputs_embeds[0][-slot : -slot + nvtokens, :] = vimg

        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
        }

        for outputs in self.stream_generate(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **gen_kwargs
        ):
            outputs = outputs.tolist()[0][len(input_ids[0]) :]
            response = tokenizer.decode(outputs)
            response = self.language_model.process_response(response)
            yield response

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids,
        inputs_embeds,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        **kwargs,
    ):
        """slightly modified from chatglm implementation to support inputs_embeds

        Args:
            input_ids (_type_): _description_
            inputs_embeds (_type_): _description_
            generation_config (Optional[GenerationConfig], optional): _description_. Defaults to None.
            logits_processor (Optional[LogitsProcessorList], optional): _description_. Defaults to None.
            stopping_criteria (Optional[StoppingCriteriaList], optional): _description_. Defaults to None.
            prefix_allowed_tokens_fn (Optional[ Callable[[int, torch.Tensor], List[int]] ], optional): _description_. Defaults to None.

        Yields:
            _type_: _description_
        """
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.language_model.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_seq_length
            )
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = (
                "decoder_input_ids"
                if self.language_model.config.is_encoder_decoder
                else "input_ids"
            )
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        logits_processor = self.language_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self.language_model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self.language_model._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, inputs_embeds=inputs_embeds, **model_kwargs
            )
            # forward pass to get next token
            outputs = self.language_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            inputs_embeds = torch.cat(
                [
                    inputs_embeds,
                    self.language_model.get_input_embeddings()(next_tokens)[:, None, :],
                ],
                dim=1,
            )
            model_kwargs = self.language_model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.language_model.config.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences.mul(
                (sum(next_tokens != i for i in eos_token_id)).long()
            )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            yield input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        past: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """slightly modified from chatglm implementation to support inputs_embeds

        Args:
            input_ids (torch.LongTensor): _description_
            inputs_embeds (Optional[torch.Tensor], optional): _description_. Defaults to None.
            past (Optional[torch.Tensor], optional): _description_. Defaults to None.
            past_key_values (Optional[torch.Tensor], optional): _description_. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            position_ids (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = self.language_model.config.mask_token_id, self.language_model.config.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.language_model.config.bos_token_id) for seq in seqs]
                if self.language_model.position_encoding_2d:
                    position_ids = torch.tensor(
                        [
                            [mask_position, seq_length - context_length]
                            for mask_position, context_length in zip(
                                mask_positions, context_lengths
                            )
                        ],
                        dtype=torch.long,
                        device=input_ids.device,
                    ).unsqueeze(-1)
                else:
                    position_ids = torch.tensor(
                        [mask_position for mask_position in mask_positions],
                        dtype=torch.long,
                        device=input_ids.device,
                    ).unsqueeze(-1)

            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        else:
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                logger.warning_once(
                    f"The dtype of attention mask ({attention_mask.dtype}) is not bool"
                )
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.language_model.get_masks(input_ids, device=input_ids.device)
            if position_ids is None:
                position_ids = self.language_model.get_position_ids(
                    input_ids,
                    device=input_ids.device,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks,
                )

            if inputs_embeds is not None:
                assert input_ids.size(1) == inputs_embeds.size(
                    1
                ), f"Make sure that both input_ids ({input_ids.size(1)}) and inputs_embeds ({inputs_embeds.size(1)}) have the same length."
                return {
                    "inputs_embeds": inputs_embeds,
                    "past_key_values": past,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "past_key_values": past,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                }
