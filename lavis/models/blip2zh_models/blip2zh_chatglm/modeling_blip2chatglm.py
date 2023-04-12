import copy
from typing import Callable, List, Optional, Tuple, Union
import torch
import warnings
from torch import Tensor, nn

from transformers import (
    PreTrainedModel,
    Blip2VisionModel,
    Blip2QFormerModel,
    GenerationConfig,
)
from transformers.utils import logging
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList

from .modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    InvalidScoreLogitsProcessor,
)
from .configuration_blip2chatglm import Blip2ChatGLMConfig


logger = logging.get_logger(__name__)


class Blip2ForChatGLM(PreTrainedModel):
    def __init__(self, config: Blip2ChatGLMConfig):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model.forward(
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
        query_outputs = self.qformer.forward(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection.forward(query_output)

        return vision_outputs, query_outputs, language_model_inputs


class Blip2ChatGLM(PreTrainedModel):
    config_class = Blip2ChatGLMConfig

    def __init__(
        self,
        config: Blip2ChatGLMConfig,
        blip2: Blip2ForChatGLM,
        lm: ChatGLMForConditionalGeneration,
    ) -> None:
        super().__init__(config)
        self.blip2 = blip2
        self.language = lm

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
        device = self.blip2.device
        # 1. Prepare token ids
        images = []
        image_slots = []

        nvtokens = self.blip2.query_tokens.size(1)
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
            vision_outputs = self.blip2.vision_model.forward(image)
            image_embeds = vision_outputs[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                device
            )

            query_tokens = self.blip2.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.blip2.qformer.forward(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
            )
            query_output = query_outputs[0]

            vtokens = self.blip2.language_projection(query_output)
        else:
            vtokens = []

        # 3. Place image embeddings into slots
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        inputs_embeds = self.language.transformer.word_embeddings(input_ids)
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

        for outputs in self.mm_stream_generate(
            input_ids=input_ids, inputs_embeds=inputs_embeds, **gen_kwargs
        ):
            outputs = outputs.tolist()[0][len(input_ids[0]) :]
            response = tokenizer.decode(outputs)
            response = self.language.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    @torch.no_grad()
    def mm_stream_generate(
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
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.language.generation_config
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
                if self.language.config.is_encoder_decoder
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

        logits_processor = self.language._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self.language._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self.language._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.language.prepare_inputs_for_generation(
                input_ids, inputs_embeds=inputs_embeds, **model_kwargs
            )
            # forward pass to get next token
            outputs = self.language(
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
                    self.language.get_input_embeddings()(next_tokens)[:, None, :],
                ],
                dim=1,
            )
            model_kwargs = self.language._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.language.config.is_encoder_decoder,
            )
            unfinished_sequences = unfinished_sequences.mul(
                (sum(next_tokens != i for i in eos_token_id)).long()
            )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            yield input_ids
