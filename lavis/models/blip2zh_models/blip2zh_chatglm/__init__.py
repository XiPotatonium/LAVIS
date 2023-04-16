from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import logging
from torch.cuda.amp import autocast as autocast
from lavis.common.registry import registry

from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip2zh_models.blip2zh_qformer import Blip2BaseZh
from .modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    InvalidScoreLogitsProcessor,
)
from .tokenization_chatglm import ChatGLMTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


@registry.register_model("blip2zh_chatglm")
class Blip2ZhChatGLM(Blip2BaseZh):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_chatglm6b": "configs/models/blip2zh/blip2zh_pretrain_chatglm6b.yaml",
        "vqa": "configs/models/blip2zh/blip2zh_vqa_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        lm_model="chatglm-6b",
        max_txt_len=64,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.lm_tokenizer: ChatGLMTokenizer = ChatGLMTokenizer.from_pretrained(
            lm_model, use_fast=False, trust_remote_code=True
        )
        self.lm_model: ChatGLMForConditionalGeneration = ChatGLMForConditionalGeneration.from_pretrained(
            lm_model
        ).half()  # chatglm is half
        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False

        self.lm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.lm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len

    def prepare_lm_input(self, vtokens: torch.FloatTensor, text_input: List[str], answer: Optional[List[str]]):
        bsz, nvtoken, _ = vtokens.size()
        tokenizer = self.lm_tokenizer
        device = self.device


        context_lengths = []
        sequence_lengths = []
        sequences = []
        labels = []
        if answer is None:
            # image text pair dataset
            def get_ids():
                for text in text_input:
                    a_ids = [tokenizer.unk_token_id] * nvtoken + tokenizer.encode("", add_special_tokens=False)
                    b_ids = tokenizer.encode(text, add_special_tokens=False)
                    yield a_ids, b_ids
        else:
            # QA dataset
            def get_ids():
                for text, ans in zip(text_input, answer):
                    a_ids = [tokenizer.unk_token_id] * nvtoken + tokenizer.encode(text, add_special_tokens=False)
                    b_ids = tokenizer.encode(ans, add_special_tokens=False)
                    yield a_ids, b_ids

        for a_ids, b_ids in get_ids():
            max_caption_length = self.max_txt_len - (len(a_ids) - nvtoken) - 2
            if len(b_ids) > max_caption_length:
                b_ids = b_ids[: max_caption_length]

            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
            context_length = input_ids.index(tokenizer.bos_token_id)
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            sequences.append(input_ids)
            label = input_ids.detach().clone()
            # -100 is the ignore index is CELoss
            label[:context_length] = -100
            labels.append(label)
            context_lengths.append(context_length)
            sequence_lengths.append(len(input_ids))

        # pad sequences
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        inputs_embeds = self.lm_model.transformer.word_embeddings(input_ids)
        inputs_embeds[:, :nvtoken] = vtokens
        return input_ids, labels, inputs_embeds

    def forward(self, samples):
        image = samples["image"]
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vtokens = self.lm_proj(query_output.last_hidden_state)
        # atts_vtokens = torch.ones((bsz, nvtoken), dtype=torch.long).to(device)

        input_ids, labels, inputs_embeds = self.prepare_lm_input(
            vtokens=vtokens, text_input=samples["text_input"], answer=samples.get("answer")
        )

        # with self.maybe_autocast():
        outputs = self.lm_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=labels,
        )
        loss = outputs.loss

        return {"loss": loss, "vtokens": vtokens, "logits": outputs.logits}


    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        lm_model = cfg.get("lm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 64)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            lm_model=lm_model,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
