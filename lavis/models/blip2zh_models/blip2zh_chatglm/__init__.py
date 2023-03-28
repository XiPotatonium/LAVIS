from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
from torch import nn
import logging
from torch.cuda.amp import autocast as autocast
from lavis.common.registry import registry

from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip2zh_models.blip2zh_qformer import Blip2BaseZh
from .modeling_chatglm import ChatGLMForConditionalGeneration, InvalidScoreLogitsProcessor
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


@registry.register_model("blip2zh_chatglm")
class Blip2ZhChatGLM(Blip2BaseZh):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_chatglm6b": "configs/models/blip2zh/blip2zh_pretrain_chatglm6b.yaml",
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
        lm_model="facebook/opt-2.7b",
        prompt="",
        use_prompt=False,
        max_txt_len=32,
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

        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model, use_fast=False, trust_remote_code=True)
        self.lm_model = ChatGLMForConditionalGeneration.from_pretrained(lm_model).half()          # chatglm is half
        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.lm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.lm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.lm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt1 = "[Round 0]\n问："
        self.prompt2 = f"{prompt}\n答："
        self.prompt1_ids = self.lm_tokenizer(
            # skip special tokens at the end of sentence
            self.prompt1, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        self.prompt2_ids = self.lm_tokenizer(
            self.prompt2, return_tensors="pt"           # 和输出保持一致，需要special tokens
        ).input_ids
        self.prompt1_embeds = self.lm_model.transformer.word_embeddings(self.prompt1_ids)
        self.prompt2_embeds = self.lm_model.transformer.word_embeddings(self.prompt2_ids)
        self.prompt1_length = len(self.prompt1_ids[0])
        self.prompt2_length = len(self.prompt2_ids[0])
        self.use_prompt = use_prompt
        if self.use_prompt:
            logging.info(f"Use prompt: {self.prompt1}<vtokens>{self.prompt2}<captions>")

    def prepare_lm_input(self, prompt_mode: str, prompt: str, vtokens: torch.FloatTensor, captions: List[str]):
        bsz, nvtoken, _ = vtokens.size()

        cap_tokens = self.lm_tokenizer(
            [t + "</s>" for t in captions],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,            # TODO: 应该减去prompt还有vtoken
        ).to(self.device).input_ids
        tokenizer_eos_id = self.lm_tokenizer.eos_token_id
        # eos_id = self.lm_tokenizer.eos_token_id
        eos_id = self.lm_model.config.eos_token_id
        # TODO: which one is correct eos?
        for i, ids in enumerate(cap_tokens):
            eos_position = ids.tolist().index(tokenizer_eos_id)
            ids[eos_position] = eos_id
        cap_embeds = self.lm_model.transformer.word_embeddings(cap_tokens)
        # -100 is the ignore index is CELoss
        cap_targets = cap_tokens.masked_fill(
            cap_tokens == self.lm_tokenizer.pad_token_id, -100
        )

        seq_lengths = [s.tolist().index(eos_id) + 1 for s in cap_tokens]
        context_lengths = []
        if prompt_mode == "None":
            # <vtokens><captions>
            inputs_embeds = torch.cat([vtokens, cap_embeds], dim=1)
            prefix_targets = (
                # do not apply loss to the prompt or vtokens
                torch.ones((bsz, nvtoken), dtype=torch.long).to(self.device).fill_(-100)
            )
        elif prompt_mode == "prefix":
            # <vtokens><prompt><captions>
            slot_offset = 0
        elif prompt_mode == "chat":
            # [Round 0]\n问：<vtokens><prompt>\n答：<captions>
            p1 = self.lm_tokenizer("[Round 0]\n问：", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            p2 = self.lm_tokenizer(f"{prompt}\n答：", return_tensors="pt").input_ids.to(self.device)
            inputs_embeds = torch.cat([
                self.lm_model.transformer.word_embeddings(p1).expand(bsz, -1, -1), vtokens,
                self.lm_model.transformer.word_embeddings(p2).expand(bsz, -1, -1), cap_embeds
            ], dim=1)
            # do not apply loss to the prompt or vtokens
            prefix_targets = torch.ones((bsz, len(p1[0]) + nvtoken + len(p2[0])), dtype=torch.long).to(self.device).fill_(-100)
        else:
            raise ValueError(f"Unknown prompt mode: {prompt_mode}")

        seq_lengths = [l + prefix_targets.size(1) for l in seq_lengths]
        targets = torch.cat([prefix_targets, cap_targets], dim=1)
        return inputs_embeds, targets, seq_lengths, context_lengths

    def forward(self, samples):
        image = samples["image"]
        device = image.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vtokens = self.lm_proj(query_output.last_hidden_state)
        bsz, nvtoken, _ = vtokens.size()
        # atts_vtokens = torch.ones((bsz, nvtoken), dtype=torch.long).to(device)

        # 注意正常情况下inference的tokenizer padding是在左边，因为要保证最后一个用于推理下一个
        # 但训练中不需要这样，因为只会做一次forward，padding在右边处理起来更方便
        self.lm_tokenizer.padding_side = "right"

        text = [t + "</s>" for t in samples["text_input"]]

        cap_tokens = self.lm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,            # TODO: 应该减去prompt还有vtoken
        ).to(device).input_ids
        tokenizer_eos_id = self.lm_tokenizer.eos_token_id
        # eos_id = self.lm_tokenizer.eos_token_id
        eos_id = self.lm_model.config.eos_token_id
        # TODO: which one is correct eos?
        for i, ids in enumerate(cap_tokens):
            eos_position = ids.tolist().index(tokenizer_eos_id)
            ids[eos_position] = eos_id
        cap_embeds = self.lm_model.transformer.word_embeddings(cap_tokens)
        # -100是ce的ignore index
        cap_targets = cap_tokens.masked_fill(
            cap_tokens == self.lm_tokenizer.pad_token_id, -100
        )

        if self.use_prompt:
            # prompt1 + vtoken + prompt2 + text
            # logging.info(f"{self.prompt1_embeds.shape}, {vtokens.shape}, {self.prompt2_embeds.shape}, {cap_embeds.shape}")
            inputs_embeds = torch.cat([
                self.prompt1_embeds.to(device).expand(bsz, -1, -1), vtokens,
                self.prompt2_embeds.to(device).expand(bsz, -1, -1), cap_embeds
            ], dim=1)

            # do not apply loss to the prompt or vtokens
            prefix_targets = torch.ones(
                (bsz, self.prompt1_length + nvtoken + self.prompt2_length), dtype=torch.long
            ).to(device).fill_(-100)
        else:
            inputs_embeds = torch.cat([vtokens, cap_embeds], dim=1)
            # vtoken + text
            prefix_targets = (
                # do not apply loss to the prompt or vtokens
                torch.ones((bsz, nvtoken), dtype=torch.long).to(device).fill_(-100)
            )

        targets = torch.cat([prefix_targets, cap_targets], dim=1)

        # NOTE: 需要在这里完成attention_mask以及position_id的计算
        # chatglm的position_ids的shape是[bsz, 2, l]，有两个是因为一个是普通的position_id另一个是block_position_id，因为rotary的关系所以有两组
        # attention_mask的shape是[bsz, 1, l, l]，其中第二个1是因为所有attention head是一样的可以broadcasst
        # chatglm的attention矩阵不是下三角矩阵，其实只是所有token看不到最后一个，之前的token的attention都是双向的
        attention_masks = torch.zeros((bsz, 1, inputs_embeds.size(1), inputs_embeds.size(1)), device=device)
        all_position_ids = torch.zeros((bsz, 2, inputs_embeds.size(1)), dtype=torch.long, device=device)
        for i, suffix in enumerate(cap_tokens):
            MASK, gMASK = 150000, 150001
            mask_token = MASK if MASK in suffix else gMASK
            use_gmask = False if MASK in suffix else gMASK
            seq = suffix.tolist()
            context_length = seq.index(self.lm_model.config.bos_token_id)
            mask_position = seq.index(mask_token)
            suffix_len = seq.index(eos_id) + 1

            if self.use_prompt:
                seq_len = self.prompt1_length + nvtoken + self.prompt2_length + suffix_len
            else:
                seq_len = nvtoken + suffix_len
            attention_mask = attention_masks[i, :, :seq_len, :seq_len]
            attention_mask.fill_(1)
            attention_mask.tril_()
            attention_mask[..., :context_length] = 1

            position_ids = all_position_ids[i]
            if self.lm_model.position_encoding_2d:
                torch.arange(context_length, out=position_ids[0][:context_length], dtype=torch.long, device=device)
                position_ids[0][context_length:seq_len] = mask_position
                torch.arange(
                    1, seq_len - context_length + 1,
                    out=position_ids[1][context_length:seq_len], dtype=torch.long, device=device
                )
            else:
                raise NotImplementedError()

        attention_masks = (attention_masks < 0.5).bool()
        with self.maybe_autocast():
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                position_ids=all_position_ids,
                attention_mask=attention_masks,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        query: Union[str, Tuple[str, torch.Tensor]],
        history: List[Tuple[Union[str, Tuple[str, torch.Tensor]], str]]=[],
        num_beams=5,
        max_length=128,
        top_p=0.9,
        do_sample=True,
        temperature=1,
    ):
        """
        Args:
            query (): The query string.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # 1. Prepare token ids
        images = []
        image_slots = []

        nvtokens = self.query_tokens.size(1)
        # if history:
        input_ids = self.lm_tokenizer(f"[Round {len(history)}]\n问：", add_special_tokens=False).input_ids
        slot_offset = len(input_ids)
        if isinstance(query, tuple):
            qtext, qimg = query
            # image slot, embedding will be replaced by image embeddings
            input_ids.extend([self.lm_tokenizer.unk_token_id] * nvtokens)
        else:
            qtext = query
            qimg = None
        input_ids += self.lm_tokenizer(qtext + f"\n答：").input_ids
        if qimg is not None:
            images.append(qimg)
            image_slots.append(len(input_ids) - slot_offset)       # count from backward

        for ri, (q, r) in enumerate(reversed(history)):
            if len(input_ids) >= self.max_txt_len:
                break
            i = len(history - ri - 1)
            cur_input_ids: List[int] = self.lm_tokenizer(f"[Round {i}]\n问：", add_special_tokens=False).input_ids
            slot_offset = len(cur_input_ids)
            if isinstance(q, tuple):
                qtext, qimg = q
                # image slot, embedding will be replaced by image embeddings
                cur_input_ids.extend([self.lm_tokenizer.unk_token_id] * nvtokens)
            else:
                qtext = q
                qimg = None
            cur_input_ids += self.lm_tokenizer(qtext + f"\n答：{r}\n", add_special_tokens=False).input_ids
            input_ids = cur_input_ids + input_ids
            if qimg is not None:
                images.append(qimg)
                image_slots.append(len(input_ids) - slot_offset)       # count from backward
        # else:
        #     raise NotImplementedError()

        if len(input_ids) >= self.max_txt_len:
            # truncate
            if image_slots[-1] > self.max_txt_len and image_slots[-1] - nvtokens < self.max_txt_len:
                # A non-intact image slot is not allowed
                input_ids = input_ids[-(image_slots[-1] - nvtokens):]
            else:
                input_ids = input_ids[-self.max_txt_len:]
            if image_slots[-1] > self.max_txt_len:
                image_slots.pop()
                images.pop()

        # 2. Prepare image embeddings
        if len(images) != 0:
            image = torch.cat(list(images), dim=0)
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            vtokens = self.lm_proj(query_output.last_hidden_state)
        else:
            vtokens = []

        # 3. Place image embeddings into slots
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).to(self.device).unsqueeze(0)
        inputs_embeds = self.lm_model.transformer.word_embeddings(input_ids)
        for slot, vimg in zip(image_slots, vtokens):
            inputs_embeds[0][-slot:-slot+nvtokens,:] = vimg

        with self.maybe_autocast(dtype=torch.bfloat16):
            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor}

            for outputs in self.lm_model.mm_stream_generate(input_ids=input_ids, inputs_embeds=inputs_embeds, **gen_kwargs):
                outputs = outputs.tolist()[0][len(input_ids[0]):]
                response = self.lm_tokenizer.decode(outputs)
                response = self.lm_model.process_response(response)
                new_history = history + [(query, response)]
                yield response, new_history

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

        prompt = cfg.get("prompt", "")
        use_prompt = cfg.get("use_prompt", False)
        max_txt_len = cfg.get("max_txt_len", 128)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            lm_model=lm_model,
            prompt=prompt,
            use_prompt=use_prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
