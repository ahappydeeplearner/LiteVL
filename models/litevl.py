"""
LiteVL - 完整的视觉语言模型
架构: SigLIP (frozen) -> MLP Projector -> Qwen2 (LLM)

低成本训练策略:
- Stage 1: 只训练 Projector (~10M 参数)
- Stage 2: 训练 Projector + LLM (LoRA, ~50M 参数)
- Stage 3: DPO 偏好对齐 (LoRA, ~50M 参数)
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from .vision_encoder import VisionEncoder
from .projector import MLPProjector


# 特殊 token 定义
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


class LiteVL(nn.Module):
    """
    LiteVL - 低成本视觉语言模型

    训练成本估算 (单卡 A100-80G):
    - Stage 1: ~2 GPU-hours (558K samples, 只训练 projector)
    - Stage 2: ~8 GPU-hours (665K samples, LoRA 微调)
    - Stage 3: ~2 GPU-hours (偏好数据, LoRA 微调)
    总计: ~12 GPU-hours
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Vision Encoder (SigLIP, frozen)
        self.vision_encoder = VisionEncoder(
            model_name=config.vision_encoder_name,
            freeze=config.freeze_vision_encoder
        )

        # 2. MLP Projector (always trainable)
        self.projector = MLPProjector(
            vision_hidden_size=config.vision_hidden_size,
            llm_hidden_size=config.llm_hidden_size,
            projector_type=config.projector_type,
        )

        # 3. LLM Backbone (Qwen2)
        llm_kwargs = dict(
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        try:
            import flash_attn  # noqa: F401
            llm_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            llm_kwargs["attn_implementation"] = "eager"
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            **llm_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name,
            trust_remote_code=True,
            padding_side="right",
        )

        # 确保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.eos_token_id

    def setup_for_stage(self, stage: str):
        """
        根据训练阶段配置可训练参数

        Stage 1 (pretrain): 冻结 vision encoder + LLM，只训练 projector
        Stage 2 (sft): 冻结 vision encoder，训练 projector + LLM (LoRA)
        Stage 3 (dpo): 同 Stage 2，加载 SFT checkpoint 后继续训练
        """
        # 始终冻结 vision encoder
        self.vision_encoder.freeze()

        if stage == "pretrain":
            # 冻结 LLM
            for param in self.llm.parameters():
                param.requires_grad = False
            # 启用 gradient checkpointing 以节省显存
            self.llm.gradient_checkpointing_enable()
            # 只训练 projector
            for param in self.projector.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[Stage 1 Pretrain] 可训练参数: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
                  f"({trainable / total * 100:.2f}%)")

        elif stage in ("sft", "dpo"):
            # 训练 projector
            for param in self.projector.parameters():
                param.requires_grad = True

            # 用 LoRA 微调 LLM
            if self.config.use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules,
                )
                self.llm = get_peft_model(self.llm, lora_config)
                self.llm.print_trainable_parameters()
            else:
                for param in self.llm.parameters():
                    param.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[Stage {stage.upper()}] 可训练参数: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
                  f"({trainable / total * 100:.2f}%)")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """编码图像: Vision Encoder -> Projector"""
        vision_features = self.vision_encoder(pixel_values)
        image_embeds = self.projector(vision_features)
        return image_embeds

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将图像 embedding 插入到文本 embedding 序列中 <image> token 的位置

        Args:
            input_ids: [B, L] 文本 token ids, <image> 位置用 IMAGE_TOKEN_INDEX 标记
            pixel_values: [B, C, H, W] 图像像素值
            attention_mask: [B, L]
        Returns:
            inputs_embeds: [B, L', D] 合并后的 embedding
            attention_mask: [B, L'] 更新后的 attention mask
            labels: [B, L'] 更新后的 labels
        """
        # 获取文本 embeddings
        text_embeds = self.llm.get_input_embeddings()(
            input_ids.clamp(min=0)  # 将 IMAGE_TOKEN_INDEX 替换为 0 来获取 embedding
        )

        if pixel_values is None:
            return text_embeds, attention_mask, None

        # 获取图像 embeddings
        image_embeds = self.encode_images(pixel_values)  # [B, N, D]
        B, N, D = image_embeds.shape

        # 找到 <image> token 的位置并替换
        new_embeds_list = []
        new_mask_list = []

        for i in range(B):
            cur_input_ids = input_ids[i]
            cur_text_embeds = text_embeds[i]
            cur_attention_mask = attention_mask[i] if attention_mask is not None else None

            # 找到 image token 位置
            image_token_positions = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

            if len(image_token_positions) == 0:
                new_embeds_list.append(cur_text_embeds)
                if cur_attention_mask is not None:
                    new_mask_list.append(cur_attention_mask)
                continue

            # 在 image token 位置插入图像 embedding
            parts = []
            mask_parts = []
            prev_pos = 0
            for pos in image_token_positions:
                pos = pos.item()
                # 添加 image token 之前的文本
                parts.append(cur_text_embeds[prev_pos:pos])
                if cur_attention_mask is not None:
                    mask_parts.append(cur_attention_mask[prev_pos:pos])
                # 添加图像 embedding
                parts.append(image_embeds[i])
                if cur_attention_mask is not None:
                    mask_parts.append(torch.ones(N, device=cur_attention_mask.device,
                                                  dtype=cur_attention_mask.dtype))
                prev_pos = pos + 1

            # 添加剩余文本
            parts.append(cur_text_embeds[prev_pos:])
            if cur_attention_mask is not None:
                mask_parts.append(cur_attention_mask[prev_pos:])

            new_embeds_list.append(torch.cat(parts, dim=0))
            if mask_parts:
                new_mask_list.append(torch.cat(mask_parts, dim=0))

        # Padding 到相同长度
        max_len = max(e.shape[0] for e in new_embeds_list)
        padded_embeds = torch.zeros(B, max_len, D, device=text_embeds.device,
                                     dtype=text_embeds.dtype)
        padded_mask = torch.zeros(B, max_len, device=text_embeds.device, dtype=torch.long)

        for i in range(B):
            L = new_embeds_list[i].shape[0]
            padded_embeds[i, :L] = new_embeds_list[i]
            if new_mask_list:
                padded_mask[i, :L] = new_mask_list[i]
            else:
                padded_mask[i, :L] = 1

        return padded_embeds, padded_mask, None

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            dict with 'loss', 'logits'
        """
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        # 如果有 labels，需要同步扩展
        if labels is not None:
            # 扩展 labels 以匹配新的序列长度
            B = labels.shape[0]
            new_labels = torch.full(
                (B, inputs_embeds.shape[1]),
                fill_value=-100,  # ignore index
                device=labels.device,
                dtype=labels.dtype,
            )
            # 图像部分的 label 为 -100 (不计算 loss)
            # 简单起见，将原始 labels 对齐到末尾
            for i in range(B):
                orig_len = labels[i].shape[0]
                offset = inputs_embeds.shape[1] - input_ids.shape[1]
                # 复制非 image token 部分的 labels
                src_ids = input_ids[i]
                new_idx = 0
                for j in range(orig_len):
                    if src_ids[j] == IMAGE_TOKEN_INDEX:
                        # 图像 token 展开后的位置，label 为 -100
                        num_patches = self.vision_encoder.num_patches
                        new_idx += num_patches
                    else:
                        if new_idx < new_labels.shape[1]:
                            new_labels[i, new_idx] = labels[i, j]
                        new_idx += 1
            labels = new_labels

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        """推理生成"""
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            **kwargs,
        )
        return outputs

    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        # 保存 projector
        torch.save(self.projector.state_dict(), os.path.join(save_path, "projector.pt"))
        # 保存 LLM (或 LoRA adapter)
        self.llm.save_pretrained(os.path.join(save_path, "llm"))
        # 保存 tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_path, "llm"))
        # 保存配置
        import json
        config_dict = {k: v for k, v in self.config.__dict__.items()
                       if not k.startswith("_")}
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"模型已保存到 {save_path}")
