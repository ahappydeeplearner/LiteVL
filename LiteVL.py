#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  LiteVL - 低成本视觉语言模型训练框架 (单文件版)                     ║
║  架构: SigLIP (Vision Encoder) + MLP Projector + Qwen2 (LLM)   ║
║                                                                  ║
║  三阶段训练策略:                                                   ║
║    Stage 1 (Pretrain): 特征对齐，只训练 Projector (~10M 参数)      ║
║    Stage 2 (SFT):      指令微调，LoRA 微调 LLM (~50M 参数)        ║
║    Stage 3 (DPO):      偏好对齐，减少幻觉                          ║
║                                                                  ║
║  Usage:                                                          ║
║    python LiteVL.py --stage pretrain                             ║
║    python LiteVL.py --stage sft                                  ║
║    python LiteVL.py --stage dpo                                  ║
║    python LiteVL.py --stage all                                  ║
║    python LiteVL.py --stage dummy    # 生成测试数据               ║
║    python LiteVL.py --stage infer --model_path outputs/xxx       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# Imports
# ============================================================================
import os
import sys
import json
import copy
import math
import time
import random
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipImageProcessor,
)
from peft import get_peft_model, LoraConfig, TaskType


# ============================================================================
# Section 1: 配置 (Configs)
# ============================================================================

@dataclass
class ModelConfig:
    """模型架构配置"""
    # Vision Encoder
    vision_encoder_name: str = "google/siglip-so400m-patch14-384"
    vision_hidden_size: int = 1152
    image_size: int = 384
    patch_size: int = 14
    freeze_vision_encoder: bool = True

    # MLP Projector
    projector_type: str = "mlp2x_gelu"  # 2层MLP + GELU
    projector_hidden_size: int = 4096

    # LLM Backbone
    llm_name: str = "Qwen/Qwen2-0.5B-Instruct"
    llm_hidden_size: int = 896
    max_length: int = 2048

    # LoRA 配置 (用于低成本微调)
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataConfig:
    """数据配置"""
    pretrain_data_path: str = "data/pretrain/llava_pretrain_lcs558k.json"
    pretrain_image_dir: str = "data/pretrain/images"
    sft_data_path: str = "data/sft/llava_v1_5_mix665k.json"
    sft_image_dir: str = "data/sft/images"
    dpo_data_path: str = "data/dpo/rlhf_v_preference.json"
    dpo_image_dir: str = "data/dpo/images"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class PretrainConfig:
    """Stage 1: 预训练 - 特征对齐"""
    output_dir: str = "outputs/stage1_pretrain"
    num_epochs: int = 1
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    deepspeed_config: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


@dataclass
class SFTConfig:
    """Stage 2: 指令微调"""
    output_dir: str = "outputs/stage2_sft"
    num_epochs: int = 1
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    deepspeed_config: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 1000
    seed: int = 42
    use_lora: bool = True


@dataclass
class DPOConfig:
    """Stage 3: DPO 偏好优化"""
    output_dir: str = "outputs/stage3_dpo"
    num_epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    beta: float = 0.1  # DPO 温度参数
    logging_steps: int = 5
    save_steps: int = 500
    seed: int = 42
    use_lora: bool = True
    sft_model_path: str = "outputs/stage2_sft/final"


# ============================================================================
# Section 2: 训练工具 (Utils)
# ============================================================================

def set_seed(seed: int = 42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, scheduler_type: str, num_training_steps: int,
                     warmup_ratio: float = 0.03):
    """获取学习率调度器"""
    from torch.optim.lr_scheduler import LambdaLR

    num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "cosine":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) /
                       float(max(1, num_training_steps - num_warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def count_parameters(model) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_ratio": trainable / total if total > 0 else 0,
    }


def save_checkpoint(model, optimizer, scheduler, step: int, epoch: int,
                    output_dir: str, metrics: Optional[dict] = None):
    """保存训练 checkpoint"""
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    train_state = {
        "step": step,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    torch.save(train_state, os.path.join(ckpt_dir, "train_state.pt"))
    print(f"Checkpoint 已保存到 {ckpt_dir}")


def load_checkpoint(model, optimizer, scheduler, ckpt_dir: str):
    """加载训练 checkpoint"""
    projector_path = os.path.join(ckpt_dir, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    state_path = os.path.join(ckpt_dir, "train_state.pt")
    if os.path.exists(state_path):
        train_state = torch.load(state_path, map_location="cpu")
        if optimizer:
            optimizer.load_state_dict(train_state["optimizer_state_dict"])
        if scheduler and train_state["scheduler_state_dict"]:
            scheduler.load_state_dict(train_state["scheduler_state_dict"])
        return train_state["step"], train_state["epoch"]
    return 0, 0


def get_gpu_info() -> str:
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        return "No GPU available"
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info.append(f"GPU {i}: {props.name} ({mem:.1f}GB)")
    return " | ".join(info)


# ============================================================================
# Section 3: 日志系统 (Logger)
# ============================================================================

class TrainLogger:
    """统一训练日志管理器，支持 Console / File / TensorBoard / WandB"""

    def __init__(self, output_dir: str, stage_name: str,
                 use_wandb: bool = False, wandb_project: str = "litevl",
                 use_tensorboard: bool = True):
        self.output_dir = output_dir
        self.stage_name = stage_name
        self.log_dir = os.path.join(output_dir, "logs")
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.logger = self._setup_file_logger()
        self.metrics_file = open(os.path.join(self.log_dir, "metrics.jsonl"), "a")
        self.step_metrics = defaultdict(list)
        self.global_step = 0
        self.epoch = 0
        self.start_time = time.time()

        self.wandb_run = None
        if use_wandb:
            self._setup_wandb(wandb_project)

        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()

        self.log_info(f"===== {stage_name} 训练日志初始化 =====")
        self.log_info(f"输出目录: {output_dir}")

    def _setup_file_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"vlm_{self.stage_name}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        fh = logging.FileHandler(os.path.join(self.log_dir, "train.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _setup_wandb(self, project: str):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=project,
                name=f"{self.stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dir=self.output_dir,
            )
            self.log_info("WandB 已启用")
        except ImportError:
            self.log_info("WandB 未安装，跳过")

    def _setup_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.log_info(f"TensorBoard 已启用: {tb_dir}")
        except ImportError:
            self.log_info("TensorBoard 未安装，跳过")

    def log_info(self, msg: str):
        self.logger.info(msg)

    def log_warning(self, msg: str):
        self.logger.warning(msg)

    def log_error(self, msg: str):
        self.logger.error(msg)

    def log_config(self, config: Any):
        config_dict = config.__dict__ if hasattr(config, "__dict__") else config
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        self.log_info(f"训练配置已保存到 {config_path}")
        if self.wandb_run:
            import wandb
            wandb.config.update(config_dict)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is not None:
            self.global_step = step
        record = {
            "step": self.global_step,
            "epoch": self.epoch,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            **metrics,
        }
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()
        for k, v in metrics.items():
            self.step_metrics[k].append(v)
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=self.global_step)
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        metrics_str = " | ".join(
            f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.log_info(f"[Step {self.global_step}][Epoch {self.epoch}][{time_str}] {metrics_str}")

    def log_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        self.epoch = epoch
        self.log_info(f"\n{'='*60}")
        self.log_info(f"Epoch {epoch} 训练完成")
        for k, v in metrics.items():
            self.log_info(f"  {k}: {v:.6f}")
        self.log_info(f"{'='*60}\n")
        self.log_metrics({f"epoch/{k}": v for k, v in metrics.items()})

    def log_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            self.log_info(
                f"GPU 显存 - 已分配: {allocated:.2f}GB, "
                f"已预留: {reserved:.2f}GB, 峰值: {max_allocated:.2f}GB"
            )

    def log_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        self.log_info(f"\n{'='*60}")
        self.log_info("模型参数统计:")
        self.log_info(f"  总参数量:     {total_params / 1e6:.2f}M")
        self.log_info(f"  可训练参数:   {trainable_params / 1e6:.2f}M")
        self.log_info(f"  冻结参数:     {frozen_params / 1e6:.2f}M")
        self.log_info(f"  可训练比例:   {trainable_params / total_params * 100:.2f}%")
        self.log_info(f"{'='*60}\n")

    def close(self):
        self.metrics_file.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()
        self.log_info("日志系统已关闭")


class MetricsTracker:
    """训练指标追踪器，支持滑动窗口平均"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.running_sum = defaultdict(float)
        self.running_count = defaultdict(int)

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.metrics[k].append(v)
            self.running_sum[k] += v
            self.running_count[k] += 1

    def get_smoothed(self, key: str) -> float:
        values = self.metrics[key][-self.window_size:]
        return sum(values) / len(values) if values else 0.0

    def get_global_avg(self, key: str) -> float:
        if self.running_count[key] == 0:
            return 0.0
        return self.running_sum[key] / self.running_count[key]

    def get_latest(self, key: str) -> float:
        return self.metrics[key][-1] if self.metrics[key] else 0.0

    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for k in self.metrics:
            summary[f"{k}_avg"] = self.get_global_avg(k)
            summary[f"{k}_smoothed"] = self.get_smoothed(k)
            summary[f"{k}_latest"] = self.get_latest(k)
        return summary

    def reset(self):
        self.metrics.clear()
        self.running_sum.clear()
        self.running_count.clear()


# ============================================================================
# Section 4: 模型 (Models)
# ============================================================================

# 特殊 token 定义
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200


class VisionEncoder(nn.Module):
    """基于 SigLIP 的视觉编码器 (参数冻结以降低训练成本)"""

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384",
                 freeze: bool = True):
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        self._frozen = False
        if freeze:
            self.freeze()

    def freeze(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self._frozen = True
        self.vision_model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._frozen:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
                vision_features = outputs.last_hidden_state
        else:
            outputs = self.vision_model(pixel_values=pixel_values)
            vision_features = outputs.last_hidden_state
        return vision_features

    def get_image_processor(self):
        return self.image_processor

    @property
    def num_patches(self):
        image_size = self.vision_model.config.image_size
        patch_size = self.vision_model.config.patch_size
        return (image_size // patch_size) ** 2


class TokenCompressor(nn.Module):
    """视觉 Token 压缩模块 (N -> N/ratio)"""

    def __init__(self, hidden_size: int, compress_ratio: int = 4):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.compress_proj = nn.Linear(hidden_size * compress_ratio, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert N % self.compress_ratio == 0, \
            f"Token数 {N} 必须能被压缩比 {self.compress_ratio} 整除"
        x = x.reshape(B, N // self.compress_ratio, D * self.compress_ratio)
        x = self.compress_proj(x)
        x = self.norm(x)
        return x


class MLPProjector(nn.Module):
    """2层 MLP + GELU，将视觉 token 映射到语言模型空间"""

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()
        self.projector_type = projector_type

        if projector_type == "mlp2x_gelu":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        elif projector_type == "mlp2x_gelu_compress":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
            self.token_compressor = TokenCompressor(llm_hidden_size, compress_ratio=4)
        elif projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        projected = self.projector(vision_features)
        if self.projector_type == "mlp2x_gelu_compress":
            projected = self.token_compressor(projected)
        return projected


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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        try:
            import flash_attn  # noqa: F401
            llm_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            llm_kwargs["attn_implementation"] = "eager"
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_name, **llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name, trust_remote_code=True, padding_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.eos_token_id

    def setup_for_stage(self, stage: str):
        """根据训练阶段配置可训练参数"""
        self.vision_encoder.freeze()

        if stage == "pretrain":
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"[Stage 1 Pretrain] 可训练参数: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
                  f"({trainable / total * 100:.2f}%)")

        elif stage in ("sft", "dpo"):
            for param in self.projector.parameters():
                param.requires_grad = True
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
        vision_features = self.vision_encoder(pixel_values)
        image_embeds = self.projector(vision_features)
        return image_embeds

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        text_embeds = self.llm.get_input_embeddings()(input_ids.clamp(min=0))

        if pixel_values is None:
            return text_embeds, attention_mask, None

        image_embeds = self.encode_images(pixel_values)
        B, N, D = image_embeds.shape

        new_embeds_list = []
        new_mask_list = []

        for i in range(B):
            cur_input_ids = input_ids[i]
            cur_text_embeds = text_embeds[i]
            cur_attention_mask = attention_mask[i] if attention_mask is not None else None

            image_token_positions = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

            if len(image_token_positions) == 0:
                new_embeds_list.append(cur_text_embeds)
                if cur_attention_mask is not None:
                    new_mask_list.append(cur_attention_mask)
                continue

            parts = []
            mask_parts = []
            prev_pos = 0
            for pos in image_token_positions:
                pos = pos.item()
                parts.append(cur_text_embeds[prev_pos:pos])
                if cur_attention_mask is not None:
                    mask_parts.append(cur_attention_mask[prev_pos:pos])
                parts.append(image_embeds[i])
                if cur_attention_mask is not None:
                    mask_parts.append(torch.ones(N, device=cur_attention_mask.device,
                                                  dtype=cur_attention_mask.dtype))
                prev_pos = pos + 1

            parts.append(cur_text_embeds[prev_pos:])
            if cur_attention_mask is not None:
                mask_parts.append(cur_attention_mask[prev_pos:])

            new_embeds_list.append(torch.cat(parts, dim=0))
            if mask_parts:
                new_mask_list.append(torch.cat(mask_parts, dim=0))

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
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
        )

        if labels is not None:
            B = labels.shape[0]
            new_labels = torch.full(
                (B, inputs_embeds.shape[1]), fill_value=-100,
                device=labels.device, dtype=labels.dtype,
            )
            for i in range(B):
                orig_len = labels[i].shape[0]
                src_ids = input_ids[i]
                new_idx = 0
                for j in range(orig_len):
                    if src_ids[j] == IMAGE_TOKEN_INDEX:
                        num_patches = self.vision_encoder.num_patches
                        new_idx += num_patches
                    else:
                        if new_idx < new_labels.shape[1]:
                            new_labels[i, new_idx] = labels[i, j]
                        new_idx += 1
            labels = new_labels

        outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            labels=labels, return_dict=True,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

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
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
        )
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=temperature > 0, **kwargs,
        )
        return outputs

    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.projector.state_dict(), os.path.join(save_path, "projector.pt"))
        self.llm.save_pretrained(os.path.join(save_path, "llm"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "llm"))
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")}
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"模型已保存到 {save_path}")


# ============================================================================
# Section 5: 数据集 (Datasets)
# ============================================================================

class PretrainDataset(Dataset):
    """Stage 1: 预训练数据集 (图文对, LCS-558K 格式)"""

    def __init__(self, data_path: str, image_dir: str,
                 tokenizer, image_processor: SiglipImageProcessor,
                 max_length: int = 2048):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"[PretrainDataset] 加载 {len(self.data)} 条预训练数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.squeeze(0)
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            pixel_values = torch.zeros(3, 384, 384)

        conversations = item["conversations"]
        human_msg = conversations[0]["value"]
        gpt_msg = conversations[1]["value"]
        prompt = human_msg.replace(IMAGE_TOKEN, "").strip()
        full_text = f"{prompt}\n{gpt_msg}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        prompt_tokens = self.tokenizer(
            f"{prompt}\n", max_length=self.max_length, truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[:len(prompt_tokens)] = -100

        input_ids_with_image = torch.cat([
            torch.tensor([IMAGE_TOKEN_INDEX]), input_ids[:-1]
        ])

        return {
            "input_ids": input_ids_with_image,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTDataset(Dataset):
    """Stage 2: 指令微调数据集 (多轮对话, LLaVA-mix-665K 格式)"""

    SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(self, data_path: str, image_dir: str,
                 tokenizer, image_processor: SiglipImageProcessor,
                 max_length: int = 2048, llm_type: str = "qwen2"):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.llm_type = llm_type
        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"[SFTDataset] 加载 {len(self.data)} 条SFT数据")

    def __len__(self):
        return len(self.data)

    def _format_conversation(self, conversations: List[Dict]) -> str:
        formatted = ""
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content = conv["value"].replace(IMAGE_TOKEN, "").strip()
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        has_image = "image" in item and item["image"]
        if has_image:
            image_path = os.path.join(self.image_dir, item["image"])
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values.squeeze(0)
            except Exception:
                pixel_values = torch.zeros(3, 384, 384)
                has_image = False
        else:
            pixel_values = torch.zeros(3, 384, 384)

        conversations = item["conversations"]
        formatted_text = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
        formatted_text += self._format_conversation(conversations)

        encoding = self.tokenizer(
            formatted_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels = self._mask_non_assistant_tokens(input_ids, labels)

        if has_image:
            input_ids = torch.cat([torch.tensor([IMAGE_TOKEN_INDEX]), input_ids[:-1]])

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values if has_image else None,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_non_assistant_tokens(self, input_ids, labels):
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        in_assistant = False
        for i in range(len(input_ids)):
            if input_ids[i] == im_start_id:
                in_assistant = False
                remaining = self.tokenizer.decode(input_ids[i:i+3])
                if "assistant" in remaining:
                    in_assistant = True
                labels[i] = -100
            elif input_ids[i] == im_end_id:
                in_assistant = False
                labels[i] = -100
            elif not in_assistant:
                labels[i] = -100
        return labels


class DPODataset(Dataset):
    """Stage 3: DPO 偏好数据集 (chosen/rejected 格式)"""

    def __init__(self, data_path: str, image_dir: str,
                 tokenizer, image_processor: SiglipImageProcessor,
                 max_length: int = 2048):
        super().__init__()
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"[DPODataset] 加载 {len(self.data)} 条偏好数据")

    def __len__(self):
        return len(self.data)

    def _tokenize(self, question: str, answer: str):
        text = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>")
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 384, 384)

        chosen_ids, chosen_mask = self._tokenize(item["question"], item["chosen"])
        rejected_ids, rejected_mask = self._tokenize(item["question"], item["rejected"])

        chosen_ids = torch.cat([torch.tensor([IMAGE_TOKEN_INDEX]), chosen_ids[:-1]])
        rejected_ids = torch.cat([torch.tensor([IMAGE_TOKEN_INDEX]), rejected_ids[:-1]])

        return {
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


def collate_fn_pretrain(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def collate_fn_sft(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    has_images = [b["pixel_values"] is not None for b in batch]
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
    if any(has_images):
        pixel_values = []
        for b in batch:
            if b["pixel_values"] is not None:
                pixel_values.append(b["pixel_values"])
            else:
                pixel_values.append(torch.zeros(3, 384, 384))
        result["pixel_values"] = torch.stack(pixel_values)
    else:
        result["pixel_values"] = None
    return result


def collate_fn_dpo(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]),
        "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]),
        "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]),
        "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]),
    }


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                     num_workers: int = 4, collate_fn=None) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )


# ============================================================================
# Section 6: 训练器 (Trainers)
# ============================================================================

class PretrainTrainer:
    """Stage 1: 预训练 Trainer - 特征对齐"""

    def __init__(self, model, train_dataloader, config, logger: TrainLogger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.logger = logger
        self.metrics = MetricsTracker(window_size=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.projector.parameters() if p.requires_grad],
            lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        num_training_steps = (
            len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer, config.lr_scheduler_type, num_training_steps, config.warmup_ratio,
        )
        self.scaler = GradScaler() if config.fp16 else None
        self.use_bf16 = config.bf16
        self.global_step = 0

    def train(self):
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 1: 预训练 - 特征对齐")
        self.logger.log_info(f"GPU: {get_gpu_info()}")
        self.logger.log_model_info(self.model)
        self.logger.log_gpu_memory()
        self.logger.log_info("=" * 60)

        self.model.train()
        self.model.vision_encoder.eval()
        self.model.llm.eval()

        for epoch in range(self.config.num_epochs):
            self.logger.epoch = epoch
            epoch_metrics = MetricsTracker()
            for step, batch in enumerate(self.train_dataloader):
                loss = self._train_step(batch)
                if loss is not None:
                    self.metrics.update({"loss": loss})
                    epoch_metrics.update({"loss": loss})
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.projector.parameters(), self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    if self.global_step % self.config.logging_steps == 0:
                        self.logger.log_metrics({
                            "train/loss": self.metrics.get_smoothed("loss"),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(self.train_dataloader),
                        }, step=self.global_step)
                    if self.global_step % self.config.save_steps == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, self.config.output_dir,
                        )
            self.logger.log_epoch_summary(epoch, {"loss": epoch_metrics.get_global_avg("loss")})
            self.logger.log_gpu_memory()

        final_dir = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(final_dir)
        self.logger.log_info(f"预训练完成! 模型已保存到 {final_dir}")

    def _train_step(self, batch) -> float:
        input_ids = batch["input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.use_bf16:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask, labels=labels,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
        elif self.scaler:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask, labels=labels,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(
                input_ids=input_ids, pixel_values=pixel_values,
                attention_mask=attention_mask, labels=labels,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.item() * self.config.gradient_accumulation_steps


class SFTTrainer:
    """Stage 2: SFT Trainer - 指令微调"""

    def __init__(self, model, train_dataloader, config, logger: TrainLogger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.logger = logger
        self.metrics = MetricsTracker(window_size=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        num_training_steps = (
            len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer, config.lr_scheduler_type, num_training_steps, config.warmup_ratio,
        )
        self.scaler = GradScaler() if config.fp16 else None
        self.use_bf16 = config.bf16
        self.global_step = 0

    def train(self):
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 2: SFT - 指令微调")
        self.logger.log_info(f"GPU: {get_gpu_info()}")
        self.logger.log_model_info(self.model)
        self.logger.log_gpu_memory()
        self.logger.log_info("=" * 60)

        self.model.train()
        self.model.vision_encoder.eval()

        for epoch in range(self.config.num_epochs):
            self.logger.epoch = epoch
            epoch_metrics = MetricsTracker()
            for step, batch in enumerate(self.train_dataloader):
                loss = self._train_step(batch)
                if loss is not None:
                    self.metrics.update({"loss": loss})
                    epoch_metrics.update({"loss": loss})
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    if self.global_step % self.config.logging_steps == 0:
                        self.logger.log_metrics({
                            "train/loss": self.metrics.get_smoothed("loss"),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(self.train_dataloader),
                        }, step=self.global_step)
                    if self.global_step % self.config.save_steps == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, self.config.output_dir,
                        )
            self.logger.log_epoch_summary(epoch, {"loss": epoch_metrics.get_global_avg("loss")})
            self.logger.log_gpu_memory()

        final_dir = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(final_dir)
        self.logger.log_info(f"SFT 训练完成! 模型已保存到 {final_dir}")

    def _train_step(self, batch) -> float:
        input_ids = batch["input_ids"].to(self.device)
        pixel_values = batch["pixel_values"]
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.use_bf16:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask, labels=labels,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(
                input_ids=input_ids, pixel_values=pixel_values,
                attention_mask=attention_mask, labels=labels,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps
        loss.backward()
        return loss.item() * self.config.gradient_accumulation_steps


class DPOTrainer:
    """Stage 3: DPO Trainer - 偏好对齐"""

    def __init__(self, model, train_dataloader, config, logger: TrainLogger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.logger = logger
        self.metrics = MetricsTracker(window_size=50)
        self.beta = config.beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.ref_model = self._create_ref_model()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        num_training_steps = (
            len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer, config.lr_scheduler_type, num_training_steps, config.warmup_ratio,
        )
        self.use_bf16 = config.bf16
        self.global_step = 0

    def _create_ref_model(self):
        ref_model = copy.deepcopy(self.model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        ref_model.to(self.device)
        return ref_model

    def _get_log_probs(self, model, input_ids, pixel_values, attention_mask):
        outputs = model(input_ids=input_ids, pixel_values=pixel_values,
                        attention_mask=attention_mask)
        logits = outputs["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].clamp(min=0).contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        mask = attention_mask[:, 1:].float()
        sum_log_probs = (token_log_probs * mask).sum(dim=-1)
        return sum_log_probs

    def _compute_dpo_loss(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)

        policy_chosen_logps = self._get_log_probs(self.model, chosen_ids, pixel_values, chosen_mask)
        policy_rejected_logps = self._get_log_probs(self.model, rejected_ids, pixel_values, rejected_mask)

        with torch.no_grad():
            ref_chosen_logps = self._get_log_probs(self.ref_model, chosen_ids, pixel_values, chosen_mask)
            ref_rejected_logps = self._get_log_probs(self.ref_model, rejected_ids, pixel_values, rejected_mask)

        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        chosen_reward = chosen_rewards.mean().item()
        rejected_reward = rejected_rewards.mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        return loss, {
            "reward_margin": reward_margin,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "accuracy": accuracy,
        }

    def train(self):
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 3: DPO - 偏好对齐")
        self.logger.log_info(f"GPU: {get_gpu_info()}")
        self.logger.log_info(f"DPO beta: {self.beta}")
        self.logger.log_model_info(self.model)
        self.logger.log_gpu_memory()
        self.logger.log_info("=" * 60)

        self.model.train()
        self.model.vision_encoder.eval()

        for epoch in range(self.config.num_epochs):
            self.logger.epoch = epoch
            epoch_metrics = MetricsTracker()
            for step, batch in enumerate(self.train_dataloader):
                if self.use_bf16:
                    device_type = "cuda" if self.device.type == "cuda" else "cpu"
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss, extra_metrics = self._compute_dpo_loss(batch)
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss, extra_metrics = self._compute_dpo_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                loss_val = loss.item() * self.config.gradient_accumulation_steps
                self.metrics.update({"loss": loss_val, **extra_metrics})
                epoch_metrics.update({"loss": loss_val, **extra_metrics})

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    if self.global_step % self.config.logging_steps == 0:
                        self.logger.log_metrics({
                            "train/loss": self.metrics.get_smoothed("loss"),
                            "train/reward_margin": self.metrics.get_smoothed("reward_margin"),
                            "train/chosen_reward": self.metrics.get_smoothed("chosen_reward"),
                            "train/rejected_reward": self.metrics.get_smoothed("rejected_reward"),
                            "train/accuracy": self.metrics.get_smoothed("accuracy"),
                            "train/lr": self.scheduler.get_last_lr()[0],
                        }, step=self.global_step)
                    if self.global_step % self.config.save_steps == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, self.config.output_dir,
                        )
            self.logger.log_epoch_summary(epoch, {
                "loss": epoch_metrics.get_global_avg("loss"),
                "reward_margin": epoch_metrics.get_global_avg("reward_margin"),
                "accuracy": epoch_metrics.get_global_avg("accuracy"),
            })
            self.logger.log_gpu_memory()

        final_dir = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(final_dir)
        self.logger.log_info(f"DPO 训练完成! 模型已保存到 {final_dir}")


# ============================================================================
# Section 7: 推理 (Inference)
# ============================================================================

def load_model(model_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        config = ModelConfig(**{k: v for k, v in config_dict.items()
                                if hasattr(ModelConfig, k)})
    else:
        config = ModelConfig()

    model = LiteVL(config)

    projector_path = os.path.join(model_path, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    llm_path = os.path.join(model_path, "llm")
    if os.path.exists(llm_path):
        from peft import PeftModel
        try:
            model.llm = PeftModel.from_pretrained(model.llm, llm_path)
        except Exception:
            pass

    model.to(device)
    model.eval()
    return model


def chat(model: LiteVL, image_path: str, question: str,
         max_new_tokens: int = 512, temperature: float = 0.7):
    """单轮对话推理"""
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    image_processor = model.vision_encoder.get_image_processor()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    image_token = torch.tensor([[IMAGE_TOKEN_INDEX]], device=device)
    input_ids = torch.cat([image_token, input_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(
        input_ids=input_ids, pixel_values=pixel_values,
        attention_mask=attention_mask, max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


# ============================================================================
# Section 8: 数据准备 (Data Preparation)
# ============================================================================

DATASETS_INFO = {
    "pretrain": {
        "name": "LLaVA-Pretrain (LCS-558K)",
        "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain",
        "description": "558K 图文对，用于 Stage 1 特征对齐预训练",
        "size": "~15GB (含图像)",
        "command": "git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain data/pretrain",
    },
    "sft": {
        "name": "LLaVA-v1.5-mix665k",
        "url": "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K",
        "description": "665K 指令微调数据，用于 Stage 2 SFT",
        "size": "~50GB (含图像)",
        "command": (
            "# 1. 下载标注文件\n"
            "wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K"
            "/resolve/main/llava_v1_5_mix665k.json -P data/sft/\n"
            "# 2. 下载图像 (COCO train2017)\n"
            "mkdir -p data/sft/images\n"
            "wget http://images.cocodataset.org/zips/train2017.zip -P data/sft/images/\n"
        ),
    },
    "dpo": {
        "name": "RLHF-V Dataset",
        "url": "https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset",
        "description": "偏好数据，用于 Stage 3 DPO 偏好对齐，减少幻觉",
        "size": "~5GB",
        "command": "git clone https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset data/dpo",
    },
}


def generate_dummy_data(stage: str, num_samples: int = 100):
    """生成模拟数据用于测试训练流程"""
    output_dir = f"data/{stage}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    for i in range(min(num_samples, 50)):
        img = Image.new("RGB", (384, 384),
                        color=(random.randint(0, 255),
                               random.randint(0, 255),
                               random.randint(0, 255)))
        img.save(os.path.join(output_dir, "images", f"dummy_{i:05d}.jpg"))

    if stage == "pretrain":
        data = []
        for i in range(num_samples):
            data.append({
                "id": f"dummy_{i:05d}",
                "image": f"images/dummy_{i % 50:05d}.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nDescribe this image."},
                    {"from": "gpt", "value": f"This is a colorful image with index {i}."},
                ]
            })
        json_path = os.path.join(output_dir, "llava_pretrain_lcs558k.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Pretrain] 生成 {num_samples} 条模拟数据 -> {json_path}")

    elif stage == "sft":
        data = []
        for i in range(num_samples):
            data.append({
                "id": f"sft_{i:05d}",
                "image": f"images/dummy_{i % 50:05d}.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nWhat do you see in this image?"},
                    {"from": "gpt", "value": "I can see a colorful pattern in this image."},
                    {"from": "human", "value": "Can you describe the colors?"},
                    {"from": "gpt", "value": "The image contains a single dominant color."},
                ]
            })
        json_path = os.path.join(output_dir, "llava_v1_5_mix665k.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SFT] 生成 {num_samples} 条模拟数据 -> {json_path}")

    elif stage == "dpo":
        data = []
        for i in range(num_samples):
            data.append({
                "image": f"images/dummy_{i % 50:05d}.jpg",
                "question": "Describe what you see in this image.",
                "chosen": "The image shows a solid color background with uniform coloring.",
                "rejected": "The image shows a sunset over the ocean with dolphins jumping.",
            })
        json_path = os.path.join(output_dir, "rlhf_v_preference.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DPO] 生成 {num_samples} 条模拟数据 -> {json_path}")


# ============================================================================
# Section 9: 训练入口 (Training Entry Points)
# ============================================================================

def run_pretrain(model_config: ModelConfig, data_config: DataConfig,
                 train_config: PretrainConfig):
    logger = TrainLogger(output_dir=train_config.output_dir, stage_name="Stage1_Pretrain",
                         use_tensorboard=True)
    set_seed(train_config.seed)
    logger.log_info(f"GPU 信息: {get_gpu_info()}")

    logger.log_info("正在加载模型...")
    model = LiteVL(model_config)
    model.setup_for_stage("pretrain")
    logger.log_model_info(model)
    logger.log_config({"model": model_config.__dict__, "data": data_config.__dict__,
                        "train": train_config.__dict__})

    logger.log_info("正在加载预训练数据...")
    dataset = PretrainDataset(
        data_path=data_config.pretrain_data_path, image_dir=data_config.pretrain_image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.get_image_processor(),
        max_length=model_config.max_length,
    )
    dataloader = build_dataloader(dataset, batch_size=train_config.batch_size,
                                   num_workers=data_config.num_workers,
                                   collate_fn=collate_fn_pretrain)

    trainer = PretrainTrainer(model, dataloader, train_config, logger)
    trainer.train()
    logger.close()


def run_sft(model_config: ModelConfig, data_config: DataConfig,
            train_config: SFTConfig, pretrain_ckpt: str = None):
    logger = TrainLogger(output_dir=train_config.output_dir, stage_name="Stage2_SFT",
                         use_tensorboard=True)
    set_seed(train_config.seed)

    logger.log_info("正在加载模型...")
    model_config.use_lora = train_config.use_lora
    model = LiteVL(model_config)

    if pretrain_ckpt is None:
        pretrain_ckpt = os.path.join("outputs/stage1_pretrain/final", "projector.pt")
    if os.path.exists(pretrain_ckpt):
        logger.log_info(f"加载预训练 projector 权重: {pretrain_ckpt}")
        model.projector.load_state_dict(torch.load(pretrain_ckpt, map_location="cpu"))
    else:
        logger.log_warning(f"未找到预训练权重: {pretrain_ckpt}，使用随机初始化")

    model.setup_for_stage("sft")
    logger.log_model_info(model)
    logger.log_config({"model": model_config.__dict__, "data": data_config.__dict__,
                        "train": train_config.__dict__})

    logger.log_info("正在加载 SFT 数据...")
    dataset = SFTDataset(
        data_path=data_config.sft_data_path, image_dir=data_config.sft_image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.get_image_processor(),
        max_length=model_config.max_length,
    )
    dataloader = build_dataloader(dataset, batch_size=train_config.batch_size,
                                   num_workers=data_config.num_workers,
                                   collate_fn=collate_fn_sft)

    trainer = SFTTrainer(model, dataloader, train_config, logger)
    trainer.train()
    logger.close()


def run_dpo(model_config: ModelConfig, data_config: DataConfig,
            train_config: DPOConfig):
    logger = TrainLogger(output_dir=train_config.output_dir, stage_name="Stage3_DPO",
                         use_tensorboard=True)
    set_seed(train_config.seed)

    logger.log_info("正在加载 SFT 模型...")
    model_config.use_lora = train_config.use_lora
    model = LiteVL(model_config)

    sft_projector = os.path.join(train_config.sft_model_path, "projector.pt")
    if os.path.exists(sft_projector):
        logger.log_info(f"加载 SFT projector 权重: {sft_projector}")
        model.projector.load_state_dict(torch.load(sft_projector, map_location="cpu"))

    model.setup_for_stage("dpo")
    logger.log_model_info(model)
    logger.log_config({"model": model_config.__dict__, "data": data_config.__dict__,
                        "train": train_config.__dict__})

    logger.log_info("正在加载 DPO 偏好数据...")
    dataset = DPODataset(
        data_path=data_config.dpo_data_path, image_dir=data_config.dpo_image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.get_image_processor(),
        max_length=model_config.max_length,
    )
    dataloader = build_dataloader(dataset, batch_size=train_config.batch_size,
                                   num_workers=data_config.num_workers,
                                   collate_fn=collate_fn_dpo)

    trainer = DPOTrainer(model, dataloader, train_config, logger)
    trainer.train()
    logger.close()


# ============================================================================
# Section 10: 主入口 (Main)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LiteVL - 低成本视觉语言模型训练框架")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["pretrain", "sft", "dpo", "all", "dummy", "infer", "data_info"],
                        help="训练阶段 / 功能选择")
    parser.add_argument("--config", type=str, default=None,
                        help="自定义配置文件路径 (JSON)")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Stage 1 checkpoint 路径 (用于 Stage 2)")
    # 推理参数
    parser.add_argument("--model_path", type=str, default=None,
                        help="[infer] 模型路径")
    parser.add_argument("--image", type=str, default=None,
                        help="[infer] 图像路径")
    parser.add_argument("--question", type=str, default="Describe this image.",
                        help="[infer] 问题")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="[infer] 最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="[infer] 采样温度")
    # 模拟数据参数
    parser.add_argument("--num_samples", type=int, default=100,
                        help="[dummy] 模拟数据样本数")
    args = parser.parse_args()

    # 加载配置
    model_config = ModelConfig()
    data_config = DataConfig()

    if args.config:
        with open(args.config) as f:
            custom = json.load(f)
        for k, v in custom.get("model", {}).items():
            setattr(model_config, k, v)
        for k, v in custom.get("data", {}).items():
            setattr(data_config, k, v)

    # ---- 功能分支 ----

    if args.stage == "dummy":
        print("生成模拟数据用于流程测试...\n")
        for stage in ["pretrain", "sft", "dpo"]:
            generate_dummy_data(stage, args.num_samples)
        print("\n模拟数据生成完成! 可以运行:")
        print("  python LiteVL.py --stage pretrain")
        return

    if args.stage == "data_info":
        for stage_name, info in DATASETS_INFO.items():
            print(f"\n{'='*60}")
            print(f"[{stage_name.upper()}] {info['name']}")
            print(f"  描述: {info['description']}")
            print(f"  大小: {info['size']}")
            print(f"  来源: {info['url']}")
            print(f"  命令:\n{info['command']}")
        print(f"{'='*60}")
        return

    if args.stage == "infer":
        assert args.model_path, "--model_path 必须指定"
        assert args.image, "--image 必须指定"
        print("正在加载模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.model_path, device=device)
        print("模型加载完成!")
        response = chat(model, args.image, args.question, args.max_tokens, args.temperature)
        print(f"\nQ: {args.question}")
        print(f"A: {response}")
        return

    # ---- 训练流程 ----

    print("\n" + "=" * 60)
    print("  LiteVL - 低成本视觉语言模型训练框架")
    print("  架构: SigLIP + MLP Projector + Qwen2")
    print("=" * 60 + "\n")

    if args.stage in ("pretrain", "all"):
        print("[1/3] Stage 1: 预训练 - 特征对齐")
        run_pretrain(model_config, data_config, PretrainConfig())

    if args.stage in ("sft", "all"):
        print("[2/3] Stage 2: SFT - 指令微调")
        run_sft(model_config, data_config, SFTConfig(), args.pretrain_ckpt)

    if args.stage in ("dpo", "all"):
        print("[3/3] Stage 3: DPO - 偏好对齐")
        run_dpo(model_config, data_config, DPOConfig())

    print("\n训练完成!")


if __name__ == "__main__":
    main()
