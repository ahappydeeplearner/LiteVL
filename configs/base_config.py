"""
LiteVL - 低成本视觉语言模型训练配置
基于 LLaVA 架构: SigLIP (Vision Encoder) + MLP Projector + Qwen2 (LLM)
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型架构配置"""
    # Vision Encoder
    vision_encoder_name: str = "models/siglip-so400m-patch14-384"
    vision_hidden_size: int = 1152
    image_size: int = 384
    patch_size: int = 14
    freeze_vision_encoder: bool = True

    # MLP Projector
    projector_type: str = "mlp2x_gelu"  # 2层MLP + GELU
    projector_hidden_size: int = 4096

    # LLM Backbone
    llm_name: str = "models/Qwen2-0.5B-Instruct"
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
    # Stage 1: 预训练数据 (特征对齐)
    pretrain_data_path: str = "data/pretrain/blip_laion_cc_sbu_558k.json"
    pretrain_image_dir: str = "data/pretrain/images"

    # Stage 2: SFT数据 (指令微调)
    sft_data_path: str = "data/sft/llava_v1_5_mix665k.json"
    sft_image_dir: str = "data/sft/images"

    # Stage 3: DPO数据 (偏好对齐)
    dpo_data_path: str = "data/dpo/rlhf_v_preference.json"
    dpo_image_dir: str = "data/dpo/images"

    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class PretrainConfig:
    """Stage 1: 预训练 - 特征对齐"""
    output_dir: str = "outputs/stage1_pretrain"
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    deepspeed_config: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 200
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
    # LoRA
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
    # 加载 Stage 2 的 checkpoint
    sft_model_path: str = "outputs/stage2_sft/final"
