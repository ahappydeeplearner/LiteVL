# LiteVL

A lightweight vision-language model training framework based on the LLaVA architecture.

## Architecture

```
SigLIP-SO400M (Vision Encoder, frozen) --> MLP Projector --> Qwen2-0.5B (LLM)
```

- **Vision Encoder**: SigLIP-SO400M-patch14-384, outputs 729 visual tokens per image, always frozen
- **MLP Projector**: 2-layer MLP (1152-dim -> 896-dim) with GELU activation
- **LLM**: Qwen2-0.5B-Instruct, frozen in Stage 1, LoRA fine-tuned in Stages 2 & 3

## Three-Stage Training Pipeline

| Stage | Task | Trainable Params | Data |
|-------|------|-----------------|------|
| Stage 1 - Pretrain | Feature alignment | Projector (~1.8M) | LCS-558K image-caption pairs |
| Stage 2 - SFT | Instruction fine-tuning | Projector + LoRA (~50M) | LLaVA-mix-665K conversations |
| Stage 3 - DPO | Preference alignment | Projector + LoRA (~50M) | RLHF-V preference pairs |

- **Stage 1**: 训练 MLP Projector 将视觉特征映射到 LLM 嵌入空间，Vision Encoder 和 LLM 均冻结
- **Stage 2**: 使用 LoRA 微调 LLM + Projector，赋予模型多轮对话和指令遵循能力
- **Stage 3**: 通过 DPO 算法进行偏好对齐，减少幻觉，提升回复质量

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional dependencies (uncomment in `requirements.txt` if needed):
- `wandb` -- Weights & Biases logging
- `flash-attn` -- Flash Attention 2 for faster training

### 2. Prepare Data

Generate dummy data for quick testing:

```bash
python prepare_data.py --dummy
```

Or follow the instructions to download real datasets:

```bash
python prepare_data.py
```

Real datasets:
- **Pretrain**: [LLaVA-Pretrain LCS-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) (~15GB)
- **SFT**: [LLaVA-v1.5-mix665k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) (~50GB)
- **DPO**: [RLHF-V Dataset](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) (~5GB)

### 3. Training

Run all three stages sequentially:

```bash
bash scripts/run_all.sh
```

Or run each stage individually:

```bash
bash scripts/run_pretrain.sh   # Stage 1
bash scripts/run_sft.sh        # Stage 2
bash scripts/run_dpo.sh        # Stage 3
```

Multi-GPU distributed training:

```bash
bash scripts/run_distributed.sh <stage> <num_gpus>
# e.g. bash scripts/run_distributed.sh pretrain 8
```

You can also use the CLI directly:

```bash
python train.py --stage pretrain
python train.py --stage sft
python train.py --stage dpo
python train.py --stage all        # all stages sequentially
python train.py --stage sft --config my_config.json  # custom config override
```

### 4. Inference

Stage 1 pretrain model (caption-style prompt):

```bash
python inference.py \
    --model_path outputs/stage1_pretrain/final \
    --image photo.jpg \
    --question "Describe this image." \
    --prompt_style pretrain
```

Stage 2/3 SFT/DPO model (ChatML dialog format, default):

```bash
python inference.py \
    --model_path outputs/stage3_dpo/final \
    --image photo.jpg \
    --question "Describe this image in detail."
```

## Single-File Version

`LiteVL.py` is a self-contained monolithic version of the entire framework:

```bash
python LiteVL.py --stage dummy   # generate test data
python LiteVL.py --stage infer   # run inference
python LiteVL.py --stage all     # full training pipeline
```

## Project Structure

```
LiteVL/
├── models/
│   ├── litevl.py           # Main LiteVL model
│   ├── projector.py        # MLP projector variants
│   └── vision_encoder.py   # SigLIP vision encoder wrapper
├── trainers/
│   ├── pretrain_trainer.py  # Stage 1 trainer
│   ├── sft_trainer.py       # Stage 2 trainer
│   └── dpo_trainer.py       # Stage 3 trainer
├── data/
│   └── dataset.py           # Dataset classes for all stages
├── configs/
│   ├── base_config.py       # Dataclass-based configurations
│   └── ds_config_zero2.json # DeepSpeed ZeRO-2 config
├── utils/
│   ├── logger.py            # Console/File/TensorBoard/WandB logging
│   └── train_utils.py       # Seed, LR scheduler, checkpoint utils
├── scripts/                 # Shell scripts for training
├── train.py                 # Main training entry point
├── inference.py             # Inference script
├── prepare_data.py          # Data preparation & dummy data generation
├── LiteVL.py               # Single-file version
├── setup.py
└── requirements.txt
```

## Key Configurations

Configurations are defined as dataclasses in `configs/base_config.py`:

- **ModelConfig**: Model paths, projector type, LoRA settings (rank=64, alpha=16)
- **DataConfig**: Dataset paths, image size (384), max sequence length (2048)
- **PretrainConfig**: lr=1e-3, batch_size=1, gradient_accumulation=32, cosine schedule, bf16
- **SFTConfig**: lr=2e-5, batch_size=16, gradient_accumulation=2, bf16
- **DPOConfig**: lr=5e-7, beta=0.1, batch_size=8, gradient_accumulation=4, bf16

Override any config via a JSON file with `--config`:

```bash
python train.py --stage sft --config '{"learning_rate": 1e-5, "num_epochs": 2}'
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- Transformers >= 4.40.0
- PEFT >= 0.10.0
- DeepSpeed >= 0.14.0
- NVIDIA GPU with sufficient VRAM (A100-80GB recommended for multi-GPU training)
