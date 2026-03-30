# LiteVL

A lightweight vision-language model training framework based on the LLaVA architecture. Trains a complete VLM on a single A100-80GB GPU in approximately 12 GPU-hours across three stages.

## Architecture

```
SigLIP-SO400M (Vision Encoder, frozen) --> MLP Projector --> Qwen2-0.5B (LLM)
```

- **Vision Encoder**: SigLIP-SO400M-patch14-384, outputs 729 visual tokens per image, always frozen
- **MLP Projector**: 2-layer MLP (1152-dim -> 896-dim) with GELU activation, ~10M parameters
- **LLM**: Qwen2-0.5B-Instruct, frozen in Stage 1, LoRA fine-tuned in Stages 2 & 3

## Three-Stage Training Pipeline

| Stage | Task | Trainable Params | Data | GPU Hours |
|-------|------|-----------------|------|-----------|
| Stage 1 - Pretrain | Feature alignment | Projector (~10M) | LCS-558K image-caption pairs | ~2h |
| Stage 2 - SFT | Instruction fine-tuning | Projector + LoRA (~50M) | LLaVA-mix-665K conversations | ~8h |
| Stage 3 - DPO | Preference alignment | Projector + LoRA | RLHF-V preference pairs | ~2h |

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

Multi-GPU distributed training with DeepSpeed:

```bash
bash scripts/run_distributed.sh <stage> <num_gpus>
# e.g. bash scripts/run_distributed.sh sft 4
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

```bash
python inference.py \
    --model_path outputs/stage3_dpo/final \
    --image photo.jpg \
    --question "Describe this image."
```

## Single-File Version

`LiteVL.py` is a self-contained monolithic version of the entire framework. It supports all training stages plus:

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
- **DataConfig**: Dataset paths, image size (384), max sequence length
- **PretrainConfig**: lr=1e-3, batch_size=32, cosine schedule
- **SFTConfig**: lr=2e-5, batch_size=16, gradient_accumulation=2
- **DPOConfig**: lr=5e-7, beta=0.1, gradient_accumulation=4

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
- 1x NVIDIA A100-80GB (or equivalent) for single-GPU training
