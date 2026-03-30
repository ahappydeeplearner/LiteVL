#!/bin/bash
# ============================================================
# LiteVL - Stage 2: SFT (指令微调)
# 训练内容: MLP Projector + LLM (LoRA)
# 冻结: Vision Encoder
# 数据: LLaVA-mix-665K
# 预估训练时间: ~8 GPU-hours (A100-80G)
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

cd "$(dirname $(dirname $(realpath $0)))"

echo "=========================================="
echo "  Stage 2: SFT - 指令微调"
echo "=========================================="

python train.py --stage sft

echo "Stage 2 完成!"
