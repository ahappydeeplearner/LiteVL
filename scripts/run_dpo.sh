#!/bin/bash
# ============================================================
# LiteVL - Stage 3: DPO (偏好对齐)
# 训练内容: MLP Projector + LLM (LoRA)
# 冻结: Vision Encoder
# 数据: RLHF-V 偏好数据
# 预估训练时间: ~2 GPU-hours (A100-80G)
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

cd "$(dirname $(dirname $(realpath $0)))"

echo "=========================================="
echo "  Stage 3: DPO - 偏好对齐"
echo "=========================================="

python train.py --stage dpo

echo "Stage 3 完成!"
