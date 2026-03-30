#!/bin/bash
# ============================================================
# LiteVL - Stage 1: 预训练 (特征对齐)
# 训练内容: MLP Projector
# 冻结: Vision Encoder + LLM
# 数据: LCS-558K 图文对
# 预估训练时间: ~2 GPU-hours (A100-80G)
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

cd "$(dirname $(dirname $(realpath $0)))"

echo "=========================================="
echo "  Stage 1: 预训练 - 特征对齐"
echo "=========================================="

python train.py --stage pretrain

echo "Stage 1 完成!"
