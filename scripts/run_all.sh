#!/bin/bash
# ============================================================
# LiteVL - 全流程训练
# Stage 1 -> Stage 2 -> Stage 3
# 预估总训练时间: ~12 GPU-hours (A100-80G)
# ============================================================

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

cd "$(dirname $(dirname $(realpath $0)))"

echo "============================================================"
echo "  LiteVL - 低成本视觉语言模型全流程训练"
echo "  架构: SigLIP + MLP Projector + Qwen2-0.5B"
echo "============================================================"
echo ""

# Step 0: 准备模拟数据 (可选，用于快速测试)
# python prepare_data.py --dummy --num_samples 100

# Step 1: 预训练
echo "[1/3] Stage 1: 预训练 - 特征对齐..."
python train.py --stage pretrain
echo "[1/3] Stage 1 完成!"
echo ""

# Step 2: SFT
echo "[2/3] Stage 2: SFT - 指令微调..."
python train.py --stage sft
echo "[2/3] Stage 2 完成!"
echo ""

# Step 3: DPO
echo "[3/3] Stage 3: DPO - 偏好对齐..."
python train.py --stage dpo
echo "[3/3] Stage 3 完成!"
echo ""

echo "============================================================"
echo "  全流程训练完成!"
echo "  模型输出:"
echo "    Stage 1: outputs/stage1_pretrain/final/"
echo "    Stage 2: outputs/stage2_sft/final/"
echo "    Stage 3: outputs/stage3_dpo/final/"
echo "============================================================"
