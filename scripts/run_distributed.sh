#!/bin/bash
# ============================================================
# DeepSpeed 分布式训练脚本 (多卡)
# ============================================================

set -e

export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"
cd "$(dirname $(dirname $(realpath $0)))"

STAGE=${1:-"pretrain"}
NUM_GPUS=${2:-2}

echo "=========================================="
echo "  分布式训练: ${STAGE} (${NUM_GPUS} GPUs)"
echo "=========================================="

deepspeed --num_gpus=${NUM_GPUS} train.py \
    --stage ${STAGE}
