"""
LiteVL - 低成本视觉语言模型训练
主入口: 支持三个训练阶段的统一启动

Usage:
    python train.py --stage pretrain   # Stage 1: 特征对齐预训练
    python train.py --stage sft        # Stage 2: 指令微调
    python train.py --stage dpo        # Stage 3: DPO偏好对齐
    python train.py --stage all        # 全流程训练
"""
import os
import sys
import argparse

import torch
import torch.distributed as dist

from configs.base_config import ModelConfig, DataConfig, PretrainConfig, SFTConfig, DPOConfig
from models.litevl import LiteVL
from data.dataset import (
    PretrainDataset, SFTDataset, DPODataset,
    collate_fn_pretrain, collate_fn_sft, collate_fn_dpo,
    build_dataloader,
)
from trainers.pretrain_trainer import PretrainTrainer
from trainers.sft_trainer import SFTTrainer
from trainers.dpo_trainer import DPOTrainer
from utils.logger import TrainLogger
from utils.train_utils import set_seed, get_gpu_info


def setup_distributed():
    """初始化分布式训练环境"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return -1
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank


def run_pretrain(model_config: ModelConfig, data_config: DataConfig,
                 train_config: PretrainConfig, local_rank: int = -1):
    """Stage 1: 预训练 - 特征对齐"""
    is_main = (local_rank <= 0)
    logger = TrainLogger(
        output_dir=train_config.output_dir,
        stage_name="Stage1_Pretrain",
        use_tensorboard=is_main,
    )

    set_seed(train_config.seed)
    if is_main:
        logger.log_info(f"GPU 信息: {get_gpu_info()}")

    # 构建模型
    if is_main:
        logger.log_info("正在加载模型...")
    model = LiteVL(model_config)
    model.setup_for_stage("pretrain")
    if is_main:
        logger.log_model_info(model)

    # 保存配置
    if is_main:
        logger.log_config({
            "model": model_config.__dict__,
            "data": data_config.__dict__,
            "train": train_config.__dict__,
        })

    # 构建数据集
    if is_main:
        logger.log_info("正在加载预训练数据...")
    dataset = PretrainDataset(
        data_path=data_config.pretrain_data_path,
        image_dir=data_config.pretrain_image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.get_image_processor(),
        max_length=model_config.max_length,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=train_config.batch_size,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn_pretrain,
        distributed=(local_rank >= 0),
    )

    # 训练
    trainer = PretrainTrainer(model, dataloader, train_config, logger, local_rank=local_rank)
    trainer.train()
    logger.close()


def run_sft(model_config: ModelConfig, data_config: DataConfig,
            train_config: SFTConfig, pretrain_ckpt: str = None):
    """Stage 2: SFT - 指令微调"""
    logger = TrainLogger(
        output_dir=train_config.output_dir,
        stage_name="Stage2_SFT",
        use_tensorboard=True,
    )

    set_seed(train_config.seed)

    # 构建模型
    logger.log_info("正在加载模型...")
    model_config.use_lora = train_config.use_lora
    model = LiteVL(model_config)

    # 加载 Stage 1 的 projector 权重
    if pretrain_ckpt is None:
        pretrain_ckpt = os.path.join("outputs/stage1_pretrain/final", "projector.pt")
    if os.path.exists(pretrain_ckpt):
        logger.log_info(f"加载预训练 projector 权重: {pretrain_ckpt}")
        model.projector.load_state_dict(torch.load(pretrain_ckpt, map_location="cpu"))
    else:
        logger.log_warning(f"未找到预训练权重: {pretrain_ckpt}，使用随机初始化")

    model.setup_for_stage("sft")
    logger.log_model_info(model)

    logger.log_config({
        "model": model_config.__dict__,
        "data": data_config.__dict__,
        "train": train_config.__dict__,
    })

    # 构建数据集
    logger.log_info("正在加载 SFT 数据...")
    dataset = SFTDataset(
        data_path=data_config.sft_data_path,
        image_dir=data_config.sft_image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.get_image_processor(),
        max_length=model_config.max_length,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=train_config.batch_size,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn_sft,
    )

    trainer = SFTTrainer(model, dataloader, train_config, logger)
    trainer.train()
    logger.close()


def run_dpo(model_config: ModelConfig, data_config: DataConfig,
            train_config: DPOConfig):
    """Stage 3: DPO - 偏好对齐"""
    logger = TrainLogger(
        output_dir=train_config.output_dir,
        stage_name="Stage3_DPO",
        use_tensorboard=True,
    )

    set_seed(train_config.seed)

    # 构建模型 (从 SFT checkpoint 加载)
    logger.log_info("正在加载 SFT 模型...")
    model_config.use_lora = train_config.use_lora
    model = LiteVL(model_config)

    # 加载 SFT 的 projector 权重
    sft_projector = os.path.join(train_config.sft_model_path, "projector.pt")
    if os.path.exists(sft_projector):
        logger.log_info(f"加载 SFT projector 权重: {sft_projector}")
        model.projector.load_state_dict(torch.load(sft_projector, map_location="cpu"))

    model.setup_for_stage("dpo")
    logger.log_model_info(model)

    logger.log_config({
        "model": model_config.__dict__,
        "data": data_config.__dict__,
        "train": train_config.__dict__,
    })

    # 构建数据集
    logger.log_info("正在加载 DPO 偏好数据...")
    dataset = DPODataset(
        data_path=data_config.dpo_data_path,
        image_dir=data_config.dpo_image_dir,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.get_image_processor(),
        max_length=model_config.max_length,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=train_config.batch_size,
        num_workers=data_config.num_workers,
        collate_fn=collate_fn_dpo,
    )

    trainer = DPOTrainer(model, dataloader, train_config, logger)
    trainer.train()
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="LiteVL Training")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["pretrain", "sft", "dpo", "all"],
                        help="训练阶段")
    parser.add_argument("--config", type=str, default=None,
                        help="自定义配置文件路径 (JSON)")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Stage 1 checkpoint 路径 (用于 Stage 2)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="DeepSpeed 分布式训练自动传入的 local rank")
    args = parser.parse_args()

    # 初始化分布式
    local_rank = setup_distributed()

    # 加载配置
    model_config = ModelConfig()
    data_config = DataConfig()

    # 如果提供了自定义配置，覆盖默认值
    if args.config:
        import json
        with open(args.config) as f:
            custom = json.load(f)
        for k, v in custom.get("model", {}).items():
            setattr(model_config, k, v)
        for k, v in custom.get("data", {}).items():
            setattr(data_config, k, v)

    if local_rank <= 0:
        print("\n" + "=" * 60)
        print("  LiteVL - 低成本视觉语言模型训练框架")
        print("  架构: SigLIP + MLP Projector + Qwen2")
        print("=" * 60 + "\n")

    if args.stage in ("pretrain", "all"):
        if local_rank <= 0:
            print("[1/3] Stage 1: 预训练 - 特征对齐")
        run_pretrain(model_config, data_config, PretrainConfig(), local_rank=local_rank)

    if args.stage in ("sft", "all"):
        if local_rank <= 0:
            print("[2/3] Stage 2: SFT - 指令微调")
        run_sft(model_config, data_config, SFTConfig(), args.pretrain_ckpt)

    if args.stage in ("dpo", "all"):
        if local_rank <= 0:
            print("[3/3] Stage 3: DPO - 偏好对齐")
        run_dpo(model_config, data_config, DPOConfig())

    if local_rank <= 0:
        print("\n训练完成!")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
