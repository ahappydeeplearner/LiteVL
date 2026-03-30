"""
训练工具函数
"""
import os
import random
import torch
import numpy as np
from typing import Optional


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
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

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

    # 保存模型
    model.save_pretrained(ckpt_dir)

    # 保存训练状态
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
    import json

    # 加载 projector
    projector_path = os.path.join(ckpt_dir, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    # 加载训练状态
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
