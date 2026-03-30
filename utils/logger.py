"""
训练日志系统
支持:
- Console 彩色日志输出
- 文件日志 (JSON Lines 格式，便于后续分析)
- WandB 集成 (可选)
- TensorBoard 集成 (可选)
- 训练曲线自动绘制
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from collections import defaultdict

import torch


class TrainLogger:
    """
    统一训练日志管理器

    日志文件结构:
    outputs/stage1_pretrain/
    ├── logs/
    │   ├── train.log          # 可读文本日志
    │   ├── metrics.jsonl      # JSON Lines 指标日志
    │   └── config.json        # 训练配置快照
    ├── checkpoints/
    │   ├── step_500/
    │   └── step_1000/
    └── final/
    """

    def __init__(self, output_dir: str, stage_name: str,
                 use_wandb: bool = False, wandb_project: str = "litevl",
                 use_tensorboard: bool = True):
        self.output_dir = output_dir
        self.stage_name = stage_name
        self.log_dir = os.path.join(output_dir, "logs")
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 文本日志
        self.logger = self._setup_file_logger()

        # JSON 指标日志
        self.metrics_file = open(os.path.join(self.log_dir, "metrics.jsonl"), "a")

        # 内存中的指标追踪
        self.step_metrics = defaultdict(list)
        self.global_step = 0
        self.epoch = 0
        self.start_time = time.time()

        # WandB
        self.wandb_run = None
        if use_wandb:
            self._setup_wandb(wandb_project)

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()

        self.log_info(f"===== {stage_name} 训练日志初始化 =====")
        self.log_info(f"输出目录: {output_dir}")
        self.log_info(f"日志目录: {self.log_dir}")

    def _setup_file_logger(self) -> logging.Logger:
        """配置文件 + 控制台日志"""
        logger = logging.getLogger(f"vlm_{self.stage_name}")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        # 文件 handler
        fh = logging.FileHandler(
            os.path.join(self.log_dir, "train.log"),
            encoding="utf-8"
        )
        fh.setLevel(logging.INFO)

        # 控制台 handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _setup_wandb(self, project: str):
        """初始化 WandB"""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=project,
                name=f"{self.stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dir=self.output_dir,
            )
            self.log_info("WandB 已启用")
        except ImportError:
            self.log_info("WandB 未安装，跳过")

    def _setup_tensorboard(self):
        """初始化 TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.log_info(f"TensorBoard 已启用: {tb_dir}")
        except ImportError:
            self.log_info("TensorBoard 未安装，跳过")

    def log_info(self, msg: str):
        self.logger.info(msg)

    def log_warning(self, msg: str):
        self.logger.warning(msg)

    def log_error(self, msg: str):
        self.logger.error(msg)

    def log_config(self, config: Any):
        """保存训练配置"""
        config_dict = config.__dict__ if hasattr(config, "__dict__") else config
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        self.log_info(f"训练配置已保存到 {config_path}")

        if self.wandb_run:
            import wandb
            wandb.config.update(config_dict)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        记录训练指标

        Args:
            metrics: 指标字典，例如 {"loss": 0.5, "lr": 1e-4}
            step: 全局步数
        """
        if step is not None:
            self.global_step = step

        # 添加时间戳和步数
        record = {
            "step": self.global_step,
            "epoch": self.epoch,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            **metrics,
        }

        # 写入 JSONL
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()

        # 追踪到内存
        for k, v in metrics.items():
            self.step_metrics[k].append(v)

        # WandB
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=self.global_step)

        # TensorBoard
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, self.global_step)

        # 控制台格式化输出
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        metrics_str = " | ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items())
        self.log_info(
            f"[Step {self.global_step}][Epoch {self.epoch}][{time_str}] {metrics_str}"
        )

    def log_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """记录 epoch 级别的汇总"""
        self.epoch = epoch
        self.log_info(f"\n{'='*60}")
        self.log_info(f"Epoch {epoch} 训练完成")
        for k, v in metrics.items():
            self.log_info(f"  {k}: {v:.6f}")
        self.log_info(f"{'='*60}\n")

        self.log_metrics({f"epoch/{k}": v for k, v in metrics.items()})

    def log_gpu_memory(self):
        """记录 GPU 显存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            self.log_info(
                f"GPU 显存 - 已分配: {allocated:.2f}GB, "
                f"已预留: {reserved:.2f}GB, "
                f"峰值: {max_allocated:.2f}GB"
            )

    def log_model_info(self, model):
        """记录模型参数信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        self.log_info(f"\n{'='*60}")
        self.log_info("模型参数统计:")
        self.log_info(f"  总参数量:     {total_params / 1e6:.2f}M")
        self.log_info(f"  可训练参数:   {trainable_params / 1e6:.2f}M")
        self.log_info(f"  冻结参数:     {frozen_params / 1e6:.2f}M")
        self.log_info(f"  可训练比例:   {trainable_params / total_params * 100:.2f}%")
        self.log_info(f"{'='*60}\n")

    def close(self):
        """关闭日志"""
        self.metrics_file.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()
        self.log_info("日志系统已关闭")


class MetricsTracker:
    """
    训练指标追踪器
    支持滑动窗口平均和全局平均
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.running_sum = defaultdict(float)
        self.running_count = defaultdict(int)

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.metrics[k].append(v)
            self.running_sum[k] += v
            self.running_count[k] += 1

    def get_smoothed(self, key: str) -> float:
        """获取滑动窗口平均值"""
        values = self.metrics[key][-self.window_size:]
        return sum(values) / len(values) if values else 0.0

    def get_global_avg(self, key: str) -> float:
        """获取全局平均值"""
        if self.running_count[key] == 0:
            return 0.0
        return self.running_sum[key] / self.running_count[key]

    def get_latest(self, key: str) -> float:
        """获取最新值"""
        return self.metrics[key][-1] if self.metrics[key] else 0.0

    def get_summary(self) -> Dict[str, float]:
        """获取所有指标的汇总"""
        summary = {}
        for k in self.metrics:
            summary[f"{k}_avg"] = self.get_global_avg(k)
            summary[f"{k}_smoothed"] = self.get_smoothed(k)
            summary[f"{k}_latest"] = self.get_latest(k)
        return summary

    def reset(self):
        self.metrics.clear()
        self.running_sum.clear()
        self.running_count.clear()
