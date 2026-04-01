"""
Stage 1: 预训练 Trainer
目标: 特征对齐 - 训练 MLP Projector 将视觉特征映射到 LLM 嵌入空间
冻结: Vision Encoder + LLM
训练: MLP Projector (~10M 参数)
数据: LCS-558K 图文对
"""
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.logger import TrainLogger, MetricsTracker
from utils.train_utils import (
    set_seed, get_lr_scheduler, save_checkpoint, get_gpu_info
)


class PretrainTrainer:
    def __init__(self, model, train_dataloader, config, logger: TrainLogger, local_rank: int = -1):
        self.config = config
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.metrics = MetricsTracker(window_size=100)
        self.local_rank = local_rank
        self.is_main = (local_rank <= 0)

        # 设备
        if local_rank >= 0:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

        # DDP 包装
        if local_rank >= 0:
            self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            self.model = model
        # 保留原始模型引用，用于保存和获取 projector 参数
        self.raw_model = model

        # 优化器 - 只优化 projector 参数
        self.optimizer = torch.optim.AdamW(
            [p for p in self.raw_model.projector.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 学习率调度
        num_training_steps = (
            len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.lr_scheduler_type,
            num_training_steps,
            config.warmup_ratio,
        )

        # 混合精度
        self.scaler = GradScaler() if config.fp16 else None
        self.use_bf16 = config.bf16

        self.global_step = 0

    def train(self):
        """执行预训练"""
        if self.is_main:
            self.logger.log_info("=" * 60)
            self.logger.log_info("Stage 1: 预训练 - 特征对齐")
            self.logger.log_info(f"GPU: {get_gpu_info()}")
            self.logger.log_model_info(self.raw_model)
            self.logger.log_gpu_memory()
            self.logger.log_info("=" * 60)

        self.model.train()
        # 但 vision encoder 和 LLM 保持 eval 模式
        self.raw_model.vision_encoder.eval()
        self.raw_model.llm.eval()

        for epoch in range(self.config.num_epochs):
            self.logger.epoch = epoch
            epoch_metrics = MetricsTracker()

            for step, batch in enumerate(self.train_dataloader):
                loss = self._train_step(batch)

                if loss is not None:
                    self.metrics.update({"loss": loss})
                    epoch_metrics.update({"loss": loss})

                # 梯度累积
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.raw_model.projector.parameters(),
                            self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # 日志
                    if self.global_step % self.config.logging_steps == 0 and self.is_main:
                        self.logger.log_metrics({
                            "train/loss": self.metrics.get_smoothed("loss"),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(self.train_dataloader),
                        }, step=self.global_step)

                    # 保存 checkpoint
                    if self.global_step % self.config.save_steps == 0 and self.is_main:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, self.config.output_dir,
                        )

            # Epoch 结束
            if self.is_main:
                self.logger.log_epoch_summary(epoch, {
                    "loss": epoch_metrics.get_global_avg("loss"),
                })
                self.logger.log_gpu_memory()

        # 保存最终模型
        if self.is_main:
            final_dir = os.path.join(self.config.output_dir, "final")
            self.raw_model.save_pretrained(final_dir)
            self.logger.log_info(f"预训练完成! 模型已保存到 {final_dir}")

    def _train_step(self, batch) -> float:
        """单步训练"""
        # 移动数据到 GPU
        input_ids = batch["input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.use_bf16:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
        elif self.scaler:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps
