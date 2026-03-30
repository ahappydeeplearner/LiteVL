"""
Stage 2: SFT (Supervised Fine-Tuning) Trainer
目标: 指令微调 - 赋予模型多模态对话能力
冻结: Vision Encoder
训练: MLP Projector + LLM (LoRA)
数据: LLaVA-mix-665K 指令数据
"""
import os
import torch
from torch.cuda.amp import GradScaler

from utils.logger import TrainLogger, MetricsTracker
from utils.train_utils import (
    set_seed, get_lr_scheduler, save_checkpoint, get_gpu_info
)


class SFTTrainer:
    def __init__(self, model, train_dataloader, config, logger: TrainLogger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.logger = logger
        self.metrics = MetricsTracker(window_size=100)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器 - projector + LLM (LoRA) 参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_training_steps = (
            len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        )
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.lr_scheduler_type,
            num_training_steps,
            config.warmup_ratio,
        )

        self.scaler = GradScaler() if config.fp16 else None
        self.use_bf16 = config.bf16
        self.global_step = 0

    def train(self):
        """执行 SFT 训练"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 2: SFT - 指令微调")
        self.logger.log_info(f"GPU: {get_gpu_info()}")
        self.logger.log_model_info(self.model)
        self.logger.log_gpu_memory()
        self.logger.log_info("=" * 60)

        self.model.train()
        self.model.vision_encoder.eval()

        for epoch in range(self.config.num_epochs):
            self.logger.epoch = epoch
            epoch_metrics = MetricsTracker()

            for step, batch in enumerate(self.train_dataloader):
                loss = self._train_step(batch)

                if loss is not None:
                    self.metrics.update({"loss": loss})
                    epoch_metrics.update({"loss": loss})

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        self.logger.log_metrics({
                            "train/loss": self.metrics.get_smoothed("loss"),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(self.train_dataloader),
                        }, step=self.global_step)

                    if self.global_step % self.config.save_steps == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, self.config.output_dir,
                        )

            self.logger.log_epoch_summary(epoch, {
                "loss": epoch_metrics.get_global_avg("loss"),
            })
            self.logger.log_gpu_memory()

        final_dir = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(final_dir)
        self.logger.log_info(f"SFT 训练完成! 模型已保存到 {final_dir}")

    def _train_step(self, batch) -> float:
        """单步训练"""
        input_ids = batch["input_ids"].to(self.device)
        pixel_values = batch["pixel_values"]
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
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
        else:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        loss.backward()
        return loss.item() * self.config.gradient_accumulation_steps
