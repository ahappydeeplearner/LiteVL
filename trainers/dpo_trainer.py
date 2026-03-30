"""
Stage 3: DPO (Direct Preference Optimization) Trainer
目标: 偏好对齐 - 减少幻觉，提升回复质量
冻结: Vision Encoder
训练: MLP Projector + LLM (LoRA)
数据: 偏好对数据 (chosen/rejected)

DPO Loss:
  L_DPO = -E[log σ(β * (log π(y_w|x) / π_ref(y_w|x) - log π(y_l|x) / π_ref(y_l|x)))]
  其中 y_w 是 chosen (优质回复), y_l 是 rejected (低质回复)
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import TrainLogger, MetricsTracker
from utils.train_utils import (
    set_seed, get_lr_scheduler, save_checkpoint, get_gpu_info
)


class DPOTrainer:
    def __init__(self, model, train_dataloader, config, logger: TrainLogger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.logger = logger
        self.metrics = MetricsTracker(window_size=50)
        self.beta = config.beta  # DPO 温度参数

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Reference model (SFT model 的冻结副本)
        self.ref_model = self._create_ref_model()

        # 优化器
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

        self.use_bf16 = config.bf16
        self.global_step = 0

    def _create_ref_model(self):
        """创建 reference model (冻结的 SFT 模型副本)"""
        ref_model = copy.deepcopy(self.model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        ref_model.to(self.device)
        return ref_model

    def _get_log_probs(self, model, input_ids, pixel_values, attention_mask):
        """计算模型的 log probabilities"""
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        logits = outputs["logits"]

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].clamp(min=0).contiguous()

        # 计算 per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

        # 只计算 response 部分的 log probs (mask 掉 prompt 部分)
        # 使用 attention_mask 的简化版本
        mask = attention_mask[:, 1:].float()
        sum_log_probs = (token_log_probs * mask).sum(dim=-1)

        return sum_log_probs

    def _compute_dpo_loss(self, batch):
        """
        计算 DPO 损失

        DPO Loss = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
        """
        pixel_values = batch["pixel_values"].to(self.device)
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)

        # Policy model log probs
        policy_chosen_logps = self._get_log_probs(
            self.model, chosen_ids, pixel_values, chosen_mask)
        policy_rejected_logps = self._get_log_probs(
            self.model, rejected_ids, pixel_values, rejected_mask)

        # Reference model log probs
        with torch.no_grad():
            ref_chosen_logps = self._get_log_probs(
                self.ref_model, chosen_ids, pixel_values, chosen_mask)
            ref_rejected_logps = self._get_log_probs(
                self.ref_model, rejected_ids, pixel_values, rejected_mask)

        # DPO loss
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # 额外指标
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        chosen_reward = chosen_rewards.mean().item()
        rejected_reward = rejected_rewards.mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        return loss, {
            "reward_margin": reward_margin,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "accuracy": accuracy,
        }

    def train(self):
        """执行 DPO 训练"""
        self.logger.log_info("=" * 60)
        self.logger.log_info("Stage 3: DPO - 偏好对齐")
        self.logger.log_info(f"GPU: {get_gpu_info()}")
        self.logger.log_info(f"DPO beta: {self.beta}")
        self.logger.log_model_info(self.model)
        self.logger.log_gpu_memory()
        self.logger.log_info("=" * 60)

        self.model.train()
        self.model.vision_encoder.eval()

        for epoch in range(self.config.num_epochs):
            self.logger.epoch = epoch
            epoch_metrics = MetricsTracker()

            for step, batch in enumerate(self.train_dataloader):
                if self.use_bf16:
                    device_type = "cuda" if self.device.type == "cuda" else "cpu"
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss, extra_metrics = self._compute_dpo_loss(batch)
                        loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss, extra_metrics = self._compute_dpo_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()

                loss_val = loss.item() * self.config.gradient_accumulation_steps
                self.metrics.update({"loss": loss_val, **extra_metrics})
                epoch_metrics.update({"loss": loss_val, **extra_metrics})

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
                            "train/reward_margin": self.metrics.get_smoothed("reward_margin"),
                            "train/chosen_reward": self.metrics.get_smoothed("chosen_reward"),
                            "train/rejected_reward": self.metrics.get_smoothed("rejected_reward"),
                            "train/accuracy": self.metrics.get_smoothed("accuracy"),
                            "train/lr": self.scheduler.get_last_lr()[0],
                        }, step=self.global_step)

                    if self.global_step % self.config.save_steps == 0:
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, epoch, self.config.output_dir,
                        )

            self.logger.log_epoch_summary(epoch, {
                "loss": epoch_metrics.get_global_avg("loss"),
                "reward_margin": epoch_metrics.get_global_avg("reward_margin"),
                "accuracy": epoch_metrics.get_global_avg("accuracy"),
            })
            self.logger.log_gpu_memory()

        final_dir = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(final_dir)
        self.logger.log_info(f"DPO 训练完成! 模型已保存到 {final_dir}")
