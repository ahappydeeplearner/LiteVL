"""
Vision Encoder 模块
使用 SigLIP-SO400M 作为视觉编码器，参数冻结以降低训练成本
"""
import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor


class VisionEncoder(nn.Module):
    """
    基于 SigLIP 的视觉编码器
    SigLIP 相比 CLIP 的优势:
    - 使用 Sigmoid Loss 替代 Softmax，无需全局归一化
    - 更好的多语言支持
    - 在相同参数量下性能更优
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384",
                 freeze: bool = True):
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        self._frozen = False

        if freeze:
            self.freeze()

    def freeze(self):
        """冻结视觉编码器参数 - 低成本训练的关键"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self._frozen = True
        self.vision_model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]
        Returns:
            vision_features: [B, N_patches, hidden_size]
        """
        if self._frozen:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
                vision_features = outputs.last_hidden_state
        else:
            outputs = self.vision_model(pixel_values=pixel_values)
            # 使用 last_hidden_state 而非 pooler_output，保留空间信息
            vision_features = outputs.last_hidden_state
        return vision_features

    def get_image_processor(self):
        return self.image_processor

    @property
    def num_patches(self):
        """计算图像 patch 数量"""
        image_size = self.vision_model.config.image_size
        patch_size = self.vision_model.config.patch_size
        return (image_size // patch_size) ** 2
