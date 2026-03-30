"""
MLP Projector 模块
将视觉特征映射到 LLM 的嵌入空间
这是预训练阶段唯一训练的模块，也是低成本训练的核心
"""
import torch
import torch.nn as nn
import math


class MLPProjector(nn.Module):
    """
    2层 MLP + GELU 激活，将视觉 token 映射到语言模型空间
    参考 LLaVA-1.5 的设计，简单有效
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()
        self.projector_type = projector_type

        if projector_type == "mlp2x_gelu":
            # 标准 2层 MLP，中间维度为 LLM hidden size
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        elif projector_type == "mlp2x_gelu_compress":
            # 带 token 压缩的 MLP (降低 visual token 数量)
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
            self.token_compressor = TokenCompressor(llm_hidden_size, compress_ratio=4)
        elif projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, N_patches, vision_hidden_size]
        Returns:
            projected_features: [B, N_tokens, llm_hidden_size]
        """
        projected = self.projector(vision_features)

        if self.projector_type == "mlp2x_gelu_compress":
            projected = self.token_compressor(projected)

        return projected


class TokenCompressor(nn.Module):
    """
    视觉 Token 压缩模块
    通过 reshape + linear 将 N 个 visual token 压缩为 N/ratio 个
    降低 LLM 的计算负担，进一步降低训练成本
    """

    def __init__(self, hidden_size: int, compress_ratio: int = 4):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.compress_proj = nn.Linear(hidden_size * compress_ratio, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            compressed: [B, N//ratio, D]
        """
        B, N, D = x.shape
        assert N % self.compress_ratio == 0, \
            f"Token数 {N} 必须能被压缩比 {self.compress_ratio} 整除"

        # Reshape: [B, N, D] -> [B, N//ratio, D*ratio]
        x = x.reshape(B, N // self.compress_ratio, D * self.compress_ratio)
        # Project back: [B, N//ratio, D*ratio] -> [B, N//ratio, D]
        x = self.compress_proj(x)
        x = self.norm(x)
        return x
