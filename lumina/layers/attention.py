import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
from .optical_linear import OpticalLinear
from ..exceptions import InvalidParameterError

# 定义有效的硬件配置类型
HardwareProfile = Literal['lumina_nano_v1', 'lumina_micro_v1', 'edge_ultra_low_power', 'datacenter_high_precision', 'custom']

class OpticalAttention(nn.Module):
    """
    Photonic-Accelerated Multi-Head Attention Layer.
    
    This layer implements the standard Multi-Head Attention mechanism but offloads
    the linear projections (Q, K, V, and Output) to the photonic core via OpticalLinear.
    
    The attention mechanism itself (Softmax(QK^T)) is currently performed digitally,
    as non-linear normalization is challenging in optics.
    
    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability on attn_output_weights. Default: 0.0.
        bias (bool): If specified, adds bias to input / output projection layers. Default: True.
        hardware_profile (str): The photonic hardware profile to simulate (e.g., 'lumina_nano_v1').
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        hardware_profile: HardwareProfile = "lumina_nano_v1",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != self.embed_dim:
            raise InvalidParameterError(f"embed_dim must be divisible by num_heads (got {embed_dim} and {num_heads})")

        # Optical Projections
        # We use separate layers for Q, K, V to allow for physical parallelization (WDM channels)
        self.q_proj = OpticalLinear(embed_dim, embed_dim, bias=bias, hardware_profile=hardware_profile)  # type: ignore[arg-type]
        self.k_proj = OpticalLinear(embed_dim, embed_dim, bias=bias, hardware_profile=hardware_profile)  # type: ignore[arg-type]
        self.v_proj = OpticalLinear(embed_dim, embed_dim, bias=bias, hardware_profile=hardware_profile)  # type: ignore[arg-type]
        self.out_proj = OpticalLinear(embed_dim, embed_dim, bias=bias, hardware_profile=hardware_profile)  # type: ignore[arg-type]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query embeddings of shape (batch, target_len, embed_dim)
            key: Key embeddings of shape (batch, source_len, embed_dim)
            value: Value embeddings of shape (batch, source_len, embed_dim)
            key_padding_mask: If specified, a mask of shape (batch, source_len) indicating which elements within key to ignore.
            need_weights: Output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions.
            is_causal: If True, applies a causal mask (auto-regressive).
            
        Returns:
            attn_output: (batch, target_len, embed_dim)
            attn_output_weights: (batch, target_len, source_len)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # 1. Optical Projections (The "Heavy Lifting")
        # [batch, seq_len, embed_dim]
        # These operations run on the photonic core (simulated or real)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Reshape for Multi-head
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention (Digital)
        # [batch, num_heads, tgt_len, src_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if is_causal:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(tgt_len, src_len, device=query.device) * float('-inf'), diagonal=1)
            attn_weights += causal_mask.unsqueeze(0).unsqueeze(0)

        if attn_mask is not None:
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask is usually boolean (True = ignore), we need to fill with -inf
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # [batch, num_heads, tgt_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # 4. Reshape back
        # [batch, tgt_len, num_heads, head_dim] -> [batch, tgt_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)

        # 5. Optical Output Projection
        attn_output = self.out_proj(attn_output)

        if average_attn_weights and need_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return attn_output, attn_weights
