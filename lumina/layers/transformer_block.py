import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal
from .optical_linear import OpticalLinear
from .attention import OpticalAttention, HardwareProfile
from ..exceptions import InvalidParameterError

class OpticalTransformerBlock(nn.Module):
    """
    Photonic-Accelerated Transformer Block.
    
    Combines OpticalAttention and an Optical MLP.
    Follows the standard Pre-Norm architecture (GPT-2/3 style), which is generally
    more stable for training deep networks.
    
    Structure:
    x -> LayerNorm -> OpticalAttention -> Residual -> x
    x -> LayerNorm -> OpticalMLP -> Residual -> x
    
    The linear transformations in both Attention and MLP are offloaded to the
    photonic core, while LayerNorm and Activations remain digital.
    
    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        dropout (float): Dropout probability. Default: 0.1.
        activation (str): Activation function ("gelu" or "relu"). Default: "gelu".
        hardware_profile (str): Photonic hardware profile.
        bias (bool): Whether to use bias in linear layers. Default: True.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        hardware_profile: HardwareProfile = "lumina_nano_v1",
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        # 1. Attention Sub-layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = OpticalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            hardware_profile=hardware_profile
        )
        
        # 2. MLP Sub-layer
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        # Optical MLP
        # Note: Activation functions are currently digital as optical non-linearities are complex
        act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()
        
        self.mlp = nn.Sequential(
            OpticalLinear(embed_dim, mlp_hidden_dim, bias=bias, hardware_profile=hardware_profile),
            act_layer,
            nn.Dropout(dropout),
            OpticalLinear(mlp_hidden_dim, embed_dim, bias=bias, hardware_profile=hardware_profile),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            is_causal: Whether to apply causal masking (for auto-regressive generation)
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # 1. Attention Block (Pre-Norm)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False
        )
        x = residual + attn_out
        
        # 2. MLP Block (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
