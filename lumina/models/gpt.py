import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..layers.transformer_block import OpticalTransformerBlock
from ..layers.optical_linear import OpticalLinear

class OpticalGPT(nn.Module):
    """
    OpticalGPT: A Photonic-Accelerated Generative Pre-trained Transformer.
    
    This model implements a GPT-style architecture where the heavy matrix multiplications
    (Attention Projections and MLP layers) are offloaded to the photonic core.
    
    Architecture:
    - Token Embedding (Digital)
    - Positional Embedding (Digital/Learned)
    - N x OpticalTransformerBlock
    - LayerNorm (Digital)
    - Head (OpticalLinear)
    
    Args:
        vocab_size (int): Size of vocabulary.
        embed_dim (int): Embedding dimension.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length. Default: 1024.
        mlp_ratio (float): Ratio of MLP hidden dimension. Default: 4.0.
        dropout (float): Dropout probability. Default: 0.1.
        activation (str): Activation function. Default: "gelu".
        hardware_profile (str): Photonic hardware profile.
        bias (bool): Whether to use bias. Default: True.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        hardware_profile: str = "lumina_nano_v1",
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            OpticalTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation,
                hardware_profile=hardware_profile,
                bias=bias
            )
            for _ in range(num_layers)
        ])
        
        # Final Norm
        self.norm_f = nn.LayerNorm(embed_dim)
        
        # Language Model Head
        # We use OpticalLinear for the final projection to vocabulary
        # This is a massive matrix multiplication (embed_dim x vocab_size)
        # which benefits greatly from photonic acceleration
        self.head = OpticalLinear(
            in_features=embed_dim,
            out_features=vocab_size,
            bias=False, 
            hardware_profile=hardware_profile
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, OpticalLinear):
            # OpticalLinear has its own initialization, but we can override if needed
            pass

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: Token indices [batch, seq_len]
            targets: Target token indices [batch, seq_len] (optional)
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Scalar loss (if targets provided)
        """
        device = idx.device
        b, t = idx.size()
        
        if t > self.max_seq_len:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.max_seq_len}")
        
        # 1. Embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device) # [t]
        
        tok_emb = self.token_embedding(idx) # [b, t, embed_dim]
        pos_emb = self.position_embedding(pos) # [t, embed_dim]
        
        x = self.dropout(tok_emb + pos_emb)
        
        # 2. Transformer Blocks
        for block in self.layers:
            x = block(x, is_causal=True)
            
        # 3. Final Norm
        x = self.norm_f(x)
        
        # 4. Head
        logits = self.head(x) # [b, t, vocab_size]
        
        loss = None
        if targets is not None:
            # Flatten for CrossEntropyLoss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens.
        
        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Indices including generated tokens [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            
            # Get last step logits
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
