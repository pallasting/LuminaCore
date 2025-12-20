import torch
import pytest
from lumina.layers import OpticalAttention

def test_optical_attention_shape():
    batch_size = 2
    seq_len = 10
    embed_dim = 32
    num_heads = 4
    
    model = OpticalAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Self-attention
    output, weights = model(x, x, x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    # weights is averaged over heads by default, so [batch, tgt_len, src_len]
    assert weights.shape == (batch_size, seq_len, seq_len)

def test_optical_attention_backward():
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 2
    
    model = OpticalAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    output, _ = model(x, x, x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

def test_causal_mask():
    batch_size = 1
    seq_len = 4
    embed_dim = 8
    num_heads = 2
    
    model = OpticalAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # With causal mask
    # Note: average_attn_weights=False returns [batch, num_heads, tgt_len, src_len]
    # But our implementation returns [batch, tgt_len, src_len] if average=True
    # Let's check the implementation logic again.
    # If average_attn_weights=False, it returns the raw weights [batch, num_heads, tgt_len, src_len]
    # Wait, let me check the code I wrote.
    
    output, weights = model(x, x, x, is_causal=True, average_attn_weights=False)
    
    # weights shape: [batch, num_heads, tgt_len, src_len]
    # We check the first head
    attn_matrix = weights[0, 0]
    
    # Upper triangle (excluding diagonal) should be 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_matrix[i, j] == 0.0

if __name__ == "__main__":
    test_optical_attention_shape()
    test_optical_attention_backward()
    test_causal_mask()
    print("All tests passed!")
