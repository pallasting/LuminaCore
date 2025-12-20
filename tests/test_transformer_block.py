import torch
import pytest
from lumina.layers import OpticalTransformerBlock

def test_transformer_block_shape():
    batch_size = 2
    seq_len = 8
    embed_dim = 32
    num_heads = 4
    
    block = OpticalTransformerBlock(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    output = block(x)
    
    assert output.shape == (batch_size, seq_len, embed_dim)

def test_transformer_block_backward():
    batch_size = 2
    seq_len = 4
    embed_dim = 16
    num_heads = 2
    
    block = OpticalTransformerBlock(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

def test_transformer_block_causal():
    # Verify that causal masking works (output at t should not depend on t+1)
    batch_size = 1
    seq_len = 4
    embed_dim = 8
    num_heads = 2
    
    block = OpticalTransformerBlock(embed_dim, num_heads)
    block.eval() # Disable dropout for deterministic check
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    x_modified = x.clone()
    x_modified[:, -1, :] += 1.0 # Modify the last token
    
    # Forward with causal mask
    out1 = block(x, is_causal=True)
    out2 = block(x_modified, is_causal=True)
    
    # The output for the first token (index 0) should be identical
    # because it cannot attend to the last token (index 3)
    # Note: LayerNorm might mix slightly if it normalizes across sequence? 
    # No, LayerNorm is usually per-token (dim=-1).
    # Let's verify LayerNorm behavior. nn.LayerNorm(embed_dim) normalizes over the last dimension.
    # So tokens are independent in LayerNorm.
    
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)
    
    # The output for the last token should be different
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])

if __name__ == "__main__":
    test_transformer_block_shape()
    test_transformer_block_backward()
    test_transformer_block_causal()
    print("All tests passed!")
