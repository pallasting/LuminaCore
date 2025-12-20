import torch
import pytest
from lumina.models import OpticalGPT

def test_gpt_forward():
    vocab_size = 100
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    max_seq_len = 20
    
    model = OpticalGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )
    
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is None

def test_gpt_training_step():
    vocab_size = 100
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    
    model = OpticalGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx, targets=targets)
    
    assert loss is not None
    loss.backward()
    
    # Check gradients
    assert model.head.weight.grad is not None

def test_gpt_generate():
    vocab_size = 100
    embed_dim = 32
    num_layers = 2
    num_heads = 4
    
    model = OpticalGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    idx = torch.zeros((1, 1), dtype=torch.long) # Start token
    generated = model.generate(idx, max_new_tokens=5)
    
    assert generated.shape == (1, 6)

if __name__ == "__main__":
    test_gpt_forward()
    test_gpt_training_step()
    test_gpt_generate()
    print("All tests passed!")
