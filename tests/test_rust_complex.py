import torch
import numpy as np
import lumina_kernel

def test_complex_matmul():
    print("Testing Rust Complex Matrix Multiplication...")
    
    # Create complex tensors in PyTorch
    batch_size = 2
    in_features = 3
    out_features = 4
    
    input_torch = torch.randn(batch_size, in_features, dtype=torch.complex64)
    weight_torch = torch.randn(out_features, in_features, dtype=torch.complex64)
    
    # Expected result using PyTorch
    expected = torch.matmul(input_torch, weight_torch.t())
    
    # Result using Rust
    # Note: Rust complex_matmul expects (batch, in) and (out, in)
    # as per parallel_complex_matmul implementation which does dot products of rows
    input_np = input_torch.numpy()
    weight_np = weight_torch.numpy()
    
    output_rust_np = lumina_kernel.complex_matmul(input_np, weight_np)
    output_rust = torch.from_numpy(output_rust_np)
    
    # Check shape
    assert output_rust.shape == expected.shape
    print(f"Shape check PASSED: {output_rust.shape}")
    
    # Check values
    diff = torch.abs(output_rust - expected).max()
    print(f"Max difference: {diff.item()}")
    assert diff < 1e-5
    print("Value check PASSED!")

if __name__ == "__main__":
    try:
        test_complex_matmul()
        print("\n✅ Rust Complex Support Test PASSED!")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
