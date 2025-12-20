import torch
from lumina.models.gpt import OpticalGPT
from lumina.compiler import export_model
import os

def demo_v03_features():
    print("=== LuminaFlow v0.3 Feature Demo ===")
    
    # 1. Initialize a v0.3 OpticalGPT model
    print("\n1. Initializing OpticalGPT (v0.3 Architecture)...")
    model = OpticalGPT(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64,
        hardware_profile="edge"
    )
    model.eval()
    
    # 2. Demonstrate Advanced Physics (via forward pass)
    # Note: In v0.3, the Rust backend now handles thermal and crosstalk noise
    print("\n2. Running Inference with Advanced Physics (Rust-accelerated)...")
    dummy_input = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        # Ensure Rust backend is simulated/enabled for this demo if possible
        # In a real environment, LUMINA_USE_RUST=1 would be set
        output = model(dummy_input)
    print(f"Inference successful. Output shape: {output.shape}")
    
    # 3. Demonstrate Lumina Compiler (Export Logic)
    print("\n3. Compiling model for Hardware Deployment...")
    export_path = "exports/gpt_edge_v03.json"
    instructions = export_model(model, export_path, profile="edge")
    
    print(f"Exported {len(instructions['layers'])} optical layers to {export_path}")
    print(f"Quantization: {instructions['metadata']['quantization_bits']}-bit")
    
    # 4. Performance Optimization Note
    print("\n4. Performance Optimization:")
    print("- Rust backend now uses loop unrolling (8x) for matrix multiplications.")
    print("- Fused kernels now include multi-physics noise injection in a single pass.")

if __name__ == "__main__":
    demo_v03_features()
