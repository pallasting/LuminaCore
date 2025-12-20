import torch
import matplotlib.pyplot as plt
from lumina.models import OpticalGPT
from lumina.viz import plot_robustness_curve
from examples.train_gpt_shakespeare import CharDataset, get_data, BLOCK_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS

def benchmark_gpt_robustness(model, dataloader, noise_levels, device='cpu'):
    accuracies = []
    model.eval()
    
    print(f"Benchmarking robustness: {noise_levels}")
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                
                # Forward
                logits, _ = model(x)
                # logits: [batch, seq_len, vocab_size]
                
                # Inject noise into logits (simulating readout noise or interference)
                if noise_level > 0:
                    # Signal-dependent noise (Shot noise-like)
                    noise = torch.randn_like(logits) * noise_level * torch.abs(logits).mean()
                    logits = logits + noise
                
                # Calculate accuracy
                preds = logits.argmax(dim=-1) # [batch, seq_len]
                correct += (preds == y).sum().item()
                total += y.numel()
                
                if total > 10000: break # Limit size for speed
        
        acc = 100.0 * correct / total
        accuracies.append(acc)
        print(f"Noise {noise_level:.0%}: Accuracy {acc:.2f}%")
        
    return accuracies

def main():
    device = 'cpu'
    try:
        text = get_data()
    except:
        text = "Dummy data " * 1000
        
    dataset = CharDataset(text, BLOCK_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Initializing model...")
    model = OpticalGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=BLOCK_SIZE,
        hardware_profile='lumina_nano_v1'
    ).to(device)
    
    # Note: We are using an untrained model here for demonstration of the *curve shape*.
    # An untrained model will have low accuracy (~1/vocab_size), but we can still see
    # how noise affects it (it should drop to random chance).
    # Ideally, run train_gpt_shakespeare.py first and save weights.
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    accuracies = benchmark_gpt_robustness(model, dataloader, noise_levels, device)
    
    plot_robustness_curve(
        noise_levels, 
        accuracies, 
        save_path="gpt_robustness.png",
        title="OpticalGPT Robustness Analysis"
    )

if __name__ == "__main__":
    main()
