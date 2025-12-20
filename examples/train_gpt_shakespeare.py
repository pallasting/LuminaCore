import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import requests
from lumina.models import OpticalGPT
from lumina.optim import NoiseAwareTrainer

# 1. Data Loading (Tiny Shakespeare)
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.data = data
        self.block_size = block_size
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[c] for c in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def get_data():
    file_path = 'input.txt'
    if not os.path.exists(file_path):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading {url}...")
        data = requests.get(url).text
        with open(file_path, 'w') as f:
            f.write(data)
    else:
        with open(file_path, 'r') as f:
            data = f.read()
    return data

# 2. Training Configuration
BLOCK_SIZE = 64 # Context length
BATCH_SIZE = 32
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
LEARNING_RATE = 3e-4
EPOCHS = 1 # Just for demo
DEVICE = 'cpu' # Using CPU for simulation

def train():
    # Prepare Data
    try:
        text = get_data()
    except Exception as e:
        print(f"Failed to download data: {e}")
        print("Using dummy data instead.")
        text = "Hello world! This is a test dataset for OpticalGPT. " * 100
        
    dataset = CharDataset(text, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset size: {len(text)} characters")
    print(f"Vocab size: {dataset.vocab_size}")
    
    # Initialize Model
    model = OpticalGPT(
        vocab_size=dataset.vocab_size,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=BLOCK_SIZE,
        hardware_profile='lumina_nano_v1' # Simulate noisy hardware
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Noise Aware Trainer
    # This will inject noise during training to make the model robust
    trainer = NoiseAwareTrainer(
        model=model,
        optimizer=optimizer,
        robustness_target=0.95,
        noise_schedule='linear'
    )
    
    # Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # NAT Training Step
            loss, robustness = trainer.train_step(x, targets=y)
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss:.4f}, Robustness: {robustness:.4f}")
            
            if i >= 50: # Stop early for demo
                break
                
    # Generation
    print("\nGenerating text...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE) # Start with 0
    generated_indices = model.generate(context, max_new_tokens=200)
    generated_text = ''.join([dataset.itos[i.item()] for i in generated_indices[0]])
    print("--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    train()
