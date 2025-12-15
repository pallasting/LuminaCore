import torch
from torch.utils.data import DataLoader, Dataset

from lumina.nn import OpticalLinear
from lumina.optim import NoiseAwareTrainer


class RandomVectorDataset(Dataset):
    def __init__(self, length: int = 16, dim: int = 8, num_classes: int = 2):
        self.length = length
        self.dim = dim
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.rand(self.dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def test_noise_aware_trainer_single_epoch_runs():
    torch.manual_seed(0)

    train_ds = RandomVectorDataset(length=12, dim=8, num_classes=2)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    model = torch.nn.Sequential(
        OpticalLinear(8, 4, hardware_profile="lumina_nano_v1"),
        torch.nn.ReLU(),
        OpticalLinear(4, 2, hardware_profile="lumina_nano_v1"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = NoiseAwareTrainer(
        model=model,
        optimizer=optimizer,
        robustness_target=0.9,
        noise_schedule="linear",
        max_noise_level=0.05,
        device=torch.device("cpu"),
    )

    trainer.fit(train_loader, epochs=1, val_loader=None, verbose=False)

    assert len(trainer.history["train_loss"]) == 1
    assert len(trainer.history["train_acc"]) == 1
