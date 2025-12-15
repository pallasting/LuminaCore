import os
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from lumina.viz import benchmark_robustness


class RandomLogitsDataset(Dataset):
    def __init__(self, length: int = 20, dim: int = 8, num_classes: int = 3):
        self.length = length
        self.dim = dim
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.rand(self.dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def test_benchmark_robustness_generates_report():
    torch.manual_seed(0)
    ds = RandomLogitsDataset()
    loader = DataLoader(ds, batch_size=5, shuffle=False)

    # 简单线性模型即可，输出 logits
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 3),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "robustness_tmp.png")
        noise_levels, accuracies = benchmark_robustness(
            model=model,
            test_loader=loader,
            noise_levels=[0.0, 0.1],
            device=torch.device("cpu"),
            save_path=save_path,
            title="Test Robustness",
        )

        assert len(noise_levels) == 2
        assert len(accuracies) == 2
        assert os.path.exists(save_path)
