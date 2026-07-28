# Generic imports
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_default_dtype(torch.float32)

# Custom Dataset Loader
class ChipDataset(torch.utils.data.Dataset):
    def __init__(self, inp, out, mu, std):
        self.inp = torch.from_numpy(inp).float()
        self.out = torch.from_numpy(out).float()

        self.mu = torch.from_numpy(mu).float()
        self.std = torch.from_numpy(std).float()

    def __len__(self):
        return self.inp.size(0)

    def __getitem__(self, idx):
        x = (self.inp[idx] - self.mu) / self.std
        y = self.out[idx]
        return x, y
