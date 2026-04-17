import numpy as np
import torch
import json
import os

class DataNormalizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data: torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True).clamp_min(1e-6)

    def transform(self, data: torch.Tensor):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fit before transform.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fit before inverse_transform.")
        return data * self.std + self.mean

    def save_stats(self, path):
        stats = {
            'mean': self.mean.cpu().numpy().tolist() if self.mean is not None else None,
            'std': self.std.cpu().numpy().tolist() if self.std is not None else None
        }
        with open(path, 'w') as f:
            json.dump(stats, f)

    def load_stats(self, path):
        with open(path, 'r') as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats['mean']) if stats['mean'] is not None else None
        self.std = torch.tensor(stats['std']) if stats['std'] is not None else None
