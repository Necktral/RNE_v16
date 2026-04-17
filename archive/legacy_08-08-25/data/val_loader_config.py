import torch
from torch.utils.data import DataLoader, random_split
from .data_normalizer import DataNormalizer

def create_train_val_loaders(data, batch_size=32, val_split=0.2, shuffle=True, seed=42, normalizer=None):
    if normalizer is not None:
        data = normalizer.transform(data)
    num_samples = data.shape[0]
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    dataset = torch.utils.data.TensorDataset(data)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
