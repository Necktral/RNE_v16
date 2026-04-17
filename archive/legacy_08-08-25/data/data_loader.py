import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def create_train_val_loaders(x, y, batch_size=32, val_split=0.2, shuffle=True, seed=42, normalizer=None):
    """
    Crea los DataLoaders de entrenamiento y validación a partir de tensores X e Y.
    Aplica normalización si se proporciona un normalizador.
    """
    if normalizer is not None:
        x = normalizer.transform(x)
    num_samples = x.shape[0]
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    dataset = TensorDataset(x, y)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
