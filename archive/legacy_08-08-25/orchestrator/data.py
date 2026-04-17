# aeon/orchestrator/data.py
# ----------------------------------------------------------------------
import os, numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch


def build_data_pipeline(cfg):
    """
    Devuelve train_loader, val_loader y vocab_size sin
    reventar la RAM (>30 GB) gracias a mmap.
    """
    if not os.path.exists(cfg.DATA_CACHE_PATH):
        from scripts.prepare_data import prepare_dataset_and_stats

        prepare_dataset_and_stats()

    data = np.load(cfg.DATA_CACHE_PATH, mmap_mode="r")  # no copia a RAM
    vocab_size = int(data.max()) + 1

    split = int(len(data) * 0.98)
    train_data, val_data = data[:split], data[split:]

    def make_loader(arr, shuffle):
        tensor = torch.from_numpy(arr.astype(np.int64))
        ds = TensorDataset(tensor[:-1], tensor[1:])
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=cfg.DATALOADER_WORKERS,
        )

    return make_loader(train_data, True), make_loader(val_data, False), vocab_size
