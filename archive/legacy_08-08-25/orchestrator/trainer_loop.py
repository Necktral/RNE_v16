# aeon/orchestrator/trainer_loop.py
# ----------------------------------------------------------------------
import torch
from typing import Tuple


def train_step(trainer, x: torch.Tensor, y: torch.Tensor, scheduler) -> Tuple[float, dict]:
    loss, metrics = trainer.train_step(x, y)  # sin grad_fn externo
    if scheduler:
        scheduler.step()
    return loss, metrics


@torch.no_grad()
def val_step(trainer, loader, device) -> float:
    trainer.model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        loss, _ = trainer.val_step(x.to(device), y.to(device))
        total += loss
        n += 1
    trainer.model.train()
    return total / max(n, 1)
