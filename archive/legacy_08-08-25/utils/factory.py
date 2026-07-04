import torch

def make_scheduler(optimizer, cfg):
    name = cfg.name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.lr_min)
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr_max, total_steps=cfg.total_steps)
    elif name == "stepdecay":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        raise ValueError(f"Scheduler '{cfg.name}' no soportado")
