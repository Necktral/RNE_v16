import torch
import logging

def log_sparsity(model, logger=None, step=None):
    logger = logger or logging.getLogger(__name__)
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if param is not None and param.requires_grad:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    msg = f"[SPARSITY]{' [step %d]' % step if step is not None else ''} sparsity={sparsity:.4f} ({zero_params}/{total_params} zeros)"
    logger.info(msg)
    return sparsity
