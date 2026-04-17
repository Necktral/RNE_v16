import torch
import logging

async def eval_loop(model, val_loader, device=None, logger=None):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    device = device or (next(model.parameters()).device)
    logger = logger or logging.getLogger(__name__)
    with torch.no_grad():
        for batch in val_loader:
            o_t = batch[0].to(device)
            z_prev = torch.zeros((o_t.shape[0], 32), device=device)
            a_prev = torch.zeros((o_t.shape[0], 32), device=device)
            output = model.generative_model(z_prev, a_prev)
            target = o_t[:, :output.shape[1]]
            loss = torch.nn.functional.mse_loss(output, target)
            total_loss += loss.item()
            total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    logger.info(f"[EVAL] Val loss: {avg_loss}")
    return avg_loss
