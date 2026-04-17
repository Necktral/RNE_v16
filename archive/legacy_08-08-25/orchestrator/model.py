# aeon/orchestrator/model.py
# ----------------------------------------------------------------------
import torch.nn as nn, torch.optim as optim
from ..models.hnet import HNet


def build_model_and_opt(cfg, device):
    emb = nn.Embedding(cfg.VOCAB_SIZE, cfg.D_MODEL).to(device)
    net = HNet(d_model=cfg.D_MODEL, n_layers=cfg.N_LAYERS).to(device)
    head = nn.Linear(cfg.D_MODEL, cfg.VOCAB_SIZE).to(device)

    params = list(emb.parameters()) + list(net.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=cfg.LEARNING_RATE)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.TRAINING_STEPS)

    return emb, net, head, opt, sched
