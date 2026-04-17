# src/core/probabilistic_models.py

import torch
import torch.nn as nn
from typing import Tuple

def _debug_dtype_tree(x, tag="input"):
    print(f"[DTYPE-DBG] {tag}: {x.dtype}  shape={tuple(x.shape)}")
    return x

class GenerativeModel(nn.Module):
    """Modelo generativo p_θ(z_t | z_{t-1}, a_{t-1})"""
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        # Importar LargeDenseModel del Orchestrator
        from src.core.module_orchestrator import LargeDenseModel
        in_dim = latent_dim * 2
        out_dim = latent_dim
        if in_dim == 4096 and out_dim == 4096:
            self.transition = LargeDenseModel(4096, 4096)
        else:
            self.transition = nn.Sequential(
                nn.Linear(in_dim, 4096),
                LargeDenseModel(4096, 4096),
                nn.Linear(4096, out_dim)
            )
        # self.half()  # Desactivado para FP32 estable

    def forward(self, z_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        z_prev = _debug_dtype_tree(z_prev, "z_prev")
        a_prev = _debug_dtype_tree(a_prev, "a_prev")
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = _debug_dtype_tree(x, "concat")
        out = self.transition(x)
        out = _debug_dtype_tree(out, "out")
        return out

class ApproximatePosterior(nn.Module):
    """Posterior aproximado q_ϕ(z_t | z_{t-1}, o_t)"""
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        from src.core.module_orchestrator import LargeDenseModel
        in_dim = latent_dim * 2
        out_dim = latent_dim * 2
        if in_dim == 4096 and out_dim == 4096:
            self.inference = LargeDenseModel(4096, 4096)
        else:
            self.inference = nn.Sequential(
                nn.Linear(in_dim, 4096),
                LargeDenseModel(4096, 4096),
                nn.Linear(4096, out_dim)
            )
        # self.half()  # Desactivado para FP32 estable

    def forward(self, z_prev: torch.Tensor, o_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.inference(torch.cat([z_prev, o_t], dim=-1))
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def sample(self, z_prev: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(z_prev, o_t)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
