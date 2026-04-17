import torch
import torch.nn as nn
from torch.distributions import Normal

class RSSMLite2(nn.Module):
    """
    Módulo de transición latente inspirado en RSSM (renombrado temporalmente).
    """
    def __init__(self, latent_dim=8, action_dim=0, hidden_dim=64):
        super().__init__()
        input_dim = latent_dim + action_dim
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, prev_z, prev_a, prev_h):
        x = torch.cat([prev_z, prev_a], dim=-1)
        h = self.rnn(x, prev_h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z_dist = Normal(mu, std)
        return z_dist, h
