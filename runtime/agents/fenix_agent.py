# [LEGACY/ROTO — AEON FENIX-Δ en cuarentena] Agente no funcional: importa
# `.rssm_lite` (inexistente → ModuleNotFoundError). No usar. Ver
# docs/analysis/LEGACY_QUARANTINE.md.
import torch
import torch.nn as nn
from torch.distributions import Normal
from .rssm_lite import RSSMLite  # Cambiado a importación relativa

class FenixAgent(nn.Module):
    """
    FenixAgent v2 — con transición latente evolutiva (RSSM-Lite).
    Este agente ya no es estático: predice, memoriza y muta secuencias latentes.
    """
    def __init__(self, input_dim=32, hidden_dim=64, latent_dim=8):
        super().__init__()
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        # RSSM-Lite para transición latente recurrente
        self.rssm = RSSMLite(latent_dim=latent_dim, hidden_dim=hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_seq):
        """
        Args:
            x_seq: Tensor [B, T, input_dim]
        Returns:
            x_pred: reconstrucción [B, T, input_dim]
            posterior: lista de distribuciones z_t (mu, std)
            prior: lista de distribuciones RSSM z_t|z_{t-1}
            cov: [B, T, latent_dim, latent_dim]
        """
        B, T, _ = x_seq.shape
        z_list, post_list, prior_list, x_rec_list = [], [], [], []

        h_rnn = torch.zeros(B, self.rssm.rnn.hidden_size, device=x_seq.device)
        prev_z = torch.zeros(B, self.rssm.to_mu.out_features, device=x_seq.device)
        prev_a = torch.zeros(B, 0, device=x_seq.device)  # no action yet

        for t in range(T):
            # === Encode observation ===
            _, h_enc = self.encoder_rnn(x_seq[:, t:t+1])
            h_enc = h_enc.squeeze(0)
            mu = self.to_mu(h_enc)
            logvar = self.to_logvar(h_enc)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_t = mu + eps * std

            post = Normal(mu, std)
            z_list.append(z_t)
            post_list.append(post)

            # === Predict prior using RSSM ===
            prior_t, h_rnn = self.rssm(prev_z, prev_a, h_rnn)
            prior_list.append(prior_t)
            prev_z = z_t

            # === Decode ===
            x_hat = self.decoder(z_t)
            x_rec_list.append(x_hat)

        # === Output ===
        x_pred = torch.stack(x_rec_list, dim=1)
        cov_matrix = torch.stack([torch.diag_embed(p.scale**2) for p in post_list], dim=1)

        print("🧠 FenixAgent v2 ejecutando con post_list tipo:", type(post_list))
        return x_pred, post_list, prior_list, cov_matrix
