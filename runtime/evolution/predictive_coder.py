# src/evolution/predictive_coder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
import math
import gc

class HierarchicalPredictiveCoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        pcfg = config.get("predictive_coder", {})
        self.cfg = SimpleNamespace(
            num_layers=pcfg.get("num_layers", 5),
            latent_dims=pcfg.get("latent_dims", [128, 64, 32, 16, 8]),
            input_dim=pcfg.get("input_dim", 256),
            inference_steps=pcfg.get("inference_steps", 5),
            learning_rate=pcfg.get("learning_rate", 0.001),
            noise_level=pcfg.get("noise_level", 0.02),
            use_residual=pcfg.get("use_residual", True),
            activation=pcfg.get("activation", "gelu"),
            attention_heads=pcfg.get("attention_heads", 2),
            dropout_rate=pcfg.get("dropout_rate", 0.1),
            max_grad_norm=pcfg.get("max_grad_norm", 1.0)
        )

        assert len(self.cfg.latent_dims) == self.cfg.num_layers, \
            "latent_dims debe tener num_layers elementos"

        self.predictive_layers = nn.ModuleList()
        self.error_layers = nn.ModuleList()
        self.attention_mechanisms = nn.ModuleList()
        self.attn_proj_layers = nn.ModuleList()
        self.input_norm = nn.LayerNorm(self.cfg.input_dim)

        for i in range(self.cfg.num_layers):
            in_dim = self.cfg.input_dim if i == 0 else self.cfg.latent_dims[i - 1]
            out_dim = self.cfg.latent_dims[i]

            self.predictive_layers.append(nn.Sequential(
                nn.Linear(out_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.Dropout(self.cfg.dropout_rate)
            ))

            self.error_layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                self._get_activation()
            ))

            if i > 0:
                self.attention_mechanisms.append(
                    nn.MultiheadAttention(out_dim, self.cfg.attention_heads, batch_first=True)
                )
                self.attn_proj_layers.append(
                    nn.Linear(self.cfg.latent_dims[i - 1], out_dim)
                )
            else:
                self.attention_mechanisms.append(nn.Identity())
                self.attn_proj_layers.append(nn.Identity())

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.reset_states()

    def _get_activation(self):
        return {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "leaky": nn.LeakyReLU(),
            "swish": nn.SiLU()
        }.get(self.cfg.activation, nn.GELU())

    def reset_states(self):
        self.latent_states = [None] * self.cfg.num_layers
        self.prediction_errors = [0.0] * self.cfg.num_layers
        self.total_loss = 0.0
        self.last_attn_scores = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, x):
        x = self._prepare_input(x)
        x = self.input_norm(x)
        error = x
        self.last_attn_scores.clear()
        states = []

        for i in range(self.cfg.num_layers):
            input_i = error if i == 0 else self.latent_states[i - 1].detach()
            encoded = self.error_layers[i](input_i)

            if self.cfg.noise_level > 0:
                encoded += torch.randn_like(encoded) * self.cfg.noise_level

            if i > 0:
                keyval = self.attn_proj_layers[i](self.latent_states[i - 1])

                # Asegurar batch shape (B, 1, D)
                query = encoded
                keyval = keyval
                if query.dim() == 1:
                    query = query.unsqueeze(0).unsqueeze(1)
                    keyval = keyval.unsqueeze(0).unsqueeze(1)
                elif query.dim() == 2:
                    query = query.unsqueeze(1)
                    keyval = keyval.unsqueeze(1)

                attn_out, attn_weights = self.attention_mechanisms[i](query, keyval, keyval)
                encoded = attn_out.squeeze(1)
                self.last_attn_scores.append(attn_weights.detach().cpu())
            else:
                self.last_attn_scores.append(None)

            if self.latent_states[i] is None:
                self.latent_states[i] = torch.zeros_like(encoded)

            if self.cfg.use_residual:
                self.latent_states[i] = self.latent_states[i] + encoded
            else:
                self.latent_states[i] = encoded

            prediction = self.predictive_layers[i](self.latent_states[i])
            loss = F.mse_loss(prediction, input_i.detach())
            self.prediction_errors[i] = loss.item()
            error = encoded
            states.append(self.latent_states[i])

        self.total_loss = sum(self.prediction_errors) / self.cfg.num_layers
        return states, self.total_loss

    def infer(self, context):
        obs = context.get("observation", None)
        if obs is None:
            return
        x = self._prepare_input(obs)
        for _ in range(self.cfg.inference_steps):
            with torch.no_grad():
                states, loss = self.forward(x)

        context["internal_representation"] = states[-1].detach().cpu().numpy()
        context["prediction_error"] = loss
        context["prediction_errors"] = self.prediction_errors.copy()
        context["attention_scores"] = [s.numpy() if s is not None else None for s in self.last_attn_scores]

    def adapt(self, x):
        x = self._prepare_input(x)
        total_loss = 0.0

        for _ in range(self.cfg.inference_steps):
            _, loss = self.forward(x)
            total_loss += loss

        entropy_penalty = sum((s ** 2).mean() for s in self.latent_states if s is not None)
        total_loss = total_loss / self.cfg.inference_steps + 0.001 * entropy_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step(total_loss)
        return total_loss.item()

    def context_update(self, context):
        self.infer(context)

    def health_check(self):
        if any(math.isnan(e) or math.isinf(e) for e in self.prediction_errors):
            return False
        for p in self.parameters():
            if not torch.isfinite(p).all():
                return False
        return self.total_loss < 10.0

    def get_memory_footprint(self):
        p = sum(t.element_size() * t.nelement() for t in self.parameters())
        b = sum(t.element_size() * t.nelement() for t in self.buffers())
        return (p + b) / (1024 ** 2)

    def _prepare_input(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)
        return x.to(self.device)

__all__ = ["HierarchicalPredictiveCoder"]
