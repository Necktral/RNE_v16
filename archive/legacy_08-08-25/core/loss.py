# aeon/core/loss.py (versión mejorada)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Tuple, Any, List
import numpy as np

from ..aeon_types import HealthStatus, SystemMode

class UnifiedSeedLoss(nn.Module):
    """
    Implementación mejorada y completamente alineada con la especificación
    matemática de AEON FENIX-Δ (v2.1), según Ecuación (5) y componentes asociados.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config.get('loss', {})

        # Pesos base (α, β, γ, δ, ε, ζ)
        self.base_weights = nn.ParameterDict({
            'alpha': nn.Parameter(torch.tensor(self.config.get('weight_alpha', 1.0))),
            'beta': nn.Parameter(torch.tensor(self.config.get('weight_beta', 0.1))),
            'gamma': nn.Parameter(torch.tensor(self.config.get('weight_gamma', 0.05))),
            'delta': nn.Parameter(torch.tensor(self.config.get('weight_delta', 0.02))),
            'epsilon': nn.Parameter(torch.tensor(self.config.get('weight_epsilon', 0.01))),
            'zeta': nn.Parameter(torch.tensor(self.config.get('weight_zeta', 0.1)))
        })

        # Coeficientes de penalización física (ω_T, ω_V, ω_W)
        self.omega = nn.ParameterDict({
            'T': nn.Parameter(torch.tensor(self.config.get('omega_T', 1.0))),
            'V': nn.Parameter(torch.tensor(self.config.get('omega_V', 1.0))),
            'W': nn.Parameter(torch.tensor(self.config.get('omega_W', 1.0)))
        })

        # Temperatura InfoNCE
        self.infonce_temp = self.config.get('infonce_temp', 0.07)

        # MLP: sensor → coeficiente dinámico (Ecuación 6)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 6),  # salida: [α_mod, β_mod, γ_mod, δ_mod, ε_mod, ζ_mod]
            nn.Sigmoid()  # garantiza salida en [0,1]
        )

        # Pérdida para logits (NLL en Friston)
        self.nll_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.kl_div_fn = lambda q, p: torch.distributions.kl.kl_divergence(q, p).mean()

    def forward(
        self,
        model_output: Dict[str, Any],
        targets: torch.Tensor,
        health: HealthStatus
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(self.parameters()).device

        # --- 1. Componentes de la pérdida (Ecuación 5) ---
        # 1a. Friston Free Energy: KL[q||p] - E[log p(o|z)]
        posterior = model_output['posterior']  # q(z|o≤t)
        prior = model_output['prior']           # p(z|z_{t-1}, π)
        kl_loss = self.kl_div_fn(posterior, prior)

        # Reconstrucción: p(o_t | z_t) → NLL
        logits = model_output['final_logits']  # (B, L, V)
        nll_loss = self.nll_loss_fn(
            rearrange(logits, 'b l v -> (b l) v'),
            rearrange(targets, 'b l -> (b l)')
        )
        friston_loss = kl_loss + nll_loss

        # 1b. InfoNCE: contraste entre estados latentes consecutivos
        latent_features = model_output.get('latent_features')  # (B, L, D)
        infonce_loss = self._infonce_loss(latent_features) if latent_features is not None else torch.tensor(0.0, device=device)

        # 1c. Regularización de ratio de compuertas (ℛ_reg)
        boundary_probs = model_output.get('boundary_probs', None)  # p_t^{(s)} ∈ [0,1]
        reg_loss = self._reg_loss(boundary_probs) if boundary_probs is not None else torch.tensor(0.0, device=device)

        # 1d. Balance de distribución de compuertas
        balance_loss = self._balance_loss(boundary_probs) if boundary_probs is not None else torch.tensor(0.0, device=device)

        # 1e. Información mutua entre estados ocultos: I(m_{t-1}, m_t)
        hidden_states = model_output.get('hidden_states', [])
        mi_loss = self._mutual_info_loss(hidden_states) if len(hidden_states) >= 2 else torch.tensor(0.0, device=device)

        # 1f. Penalización física P_phys
        phys_penalty = self._physical_penalty(health)

        # --- 2. Modulación homeostática de pesos (Ecuación 6) ---
        sensor_vec = self._get_sensor_vector(health)
        weights = self._modulate_weights(sensor_vec)

        # --- 3. Pérdida total unificada (Ecuación 5) ---
        total_loss = (
            weights['alpha'] * friston_loss +
            weights['beta'] * infonce_loss +
            weights['gamma'] * reg_loss +
            weights['delta'] * balance_loss +
            weights['epsilon'] * mi_loss +
            weights['zeta'] * phys_penalty
        )

        # --- 4. Métricas ---
        metrics = {
            'total_loss': total_loss.item(),
            'L_friston': friston_loss.item(),
            'L_infonce': infonce_loss.item(),
            'L_reg': reg_loss.item(),
            'L_balance': balance_loss.item(),
            'L_mutual_info': mi_loss.item(),
            'P_phys': phys_penalty.item(),
            'w_alpha': weights['alpha'].item(),
            'w_beta': weights['beta'].item(),
            'w_gamma': weights['gamma'].item(),
            'w_delta': weights['delta'].item(),
            'w_epsilon': weights['epsilon'].item(),
            'w_zeta': weights['zeta'].item(),
            'sensor_vram': sensor_vec[0].item(),
            'sensor_power': sensor_vec[1].item(),
            'sensor_stress': sensor_vec[2].item(),
            'sensor_temp': sensor_vec[3].item(),
            'system_mode': health.system_mode.value
        }

        return total_loss, metrics

    def _infonce_loss(self, features: torch.Tensor) -> torch.Tensor:
        """InfoNCE: contraste entre estados latentes consecutivos."""
        B, L, D = features.shape
        if L < 2:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, p=2, dim=-1)
        queries = features[:, :-1].reshape(-1, D)    # z_t
        keys = features[:, 1:].reshape(-1, D)       # z_{t+1}

        logits = torch.mm(queries, keys.t()) / self.infonce_temp  # (N, N)
        labels = torch.arange(logits.size(0), device=logits.device)

        # InfoNCE: -log diag + log sum exp
        return F.cross_entropy(logits, labels)

    def _reg_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Regularización para mantener ratio objetivo de compuertas activas."""
        target_ratio = self.config.get('target_ratio', 6.0)
        seq_len = probs.size(1)
        num_boundaries = probs.sum(dim=1)  # (B,)
        actual_ratio = seq_len / (num_boundaries + 1e-8)  # R = L / N_c

        # Penalización cuadrática sobre desviación del ratio objetivo
        reg_error = (actual_ratio - target_ratio).pow(2).mean()
        return reg_error

    def _balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Fomenta distribución uniforme de eventos de chunking en el tiempo."""
        p_t = probs.mean(dim=0)  # Probabilidad promedio por paso
        p_uniform = torch.full_like(p_t, 1.0 / len(p_t))
        return F.kl_div(
            F.log_softmax(p_t, dim=-1),
            F.softmax(p_uniform, dim=-1),
            reduction='sum'
        )

    def _mutual_info_loss(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Estimación de I(m_{t-1}, m_t) usando inner product normalizado."""
        prev = hidden_states[-2].flatten()  # m_{t-1}
        curr = hidden_states[-1].flatten()  # m_t

        prev = (prev - prev.mean()) / (prev.std() + 1e-8)
        curr = (curr - curr.mean()) / (curr.std() + 1e-8)

        # Correlación como proxy de MI
        corr = torch.dot(prev, curr) / len(prev)
        return 1.0 - torch.abs(corr)  # Minimizar → maximizar dependencia

    def _physical_penalty(self, health: HealthStatus) -> torch.Tensor:
        """Penalización física según (5a)"""
        T_norm = health.gpu_temp / health.gpu_temp_max
        VRAM_norm = health.vram_used / health.vram_max
        W_norm = health.power_dissipated / health.power_max

        penalty = (
            self.omega['T'] * T_norm +
            self.omega['V'] * VRAM_norm +
            self.omega['W'] * W_norm
        )
        return torch.clamp(penalty, 0.0, 3.0)  # Acotado como en especificación

    def _get_sensor_vector(self, health: HealthStatus) -> torch.Tensor:
        """Vector de sensores normalizados (Ecuación 6)"""
        vec = torch.tensor([
            health.vram_used / health.vram_max,
            health.power_dissipated / health.power_max,
            health.compute_stress / health.compute_stress_max,
            health.gpu_temp / health.gpu_temp_critical  # T_crit, no T_max
        ], dtype=torch.float32, device=next(self.parameters()).device)
        return torch.clamp(vec, 0.0, 1.0)

    def _modulate_weights(self, sensor_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """MLP que mapea sensores a factores de modulación (Ecuación 6)"""
        mod_factors = self.sensor_mlp(sensor_vec.unsqueeze(0)).squeeze(0)
        return {
            k: self.base_weights[k] * mod_factors[i]
            for i, k in enumerate(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'])
        }

    def apply_safety_hooks(self, health: HealthStatus, learning_rate: float) -> Tuple[float, Dict[str, Any]]:
        """
        Aplica hooks de seguridad (Ecuación 7)
        Devuelve: (nuevo_lr, acciones_adicionales)
        """
        actions = {}
        lr = learning_rate

        # Hook 1: Reducir tasa de aprendizaje si temperatura crítica
        if health.gpu_temp > health.gpu_temp_critical:
            lr *= 0.5
            actions['throttle_lr'] = True

        # Hook 2: Resetear compuertas binarias si baja confianza
        min_conf = self.config.get('min_confidence', 0.2)
        if health.confidence < min_conf:
            actions['reset_boundaries'] = True  # Indicar al modelo que reinicie `b_t^{(s)}`

        return lr, actions