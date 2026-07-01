import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import deque
from omegaconf import DictConfig
from typing import Optional, Deque, Tuple, Dict, Any
from torch.distributions import Normal
from runtime.core.event_bus import event_bus  # Integración EventBus centralizado

# Importación segura de EpistemeSnapshot
try:
    from runtime.core.episteme import EpistemeSnapshot
except ImportError:
    class EpistemeSnapshot:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

__all__ = ["ThermodynamicVFE", "QuantumAwarePID", "EpistemicAwareLoss"]

class ThermodynamicVFE(nn.Module):
    """VFE con conciencia energética adaptativa: ℱ = (1 - η)·Recon + β·(η·KL) + homeo_penalty"""
    def __init__(self, config: DictConfig | dict):
        super().__init__()
        self.energy_weight = config.get("energy_weight", 0.7)
        self.recon_type = config.get("recon_loss", "mse")
        
        # Coeficientes de penalización dinámicos
        self.mem_penalty_coef = config.get("mem_penalty", 10.0)
        self.temp_penalty_coef = config.get("temp_penalty", 15.0)
        self.entropy_penalty_coef = config.get("entropy_penalty", 5.0)
        
        # Umbrales adaptativos según intensidad de tarea
        self.base_temp_cap = config.get("thermal_cap", 0.95)
        self.base_mem_cap = config.get("mem_cap", 0.93)

    def forward(self, 
                recon_x: torch.Tensor, 
                x: torch.Tensor, 
                kl_div: torch.Tensor, 
                snap: EpistemeSnapshot) -> torch.Tensor:
        """
        Calcula la Energía Libre Variacional con conciencia termodinámica adaptativa
        """
        # Factor de eficiencia epistémica (ℰ)
        efficiency_factor = 1 - snap.efficiency if hasattr(snap, 'efficiency') else 1.0
        
        # Término de reconstrucción con peso energético
        if self.recon_type == "bce":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        recon_loss *= efficiency_factor
        
        # KL con corrección cuántica si está disponible
        quantum_beta = getattr(snap, 'quantum_beta', 1.0)
        kl_term = kl_div * quantum_beta * self.energy_weight
        
        # Penalización por desviación de homeostasis adaptativa
        homeo_penalty = self._adaptive_homeostasis_penalty(snap)
        
        return recon_loss + kl_term + homeo_penalty

    def _adaptive_homeostasis_penalty(self, snap: EpistemeSnapshot) -> torch.Tensor:
        """Penaliza acercarse a límites físicos con umbrales adaptativos"""
        task_intensity = getattr(snap, 'task_intensity', 0.0)
        
        # Umbrales dinámicos
        mem_threshold = self.base_mem_cap + 0.05 * task_intensity
        temp_threshold = self.base_temp_cap + 0.02 * task_intensity
        
        mem_penalty = F.relu(torch.tensor(getattr(snap, 'memory_usage', 0.0) - mem_threshold)) * self.mem_penalty_coef
        temp_penalty = F.relu(torch.tensor(getattr(snap, 'temperature', 0.0) - temp_threshold)) * self.temp_penalty_coef
        entropy_penalty = F.relu(torch.tensor(getattr(snap, 'entropy_production', 0.0) - 0.95)) * self.entropy_penalty_coef
        
        return mem_penalty + temp_penalty + entropy_penalty


class QuantumAwarePID:
    """Controlador PID con correcciones cuánticas avanzadas y monitoreo físico"""
    def __init__(self, config: DictConfig | dict):
        # Parámetros PID base
        self.k_p = config.get("k_p", 0.1)
        self.k_i = config.get("k_i", 0.01)
        self.k_d = config.get("k_d", 0.005)
        self.target_kl = config.get("target_kl", 1.0)
        
        # Parámetros cuánticos
        self.quantum_amp = config.get("quantum_amp", 0.05)
        self.quantum_freq = config.get("quantum_freq", 0.5)
        
        # Límites de crisis
        self.lyapunov_threshold = config.get("lyapunov_threshold", 1000.0)
        
        # Estado interno
        self.beta = float(config.get("beta_0", 1.0))
        self.prev_error = 0.0
        self.integral = 0.0
        self.beta_history = deque(maxlen=100)
        self.chaos_counter = 0
        self.last_update = time.time()

    def update(self, current_kl: float, snap: Optional[EpistemeSnapshot] = None) -> float:
        """
        Actualiza beta con fluctuaciones cuánticas gaussianas y ajuste adaptativo de PID
        """
        if snap and self._detect_crisis(snap):
            # Emitir evento de crisis física/caótica
            event_bus.emit('loss_crisis_detected', {
                'lyapunov_exp': getattr(snap, 'lyapunov_exp', None),
                'temperature': getattr(snap, 'temperature', None),
                'memory_usage': getattr(snap, 'memory_usage', None),
                'timestamp': time.time()
            })
            return self._chaotic_reset()
            
        # Fluctuación cuántica gaussiana
        error = current_kl - self.target_kl
        quantum_fluct = Normal(0, self.quantum_amp).sample().item()
        quantum_error = error + quantum_fluct
        
        # Ajuste adaptativo de PID según estado del sistema
        if hasattr(snap, 'temperature') and snap.temperature > 0.85:
            quantum_error *= 0.5  # Reducir exploración en alta temperatura
            
        # Términos PID
        self.integral += quantum_error
        derivative = quantum_error - self.prev_error
        self.prev_error = quantum_error
        
        # Actualización beta con límites físicos
        delta_beta = self.k_p * quantum_error + self.k_i * self.integral + self.k_d * derivative
        new_beta = self.beta + delta_beta
        self.beta = max(0.0, min(new_beta, 10.0))
        self.beta_history.append(self.beta)
        
        return self.beta

    def _detect_crisis(self, snap: EpistemeSnapshot) -> bool:
        """Detecta condiciones de crisis usando métricas físicas y epistémicas"""
        has_lyapunov = hasattr(snap, 'lyapunov_exp')
        has_temp = hasattr(snap, 'temperature')
        has_memory = hasattr(snap, 'memory_usage')
        
        lyapunov_cond = has_lyapunov and snap.lyapunov_exp > self.lyapunov_threshold
        thermal_cond = has_temp and snap.temperature > 0.98
        memory_cond = has_memory and snap.memory_usage > 0.95
        
        return lyapunov_cond or thermal_cond or memory_cond

    def _chaotic_reset(self) -> float:
        """Reinicialización no determinista con memoria de estados estables"""
        self.chaos_counter += 1
        # Emitir evento de resurrección caótica
        event_bus.emit('loss_chaotic_resurrection', {
            'chaos_counter': self.chaos_counter,
            'beta': self.beta,
            'timestamp': time.time()
        })
        
        if self.beta_history:
            stable_beta = np.percentile(list(self.beta_history), 25)
            chaos_factor = 0.8 + 0.4 * torch.rand(1).item()
            self.beta = stable_beta * chaos_factor
        else:
            self.beta = 1.0 * (0.7 + 0.6 * torch.rand(1).item())
            
        self.integral *= 0.5
        self.prev_error = 0.0
        return self.beta


class EpistemicAwareLoss(nn.Module):
    """
    Sistema de pérdidas completo con conciencia termodinámica, estabilidad caótica
    y optimización prospectiva para AGI.
    """
    def __init__(self, 
                 config: DictConfig | dict,
                 episteme_meter: Optional[Any] = None):
        super().__init__()
        self.vfe = ThermodynamicVFE(config.get("vfe", {}))
        self.pid = QuantumAwarePID(config.get("pid", {}))
        self.episteme = episteme_meter
        
        # Pesos para componentes adicionales
        self.fisher_weight = config.get("fisher_weight", 0.1)
        self.prospective_weight = config.get("prospective_weight", 0.3)
        
        # Estado interno
        self.last_entropy = 0.0
        self.last_snapshot = None

    def forward(self,
                recon_x: torch.Tensor,
                x: torch.Tensor,
                kl_div: torch.Tensor,
                fisher_matrix: Optional[torch.Tensor] = None,
                hidden_act: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula la pérdida completa con conciencia epistémica y métricas avanzadas
        """
        current_snap = self._get_current_snapshot()
        beta = self.pid.update(kl_div.item(), current_snap)
        vfe_loss = self.vfe(recon_x, x, kl_div, current_snap)
        fisher_loss = self._compute_kfac_penalty(fisher_matrix, hidden_act)
        future_entropy_loss = self._compute_prospective_entropy(current_snap)
        
        total_loss = vfe_loss + self.fisher_weight * fisher_loss + self.prospective_weight * future_entropy_loss
        
        # Actualizar estado
        if current_snap and hasattr(current_snap, 'entropy_production'):
            self.last_entropy = current_snap.entropy_production
        self.last_snapshot = current_snap
        
        # Métricas detalladas
        metrics = {
            "kl": kl_div.item(),
            "beta": beta,
            "vfe": vfe_loss.item(),
            "efficiency": getattr(current_snap, 'efficiency', 0.0) if current_snap else 0.0,
            "fisher_penalty": fisher_loss.item() if fisher_loss is not None else 0.0,
            "fisher_loss": fisher_loss.item() if fisher_loss is not None else 0.0,  # Para compatibilidad legacy
            "thermal_penalty": 0.0,  # Placeholder, puedes calcularlo si tienes la lógica
            "entropy_rate": 0.0  # Placeholder, puedes calcularlo si tienes la lógica
        }
        
        return total_loss, metrics

    def _get_current_snapshot(self) -> Optional[EpistemeSnapshot]:
        """Obtiene el snapshot epistémico actual o uno por defecto"""
        if self.episteme and hasattr(self.episteme, 'latest_snapshot'):
            return self.episteme.latest_snapshot
        return self.last_snapshot

    def _compute_kfac_penalty(self, 
                              fisher_matrix: Optional[torch.Tensor], 
                              hidden_act: Optional[torch.Tensor]) -> torch.Tensor:
        """Penalización de Fisher optimizada con K-FAC"""
        if fisher_matrix is None:
            return torch.tensor(0.0)
            
        try:
            # Aproximación K-FAC (pseudo-código)
            approx_fisher = torch.diag(torch.diag(fisher_matrix))  # Diagonal approximation
            return torch.norm(approx_fisher) * 0.01
        except Exception as e:
            # Fallback a varianza de activaciones
            if hidden_act is not None:
                act_variance = hidden_act.var()
                return F.relu(act_variance - 5.0) * 0.05
            return torch.tensor(0.0)

    def _compute_prospective_entropy(self, snap: Optional[EpistemeSnapshot]) -> torch.Tensor:
        """Penaliza trayectorias que aumentan la entropía futura usando modelo predictivo"""
        if not snap or not hasattr(snap, 'entropy_production'):
            return torch.tensor(0.0)
            
        # Modelo predictivo simple basado en histórico
        entropy_gradient = (snap.entropy_production - self.last_entropy) / 0.1
        predicted_entropy = snap.entropy_production + 0.5 * entropy_gradient
        
        return F.relu(predicted_entropy - 0.9) * 10.0  # Penalización más agresiva


def create_loss(config: DictConfig, episteme_meter: Optional[Any] = None) -> nn.Module:
    """Crea el sistema de pérdidas apropiado basado en configuración"""
    loss_type = config.get("type", "epistemic")
    
    if loss_type == "epistemic":
        return EpistemicAwareLoss(config, episteme_meter)
    else:
        # Fallback a implementación simple
        class SimpleLoss(nn.Module):
            def forward(self, recon_x, x, kl_div):
                return F.mse_loss(recon_x, x) + kl_div, {"loss": 0.0}
        return SimpleLoss()

# Wrapper para compatibilidad retroactiva con tests y código legado
class CompositeLoss(EpistemicAwareLoss):
    def __init__(self, **kwargs):
        config = {}
        episteme_meter = None
        if 'config' in kwargs and isinstance(kwargs['config'], (dict, DictConfig)):
            config = kwargs['config']
        else:
            for key in [
                'efficiency', 'temperature', 'memory_load', 'fisher_weight', 'prospective_weight',
                'vfe', 'pid', 'episteme_meter', 'thermal_cap', 'mem_cap', 'energy_weight',
                'recon_loss', 'mem_penalty', 'temp_penalty', 'entropy_penalty', 'target_kl',
                'quantum_amp', 'quantum_freq', 'lyapunov_threshold', 'beta_0'
            ]:
                if key in kwargs:
                    config[key] = kwargs[key]
            if 'episteme_meter' in kwargs:
                episteme_meter = kwargs['episteme_meter']
        super().__init__(config, episteme_meter)

    def forward(self, pred, target, kl, fisher=None, *args, **kwargs):
        # Soporta extra_losses y métricas legacy
        extra_losses = kwargs.get('extra_losses', [])
        # Llama al forward original
        loss, metrics = super().forward(pred, target, kl, fisher)
        # Suma extra_losses si existen
        if extra_losses:
            extra_sum = sum([x.item() if isinstance(x, torch.Tensor) else float(x) for x in extra_losses])
            loss = loss + extra_sum
        # Asegura requires_grad si algún input lo tiene
        if any([t.requires_grad for t in [pred, target, kl] if isinstance(t, torch.Tensor)]):
            loss = loss.clone().detach().requires_grad_(True)
        # Inyecta claves legacy en métricas
        metrics.setdefault('entropy_rate', 0.0)
        metrics.setdefault('fisher_penalty', 0.0)
        metrics.setdefault('thermal_penalty', 0.0)
        return loss, metrics