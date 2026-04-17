import torch
import torch.nn as nn
import math
import numpy as np
import logging
import os
from src.utils.influx_logger import InfluxLogger
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

WARMUP_STEPS = 5000  # Número de pasos antes de permitir la poda

class KatanaPruner:
    def __init__(self, model: nn.Module, base_tau=0.01, patio_cfg=None,
                 mask_temp=1.0, granularity="weight", cycle_length=1000):
        self.model = model
        self.base_tau = base_tau
        self.mask_temp = mask_temp
        self.granularity = granularity
        self.cycle_length = cycle_length
        self.step_count = 0
        self.rollback_threshold = 0.02
        patio_cfg = patio_cfg or {}
        self.beta_T = patio_cfg.get("beta_T", 2.0)
        self.beta_F = patio_cfg.get("beta_F", 0.5)
        self.min_tau = patio_cfg.get("min_tau", 1e-3)
        self.max_tau = patio_cfg.get("max_tau", 0.1)
        self.amp = patio_cfg.get("amp", 0.01)
        self.log_alpha = nn.ParameterDict()
        self.current_masks = {}  # Almacena las máscaras activas
        self._init_concrete_params()
        self.influx_logger = InfluxLogger(
            url=os.environ.get("INFLUXDB_URL", "http://localhost:8181"),
            token=os.environ.get("INFLUXDB_TOKEN", "<TOKEN>"),
            database=os.environ.get("INFLUXDB_BUCKET", "aeon_metrics")
        )

    def _init_concrete_params(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                sanitized_name = name.replace('.', '_')
                shape = module.weight.shape
                # Inicializar log_alpha basado en la magnitud media de los pesos
                init_val = np.log(module.weight.abs().mean().item() + 1e-8)
                param = nn.Parameter(torch.full(shape, init_val, device=module.weight.device))
                self.log_alpha[sanitized_name] = param
                # Inicializar máscaras a 1 (sin poda)
                self.current_masks[name] = torch.ones(shape, device=module.weight.device)

    def _schedule_tau(self, ctx):
        # Telemetría: log de decisión de tau
        try:
            self.influx_logger.log_event(
                name="katana_tau_schedule",
                tags={
                    "thermal_risk": str(ctx.get("thermal_risk", 0.0)),
                    "fisher_density": str(ctx.get("fisher_density", 1.0)),
                    "step_count": str(self.step_count)
                }
            )
        except Exception as e:
            logger.warning(f"No se pudo enviar evento a InfluxDB: {e}")
        thermal_risk = ctx.get("thermal_risk", 0.0)
        fisher_density = max(ctx.get("fisher_density", 1.0), 1e-9)
        cyc = 0.5 * (1 + math.cos(2 * math.pi * self.step_count / self.cycle_length))
        tau = (
            self.base_tau +
            self.beta_T * thermal_risk +
            self.beta_F * (1 / fisher_density) +
            cyc * self.amp
        )
        return max(self.min_tau, min(tau, self.max_tau))

    def _horseshoe_kl(self, module):
        w = module.weight
        if not hasattr(module, 'lambda_global'):
            module.lambda_global = nn.Parameter(torch.tensor(1.0, device=w.device))
        w_sq = w.pow(2)
        kl_local = torch.log1p(w_sq / (module.lambda_global**2 + 1e-9))
        kl_global = torch.log1p(1 / (module.lambda_global**2 + 1e-9))
        return kl_local + kl_global

    def _hybrid_score(self, module, name, ctx):
        # Si no hay Fisher ni gradientes, usamos |w| normalizada
        fisher = ctx.get("fisher", {}).get(name, None)
        grad = module.weight.grad if module.weight.grad is not None else None
        if fisher is None and grad is None:
            score = module.weight.abs()
            score = score.float()  # Asegura tipo correcto para quantile
            q10, q90 = score.quantile(0.1), score.quantile(0.9)
            return (score - q10) / (q90 - q10 + 1e-9)
        if fisher is None:
            fisher = torch.ones_like(module.weight)  # type: ignore
        if grad is None:
            grad = torch.zeros_like(module.weight)  # type: ignore
        kl = self._horseshoe_kl(module)
        score = module.weight.pow(2) * fisher + grad.abs() + kl
        score = score.float()  # Asegura tipo correcto para quantile
        q10, q90 = score.quantile(0.1), score.quantile(0.9)
        return (score - q10) / (q90 - q10 + 1e-9)

    def _concrete_mask(self, log_alpha, tau, score):
        u = torch.rand_like(log_alpha)
        g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        mask = torch.sigmoid((log_alpha + g) / self.mask_temp)
        keep = torch.sigmoid(10 * (score - tau))
        return mask * keep

    def step(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calcula nuevas máscaras de poda. Si cambian, devuelve un 'Adaptation Payload'.
        Ya no modifica los pesos del modelo directamente.
        """
        if self.step_count < WARMUP_STEPS:
            logger.info(f"[Katana] Warmup activo ({self.step_count}/{WARMUP_STEPS}). No se realiza poda.")
            self.step_count += 1
            return None

        if self._needs_rollback(ctx):
            logger.warning("[Katana] Condición de rollback detectada. Revirtiendo toda la poda.")
            return self._rollback()

        tau = self._schedule_tau(ctx)
        new_masks = {}
        masks_changed = False

        for name, module_proxy in self._get_all_linear_layers():
            sanitized_name = name.replace('.', '_')
            if sanitized_name not in self.log_alpha:
                continue
            
            module = module_proxy.layer
            score = self._hybrid_score(module, name, ctx)
            
            # Si el score es plano (sin fisher ni grad), no se poda
            if torch.all(score.abs() < 1e-6):
                mask = torch.ones_like(module.weight)  # type: ignore
            else:
                mask = self._concrete_mask(self.log_alpha[sanitized_name], tau, score)
            
            new_masks[name] = mask.detach()

            # Comprobar si la máscara ha cambiado significativamente
            if name not in self.current_masks or not torch.allclose(self.current_masks[name], new_masks[name], atol=1e-3):
                masks_changed = True

        self.step_count += 1

        if masks_changed:
            self.current_masks = new_masks
            pruning_ratios = {name: 1.0 - mask.mean().item() for name, mask in new_masks.items()}
            logger.info(f"[Katana] Nuevas máscaras de poda calculadas. Ratios: {pruning_ratios}")
            
            payload = {
                "action": "PRUNING_MASKS_UPDATED",
                "impact": {"pruning_ratios": pruning_ratios},
                "artifacts": {
                    "pruning_masks": self.current_masks
                }
            }
            return payload

        return None # No hubo cambios en las máscaras

    def _needs_rollback(self, ctx):
        delta_epist = ctx.get("delta_epist", 0)
        acc_trend = ctx.get("acc_trend", 0)
        risk_factor = 1.0 + ctx.get("thermal_risk", 0)
        threshold = self.rollback_threshold * risk_factor
        return delta_epist < -threshold and acc_trend < -threshold

    def _rollback(self) -> Dict[str, Any]:
        """Genera un payload para revertir la poda, creando máscaras de unos."""
        rollback_masks = {}
        for name, module_proxy in self._get_all_linear_layers():
            rollback_masks[name] = torch.ones_like(module_proxy.layer.weight)  # type: ignore
        
        self.current_masks = rollback_masks
        payload = {
            "action": "PRUNING_MASKS_UPDATED",
            "impact": {"pruning_ratios": {name: 0.0 for name in rollback_masks}},
            "artifacts": {
                "pruning_masks": self.current_masks
            }
        }
        return payload

    def _get_all_linear_layers(self):
        """Generador que encuentra todas las capas lineales anidadas."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                yield name, self._get_layer_proxy(name)

    def _get_layer_proxy(self, layer_name: str) -> 'LayerProxy':
        """Obtiene un proxy para acceder a una capa, incluso si es anidada."""
        parts = layer_name.split('.')
        obj = self.model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return LayerProxy(obj, parts[-1])

    def compute_kl_divergence(self):
        # Para el test_horseshoe_kl_stability
        total_kl = 0.0
        for name, module in self.model.named_modules():
            if name in self.log_alpha:
                kl = self._horseshoe_kl(module)
                total_kl += kl.sum().item()
        return float(total_kl)

# La clase LayerProxy debe ser definida para que _get_layer_proxy funcione
class LayerProxy:
    """Un objeto intermediario para obtener y establecer atributos de capa de forma segura."""
    def __init__(self, parent_module: nn.Module, layer_attr: str):
        self.parent_module = parent_module
        self.layer_attr = layer_attr

    @property
    def layer(self) -> nn.Module:
        return getattr(self.parent_module, self.layer_attr)

    def set(self, new_layer: nn.Module):
        setattr(self.parent_module, self.layer_attr, new_layer)
