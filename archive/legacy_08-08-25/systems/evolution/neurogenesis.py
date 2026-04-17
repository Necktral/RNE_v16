import torch
import torch.nn as nn
import warnings
import os
import logging
from aeon.utils.influx_logger import InfluxLogger
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class NeurogenesisManager:
    def __init__(self, model: nn.Module,
                 layers_to_expand: List[str],
                 dependent_layers: Dict[str, str],
                 growth_unit: int = 4,
                 plasticity_factor: float = 0.05,
                 eff_threshold: float = 0.2,
                 fisher_threshold: float = 0.2,
                 delta_epist_min: float = 0.01,
                 mutual_info_min: float = 0.05,
                 soft_grow_steps: int = 10,
                 cooldown_steps: int = 500,
                 layer_thresholds: Optional[Dict[str, float]] = None):
        self.model = model
        self.layers_to_expand = layers_to_expand
        self.dependent_layers = dependent_layers
        self.growth_unit = growth_unit
        self.plasticity_factor = plasticity_factor
        self.eff_threshold = eff_threshold
        self.fisher_threshold = fisher_threshold
        self.delta_epist_min = delta_epist_min
        self.mutual_info_min = mutual_info_min
        self.soft_grow_steps = soft_grow_steps
        self.layer_thresholds = layer_thresholds or {}
        self.cooldown_steps = cooldown_steps
        self.cooldown_counter = {name: 0 for name in self.layers_to_expand}
        self.new_units = {}
        self.influx_logger = InfluxLogger(
            url=os.environ.get("INFLUXDB_URL", "http://localhost:8181"),
            token=os.environ.get("INFLUXDB_TOKEN", "<TOKEN>"),
            database=os.environ.get("INFLUXDB_BUCKET", "aeon_metrics")
        )

    def should_grow_layer(self, name: str, ctx: Dict[str, Any]) -> bool:
        if self.cooldown_counter[name] > 0:
            self.cooldown_counter[name] -= 1
            return False

        delta_epist = ctx.get("delta_epist", 0.0)
        mutual_info = ctx.get("mutual_info", 0.0)
        efficiency = ctx.get("efficiency", 1.0)
        fisher_density = ctx.get("fisher_density", 1.0)
        conscious_focus = ctx.get("conscious_focus", 1.0)

        if conscious_focus < 0.7:
            return False

        if not (0 <= efficiency <= 1.0):
            warnings.warn(f"Efficiency fuera de rango: {efficiency}")
            return False

        eff_thr = self.layer_thresholds.get(name, self.eff_threshold)

        should = (
            efficiency < eff_thr and
            fisher_density < self.fisher_threshold and
            delta_epist < -self.delta_epist_min and
            mutual_info > self.mutual_info_min
        )

        # Telemetría: log de decisión de crecimiento
        try:
            self.influx_logger.log_event(
                name="neurogenesis_decision",
                tags={
                    "layer": name,
                    "should_grow": str(should),
                    "efficiency": str(efficiency),
                    "fisher_density": str(fisher_density),
                    "delta_epist": str(delta_epist),
                    "mutual_info": str(mutual_info)
                }
            )
        except Exception as e:
            logger.warning(f"No se pudo enviar evento a InfluxDB: {e}")

        if should:
            self.cooldown_counter[name] = self.cooldown_steps
            logger.info(f"[Neurogénesis] Condición de crecimiento cumplida para la capa '{name}'. Iniciando cooldown.")

        return should

    def compute_diversity(self, acts: torch.Tensor) -> torch.Tensor:
        """Calcula diversidad entre activaciones usando correlación inversa."""
        if acts.shape[0] < 2:
            return torch.ones(acts.shape[1])
        corr = torch.corrcoef(acts.T)
        diversity = 1.0 - corr.abs().mean(dim=1)
        return diversity

    def grow(self, context):
        # Dummy: solo marca que se llamó
        context['neurogenesis_grow'] = True

    def step(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evalúa y ejecuta la neurogénesis. Si se realiza un cambio, devuelve un 
        'Adaptation Payload' con los artefactos necesarios para que el sistema se adapte.
        """
        fisher = context.get("fisher", {})
        grads = context.get("grad_accum", {})
        acts = context.get("activations", {})  # activaciones por capa
        impact = context.setdefault("neurogenesis_impact", {})
        adaptation_performed = False

        for name in self.layers_to_expand:
            if not self.should_grow_layer(name, context):
                continue

            layer_proxy = self._get_layer_proxy(name)
            if not isinstance(layer_proxy.layer, nn.Linear):
                continue

            out_old = layer_proxy.layer.out_features
            out_new = out_old + self.growth_unit
            
            logger.info(f"[Neurogénesis] Creciendo capa '{name}' de {out_old} a {out_new} neuronas.")

            # 1. Crear la nueva capa expandida
            new_layer = self._create_expanded_layer(layer_proxy.layer, out_new, grads.get(name), acts.get(name, []))

            # 2. Actualizar el modelo con la nueva capa
            layer_proxy.set(new_layer)

            # 3. Actualizar capas dependientes
            if name in self.dependent_layers:
                dep_name = self.dependent_layers[name]
                self.update_dependent_layer(dep_name, out_old, out_new)

            # 4. Registrar metadatos del crecimiento
            self.new_units[name] = {"start": out_old, "end": out_new, "step": 0}
            impact[name] = new_layer.weight[out_old:].abs().mean().item()
            adaptation_performed = True

        if adaptation_performed:
            # 5. Si hubo cambios, construir y devolver el Adaptation Payload
            payload = {
                "action": "NEUROGENESIS_PERFORMED",
                "impact": impact,
                "artifacts": {
                    # El artefacto clave es el conjunto completo de nuevos parámetros del modelo
                    "new_model_params": self.model.parameters()
                }
            }
            logger.info("[Neurogénesis] Payload de adaptación generado.")
            return payload

        return None # No se hizo nada

    def _create_expanded_layer(self, old_layer: nn.Linear, new_out_features: int, grad_tensor: Optional[torch.Tensor], activations: List[torch.Tensor]) -> nn.Linear:
        """Crea una nueva capa lineal, copiando pesos y expandiendo con una estrategia de diversidad."""
        new_layer = nn.Linear(old_layer.in_features, new_out_features, device=old_layer.weight.device)
        out_old = old_layer.out_features

        with torch.no_grad():
            # Copiar pesos y sesgos existentes
            new_layer.weight[:out_old] = old_layer.weight
            if old_layer.bias is not None:
                new_layer.bias[:out_old] = old_layer.bias
                new_layer.bias[out_old:] = 0

            # Estrategia de inicialización para nuevas neuronas basada en diversidad y gradientes
            if grad_tensor is None:
                grad_tensor = torch.ones_like(old_layer.weight)
            
            score = grad_tensor.abs().mean(dim=1) # Shape: [out_features]
            
            act_tensor = torch.stack(activations) if activations else torch.zeros(1, *score.shape, device=score.device)
            diversity = self.compute_diversity(act_tensor)

            if score.shape != diversity.shape:
                warnings.warn(f"Discrepancia de formas en neurogénesis: score {score.shape}, diversity {diversity.shape}. Usando solo score.")
                combined_score = score
            else:
                combined_score = score * diversity
            
            # Seleccionar las neuronas existentes más 'prometedoras' para clonar sus pesos
            num_to_clone = min(self.growth_unit, len(combined_score))
            idxs = torch.topk(combined_score, num_to_clone).indices
            new_layer.weight[out_old:] = old_layer.weight[idxs]

        # Registrar buffers para futura introspección
        new_layer.register_buffer("associative_memory", torch.zeros((1, new_out_features), device=new_layer.weight.device))
        return new_layer

    def update_dependent_layer(self, dep_name: str, old_size: int, new_size: int):
        dep_proxy = self._get_layer_proxy(dep_name)
        old_dep_layer = dep_proxy.layer
        
        if not isinstance(old_dep_layer, nn.Linear):
            logger.warning(f"Capa dependiente '{dep_name}' no es nn.Linear, no se puede actualizar.")
            return

        logger.info(f"[Neurogénesis] Actualizando capa dependiente '{dep_name}' para aceptar {new_size} entradas.")
        new_dep_layer = nn.Linear(new_size, old_dep_layer.out_features, device=old_dep_layer.weight.device)

        with torch.no_grad():
            # Copiar pesos para las conexiones existentes
            new_dep_layer.weight[:, :old_size] = old_dep_layer.weight
            # Inicializar nuevos pesos con Xavier para mantener la varianza de la señal
            nn.init.xavier_uniform_(new_dep_layer.weight[:, old_size:])
            
            if old_dep_layer.bias is not None:
                new_dep_layer.bias = nn.Parameter(old_dep_layer.bias.clone())

            if hasattr(old_dep_layer, "associative_memory"):
                mem = torch.zeros((1, new_size), device=new_dep_layer.weight.device)
                mem[:, :old_size] = old_dep_layer.associative_memory
                new_dep_layer.register_buffer("associative_memory", mem)

        dep_proxy.set(new_dep_layer)

    def _get_layer_proxy(self, layer_name: str) -> 'LayerProxy':
        """Obtiene un proxy para acceder y modificar una capa, incluso si es anidada."""
        parts = layer_name.split('.')
        obj = self.model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return LayerProxy(obj, parts[-1])

    def apply_soft_grow(self):
        for name, info in list(self.new_units.items()):
            module = getattr(self.model, name)
            start, end = info["start"], info["end"]

            if module.weight.grad is not None:
                with torch.no_grad():
                    delta_w = self.plasticity_factor * module.weight.grad[start:end]
                    module.weight.data[start:end] += delta_w
                    module.weight.grad[start:end] = 0

                    if module.bias is not None and module.bias.grad is not None:
                        delta_b = self.plasticity_factor * module.bias.grad[start:end]
                        module.bias.data[start:end] += delta_b
                        module.bias.grad[start:end] = 0

                    if hasattr(module, "associative_memory"):
                        module.associative_memory[:, start:end] += delta_w.mean(dim=1)

                info["step"] += 1
                if info["step"] >= self.soft_grow_steps:
                    del self.new_units[name]

    def consolidate(self, stability_threshold: float = 0.9):
        for name in self.layers_to_expand:
            module = getattr(self.model, name)
            if hasattr(module, "associative_memory"):
                mem = module.associative_memory
                std = mem.std()
                if std < stability_threshold:
                    for param in module.parameters():
                        param.requires_grad_(False)

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
