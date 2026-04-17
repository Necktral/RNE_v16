# aeon/core/trainer.py
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# Dependencias del ecosistema AEON
from ..models.hnet import HNet
from ..core.loss import UnifiedSeedLoss
from ..aeon_types import HealthStatus

import logging
log = logging.getLogger("AEON.Trainer")

class HNetTrainer:
    """
    Gestiona el ciclo de entrenamiento para el modelo HNet.
    Actúa como el puente entre el modelo (HNet), la función objetivo (UnifiedSeedLoss)
    y el estado del sistema (HealthStatus).
    """
    def __init__(self,
                 model: HNet,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: UnifiedSeedLoss,
                 config: Dict[str, Any]):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = next(model.parameters()).device
        
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.target_ratio = self.config.get('model', {}).get('target_ratio', 6.0)

    def train_step(self,
                   x_batch: torch.Tensor,
                   y_batch: torch.Tensor,
                   health: HealthStatus) -> Tuple[float, Dict[str, float]]:
        """
        Realiza un único paso de entrenamiento completo, integrando el estado de salud.
        """
        self.model.train()

        # Mover datos al dispositivo
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

        # 1. FORWARD PASS
        # NOTA CRÍTICA: Se asume que el modelo HNet se ha actualizado para devolver
        # los tensores 'posterior', 'prior' y 'hidden_states' que la loss `loos3q.txt` espera.
        # Esto es necesario para la alineación completa con la fórmula.
        # Ejemplo de llamada esperada:
        # model_output = self.model(x_batch_embedded, target_ratio=self.target_ratio)
        model_output = self.model(x_batch, target_ratio=self.target_ratio)

        # 2. CÁLCULO DE LA PÉRDIDA UNIFICADA
        # La función de pérdida consume la salida del modelo y el estado de salud
        # para calcular la pérdida total y las métricas detalladas.
        total_loss, loss_metrics = self.loss_fn(model_output, y_batch, health)

        # 3. APLICACIÓN DE HOOKS DE SEGURIDAD (ECUACIÓN 7 de la fórmula)
        # La función de pérdida puede sugerir acciones inmediatas. El trainer las ejecuta.
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr, actions = self.loss_fn.apply_safety_hooks(health, current_lr)

        if 'throttle_lr' in actions and new_lr != current_lr:
            log.warning(f"HOOK DE SEGURIDAD: Temperatura crítica detectada. Reduciendo LR a {new_lr:.6f}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            loss_metrics['action_hook'] = f"lr_throttled_to_{new_lr:.6f}"
        
        if 'reset_boundaries' in actions:
            log.warning("HOOK DE SEGURIDAD: Baja confianza detectada. Señal de reseteo de fronteras enviada.")
            # La lógica para resetear b_t se manejaría dentro del modelo o con un hook.
            loss_metrics['action_hook'] = "boundary_reset_signaled"

        # 4. BACKWARD PASS
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Recorte de gradientes para estabilidad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        
        self.optimizer.step()
        
        return total_loss.item(), loss_metrics