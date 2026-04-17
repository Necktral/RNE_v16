# aeon/systems/actuators.py

import torch
import logging
from torch.optim.optimizer import Optimizer

from aeon.models.hnet import HNet
# Importamos los sistemas evolutivos con los que interactuará
from aeon.systems.evolution.katana_pruner import KatanaPruner
from aeon.systems.evolution.neurogenesis import NeurogenesisManager

logger = logging.getLogger("AEON.Actuators")

class AeonActuators:
    """
    Brazo ejecutor de AEON. Traduce decisiones estratégicas del Planner y el AutoMutator
    en operaciones concretas sobre el modelo, el optimizador y otros componentes.
    """
    def __init__(self, model: HNet, optimizer: Optimizer, pruner: KatanaPruner, neurogenesis: NeurogenesisManager, config: dict):
        """
        Inicializa los actuadores con acceso a los componentes clave del sistema.
        """
        self.model = model
        self.optimizer = optimizer
        self.pruner = pruner
        self.neurogenesis = neurogenesis
        self.config = config
        self.original_lr = optimizer.param_groups[0]['lr']

    def execute_planner_action(self, action: str, context: dict):
        """
        Punto de entrada que recibe una acción del Planner y la ejecuta.
        """
        action_method = getattr(self, f"action_{action}", self.action_unknown)
        action_method(context)

    def apply_adaptation_payloads(self, payloads: list):
        """
        Aplica los 'Adaptation Payloads' generados por el sistema evolutivo.
        """
        if not payloads:
            return

        for payload in payloads:
            action = payload.get("action")
            if action == "PRUNING_MASKS_UPDATED":
                self._apply_pruning_masks(payload.get("artifacts", {}).get("pruning_masks", {}))
            elif action == "NEUROGENESIS_PERFORMED":
                # La neurogénesis modifica el modelo in-situ, pero puede requerir
                # reiniciar el estado del optimizador si la arquitectura cambia drásticamente.
                self._reset_optimizer_state()
            else:
                logger.warning(f"Payload de adaptación desconocido: {action}")

    # --- Métodos de Acción para el Planner ---

    def action_train(self, context: dict):
        """Acción por defecto: continuar el entrenamiento."""
        logger.debug("ACTUATOR: Acción -> TRAIN. Operación normal.")
        pass

    def action_rest(self, context: dict):
        """Acción: Pausar el aprendizaje. Ya manejado por el Orchestrator."""
        logger.info("ACTUATOR: Acción -> REST. El Orchestrator pausará el ciclo.")
        pass

    def action_pause(self, context: dict):
        """Acción: Pausar el aprendizaje. Ya manejado por el Orchestrator."""
        logger.info("ACTUATOR: Acción -> PAUSE. El Orchestrator pausará el ciclo.")
        pass

    def action_unknown(self, context: dict):
        """Acción para decisiones no reconocidas."""
        logger.error(f"ACTUATOR: Recibida acción desconocida del Planner en contexto: {context}")
        pass

    # --- Métodos de Aplicación de Payloads ---

    def _apply_pruning_masks(self, masks: dict):
        """
        Aplica las máscaras de poda a los gradientes del modelo.
        Esta es una forma de poda "suave" que anula las actualizaciones de los pesos podados.
        """
        logger.info("ACTUATOR: Aplicando nuevas máscaras de poda a los gradientes.")
        for name, param in self.model.named_parameters():
            if name in masks and param.grad is not None:
                param.grad.data.mul_(masks[name])

    def _reset_optimizer_state(self):
        """
        Reinicia el estado del optimizador. Necesario después de que la neurogénesis
        añade nuevos parámetros que no tienen momentos (momenta) acumulados.
        """
        logger.warning("ACTUATOR: Reiniciando estado del optimizador debido a neurogénesis.")
        self.optimizer.state = {} # Forma simple de resetear. Puede ser más sofisticado.

    def apply_pruning_masks(self, masks: dict):
        """
        Aplica las máscaras de poda a los gradientes del modelo desde el Orchestrator.
        """
        self._apply_pruning_masks(masks)