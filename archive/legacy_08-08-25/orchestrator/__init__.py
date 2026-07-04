# aeon/orchestrator/__init__.py (Versión Unificada v3.1 - Estratégica)

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

# --- Importaciones de AEON ---
from ..protocols import HomeoProto, PlannerProto
from ..state import AEONState
from ..core.infrastructure import EventBus
from ..core.trainer import HNetTrainer
from .data import build_data_pipeline
from .model import build_model_and_opt
from .vitals import monitor_vitals
from aeon.systems.evolution.katana_pruner import KatanaPruner
from aeon.systems.evolution.neurogenesis import NeurogenesisManager
from aeon.systems.evolution.auto_mutator import AutoMutator
from aeon.systems.actuators import AeonActuators

# --- Sistemas de Alto Nivel ---
from aeon.systems.homeostasis.controller import HomeoController
from aeon.systems.planning.planner import AEONPlanner

log = logging.getLogger("AEON.Orchestrator")

class Orchestrator:
    """
    Núcleo operativo unificado de AEON (v3.1).
    Orquesta el ciclo de vida completo: percepción, decisión, acción y evolución,
    con una lógica de planificación y evolución mejorada.
    """
    def __init__(self, cfg: Any, state: AEONState, bus: "EventBus", planner: PlannerProto, homeo: HomeoProto, executor: ThreadPoolExecutor | None = None):
        log.info("Iniciando conciencia ejecutiva de AEON (v3.1)...")
        self.cfg = cfg
        self.state = state
        self.bus = bus
        self.planner = planner
        self.homeo = homeo
        self.executor = executor or ThreadPoolExecutor(max_workers=min(4, os.cpu_count()))
        self._shutdown = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []

        self._setup_components()

    def _setup_components(self):
        """
        Inicializa y conecta todos los módulos de AEON.
        """
        log.info("Ensamblando arquitectura cognitiva...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader, self.cfg.VOCAB_SIZE = build_data_pipeline(self.cfg)
        
        # Construcción del modelo y componentes asociados
        embedding, model, head, optimizer, sched = build_model_and_opt(self.cfg, self.device)
        self.model = model
        self.optimizer = optimizer
        self.embedding = embedding
        self.head = head
        self.sched = sched
        
        self.trainer = HNetTrainer(
            model=self.model,
            embedding=self.embedding,
            output_head=self.head,
            optimizer=self.optimizer,
            config=self.cfg
        )
        
        # --- Sistemas Evolutivos ---
        self.katana_pruner = KatanaPruner(self.model, base_tau=self.cfg.get('BASE_TAU', 0.01))
        self.neurogenesis_manager = NeurogenesisManager(
            self.model,
            layers_to_expand=self.cfg.get('LAYERS_TO_EXPAND', []),
            dependent_layers=self.cfg.get('DEPENDENT_LAYERS', {})
        )
        self.auto_mutator = AutoMutator(
            neurogenesis_manager=self.neurogenesis_manager,
            katana_pruner=self.katana_pruner
        )
        
        # --- Actuadores ---
        self.actuators = AeonActuators(
            self.model,
            self.optimizer,
            self.katana_pruner,
            self.neurogenesis_manager,
            self.cfg
        )
        log.info("Todos los sistemas están instanciados y listos.")

    def _build_context_for_evolution(self):
        """
        Centraliza la construcción del contexto evolutivo para el AutoMutator.
        """
        return {
            'metrics': self.state.metrics.as_dict(),
            'health': self.homeo.health_status().as_dict(),
            'step': self.state.step,
            'loss': self.state.metrics.loss,
            'loss_history': self.state.loss_history
        }

    async def _main_loop(self) -> None:
        """
        El único y autoritativo bucle de vida de AEON.
        Fases:
        1. Percepción (Homeostasis y Epistemología)
        2. Decisión (Planner)
        3. Acción (Trainer / Actuadores)
        4. Evolución (AutoMutator)
        5. Logging y Validación
        """
        log.info("Iniciando bucle de vida principal...")
        self.model.train()
        
        async for x_batch, y_batch in self._async_loader(self.train_loader):
            if self._shutdown.is_set() or self.state.step >= self.cfg.TRAINING_STEPS:
                break
            
            # --- FASE 1: PERCEPCIÓN (Holística) ---
            # Se captura el estado del hardware. El estado cognitivo (loss, etc.)
            # ya está presente en `self.state` desde el ciclo anterior.
            health = self.homeo.health_status()
            self.state.metrics.update_from_health(health)

            # --- FASE 2: DECISIÓN (Planificación Informada) ---
            # ANÁLISIS ESTRATÉGICO: El planificador ahora recibe el estado completo (físico y cognitivo)
            # para tomar decisiones que equilibren supervivencia y crecimiento.
            action = self.planner.decide(self.state)
            self.bus.emit("planner_decision", {"action": action, "state": self.state.as_dict()})

            # --- FASE 3: ACCIÓN ---
            if action == "train":
                x, y = x_batch.to(self.device), y_batch.to(self.device)
                
                # El Trainer ejecuta el paso de aprendizaje y actualiza el estado cognitivo
                loss, metrics = await asyncio.to_thread(self.trainer.train_step, x, y)
                
                # Actualizamos el estado con la nueva información cognitiva para el *próximo* ciclo
                self.state.loss_history.append(loss)
                self.state.metrics.loss = loss
                # Aquí se actualizarían otras métricas cognitivas retornadas por el trainer
            
            elif action in {"pause", "rest"}:
                log.warning(f"PLANNER -> {action.upper()}. Pausando ciclo por 5s para recuperación homeostática.")
                await asyncio.sleep(5)
                continue # Saltamos la evolución si la acción es no hacer nada
            
            # --- FASE 4: EVOLUCIÓN (Proceso Continuo) ---
            # ANÁLISIS ESTRATÉGICO: La evolución ahora es un proceso fundamental del ciclo de vida,
            # no un subproducto del entrenamiento. Puedo decidir mutar incluso si estoy en un estado de "reflexión".
            context = self._build_context_for_evolution()
            adaptation_payloads = self.auto_mutator.step(context)
            if adaptation_payloads:
                self.bus.emit("evolution_trigger", {"payloads": adaptation_payloads})
                self.actuators.apply_adaptation_payloads(adaptation_payloads)

            # --- FASE 5: LOGGING Y VALIDACIÓN ---
            self.state.step += 1
            if self.state.step % self.cfg.LOG_INTERVAL == 0:
                self._log_status(action)
            if self.state.step % self.cfg.VAL_INTERVAL == 0:
                self._validate()

        self._shutdown.set()

    def _log_status(self, action: str):
        """Lógica de logging centralizada."""
        health = self.state.metrics
        log.info(
            f"Step {self.state.step} | Loss: {self.state.metrics.loss:.4f} | Acción: {action.upper()} | "
            f"Temp: {health.temp_c:.1f}°C | VRAM: {health.vram_gb:.2f}/{health.vram_total_gb:.2f} GB"
        )

    def _validate(self):
        """Lógica de validación centralizada."""
        # Se podría ejecutar de forma asíncrona para no bloquear el bucle principal
        log.info(f"Step {self.state.step} -> Iniciando ciclo de validación...")
        # ...implementa validación aquí...
        pass

    async def _async_loader(self, loader):
        """Convierte un dataloader de PyTorch en un generador asíncrono."""
        while True:
            for batch in loader:
                yield await asyncio.to_thread(lambda: batch)