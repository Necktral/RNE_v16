# aeon/orchestrator/__init__.py (Versión Refactorizada)

import asyncio, logging, os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

# --- Importaciones de AEON ---
from ..protocols import HomeoProto, PlannerProto
from ..state import AEONState
from .data import build_data_pipeline
from .model import build_model_and_opt
from .trainer_loop import train_step, val_step
from .vitals import monitor_vitals

# Importar gestores evolutivos
from aeon.systems.evolution.auto_mutator import AutoMutator
from aeon.systems.evolution.neurogenesis import NeurogenesisManager
from aeon.systems.evolution.katana_pruner import KatanaPruner
from aeon.systems.evolution.predictive_coder import PredictiveCoder 
from aeon.systems.evolution.meta_optimizer import QuantumExponentialOptimizer
from aeon.systems.evolution.EvolutionaryRehabilitationCenter import QuantumQuarantineManager
from aeon.systems.evolution.predictive_coder import HierarchicalPredictiveCoder
from aeon.systems.homeostasis.energy_sensors import EnergySensors
from aeon.systems.homeostasis.thermodynamic_governor import ThermodynamicGovernor

log = logging.getLogger("AEON.Orchestrator")

class Orchestrator:
    """
    Núcleo operativo refactorizado de AEON.
    Recibe sus dependencias y opera sobre un estado centralizado.
    """
    def __init__(self, cfg: Any, state: AEONState, bus: "EventBus", planner: PlannerProto, homeo: HomeoProto, executor: ThreadPoolExecutor | None = None):
        log.info("Booting AEON executive consciousness (v2.0)...")
        self.cfg = cfg
        self.state = state
        self.bus = bus
        self.planner = planner
        self.homeo = homeo
        self.executor = executor or ThreadPoolExecutor(max_workers=min(4, os.cpu_count()))
        self._shutdown = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []

        # --- Setup de componentes a partir de la configuración ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader, vocab_size = build_data_pipeline(self.cfg)
        self.cfg.VOCAB_SIZE = vocab_size # Actualizar config con vocab_size real
        
        # Build del modelo y optimizador
        components = build_model_and_opt(self.cfg, self.device)
        self.trainer = self._setup_trainer(components, self.cfg, self.device)

        # Instanciación centralizada de gestores evolutivos
        self.model, self.opt = build_model_and_opt(cfg)
        self.auto_mutator = AutoMutator(self.model, cfg, state)
        self.neurogenesis_manager = NeurogenesisManager(self.model, layers_to_expand=[], dependent_layers={})
        self.katana_pruner = KatanaPruner(self.model, base_tau=0.01)
        self.predictive_coder = HierarchicalPredictiveCoder(cfg)
        self.evolution_center = QuantumQuarantineManager()
        self.quantum_optimizer = QuantumExponentialOptimizer(cfg)

        # Instanciación de módulos de homeostasis
        self.energy_sensors = EnergySensors(cfg.get('energy', {}))
        self.thermo_governor = ThermodynamicGovernor(cfg.get('thermo', {}))

    def _setup_trainer(self, components, cfg, device):
        from ..core.trainer import HNetTrainer
        trainer_config = {
            'device': device, 'dtype': torch.float16 if device.type == "cuda" else torch.float32,
            'vocab_size': cfg.VOCAB_SIZE, 'loss_config': cfg.LOSS_CONFIG,
            'chunk_ratio': cfg.CHUNK_RATIO,
        }
        return HNetTrainer(
            model=components['model'], embedding=components['embedding'], output_head=components['head'],
            optimizer=components['optimizer'], config=trainer_config
        )

    async def _train_forever(self) -> None:
        try:
            async for x, y in self._async_loader(self.train_loader):
                if self._shutdown.is_set(): break
                self.state.step += 1

                health = self.homeo.health_status()
                action = self.planner.decide(health.as_dict())

                if action in {"pause", "rest"}: continue

                loss, aux = await asyncio.to_thread(train_step, self.trainer, x.to(self.device), y.to(self.device), None)
                
                self.state.loss_history.append(loss)
                self.state.metrics.loss = loss
                
                # Enriquecimiento del contexto para sistemas evolutivos
                context = {
                    'metrics': train_metrics,
                    'homeostasis': self.homeo.get_status() if hasattr(self.homeo, 'get_status') else None,
                    'episteme': getattr(self.state, 'episteme', None),
                    'surprise': getattr(self.state, 'surprise_level', None),
                    # ...agrega aquí otros datos relevantes...
                }
                # Invocación del sistema evolutivo
                self.auto_mutator.step(context)

                # ... (resto de la lógica de logging y validación) ...

        except Exception as ex:
            log.exception("FALLO CATASTRÓFICO en el bucle de entrenamiento.")
            self.state.crisis = True
            self._shutdown.set()

    async def run(self) -> None:
        """Arranca todos los sistemas y gestiona el ciclo de vida."""
        try:
            # ... (código para iniciar bus y tasks) ...
            await self._shutdown.wait()
        finally:
            # ... (código de limpieza) ...
            if self.state.crisis:
                log.error("Parada por crisis homeostática o excepción fatal.")

    def _main_loop(self):
        """Ciclo principal que gestiona el estado y las decisiones de AEON."""
        while not self.state.done:
            if self._shutdown.is_set(): break

            # --- Monitoreo y ajustes de homeostasis ---
            self.energy_sensors.update()
            self.thermo_governor.update()

            context = {
                'metrics': self.state.metrics.as_dict(),
                'homeostasis': self.homeo.get_status() if hasattr(self.homeo, 'get_status') else None,
                'episteme': getattr(self.state, 'episteme', None),
                'surprise': getattr(self.state, 'surprise_level', None),
                'energy': self.energy_sensors.system_snapshot(),
                'health': getattr(self.thermo_governor, 'health_status', None),
                # ...agrega aquí otros datos relevantes...
            }

            # --- Toma de decisiones ---
            action = self.planner.decide(context)
            self._execute_action(action)

            # --- Registro de vitals ---
            monitor_vitals(self.state, self.cfg)

        log.info("Ciclo principal finalizado.")

    # ... (resto de métodos como _async_loader, etc.) ...