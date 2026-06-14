# [LEGACY — AEON FENIX-Δ en cuarentena] Orquestador AGI antiguo. Solo alcanzable
# vía exocortex/channels/cli/aeon_main_loop; el camino vivo (organismo/RTCME) no
# lo importa. Entrena sobre datos aleatorios. Ver docs/analysis/LEGACY_QUARANTINE.md.
"""
aeon_fenix_delta.py – Orquestador AGI para AEON FENIX-Δ
-------------------------------------------------------
• Modelos probabilísticos (VFE/EFE)
• Monitoreo de hardware en tiempo real
• Homeostasis avanzado (poda, veto térmico, resurrección)
• Basado en el marco teórico de AEON FENIX-Δ Fase 0
"""

import asyncio
import logging
import time
import tracemalloc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Set, Tuple, Awaitable, Optional
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor
import os


from .infrastructure import Event, EventBus, WorkerPool, ConfigLoader
from .probabilistic_models import GenerativeModel, ApproximatePosterior
from .metrics import SelfAwarenessMetrics, MAX_VRAM_GB, THERMAL_THRESHOLD, MAX_ENTROPY, CRITICAL_TEMP
from .train import QuantumDistributedTrainer
from .orchestration import OrchestratorLifecycle, RuntimeRunner
from .training.training_loop import TrainingLoop
from ..evolution.neurogenesis import NeurogenesisManager
from ..evolution.katana_pruner import KatanaPruner
from ..evolution.auto_mutator import AutoMutator
from ..control.adaptation_controller import AdaptationController
from ..control.crisis_router import CrisisRouter
from .homeo_controller import HomeoController
from ..telemetry.collector import TelemetryCollector
from ..telemetry.snapshot_service import SnapshotService
from src.core.epistemic_drift_predictor import EpistemicDriftPredictor
from src.evolution.meta_optimizer import QuantumExponentialOptimizer, QuantumExponentialConfig
from src.core.event_bus import event_bus  # EventBus centralizado para integración AGI

# Wrapper para los modelos
class CombinedModel(nn.Module):
    def __init__(self, generative_model, posterior_model):
        super().__init__()
        self.generative_model = generative_model
        self.posterior = posterior_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ────────────────────────────  ORCHESTRATOR PRINCIPAL  ────────────────────────────
class Orchestrator:
    HEARTBEAT_TOPIC = "heartbeat"
    CRISIS_TOPICS = ["VRAMUsageHigh", "ThermalAlert", "EntropyMax", "StabilityLoss"]

    def __init__(self, latent_dim: int = 0, tensorboard_writer=None, testing=False,
                 max_cycles: int = 1000, batch_size: int = 32, katana_warmup: int = 0,
                 noise_sigma: float = 0.0, val_interval: int = 1000):
        # --- MODO TEST/LITE: saltar inicialización pesada si AEON_TEST_MODE o --fast ---
        self.testing = testing or os.getenv("AEON_TEST_MODE") == "1"
        if self.testing:
            self.bus = EventBus()
            self.pool = WorkerPool()
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.metrics = SelfAwarenessMetrics()
            self._shutdown = asyncio.Event()
            self._tasks = []
            self.latent_dim = 4
            self.memory_capacity = 8
            self.max_cycles = 1
            self.batch_size = 2
            self.katana_warmup = 0
            self.noise_sigma = 0.0
            self.val_interval = 1
            self.device = torch.device("cpu")
            self.generative_model = nn.Identity()
            self.posterior = nn.Identity()
            self.combined_model = nn.Identity()
            self.optimizer = optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4)
            @dataclass
            class TrainerConfig:
                device: Any
                lr: float = 1e-4
                epochs: int = 1
                latent_dim: int = self.latent_dim
            self.trainer_config = TrainerConfig(device=self.device, latent_dim=self.latent_dim)
            self.trainer = type('DummyTrainer', (), {'_distributed_train_step': lambda *a, **kw: (0.0, torch.zeros(1, self.latent_dim)), 'optimizer': self.optimizer})()
            self.train_loader = [[torch.zeros(self.batch_size, self.latent_dim)]]
            self.val_loader = [[torch.zeros(self.batch_size, self.latent_dim)]]
            self.train_iter = iter(self.train_loader)
            self.neurogenesis_manager = None
            self.katana_pruner = None
            self.auto_mutator = None
            self.z = torch.zeros(1, self.latent_dim)
            self.a = torch.zeros(1, self.latent_dim)
            self.history = deque([0.0], maxlen=100)
            self.pynvml = None
            self.tensorboard_writer = None
            self.logger = logging.getLogger("aeon.orchestrator.test")
            self.log_sparsity = lambda *a, **kw: None
            self.homeo_controller = HomeoController(lambda: self.metrics.as_dict())
            self.event_bus = event_bus
            self.current_run_id = "test-run"
            self.drift_predictor = type(
                "DummyDriftPredictor",
                (),
                {
                    "update": lambda *a, **kw: None,
                    "check_drift": lambda *a, **kw: (False, ""),
                    "force_mutation": lambda *a, **kw: None,
                },
            )()
            self.meta_optimizer = type("DummyMetaOptimizer", (), {"step": lambda *a, **kw: None})()
            self.scheduler = None
            self._init_runtime_services()
            return

        self.bus = EventBus()
        self.pool = WorkerPool()
        self.executor = ThreadPoolExecutor(max_workers=self.pool.max_workers)
        self.metrics = SelfAwarenessMetrics()
        self._shutdown = asyncio.Event()
        self._tasks = []

        # Cargar configuración centralizada
        config = ConfigLoader()
        model_cfg = config.model_cfg
        self.latent_dim = latent_dim if latent_dim else model_cfg.latent_dim
        self.memory_capacity = model_cfg.memory_capacity

        # Guardar parámetros experimentales
        self.max_cycles = max_cycles
        self.batch_size = batch_size
        self.katana_warmup = katana_warmup
        self.noise_sigma = noise_sigma
        self.val_interval = val_interval

        # Inicializar modelos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generative_model = GenerativeModel(self.latent_dim).to(self.device)
        self.posterior = ApproximatePosterior(self.latent_dim).to(self.device)
        self.combined_model = CombinedModel(self.generative_model, self.posterior).to(self.device)

        self.optimizer = optim.AdamW(
            self.combined_model.parameters(), lr=1e-4
        )

        # Configuración del Trainer
        @dataclass
        class TrainerConfig:
            device: Any # torch.device object
            lr: float = 1e-4
            epochs: int = 1 # El orquestador controla el bucle, no el trainer
            latent_dim: int = self.latent_dim
        self.trainer_config = TrainerConfig(device=self.device, latent_dim=self.latent_dim)


        # --- INTEGRACIÓN DATA NORMALIZER ---
        from src.data.data_normalizer import DataNormalizer
        # Simula datos de entrada (en producción, cargarías tu dataset real)
        raw_data = torch.randn(12000, 64)
        normalizer = DataNormalizer()
        normalizer.fit(raw_data)
        normalizer.save_stats("config/data_stats.json")
        norm_data = normalizer.transform(raw_data)
        # --- INTEGRACIÓN val_loader_config.py ---
        from src.data.val_loader_config import create_train_val_loaders
        # Asume que norm_data ya está normalizado y disponible
        self.train_loader, self.val_loader = create_train_val_loaders(
            norm_data,
            batch_size=self.batch_size,
            val_split=2000/12000,
            shuffle=True,
            seed=42,
            normalizer=None  # Ya normalizado
        )
        self.train_iter = iter(self.train_loader)
        # Log de diagnóstico: muestra un batch de entrenamiento y validación para verificar normalización
        train_batch = next(iter(self.train_loader))[0]
        val_batch = next(iter(self.val_loader))[0]
        logger.info(f"[AEON][DEBUG] Batch train: mean={train_batch.mean():.3f}, std={train_batch.std():.3f}, shape={train_batch.shape}")
        logger.info(f"[AEON][DEBUG] Batch val: mean={val_batch.mean():.3f}, std={val_batch.std():.3f}, shape={val_batch.shape}")


        # Instanciar el Trainer
        self.trainer = QuantumDistributedTrainer(
            model=self.combined_model,
            train_loader=self.train_loader, # No se usará directamente en el bucle integrado
            config=self.trainer_config,
            tensorboard_writer=tensorboard_writer
        )
        # El optimizador del trainer debe ser el mismo que el del orquestador
        self.trainer.optimizer = self.optimizer


        # Inicializar pilar de Evolución
        layers_to_expand = ['generative_model.transition.0', 'posterior.inference.0']
        dependent_layers = {
            'generative_model.transition.0': 'generative_model.transition.2',
            'posterior.inference.0': 'posterior.inference.2'
        }
        self.neurogenesis_manager = NeurogenesisManager(
            model=self.combined_model,
            layers_to_expand=layers_to_expand,
            dependent_layers=dependent_layers
        )
        self.katana_pruner = KatanaPruner(model=self.combined_model)
        self.auto_mutator = AutoMutator(
            neurogenesis_manager=self.neurogenesis_manager,
            katana_pruner=self.katana_pruner,
            testing=testing
        )

        # Estado interno
        self.z = torch.randn(1, latent_dim, device=self.device).float()
        self.a = torch.zeros(1, latent_dim, device=self.device).float() # Acción inicial nula
        self.history = deque(maxlen=100)
        self.history.append(0.0) # Pérdida inicial

        # Inicializar NVML
        try:
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            self.gpu_handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
        except ImportError:
            self.pynvml = None

        self.tensorboard_writer = tensorboard_writer

        # --- INTEGRACIÓN eval_loop.py ---
        from src.core.eval_loop import eval_loop
        self.eval_loop = eval_loop  # Asigna la función para uso en validación

        # --- INTEGRACIÓN ReduceLROnPlateau scheduler ---
        from src.core.scheduler import LRSchedulerFactory
        self.scheduler = LRSchedulerFactory.create(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # --- INTEGRACIÓN logging_utils.py ---
        from src.utils.logging_utils import get_logger, log_metrics
        self.logger = get_logger("aeon.orchestrator", level=logging.INFO)
        # Reemplaza logger por self.logger en todo el Orchestrador para logs estructurados

        # --- INTEGRACIÓN sparsity_logger.py (opcional, post Katana) ---
        from src.core.sparsity_logger import log_sparsity
        self.log_sparsity = log_sparsity
        # Para loguear sparsity tras cada ciclo, puedes llamar:
        # self.log_sparsity(self.combined_model, logger=self.logger, step=cycle)

        # Inicializar HomeoController
        self.homeo_controller = HomeoController(lambda: self.metrics.as_dict())

        # Inicializar predictor de deriva epistémica y meta-optimizador cuántico
        self.drift_predictor = EpistemicDriftPredictor(window_size=50, threshold=0.0005, cooldown=1000)
        self.meta_optimizer = QuantumExponentialOptimizer(QuantumExponentialConfig())
        if hasattr(self, 'modules'):
            self.drift_predictor.modules = self.modules

        # Integración EventBus centralizado
        self.event_bus = event_bus
        self.current_run_id = f"run-{int(time.time())}"
        self._init_runtime_services()

    def _init_runtime_services(self):
        self.lifecycle = OrchestratorLifecycle()
        self.telemetry_collector = TelemetryCollector()
        self.snapshot_service = SnapshotService()
        self.adaptation_controller = AdaptationController()
        self.crisis_router = CrisisRouter(
            logger=self.logger,
            bus=self.bus,
            global_event_bus=self.event_bus,
            metrics=self.metrics,
            shutdown_event=self._shutdown,
        )
        self.crisis_router.wire_global_handlers()
        self.training_loop_service = TrainingLoop(
            orchestrator=self,
            telemetry_collector=self.telemetry_collector,
            snapshot_service=self.snapshot_service,
            adaptation_controller=self.adaptation_controller,
            lifecycle=self.lifecycle,
        )
        self.runner = RuntimeRunner(
            orchestrator=self,
            lifecycle=self.lifecycle,
            training_loop=self.training_loop_service,
            crisis_router=self.crisis_router,
        )

    def _on_crisis_event(self, payload):
        self.crisis_router._on_crisis_event(payload)

    def _on_homeostasis_violation(self, payload):
        self.crisis_router._on_homeostasis_violation(payload)

    def _on_quantum_mutation(self, payload):
        self.crisis_router._on_quantum_mutation(payload)

    def _on_module_pruned(self, payload):
        self.crisis_router._on_module_pruned(payload)

    def _on_module_spawned(self, payload):
        self.crisis_router._on_module_spawned(payload)

    async def run_forever(self):
        await self.runner.run_forever()

    async def _get_next_observation(self):
        return await self.training_loop_service._get_next_observation()

    async def _main_loop(self):
        await self.training_loop_service.run()

    async def _heartbeat_loop(self):
        """Compatibilidad temporal: heartbeat se emite dentro de TrainingLoop."""
        pass

    async def _monitor_vitals(self):
        await self.crisis_router.monitor_vitals()

    def _update_metrics(self):
        self.telemetry_collector.update_metrics(
            metrics=self.metrics,
            history=list(self.history),
            pynvml_module=self.pynvml,
            gpu_handle=getattr(self, "gpu_handle", None),
        )

    def _build_adaptation_context(self, event: Optional[Event] = None) -> Dict[str, Any]:
        return self.adaptation_controller.build_context(
            metrics=self.metrics,
            history=list(self.history),
            model=self.combined_model,
            optimizer=self.optimizer,
            device=self.device,
            event_topic=event.topic if event else None,
            event_severity=event.severity if event else None,
        )

    async def handle_adaptation_pressure(self, event: Event):
        await self.crisis_router.handle_adaptation_pressure(event)

    async def eval_loop(self, model, val_loader):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                o_t = batch[0]
                z_prev = torch.zeros((o_t.shape[0], 32), device=o_t.device)  # dummy
                a_prev = torch.zeros((o_t.shape[0], 32), device=o_t.device)  # dummy
                output = model.generative_model(z_prev, a_prev)
                target = o_t[:, :output.shape[1]]
                loss = torch.nn.functional.mse_loss(output, target)
                total_loss += loss.item()
                total_batches += 1
        return total_loss / max(1, total_batches)

    def health_status(self):
        return self.homeo_controller.health_status()

    async def shutdown(self):
        # Cierre robusto: EventBus, WorkerPool, ThreadPoolExecutor
        await self.bus.shutdown()  # Asegúrate de que EventBus implemente shutdown/close correctamente
        self.pool.close()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        self.logger.info("[DEBUG] Orchestrator.shutdown ejecutado")

# ───────────────────────────  DEMO  ──────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import logging
    orch = Orchestrator()
    async def main():
        try:
            await orch.run_forever()
        finally:
            # Cancelar todas las tareas vivas
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            logging.debug("Tareas vivas al salir: %s", pending)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logging.info("[AEON] Shutdown limpio con %d tareas canceladas", len(pending))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Apagado por usuario")
