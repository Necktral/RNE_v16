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
from ..evolution.neurogenesis import NeurogenesisManager
from ..evolution.katana_pruner import KatanaPruner
from ..evolution.auto_mutator import AutoMutator
from .homeo_controller import HomeoController
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
        # Suscribirse a eventos críticos globales
        self.event_bus.on('crisis', self._on_crisis_event)
        self.event_bus.on('homeostasis_violation', self._on_homeostasis_violation)
        self.event_bus.on('quantum_mutation_applied', self._on_quantum_mutation)
        self.event_bus.on('module_pruned', self._on_module_pruned)
        self.event_bus.on('module_spawned', self._on_module_spawned)

    def _on_crisis_event(self, payload):
        self.logger.warning(f"[EventBus] Crisis global detectada en Orchestrator | Payload: {payload}")
        # Hook: lógica de respuesta a crisis global (extensible)

    def _on_homeostasis_violation(self, payload):
        self.logger.warning(f"[EventBus] Homeostasis violation detectada | Payload: {payload}")
        # Hook: lógica de respuesta a violación homeostática

    def _on_quantum_mutation(self, payload):
        self.logger.info(f"[EventBus] Mutación cuántica aplicada | Payload: {payload}")
        # Hook: lógica de respuesta a mutación cuántica

    def _on_module_pruned(self, payload):
        self.logger.info(f"[EventBus] Módulo podado | Payload: {payload}")
        # Hook: lógica de respuesta a poda

    def _on_module_spawned(self, payload):
        self.logger.info(f"[EventBus] Nuevo módulo creado | Payload: {payload}")
        # Hook: lógica de respuesta a creación de módulo

    async def run_forever(self):
        tracemalloc.start()
        self.logger.info("[DEBUG] run_forever iniciado")
        await self.bus.start()

        # Suscribir handlers
        self.bus.subscribe("VRAMUsageHigh", self.handle_adaptation_pressure)
        self.bus.subscribe("ThermalAlert", self.handle_adaptation_pressure)
        self.bus.subscribe("EntropyMax", self.handle_adaptation_pressure)
        self.bus.subscribe("StabilityLoss", self.handle_adaptation_pressure)

        # Tareas principales
        self._tasks += [
            asyncio.create_task(self._main_loop()),
            asyncio.create_task(self._monitor_vitals())
        ]

        await self._shutdown.wait()
        self.logger.info("[DEBUG] run_forever: shutdown recibido, cancelando tareas...")
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.executor.shutdown()
        self.logger.info("[DEBUG] run_forever finalizado")

    async def _get_next_observation(self):
        """Obtiene la siguiente observación del 'entorno' (DataLoader)."""
        try:
            return next(self.train_iter)[0]
        except StopIteration:
            # Reiniciar el iterador si se acaban los datos
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)[0]

    async def _main_loop(self):
        self.logger.info("[DEBUG] _main_loop iniciado")
        challenger = None
        if hasattr(self, 'modules') and self.modules:
            from src.cognition.cognitive_self_challenge import CognitiveSelfChallenge
            challenger = CognitiveSelfChallenge(self.modules)
        cycle = 0
        # Log de hiperparámetros al inicio
        if self.tensorboard_writer is not None:
            hparams = {
                'latent_dim': getattr(self, 'latent_dim', 32),
                'optimizer': type(self.optimizer).__name__,
                'lr': self.optimizer.param_groups[0]['lr'] if hasattr(self.optimizer, 'param_groups') else None,
            }
            self.tensorboard_writer.add_text('hparams', str(hparams), 0)
        while not self._shutdown.is_set():
            try:
                cycle += 1
                self.logger.info(f"Ciclo {cycle} iniciado.")
                # 1. Obtener estado y observación
                z_prev = self.z
                a_prev = self.a
                o_t = await self._get_next_observation()
                self.logger.info(f"Observación obtenida: {o_t.shape if hasattr(o_t, 'shape') else type(o_t)}")

                # --- Asegurar que todos los tensores estén en el mismo dispositivo ---
                z_prev = z_prev.to(self.device)
                a_prev = a_prev.to(self.device)
                o_t = o_t.to(self.device)

                # 2. Ejecutar un paso de entrenamiento y obtener la pérdida y el nuevo estado
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.trainer._distributed_train_step(z_prev, a_prev, o_t.view(1, -1))
                )
                if result is None:
                    self.logger.warning("Paso de entrenamiento devolvió None. Esperando...")
                    await asyncio.sleep(0.1)
                    continue
                loss, new_z = result
                if loss is None or new_z is None:
                    self.logger.warning("Paso de entrenamiento devolvió None para loss o new_z. Ciclo omitido.")
                    await asyncio.sleep(0.1)
                    continue

                self.logger.info(f"Loss: {loss}")
                self.z = new_z
                self.history.append(loss)
                # --- LOGGING TENSORBOARD ---
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar("train/loss", loss, cycle)
                    self.tensorboard_writer.add_scalar("system/vram_gb", self.metrics.vram_usage_gb, cycle)
                    self.tensorboard_writer.add_scalar("system/temp", self.metrics.temperature, cycle)
                    self.tensorboard_writer.add_scalar("system/entropy", self.metrics.entropy, cycle)
                    self.tensorboard_writer.add_scalar("system/stability", self.metrics.stability, cycle)
                    # Histogramas de pesos y gradientes cada 1000 ciclos
                    if cycle % 1000 == 0:
                        for name, param in getattr(self.combined_model, 'named_parameters', lambda:[])():
                            self.tensorboard_writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), cycle)
                            if param.grad is not None:
                                self.tensorboard_writer.add_histogram(f"grads/{name}", param.grad.cpu().numpy(), cycle)
                # 3. Actualizar métricas
                self._update_metrics()
                self.logger.info(f"Métricas actualizadas: {self.metrics.as_dict()}")
                # Emitir evento de métricas actualizadas
                self.event_bus.emit('orchestrator_metrics', {
                    'cycle': cycle,
                    'metrics': self.metrics.as_dict()
                })

                # --- INTEGRACIÓN DERIVA EPISTÉMICA Y META-OPTIMIZADOR ---
                vfe = getattr(self.metrics, 'vfe', None)
                eta = getattr(self.metrics, 'eta_bayes', None)
                self.drift_predictor.update(eta, vfe)
                alerta, razon = self.drift_predictor.check_drift(cycle)
                if alerta:
                    self.logger.warning(f"[DERIVA EPISTÉMICA] Detectada: {razon}. Ejecutando intervención de emergencia.")
                    self.drift_predictor.force_mutation(razon)
                if self.metrics.vram_usage_gb < 0.95 * MAX_VRAM_GB and self.metrics.temperature < 0.95 * CRITICAL_TEMP:
                    self.meta_optimizer.step(
                        {
                            'vram': self.metrics.vram_usage_gb / MAX_VRAM_GB,
                            'thermal': self.metrics.temperature / CRITICAL_TEMP,
                            'entropy': self.metrics.entropy,
                            'cognitive_load': self.metrics.stability
                        },
                        lambda uid: eta if eta is not None else 1.0
                    )
                else:
                    self.logger.warning(f"Mutación/NAS bloqueada por límites físicos: vram={self.metrics.vram_usage_gb:.2f}, temp={self.metrics.temperature:.2f}")

                # 4. Construir contexto y tomar decisiones de adaptación
                ctx = self._build_adaptation_context()
                self.logger.info(f"Contexto de adaptación construido.")
                adaptation_payloads = None
                if self.auto_mutator is not None:
                    adaptation_payloads = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.auto_mutator.step(ctx)
                    )

                # 5. Aplicar las adaptaciones si existen
                if adaptation_payloads:
                    self.logger.info(f"Aplicando {len(adaptation_payloads)} adaptación(es)...")
                    for payload in adaptation_payloads:
                        if hasattr(self.trainer, 'apply_adaptation'):
                            await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                lambda: self.trainer.apply_adaptation(payload)
                            )
                    if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                        self.optimizer = self.trainer.optimizer
                    self.logger.info("Adaptación(es) aplicada(s) correctamente.")

                # --- DESAFÍO COGNITIVO INTERNO (Self-Challenge) ---
                if challenger is not None:
                    resultado = challenger.generate_challenge(cycle)
                    if resultado is not None:
                        self.logger.info(f"[AEON] Resultado del desafío cognitivo en ciclo {cycle}: {resultado}")
                        # Emitir evento de desafío cognitivo
                        self.event_bus.emit('cognitive_challenge', {
                            'cycle': cycle,
                            'result': resultado
                        })

                # 6. Publicar heartbeat con las métricas actualizadas
                await self.bus.publish(Event(
                    topic=self.HEARTBEAT_TOPIC,
                    payload=self.metrics.as_dict(),
                    severity="INFO"
                ))
                # Emitir evento heartbeat global
                self.event_bus.emit('heartbeat', {
                    'cycle': cycle,
                    'metrics': self.metrics.as_dict()
                })
                # 7. Validación y checkpoint cada 1000 pasos
                if cycle % 1000 == 0 and hasattr(self, 'scheduler') and self.scheduler is not None:
                    try:
                        val_loss = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: self.eval_loop(self.combined_model, self.val_loader)
                        )
                        if not isinstance(val_loss, (float, int)) or not np.isfinite(val_loss):
                            self.logger.warning(f"Val Loss no numérico o infinito: {val_loss}")
                            val_loss = float('nan')
                        self.logger.info(f"Val Loss: {val_loss}")
                        if self.tensorboard_writer is not None:
                            self.tensorboard_writer.add_scalar("val/loss", val_loss, cycle)
                        if hasattr(torch, 'save') and hasattr(self.combined_model, 'state_dict'):
                            torch.save(self.combined_model.state_dict(), f"checkpoints/aeon_{cycle}.pt")
                        # Actualiza el scheduler con el val_loss
                        if hasattr(self, 'scheduler') and self.scheduler is not None and val_loss is not None and np.isfinite(val_loss):
                            self.scheduler.step(val_loss)
                            lr = self.optimizer.param_groups[0]['lr'] if hasattr(self.optimizer, 'param_groups') and getattr(self.optimizer, 'param_groups', None) else None
                            self.logger.info(f"LR actualizado: {lr}")
                    except Exception as e:
                        self.logger.exception(f"Error en validación/checkpoint: {e}")
                self.logger.info(f"Ciclo {cycle} finalizado. Esperando próximo ciclo...")
                await asyncio.sleep(0.1)

                # ─── DETENER SI ALCANZAMOS EL LÍMITE ───
                if self.max_cycles and cycle >= self.max_cycles:
                    self.logger.info(f"✓ Alcanzados {cycle} ciclos — apagando Orchestrator.")
                    self._shutdown.set()
                    break
            except Exception as e:
                self.logger.exception(f"Excepción inesperada en ciclo {cycle}: {e}")
                break
        # --- Cierre seguro del writer ---
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        self.logger.info("[DEBUG] _main_loop finalizado")

    async def _heartbeat_loop(self):
        """Este método queda obsoleto y es reemplazado por _main_loop."""
        pass

    async def _monitor_vitals(self):
        self.logger.info("[DEBUG] _monitor_vitals iniciado")
        while not self._shutdown.is_set():
            m = self.metrics.as_dict()
            if m["Mem"] >= 0.95:
                await self.bus.publish(Event("VRAMUsageHigh", m, "CRITICAL"))
            if m["Temp"] >= 0.9:
                await self.bus.publish(Event("ThermalAlert", m, "CRITICAL"))
            if m["Entropy"] >= 0.98:
                await self.bus.publish(Event("EntropyMax", m, "WARN"))
            if m["Stability"] >= 1e3:
                await self.bus.publish(Event("StabilityLoss", m, "CRITICAL"))
            await asyncio.sleep(5)
        self.logger.info("[DEBUG] _monitor_vitals finalizado")

    def _update_metrics(self):
        # VRAM y temperatura
        if self.pynvml:
            vmem = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.metrics.vram_usage_gb = vmem.used / (1024 ** 3)
            self.metrics.temperature = self.pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle, self.pynvml.NVML_TEMPERATURE_GPU
            )
            self.metrics.dissipated_power = self.pynvml.nvmlDeviceGetPowerUsage(
                self.gpu_handle
            ) / 1000
        else:
            cur, _ = tracemalloc.get_traced_memory()
            self.metrics.vram_usage_gb = cur / (1024 ** 3)
            self.metrics.temperature = 40.0

        # Entropía del sistema
        try:
            import psutil
            cpu = psutil.cpu_percent() / 100
            mem = psutil.virtual_memory().percent / 100
            self.metrics.entropy = 0.7 * cpu + 0.3 * mem
        except ImportError:
            self.metrics.entropy = 0.5

        # Estabilidad (λ_max) -> Ahora basado en la VFE real
        if len(self.history) > 1:
            # Usamos la fluctuación de la pérdida como indicador de estabilidad
            self.metrics.stability = abs(self.history[-1] - self.history[-2])
        else:
            self.metrics.stability = 0.0

    def _build_adaptation_context(self, event: Optional[Event] = None) -> Dict[str, Any]:
        """Construye el diccionario de contexto para el AutoMutator."""
        ctx = {
            "delta_epist": self.history[-1] - self.history[-2] if len(self.history) > 1 else 0.0,
            "mutual_info": self.metrics.stability, # Proxy para información mutua
            "thermal_risk": self.metrics.temperature / CRITICAL_TEMP,
            "vram_usage": self.metrics.vram_usage_gb / MAX_VRAM_GB,
            "pruning_intensity": 0.0, # Será actualizado por el mutator
            "neurogenesis_impact": {}, # Será actualizado por el mutator
            "model": self.combined_model,
            "optimizer": self.optimizer,
            "device": self.device,
            # --- Claves necesarias para testing ---
            "vram_usage_gb": self.metrics.vram_usage_gb,
            "MAX_VRAM_GB": MAX_VRAM_GB,
            "history": list(self.history),
        }
        if event:
            ctx.update({
                "event_topic": event.topic,
                "event_severity": event.severity
            })
        return ctx

    # ─────────────────── HANDLERS DE CRISIS ───────────────────
    async def handle_adaptation_pressure(self, event: Event):
        """Centraliza la respuesta a cualquier tipo de presión homeostática."""
        logging.info(f"Evento de adaptación recibido: {event.topic} con severidad {event.severity}. La lógica de adaptación ahora está en el bucle principal.")
        # La lógica de adaptación ahora es proactiva en el _main_loop.
        # Este handler podría usarse en el futuro para acciones de emergencia
        # que no forman parte del ciclo normal de adaptación (ej. guardado forzoso).
        pass

    # ─────────────────── ACCIONES HOMEOSTÁTICAS (OBSOLETAS) ───────────────────
    # Las siguientes funciones son ahora manejadas por el AutoMutator y sus sub-módulos.
    # Se mantienen aquí como referencia de la lógica anterior pero no serán llamadas.

    async def _trigger_pruning(self):
        logging.info("Poda simulada... (OBSOLETO)")
        # Lógica ahora en KatanaPruner

    async def _memory_offloading(self):
        logging.info("Off-loading a CPU… (OBSOLETO)")
        # Esta lógica podría ser re-introducida si es necesario

    async def _thermal_veto(self):
        logging.info("Veto térmico: pausa workers (OBSOLETO)")
        # Lógica de control de workers puede ser reimplementada si es necesario

    async def _inject_noise(self):
        logging.info("Inyectando ruido latente… (OBSOLETO)")
        # Lógica ahora puede ser parte de una estrategia de exploración

    async def _reduce_complexity(self):
        logging.info("Reduciendo complejidad de transición… (OBSOLETO)")
        # Lógica ahora en KatanaPruner/NeurogenesisManager

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