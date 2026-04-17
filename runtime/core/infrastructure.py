# src/core/infrastructure.py

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import yaml
from pydantic import BaseModel, ValidationError
import threading

# ────────────────────────────  EVENTOS Y MÉTRICAS  ────────────────────────────
@dataclass(frozen=True, slots=True)
class Event:
    topic: str
    payload: Any = None
    severity: str = "INFO"
    timestamp: float = field(default_factory=time.time)
    source: str = "system"

# ────────────────────────────  COLAS DE TRABAJO  ────────────────────────────
class WorkerPool:
    def __init__(self, num_workers: int = 4):
        self.max_workers = num_workers
        self.queues = {
            "CRITICAL": deque(), "HIGH": deque(),
            "MEDIUM": deque(), "LOW": deque()
        }
        self.workers = [asyncio.create_task(self._worker(f"worker-{i}")) for i in range(num_workers)]
        self.lock = asyncio.Lock()

    async def _worker(self, name: str):
        while True:
            async with self.lock:
                for qtype in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    if self.queues[qtype]:
                        func, desc = self.queues[qtype].popleft()
                        logging.info(f"{name} ejecutando {qtype}: {desc}")
                        asyncio.create_task(self._execute(func, desc))
                        break
                else:
                    await asyncio.sleep(0.01)

    async def _execute(self, func: Callable, desc: str):
        try:
            await func()
        except Exception as e:
            logging.error(f"Error en tarea '{desc}': {e}")

    def submit(self, qtype: str, task: Callable, desc: str = "sin descripción"):
        if qtype not in self.queues:
            raise ValueError(f"Cola desconocida: {qtype}")
        self.queues[qtype].append((task, desc))

    def close(self):
        # No hay threads explícitos, pero limpiamos workers y colas
        for w in self.workers:
            w.cancel()
        for q in self.queues.values():
            q.clear()

# ────────────────────────────  BUS DE EVENTOS  ──────────────────────────────
class EventBus:
    _SENTINEL = object()
    def __init__(self, capacity: int = 4096):
        self._queues = {s: asyncio.Queue(capacity) for s in ["CRITICAL", "WARN", "INFO"]}
        self._subs = defaultdict(set)
        self._running = True
        self._router_task = None

    async def publish(self, event: Event):
        await self._queues[event.severity].put(event)

    def subscribe(self, topic: str, handler: Callable[[Event], Awaitable[None]]):
        self._subs[topic].add(handler)

    async def _router(self):
        while self._running:
            for sev in ["CRITICAL", "WARN", "INFO"]:
                if not self._queues[sev].empty():
                    event = await self._queues[sev].get()
                    if event is self._SENTINEL:
                        self._running = False
                        break
                    for h in self._subs.get(event.topic, ()):
                        asyncio.create_task(h(event))
                    break
            else:
                await asyncio.sleep(0.001)

    async def start(self):
        if not self._router_task or self._router_task.done():
            self._router_task = asyncio.create_task(self._router())

    async def shutdown(self):
        self._running = False
        # Desbloquea cualquier get pendiente
        for q in self._queues.values():
            await q.put(self._SENTINEL)
        if self._router_task:
            await self._router_task
        self._subs.clear()
        # Vacía las colas restantes
        for q in self._queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    pass

# ────────────────────────────  CARGA DE CONFIGURACIÓN  ────────────────────────────
class ModelConfig(BaseModel):
    latent_dim: int
    memory_capacity: int

class ConfigLoader:
    _instance = None
    _lock = threading.Lock()
    config_path = 'config/config.yaml'

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._load()
        return cls._instance

    def _load(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        try:
            self.model = ModelConfig(**raw.get('model', {}))
        except ValidationError as e:
            raise RuntimeError(f"Error de validación en config.yaml: {e}")

    @property
    def model_cfg(self):
        return self.model
