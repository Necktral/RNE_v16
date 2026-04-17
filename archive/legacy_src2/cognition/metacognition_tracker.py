# metacognition_tracker.py — AEON ∆ v1.0
# Registro de desafíos cognitivos superados y activación de autoconciencia estructural
# Seguimiento metacognitivo: eventos, métricas EWMA, severidad, persistencia y atomicidad

import time
from typing import List, Dict, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
import logging
import os
import json
from datetime import datetime
import threading

@dataclass(frozen=True, slots=True)
class MetaEvent:
    timestamp: float  # Marca de tiempo del evento
    success: bool     # Indica si el desafío fue superado
    loss_before: Optional[float] = None  # Pérdida antes del evento
    loss_after: Optional[float] = None   # Pérdida después del evento

@dataclass(slots=True)
class Severity:
    delta_loss: float  # Diferencia de pérdida
    duration: int      # Duración del desafío

class MetacognitionTracker:
    _instance = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, history_size=1000, ewma_alpha=0.1, prometheus_registry=None, autoload=True, save_dir=None):
        self._history: Deque[MetaEvent] = deque(maxlen=history_size)
        self._ewma_success: float = 1.0
        self._severity_score: float = 0.0
        self._lock = threading.RLock()
        self.save_every: int = 100
        self.max_files: int = 10
        self.save_dir: str = save_dir or os.path.join(os.path.dirname(__file__), '../../logs/meta')
        os.makedirs(self.save_dir, exist_ok=True)
        self._ewma_alpha = ewma_alpha
        self._logger = logging.getLogger("aeon.metacognition")
        self._event_count = 0
        # Instrumentación Prometheus (opcional)
        # --- Singleton Prometheus: solo registrar métricas si no existen ---
        try:
            from prometheus_client import Counter, Gauge, REGISTRY
            kwargs_prom = {'registry': prometheus_registry} if prometheus_registry is not None else {}
            # Detectar si ya existen métricas con el mismo nombre
            metric_names = set(m.name for m in REGISTRY.collect())
            if 'aeon_metacog_event_total' not in metric_names:
                self.prom_event_count = Counter('aeon_metacog_event_total', 'Total de eventos metacognitivos', **kwargs_prom)
            else:
                self.prom_event_count = None
            if 'aeon_metacog_success_total' not in metric_names:
                self.prom_success_count = Counter('aeon_metacog_success_total', 'Total de eventos exitosos', **kwargs_prom)
            else:
                self.prom_success_count = None
            if 'aeon_metacog_severity' not in metric_names:
                self.prom_severity_gauge = Gauge('aeon_metacog_severity', 'Severidad EWMA actual', **kwargs_prom)
            else:
                self.prom_severity_gauge = None
        except ImportError:
            self.prom_event_count = None
            self.prom_success_count = None
            self.prom_severity_gauge = None
        # Recarga automática del último historial si existe
        if autoload:
            self._try_autoload()

    def clear(self):
        with self._lock:
            self._history.clear()
            self._event_count = 0
            self._ewma_success = 1.0
            self._severity_score = 0.0

    def _try_autoload(self):
        try:
            files = [f for f in os.listdir(self.save_dir) if f.startswith("meta_") and f.endswith(".jsonl")]
            if files:
                files.sort(reverse=True)
                self.load(os.path.join(self.save_dir, files[0]))
        except Exception:
            pass

    def register_event(self, success: bool, loss_before: Optional[float]=None, loss_after: Optional[float]=None, duration: int=1):
        """
        Registra un evento metacognitivo y actualiza las métricas EWMA y severidad.
        Instrumenta métricas Prometheus y logs estructurados para Loki.
        """
        with self._lock:
            event = MetaEvent(
                timestamp=time.time(),
                success=success,
                loss_before=loss_before,
                loss_after=loss_after
            )
            self._history.append(event)
            self._event_count += 1
            # EWMA de éxito
            self._ewma_success = self._ewma_alpha * (1.0 if success else 0.0) + (1 - self._ewma_alpha) * self._ewma_success
            # Score de severidad
            delta_loss = (loss_after - loss_before) if (loss_before is not None and loss_after is not None) else 0.0
            sev = self._compute_severity(delta_loss, duration)
            self._severity_score = self._ewma_alpha * sev + (1 - self._ewma_alpha) * self._severity_score
            # Prometheus
            if self.prom_event_count:
                self.prom_event_count.inc()
            if self.prom_success_count and success:
                self.prom_success_count.inc()
            if self.prom_severity_gauge:
                self.prom_severity_gauge.set(self._severity_score)
            # Log estructurado para Loki
            log_event = {
                "timestamp": event.timestamp,
                "success": event.success,
                "loss_before": event.loss_before,
                "loss_after": event.loss_after,
                "ewma_success": self._ewma_success,
                "severity": self._severity_score,
                "tipo": "metacognition_event"
            }
            self._logger.info(json.dumps(log_event, ensure_ascii=False))
            # Guardado periódico automático
            if self._event_count % self.save_every == 0:
                self.save()

    def _compute_severity(self, delta_loss: float, duration: int) -> float:
        # Normaliza delta_loss con sigmoide y duration con tangente hiperbólica
        import math
        norm_loss = 1 / (1 + math.exp(-delta_loss))
        norm_duration = math.tanh(duration / 10)
        return 0.5 * (norm_loss + norm_duration)
    def get_snapshot(self):
        """
        Devuelve un resumen rápido de las métricas actuales.
        """
        with self._lock:
            return {
                "ewma_success": round(self._ewma_success, 4),
                "severity_score": round(self._severity_score, 4),
                "total_events": len(self._history)
            }
    def save(self, path=None, snapshot=True):
        """
        Guarda el historial en formato JSONL. Si 'path' es None, genera una ruta con timestamp en logs/meta.
        Escribe en archivo temporal y reemplaza de forma atómica.
        Añade un snapshot agregado al final.
        Rota archivos viejos si excede 'max_files'.
        """
        with self._lock:
            if path is None:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                base = f"meta_{ts}"
                path = os.path.join(self.save_dir, f"{base}.jsonl")
                # Si el archivo existe, añade sufijo incremental
                suffix = 1
                while os.path.exists(path):
                    path = os.path.join(self.save_dir, f"{base}_{suffix}.jsonl")
                    suffix += 1
            tmp_path = path + ".tmp"
            from dataclasses import asdict
            with open(tmp_path, "w", encoding="utf-8") as f:
                for event in self._history:
                    f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                if snapshot:
                    snap = self.summary()
                    snap["snapshot"] = True
                    f.write(json.dumps(snap, ensure_ascii=False) + "\n")
            os.replace(tmp_path, path)
            self._rotate_files()

    def load(self, path):
        """
        Recarga el historial desde un archivo JSONL.
        """
        with self._lock:
            self._history.clear()
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj.get("snapshot"):
                        continue
                    event = MetaEvent(
                        timestamp=obj["timestamp"],
                        success=obj["success"],
                        loss_before=obj.get("loss_before"),
                        loss_after=obj.get("loss_after")
                    )
                    self._history.append(event)

    def _rotate_files(self):
        """
        Mantiene sólo los últimos 'max_files' en 'save_dir', elimina los más antiguos.
        """
        with self._lock:
            files = [f for f in os.listdir(self.save_dir) if f.startswith("meta_") and f.endswith(".jsonl")]
            files.sort(reverse=True)
            for old in files[self.max_files:]:
                try:
                    os.remove(os.path.join(self.save_dir, old))
                except Exception:
                    pass

    def summary(self):
        """
        Devuelve un resumen estadístico del historial de eventos.
        """
        with self._lock:
            total = len(self._history)
            successes = sum(1 for e in self._history if e.success)
            failures = total - successes
            success_rate = (successes / total) if total else 0.0
            return {
                "total_challenges": total,
                "success_rate": round(success_rate, 3),
                "ewma_success": round(self._ewma_success, 4),
                "severity_score": round(self._severity_score, 4),
                "successes": successes,
                "failures": failures
            }

    def requires_intervention(self, ewma_threshold=0.3, severity_threshold=0.7):
        """
        Retorna True si el EWMA de éxito es bajo o la severidad es alta.
        """
        snap = self.get_snapshot()
        return snap["ewma_success"] < ewma_threshold or snap["severity_score"] > severity_threshold

    def log_challenge(self, challenge_type, cycle, outcome):
        """
        Registra un desafío cognitivo/metacognitivo en el historial y lo expone para dashboards.
        """
        with self._lock:
            event = {
                "timestamp": time.time(),
                "challenge_type": challenge_type,
                "cycle": cycle,
                "outcome": outcome,
                "ewma_success": self._ewma_success,
                "severity": self._severity_score
            }
            # Log estructurado para dashboards
            self._logger.info(json.dumps(event, ensure_ascii=False))
            # Opcional: guardar en disco cada vez
            if hasattr(self, 'save_every') and self._event_count % self.save_every == 0:
                self.save()

if __name__ == "__main__":
    tracker = MetacognitionTracker()
    tracker.register_event(success=True, loss_before=1.2, loss_after=0.9)
    tracker.register_event(success=False, loss_before=0.9, loss_after=1.5)
    tracker.register_event(success=True, loss_before=1.5, loss_after=1.1)

    print(tracker.summary())
    if tracker.requires_intervention():
        print("[META] ⚠️ Nivel de éxito bajo — activar asistencia cognitiva.")
    # Ejemplo de guardado manual
    tracker.save()
    # Ejemplo de recarga
    # tracker.load("ruta/a/archivo.jsonl")
