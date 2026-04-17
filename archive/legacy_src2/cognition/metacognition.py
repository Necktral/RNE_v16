import json
from collections import deque
from typing import Deque

__all__ = [
    "load_events_from_jsonl",
    "compute_ewma",
    "snapshot_metrics"
]

def load_events_from_jsonl(path):
    """
    Carga eventos desde un archivo JSONL, ignorando líneas con 'snapshot'.
    """
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("snapshot"):
                continue
            events.append(obj)
    return events

def compute_ewma(events: list, alpha: float = 0.1) -> float:
    """
    Exponential-weighted moving average de éxitos (True = 1, False = 0).
    α=1 implica “sólo el último evento”; α→0 se acerca a media aritmética.
    """
    ewma = 1.0  # arranca optimista
    for flag in events:
        ewma = alpha * float(flag) + (1 - alpha) * ewma
    return ewma

def snapshot_metrics(history: Deque) -> dict:
    """
    Devuelve un resumen compacto del historial:
      - n_events
      - success_ratio
      - ewma_success
      - avg_severity
    """
    if not history:
        return dict(n_events=0, success_ratio=1.0, ewma_success=1.0, avg_severity=0.0)
    successes = [getattr(e, "success", e.get("success", False)) for e in history]
    severity  = [getattr(e, "severity", e.get("severity", None)) for e in history if getattr(e, "severity", e.get("severity", None)) is not None]
    return {
        "n_events":      len(successes),
        "success_ratio": sum(successes) / len(successes),
        "ewma_success":  compute_ewma(successes),
        "avg_severity":  (sum(severity) / len(severity)) if severity else 0.0,
    }