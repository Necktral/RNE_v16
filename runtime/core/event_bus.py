"""
event_bus.py – EventBus centralizado para AEON FENIX-Δ
-----------------------------------------------------
• Permite emitir, escuchar y responder a eventos críticos entre módulos
• Soporta callbacks, logging y trazabilidad
"""

import threading
from collections import defaultdict
from typing import Callable, Dict, List, Any
import json
import os
from datetime import datetime

from runtime.storage import get_storage

EVENT_LOG_PATH = os.environ.get("AEON_EVENT_LOG", "aeon_event_log.jsonl")

class EventBus:
    def __init__(self):
        self._listeners: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def emit(self, event_type: str, payload: Any = None):
        """Emite un evento a todos los listeners registrados y lo persiste en archivo y SQLite"""
        # Persistencia a archivo
        try:
            with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
                json.dump({
                    "event": event_type,
                    "payload": payload,
                    "timestamp": datetime.utcnow().isoformat()
                }, f)
                f.write("\n")
        except Exception as e:
            print(f"[EventBus] Error al escribir evento en log: {e}")
        # Persistencia en capa de storage (sqlite/postgres/hybrid)
        try:
            storage = get_storage()
            storage.append_event(
                event_type=event_type,
                payload=payload if isinstance(payload, dict) else {"value": payload},
                timestamp=datetime.utcnow().isoformat(),
                source="event_bus",
            )
        except Exception as e:
            print(f"[EventBus] Error al persistir evento: {e}")
        # Emisión del evento
        with self._lock:
            listeners = list(self._listeners[event_type])
        for callback in listeners:
            try:
                callback(payload)
            except Exception as e:
                print(f"[EventBus] Error en callback de '{event_type}': {e}")

    def on(self, event_type: str, callback: Callable[[Any], None]):
        """Registra un callback para un tipo de evento"""
        with self._lock:
            self._listeners[event_type].append(callback)

    def off(self, event_type: str, callback: Callable[[Any], None]):
        """Elimina un callback de un tipo de evento"""
        with self._lock:
            if callback in self._listeners[event_type]:
                self._listeners[event_type].remove(callback)

# Instancia global (puede ser importada por todos los módulos)
event_bus = EventBus()
