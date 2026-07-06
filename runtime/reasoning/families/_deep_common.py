"""Utilidades compartidas para las variantes profundas de las familias overlay.

Sin dependencias del resto del sistema: helpers puros para leer el estado
acumulado del episodio y las features del scheduler de forma defensiva.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(x, lo), hi)


def features(state: Mapping[str, Any]) -> Dict[str, Any]:
    meta = state.get("_meta")
    feats = meta.get("features") if isinstance(meta, dict) else None
    return feats if isinstance(feats, dict) else {}
