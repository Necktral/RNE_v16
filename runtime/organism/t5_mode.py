"""Configuracion de modo RNFE-T5.

RNFE_T5_MODE:
- off: usa pipeline legacy
- shadow: ejecuta T5 en paralelo sin gatear decisiones legacy
- on: T5 es fuente primaria de decision

Compatibilidad:
- si RNFE_T5_MODE no esta definido, se lee RNFE_T4_MODE.
"""

from __future__ import annotations

import os
from typing import Literal


T5Mode = Literal["off", "shadow", "on"]


def get_t5_mode() -> T5Mode:
    raw = os.environ.get("RNFE_T5_MODE")
    if raw is None:
        raw = os.environ.get("RNFE_T4_MODE", "on")
    normalized = str(raw).strip().lower()
    if normalized in {"off", "shadow", "on"}:
        return normalized  # type: ignore[return-value]
    return "on"


def is_t5_enabled() -> bool:
    return get_t5_mode() in {"shadow", "on"}


def is_t5_primary() -> bool:
    return get_t5_mode() == "on"

