"""Adaptador legacy de modo T4.

Mantiene compatibilidad con imports existentes delegando en T5.
"""

from __future__ import annotations

from typing import Literal

from .t5_mode import get_t5_mode, is_t5_enabled, is_t5_primary


T4Mode = Literal["off", "shadow", "on"]


def get_t4_mode() -> T4Mode:
    return get_t5_mode()  # type: ignore[return-value]


def is_t4_enabled() -> bool:
    return is_t5_enabled()


def is_t4_primary() -> bool:
    return is_t5_primary()

