"""Adapter estable para la familia DED.

DED v1 soporta solo un corte pequeno y auditable de LOT-F booleano:
- proposiciones atomicas
- NOT / AND / OR / ->
- parentesis

La integracion con Z3 queda encapsulada dentro de ``runtime.reasoning.families.ded``.
"""

from __future__ import annotations

from .engine import run_ded_engine


FAMILY_ID = "DED"


def execute(state):
    return run_ded_engine(state)
