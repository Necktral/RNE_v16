"""Evaluación numérica segura para expresiones EML."""

from __future__ import annotations

import math
from typing import Any, Dict

from .tree import ExprNode


class DomainError(ValueError):
    """Error de dominio para expresiones no válidas."""


def _clip(value: float, *, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def safe_eval(
    expr: ExprNode,
    variables: Dict[str, Any],
    *,
    clip_abs: float = 1_000_000.0,
) -> float:
    if expr.op == "const":
        if expr.value is None:
            raise DomainError("const sin value")
        return _clip(float(expr.value), limit=clip_abs)
    if expr.op == "var":
        if not expr.name:
            raise DomainError("var sin name")
        raw = variables.get(expr.name, 0.0)
        if not isinstance(raw, (int, float)):
            raise DomainError(f"variable no numérica: {expr.name}")
        return _clip(float(raw), limit=clip_abs)

    left = safe_eval(expr.left, variables, clip_abs=clip_abs) if expr.left else 0.0
    right = safe_eval(expr.right, variables, clip_abs=clip_abs) if expr.right else 0.0

    if expr.op == "add":
        out = left + right
    elif expr.op == "sub":
        out = left - right
    elif expr.op == "mul":
        out = left * right
    elif expr.op == "div":
        if abs(right) < 1e-8:
            raise DomainError("división por cero")
        out = left / right
    elif expr.op == "pow2":
        out = left * left
    elif expr.op == "log1p":
        if left <= -1.0:
            raise DomainError("log1p fuera de dominio")
        out = math.log1p(left)
    elif expr.op == "exp":
        out = math.exp(_clip(left, limit=40.0))
    else:
        raise DomainError(f"operador no soportado: {expr.op}")

    if not math.isfinite(out):
        raise DomainError("resultado no finito")
    return _clip(out, limit=clip_abs)

