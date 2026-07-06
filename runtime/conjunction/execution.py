"""Mapeo tier de cómputo -> directivas de ejecución del episodio.

El router elige un tier; hasta ahora ese tier era decorativo (se persistía pero
nada ejecutaba distinto). Este módulo traduce el tier a directivas que el
LifeKernel aplica sobre la construcción del runner: perfil de cierre, límite de
recuperación de memoria y habilitación del razonador externo.

Todo el efecto está detrás de ``RNFE_CONJUNCTION_ROUTING_ENFORCED`` (off por
defecto); apagado, el kernel ignora estas directivas y la conducta nominal queda
byte-idéntica.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from .contracts import ComputeTier

_TRUE = {"1", "true", "yes", "on"}


def routing_enforced() -> bool:
    """True si el tier del router debe ejecutarse (no solo persistirse)."""
    return os.environ.get("RNFE_CONJUNCTION_ROUTING_ENFORCED", "").strip().lower() in _TRUE


@dataclass(frozen=True, slots=True)
class TierExecutionDirectives:
    """Parámetros de ejecución derivados de un tier de cómputo."""

    tier: ComputeTier
    closure_profile: str
    memory_retrieval_limit: int
    external_reasoner_enabled: bool
    gpu_backed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "closure_profile": self.closure_profile,
            "memory_retrieval_limit": self.memory_retrieval_limit,
            "external_reasoner_enabled": self.external_reasoner_enabled,
            "gpu_backed": self.gpu_backed,
        }


# Mapeo declarativo. Reutiliza los knobs que YA existen en el runner:
#   - closure_profile: baseline_fixed (núcleo) vs adaptive_min (overlays).
#   - memory_retrieval_limit: knob real del ScenarioEpisodeRunner (default 3).
#   - external_reasoner_enabled: habilita el perfil gated del razonador (tier_3).
# NUNCA introduce max_steps en run_episode (invariante ADR_MSRC): el presupuesto
# se sigue derivando de features dentro del scheduler.
_TIER_MAP: Dict[str, Dict[str, Any]] = {
    "tier_0_deterministic": {
        "closure_profile": "baseline_fixed",
        "memory_retrieval_limit": 1,
        "external_reasoner_enabled": False,
    },
    "tier_1_local_light": {
        "closure_profile": "baseline_fixed",
        "memory_retrieval_limit": 3,
        "external_reasoner_enabled": False,
    },
    "tier_2_specialized": {
        "closure_profile": "adaptive_min",
        "memory_retrieval_limit": 5,
        "external_reasoner_enabled": False,
    },
    "tier_3_external": {
        "closure_profile": "adaptive_min",
        "memory_retrieval_limit": 5,
        "external_reasoner_enabled": True,
    },
}


def tier_execution_directives(
    tier: str,
    *,
    gpu_backed: bool = False,
) -> TierExecutionDirectives:
    """Devuelve las directivas de ejecución para ``tier`` (default tier_1)."""
    spec = _TIER_MAP.get(str(tier), _TIER_MAP["tier_1_local_light"])
    return TierExecutionDirectives(
        tier=str(tier),  # type: ignore[arg-type]
        closure_profile=str(spec["closure_profile"]),
        memory_retrieval_limit=int(spec["memory_retrieval_limit"]),
        external_reasoner_enabled=bool(spec["external_reasoner_enabled"]),
        gpu_backed=bool(gpu_backed),
    )
