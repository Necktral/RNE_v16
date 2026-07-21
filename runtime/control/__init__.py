"""Subsistema de control del runtime.

**MSRC** (Multi-Scale Resolution Controller) es el subsistema **vivo** del organismo
y se importa de forma *eager*.

`AdaptationController` y `CrisisRouter` son soporte del orquestador **legacy**
(AEON FENIX-Δ, en cuarentena) y se exponen de forma **perezosa** vía ``__getattr__``
para que `import runtime.control` (el camino vivo de MSRC) **no** arrastre la cadena
legacy ``crisis_router → infrastructure → pydantic``. Siguen siendo accesibles con
``from runtime.control import CrisisRouter`` (resuelto al vuelo) o por import directo
del submódulo. Ver ``docs/analysis/LEGACY_QUARANTINE.md``.
"""

from .msrc import (
    CrossScaleMemoryGuard,
    FixedVRAMSampler,
    MSRCController,
    NvidiaVRAMSampler,
    NullVRAMSampler,
    ScaleAuditLogger,
    ScaleCatalog,
    ScaleEstimator,
    ScalePolicyEngine,
    ScalePolicyState,
    RegimeClassifier,
    RegimeClassification,
    ScaleTransitionManager,
)

# Solo lo VIVO en __all__ → `from runtime.control import *` no carga el legacy.
__all__ = [
    "MSRCController",
    "ScaleCatalog",
    "ScaleEstimator",
    "ScalePolicyEngine",
    "ScalePolicyState",
    "RegimeClassifier",
    "RegimeClassification",
    "ScaleTransitionManager",
    "CrossScaleMemoryGuard",
    "ScaleAuditLogger",
    "FixedVRAMSampler",
    "NvidiaVRAMSampler",
    "NullVRAMSampler",
]

# Nombres legacy disponibles bajo demanda (cuarentena).
_LEGACY_EXPORTS = {"AdaptationController", "CrisisRouter"}


def __getattr__(name: str):
    """Carga perezosa del soporte legacy del orquestador FENIX-Δ (cuarentena).

    Evita acoplar el camino vivo a ``crisis_router → infrastructure → pydantic``.
    """
    if name == "AdaptationController":
        from .adaptation_controller import AdaptationController

        return AdaptationController
    if name == "CrisisRouter":
        from .crisis_router import CrisisRouter

        return CrisisRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*__all__, *_LEGACY_EXPORTS])
