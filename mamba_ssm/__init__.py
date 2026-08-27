"""Puente de vendor hacia `engines/mamba_vendor/mamba_ssm` — NO BORRAR.

A diferencia de los shims de forwarding eliminados en la reorg 2026-07-01,
este paquete extiende `__path__` para que el nombre top-level `mamba_ssm`
(que el paquete vendorizado importa internamente en absoluto) siga resolviendo.

IMPORTS PEREZOSOS (PEP 562) — por qué
=====================================
Este shim re-exportaba CUATRO símbolos de forma ANSIOSA, copiados del ``__init__`` del
vendor.  Ese ``__init__`` está escrito para usar Mamba **como modelo de lenguaje**.  RNFE lo
usa como **librería de capas** (H-Net importa ``Mamba2`` y el kernel SSD).  Son dos casos de
uso distintos, y el ansioso hacía IMPOSIBLE el nuestro:

  ``selective_scan_fn`` / ``mamba_inner_fn`` / ``Mamba``  → Mamba-**1** ⇒ ``import
      selective_scan_cuda``, una extensión CUDA que exige **nvcc** (CUDA toolkit).  RNFE no
      lo tiene instalado y NO LO NECESITA.
  ``MambaLMHeadModel``  → el LM ⇒ ``transformers.generation`` con símbolos
      (``GreedySearchDecoderOnlyOutput``) **eliminados** en transformers >= 5.

Resultado: ``import mamba_ssm`` reventaba, y con él **todo Mamba-2** — cuyo kernel SSD
(``mamba_ssm/ops/triton/ssd_combined.py``) es **Triton PURO**: compila en runtime, NO
necesita nvcc, y está VERIFICADO corriendo en Turing (sm_75, RTX 2070) en fp16/fp32/bf16.

O sea: no fallaba Mamba-2.  Fallaba **importar el Mamba-1 que no usamos**.

Con ``__getattr__`` perezoso:
  - ``import mamba_ssm`` sólo extiende ``__path__``.  No importa nada pesado.
  - ``from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined`` → funciona.
  - ``from mamba_ssm.modules.mamba2 import Mamba2`` → funciona.
  - ``mamba_ssm.selective_scan_fn`` → sólo falla **si alguien realmente lo pide**, con un
    mensaje que dice qué falta.  La ausencia de nvcc deja de ser un fallo GLOBAL.

Los cuatro nombres siguen accesibles: el contrato público NO cambia.
"""

from pathlib import Path
from typing import Any

_engine_path = Path(__file__).resolve().parents[1] / "engines" / "mamba_vendor" / "mamba_ssm"
if str(_engine_path) not in __path__:
    __path__.append(str(_engine_path))


# nombre → (submódulo que lo define, qué necesita de verdad).
# El requisito se usa SOLO para el mensaje de error: nada acá chequea disponibilidad por
# adelantado.  Adivinar qué está disponible es exactamente lo que rompía antes.
_LAZY: dict[str, tuple[str, str]] = {
    "selective_scan_fn": (
        "mamba_ssm.ops.selective_scan_interface",
        "la extensión CUDA `selective_scan_cuda` (Mamba-1; requiere nvcc)",
    ),
    "mamba_inner_fn": (
        "mamba_ssm.ops.selective_scan_interface",
        "la extensión CUDA `selective_scan_cuda` (Mamba-1; requiere nvcc)",
    ),
    "Mamba": (
        "mamba_ssm.modules.mamba_simple",
        "la extensión CUDA `selective_scan_cuda` (Mamba-1; requiere nvcc)",
    ),
    "Mamba2": (
        "mamba_ssm.modules.mamba2",
        "torch + triton (Mamba-2 es Triton puro: NO requiere nvcc)",
    ),
    "MambaLMHeadModel": (
        "mamba_ssm.models.mixer_seq_simple",
        "`transformers` con la API de generación PRE-5.x (`GreedySearchDecoderOnlyOutput`)",
    ),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    """PEP 562: resolver el símbolo cuando lo piden, no al importar el paquete."""
    if name not in _LAZY:
        raise AttributeError(f"module 'mamba_ssm' has no attribute {name!r}")
    module_path, requisito = _LAZY[name]
    from importlib import import_module

    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"`mamba_ssm.{name}` no está disponible en este entorno: importar "
            f"`{module_path}` falló ({exc}). Necesita {requisito}.\n"
            f"Si lo que buscabas era Mamba-2, NO hace falta esto: el kernel SSD vive en "
            f"`mamba_ssm.ops.triton.ssd_combined` (Triton puro, corre en Turing sin nvcc)."
        ) from exc

    value = getattr(module, name)
    globals()[name] = value  # cachear: el segundo acceso no re-importa
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
