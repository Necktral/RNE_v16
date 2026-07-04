"""Puente de vendor hacia `engines/mamba_vendor/mamba_ssm` — NO BORRAR.

A diferencia de los shims de forwarding eliminados en la reorg 2026-07-01,
este paquete extiende `__path__` para que el nombre top-level `mamba_ssm`
(que el paquete vendorizado importa internamente en absoluto) siga resolviendo.
"""

from pathlib import Path

_engine_path = Path(__file__).resolve().parents[1] / "engines" / "mamba_vendor" / "mamba_ssm"
if str(_engine_path) not in __path__:
    __path__.append(str(_engine_path))

from .ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn  # noqa: E402,F401
from .modules.mamba_simple import Mamba  # noqa: E402,F401
from .modules.mamba2 import Mamba2  # noqa: E402,F401
from .models.mixer_seq_simple import MambaLMHeadModel  # noqa: E402,F401

