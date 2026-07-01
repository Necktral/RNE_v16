"""Puente de vendor hacia `engines/hnet` — NO BORRAR.

A diferencia de los shims de forwarding eliminados en la reorg 2026-07-01,
este paquete extiende `__path__` para que el nombre top-level `hnet` (que el
código del engine espera) resuelva a `engines/hnet`.
"""

from pathlib import Path

_engine_path = Path(__file__).resolve().parents[1] / "engines" / "hnet"
if str(_engine_path) not in __path__:
    __path__.append(str(_engine_path))

from .models.mixer_seq import HNetForCausalLM  # noqa: E402,F401
from .modules import *  # noqa: E402,F401,F403
from .modules.utils import *  # noqa: E402,F401,F403

