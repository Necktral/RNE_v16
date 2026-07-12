"""ESTO NO ES FLASHATTENTION.

Es una **capa de compatibilidad en torch puro** que ocupa el nombre ``flash_attn`` para que
``engines/hnet`` sea importable y ejecutable en una máquina donde FlashAttention **no se
puede instalar** (no hay ``nvcc``; la GPU es una RTX 2070 Max-Q, Turing ``sm_75``, y
FlashAttention exige Ampere+).

Qué es y qué no es
==================
FlashAttention es una optimización de **memoria**, no una matemática distinta: hace tiling
de la atención en SRAM y **nunca materializa** la matriz ``QK^T`` completa ⇒ memoria
**O(L)** en la longitud de secuencia.  Este paquete calcula **la misma función matemática**
con ``torch.nn.functional.scaled_dot_product_attention`` y máscaras explícitas.

⚠ REGRESIÓN DE MEMORIA — MEDIDA, no supuesta
============================================
``scaled_dot_product_attention`` tiene varios backends, y **cuál te toca no es obvio**.  En
**Turing (sm_75)** con ``torch 2.13.0+cu130``, medido el 2026-07-12:

  * ``FLASH_ATTENTION``     → **NO existe**: "only supports gpu architectures in the range
    [sm80, sm121]".
  * ``EFFICIENT_ATTENTION`` (mem-efficient / cutlass) → **SÍ existe en Turing**, pero **sólo
    en fp16 y fp32**.  En **bf16 NO**: "Expected query, key and value to all be of dtype:
    {Half, Float}. Got BFloat16".
  * ``MATH``                → siempre disponible, y **materializa la matriz L×L** ⇒ **O(L²)**.

Pico de VRAM MEDIDO (B=1, H=16, D=96 — las dims reales de la red ``T22`` del
``hnet_1stage_L``; GPU con 7.6 GiB útiles).  Ver
``tests/engines/test_flash_attn_compat.py::test_LA_REGRESION_DE_MEMORIA_es_real_y_depende_del_dtype``::

    L        fp16 causal    fp16 ventana    bf16 causal
             (is_causal)     (máscara)      (is_causal)
     1024       3.0 MB         10.0 MB        181.1 MB
     2048       6.0 MB         40.0 MB        640.0 MB
     4096      12.0 MB        160.1 MB       2464.1 MB
     8192      24.0 MB        640.2 MB            OOM
    16384      72.0 MB       2560.4 MB            OOM
    32768     144.0 MB            OOM             OOM
    65536     288.0 MB            OOM             OOM

Léase:

* **fp16 + causal denso** (``causal=True``, ``Sq == Sk``, sin ventana) ⇒ usamos
  ``is_causal=True`` ⇒ backend mem-efficient ⇒ **LINEAL**.  64 K tokens en 288 MB.
  **Ahí NO hay regresión ninguna.**
* **Cualquier máscara explícita** (ventana deslizante, ``varlen_*``, ``with_kvcache``) ⇒
  máscara (Sq × Sk) materializada ⇒ **CUADRÁTICO**.  Empieza a doler cerca de **L ≈ 16 K** y
  revienta a 32 K.
* **bf16, SIEMPRE** ⇒ el mem-efficient no lo acepta en Turing ⇒ cae a ``MATH`` ⇒ **cuadrático
  y peor**: OOM ya a **L = 8192**.  Es la trampa más fea: bf16 es el dtype "obvio" y acá te
  cuesta ~100× la memoria, **en silencio**.

⚠ **En Turing, para atención larga: fp16, NO bf16.**  Este paquete emite un ``UserWarning``
(una vez por motivo) cuando está por tomar un camino cuadrático caro, para que la regresión
no sea silenciosa.

Además ``varlen_*`` y ``with_kvcache`` iteran la batch en un **bucle de Python** (una llamada
de atención por secuencia).  Es exacto; no es rápido.

PARA QUÉ SIRVE
--------------
* Importar ``hnet`` sin FlashAttention.
* Correr el **encoder** de H-Net (4 capas Mamba-2 en ``hnet_1stage_*``) y el
  ``RoutingModule``: el camino que RNFE necesita (``N5`` ⇒ una probabilidad de frontera por
  byte).  Ese camino **no usa atención en absoluto**: sólo ``RMSNorm`` ⇒ **la regresión de
  memoria ni lo roza**.
* Correr la red Transformer interna (``T22``) — en fp16 y causal global (que es lo que pide
  el ``hnet_1stage_L``: ``window_size: [1023, -1]``, y la etapa de la ``T22`` usa el ``-1``).

PARA QUÉ **NO** SIRVE
---------------------
* Entrenar Transformers de contexto largo **en bf16** en 8 GB: OOM a 8 K.
* Atención con **ventana deslizante** a contexto largo: es cuadrática.
* Sustituir a FlashAttention en **velocidad**.  No lo hace y no lo pretende.

REGLAS QUE ESTE PAQUETE RESPETA
===============================
1. **Defiere al real.**  Si hay un ``flash_attn`` de verdad instalado (fuera de la raíz del
   repo), este módulo **se aparta** y deja pasar al real (ver ``_find_installed_flash_attn``).
   No lo tapa.
2. **No fabrica.**  Todo símbolo o bien está implementado con equivalencia matemática, o
   bien tira ``NotImplementedError`` diciendo qué falta.  **Nunca** devuelve ceros, un
   ``pass``, ni una constante plausible.  Sin defaults benignos.
3. **Dice qué es.**  Este docstring.  Y ``__version__`` es ``0.0.0+...``: cualquier chequeo
   de versión contra FlashAttention **va a fallar**, que es lo correcto.

Símbolos NO implementados a propósito (tiran ``NotImplementedError`` con motivo):
  * ``dropout_p > 0``      — el dropout de FA depende de su RNG por bloques; no es replicable.
  * ``alibi_slopes``       — no lo usa nadie en el repo.
  * ``softcap != 0``       — idem.
  * ``return_attn_probs``  — SDPA no expone las probabilidades.
  * ``block_table``        — paged KV cache.
  * ``GenerationMixin.generate`` — andamiaje de generación que RNFE no usa (ver
    ``flash_attn/utils/generation.py``).
"""

from __future__ import annotations

import os
import sys
from importlib.machinery import PathFinder

# Deliberadamente 0.0.0: si algún día alguien chequea `flash_attn.__version__ >= "2"`,
# tiene que fallar.  Mentir acá sería exactamente el bug que esta campaña persigue.
__version__ = "0.0.0+rnfe.torchcompat"

_SHIM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SHIM_DIR)


def _find_installed_flash_attn():
    """Buscar un ``flash_attn`` REAL, instalado fuera de la raíz del repo.

    Devuelve su ``ModuleSpec`` o ``None``.  La raíz del repo suele ir **antes** que
    ``site-packages`` en ``sys.path``, así que si FlashAttention llegara a instalarse este
    shim lo taparía sin querer.  Esta función existe para que eso no pase.
    """
    candidates: list[str] = []
    for entry in sys.path:
        resolved = os.path.abspath(entry or os.getcwd())
        if resolved == _REPO_ROOT:
            continue
        candidates.append(entry or os.getcwd())

    spec = PathFinder.find_spec("flash_attn", candidates)
    if spec is None or spec.origin is None:
        # `origin is None` ⇒ namespace package: no es una instalación real.
        return None
    origin = os.path.abspath(spec.origin)
    if origin == _SHIM_DIR or origin.startswith(_SHIM_DIR + os.sep):
        return None  # nos encontramos a nosotros mismos
    return spec


_real_spec = _find_installed_flash_attn()

if _real_spec is not None:
    # DEFERIR AL REAL.  Reemplazamos nuestra entrada en `sys.modules` por el módulo real:
    # `importlib._bootstrap._load` re-lee `sys.modules[name]` después de `exec_module`, así
    # que el que queda instalado es el real, con su propio `__path__` (⇒ sus submódulos
    # también resuelven al real).  Si el real está instalado pero roto, esto REVIENTA: no
    # lo enmascaramos.
    import importlib.util as _ilu

    _real_module = _ilu.module_from_spec(_real_spec)
    sys.modules[__name__] = _real_module
    _real_spec.loader.exec_module(_real_module)

else:
    from .flash_attn_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )

    IS_RNFE_TORCH_COMPAT_SHIM = True
    """Marcador explícito. Si esto es True, NO estás usando FlashAttention."""

    __all__ = [
        "IS_RNFE_TORCH_COMPAT_SHIM",
        "flash_attn_func",
        "flash_attn_kvpacked_func",
        "flash_attn_qkvpacked_func",
        "flash_attn_varlen_func",
        "flash_attn_varlen_kvpacked_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_with_kvcache",
    ]
