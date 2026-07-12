"""``flash_attn.ops.activations`` en torch puro. NO es FlashAttention (no hay fusión).

Único símbolo que ``engines/`` necesita: ``swiglu``, usado en
``engines/hnet/modules/mlp.py:31`` como ``swiglu(gate, y)`` con
``y, gate = self.fc1(x).chunk(2, dim=-1)``.

Definición (idéntica a la de FlashAttention, que computa el silu en fp32 y castea de vuelta):

    swiglu(x, y) = silu(x) * y = (x · sigmoid(x)) * y

El **orden de los argumentos importa**: el primer argumento es el que pasa por el silu.  En
``mlp.py`` el primero es ``gate`` ⇒ ``silu(gate) * y``.  Invertirlo daría otra red.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["swiglu", "swiglu_fwd"]


def swiglu_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``silu(x) * y``, computado en fp32 y devuelto en el dtype de ``x``.

    El upcast a fp32 replica lo que hace FlashAttention (``F.silu(x.float()) * y``): en bf16
    el silu pierde bastante precisión si se computa en el dtype nativo.
    """
    return (F.silu(x.float()) * y.float()).to(dtype=x.dtype)


# FlashAttention expone `swiglu` como una `torch.autograd.Function` fusionada.  Acá no hay
# fusión: la composición de ops de torch ya es diferenciable, y el gradiente es el mismo.
swiglu = swiglu_fwd
