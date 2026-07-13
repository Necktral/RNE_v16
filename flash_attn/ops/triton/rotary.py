"""``flash_attn.ops.triton.rotary.apply_rotary`` en torch puro. NO hay Triton acá.

Consumidor en el repo: ``engines/hnet/modules/rotary.py:7`` (que es, a su vez, una copia del
``flash_attn/layers/rotary.py`` upstream).  Ese archivo llama a ``apply_rotary`` con:

  * ``inplace=True``  sobre **vistas** de ``qkv`` (``qkv[:, :, :2].reshape(...)``)  ⇒ la
    escritura tiene que propagarse al tensor padre.  Acá se hace con asignación indexada,
    que sí propaga por la vista.
  * ``conjugate=True`` en el backward (rotación inversa).
  * ``cu_seqlens`` / ``max_seqlen`` en modo empaquetado.
  * ``seqlen_offsets`` int o tensor ``(batch,)``.

Matemática (idéntica al kernel upstream, computada en fp32):

    rotary_dim = 2 · cos.shape[-1];  sólo se rota ``x[..., :rotary_dim]``.
    no-interleaved (GPT-NeoX):  x1 = x[..., :rd/2],  x2 = x[..., rd/2:rd]
    interleaved   (GPT-J):      x1 = x[..., 0::2],   x2 = x[..., 1::2]
    o1 = x1·cos − x2·sin
    o2 = x1·sin + x2·cos          (``conjugate`` ⇒ sin ↦ −sin)
"""

from __future__ import annotations

from typing import Optional, Union

import torch

__all__ = ["apply_rotary", "apply_rotary_positions"]


def _split(x: torch.Tensor, rotary_dim: int, interleaved: bool):
    if interleaved:
        return x[..., :rotary_dim:2], x[..., 1:rotary_dim:2]
    half = rotary_dim // 2
    return x[..., :half], x[..., half:rotary_dim]


def _merge(o1: torch.Tensor, o2: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        return torch.stack((o1, o2), dim=-1).flatten(start_dim=-2)
    return torch.cat((o1, o2), dim=-1)


def _rotate(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    interleaved: bool,
    conjugate: bool,
) -> torch.Tensor:
    """``x`` (..., D) con ``cos``/``sin`` broadcasteables a (..., rotary_dim/2)."""
    x1, x2 = _split(x, rotary_dim, interleaved)
    x1f, x2f = x1.float(), x2.float()
    cosf, sinf = cos.float(), sin.float()
    if conjugate:
        sinf = -sinf
    o1 = x1f * cosf - x2f * sinf
    o2 = x1f * sinf + x2f * cosf
    return _merge(o1, o2, interleaved).to(x.dtype)


def _positions_from_cu_seqlens(
    cu_seqlens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """(total,) con la posición de cada token DENTRO de su secuencia."""
    bounds = cu_seqlens.to(device=device, dtype=torch.long)
    lengths = bounds[1:] - bounds[:-1]
    total = int(bounds[-1])
    starts = torch.repeat_interleave(bounds[:-1], lengths)
    return torch.arange(total, device=device) - starts


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Misma firma que ``flash_attn.ops.triton.rotary.apply_rotary``.

    Arguments:
        x: ``(batch, seqlen, nheads, headdim)`` si ``cu_seqlens is None``,
           si no ``(total_seqlen, nheads, headdim)``.
        cos, sin: ``(seqlen_ro, rotary_dim / 2)``.
        seqlen_offsets: int o tensor ``(batch,)``.
        cu_seqlens: ``(batch + 1,)`` o None.
        max_seqlen: int (sólo se valida; no cambia el resultado).
    Returns:
        ``x`` rotado, misma forma.  Con ``inplace=True`` devuelve el MISMO tensor, mutado.
    """
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(
            f"rotary_dim={rotary_dim} no entra en headdim={x.shape[-1]}"
        )
    if cos.shape != sin.shape:
        raise ValueError(f"cos {tuple(cos.shape)} y sin {tuple(sin.shape)} deben coincidir")

    device = x.device

    if cu_seqlens is None:
        if x.dim() != 4:
            raise ValueError(
                f"x debe ser (batch, seqlen, nheads, headdim); recibí {tuple(x.shape)}"
            )
        batch, seqlen = x.shape[0], x.shape[1]
        base = torch.arange(seqlen, device=device)
        if isinstance(seqlen_offsets, int):
            positions = (base + seqlen_offsets).unsqueeze(0).expand(batch, seqlen)
        else:
            offsets = seqlen_offsets.to(device=device, dtype=torch.long).reshape(-1, 1)
            positions = base.unsqueeze(0) + offsets  # (batch, seqlen)
        # (batch, seqlen, 1, rotary_dim/2) → broadcast sobre nheads
        cos_sel = cos[positions].unsqueeze(2)
        sin_sel = sin[positions].unsqueeze(2)
        rotated = _rotate(x, cos_sel, sin_sel, rotary_dim, interleaved, conjugate)
        if inplace:
            x[..., :rotary_dim] = rotated
            return x
        if rotary_dim == x.shape[-1]:
            return rotated
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    # ---- modo empaquetado (varlen) ----
    if x.dim() != 3:
        raise ValueError(
            f"con cu_seqlens, x debe ser (total_seqlen, nheads, headdim); "
            f"recibí {tuple(x.shape)}"
        )
    if max_seqlen is None:
        raise ValueError("con cu_seqlens hay que pasar max_seqlen (contrato de FlashAttention)")

    positions = _positions_from_cu_seqlens(cu_seqlens, device)  # (total,)
    total = positions.shape[0]
    if total > x.shape[0]:
        raise ValueError(
            f"cu_seqlens cubre {total} tokens pero x tiene {x.shape[0]}"
        )
    if not isinstance(seqlen_offsets, int) or seqlen_offsets != 0:
        raise NotImplementedError(
            "apply_rotary(cu_seqlens=..., seqlen_offsets!=0) no está implementado en la "
            "capa de compatibilidad torch de RNFE: FlashAttention lo usa sólo para "
            "generación con KV-cache empaquetada, que este repo no ejercita. "
            "No lo fabrico: implementalo con equivalencia exacta si lo necesitás."
        )

    view = x[:total]
    cos_sel = cos[positions].unsqueeze(1)  # (total, 1, rotary_dim/2)
    sin_sel = sin[positions].unsqueeze(1)
    rotated = _rotate(view, cos_sel, sin_sel, rotary_dim, interleaved, conjugate)
    if inplace:
        x[:total, ..., :rotary_dim] = rotated
        return x
    out = x.clone()
    out[:total, ..., :rotary_dim] = rotated
    return out


def apply_rotary_positions(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    interleaved: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Variante con posiciones explícitas por token — la que necesita la KV-cache.

    x: ``(batch, seqlen, nheads, headdim)``; positions: ``(batch, seqlen)`` (índices en cos/sin).
    Fuera de la API pública de FlashAttention; es un helper interno de este shim.
    """
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary_dim={rotary_dim} no entra en headdim={x.shape[-1]}")
    pos = positions.to(device=x.device, dtype=torch.long)
    cos_sel = cos[pos].unsqueeze(2)
    sin_sel = sin[pos].unsqueeze(2)
    rotated = _rotate(x, cos_sel, sin_sel, rotary_dim, interleaved, conjugate)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
