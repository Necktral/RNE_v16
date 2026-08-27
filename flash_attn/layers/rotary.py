"""``flash_attn.layers.rotary`` en torch puro. NO es FlashAttention.

Consumidor en el repo: ``engines/mamba_vendor/mamba_ssm/modules/mha.py:16`` (dentro de un
``try/except ImportError``).  H-Net **no** usa esta clase: tiene su propia copia en
``engines/hnet/modules/rotary.py``, que consume ``flash_attn.ops.triton.rotary.apply_rotary``
(implementado en este mismo paquete).

Divergencia declarada con el upstream (importante)
==================================================
Upstream, ``apply_rotary_emb_qkv_`` y ``apply_rotary_emb_kv_`` mutan ``qkv``/``kv``
**in-place** (por eso el ``_`` final).  Acá **devolvemos tensores nuevos** y NO mutamos la
entrada, para que el autograd de torch funcione sin envolver todo en ``autograd.Function``.

Es seguro para los dos únicos consumidores del repo, que usan el **valor de retorno**:
``engines/hnet/modules/mha.py:389`` (``qkv = self.rotary_emb(qkv, ...)``) y
``engines/mamba_vendor/mamba_ssm/modules/mha.py:276`` (``q, kv = self.rotary_emb(q, kv, ...)``).
Si algún día aparece un caller que dependa del **efecto** in-place y descarte el retorno, va a
romperse en silencio.  Queda escrito acá.

(``flash_attn.ops.triton.rotary.apply_rotary`` **sí** respeta ``inplace=True``, porque la copia
de rotary que trae H-Net lo necesita para escribir a través de una vista de ``qkv``.)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from ..ops.triton.rotary import apply_rotary

__all__ = [
    "RotaryEmbedding",
    "apply_rotary_emb",
    "apply_rotary_emb_func",
    "apply_rotary_emb_kv_",
    "apply_rotary_emb_qkv_",
]


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """x: ``(batch, seqlen, nheads, headdim)`` (o ``(total, nheads, headdim)`` con cu_seqlens)."""
    return apply_rotary(
        x,
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        interleaved=interleaved,
        inplace=inplace,
    )


apply_rotary_emb_func = apply_rotary_emb


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """qkv: ``(batch, seqlen, 3, nheads, headdim)``. Devuelve un tensor NUEVO (ver docstring)."""
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise NotImplementedError(
            "apply_rotary_emb_qkv_ en la capa de compatibilidad torch de RNFE sólo soporta "
            f"qkv (batch, seqlen, 3, nheads, headdim); recibí {tuple(qkv.shape)}. El layout "
            "MQA/GQA de 4 dims (num_heads_q + 2·num_heads_k) no lo usa nadie en `engines/` y "
            "no lo fabrico."
        )
    cos_k = cos if cos_k is None else cos_k
    sin_k = sin if sin_k is None else sin_k
    q, k, v = qkv.unbind(dim=2)
    q = apply_rotary(q, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved)
    k = apply_rotary(k, cos_k, sin_k, seqlen_offsets=seqlen_offsets, interleaved=interleaved)
    return torch.stack((q, k, v), dim=2)


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """kv: ``(batch, seqlen, 2, nheads, headdim)``. Devuelve un tensor NUEVO (ver docstring)."""
    if kv.dim() != 5 or kv.shape[2] != 2:
        raise ValueError(f"kv debe ser (batch, seqlen, 2, nheads, headdim); recibí {tuple(kv.shape)}")
    k, v = kv.unbind(dim=2)
    k = apply_rotary(k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved)
    return torch.stack((k, v), dim=2)


class RotaryEmbedding(torch.nn.Module):
    """RoPE (Su et al.), y xPos si ``scale_base is not None``.

    Misma API pública que la clase upstream: ``dim``, ``base``, ``interleaved``,
    ``scale_base``, ``pos_idx_in_fp32``, ``_update_cos_sin_cache``, ``_cos_cached``,
    ``_sin_cached``, ``scale``.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None) -> None:
        if (
            seqlen <= self._seq_len_cached
            and self._cos_cached is not None
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
            and not (self.training and self._cos_cached.is_inference())
        ):
            return

        self._seq_len_cached = seqlen
        if self.pos_idx_in_fp32:
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            inv_freq = (
                self._compute_inv_freq(device=device)
                if self.inv_freq.dtype != torch.float32
                else self.inv_freq
            )
        else:
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            inv_freq = self.inv_freq
        freqs = torch.outer(t, inv_freq.to(device=device))

        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if cu_seqlens is not None:
            raise NotImplementedError(
                "RotaryEmbedding(cu_seqlens=...) no está implementado en la capa de "
                "compatibilidad torch de RNFE. H-Net NO usa esta clase (tiene su propia copia "
                "en `engines/hnet/modules/rotary.py`, que sí soporta el modo empaquetado vía "
                "`flash_attn.ops.triton.rotary.apply_rotary`). No lo fabrico."
            )

        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(
                seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
            )
        else:
            raise ValueError(
                "con seqlen_offset tensorial hay que pasar max_seqlen (contrato upstream)"
            )

        if kv is None:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached,
                self._sin_cached,
                self._cos_k_cached,
                self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                num_heads_q=num_heads_q,
            )

        q = apply_rotary_emb(
            qkv,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            seqlen_offsets=seqlen_offset,
        )
        if self.scale is None:
            kv = apply_rotary_emb_kv_(
                kv,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
        else:
            kv = apply_rotary_emb_kv_(
                kv,
                self._cos_k_cached,
                self._sin_k_cached,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
        return q, kv
