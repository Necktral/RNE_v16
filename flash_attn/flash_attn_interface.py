"""Las funciones de atención de FlashAttention, reimplementadas en **torch puro**.

NO es FlashAttention: no hay tiling en SRAM.  Ver el docstring de ``flash_attn/__init__.py``.

Equivalencia matemática que se sostiene acá
===========================================
``out = softmax(scale · Q Kᵀ + M) V``  con ``scale = 1/√D`` por defecto y ``M`` la máscara.

Alineación de la máscara causal (la trampa clásica)
---------------------------------------------------
FlashAttention alinea la máscara causal a la **esquina inferior derecha** cuando
``seqlen_q != seqlen_k`` (documentado en su README desde 2.1).  ``torch``'s
``is_causal=True`` alinea a la **superior izquierda**.  Acá construimos la máscara a mano con
``offset = Sk - Sq`` y sólo usamos el atajo ``is_causal=True`` cuando ``Sq == Sk`` (donde las
dos alineaciones coinciden).

Ventana deslizante (``window_size=(left, right)``)
--------------------------------------------------
Se conserva la clave ``j`` para la consulta ``i`` (con ``i`` ya desplazada por ``offset``) si
``(j - i) <= right`` y ``(i - j) <= left``.  ``-1`` significa "sin límite".  ``causal=True``
fuerza ``right = 0``, igual que FlashAttention.

Filas totalmente enmascaradas
-----------------------------
Si ``Sq > Sk`` con causal, las primeras filas no ven ninguna clave.  FlashAttention devuelve
**ceros** ahí.  ``softmax`` sobre una fila toda ``-inf`` da ``NaN``, así que desenmascaramos
la fila para el softmax y **anulamos la salida después**.  Los ceros de esas filas son
deliberados y equivalentes al upstream — no son un default fabricado.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .ops.triton.rotary import apply_rotary_positions

__all__ = [
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
]

_UNSUPPORTED = (
    "`{feature}` no está implementado en la capa de compatibilidad torch de RNFE "
    "(`flash_attn/` en la raíz del repo). NO ES FlashAttention: es torch puro. "
    "{why} "
    "Si de verdad lo necesitás, instalá FlashAttention de verdad (requiere nvcc y "
    "Ampere+; esta máquina es Turing sm_75) o implementalo acá con equivalencia exacta."
)


def _reject_unsupported(
    *,
    fn: str,
    dropout_p: float = 0.0,
    alibi_slopes=None,
    softcap: float = 0.0,
    return_attn_probs: bool = False,
) -> None:
    """Fallar RUIDOSAMENTE en vez de devolver algo plausible-pero-falso."""
    if dropout_p:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature=f"{fn}(dropout_p={dropout_p})",
                why=(
                    "El dropout de FlashAttention usa su propio RNG por bloque de tiles; "
                    "reproducir la MISMA máscara con torch es imposible, y una máscara "
                    "distinta daría gradientes distintos."
                ),
            )
        )
    if alibi_slopes is not None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature=f"{fn}(alibi_slopes=...)",
                why="ALiBi no se usa en `engines/` (H-Net usa rotary).",
            )
        )
    if softcap:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature=f"{fn}(softcap={softcap})",
                why="El tanh-softcap de logits no se usa en `engines/`.",
            )
        )
    if return_attn_probs:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature=f"{fn}(return_attn_probs=True)",
                why=(
                    "`scaled_dot_product_attention` no expone la matriz de probabilidades. "
                    "Devolver cualquier otra cosa sería fabricar un dato."
                ),
            )
        )


def _normalize_window(window_size, causal: bool) -> Tuple[int, int]:
    if window_size is None:
        left, right = -1, -1
    else:
        left, right = int(window_size[0]), int(window_size[1])
    if causal:
        # FlashAttention: `causal=True` ⇒ nada a la derecha de la diagonal.
        right = 0
    return left, right


def _expand_kv_heads(t: torch.Tensor, nheads_q: int, head_dim_pos: int = 2) -> torch.Tensor:
    """MQA/GQA: replicar cabezas de K/V hasta las de Q (equivale a lo que hace FA)."""
    nheads_kv = t.shape[head_dim_pos]
    if nheads_kv == nheads_q:
        return t
    if nheads_q % nheads_kv != 0:
        raise ValueError(
            f"nheads_q={nheads_q} no es múltiplo de nheads_kv={nheads_kv}: no es GQA válido."
        )
    return t.repeat_interleave(nheads_q // nheads_kv, dim=head_dim_pos)


def _build_allowed(
    seqlen_q: int,
    seqlen_k: int,
    left: int,
    right: int,
    key_lengths: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Máscara booleana (True = la clave participa).

    Devuelve ``None`` si no hace falta máscara (atención densa completa).
    Forma: ``(1, 1, Sq, Sk)`` o ``(B, 1, Sq, Sk)`` si hay largos de clave por batch.
    """
    if left < 0 and right < 0 and key_lengths is None:
        return None

    j = torch.arange(seqlen_k, device=device)
    q_idx = torch.arange(seqlen_q, device=device)

    if key_lengths is None:
        # Alineación abajo-derecha, igual que FlashAttention.
        i = (q_idx + (seqlen_k - seqlen_q)).unsqueeze(1)  # (Sq, 1)
        jj = j.unsqueeze(0)  # (1, Sk)
        allowed = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=device)
        if right >= 0:
            allowed &= (jj - i) <= right
        if left >= 0:
            allowed &= (i - jj) <= left
        return allowed.view(1, 1, seqlen_q, seqlen_k)

    klen = key_lengths.to(device=device, dtype=torch.long).view(-1, 1, 1)  # (B,1,1)
    i = q_idx.view(1, -1, 1) + (klen - seqlen_q)  # (B, Sq, 1)
    jj = j.view(1, 1, -1)  # (1,1,Sk)
    allowed = (jj < klen).expand(klen.shape[0], seqlen_q, seqlen_k).clone()
    if right >= 0:
        allowed &= (jj - i) <= right
    if left >= 0:
        allowed &= (i - jj) <= left
    return allowed.unsqueeze(1)


# Umbral a partir del cual el camino cuadrático deja de ser gratis (medido: a L=2048 en bf16
# el pico ya es de 640 MB).  Sq·Sk >= 2^22 ⇒ L >= 2048 en el caso cuadrado.
_UMBRAL_AVISO_CUADRATICO = 2**22
_avisos_emitidos: set[str] = set()


def _avisar_una_vez(clave: str, mensaje: str) -> None:
    if clave in _avisos_emitidos:
        return
    _avisos_emitidos.add(clave)
    warnings.warn(mensaje, UserWarning, stacklevel=4)


def _avisar_si_cuadratico(
    seqlen_q: int, seqlen_k: int, dtype: torch.dtype, device: torch.device, con_mascara: bool
) -> None:
    """Que la regresión de memoria NO sea silenciosa.

    En Turing el backend mem-efficient de torch **no acepta bf16** y el de flash no existe:
    cualquiera de esas dos condiciones tira el cómputo al backend ``MATH``, que materializa la
    matriz L×L.  Avisamos una vez por motivo, y sólo cuando el tamaño ya duele.
    """
    if seqlen_q * seqlen_k < _UMBRAL_AVISO_CUADRATICO or device.type != "cuda":
        return

    if con_mascara:
        _avisar_una_vez(
            "mascara",
            f"flash_attn (capa de compatibilidad torch de RNFE, NO es FlashAttention): "
            f"esta atención necesita una máscara explícita (ventana deslizante / varlen / "
            f"kvcache) con Sq={seqlen_q}, Sk={seqlen_k}. La máscara ({seqlen_q}x{seqlen_k}) "
            f"se materializa ⇒ memoria O(L²), no O(L). FlashAttention de verdad no pagaría "
            f"esto. Medido: OOM cerca de L=32K en una GPU de 8 GB.",
        )
        return

    if dtype == torch.bfloat16:
        _avisar_una_vez(
            "bf16",
            f"flash_attn (capa de compatibilidad torch de RNFE, NO es FlashAttention): "
            f"atención en bf16 con L={seqlen_q}. En Turing (sm_75) el backend mem-efficient "
            f"de torch NO acepta bf16 ('Expected ... {{Half, Float}}. Got BFloat16') y el de "
            f"flash exige sm_80+ ⇒ se cae al backend MATH, que materializa la matriz L×L "
            f"⇒ memoria O(L²). Medido: 640 MB a L=2048 y OOM a L=8192, contra 6 MB / 24 MB "
            f"en fp16. USÁ fp16, no bf16, para atención larga en esta GPU.",
        )


def _sdpa(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    allowed: Optional[torch.Tensor],
    scale: float,
    use_is_causal: bool,
) -> torch.Tensor:
    if use_is_causal:
        # Camino rápido y SIN O(L²): el backend mem-efficient de torch corre en Turing.
        return F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd, is_causal=True, scale=scale
        )
    if allowed is None:
        return F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, scale=scale)

    row_valid = allowed.any(dim=-1, keepdim=True)  # (..., Sq, 1)
    safe = allowed | (~row_valid)  # evita softmax(-inf en toda la fila) ⇒ NaN
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd, attn_mask=safe, scale=scale
    )
    return out * row_valid.to(out.dtype)


def _attention_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    window_size,
    key_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q: (B, Sq, H, D); k, v: (B, Sk, Hk, D) → out: (B, Sq, H, D)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"Se esperaban tensores (B, S, H, D); recibí q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    _, seqlen_q, nheads, head_dim = q.shape
    seqlen_k = k.shape[1]

    k = _expand_kv_heads(k, nheads)
    v = _expand_kv_heads(v, nheads)

    scale = float(softmax_scale) if softmax_scale is not None else head_dim**-0.5
    left, right = _normalize_window(window_size, causal)

    use_is_causal = (
        causal
        and key_lengths is None
        and left < 0
        and seqlen_q == seqlen_k
        and seqlen_q > 0
    )
    allowed = (
        None
        if use_is_causal
        else _build_allowed(seqlen_q, seqlen_k, left, right, key_lengths, q.device)
    )
    _avisar_si_cuadratico(
        seqlen_q, seqlen_k, q.dtype, q.device, con_mascara=allowed is not None
    )

    out = _sdpa(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        allowed,
        scale,
        use_is_causal,
    )
    return out.transpose(1, 2).contiguous()


# --------------------------------------------------------------------------------------
# API pública: mismas firmas que FlashAttention 2.x
# --------------------------------------------------------------------------------------


def flash_attn_func(
    q,
    k,
    v,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """q: (B, Sq, H, D); k, v: (B, Sk, Hk, D) → (B, Sq, H, D)."""
    _reject_unsupported(
        fn="flash_attn_func",
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
    )
    return _attention_bshd(q, k, v, softmax_scale, causal, window_size)


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """qkv: (B, S, 3, H, D) → (B, S, H, D)."""
    _reject_unsupported(
        fn="flash_attn_qkvpacked_func",
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
    )
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(f"qkv debe ser (B, S, 3, H, D); recibí {tuple(qkv.shape)}")
    q, k, v = qkv.unbind(dim=2)
    return _attention_bshd(q, k, v, softmax_scale, causal, window_size)


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """q: (B, Sq, H, D); kv: (B, Sk, 2, Hk, D) → (B, Sq, H, D)."""
    _reject_unsupported(
        fn="flash_attn_kvpacked_func",
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
    )
    if kv.dim() != 5 or kv.shape[2] != 2:
        raise ValueError(f"kv debe ser (B, Sk, 2, H, D); recibí {tuple(kv.shape)}")
    k, v = kv.unbind(dim=2)
    return _attention_bshd(q, k, v, softmax_scale, causal, window_size)


def _varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale,
    causal,
    window_size,
):
    """q: (total_q, H, D); k, v: (total_k, Hk, D) → (total_q, H, D).

    Implementación EXACTA por bucle de Python sobre la batch: cada secuencia se atiende
    aislada, que es justo lo que garantiza `cu_seqlens` (bloque-diagonal).  Es exacto; no es
    rápido.  Ver el docstring del paquete.
    """
    for name, t, dim in (("q", q, 3), ("k", k, 3), ("v", v, 3)):
        if t.dim() != dim:
            raise ValueError(
                f"{name} debe ser (total, H, D) en modo varlen; recibí {tuple(t.shape)}"
            )
    bounds_q = cu_seqlens_q.tolist()
    bounds_k = cu_seqlens_k.tolist()
    if len(bounds_q) != len(bounds_k):
        raise ValueError("cu_seqlens_q y cu_seqlens_k deben tener el mismo largo")

    pieces = []
    for b in range(len(bounds_q) - 1):
        q_i = q[bounds_q[b] : bounds_q[b + 1]].unsqueeze(0)
        k_i = k[bounds_k[b] : bounds_k[b + 1]].unsqueeze(0)
        v_i = v[bounds_k[b] : bounds_k[b + 1]].unsqueeze(0)
        pieces.append(
            _attention_bshd(q_i, k_i, v_i, softmax_scale, causal, window_size).squeeze(0)
        )
    if not pieces:
        raise ValueError("cu_seqlens vacío: no hay secuencias que atender")
    return torch.cat(pieces, dim=0)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table=None,
):
    _reject_unsupported(
        fn="flash_attn_varlen_func",
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
    )
    if block_table is not None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="flash_attn_varlen_func(block_table=...)",
                why="La KV cache paginada no se usa en `engines/`.",
            )
        )
    return _varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, window_size
    )


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """qkv: (total, 3, H, D) → (total, H, D)."""
    _reject_unsupported(
        fn="flash_attn_varlen_qkvpacked_func",
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
    )
    if qkv.dim() != 4 or qkv.shape[1] != 3:
        raise ValueError(f"qkv debe ser (total, 3, H, D); recibí {tuple(qkv.shape)}")
    q, k, v = qkv.unbind(dim=1)
    return _varlen_attention(
        q, k, v, cu_seqlens, cu_seqlens, softmax_scale, causal, window_size
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    """q: (total_q, H, D); kv: (total_k, 2, Hk, D) → (total_q, H, D)."""
    _reject_unsupported(
        fn="flash_attn_varlen_kvpacked_func",
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
        return_attn_probs=return_attn_probs,
    )
    if kv.dim() != 4 or kv.shape[1] != 2:
        raise ValueError(f"kv debe ser (total, 2, H, D); recibí {tuple(kv.shape)}")
    k, v = kv.unbind(dim=1)
    return _varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, window_size
    )


def _as_cache_seqlens(cache_seqlens, batch: int, device) -> torch.Tensor:
    if cache_seqlens is None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="flash_attn_with_kvcache(cache_seqlens=None)",
                why=(
                    "Sin `cache_seqlens` no se sabe dónde termina la caché; FlashAttention "
                    "lo trata como 0 pero acá NO adivinamos un default benigno."
                ),
            )
        )
    if isinstance(cache_seqlens, int):
        return torch.full((batch,), cache_seqlens, dtype=torch.long, device=device)
    return cache_seqlens.to(device=device, dtype=torch.long).reshape(-1)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Union[int, torch.Tensor, None] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table=None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size=(-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes=None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
):
    """Atención contra una KV-cache, con actualización **in-place** de la caché.

    q: (B, Sq, H, D); k_cache/v_cache: (B_cache, S_cache, Hk, D).
    Si se pasan ``k``/``v`` (B, Sq, Hk, D), se escriben en la caché en las posiciones
    ``cache_seqlens[b] .. cache_seqlens[b]+Sq-1`` y participan de la atención.

    Camino de **generación**, que RNFE (N5 ⇒ fronteras) no usa.  Está implementado igual, en
    torch puro y exacto, por bucle sobre la batch.
    """
    _reject_unsupported(
        fn="flash_attn_with_kvcache",
        alibi_slopes=alibi_slopes,
        softcap=softcap,
    )
    if block_table is not None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="flash_attn_with_kvcache(block_table=...)",
                why="La KV cache paginada no se usa en `engines/`.",
            )
        )
    if cache_leftpad is not None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="flash_attn_with_kvcache(cache_leftpad=...)",
                why="El padding izquierdo de la caché no se usa en `engines/`.",
            )
        )
    if return_softmax_lse:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="flash_attn_with_kvcache(return_softmax_lse=True)",
                why="`scaled_dot_product_attention` no devuelve el logsumexp.",
            )
        )
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos y rotary_sin se pasan juntos o no se pasan")

    batch, seqlen_q, _, _ = q.shape
    device = q.device
    seqlens = _as_cache_seqlens(cache_seqlens, batch, device)
    if seqlens.shape[0] != batch:
        raise ValueError(
            f"cache_seqlens tiene {seqlens.shape[0]} entradas y la batch es {batch}"
        )

    if cache_batch_idx is None:
        cache_idx = torch.arange(batch, device=device, dtype=torch.long)
    else:
        cache_idx = cache_batch_idx.to(device=device, dtype=torch.long).reshape(-1)

    # Posiciones absolutas de los tokens nuevos, por secuencia: (B, Sq)
    positions = seqlens.view(-1, 1) + torch.arange(seqlen_q, device=device).view(1, -1)

    if rotary_cos is not None:
        q = apply_rotary_positions(
            q, rotary_cos, rotary_sin, positions, interleaved=rotary_interleaved
        )
        if k is not None:
            k = apply_rotary_positions(
                k, rotary_cos, rotary_sin, positions, interleaved=rotary_interleaved
            )

    if (k is None) != (v is None):
        raise ValueError("k y v se pasan juntos o no se pasan")

    if k is not None:
        with torch.no_grad():
            for b in range(batch):
                start = int(seqlens[b])
                stop = start + seqlen_q
                if stop > k_cache.shape[1]:
                    raise ValueError(
                        f"la KV-cache (S={k_cache.shape[1]}) no entra {stop} posiciones "
                        f"para la secuencia {b}"
                    )
                k_cache[cache_idx[b], start:stop] = k[b].to(k_cache.dtype)
                v_cache[cache_idx[b], start:stop] = v[b].to(v_cache.dtype)
        total_lens = seqlens + seqlen_q
    else:
        total_lens = seqlens

    outs = []
    for b in range(batch):
        length = int(total_lens[b])
        k_b = k_cache[cache_idx[b], :length].unsqueeze(0)
        v_b = v_cache[cache_idx[b], :length].unsqueeze(0)
        outs.append(
            _attention_bshd(
                q[b : b + 1].to(k_b.dtype),
                k_b,
                v_b,
                softmax_scale,
                causal,
                window_size,
            ).squeeze(0)
        )
    return torch.stack(outs, dim=0).to(q.dtype)
