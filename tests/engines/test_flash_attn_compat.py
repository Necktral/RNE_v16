"""La capa de compatibilidad `flash_attn/` es matemáticamente equivalente — y lo probamos.

Contexto (RTX 2070 Max-Q, Turing sm_75, 8 GB, sin nvcc, 2026-07-12):

FlashAttention **no se puede instalar acá** (necesita nvcc; oficialmente exige Ampere+).  Pero
`engines/hnet` la importa ⇒ H-Net era **inimportable**.  FlashAttention es una optimización de
**memoria** (tiling en SRAM), **no una matemática distinta**: todo lo que H-Net le pide tiene
equivalente exacto en torch puro.  `flash_attn/` (raíz del repo) es esa capa.

Qué pinea este archivo
======================
1. Equivalencia numérica de `swiglu`, `RMSNorm` y `apply_rotary` contra su definición
   matemática, escrita a mano acá.
2. La atención contra `softmax(QKᵀ/√d)V` ingenuo, con máscara causal; y que `causal=True`
   **realmente enmascare** (no que "no rompa").
3. Que H-Net **importe** (hoy eso era un ImportError).
4. **El que importa**: el encoder de H-Net (4 capas Mamba-2 del `1stage`) + `RoutingModule`
   end-to-end ⇒ **una probabilidad de frontera por posición**.  Ése es el camino que RNFE
   necesita (`runtime/neural/organs/n5_ingest.py`, `infer_boundaries(model, utf8_bytes)`).
5. Que el shim **defiera** a un FlashAttention real si alguna vez se instala.
6. **Falsificación**: un test que no puede fallar es otra mentira.  Los `test_*_FALSIFICACION_*`
   rompen la implementación a propósito y exigen que el test correspondiente se ponga rojo.

Tolerancias
===========
- fp32/fp64: `atol=1e-6` — es error de redondeo de fp32 acumulado sobre D≤128 términos.
- bf16: se compara contra la referencia calculada en **fp32** con `atol=2e-2`.  bf16 tiene 8
  bits de mantisa (~2 decimales); pedir más sería pedirle a bf16 que no sea bf16.
- Atención vs. referencia ingenua en fp32: `atol=1e-5` (SDPA reordena la suma; el softmax
  amplifica el reordenamiento).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="el sustrato neural necesita torch")
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

requiere_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requiere GPU CUDA (el kernel SSD de Mamba-2 es Triton sobre GPU)",
)


# =====================================================================================
# 0. El shim se declara como lo que es
# =====================================================================================


def test_el_paquete_admite_que_NO_es_flash_attention() -> None:
    """Si esto alguna vez dice que es FlashAttention, es mentira. Pineamos la honestidad."""
    import flash_attn

    assert flash_attn.IS_RNFE_TORCH_COMPAT_SHIM is True
    # __version__ deliberadamente 0.0.0: cualquier chequeo `>= 2.x` DEBE fallar.
    assert flash_attn.__version__.startswith("0.0.0+")
    doc = flash_attn.__doc__ or ""
    assert "NO ES FLASHATTENTION" in doc.upper()
    assert "O(L²)" in doc  # la regresión de memoria está declarada, no escondida


# =====================================================================================
# 1. swiglu
# =====================================================================================


def _swiglu_referencia(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Definición matemática, a mano: silu(x)·y = (x·σ(x))·y."""
    sigmoide = 1.0 / (1.0 + torch.exp(-x.double()))
    return (x.double() * sigmoide) * y.double()


def test_swiglu_es_silu_del_PRIMER_argumento_por_el_segundo() -> None:
    """`swiglu(a, b) == silu(a) * b`. El ORDEN importa: silu va sobre el primero.

    `engines/hnet/modules/mlp.py:29-31` hace `y, gate = fc1(x).chunk(2, -1)` y luego
    `swiglu(gate, y)` ⇒ la compuerta (`gate`) es la que pasa por el silu.  Invertirlo sería
    otra red.
    """
    from flash_attn.ops.activations import swiglu

    torch.manual_seed(0)
    a = torch.randn(4, 7, 32, dtype=torch.float32)
    b = torch.randn(4, 7, 32, dtype=torch.float32)

    obtenido = swiglu(a, b)
    esperado = _swiglu_referencia(a, b).to(torch.float32)

    assert obtenido.shape == a.shape
    assert torch.allclose(obtenido, esperado, atol=1e-6, rtol=0)

    # Y NO es simétrico: silu(a)·b != silu(b)·a. Si lo fuera, el test de orden no probaría nada.
    assert not torch.allclose(swiglu(a, b), swiglu(b, a), atol=1e-3)


def test_swiglu_en_bf16_computa_en_fp32_por_dentro() -> None:
    """FlashAttention hace `F.silu(x.float()) * y`; replicamos el upcast."""
    from flash_attn.ops.activations import swiglu

    torch.manual_seed(1)
    a = torch.randn(64, 128, dtype=torch.bfloat16)
    b = torch.randn(64, 128, dtype=torch.bfloat16)

    obtenido = swiglu(a, b)
    esperado = _swiglu_referencia(a, b).to(torch.bfloat16)

    assert obtenido.dtype == torch.bfloat16
    assert torch.allclose(obtenido.float(), esperado.float(), atol=2e-2, rtol=0)


def test_el_SwiGLU_de_hnet_usa_swiglu_y_da_lo_mismo_que_la_definicion() -> None:
    """El consumidor real (`hnet.modules.mlp.SwiGLU`) tiene que dar la matemática esperada."""
    from hnet.modules.mlp import SwiGLU

    torch.manual_seed(2)
    capa = SwiGLU(d_model=64, d_intermediate=128, bias=False)
    x = torch.randn(3, 5, 64)

    obtenido = capa(x)

    y, gate = capa.fc1(x).chunk(2, dim=-1)
    esperado = capa.fc2(_swiglu_referencia(gate, y).to(x.dtype))
    assert torch.allclose(obtenido, esperado, atol=1e-5, rtol=0)


# =====================================================================================
# 2. RMSNorm
# =====================================================================================


def _rmsnorm_referencia(x, weight, eps, residual=None):
    """A mano, en fp64: rstd = 1/√(mean(x²)+eps); y = x·rstd·w."""
    acc = x.double()
    if residual is not None:
        acc = acc + residual.double()
    rstd = torch.rsqrt(acc.square().mean(dim=-1, keepdim=True) + eps)
    return (acc * rstd) * weight.double(), acc


def test_rmsnorm_coincide_con_la_definicion_matematica() -> None:
    from flash_attn.ops.triton.layer_norm import RMSNorm

    torch.manual_seed(3)
    norma = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        norma.weight.copy_(torch.randn(64))
    x = torch.randn(2, 9, 64)

    obtenido = norma(x)
    esperado, _ = _rmsnorm_referencia(x, norma.weight, 1e-5)

    assert torch.allclose(obtenido.double(), esperado, atol=1e-6, rtol=0)


def test_rmsnorm_prenorm_suma_el_residual_ANTES_de_normalizar() -> None:
    from flash_attn.ops.triton.layer_norm import RMSNorm

    torch.manual_seed(4)
    norma = RMSNorm(32, eps=1e-5)
    x = torch.randn(2, 5, 32)
    residual = torch.randn(2, 5, 32)

    y, residual_out = norma(x, residual=residual, prenorm=True, residual_in_fp32=True)
    esperado_y, esperado_res = _rmsnorm_referencia(x, norma.weight, 1e-5, residual=residual)

    assert torch.allclose(y.double(), esperado_y, atol=1e-6, rtol=0)
    assert residual_out is not None
    assert torch.allclose(residual_out.double(), esperado_res, atol=1e-6, rtol=0)


def test_rmsnorm_devuelve_residual_None_igual_que_upstream_en_fp32_sin_residual() -> None:
    """Comportamiento CONTRAINTUITIVO del upstream, replicado a propósito.

    En `_layer_norm_fwd` (`engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:336-347`)
    `residual_out` sólo se materializa si `residual is not None` **o**
    `residual_dtype != x.dtype`.  Con `x` en fp32, `residual=None` y `residual_in_fp32=True`
    esas dos condiciones son falsas ⇒ upstream devuelve `(y, None)`.

    Consecuencia real: correr H-Net en **fp32** deja el stream residual en `None` y **no lo
    acumula**.  H-Net está pensado para bf16/fp16, donde fp32 != x.dtype y el residual sí se
    materializa.  Este test existe para que nadie "arregle" el shim y lo aleje del modelo real.
    """
    from flash_attn.ops.triton.layer_norm import RMSNorm

    norma = RMSNorm(16, eps=1e-5)

    x_fp32 = torch.randn(1, 4, 16, dtype=torch.float32)
    _, residual_out = norma(x_fp32, residual=None, prenorm=True, residual_in_fp32=True)
    assert residual_out is None, "en fp32 upstream NO materializa el residual"

    x_bf16 = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    norma_bf16 = RMSNorm(16, eps=1e-5, dtype=torch.bfloat16)
    _, residual_out_bf16 = norma_bf16(
        x_bf16, residual=None, prenorm=True, residual_in_fp32=True
    )
    assert residual_out_bf16 is not None
    assert residual_out_bf16.dtype == torch.float32


def test_rmsnorm_acepta_la_firma_que_usan_isotropic_y_block() -> None:
    """`isotropic.py:98`: `RMSNorm(d_model, eps=1e-5, **factory_kwargs)`.
    `block.py:74`:     `partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)`.
    donde `factory_kwargs = {"device": ..., "dtype": ...}`.
    """
    from functools import partial

    from flash_attn.ops.triton.layer_norm import RMSNorm

    factory_kwargs = {"device": torch.device("cpu"), "dtype": torch.float32}
    directa = RMSNorm(48, eps=1e-5, **factory_kwargs)
    assert directa.weight.shape == (48,)
    assert directa.weight.dtype == torch.float32

    norm_cls = partial(RMSNorm, eps=1e-5, **factory_kwargs)
    via_partial = norm_cls(48)
    assert isinstance(via_partial, RMSNorm)


def test_LA_TRAMPA_el_assert_isinstance_de_block_py_pasa() -> None:
    """`engines/hnet/modules/block.py:106` hace `assert isinstance(self.norm1, RMSNorm)`.

    Si nuestro `RMSNorm` no es LA clase que efectivamente se instancia, ese assert revienta.
    Este test es el que prueba que no la esquivamos con un alias o un wrapper.
    """
    from flash_attn.ops.triton.layer_norm import RMSNorm
    from hnet.modules.block import create_block

    bloque = create_block(
        "m",
        d_model=64,
        d_intermediate=0,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2, "chunk_size": 64},
        layer_idx=0,
    )
    assert isinstance(bloque.norm1, RMSNorm)
    assert type(bloque.norm1) is RMSNorm


@requiere_cuda
def test_rmsnorm_coincide_con_el_kernel_triton_del_vendor() -> None:
    """ORÁCULO EXTERNO: contra el `RMSNorm` Triton del vendor de Mamba.

    `engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:953` es una **copia literal** del
    `RMSNorm` de FlashAttention.  O sea: acá no comparamos contra una referencia que escribí
    yo, comparamos contra el código de Tri Dao corriendo de verdad en la GPU.  Si esto pasa, la
    equivalencia con FlashAttention no es una opinión.

    Resultado MEDIDO (RTX 2070, 2026-07-12):
      * fp16  → diferencia **0.000e+00**: BIT A BIT idéntico.
      * bf16  → diferencia **0.000e+00**: BIT A BIT idéntico.
      * fp32  → max|Δ| = 1.9e-06, max relativo = 2.4e-07 ≈ **2 ulps** de fp32.  Es la
        diferencia entre `torch.rsqrt(v)` y el `1/tl.sqrt(v)` del kernel: redondeo, no
        matemática.  Por eso acá el fp32 va con tolerancia RELATIVA (rtol=1e-6 ≈ 8 ulps) y
        atol=0: un error de verdad (eps mal, rstd olvidado) mueve el valor órdenes de
        magnitud, no 2 ulps.
    """
    from mamba_ssm.ops.triton.layer_norm import RMSNorm as RMSNormTriton

    from flash_attn.ops.triton.layer_norm import RMSNorm as RMSNormTorch

    torch.manual_seed(5)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        triton_norm = RMSNormTriton(128, eps=1e-5, device="cuda", dtype=dtype)
        torch_norm = RMSNormTorch(128, eps=1e-5, device="cuda", dtype=dtype)
        with torch.no_grad():
            pesos = torch.randn(128, device="cuda", dtype=dtype)
            triton_norm.weight.copy_(pesos)
            torch_norm.weight.copy_(pesos)

        x = torch.randn(2, 64, 128, device="cuda", dtype=dtype)
        residual = torch.randn(2, 64, 128, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            y_t, res_t = triton_norm(x, residual=residual, prenorm=True, residual_in_fp32=True)
            y_p, res_p = torch_norm(x, residual=residual, prenorm=True, residual_in_fp32=True)

        assert y_p.dtype == y_t.dtype
        assert res_p.dtype == res_t.dtype
        # el residual (fp32, pura suma) tiene que salir IDÉNTICO en los tres dtypes
        assert torch.equal(res_t, res_p), dtype

        if dtype == torch.float32:
            assert torch.allclose(y_t, y_p, rtol=1e-6, atol=0.0), dtype
        else:
            # en fp16/bf16 el redondeo del dtype de salida absorbe los 2 ulps ⇒ exacto.
            assert torch.equal(y_t, y_p), f"{dtype} dejó de ser bit-exacto vs el kernel Triton"


# =====================================================================================
# 3. apply_rotary
# =====================================================================================


def _rotary_referencia(x, cos, sin, interleaved=False, conjugate=False):
    """A mano, en fp64. Rota sólo las primeras `2·cos.shape[-1]` dims."""
    rd = cos.shape[-1] * 2
    xr = x[..., :rd].double()
    c = cos.double()
    s = -sin.double() if conjugate else sin.double()
    if interleaved:
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
    else:
        x1, x2 = xr[..., : rd // 2], xr[..., rd // 2 :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    if interleaved:
        rot = torch.stack((o1, o2), dim=-1).flatten(start_dim=-2)
    else:
        rot = torch.cat((o1, o2), dim=-1)
    return torch.cat((rot, x[..., rd:].double()), dim=-1)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("conjugate", [False, True])
def test_apply_rotary_coincide_con_la_rotacion_de_pares(interleaved, conjugate) -> None:
    from flash_attn.ops.triton.rotary import apply_rotary

    torch.manual_seed(6)
    batch, seqlen, nheads, headdim, rotary_dim = 2, 12, 3, 32, 16
    x = torch.randn(batch, seqlen, nheads, headdim)
    cos = torch.randn(seqlen, rotary_dim // 2).cos()
    sin = torch.randn(seqlen, rotary_dim // 2).sin()

    obtenido = apply_rotary(x, cos, sin, interleaved=interleaved, conjugate=conjugate)
    # (seqlen, rd/2) -> (1, seqlen, 1, rd/2) para broadcastear sobre batch y cabezas
    esperado = _rotary_referencia(
        x,
        cos[None, :, None, :],
        sin[None, :, None, :],
        interleaved=interleaved,
        conjugate=conjugate,
    )

    assert obtenido.shape == x.shape
    assert torch.allclose(obtenido.double(), esperado, atol=1e-6, rtol=0)
    # Las dims fuera de rotary_dim quedan intactas.
    assert torch.equal(obtenido[..., rotary_dim:], x[..., rotary_dim:])


def test_apply_rotary_respeta_seqlen_offsets() -> None:
    """`seqlen_offsets=k` desplaza la posición: el token i usa cos[i+k]."""
    from flash_attn.ops.triton.rotary import apply_rotary

    torch.manual_seed(7)
    x = torch.randn(1, 4, 2, 16)
    cos = torch.randn(32, 8).cos()
    sin = torch.randn(32, 8).sin()

    con_offset = apply_rotary(x, cos, sin, seqlen_offsets=5)
    # equivalente: mirar cos/sin ya desplazados
    sin_offset = apply_rotary(x, cos[5:9], sin[5:9], seqlen_offsets=0)
    assert torch.allclose(con_offset, sin_offset, atol=1e-6, rtol=0)

    # y NO es lo mismo que sin offset (si lo fuera, el test no probaría nada)
    assert not torch.allclose(con_offset, apply_rotary(x, cos, sin), atol=1e-3)


def test_apply_rotary_inplace_escribe_a_traves_de_una_VISTA_de_qkv() -> None:
    """`hnet/modules/rotary.py:162-176` rota `qkv[:, :, :2].reshape(...)` con `inplace=True`.

    Eso es una VISTA de `qkv`: la escritura tiene que llegar al tensor padre, y `v` (el tercer
    bloque) NO puede tocarse.  Si el shim hiciera una copia, H-Net rotaría al vacío y nadie se
    enteraría.
    """
    from flash_attn.ops.triton.rotary import apply_rotary

    torch.manual_seed(8)
    batch, seqlen, nheads, headdim = 1, 6, 2, 16
    qkv = torch.randn(batch, seqlen, 3, nheads, headdim)
    qkv_original = qkv.clone()
    cos = torch.randn(seqlen, 8).cos()
    sin = torch.randn(seqlen, 8).sin()

    qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
    devuelto = apply_rotary(qk, cos, sin, inplace=True)

    assert devuelto.data_ptr() == qk.data_ptr(), "inplace debe devolver el MISMO tensor"
    assert not torch.equal(qkv[:, :, 0], qkv_original[:, :, 0]), "q no se rotó en el padre"
    assert not torch.equal(qkv[:, :, 1], qkv_original[:, :, 1]), "k no se rotó en el padre"
    assert torch.equal(qkv[:, :, 2], qkv_original[:, :, 2]), "v NO se debe tocar"

    esperado = _rotary_referencia(
        qkv_original[:, :, 0], cos[None, :, None, :], sin[None, :, None, :]
    )
    assert torch.allclose(qkv[:, :, 0].double(), esperado, atol=1e-6, rtol=0)


def test_apply_rotary_varlen_reinicia_las_posiciones_en_cada_secuencia() -> None:
    """Con `cu_seqlens`, cada secuencia empieza en la posición 0."""
    from flash_attn.ops.triton.rotary import apply_rotary

    torch.manual_seed(9)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    x = torch.randn(7, 2, 16)
    cos = torch.randn(8, 8).cos()
    sin = torch.randn(8, 8).sin()

    obtenido = apply_rotary(x, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=4)

    for inicio, fin in ((0, 3), (3, 7)):
        trozo = x[inicio:fin].unsqueeze(0)  # (1, L, H, D)
        esperado = apply_rotary(trozo, cos, sin).squeeze(0)
        assert torch.allclose(obtenido[inicio:fin], esperado, atol=1e-6, rtol=0)


def test_apply_rotary_coincide_con_la_referencia_torch_QUE_TRAE_HNET() -> None:
    """ORÁCULO EXTERNO #2: `hnet.modules.rotary.apply_rotary_emb_torch`.

    Esa función es **código upstream de FlashAttention** que ya vive en el repo (la referencia
    en torch con la que Tri Dao valida su propio kernel Triton).  No la escribí yo.
    """
    from hnet.modules.rotary import apply_rotary_emb_torch

    from flash_attn.ops.triton.rotary import apply_rotary

    torch.manual_seed(10)
    x = torch.randn(2, 10, 4, 32)
    cos = torch.randn(10, 8).cos()
    sin = torch.randn(10, 8).sin()

    nuestro = apply_rotary(x, cos, sin, interleaved=False)
    upstream = apply_rotary_emb_torch(x, cos, sin, interleaved=False)
    assert torch.allclose(nuestro, upstream, atol=1e-6, rtol=0)


# =====================================================================================
# 4. Atención
# =====================================================================================


def _atencion_referencia(q, k, v, causal=False, softmax_scale=None, window_left=-1):
    """Ingenua y explícita: softmax(QKᵀ·escala + máscara)·V, en fp64.

    q: (B, Sq, H, D); k, v: (B, Sk, H, D).  Máscara causal alineada ABAJO-DERECHA (como
    FlashAttention cuando Sq != Sk).
    """
    b, sq, h, d = q.shape
    sk = k.shape[1]
    escala = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    qd = q.double().permute(0, 2, 1, 3)  # (B,H,Sq,D)
    kd = k.double().permute(0, 2, 1, 3)
    vd = v.double().permute(0, 2, 1, 3)

    logits = torch.matmul(qd, kd.transpose(-1, -2)) * escala  # (B,H,Sq,Sk)

    if causal or window_left >= 0:
        i = torch.arange(sq).unsqueeze(1) + (sk - sq)  # alineación abajo-derecha
        j = torch.arange(sk).unsqueeze(0)
        permitido = torch.ones(sq, sk, dtype=torch.bool)
        if causal:
            permitido &= (j - i) <= 0
        if window_left >= 0:
            permitido &= (i - j) <= window_left
        logits = logits.masked_fill(~permitido, float("-inf"))
        pesos = torch.softmax(logits, dim=-1)
        pesos = torch.nan_to_num(pesos, nan=0.0)  # filas sin ninguna clave ⇒ salida 0
    else:
        pesos = torch.softmax(logits, dim=-1)

    return torch.matmul(pesos, vd).permute(0, 2, 1, 3)  # (B,Sq,H,D)


@pytest.mark.parametrize("causal", [False, True])
def test_qkvpacked_coincide_con_softmax_QKt_V_ingenuo(causal) -> None:
    from flash_attn import flash_attn_qkvpacked_func

    torch.manual_seed(11)
    qkv = torch.randn(2, 16, 3, 4, 32)

    obtenido = flash_attn_qkvpacked_func(qkv, causal=causal)
    q, k, v = qkv.unbind(dim=2)
    esperado = _atencion_referencia(q, k, v, causal=causal)

    assert obtenido.shape == (2, 16, 4, 32)
    assert torch.allclose(obtenido.double(), esperado, atol=1e-5, rtol=0)


def test_causal_REALMENTE_enmascara_el_futuro() -> None:
    """No basta con que `causal=True` no rompa: tiene que CAMBIAR el resultado.

    Prueba operativa: si cambio el token del final, la salida de los tokens anteriores NO
    puede moverse.  Con `causal=False` sí se mueve.  Eso es enmascarar de verdad.
    """
    from flash_attn import flash_attn_qkvpacked_func

    torch.manual_seed(12)
    qkv = torch.randn(1, 8, 3, 2, 16)

    causal_a = flash_attn_qkvpacked_func(qkv, causal=True)
    denso_a = flash_attn_qkvpacked_func(qkv, causal=False)

    qkv_b = qkv.clone()
    qkv_b[:, -1] = torch.randn(1, 3, 2, 16)  # sólo el ÚLTIMO token cambia

    causal_b = flash_attn_qkvpacked_func(qkv_b, causal=True)
    denso_b = flash_attn_qkvpacked_func(qkv_b, causal=False)

    # Causal: los tokens 0..6 no ven el 7 ⇒ idénticos.
    assert torch.allclose(causal_a[:, :-1], causal_b[:, :-1], atol=1e-6, rtol=0)
    # No causal: todos ven al 7 ⇒ se mueven.  Si esto no se moviera, el test de arriba sería vacío.
    assert not torch.allclose(denso_a[:, :-1], denso_b[:, :-1], atol=1e-3)
    # Y causal != denso: la máscara hace algo.
    assert not torch.allclose(causal_a, denso_a, atol=1e-3)


def test_causal_con_Sq_distinto_de_Sk_se_alinea_ABAJO_DERECHA() -> None:
    """FlashAttention alinea la causal a la esquina inferior derecha; torch a la superior izq.

    Si el shim usara `is_causal=True` de torch acá, la máscara sería la equivocada y el
    resultado, silenciosamente distinto.  Este test lo caza.
    """
    from flash_attn import flash_attn_kvpacked_func

    torch.manual_seed(13)
    q = torch.randn(1, 4, 2, 16)
    kv = torch.randn(1, 10, 2, 2, 16)
    k, v = kv.unbind(dim=2)

    obtenido = flash_attn_kvpacked_func(q, kv, causal=True)
    esperado_abajo_derecha = _atencion_referencia(q, k, v, causal=True)
    assert torch.allclose(obtenido.double(), esperado_abajo_derecha, atol=1e-5, rtol=0)

    # La alineación arriba-izquierda (la de torch) daría OTRA cosa: el test no es vacío.
    arriba_izquierda = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=True
    ).permute(0, 2, 1, 3)
    assert not torch.allclose(obtenido, arriba_izquierda, atol=1e-3)


def test_softmax_scale_se_respeta() -> None:
    from flash_attn import flash_attn_func

    torch.manual_seed(14)
    q, k, v = (torch.randn(1, 6, 2, 16) for _ in range(3))

    obtenido = flash_attn_func(q, k, v, softmax_scale=0.05, causal=True)
    esperado = _atencion_referencia(q, k, v, causal=True, softmax_scale=0.05)
    assert torch.allclose(obtenido.double(), esperado, atol=1e-5, rtol=0)

    # con la escala por defecto (1/√d = 0.25) da otra cosa
    assert not torch.allclose(obtenido, flash_attn_func(q, k, v, causal=True), atol=1e-3)


def test_window_size_hace_atencion_local() -> None:
    """`window_size=(left, -1)` + causal ⇒ cada query ve sólo `left` claves hacia atrás."""
    from flash_attn import flash_attn_qkvpacked_func

    torch.manual_seed(15)
    qkv = torch.randn(1, 12, 3, 2, 16)
    q, k, v = qkv.unbind(dim=2)

    obtenido = flash_attn_qkvpacked_func(qkv, causal=True, window_size=(2, -1))
    esperado = _atencion_referencia(q, k, v, causal=True, window_left=2)
    assert torch.allclose(obtenido.double(), esperado, atol=1e-5, rtol=0)

    # y la ventana cambia el resultado respecto de la causal global
    assert not torch.allclose(
        obtenido, flash_attn_qkvpacked_func(qkv, causal=True), atol=1e-3
    )


def test_varlen_qkvpacked_atiende_cada_secuencia_AISLADA() -> None:
    """`cu_seqlens` ⇒ bloque-diagonal: la secuencia 1 no puede ver a la 0."""
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func

    torch.manual_seed(16)
    cu_seqlens = torch.tensor([0, 5, 12], dtype=torch.int32)
    qkv = torch.randn(12, 3, 2, 16)

    obtenido = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=7, causal=True)
    assert obtenido.shape == (12, 2, 16)

    for inicio, fin in ((0, 5), (5, 12)):
        trozo = qkv[inicio:fin].unsqueeze(0)
        esperado = flash_attn_qkvpacked_func(trozo, causal=True).squeeze(0)
        assert torch.allclose(obtenido[inicio:fin], esperado, atol=1e-6, rtol=0)

    # FALSIFICACIÓN: si la segunda secuencia VIERA a la primera, el resultado sería otro.
    denso = flash_attn_qkvpacked_func(qkv.unsqueeze(0), causal=True).squeeze(0)
    assert not torch.allclose(obtenido[5:], denso[5:], atol=1e-3)


def test_varlen_kvpacked_coincide_con_la_referencia() -> None:
    from flash_attn import flash_attn_varlen_kvpacked_func

    torch.manual_seed(17)
    cu_q = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 8], dtype=torch.int32)
    q = torch.randn(8, 2, 16)
    kv = torch.randn(8, 2, 2, 16)

    obtenido = flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, 5, 5, causal=True)
    assert obtenido.shape == (8, 2, 16)

    k, v = kv.unbind(dim=1)
    for inicio, fin in ((0, 3), (3, 8)):
        esperado = _atencion_referencia(
            q[inicio:fin].unsqueeze(0),
            k[inicio:fin].unsqueeze(0),
            v[inicio:fin].unsqueeze(0),
            causal=True,
        ).squeeze(0)
        assert torch.allclose(obtenido[inicio:fin].double(), esperado, atol=1e-5, rtol=0)


def test_gqa_replica_cabezas_de_kv() -> None:
    from flash_attn import flash_attn_func

    torch.manual_seed(18)
    q = torch.randn(1, 6, 8, 16)  # 8 cabezas de query
    k = torch.randn(1, 6, 2, 16)  # 2 de kv  ⇒ GQA 4:1
    v = torch.randn(1, 6, 2, 16)

    obtenido = flash_attn_func(q, k, v, causal=True)
    k_exp = k.repeat_interleave(4, dim=2)
    v_exp = v.repeat_interleave(4, dim=2)
    esperado = _atencion_referencia(q, k_exp, v_exp, causal=True)
    assert torch.allclose(obtenido.double(), esperado, atol=1e-5, rtol=0)


def test_with_kvcache_escribe_la_cache_y_atiende_al_prefijo() -> None:
    """Camino de generación. RNFE no lo usa, pero está implementado de verdad."""
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    torch.manual_seed(19)
    batch, cache_len, nheads, headdim = 2, 16, 2, 16
    k_cache = torch.zeros(batch, cache_len, nheads, headdim)
    v_cache = torch.zeros(batch, cache_len, nheads, headdim)
    prefijo_k = torch.randn(batch, 5, nheads, headdim)
    prefijo_v = torch.randn(batch, 5, nheads, headdim)
    k_cache[:, :5] = prefijo_k
    v_cache[:, :5] = prefijo_v

    q = torch.randn(batch, 1, nheads, headdim)
    k_nuevo = torch.randn(batch, 1, nheads, headdim)
    v_nuevo = torch.randn(batch, 1, nheads, headdim)

    obtenido = flash_attn_with_kvcache(
        q, k_cache, v_cache, k_nuevo, v_nuevo, cache_seqlens=5, causal=True
    )

    # la caché quedó escrita en la posición 5
    assert torch.equal(k_cache[:, 5:6], k_nuevo)
    assert torch.equal(v_cache[:, 5:6], v_nuevo)

    # y el resultado es atender a las 6 claves (prefijo + la nueva)
    esperado = flash_attn_func(
        q, k_cache[:, :6], v_cache[:, :6], causal=True
    )
    assert obtenido.shape == (batch, 1, nheads, headdim)
    assert torch.allclose(obtenido, esperado, atol=1e-6, rtol=0)


# =====================================================================================
# 5. Lo que NO está implementado falla RUIDOSAMENTE (no fabrica)
# =====================================================================================


def test_lo_no_implementado_tira_NotImplementedError_y_dice_QUE_falta() -> None:
    """El hallazgo central de la campaña: "ausencia de dato = evidencia favorable".

    Este shim no reproduce eso.  Lo que no sabe hacer, lo grita.  Nunca devuelve ceros, ni un
    `pass`, ni una constante plausible.
    """
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_with_kvcache
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
    from flash_attn.utils.generation import GenerationMixin

    qkv = torch.randn(1, 4, 3, 2, 16)

    for kwargs in (
        {"dropout_p": 0.1},
        {"alibi_slopes": torch.ones(2)},
        {"softcap": 30.0},
        {"return_attn_probs": True},
    ):
        with pytest.raises(NotImplementedError) as exc:
            flash_attn_qkvpacked_func(qkv, **kwargs)
        mensaje = str(exc.value)
        assert "NO ES FlashAttention" in mensaje
        assert "torch puro" in mensaje

    with pytest.raises(NotImplementedError, match="paginada"):
        flash_attn_with_kvcache(
            torch.randn(1, 1, 2, 16),
            torch.zeros(1, 4, 2, 16),
            torch.zeros(1, 4, 2, 16),
            cache_seqlens=0,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )

    # sin cache_seqlens NO adivinamos 0: fallamos.
    with pytest.raises(NotImplementedError, match="cache_seqlens"):
        flash_attn_with_kvcache(
            torch.randn(1, 1, 2, 16),
            torch.zeros(1, 4, 2, 16),
            torch.zeros(1, 4, 2, 16),
        )

    with pytest.raises(NotImplementedError, match="dropout"):
        rms_norm_fn(torch.randn(1, 8), torch.ones(8), None, dropout_p=0.2)

    with pytest.raises(NotImplementedError, match="generate"):
        GenerationMixin().generate()


# =====================================================================================
# 6. Deferir al FlashAttention real si alguna vez se instala
# =====================================================================================


def test_el_shim_SE_APARTA_si_hay_un_flash_attn_de_verdad_instalado(tmp_path) -> None:
    """La raíz del repo va ANTES que site-packages en sys.path: sin esto, el shim taparía al
    FlashAttention real y nadie se enteraría.

    Fabricamos un `flash_attn` falso "instalado" en otro directorio de `sys.path` y exigimos
    que `import flash_attn` resuelva a ÉSE, no a nosotros.
    """
    fake_dir = tmp_path / "site-packages-falso"
    paquete = fake_dir / "flash_attn"
    paquete.mkdir(parents=True)
    (paquete / "__init__.py").write_text(
        '__version__ = "2.7.4.post1"\nSOY_EL_FLASH_ATTN_REAL = True\n', encoding="utf-8"
    )

    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])   # la raíz del repo, PRIMERO (donde vive el shim)
        sys.path.append(sys.argv[2])      # el "site-packages" con el flash_attn "real"
        import flash_attn
        print(flash_attn.__version__)
        print(getattr(flash_attn, "SOY_EL_FLASH_ATTN_REAL", False))
        print(getattr(flash_attn, "IS_RNFE_TORCH_COMPAT_SHIM", False))
        print(flash_attn.__file__)
        """
    )
    resultado = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT), str(fake_dir)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),  # cwd != raíz del repo, para no meterla por la puerta de atrás
    )
    version, es_real, es_shim, archivo = resultado.stdout.strip().splitlines()

    assert version == "2.7.4.post1", "el shim tapó al flash_attn real"
    assert es_real == "True"
    assert es_shim == "False"
    assert str(fake_dir) in archivo


def test_sin_flash_attn_real_el_shim_SI_atiende() -> None:
    """El contrapunto del test anterior: hoy, acá, no hay ninguno instalado ⇒ manda el shim.

    Si este test se pone rojo, es que alguien instaló FlashAttention de verdad — y entonces el
    de arriba pasa a ser el que manda.  Los dos juntos cubren los dos mundos.
    """
    import flash_attn

    assert getattr(flash_attn, "IS_RNFE_TORCH_COMPAT_SHIM", False) is True
    assert str(REPO_ROOT) in flash_attn.__file__


# =====================================================================================
# 7. H-Net importa (esto era un ImportError)
# =====================================================================================


def test_hnet_importa_sin_flash_attention() -> None:
    """Antes de este paquete, CADA una de estas líneas era `ModuleNotFoundError: flash_attn`."""
    from hnet.models.mixer_seq import HNetForCausalLM
    from hnet.modules.block import Block, create_block
    from hnet.modules.dc import ChunkLayer, DeChunkLayer, RoutingModule
    from hnet.modules.isotropic import Isotropic
    from hnet.modules.mha import CausalMHA
    from hnet.modules.mlp import SwiGLU

    for simbolo in (
        HNetForCausalLM,
        Block,
        create_block,
        RoutingModule,
        ChunkLayer,
        DeChunkLayer,
        Isotropic,
        CausalMHA,
        SwiGLU,
    ):
        assert simbolo is not None


def test_hnet_1stage_L_declara_un_encoder_de_4_capas_mamba_SIN_atencion() -> None:
    """El camino de RNFE (fronteras) no toca la atención. Lo pineamos leyendo el config real.

    `arch_layout: ["m4", ["T22"], "m4"]` ⇒ encoder = 4 Mamba (minúscula ⇒ sin MLP).
    `d_intermediate: [0, 4096]` ⇒ la etapa 0 no tiene FFN ⇒ ni `swiglu` ni rotary.
    La red `T22` (atención) va DESPUÉS del chunker y sólo sirve para generar texto.
    """
    config = json.loads(
        (REPO_ROOT / "engines" / "hnet" / "configs" / "hnet_1stage_L.json").read_text()
    )
    assert config["arch_layout"][0] == "m4"
    assert config["arch_layout"][2] == "m4"
    assert config["arch_layout"][1] == ["T22"]
    assert config["d_intermediate"][0] == 0


# =====================================================================================
# 8. EL TEST QUE IMPORTA: fronteras end-to-end
# =====================================================================================


def _config_1stage(use_mem_eff_path: bool = False):
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

    crudo = json.loads(
        (REPO_ROOT / "engines" / "hnet" / "configs" / "hnet_1stage_L.json").read_text()
    )
    return HNetConfig(
        arch_layout=crudo["arch_layout"],
        d_model=crudo["d_model"],
        d_intermediate=crudo["d_intermediate"],
        vocab_size=crudo["vocab_size"],
        ssm_cfg=SSMConfig(**crudo["ssm_cfg"], use_mem_eff_path=use_mem_eff_path),
        attn_cfg=AttnConfig(**crudo["attn_cfg"]),
        tie_embeddings=crudo["tie_embeddings"],
    )


@requiere_cuda
def test_EL_CAMINO_DE_RNFE_encoder_de_hnet_mas_routing_da_una_prob_por_byte() -> None:
    """El encoder real del `hnet_1stage_L` (4 capas Mamba-2, d_model=1024, 26 M params) +
    `RoutingModule`, end-to-end, sobre bytes UTF-8 ⇒ **una probabilidad de frontera por byte**.

    Ése es exactamente el contrato que pide `runtime/neural/organs/n5_ingest.py:230`:
    `infer_boundaries(model, utf8_bytes) -> Sequence[float]`, `len(probs) == len(bytes)`.

    ⚠ LO QUE ESTE TEST **NO** PRUEBA: que las fronteras sean BUENAS.  Los pesos son aleatorios
    (no hay checkpoint entrenado en esta máquina) ⇒ las probabilidades rondan 0.5, que es
    justo lo que da un `cos_sim ≈ 0` entre estados ocultos aleatorios.  Esto prueba la
    **cañería**, no la semántica.  Decirlo es parte del trato.
    """
    from hnet.modules.dc import RoutingModule
    from hnet.modules.isotropic import Isotropic

    torch.manual_seed(20)
    config = _config_1stage()
    d_model = config.d_model[0]

    encoder = Isotropic(config, pos_idx=0, stage_idx=0, device="cuda", dtype=torch.bfloat16)
    router = RoutingModule(d_model, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.nn.Embedding(
        config.vocab_size, d_model, device="cuda", dtype=torch.bfloat16
    )

    assert encoder.arch_full == ["m", "m", "m", "m"], "el encoder del 1stage son 4 Mamba"

    texto = "El organismo no fabrica un default benigno cuando no sabe."
    utf8 = texto.encode("utf-8")
    ids = torch.tensor([list(utf8)], device="cuda", dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)

    with torch.no_grad():
        hidden = encoder(embeddings(ids), mask=mask)
        salida = router(hidden, mask=mask)

    probabilidades = salida.boundary_prob[..., 1].float()

    # Contrato N5: UNA probabilidad por byte.
    assert hidden.shape == (1, len(utf8), d_model)
    assert probabilidades.shape == (1, len(utf8))
    assert torch.isfinite(hidden).all(), "el encoder dio NaN/Inf"
    assert torch.isfinite(probabilidades).all()

    # Son probabilidades de verdad: en [0,1] y las dos clases suman 1.
    assert float(probabilidades.min()) >= 0.0
    assert float(probabilidades.max()) <= 1.0
    assert torch.allclose(
        salida.boundary_prob.sum(dim=-1).float(),
        torch.ones(1, len(utf8), device="cuda"),
        atol=1e-2,
    )

    # El primer byte es SIEMPRE frontera (`dc.py:95-96`, PAD_PROB = 1.0).
    assert float(probabilidades[0, 0]) == pytest.approx(1.0, abs=1e-3)
    assert bool(salida.boundary_mask[0, 0])

    # La máscara es el argmax de la probabilidad: coherencia interna.
    assert torch.equal(salida.boundary_mask[0], probabilidades[0] > 0.5)


@requiere_cuda
def test_el_camino_de_RNFE_no_necesita_atencion_en_absoluto() -> None:
    """FALSIFICACIÓN del argumento: si el encoder usara atención, romper `flash_attn_func`
    rompería las fronteras.  No las rompe ⇒ el encoder no la usa.

    (Es la razón por la que la regresión de memoria O(L²) NO afecta a N5.)
    """
    import flash_attn
    from hnet.modules.dc import RoutingModule
    from hnet.modules.isotropic import Isotropic

    torch.manual_seed(21)
    config = _config_1stage()
    encoder = Isotropic(config, pos_idx=0, stage_idx=0, device="cuda", dtype=torch.bfloat16)
    router = RoutingModule(config.d_model[0], device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 64, config.d_model[0], device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 64, dtype=torch.bool, device="cuda")

    def explota(*args, **kwargs):
        raise AssertionError("el encoder NO debería llamar a la atención")

    originales = {
        nombre: getattr(flash_attn, nombre)
        for nombre in (
            "flash_attn_func",
            "flash_attn_qkvpacked_func",
            "flash_attn_kvpacked_func",
            "flash_attn_varlen_qkvpacked_func",
            "flash_attn_varlen_kvpacked_func",
            "flash_attn_with_kvcache",
        )
    }
    try:
        for nombre in originales:
            setattr(flash_attn, nombre, explota)
        with torch.no_grad():
            salida = router(encoder(x, mask=mask), mask=mask)
    finally:
        for nombre, fn in originales.items():
            setattr(flash_attn, nombre, fn)

    assert salida.boundary_prob.shape == (1, 64, 2)


@requiere_cuda
def test_el_bloque_de_atencion_de_hnet_corre_en_turing() -> None:
    """La red interna `T22` (la que SÍ usa atención) también corre. Sin FlashAttention."""
    from hnet.modules.block import create_block

    torch.manual_seed(22)
    bloque = create_block(
        "T",
        d_model=256,
        d_intermediate=512,
        attn_cfg={"num_heads": 4, "rotary_emb_dim": 32, "window_size": -1},
        layer_idx=0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        salida, residual = bloque(x, residual=None)

    assert salida.shape == (1, 128, 256)
    assert torch.isfinite(salida).all()
    assert residual is not None and residual.dtype == torch.float32


# =====================================================================================
# 9. La regresión de memoria: MEDIRLA, no suponerla
# =====================================================================================


def _pico_vram(fn) -> float:
    """Pico de VRAM (MiB) que asigna `fn`, descontando lo que ya estaba."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        salida = fn()
    pico = (torch.cuda.max_memory_allocated() - base) / 2**20
    del salida
    gc.collect()
    torch.cuda.empty_cache()
    return pico


@requiere_cuda
def test_LA_REGRESION_DE_MEMORIA_es_real_y_depende_del_dtype() -> None:
    """La regresión O(L²) NO es uniforme, y la diferencia la hace el **dtype**.

    Medido en RTX 2070 (Turing sm_75) con torch 2.13.0+cu130, B=1 H=16 D=96 (las dims de la
    red `T22` del `hnet_1stage_L`):

      * `EFFICIENT_ATTENTION` (el backend mem-efficient de torch) **SÍ corre en Turing**…
        pero **sólo en fp16/fp32**.  En **bf16 lo rechaza**:
        "Expected query, key and value to all be of dtype: {Half, Float}. Got BFloat16".
      * `FLASH_ATTENTION` de torch **no existe** en Turing: exige sm_80+.
      * ⇒ en **bf16** todo cae al backend `MATH`, que materializa la matriz L×L ⇒ **O(L²)**.

    Números medidos (pico de VRAM):

        L      fp16 causal   fp16 ventana   bf16 causal
         1024      3.0 MB        10.0 MB      181.1 MB
         2048      6.0 MB        40.0 MB      640.0 MB
         4096     12.0 MB       160.1 MB     2464.1 MB
         8192     24.0 MB       640.2 MB          OOM
        16384     72.0 MB      2560.4 MB          OOM
        32768    144.0 MB           OOM           OOM
        65536    288.0 MB           OOM           OOM

    Conclusión operativa: **en Turing, fp16 — no bf16.**  Y la ventana deslizante es
    cuadrática siempre (la máscara se materializa).

    Este test pinea las tres afirmaciones con márgenes anchos (2x) para no ser frágil ante
    cambios de allocator: lo que se pinea es el ORDEN DE MAGNITUD, que es lo que importa.
    """
    from flash_attn import flash_attn_qkvpacked_func

    B, H, D, L = 1, 16, 96, 2048

    def corre(dtype, **kwargs):
        qkv = torch.randn(B, L, 3, H, D, device="cuda", dtype=dtype)
        return _pico_vram(lambda: flash_attn_qkvpacked_func(qkv, **kwargs))

    fp16_causal = corre(torch.float16, causal=True)
    fp16_ventana = corre(torch.float16, causal=True, window_size=(1023, -1))
    bf16_causal = corre(torch.bfloat16, causal=True)

    # 1. fp16 + causal denso es LINEAL: el pico es del orden de la salida (L·H·D·2 B ≈ 6 MB),
    #    no de la matriz L² (que en fp16 serían 16·2048²·2 B = 128 MB por cabeza-batch).
    assert fp16_causal < 32, f"fp16 causal debería ser ~6 MB, midió {fp16_causal:.1f} MB"

    # 2. La máscara explícita (ventana) cuesta un orden de magnitud más: es O(L²).
    assert fp16_ventana > fp16_causal * 3, (
        f"la ventana deslizante debería materializar una máscara L² "
        f"(medido: {fp16_ventana:.1f} MB vs {fp16_causal:.1f} MB causal)"
    )

    # 3. bf16 es la trampa: ~100x más caro que fp16 por caer al backend MATH.
    assert bf16_causal > fp16_causal * 20, (
        f"bf16 debería caer al backend MATH y costar ~100x "
        f"(medido: {bf16_causal:.1f} MB vs {fp16_causal:.1f} MB en fp16). "
        f"Si esto se pone rojo, torch habilitó mem-efficient para bf16 en Turing: "
        f"BUENA noticia, actualizá el docstring de flash_attn/__init__.py."
    )


@requiere_cuda
def test_el_camino_cuadratico_AVISA_no_es_silencioso() -> None:
    """Una regresión de memoria silenciosa es una mentira por omisión. Ésta grita."""
    import flash_attn.flash_attn_interface as interfaz
    from flash_attn import flash_attn_qkvpacked_func

    interfaz._avisos_emitidos.clear()
    qkv = torch.randn(1, 2048, 3, 4, 64, device="cuda", dtype=torch.bfloat16)

    with pytest.warns(UserWarning, match="USÁ fp16, no bf16"):
        with torch.no_grad():
            flash_attn_qkvpacked_func(qkv, causal=True)

    interfaz._avisos_emitidos.clear()
    with pytest.warns(UserWarning, match=r"O\(L²\)"):
        with torch.no_grad():
            flash_attn_qkvpacked_func(qkv, causal=True, window_size=(64, -1))

    interfaz._avisos_emitidos.clear()


# =====================================================================================
# 10. FALSIFICACIÓN: si rompo la implementación, los tests se tienen que poner ROJOS
# =====================================================================================


def test_FALSIFICACION_swiglu_con_los_argumentos_al_reves_falla_el_test() -> None:
    """Rompo `swiglu` invirtiendo los argumentos (el bug clásico) y exijo rojo."""
    import flash_attn.ops.activations as activations

    original = activations.swiglu
    try:
        activations.swiglu = lambda x, y: original(y, x)  # ¡al revés!
        with pytest.raises(AssertionError):
            a = torch.randn(4, 32)
            b = torch.randn(4, 32)
            assert torch.allclose(
                activations.swiglu(a, b),
                _swiglu_referencia(a, b).to(torch.float32),
                atol=1e-6,
            )
    finally:
        activations.swiglu = original


def test_FALSIFICACION_rmsnorm_sin_eps_ni_rstd_falla_el_test() -> None:
    """Un `RMSNorm` que no normaliza (devuelve x·w) NO pasa el test de equivalencia."""
    torch.manual_seed(23)
    x = torch.randn(2, 5, 64) * 3.0
    w = torch.randn(64)

    esperado, _ = _rmsnorm_referencia(x, w, 1e-5)
    falso = (x * w).double()  # se "olvida" del rstd
    assert not torch.allclose(falso, esperado, atol=1e-6)


def test_FALSIFICACION_una_atencion_de_ceros_no_pasa_el_test_de_atencion() -> None:
    """El anti-patrón que esta campaña persigue: devolver un tensor plausible-pero-falso.

    Si `flash_attn_qkvpacked_func` devolviera ceros (o cualquier constante), el test de
    equivalencia con la referencia ingenua tendría que ponerse rojo.  Lo comprobamos.
    """
    torch.manual_seed(24)
    qkv = torch.randn(1, 8, 3, 2, 16)
    q, k, v = qkv.unbind(dim=2)
    esperado = _atencion_referencia(q, k, v, causal=True)

    ceros = torch.zeros_like(esperado)
    assert not torch.allclose(ceros, esperado, atol=1e-5)

    # y tampoco pasaría "devolver v tal cual" (el otro atajo tentador)
    assert not torch.allclose(v.double(), esperado, atol=1e-5)


def test_FALSIFICACION_una_causal_que_no_enmascara_falla_el_test_de_causalidad() -> None:
    """Si `causal=True` fuera un no-op, el test `test_causal_REALMENTE_enmascara_el_futuro`
    se pondría rojo.  Lo simulamos: la atención densa NO cumple la invariante causal.
    """
    from flash_attn import flash_attn_qkvpacked_func

    torch.manual_seed(25)
    qkv = torch.randn(1, 8, 3, 2, 16)
    qkv_b = qkv.clone()
    qkv_b[:, -1] = torch.randn(1, 3, 2, 16)

    # con causal=False (el "no-op"), los tokens previos SÍ se mueven ⇒ la invariante falla
    a = flash_attn_qkvpacked_func(qkv, causal=False)
    b = flash_attn_qkvpacked_func(qkv_b, causal=False)
    with pytest.raises(AssertionError):
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
