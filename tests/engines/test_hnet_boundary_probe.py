"""Tests del camino de fronteras de H-Net (embeddings -> encoder -> RoutingModule).

═══════════════════════════════════════════════════════════════════════════════
EL TEST QUE IMPORTA: `test_en_fp32_el_stream_residual_del_encoder_esta_MUERTO`
═══════════════════════════════════════════════════════════════════════════════

No es una curiosidad numérica.  En fp32, `flash_attn/ops/triton/layer_norm.py`
devuelve `residual_out=None` donde el kernel REAL
(`engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:414`) devuelve `x`.
`Block.forward` toma ese None y lo propaga: **las 4 capas Mamba del encoder
pierden TODAS sus conexiones residuales.**

Consecuencia práctica: cualquiera que evalúe (o peor: fine-tunee) el chunker en
fp32 está midiendo/optimizando OTRA función.  Estos tests lo pinean para que el
día que alguien arregle el shim, se enteren — y para que nadie vuelva a
"corregir" el evaluador hacia fp32.

Requieren GPU y los pesos; se saltan solos si no están.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lab.hnet_chunker.model import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_WEIGHTS,
    BoundaryProbe,
    build_config,
    residual_stream_is_alive,
)

_ENGLISH = (
    b"The quick brown fox jumps over the lazy dog. Hierarchical networks learn "
    b"their own boundaries from raw bytes."
)

requires_hnet = pytest.mark.skipif(
    not (torch.cuda.is_available() and DEFAULT_WEIGHTS.exists()),
    reason="requiere GPU CUDA y los pesos de H-Net",
)


@pytest.fixture(scope="module")
def probe():
    p = BoundaryProbe(build_config(), device="cuda", dtype=torch.float16)
    p.load_pretrained(DEFAULT_WEIGHTS)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# EL BUG DEL RESIDUAL (no necesita GPU ni pesos: es una propiedad del shim)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.requires_torch
def test_en_fp32_el_stream_residual_del_encoder_esta_MUERTO():
    """El shim devuelve None; el kernel real devuelve `x`. Una línea de diferencia.

    Ver `engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:414`:
        return (..., residual_out if residual_out is not None else x, ...)
    contra `flash_attn/ops/triton/layer_norm.py:142`:
        return y if not prenorm else (y, residual_out)

    `Block.forward` (engines/hnet/modules/block.py:115) usa el valor devuelto como
    el residual del bloque siguiente. Con None, el residual NUNCA se acumula.

    Si este test se pone en ROJO es una BUENA noticia: alguien arregló el shim.
    Cuando pase, revisar el guard de `BoundaryProbe` y re-medir la baseline.
    """
    assert residual_stream_is_alive(torch.float16) is True
    assert residual_stream_is_alive(torch.bfloat16) is True
    assert residual_stream_is_alive(torch.float32) is False, (
        "el residual en fp32 revivió: el shim de flash_attn se arregló. "
        "Re-medir la baseline: los números de fp16 y fp32 deberían converger."
    )


@pytest.mark.requires_torch
def test_el_probe_se_niega_a_correr_en_fp32_mientras_el_bug_exista():
    """No se elige el dtype por gusto: se rechaza el que computa otra función."""
    if residual_stream_is_alive(torch.float32):
        pytest.skip("el shim ya está arreglado; fp32 es legítimo")
    with pytest.raises(ValueError, match="stream residual"):
        BoundaryProbe(build_config(), device="cpu", dtype=torch.float32)


@pytest.mark.requires_torch
def test_el_bug_es_el_UNICO_lugar_donde_el_shim_se_aparta():
    """La matemática del shim (sumar el residual ANTES de normalizar) es correcta;
    lo que falta es el fallback del return. Se verifica que, PASÁNDOLE un residual,
    fp32 sí lo devuelve — o sea que el problema es el caso `residual=None`."""
    from flash_attn.ops.triton.layer_norm import RMSNorm

    norm = RMSNorm(8, eps=1e-5, dtype=torch.float32)
    x = torch.randn(1, 2, 8, dtype=torch.float32)
    res = torch.randn(1, 2, 8, dtype=torch.float32)

    _, out_con = norm(x, residual=res, prenorm=True, residual_in_fp32=True)
    assert out_con is not None
    torch.testing.assert_close(out_con, x + res)  # la suma está bien

    _, out_sin = norm(x, residual=None, prenorm=True, residual_in_fp32=True)
    assert out_sin is None, "acá está el bug: debería devolver `x`"


# ══════════════════════════════════════════════════════════════════════════════
# El camino de fronteras
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.requires_torch
def test_config_necesita_ssm_cfg_construido_a_mano():
    """`HNetConfig(**raw)` deja `ssm_cfg` como dict crudo (dataclass no valida
    tipos). Y el JSON de upstream NO trae `use_mem_eff_path`, que acá DEBE ser
    False: el camino fusionado de Mamba-2 llama a causal_conv1d => exige nvcc."""
    import json

    from hnet.models.config_hnet import HNetConfig, SSMConfig

    raw = json.loads(Path(DEFAULT_CONFIG).read_text())
    assert "use_mem_eff_path" not in raw["ssm_cfg"], "el JSON de upstream NO lo trae"
    assert not isinstance(HNetConfig(**raw).ssm_cfg, SSMConfig), "sigue siendo un dict"

    cfg = build_config()
    assert isinstance(cfg.ssm_cfg, SSMConfig)
    assert cfg.ssm_cfg.use_mem_eff_path is False


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
@requires_hnet
def test_el_camino_de_fronteras_son_28_7M_de_679M(probe):
    assert sum(p.numel() for p in probe.parameters()) == 28_764_544
    n_enc = sum(p.numel() for p in probe.encoder.parameters())
    n_rm = sum(p.numel() for p in probe.routing_module.parameters())
    assert round((n_enc + n_rm) / 1e6, 1) == 28.5  # lo que el paquete propone entrenar


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
@requires_hnet
def test_devuelve_una_probabilidad_por_byte(probe):
    data = "el órgano cortó acá".encode("utf-8")
    p = probe.boundary_prob(data)
    assert p.shape == (len(data),)
    assert p.dtype == np.float32
    assert (p >= 0).all() and (p <= 1).all()
    assert p[0] == 1.0, "el RoutingModule fuerza PAD_PROB=1.0 en la posición 0"


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
@requires_hnet
def test_en_fp16_H_Net_corta_en_palabras_y_en_fp32_no(probe):
    """La evidencia dura de que fp16 es el dtype fiel y fp32 el roto.

    fp16 : |The| quick| brown| fo|x| ju|mps| over| the la|zy| dog|.  -> 4.5 B/chunk
    fp32 : |The |q|ui|c|k |brown| fox| |j|umps |o|ve|r| |th|e| laz|y| -> 2.3 B/chunk
    """
    espacios = {i for i, b in enumerate(_ENGLISH) if b == 0x20}

    def alineacion(p):
        cortes = np.flatnonzero(p >= 0.5)
        junto = sum(1 for c in cortes if c in espacios or (c - 1) in espacios)
        return cortes.size, len(_ENGLISH) / cortes.size, junto / cortes.size

    n16, bpc16, frac16 = alineacion(probe.boundary_prob(_ENGLISH))

    roto = BoundaryProbe.allow_lossy_dtype(build_config(), device="cuda", dtype=torch.float32)
    roto.load_pretrained(DEFAULT_WEIGHTS)
    n32, bpc32, frac32 = alineacion(roto.boundary_prob(_ENGLISH))
    del roto

    # La afirmación es RELATIVA a propósito: "el 67 % de los cortes cae junto a un
    # espacio" no significa nada en abstracto (depende del texto). Lo que significa
    # algo es que fp16 se alinea con las palabras MUCHO más que fp32, y que corta la
    # mitad de veces. Un umbral absoluto acá sería un número ajustado hasta que pase.
    assert bpc16 > 4.0, f"fp16 debería chunkear a nivel palabra (~4.5 B/chunk), dio {bpc16:.2f}"
    assert bpc32 < 3.0, f"fp32 (residual muerto) debería sobre-cortar, dio {bpc32:.2f}"
    assert n32 > 1.5 * n16, f"fp32 corta {n32} y fp16 {n16}: la sobre-segmentación desapareció"
    assert frac16 - frac32 > 0.15, (
        f"fp16 alinea {frac16:.0%} de sus cortes con espacios y fp32 {frac32:.0%}: "
        "la brecha se cerró, revisar el bug del residual"
    )


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
@requires_hnet
def test_fp16_y_bf16_coinciden(probe):
    """Dos formatos de baja precisión INDEPENDIENTES dan lo mismo. Es la
    contraprueba de que el outlier es fp32, no bf16 (como decía la premisa)."""
    bf = BoundaryProbe(build_config(), device="cuda", dtype=torch.bfloat16)
    bf.load_pretrained(DEFAULT_WEIGHTS)
    p16 = probe.boundary_prob(_ENGLISH)
    pbf = bf.boundary_prob(_ENGLISH)
    del bf
    n16 = int((p16[1:] >= 0.5).sum())
    nbf = int((pbf[1:] >= 0.5).sum())
    assert abs(n16 - nbf) <= 2, f"fp16 cortó {n16} y bf16 {nbf}: no deberían separarse"
    assert np.abs(p16 - pbf).mean() < 0.02


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
@requires_hnet
def test_el_ventaneo_no_inventa_fronteras_en_los_bordes(probe):
    """El RoutingModule fuerza p=1.0 en la posición 0 de CADA ventana. Sin warmup,
    el ventaneo sembraría una frontera falsa cada `window` bytes."""
    data = b'{"k": "' + b"a" * 6000 + b'"}'
    p = probe.boundary_prob(data)
    assert p.shape == (len(data),)
    stride = probe.window - probe.warmup
    for k in range(1, len(data) // stride + 1):
        i = k * stride
        if i < len(data):
            assert p[i] < 0.999, f"frontera fabricada por el ventaneo en {i}"


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
@requires_hnet
def test_documento_vacio_no_revienta(probe):
    assert probe.boundary_prob(b"").shape == (0,)


