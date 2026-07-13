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
def test_el_stream_residual_esta_VIVO_en_los_TRES_dtypes():
    """El stream residual NO se corta. En ninguna precisión. Nunca.

    ─── Este test estaba AL REVÉS, y el alambre trampa funcionó ──────────────────────
    La versión anterior afirmaba que en fp32 el residual estaba MUERTO, y cerraba con:
    *"si este test se pone en ROJO es una BUENA noticia: alguien arregló el shim"*.

    Se puso en rojo. Alguien lo arregló. `flash_attn/ops/triton/layer_norm.py` ahora
    replica el fallback del kernel real (`mamba_ssm/ops/triton/layer_norm.py:414`):

        residual_out if residual_out is not None else x     # <-- CAE A `x`

    `store_residual_out` decide si se ASIGNA un tensor nuevo; el return **siempre** entrega
    el stream residual. El shim copiaba la condición de asignación y se comía el fallback
    ⇒ en fp32 devolvía `None` ⇒ `Block.forward` perdía el residual de las 4 capas Mamba.

    Contraprueba ejecutada: con el fix, **fp32 y fp16 segmentan idéntico byte a byte**.
    La matemática correcta en precisión completa no puede diferir de la reducida.
    """
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert residual_stream_is_alive(dtype) is True, (
            f"REGRESIÓN GRAVE: el stream residual volvió a cortarse en {dtype}. "
            "Revisar el fallback del return en flash_attn/ops/triton/layer_norm.py."
        )


@pytest.mark.requires_torch
def test_el_probe_se_niega_a_correr_en_fp32_mientras_el_bug_exista():
    """No se elige el dtype por gusto: se rechaza el que computa otra función."""
    if residual_stream_is_alive(torch.float32):
        pytest.skip("el shim ya está arreglado; fp32 es legítimo")
    with pytest.raises(ValueError, match="stream residual"):
        BoundaryProbe(build_config(), device="cpu", dtype=torch.float32)


@pytest.mark.requires_torch
def test_el_shim_devuelve_el_residual_con_y_sin_residual_de_entrada():
    """Los dos casos del return, que es donde estaba el bug.

    CON residual de entrada  -> devuelve `x + residual` (la suma siempre estuvo bien).
    SIN residual de entrada  -> devuelve `x`            (esto era lo que devolvía None).
    """
    from flash_attn.ops.triton.layer_norm import RMSNorm

    norm = RMSNorm(8, eps=1e-5, dtype=torch.float32)
    x = torch.randn(1, 2, 8, dtype=torch.float32)
    res = torch.randn(1, 2, 8, dtype=torch.float32)

    _, out_con = norm(x, residual=res, prenorm=True, residual_in_fp32=True)
    assert out_con is not None
    torch.testing.assert_close(out_con, x + res)

    _, out_sin = norm(x, residual=None, prenorm=True, residual_in_fp32=True)
    assert out_sin is not None, "REGRESIÓN: volvió a devolver None (el bug del residual)"
    torch.testing.assert_close(out_sin, x)  # sin residual de entrada, el stream ES `x`


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
def test_fp32_y_fp16_dan_la_MISMA_segmentacion(probe):
    """LA CONTRAPRUEBA del fix del residual: precisión completa ≡ precisión reducida.

    Antes del fix, fp32 sobre-segmentaba (2.3 B/chunk) y fp16 cortaba en palabras
    (4.5 B/chunk). Se lo leyó como "fp32 es más preciso y ve más fronteras" — y era al
    revés: **fp32 corría un encoder SIN conexiones residuales**, o sea otra función.

    Con el residual restaurado los dos dan `The| quick| brown| fo|x| ju|mps| over| the la|zy|
    dog|.` — **byte a byte**. Si este test se pone en rojo, el residual se volvió a cortar
    (o alguien tocó el chunker), y la baseline hay que re-medirla.

    La afirmación es de IDENTIDAD, no de umbral: no hay ningún número ajustado hasta que pase.
    """
    p16 = probe.boundary_prob(_ENGLISH)
    cortes16 = set(np.flatnonzero(p16 >= 0.5).tolist())

    fp32 = BoundaryProbe(build_config(), device="cuda", dtype=torch.float32)
    fp32.load_pretrained(DEFAULT_WEIGHTS)
    cortes32 = set(np.flatnonzero(fp32.boundary_prob(_ENGLISH) >= 0.5).tolist())
    del fp32

    assert cortes32 == cortes16, (
        f"fp32 y fp16 segmentan distinto — fp32 corta en {sorted(cortes32)}, "
        f"fp16 en {sorted(cortes16)}. La matemática correcta en precisión completa NO puede "
        "diferir de la reducida: revisar el fallback del residual en flash_attn."
    )
    # Y que el corte sea a nivel PALABRA, no a nivel byte (lo que hacía el modelo roto).
    bytes_por_chunk = len(_ENGLISH) / max(len(cortes16), 1)
    assert bytes_por_chunk > 3.5, (
        f"chunkea a {bytes_por_chunk:.2f} B/chunk: eso es sub-palabra. "
        "El modelo sano corta a ~4.5 B/chunk en prosa inglesa."
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


