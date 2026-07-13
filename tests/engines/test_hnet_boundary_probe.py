"""Tests del camino de fronteras de H-Net (embeddings -> encoder -> RoutingModule).

Requieren GPU y los pesos preentrenados; se saltan solos si no están.  El test
que hay que mirar es `test_bf16_destruye_el_chunker`: NO es una curiosidad, es la
razón por la que el evaluador exige fp32.  Si alguien mide el chunker en bf16,
mide ruido y lo reporta como si fuera el modelo.
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
)

pytestmark = [
    pytest.mark.requires_torch,
    pytest.mark.requires_cuda,
    pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="pesos de H-Net no presentes"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="sin GPU"),
]


@pytest.fixture(scope="module")
def probe():
    p = BoundaryProbe(build_config(), device="cuda", dtype=torch.float32)
    p.load_pretrained(DEFAULT_WEIGHTS)
    return p


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


def test_el_camino_de_fronteras_son_28_7M_de_679M(probe):
    n = sum(p.numel() for p in probe.parameters())
    assert n == 28_764_544
    n_enc = sum(p.numel() for p in probe.encoder.parameters())
    n_rm = sum(p.numel() for p in probe.routing_module.parameters())
    assert round((n_enc + n_rm) / 1e6, 1) == 28.5  # lo que el paquete propone entrenar


def test_devuelve_una_probabilidad_por_byte(probe):
    data = "el órgano cortó acá".encode("utf-8")
    p = probe.boundary_prob(data)
    assert p.shape == (len(data),)
    assert p.dtype == np.float32
    assert (p >= 0).all() and (p <= 1).all()
    assert p[0] == 1.0, "el RoutingModule fuerza PAD_PROB=1.0 en la posición 0"


def test_fp32_es_obligatorio():
    with pytest.raises(ValueError, match="fp32"):
        BoundaryProbe(build_config(), device="cuda", dtype=torch.bfloat16)


def test_bf16_destruye_el_chunker():
    """MEDIDO, no supuesto. `boundary_prob = (1 - cos_sim)/2`; en texto homogéneo
    cos_sim ~ 0.99 y los 8 bits de mantisa de bf16 no resuelven la resta. El
    chunker no queda "peor calibrado": queda ROTO — corta en TODAS las posiciones.
    """
    cfg = build_config()
    data = b"a" * 24

    p32 = BoundaryProbe(cfg, device="cuda", dtype=torch.float32)
    p32.load_pretrained(DEFAULT_WEIGHTS)
    cuts32 = int((p32.boundary_prob(data)[1:] >= 0.5).sum())
    del p32

    pbf = BoundaryProbe.allow_lossy_dtype(cfg, device="cuda", dtype=torch.bfloat16)
    pbf.load_pretrained(DEFAULT_WEIGHTS)
    cutsbf = int((pbf.boundary_prob(data)[1:] >= 0.5).sum())
    del pbf

    assert cuts32 <= 8, f"fp32 debería cortar poco en texto homogéneo, cortó {cuts32}/23"
    assert cutsbf >= 20, f"bf16 debería cortar en (casi) todo, cortó {cutsbf}/23"
    assert cutsbf > 2 * cuts32, "si esta relación se cae, la premisa del dtype cambió"


def test_el_ventaneo_no_inventa_fronteras_en_los_bordes(probe):
    """El RoutingModule fuerza p=1.0 en la posición 0 de CADA ventana. Sin warmup,
    el ventaneo sembraría una frontera falsa cada `window` bytes."""
    data = (b'{"k": "' + b"a" * 6000 + b'"}')
    p = probe.boundary_prob(data)
    assert p.shape == (len(data),)
    # la única posición con p == 1.0 exacto debe ser la 0 del documento
    unos = np.flatnonzero(p >= 0.999)
    assert unos.tolist() == [0] or all(u == 0 for u in unos[:1])
    # no hay un pico sistemático en los múltiplos del stride
    stride = probe.window - probe.warmup
    for k in range(1, len(data) // stride + 1):
        i = k * stride
        if i < len(data):
            assert p[i] < 0.999, f"frontera fabricada por el ventaneo en {i}"


def test_documento_vacio_no_revienta(probe):
    assert probe.boundary_prob(b"").shape == (0,)
