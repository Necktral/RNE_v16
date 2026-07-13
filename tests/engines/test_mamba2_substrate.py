"""El sustrato de Mamba-2 corre SIN nvcc y SIN extensiones CUDA compiladas.

Contexto (medido en RTX 2070 Max-Q, Turing sm_75, 2026-07-12):

RNFE vendoriza `engines/mamba_vendor/mamba_ssm` pero NUNCA pudo usarlo, y el diagnóstico
obvio ("hace falta nvcc / hace falta una GPU más nueva") era FALSO.  Lo que bloqueaba era:

1. El shim ``mamba_ssm/__init__.py`` —archivo DEL PROYECTO, no del vendor— re-exportaba
   ansiosamente cuatro símbolos, tres de los cuales RNFE no usa:
     - ``selective_scan_fn`` / ``Mamba``  → Mamba-**1** ⇒ extensión CUDA ``selective_scan_cuda``
       ⇒ **requiere nvcc**.
     - ``MambaLMHeadModel``               → el LM ⇒ ``transformers.generation`` con símbolos
       **eliminados** en transformers >= 5.
   Un solo ``import mamba_ssm`` reventaba y se llevaba puesto a **todo Mamba-2**.

2. ``Mamba2(use_mem_eff_path=True)`` (el default) toma el camino rápido, que llama a
   ``causal_conv1d_fwd_function`` — OTRA extensión CUDA.  El camino lento
   (``use_mem_eff_path=False``) ya tiene fallback a ``F.conv1d`` de torch puro.

El kernel SSD de Mamba-2 (``ops/triton/ssd_combined.py``) es **Triton PURO**: compila en
runtime, no necesita toolkit, y corre en Turing.  Medido: 1.16 ms/forward en estado estable
(la primera llamada paga ~5 s de JIT), 1.7 GB de VRAM.

Estos tests PINEAN esa verdad.  Si alguien vuelve a poner imports ansiosos en el shim, o
alguien asume que hace falta nvcc y borra el vendor, esto se pone en rojo.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="el sustrato neural necesita torch")

requiere_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requiere GPU CUDA (el kernel SSD es Triton sobre GPU)",
)


def test_import_mamba_ssm_no_requiere_nvcc() -> None:
    """`import mamba_ssm` NO debe arrastrar Mamba-1 ni el LM.

    Éste es el test que importa: antes del shim perezoso, esta línea sola era un
    ModuleNotFoundError('selective_scan_cuda') y dejaba Mamba-2 inalcanzable.
    """
    import mamba_ssm  # noqa: F401


def test_kernel_ssd_de_mamba2_es_importable_sin_toolkit() -> None:
    """El kernel SSD es Triton puro: importable sin extensión CUDA compilada."""
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    assert callable(mamba_chunk_scan_combined)


def test_simbolos_de_mamba1_fallan_SOLO_al_pedirlos_y_explican_por_que() -> None:
    """La ausencia de nvcc debe ser un fallo LOCAL, no global.

    `selective_scan_fn` (Mamba-1) sigue accesible por nombre; sólo revienta si alguien lo
    pide de verdad, y el mensaje tiene que decir qué falta y que Mamba-2 no lo necesita.
    """
    import mamba_ssm

    assert "selective_scan_fn" in dir(mamba_ssm)  # el contrato público no cambió
    with pytest.raises(ImportError) as exc:
        _ = mamba_ssm.selective_scan_fn
    mensaje = str(exc.value)
    assert "nvcc" in mensaje
    assert "ssd_combined" in mensaje  # le dice al lector dónde SÍ está Mamba-2


@requiere_cuda
def test_kernel_ssd_corre_en_gpu_y_da_numeros_finitos() -> None:
    """El SSD de Mamba-2 CORRE en la GPU (incl. Turing sm_75), sin nvcc.

    OJO con `A`: en Mamba es SIEMPRE negativa (`A = -exp(A_log)`).  Con `A` positiva la
    recurrencia diverge y la salida sale NaN/Inf — eso sería un bug DEL TEST, no del kernel.
    """
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    batch, seqlen, nheads, headdim, ngroups, dstate = 1, 256, 4, 32, 1, 16
    for dtype in (torch.float16, torch.float32):
        salida = mamba_chunk_scan_combined(
            x=torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype),
            dt=torch.rand(batch, seqlen, nheads, device="cuda", dtype=torch.float32),
            A=-torch.exp(torch.randn(nheads, device="cuda", dtype=torch.float32)),
            B=torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype),
            C=torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype),
            chunk_size=64,
            D=torch.randn(nheads, device="cuda", dtype=torch.float32),
        )
        assert tuple(salida.shape) == (batch, seqlen, nheads, headdim)
        assert torch.isfinite(salida).all(), f"el kernel SSD dio NaN/Inf en {dtype}"


@requiere_cuda
def test_capa_mamba2_corre_por_el_camino_sin_causal_conv1d() -> None:
    """La capa Mamba2 completa corre con `use_mem_eff_path=False`, sin extensiones CUDA.

    El default (`True`) exige `causal_conv1d`, que necesita nvcc.  El camino lento cae a
    `F.conv1d` de torch puro y funciona.  Medido: 1.16 ms/forward en estado estable.
    """
    from mamba_ssm.modules.mamba2 import Mamba2

    capa = Mamba2(d_model=128, d_state=16, headdim=32, use_mem_eff_path=False)
    capa = capa.to("cuda").to(torch.float16)
    entrada = torch.randn(1, 128, 128, device="cuda", dtype=torch.float16)

    salida = capa(entrada)

    assert tuple(salida.shape) == (1, 128, 128)
    assert torch.isfinite(salida).all()
