# 12 — `engines/mamba_vendor/` (Mamba SSM vendoreado — terceros)

~9.5K LOC. Es el paquete **Mamba SSM oficial** (`state-spaces/mamba`, Albert Gu & Tri Dao)
**vendoreado verbatim**. No contiene lógica del proyecto.

## Contenido (estructura estándar upstream)
- `mamba_ssm/modules/`: `mamba_simple.py`, `mamba2.py`, `mamba2_simple.py`, `block.py`, `mha.py`,
  `mlp.py`, `ssd_minimal.py`.
- `mamba_ssm/ops/triton/`: kernels Triton — `ssd_combined.py` (998), `ssd_chunk_scan.py` (1834),
  `ssd_chunk_state.py` (997), `layer_norm.py` (1113), `layernorm_gated.py`, `ssd_bmm.py`,
  `selective_state_update.py`, `k_activations.py`, `softplus.py`, `ssd_state_passing.py`.
- `mamba_ssm/models/`: `mixer_seq_simple.py`, `config_mamba.py`.
- `mamba_ssm/distributed/`: `tensor_parallel.py`, `distributed_utils.py`.
- `mamba_ssm/utils/`: `generation.py`, `hf.py`, `torch.py`.
- `csrc/selective_scan/`: 9 kernels CUDA (`selective_scan_{fwd,bwd}_{fp16,fp32,bf16}_{real,complex}.cu`).

## Única modificación del proyecto
- **`patches/rocm6_0.patch`** (2305 bytes): parche de compatibilidad **ROCm 6.0** (AMD). Es la
  aportación propia del repo sobre el vendor (adaptar Mamba a ROCm además de CUDA).

## Uso
Importado **solo** por `engines/hnet/modules/{dc,block}.py` (Mamba2 como bloque de H-Net). **No lo
usa ningún módulo de `runtime/`** → es una dependencia transitiva de H-Net, ajena al organismo AEON.

## Hallazgos / tratamiento
- **[TERCEROS]** Auditar línea por línea los kernels Triton/CUDA de selective-scan/SSD equivale a
  revisar Mamba upstream — fuera del alcance útil para este proyecto. Lo relevante es:
  1. Es **terceros vendoreado**; preservar licencia/atribución (Apache-2.0 de Mamba).
  2. La diferencia frente a upstream a vigilar es `patches/rocm6_0.patch` (lo único propio).
  3. Acopla el repo a un toolchain CUDA/ROCm + Triton (compilación de kernels); irrelevante salvo
     que se ejecute H-Net.

## Veredicto
Dependencia de terceros pura (Mamba SSM) con un parche ROCm propio; sin lógica del organismo. Se
documenta su presencia, procedencia y el parche; no procede análisis de corrección como código del
proyecto.
