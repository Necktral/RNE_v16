# RNFE External Reasoner Replication

Este documento deja replicable el estado validado de `EXT_OPEN_THINKER` como resolver externo gated para conflicto causal/contrafactual. No versiona pesos, builds locales ni caches.

## 1. Requisitos

- Windows 11 + WSL2.
- Ubuntu en WSL.
- GPU NVIDIA visible desde WSL con `nvidia-smi`.
- Espacio recomendado en `D:\rnfe_models` para modelos, caches y builds.
- Python 3.10; el entorno actual fue validado con Python 3.10.14.
- Git.
- CMake y build tools (`cmake`, `build-essential`, `pkg-config`).
- Entorno virtual Python para RNFE.

## 2. Layout Esperado

En Windows:

```text
D:\rnfe_models
```

En WSL:

```text
/mnt/d/rnfe_models
```

Subdirectorios:

```text
/mnt/d/rnfe_models/local/
/mnt/d/rnfe_models/gguf/
/mnt/d/rnfe_models/tools/
/mnt/d/rnfe_models/cache/
/mnt/d/rnfe_models/manifests/
/mnt/d/rnfe_models/logs/
/mnt/d/rnfe_models/scripts/
```

Crear el layout:

```bash
mkdir -p /mnt/d/rnfe_models/{local,gguf,tools,cache,manifests,logs,scripts}
mkdir -p /mnt/d/rnfe_models/cache/{huggingface,torch,pip}
```

## 3. Variables De Entorno

Usar `.env.external_reasoner.example` como plantilla. Ajustar rutas segun usuario, venv y ubicacion real del CUDA runtime.

```bash
export RNFE_MODELS_ROOT=/mnt/d/rnfe_models
export RNFE_REASONING_GGUF=/mnt/d/rnfe_models/gguf/OpenThinker3-7B/OpenThinker3-7B-Q4_K_M.gguf
export RNFE_LLAMA_CLI_CUDA=/mnt/d/rnfe_models/tools/llama.cpp/build-cuda/bin/llama-cli
export RNFE_LLAMA_CLI_CPU=/mnt/d/rnfe_models/tools/llama.cpp/build-cpu/bin/llama-cli

export HF_HOME=/mnt/d/rnfe_models/cache/huggingface
export HF_HUB_CACHE=/mnt/d/rnfe_models/cache/huggingface/hub
export TORCH_HOME=/mnt/d/rnfe_models/cache/torch
export PIP_CACHE_DIR=/mnt/d/rnfe_models/cache/pip

export CUDA_ROOT=/home/necktral/.venvs/rnfe-reasoning-models/lib/python3.10/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:/usr/lib/wsl/lib:/mnt/d/rnfe_models/tools/llama.cpp/build-cuda/bin:$LD_LIBRARY_PATH"
```

`CUDA_ROOT` puede cambiar segun nombre de usuario, version de Python, nombre del venv o si se instala CUDA Toolkit del sistema.

## 4. Restaurar Modelo

Modelo validado:

- `open-thoughts/OpenThinker3-7B`
- GGUF: `OpenThinker3-7B-Q4_K_M.gguf`
- cuantizacion: `Q4_K_M`
- SHA256 esperado: `0b7344e4bf1c68fc40c4a10b14b9bd51f367423b8453d83544ea5bdbe08e7e5e`

### Ruta A: copiar desde backup externo

Copiar:

```text
OpenThinker3-7B-Q4_K_M.gguf
```

a:

```text
/mnt/d/rnfe_models/gguf/OpenThinker3-7B/
```

Verificar:

```bash
sha256sum /mnt/d/rnfe_models/gguf/OpenThinker3-7B/OpenThinker3-7B-Q4_K_M.gguf
```

Debe coincidir con:

```text
0b7344e4bf1c68fc40c4a10b14b9bd51f367423b8453d83544ea5bdbe08e7e5e
```

### Ruta B: reconstruir desde Hugging Face

Usar esta ruta solo si no existe backup del GGUF.

1. Descargar `open-thoughts/OpenThinker3-7B` hacia `RNFE_MODELS_ROOT`, no al repo.
2. Convertir a GGUF F16 con las herramientas de `llama.cpp`.
3. Cuantizar a `Q4_K_M`.
4. Verificar tamano y hash.
5. Si el hash difiere por version de modelo o herramienta, actualizar `docs/setup/external_reasoner_state_manifest.json` y repetir smoke/repetibilidad antes de admitir el nuevo artefacto.

No versionar el modelo ni los pesos originales.

## 5. Compilar llama.cpp CUDA

Instalar dependencias base:

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
```

Si no hay `nvcc` del sistema, instalar el paquete CUDA compatible con WSL o usar el CUDA runtime del venv. Evitar instalar drivers Linux completos si WSL ya ve la GPU via driver Windows.

Clonar o actualizar `llama.cpp` fuera del repo RNFE:

```bash
cd /mnt/d/rnfe_models/tools
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Build CPU fallback:

```bash
cmake -B build-cpu
cmake --build build-cpu --config Release -j
```

Build CUDA:

```bash
export CUDA_ROOT=/home/necktral/.venvs/rnfe-reasoning-models/lib/python3.10/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j
```

Verificar:

```bash
/mnt/d/rnfe_models/tools/llama.cpp/build-cuda/bin/llama-cli --help | grep -- --json-schema-file
```

## 6. Smoke Test

Desde el repo RNFE, con env cargado:

```bash
python scripts/benchmark_external_reasoner_gain.py \
  --campaign-id smoke-replication \
  --output-root /tmp/rnfe_external_reasoner_replication \
  --episodes 1 \
  --backend cuda \
  --profiles core_plus_external_reasoner_gated_v1 \
  --regimes causal_counterfactual_conflict
```

Resultado minimo esperado:

- `external_reasoner_ok_rate = 1.0`
- `schema_validated_rate = 1.0`
- `invalid_intervention_accepted = 0`
- salida estructurada validada por schema
- fallback core intacto si gate/guard/schema fallan

## 7. Tests Minimos Post-Clone

```bash
pytest tests/regression/test_external_reasoner_profiles.py \
  tests/benchmarks/test_external_reasoner_gain.py \
  tests/regression/test_meta_scheduler_family_profiles.py \
  tests/regression/test_meta_scheduler_policy_units.py \
  tests/regression/test_ext_open_thinker_adapter.py \
  tests/regression/test_external_reasoner_client.py -q
```

Resultado esperado del checkpoint:

```text
46 passed
```

## 8. Verificacion De Replicacion

```bash
python scripts/check_external_reasoner_replication.py
```

El script imprime cada requisito y sale con:

- `0` si el entorno esta listo.
- `1` si faltan variables, binarios, schema, GPU visible, hash correcto, o si aparecen pesos nuevos dentro del repo.

El script puede reportar advertencia por pesos historicos ya trackeados antes del checkpoint; no debe admitir pesos nuevos o no versionados.

## 9. Estado Validado

- componente: `EXT_OPEN_THINKER`
- estado: `external_conflict_resolver_governed`
- regimen validado: `causal_counterfactual_conflict`
- dictamen experimental: `conflict_resolver_repetible`
- `latency_mean_s = 96.115`
- `latency_p95_s = 98.953`
- `corrected_core_failure_rate = 0.875`
- runtime nominal: no activa `EXT_OPEN_THINKER`
