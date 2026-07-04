# 11 — `engines/hnet/` (modelo H-Net — terceros adaptado)

~2.5K LOC. Implementación de **H-Net** (modelo de secuencia jerárquico con *dynamic chunking*,
paper Goombalab 2025) más módulos de **flash-attn (Tri Dao)**. **No es código original del
proyecto**, y está **aislado del pipeline AEON** (razonamiento/episodios). Solo lo consume la
herramienta `exocortex/tools/generate_hnet.py` (la entrada histórica `generate.py`).

## Evidencia de origen tercero
- `modules/rotary.py` y `modules/mha.py`: cabecera **"Copyright (c) 2023, Tri Dao."** (copiados de
  flash-attn).
- Imports absolutos `from hnet.modules...` (dependen del shim raíz `hnet/` → `engines.hnet`).
- Dependencias externas pesadas: `flash_attn`, `mamba_ssm` (el vendoreado), `einops`,
  `causal_conv1d` (implícito). Configs oficiales de release: `hnet_{1,2}stage_{L,XL}`,
  `_XL_chinese`, `_XL_code`.

## Estructura
- `models/hnet.py` (`HNet`): jerarquía recursiva con `STE` (straight-through estimator para
  fronteras de chunk), encoder/decoder por etapa.
- `modules/dc.py`: la **novedad de H-Net** — `RoutingModule` (predice fronteras), `ChunkLayer`
  (comprime), `DeChunkLayer` (descomprime). Dynamic chunking.
- `modules/isotropic.py` (`Isotropic`): backbone de bloques homogéneos.
- `modules/block.py`: `Mamba2Wrapper(Mamba2)` + `create_block` (Mamba2 / MHA / MLP).
- `modules/mha.py` (404), `modules/rotary.py` (683), `modules/mlp.py`: atención/rotary/FFN (Tri Dao).
- `models/mixer_seq.py` (`HNetForCausalLM` con `GenerationMixin`): el LM completo.
- `utils/tokenizers.py` (`ByteTokenizer`), `utils/train.py`, `configs/*.json`.

## Hallazgos
- **[DISEÑO/DEPENDENCIA]** Requiere `flash_attn`, `mamba_ssm`, `causal_conv1d`, `einops` — toolchain
  CUDA pesado. En un entorno sin estas libs, importar `HNetForCausalLM` falla; `generate_hnet.py`
  no es ejecutable sin GPU + dependencias compiladas.
- **[DISEÑO] Aislado del organismo AEON.** No participa en `runtime/*`; es una capacidad de
  generación de texto independiente. El "razonador externo" del pipeline NO es H-Net (es
  OpenThinker vía llama.cpp; ver [07_reasoning.md](07_reasoning.md)).
- **Atribución:** al ser código de terceros adaptado (H-Net + flash-attn), conviene preservar las
  cabeceras de copyright y documentar la procedencia (licencias H-Net / flash-attn).

## Veredicto
Código de investigación de terceros (H-Net + flash-attn de Tri Dao) integrado como motor de
generación opcional, **desacoplado del sistema cognitivo**. No requiere análisis de corrección como
código propio; sí requiere claridad de **procedencia/licencia** y de que es una pieza separada del
"organismo". Análisis línea-por-línea de su corrección equivaldría a auditar H-Net/flash-attn
upstream, fuera del alcance útil para este proyecto.
