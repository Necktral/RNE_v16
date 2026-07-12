# ADR: integración simbiótica neuronal N0–N6

## Estado

Implementado en `integration/symbiotic-organism-v1`. La evidencia normativa sigue
en el canon; este ADR describe solamente el cableado ejecutable y sus límites.

## Decisión

`ScenarioEpisodeRunner` llama una única frontera,
`SymbioticNeuralCoordinator`. El coordinador usa N0 para ejecutar referencias CPU,
propaga una identidad causal común y transforma resultados en evidencia. No agenda
familias, selecciona intervenciones, certifica, promociona memoria ni aplica cambios.

El contrato `neural-symbiosis-trace-v1` enlaza identidad, input hash, candidato,
consumidor, veredicto del consumidor, fallback, coste, resultado y certificado. Se
persiste en el ledger existente; un fallo de storage queda en el health/buffer de N0.

## Flujo vivo

1. N5 segmenta observación, fórmula y memoria autorizada con offsets deterministas;
   el runner materializa candidatos en SMG y los deja en el contexto que MFM condensa.
2. N3 actualiza estado con clave `(organism_id, scenario_id, lineage_id)` y su salida
   entra al siguiente contexto, a MFM candidata y a auditoría de continuidad.
3. N1 propone familias; el scheduler conserva la selección autoritativa. Después se
   registra propuesta, selección, intersección, recompensa y certificado.
4. N2 produce una proposición y la somete a los resultados reales de DED, LOT-F y
   NESY. Solo queda evidencia shadow aceptada o rechazada.
5. N4 formula una relación causal tipada desde el snapshot vivo y se compara con
   CAU, CTF y C-GWM. El desacuerdo llega a certificación y experiencia sin autoridad.
6. N6 propone o se abstiene según viabilidad/coste, pasa por evaluación sandbox y
   siempre conserva `applied=false`; no existe `apply_fn` en esta frontera.

## Censo de código vivo

| Componente | Clasificación | Caller vivo | Consumidor vivo |
|---|---|---|---|
| N0 | LIVE | ScenarioEpisodeRunner/coordinador | N1–N6 y trace health |
| N1 | SHADOW_CONSUMED | coordinador | comparación scheduler + resultado |
| N2 | SHADOW_CONSUMED | coordinador | DED + LOT-F + NESY + certificación |
| N3 | LIVE | coordinador | siguiente episodio + MFM + continuidad |
| N4 | SHADOW_CONSUMED | coordinador | CAU/CTF/C-GWM + certificación + experiencia |
| N5 | LIVE | coordinador | SMG + MFM + reasoning context |
| N6 | SHADOW_CONSUMED | coordinador | sandbox + certificación + autoevolution evidence |
| NESY | SHADOW_CONSUMED | verificador N2 / scheduler deep | N2 trace / PROB |
| EVO_SEARCH | REFERENCE_ONLY | scheduler deep | PROB |
| IMAGINATION/A11 | REFERENCE_ONLY | scheduler deep | A12 / PROB |
| A12 | REFERENCE_ONLY | scheduler/guard gated | override guard |
| CAU, CTF, DED | LIVE | MetaScheduler | reasoning, N2/N4, certificación |
| C-GWM | LIVE | ScenarioEpisodeRunner | transición factual y comparador N4 |
| LOT-F | LIVE | ScenarioEpisodeRunner | DED, N2 y certificación |
| MFM, SMG | LIVE | runner / PromotionGate | reasoning, memoria y certificación |
| scheduler | LIVE | ScenarioEpisodeRunner | intervención gobernada y certificación |
| certification | LIVE | ScenarioEpisodeRunner | promoción, experiencia y autoevolución |
| experience | LIVE cuando está habilitado | ScenarioEpisodeRunner | sesgo futuro |
| autoevolution | LIVE | ScenarioEpisodeRunner | linaje y knobs gobernados |

La matriz ejecutable y el gate de callers/consumidores viven en
`runtime/neural/integration/census.py`; documentación no se usa como prueba.

## Backends retirados de perfiles activos

- H-Net: reference-only hasta artefacto, hash, licencia y evidencia real.
- Mamba2: deshabilitado; N3 se declara `reference_temporal_filter`.
- MLP N1 sin entrenamiento: no dirige el scheduler; se usa una propuesta heurística
  determinista y declarada.
- N4 message-passing sin entrenamiento: reference-only; el camino vivo usa una
  propuesta contractual determinista, nunca un claim predictivo aprendido.

## Autoridad

- N1, N2, N4 y N6 tienen techo SHADOW.
- Integración no implica promoción.
- `neural_symbiosis` es metadata aditiva y no entra en `verdict`, `risk_score` ni
  `promotion_candidate`.
- OFF no llama productores ni adquiere artefactos.
- La presión alta conserva N5/N3 CPU y degrada N1/N2/N4/N6 de forma visible.

## Consolidación con MSRC

MSRC sigue siendo la única autoridad de escala. `LifeKernel` transmite el mismo
snapshot normalizado que reciben vitales y reasoning, añadiendo únicamente
`msrc_budget_available` y `msrc_scale_id`. El coordinador no elige escala ni define
umbrales; N0 aplica su presupuesto neuronal configurado.

Si MSRC falla, el siguiente episodio recibe `msrc_budget_available=false` y N0 no
ejecuta ningún órgano sin presupuesto. Una recuperación posterior de MSRC reabre el
presupuesto sin intervención manual. Las decisiones y transiciones MSRC incorporan
el mismo `run_id`, `episode_id` y `trace_group_id` que la traza simbiótica.

El checkpoint soberano incluye `n3-temporal-checkpoint-v1`: únicamente estado
determinista y claves `(organism_id, scenario_id, lineage_id)`, nunca pesos, modelos o
buffers. Esto conserva continuidad N3 entre corridas con nuevo `run_id`.

## Clasificación del gate estático anti-stub

La búsqueda obligatoria `return .*idle|if False|pass$|NotImplemented` no encuentra
coincidencias en el coordinador ni en sus perfiles activos. Las coincidencias restantes
se clasifican así:

- `runtime/legacy/*`: DEAD respecto al runtime vivo; no tiene importadores fuera de
  `runtime/legacy` y no forma parte del perfil simbiótico.
- `imagination` y `a12`: REFERENCE_ONLY en baseline; sus `idle` son kill-switches de
  overlays deep no activados, no órganos contados como vivos.
- `EpistemeMeter.apply_noise` y `Hook.__call__`: errores explícitos para capacidades no
  implementadas/abstractas; no tienen caller en el flujo simbiótico.
- `FormulaNode` y `BackendRegistryError`: clases base/excepción sin cuerpo; no producen
  outputs de órgano.
- Los `pass` restantes están exclusivamente en manejadores de excepción best-effort
  (`host_sampler`, continuidad, teacher, ecology, experience, storage config) y no son
  retornos sintéticos de un órgano.

No queda `if False` en `ScenarioEpisodeRunner`; `trajectory_window` usa ahora la ventana
real de trayectoria.
