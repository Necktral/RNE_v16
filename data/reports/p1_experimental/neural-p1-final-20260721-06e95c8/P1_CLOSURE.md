# P1 canonical closure

## 1. Identidad del experimento

- Campaign: `neural-p1-final-20260721-06e95c8`
- Experiment commit: `06e95c8f45c132be87f81f03c19a966674dfb51b`
- Evidence publication HEAD: `060101975a752793c4b5bae6872002ce69c0f8ec`
- N3 attribution audit HEAD: `d2d54a649309e54d9cc434871e8a4e40c6325ad0`

## 2. Objetivo y alcance de P1

P1 midió atribución cognitiva en SHADOW. Este cierre es derivado, no repite la
campaña, no modifica runtime y no diseña ni autoriza P2.

## 3. Evidencia y cadena de commits

Se preservan `matrix.json`, `matrix.audit-v2.json` y
`n3-attribution.audit-v1.json`, ligados por SHA-256 en `SHA256SUMS`.

## 4. Integridad y reproducibilidad

Diez perfiles, doce seeds por perfil y 32 pasos por lane. Cerraron 3.250/3.250
episodios; certificación 1.0; cero violaciones de seguridad; 3.456 comparaciones
canónicas y cero mismatches. JSON y Markdown se generan offline y atómicamente.

## 5. Resultado N2

**FAILED.** `retry_false_accepts=0`,
`valid_corrections=0` y
`final_false_rejections=70`. La política de segunda
verificación fue segura pero no demostró utilidad.

## 6. Resultado N3

**SUPPORTED_LIMITED.** La señal aislada
`paired_binary_normalized_dcg_delta_v1` tiene media `0.01835533397030496`, IC95%
`[0.014940696408712381, 0.02188782420647339]`, 12 seeds positivas y p exacto
`0.00048828125`. Los tres contrastes contextuales son cero en
las doce seeds.

N3 demuestra una contribución cognitiva limitada y reproducible dentro del
experimento P1 SHADOW.

La señal aislada mejora la métrica pareada de ranking interno frente al
retrieval canónico y permanece inalterada al habilitar o retirar N2 y N4.

El backend trained mejora sustancialmente el Brier score frente al backend
reference, pero no demuestra superioridad global: la ventaja de ranking
trained-reference es inconclusa, MRR no mejora y balanced accuracy empeora.

P1 no demuestra influencia sobre decisión, memoria, scheduler, actuación,
certificación ni comportamiento canónico.

## 7. Resultado N4

**FAILED.** Cobertura `0.917`, top-1
`1.0`, cobertura aislada `0.0`. N4 no
demostró valor incremental sobre el prior causal y permanece rechazado.

## 8. Resolución semántica trained vs reference

- Ranking: **INCONCLUSIVE**, delta `0.00451492140564504`,
  4 seeds positivas, 8 cero, p exacto `0.125`.
- Brier score: **SUPPORTED**, mejora emparejada
  `0.3178849374622885`.
- MRR: **NOT_SUPPORTED**, delta `0.0`.
- Balanced accuracy: **REFUTED**, delta
  `-0.2451402185777186`.
- Superioridad global: **NOT_DEMONSTRATED**.

El gate fuente ambiguo `trained_vs_reference` queda deprecado y no aparece como
gate final.

## 9. Limitaciones científicas

- No puede recomputarse nDCG convencional.
- No puede descomponerse Brier en reliability, resolution y uncertainty.
- No hubo mejora de MRR y balanced accuracy empeoró.
- No hubo influencia canónica ni evaluación live/OOD suficiente.
- XPASS externo no bloqueante:
  `tests/reasoning_stress/test_temporal_hysteresis.py::test_no_undesired_memory_effects`;
  caracteriza histéresis/discretización del scheduler y no pertenece a P1.

## 10. Inferencias prohibidas

- AGI
- general intelligence
- operational autonomy
- autopoiesis
- effective self-evolution
- improved final decision
- improved actuation
- improved scheduler
- top-rank improvement
- conventional nDCG
- decomposed calibration
- trained-reference global superiority
- sufficient OOD robustness
- live safety
- production readiness
- P2 authorization
- main merge authorization

## 11. Decisiones de autoridad

Influencia canónica: `none`. Autoridad live, staging, promoción, merge a `main` y
P2 permanecen explícitamente no autorizados.

## 12. Estado final de P1

`P1_STATUS=CLOSED`; `P1_N2=FAILED`; `P1_N3=SUPPORTED_LIMITED`;
`P1_N4=FAILED`.

## 13. Condición para seleccionar un nuevo objetivo

La selección de un nuevo objetivo requiere una decisión humana explícita. Este
cierre no selecciona, diseña ni inicia P2.

P1 queda CLOSED como experimento SHADOW de atribución cognitiva.

N2: FAILED.
N3: SUPPORTED_LIMITED.
N4: FAILED.

La contribución aislada de N3 queda demostrada dentro de las métricas y
condiciones de P1. La superioridad global del backend trained frente al
reference no queda demostrada.

Ningún resultado concede autoridad operativa, staging, promoción, merge a
main ni autorización de P2.

La selección de un nuevo objetivo requiere una decisión humana explícita.
