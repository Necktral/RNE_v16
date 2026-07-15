# Held-out y currículo Codex → 7B

Fecha: 2026-07-14. Campaña ejecutada en el volumen Linux nativo.

## Escala ejecutada

- 3 escenarios;
- 2 perturbaciones inéditas por escenario;
- 10 seeds;
- 3 variantes;
- horizonte 5;
- 180 trials, 60 inferencias 7B y 1,080 episodios;
- duración: 346.149 s.

El 7B alcanzó 80% de calidad semántica, 45.22 tokens/s y 3.43 s de latencia
media. Codex conservó 100% de calidad semántica. El promedio agregado favoreció la
guía, pero el selector curricular evalúa cada par, no sólo el promedio.

## Hallazgo estratificado

| Escenario | Perturbación | Δ reward Codex-control | Δ severidad Codex-control |
|---|---|---:|---:|
| térmico | baja | -0.003250 | -0.024760 |
| térmico | alta | +0.046333 | +0.001720 |
| recursos | baja | -0.003000 | -0.022300 |
| recursos | alta | +0.045333 | +0.002200 |
| carga diferida | baja | -0.000100 | -0.004320 |
| carga diferida | alta | +0.000500 | -0.000220 |

Una política fija no domina al control en térmico ni recursos: en perturbación baja
intercambia reward por seguridad; en alta intercambia una pequeña severidad por
reward. En carga diferida sólo 10 de 20 pares cumplen ambos objetivos.

## Dataset construido y rechazado

`build_teacher_curriculum.py` produjo tres registros candidatos con procedencia,
seeds, perturbaciones, hashes y outcomes. El gate exige 20 soportes, 10 seeds, dos
perturbaciones, ≥90% de éxitos, semántica perfecta y cero regresiones.

Resultado: 0/3 registros elegibles. Veredicto:
`dataset_rejected_by_heldout_gate`. Entrenamiento y promoción permanecen falsos.
`stratified_reanalysis.json` sustituye explícitamente la lectura agregada inicial y
fija `codex_cross_scenario_gate_passed=false` al evaluar cada perturbación.

El veredicto, resumen, reporte y manifiesto de evidencia fueron reconciliados con
`benchmark_teacher_advanced.py --reconcile-existing`. Desde entonces
`verdict.json` declara la reanálisis estratificada como `evidence_basis`, elimina el
candidato docente agregado y mantiene `training_authorized=false`. Los seis hashes
del manifiesto fueron regenerados y verificados.

Este rechazo no significa que la guía sea inútil. Significa que el input docente no
incluye información suficiente para resolver el tradeoff. La siguiente versión del
contrato debe permitir `abstain_pending_measurement`, declarar qué variable falta y
pedir predicción/foresight antes de recomendar una acción. Sólo entonces conviene
generar un currículo condicional y repetir held-out.

Evidencia:

- `data/reports/teacher_advanced/teacher-heldout-native-v1-10x2-h5/`;
- `data/curricula/teacher_codex_to_7b/codex-to-7b-heldout-v1/`.
