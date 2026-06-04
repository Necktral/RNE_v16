# ADR: EXT_OPEN_THINKER como resolver externo de conflicto

## Decision

`EXT_OPEN_THINKER` se admite solo como resolver externo gated para conflicto causal/contrafactual.

La admision queda con estado `experimental_admitted_conditionally`. No entra al runtime nominal, no sustituye al core, no es augmenter general y no se declara como inteligencia general.

## Contexto

El benchmark de repetibilidad valido una mejora acotada en el regimen `causal_counterfactual_conflict`. La ruta evaluada fue `core_only` contra `core_plus_external_reasoner_gated_v1`, con `ExternalReasonerGate v1`, schema JSON obligatorio, guard obligatorio y fallback al core.

`ScenarioEpisodeRunner` no queda modificado por esta decision. El uso admitido vive en benchmark de laboratorio.

## Evidencia Experimental

Artefactos canonicos:

- `data/benchmarks/external_reasoner_gain/conflict-repeatability-gated-v1-4x4/summary.json`
- `data/benchmarks/external_reasoner_gain/conflict-repeatability-gated-v1-4x4/external_reasoner_conflict_repeatability_report.md`
- `data/benchmarks/external_reasoner_gain/conflict-repeatability-gated-v1-4x4/external_reasoner_conflict_repeatability_verdict.json`

Resultado global:

- `core_only.ivc_r = 0.000691`
- `core_only.intervention_precision = -0.001818`
- `core_only.viability_margin = -0.031600`
- `core_only.success_rate = 0.000`
- `core_only.closure_stable = 0.000`
- `core_plus_external_reasoner_gated_v1.ivc_r = 0.231401`
- `core_plus_external_reasoner_gated_v1.intervention_precision = 0.067784`
- `core_plus_external_reasoner_gated_v1.viability_margin = 0.029650`
- `core_plus_external_reasoner_gated_v1.success_rate = 0.875`
- `core_plus_external_reasoner_gated_v1.closure_stable = 0.875`

Deltas:

- `delta_ivc_r = +0.230710`
- `delta_intervention_precision = +0.069602`
- `delta_viability_margin = +0.061250`
- `delta_success_rate = +0.875`
- `delta_closure_stable_rate = +0.875`

Repetibilidad:

- bloques positivos: `4/4`
- llamadas externas: `16`
- guard_pass_rate: `0.875`
- guard_reject_rate: `0.125`
- invalid_intervention_accepted: `0`
- latency_mean: `96.115 s`
- latency_p95: `98.953 s`
- corrected_core_failure_rate: `0.875`

## Regimen Validado

Unico regimen validado:

- `causal_counterfactual_conflict`

Regimenes no admitidos por defecto:

- `viability_edge`
- `heterogeneous_warning`
- `homogeneous_safe`

## Limites

- No se activa en runtime nominal.
- No se usa como familia nominal del scheduler.
- No se usa fuera de `causal_counterfactual_conflict`.
- No se aceptan outputs sin schema validado.
- No se acepta intervencion externa sin guard.
- Si gate, schema, guard o fallback no estan presentes, el uso debe rechazarse.
- El fallback core debe permanecer intacto.

## Riesgos

- La latencia media actual (`96.115 s`) esta muy cerca del presupuesto (`100 s`).
- Dos de dieciseis llamadas fueron rechazadas por guard.
- La evidencia solo cubre un regimen sintetico de conflicto causal/contrafactual.
- El modelo externo puede fallar por entorno CUDA, timeout o salida invalida.

## Rollback

Rollback operativo:

1. Mantener `EXT_OPEN_THINKER` fuera de perfiles nominales.
2. Deshabilitar o eliminar `core_plus_external_reasoner_gated_v1` de benchmarks.
3. Conservar `core_only` como decision por defecto.
4. Invalidar la admision cambiando `nominal_status` a `rejected_lab` si evidencia futura degrada cierre, precision o seguridad.

## Criterios de Revalidacion

Revalidar si ocurre cualquiera:

- cambia el modelo, cuantizacion, prompt, schema o wrapper de `llama.cpp`;
- cambia `ExternalReasonerGate v1`;
- cambia el escenario `causal_counterfactual_conflict`;
- `latency_mean` supera `100 s`;
- `external_reasoner_ok_rate` o `schema_validated_rate` cae bajo `0.95`;
- `guard_pass_rate` cae bajo `0.80`;
- aparece cualquier intervencion invalida aceptada;
- `delta_ivc_r`, `delta_intervention_precision` o `delta_viability_margin` deja de ser positivo/no negativo en repetibilidad.

## Por Que No Entra Al Runtime Nominal

No entra al runtime nominal porque la evidencia es condicionada a un solo regimen, el coste es alto, y el valor aparece como resolucion de conflicto causal/contrafactual, no como mejora general del organismo. El runtime nominal conserva el core y sus perfiles adaptativos sin `EXT_OPEN_THINKER`.
