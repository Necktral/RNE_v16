# Integración neural por prioridad — bloques P0 y N1

Fecha: 2026-07-15. Worktree: `codex/neural-agent-suite-v1`.

## Resultado

El bloque previo a cualquier entrenamiento quedó integrado en este orden:

1. cierre de feedback conectómico;
2. actividad explícita de MSRC y persistencia;
3. veredicto docente held-out soberano;
4. preparación segura de artefactos de laboratorio.

Esto habilita comparación `SHADOW`; no habilita promoción.

## Conectoma funcional

La topología canónica conserva 22 nodos y pasa de 23 a 38 aristas al representar
los 15 retornos consumidor→órgano. Cada retorno reutiliza los receipts validados,
entra por el puerto `feedback`, conserva `evidence_only` y no participa como una
segunda observación plástica. `MSRC→N0` informa `available`, `constrained` o
`blocked`; `StorageFacade→N0` informa `durable`, `degraded` o `unavailable`.

Una ejecución real de `thermal_homeostasis` con N1/N3/N4 entrenados en laboratorio
cerró 38 conexiones activas: seis gates N0, quince consumos, quince retornos y dos
entradas de gobierno. El grafo no mutó.

## Evidencia docente

La reanálisis estratificada es ahora la única base del veredicto held-out:

- `codex_cross_scenario_gate_passed=false`;
- `codex_teacher_candidate=false`;
- `training_authorized=false`;
- `curriculum_promotion_authorized=false`;
- razón: `stratified_tradeoff_detected`.

El reconciliador regenera summary, verdict, REPORT y `evidence_manifest.json` como
una unidad y falla cerrado si la campaña no coincide o algún permiso es verdadero.

## Artifact plane

`scripts/stage_neural_lab_artifacts.py` valida órgano, backend, contrato, SHA-256 y
procedencia antes de copiar de forma atómica. Sólo acepta artefactos con
`promotion_eligible=false`, genera `activation_profile.json` y nunca exporta
variables de entorno automáticamente.

Se prepararon N1, N3 y N4 bajo `rnfe_artifacts/neural/`. N5 permanece `missing`:
su trainer funciona físicamente, pero no existe corpus ni artefacto semántico final.

## N1 — campaña contrafactual nativa

`scripts/benchmark_n1_counterfactual.py` ejecuta ramas aisladas `core_only` y
`core_plus_<familia>` con storage vacío por episodio. La semilla modifica parámetros
físicos del mundo; la identidad del par usa una huella pre-tratamiento que excluye
IDs y la política tratada, pero conserva mundo, régimen, organismo basal, recursos
y homeostasis. Ambas ramas usan `adaptive_min`: comparar overlays bajo
`baseline_fixed` sería inválido porque ese perfil prohíbe familias opcionales.

La utilidad causal combina recompensa/coste (0.25), efectividad física (0.20),
cierre (0.15), certificación (0.10), continuidad (0.15) y viabilidad (0.15). El
gate exige además de volumen: al menos 30 pares positivos, 30 negativos y rango
de utilidad >= 0.02. Esto impide que un modelo trivial de una sola clase pase por
estar bien calibrado.

La campaña soberana `n1-counterfactual-native-v2` produjo:

- 720 registros, 360 pares válidos y cero rechazos;
- 60 contextos, tres generadores y seis familias;
- 108 pares positivos y 252 negativos;
- splits agrupados 240/60/60 para train/validación/test;
- ECE 0.1545 en validación y 0.1821 en test;
- exactitud positiva 0.80 en validación y 0.95 en test.

El dataset supera la compuerta estructural, pero el artefacto de 676 parámetros
falla ECE <= 0.10. Queda en cuarentena dentro de la campaña, no se prepara en el
artifact plane vivo y `promotion_authorized=false`.

## Próxima prioridad

N1 debe recalibrarse sin tocar test, después compararse en `SHADOW` contra el
scheduler canónico y producir `OrganismImpactReport`. Hasta superar ECE, no se
habilita ese hook como candidato. N3 sigue después como memoria temporal; N5 y N4
esperan datasets semántico y causal respectivamente. N2/N6 no se entrenan hasta
definir objetivos aprendibles independientes.
