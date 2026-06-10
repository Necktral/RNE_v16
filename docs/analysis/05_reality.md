# 05 — `runtime/reality/` (validación de realidad operativa)

4579 LOC, 17 archivos. Capa que valida que los episodios "cierran" correctamente y que el
organismo mantiene **continuidad** (no colapsa) a través de escenarios. Núcleo + harnesses de
benchmark + labs de transferencia/analogía.

**Núcleo:** `belief_state` (estado de creencia + shift + evidencia de transición), `evaluator`
(cierre triádico + perfiles de cierre), `continuity` (score escalar), `collapse` (detector),
`transition_analysis` (vector/tensor de continuidad), `service` (orquestador de benchmarks),
`hook` (integración con RuntimeRunner). **Harnesses/labs:** `msrc_policy_benchmark` (755),
`transition_matrix`, `analogical_lab`, `ablation_lab`, `transfer_dynamics`, `transition_stress`,
`analogical_protocols`, `edge_benchmark`.

El `__init__` usa `__getattr__` perezoso (bien: no importa los harnesses pesados al cargar).

---

## Hallazgos

### [RIESGO/BUG] El benchmark heterogéneo resetea la cadena del organismo (service.py:447)
`run_heterogeneous_benchmark` crea un **nuevo `ScenarioEpisodeRunner` por cada paso** del
`scenario_sequence`, con el mismo `run_id`. Cada runner arranca con `_previous_belief = None`,
`OrganismTrajectory` nueva y `OrganismState` episode_count=0. Por tanto:
- La **cadena de belief** (prior→posterior) se reinicia en cada episodio: el prior siempre es None
  → `BeliefShift`/hysteresis/recovery entre escenarios no se capturan vía el runner.
- La **trayectoria del organismo** (T5) no acumula entre pasos; cada paso es `state-0/state-1`.
La continuidad escalar inter-episodio sí se calcula (vía `previous_result` en el servicio), pero la
maquinaria de creencia/trayectoria del propio runner queda inutilizada en el benchmark que más la
necesita (transferencia cross-scenario).

### [RIESGO cross-módulo] `trace_integrity` depende del bug de `list_events` en SQLite (evaluator.py:84-92)
`_has_episode_closed_event` hace `storage.list_events(run_id=run_id, limit=500)` y busca el evento
`episode.closed`. Con el backend SQLite, `list_events` filtra `run_id` **después** del LIMIT
(ver [02_storage.md](02_storage.md)); si hay >500 eventos globales, el `episode.closed` del run
puede no aparecer → `has_episode_closed=False` → `trace_integrity=False` → cierre/certificación
fallan por un artefacto del storage, no por el episodio. En Postgres no ocurre.

### [DISEÑO] Dos rutas de runner inconsistentes en el servicio
`run_benchmark` (homogéneo) usa el **legacy** `MinimalCognitiveEpisodeRunner` (service.py:198);
`run_heterogeneous_benchmark` usa el nuevo `ScenarioEpisodeRunner` (service.py:447). El homogéneo
arrastra los defectos del runner legacy (memoria no-op, hardcode térmico).

### [DISEÑO] Dos implementaciones de estabilidad causal; la "viva" es térmica-céntrica
- `continuity._causal_consistency` (continuity.py:28-44) compara `factual["temperature"]` vs
  `counterfactual["temperature"]` — **hardcodeado a temperatura**. Es la que usa el
  `continuity_score` escalar que consume el servicio.
- `transition_analysis._causal_stability_generic` usa `main_variable` genérico (correcto).
Para escenarios no térmicos (resource), el path escalar cae al fallback por `relation_kind`
(funciona), pero la comparación factual/contrafactual por variable está muerta. Conviene unificar
hacia la versión genérica.

### [DISEÑO] "kl_divergence_approx" no es KL (belief_state.py:203-207)
`compute_belief_shift` calcula `kl_approx = mean(|Δcomponentes|)` (L1 uniforme), pero lo nombra
`kl_divergence_approx`. El docstring lo admite ("Simplified to a weighted L1") aunque ni siquiera
es ponderado. Naming engañoso para una métrica que dispara `recovery_needed`/`is_large_shift`.

### [DISEÑO] Heurísticos con números mágicos en `build_belief_state` (belief_state.py:118-163)
`alarm_prob` ∈ {0.9, 0.1}, `policy_conf = 0.5 + 0.2·support_count`, `causal_conf` ∈ {0.9,0.2,0.5},
penalizaciones de pureza −0.15/−0.10·cross, boost de certificación +0.10. Proxy razonable, pero
toda la "creencia" del organismo se construye sobre constantes ad-hoc sin calibración.

### [DISEÑO] `RealityValidationHook.run_on_shutdown=False` por defecto (hook.py:19)
El hook de validación al cierre del runner viene **desactivado** por defecto; combinado con el bug
de `RuntimeRunner` que pasa `run_id=None` ([03_core.md](03_core.md)), la validación de realidad
al shutdown del orquestador legacy prácticamente no corre con identidad de run.

---

## Harnesses / labs (revisión estructural)
- `msrc_policy_benchmark.MSRCPolicyBenchmarkRunner` (755): corre políticas de escala
  (`always_1x1`, `always_5x5`, `adaptive_msrc*`, `probe_before_switch`…) sobre el grid; maneja
  tanto `world_level` (grid) como `temperature` (térmico) (l.378, 674-675). Genera los reportes en
  `data/benchmarks/msrc/`.
- `transition_matrix`: benchmark NxN de transiciones entre escenarios + gate de matriz.
- `analogical_lab` / `analogical_protocols`: experimentos de transferencia analógica y comparación
  de regímenes; `eml_concurrence_score`.
- `ablation_lab`: estudio de ablación con "EML secondary judgment".
- `transfer_dynamics`: `compute_transfer_stability`, `compute_hysteresis`, `compute_recovery_profile`
  (las métricas A→B→A descritas en `belief_state`).
- `transition_stress` / `edge_benchmark`: stress en los bordes (clasificación de edges).
No se detectaron bugs de corrección en la inspección estructural; son harnesses procedurales que
producen los artefactos `data/benchmarks/*` y `data/reports/*`.

## Aspectos positivos
- Perfiles de cierre (`baseline_fixed` exacto vs `adaptive_min` con orden parcial + opcionales)
  bien especificados; validación de secuencia clara (evaluator.py).
- Gate profiles (`ci`/`extended`/`heterogeneous_ci`) con criterios explícitos; gate heterogéneo
  exige `trace_integrity_rate==1.0` y memoria estricta limpia.
- `transition_analysis`: vector compuesto con pesos explícitos + **belief-enhancement** (mezcla
  heurístico con confianza del belief state); tensor NxN agregado.
- `CollapseDetector` con umbral + racha; `__init__` perezoso.

## Veredicto
Capa conceptualmente rica y mayormente sólida. Riesgos reales: el **reset de la cadena del
organismo en el benchmark heterogéneo** y la **dependencia de `trace_integrity` en el bug de
`list_events`** (acopla un fallo de storage a la certificación). Deuda: unificar las dos
estabilidades causales y el runner homogéneo legacy.
