# 04 — `runtime/world/` (escenarios y entornos)

3770 LOC, 14 archivos. Capa de **mundos/escenarios** sobre la que corren los episodios. Diseño
en **dos capas**:

- **Legacy (thermal-only):** `cgwm_min.py` (`CGWMMin`, mundo mínimo) + `min_cognitive_episode.py`
  (`MinimalCognitiveEpisodeRunner`). Hardcodeado a temperatura.
- **Generalizada (RTCME / morfismos):** ABC `scenario.CognitiveScenario` + `registry.py` +
  `scenario_runner.ScenarioEpisodeRunner`, con escenarios concretos `thermal_scenario`,
  `resource_scenario`, `grid_thermal_scenario` (5x5), y el aparato de transferencia
  `compatibility.py` (grafo simétrico), `morphism_engine.py` (morfismos dirigidos),
  `alignment.py` (asignación bipartita), `causal_signature.py` (firmas causales).

`ScenarioEpisodeRunner.run_episode` es **el corazón del pipeline vivo**: observe → SMG (signos) →
LOTF (fórmula) → memoria → intervención → contrafactual/factual → MetaScheduler (razonamiento) →
belief_state → organismo T5 (state/trajectory/constitution/viability) → PromotionGate
(certificación) → EML shadow opcional.

---

## Hallazgos

### [BUG/MUERTO] La memoria recuperada no influye en la decisión (scenario_runner.py:214-219, min_cognitive_episode.py:92-99)
En ambos runners, tras `memory_retrieval.retrieve(...)`:
```python
intervention = self.scenario.select_intervention(observation)
if memory_hits:
    top = memory_hits[0].get("structure", {})
    if top.get("relation_kind") == "support" and observation.alarm:
        intervention = self.scenario.select_intervention(observation)  # ← mismo valor
```
La rama recomputa **exactamente el mismo** `select_intervention(observation)`. La memoria se
recupera (y se loguea/persiste) pero **no cambia la intervención elegida**. El "uso de memoria"
es un no-op funcional.

### [MUERTO] `trajectory_window` siempre `None` por `if False` (scenario_runner.py:400)
```python
episode_result["trajectory_window"] = self._organism_trajectory.get_window(window_size=5).to_dict() if False else None  # Will enable in certification update
```
Código muerto con comentario "Will enable". La ventana de trayectoria nunca se puebla en el
resultado del episodio (la certificación T5 no recibe la ventana por esta vía).

### [RIESGO] Contrafactual siempre = `interventions[1]` (scenario_runner.py:222-226)
```python
counter_intervention = interventions[1] if len(interventions) > 1 else interventions[0]
```
Se fija como contrafactual la **segunda** intervención, sin relación con la elegida factualmente.
Si la intervención factual ES `interventions[1]` (p. ej. en recursos al no haber alarma se elige
`stop_production` = índice 1), factual y contrafactual coinciden → `evaluate_relation_kind`
compara un estado consigo mismo y siempre da `support`. Sesga la señal de relación.

### [DOC] `alignment.py` dice "algoritmo húngaro" pero usa greedy (alignment.py:6 vs 213)
El docstring del módulo afirma *"Usa el algoritmo húngaro (scipy-free, implementación interna)"*,
pero `_greedy_assignment` es asignación **greedy** por mínimo costo (el propio docstring de la
función lo admite). El greedy no garantiza el óptimo global; para N=2-4 intervenciones suele
coincidir, pero la afirmación del módulo es falsa. (Coherente con "docs desfasados".)

### [DISEÑO] Duplicación legacy vs. generalizado
- `cgwm_min.CGWMMin` duplica la dinámica de `ThermalScenario` (mismo `0.07` de enfriamiento
  hardcodeado, cgwm_min.py:44).
- `MinimalCognitiveEpisodeRunner` duplica ~80% de `ScenarioEpisodeRunner` pero fijo a térmico.
  Ambos siguen exportados (el `__init__` los marca "Legacy"). Mantener dos rutas de episodio
  arriesga drift de comportamiento.

### [DISEÑO] `__init__.py` con docstring duplicado (world/__init__.py:1-2)
Dos docstrings de módulo seguidos; solo el segundo cuenta. Inocuo.

### [DISEÑO] `evaluate_relation_kind` por defecto asume "menor es mejor" (scenario.py:218)
Correcto para térmico; `ResourceScenario` lo **sobreescribe** bien (mayor stock = mejor,
resource_scenario.py:264). Riesgo si un escenario nuevo "maximize" olvida sobreescribirlo →
relación invertida silenciosa. (El `structural_profile`/`causal_signature` ya marcan la polaridad,
pero la base no la usa.)

---

## `grid_thermal_scenario.py` (1043 LOC) — el escenario más complejo
Extiende el térmico a una **malla 5x5** de celdas (`CellState`/`GridState`), Python puro
(`math`/`random`, sin numpy). Aporta:
- Topologías iniciales: `uniform`, `hotspot_center/corner`, `gradient_ns/ew`, `checkerboard`,
  `quadrants` (l.167-271).
- Métricas espaciales: detección de hotspots, gradiente térmico, distribución por cuadrantes,
  índice de concentración, **entropía espacial** (l.448-617).
- Niveles de mundo discretos 1-4 (SAFE/ELEVATED/WARNING/CRITICAL) y proposiciones espaciales
  generadas (l.393-770).
Implementa correctamente la interfaz `CognitiveScenario`. Es código nuevo y estructurado; no se
detectaron bugs en la inspección estructural (1043 líneas de helpers deterministas).

---

## Aspectos positivos
- ABC `CognitiveScenario` clara y completa; escenarios concretos pequeños y legibles.
- `compatibility.py` y `morphism_engine.py`: scoring ponderado bien factorizado; la matriz de
  morfismos es **asimétrica** a propósito (M[A][B] ≠ M[B][A]) — modela transferencia dirigida.
- `thermal` (minimize/lower_is_better) y `resource` (maximize/higher_is_better) son **inversos a
  propósito** → buenos para probar transferencia/penalización de direccionalidad.
- Artefactos de episodio content-addressed vía `materialize_artifact`.
- `causal_signature` con grafo causal mínimo (DAG) por escenario.

## Veredicto
Capa de world **sólida y moderna** salvo tres defectos concretos del runner principal: el no-op de
memoria, la `trajectory_window` muerta y el contrafactual fijo. La capa legacy
(`cgwm_min`/`min_cognitive_episode`) es candidata a deprecación una vez todo migre a
`ScenarioEpisodeRunner`.
