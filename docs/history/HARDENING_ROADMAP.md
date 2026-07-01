# Roadmap de Hardening Estructural PR #11
**Status**: P0 Completado ✅ | **Fecha**: 2026-04-17

---

## Contexto Estratégico

**Orientación correcta**: No abrir más frentes teóricos. El cuello de botella es cerrar la transición de **módulos sueltos → organismo operativo coherente**.

El proyecto ya tiene nacimiento cognitivo mínimo. Ahora necesita **disciplina fisiológica**: estabilidad, continuidad y ecología de razón gobernada.

---

## ✅ P0 — Cerrar correctamente PR #11 [COMPLETADO]

### Objetivo
Blindar el runtime contra regresión, integrar validación al ciclo de vida, y eliminar deuda técnica crítica.

### Acciones Implementadas

#### 1. ✅ Blindar `baseline_fixed` como secuencia exacta
**Archivo**: `runtime/reality/evaluator.py`

**Cambios**:
- Agregado `_validate_exact_sequence(sequence, required)` → `sequence == required`
- `baseline_fixed` usa validación estricta: `exact_sequence_match=True`
- `adaptive_min` mantiene validación flexible: orden parcial + opcionales
- Documentación explícita del contrato duro en comentarios

**Garantía**: 
- `baseline_fixed` = modo canon (inmutable, comparable históricamente)
- `adaptive_min` = modo ecología (permite DIA_ADV, HEUR, FAL_GUARD, EML_SR)

**Tests**: 21/21 passed
```python
# baseline_fixed rechaza extras
sequence = ["ABD", "HEUR", "ANA", "CAU", "CTF", "DED", "PROB"]
assert validate_sequence_with_profile(sequence, BASELINE_FIXED_PROFILE)["passed"] is False

# adaptive_min acepta opcionales
assert validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)["passed"] is True
```

---

#### 2. ✅ Conectar `RealityValidationHook` al RuntimeRunner
**Archivo**: `runtime/core/orchestration/runner.py`

**Cambios**:
- `RuntimeRunner.__init__` acepta parámetro opcional `reality_hook`
- Hook ejecuta durante `run_forever()` shutdown:
  ```python
  await orch._shutdown.wait()
  # ... cancelar tareas ...
  if self.reality_hook is not None:
      validation_result = self.reality_hook.on_shutdown(run_id=run_id)
  orch.executor.shutdown()
  ```
- Logs `[REALITY] Validación completada` o `[REALITY] Validación falló`
- Manejo de excepciones: no bloquea shutdown si hook falla

**Garantía**: Validación de realidad integrada en ciclo de vida del organismo

**Tests**: 9/9 passed
```python
hook = RealityValidationHook(storage=storage, run_on_shutdown=True)
runner = RuntimeRunner(..., reality_hook=hook)
# hook.on_shutdown() se ejecuta automáticamente durante shutdown
```

---

#### 3. ✅ Hacer `RealityValidationService` sensible a escenarios
**Archivo**: `runtime/reality/service.py`

**Cambios**:
- `evaluate_episode_result()` acepta parámetro `scenario_name`
- Agregado `_compute_scenario_metrics(assessments)`:
  ```python
  {
    "thermal_homeostasis": {
      "total_episodes": 5,
      "closure_rate": 1.0,
      "continuity_mean": 0.85,
      "collapse_count": 0
    },
    "resource_management": {
      "total_episodes": 5,
      "closure_rate": 1.0,
      "continuity_mean": 0.78,
      "collapse_count": 0
    }
  }
  ```
- `summary` incluye `scenario_metrics` además de métricas globales
- Assessment `details` incluyen `scenario_name` para trazabilidad

**Garantía**: Benchmark puede evaluar thermal vs resource independientemente

**Tests**: 22/22 passed (scenario_runner)

---

#### 4. ✅ Eliminar hardcode `/mnt/d`
**Archivo**: `runtime/storage/config.py`

**Cambios**:
```python
# ANTES
artifact_root = Path(os.environ.get("RNFE_ARTIFACT_ROOT", "/mnt/d/rnfe_artifacts"))

# DESPUÉS
default_artifact_root = os.path.join(os.getcwd(), "rnfe_artifacts")
artifact_root = Path(os.environ.get("RNFE_ARTIFACT_ROOT", default_artifact_root))
```

**Garantía**: Portabilidad multiplataforma sin paths específicos de Windows/WSL

---

### Resumen P0
```
✅ baseline_fixed = contrato duro inmutable
✅ RealityValidationHook vive en el ciclo del organismo
✅ Benchmark desagrega métricas por escenario
✅ Sin deuda de portabilidad en paths

Tests: 52/52 PASSED
```

---

## 🚧 P1 — Consolidación del modelo por escenarios [SIGUIENTE FASE]

### Objetivo
Convertir `ScenarioEpisodeRunner` de incubación a camino oficial sin romper baseline histórico.

### A. Introducir identidad de escenario en toda la cadena

**Qué hacer**:
1. Agregar campos obligatorios:
   - `scenario_name` (string): "thermal_homeostasis", "resource_management"
   - `scenario_version` (string): "v1.0.0"
   - `scenario_config_hash` (string): hash de config usado

2. Persistir en:
   - **Episodio**: `episode["scenario_metadata"]`
   - **Assessment**: `assessment.details["scenario_metadata"]`
   - **Certificado**: `certificate.metadata["scenario"]`
   - **Memoria**: `memory_item.context["scenario"]`
   - **Artifact metadata**: `artifact.metadata["scenario"]`

**Por qué**:
- Evitar contaminación de memoria entre escenarios
- Permitir análisis histórico por tipo de mundo
- Habilitar comparación cross-scenario controlada

**Archivos a modificar**:
```
runtime/world/scenario_runner.py       # Agregar metadata al episodio
runtime/reality/service.py             # Persistir en assessment
runtime/certification/promotion_gate.py # Persistir en certificado
runtime/memory/mfm_lite/storage.py     # Filtro en write/retrieve
runtime/storage/records.py             # Schemas con scenario_metadata
```

**Tests a agregar**:
```python
def test_episode_includes_scenario_metadata():
    runner = ScenarioEpisodeRunner(scenario="thermal_homeostasis")
    result = runner.run_episode(external_input=0.05)
    assert result["episode"]["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
    assert "scenario_version" in result["episode"]["scenario_metadata"]
    assert "scenario_config_hash" in result["episode"]["scenario_metadata"]
```

---

### B. Hacer retrieval de memoria con filtro de compatibilidad

**Qué hacer**:
1. Agregar parámetro `scenario_filter_mode` a `MemoryRetrieval.retrieve()`:
   - `"strict_same_scenario"` (default): solo recupera del mismo escenario
   - `"cross_scenario_analogical"`: permite otros escenarios con penalización de score

2. Implementar scoring:
   ```python
   if mode == "strict_same_scenario":
       if memory.scenario != query.scenario:
           return None  # Filtrar completamente
   elif mode == "cross_scenario_analogical":
       if memory.scenario != query.scenario:
           score *= 0.5  # Penalizar analogías cross-scenario
   ```

**Por qué**:
- Por ahora, usar **strict** como default protege contra contaminación
- En el futuro, `cross_scenario_analogical` será útil para transferencia

**Archivos a modificar**:
```
runtime/memory/mfm_lite/retrieval.py   # Agregar filtro
runtime/world/scenario_runner.py       # Usar strict por defecto
```

**Tests a agregar**:
```python
def test_strict_mode_filters_different_scenario():
    storage.write_memory(scenario="thermal", content="hot")
    storage.write_memory(scenario="resource", content="low")
    
    retrieval = MemoryRetrieval(storage, scenario_filter="strict_same_scenario")
    results = retrieval.retrieve(query="temp", scenario="thermal")
    
    assert all(r.scenario == "thermal" for r in results)
```

---

### C. Unificar runners por interfaz

**Qué hacer**:
1. Crear política explícita en documentación:
   ```
   MinimalCognitiveEpisodeRunner = baseline histórico (thermal only)
   ScenarioEpisodeRunner = camino oficial futuro (multi-scenario)
   ```

2. No eliminar `MinimalCognitiveEpisodeRunner` todavía:
   - Mantener como smoke baseline para regresión
   - Usar en tests de comparación: legacy vs scenario runner

3. Agregar tests de equivalencia:
   ```python
   def test_scenario_runner_equivalent_to_legacy():
       legacy_runner = MinimalCognitiveEpisodeRunner()
       scenario_runner = ScenarioEpisodeRunner(scenario="thermal_homeostasis")
       
       legacy_result = legacy_runner.run_episode(external_heat=0.05)
       scenario_result = scenario_runner.run_episode(external_input=0.05)
       
       assert_closure_equivalent(legacy_result, scenario_result)
   ```

**Por qué**:
- Transición segura: ambos coexisten hasta validar estabilidad
- Legacy runner = adapter eventual o smoke baseline permanente

**Archivos a modificar**:
```
docs/RUNNER_TRANSITION.md          # Política de migración
tests/comparison/test_runner_parity.py  # Tests de equivalencia
runtime/reality/service.py          # Soportar ambos runners
```

---

### Resumen P1
```
🎯 Objetivo: ScenarioEpisodeRunner listo para producción
📝 Archivos: ~8 archivos modificados
🧪 Tests: ~15 tests nuevos
⏱️  Estimado: 2-3 días de trabajo
```

---

## 🔬 P2 — Benchmark de continuidad heterogénea [EXPERIMENTO CLAVE]

### Objetivo
Demostrar que el organismo no colapsa al alternar mundos mínimos.

### Diseño del Experimento

**Secuencia**:
```
Episode 1: thermal (external_heat=0.03)
Episode 2: thermal (external_heat=0.05)
Episode 3: resource (external_input=0.04)  ← cambio de escenario
Episode 4: thermal (external_heat=0.06)    ← regreso a thermal
Episode 5: resource (external_input=0.03)  ← cambio nuevamente
```

**Métricas a medir**:
1. **Closure rate global**: ¿Se mantiene ≥90%?
2. **Continuity degradation en cambios**:
   - Continuity(ep2→ep3): esperado drop porque cambia escenario
   - Continuity(ep3→ep4): esperado drop
   - Continuity(ep4→ep5): esperado drop
3. **Memoria cross-scenario pollution**:
   - ¿Episode 3 (resource) recupera memoria de episodes 1-2 (thermal)?
   - Con `strict_same_scenario` debe ser: NO
4. **Scheduler adaptation sin degradar trazabilidad**:
   - ¿Familia sequence sigue valid con perfil?
   - ¿Trace integrity se mantiene?

**Código del benchmark**:
```python
def run_heterogeneous_continuity_benchmark(storage):
    scenarios = [
        ("thermal_homeostasis", 0.03),
        ("thermal_homeostasis", 0.05),
        ("resource_management", 0.04),  # cambio
        ("thermal_homeostasis", 0.06),  # regreso
        ("resource_management", 0.03),  # cambio
    ]
    
    previous_result = None
    assessments = []
    
    for scenario_name, external_input in scenarios:
        runner = ScenarioEpisodeRunner(scenario=scenario_name)
        result = runner.run_episode(external_input=external_input)
        
        assessment = evaluate_episode_result(
            result=result,
            previous_result=previous_result,
            scenario_name=scenario_name,
        )
        assessments.append(assessment)
        previous_result = result
    
    # Análisis
    closure_rate = sum(a.closure_passed for a in assessments) / len(assessments)
    continuity_at_transitions = [
        assessments[2].continuity_score,  # thermal→resource
        assessments[3].continuity_score,  # resource→thermal
        assessments[4].continuity_score,  # thermal→resource
    ]
    
    return {
        "closure_rate": closure_rate,
        "continuity_at_transitions": continuity_at_transitions,
        "mean_transition_continuity": mean(continuity_at_transitions),
    }
```

**Criterio de éxito**:
```
✅ closure_rate ≥ 0.90
✅ mean_transition_continuity ≥ 0.40  (puede ser menor que intra-scenario)
✅ No pollution: memoria no mezcla escenarios en modo strict
✅ Trace integrity: todas las secuencias válidas con perfil adaptive_min
```

---

### Resumen P2
```
🎯 Objetivo: Probar estabilidad multi-escenario
📊 Métricas: 4 dimensiones críticas
🧪 Tests: 1 benchmark + análisis
⏱️  Estimado: 1 día experimento + 1 día análisis
```

---

## Criterios de Éxito por Fase

### P0 ✅ [Completado]
- [x] baseline_fixed es verdaderamente estricto
- [x] Hook vive en RuntimeRunner
- [x] Benchmark desagrega por escenario
- [x] Sin paths hardcodeados

### P1 🚧 [Siguiente]
- [ ] Episodios incluyen scenario_metadata
- [ ] Memoria filtra por escenario (strict mode)
- [ ] Legacy y scenario runner coexisten con política clara

### P2 🔬 [Futuro]
- [ ] Benchmark heterogéneo pasa closure ≥90%
- [ ] Continuity transitions ≥40%
- [ ] Sin contamination cross-scenario
- [ ] Scheduler adapta sin degradar trace

---

## Anti-Patrones a Evitar

### ❌ No hagas esto:
1. **Ampliar LOTF todavía**: El parser mínimo sirve para esta fase
2. **Agregar más familias**: Ya hay `adaptive_min` con opcionales
3. **Crear otro escenario sin validar los 2 existentes**: Primero P2
4. **Eliminar MinimalCognitiveEpisodeRunner prematuramente**: Mantener como baseline
5. **Mezclar P1 y P2**: Consolidar primero, experimentar después

### ✅ Sí haz esto:
1. **Ejecutar P0→P1→P2 en orden**: Sin saltos ni paralelización
2. **Medir antes de expandir**: Cada fase tiene métricas claras
3. **Mantener backward compatibility**: baseline_fixed inmutable
4. **Documentar decisiones**: ADRs para cambios arquitectónicos
5. **Tests antes de integrar**: No merge sin cobertura

---

## Archivo por Archivo - P1 Detallado

### 1. `runtime/world/scenario_runner.py`
**Cambio**: Agregar metadata de escenario al episodio
```python
def run_episode(self, *, external_input: float):
    # ... código existente ...
    
    scenario_metadata = {
        "scenario_name": self.scenario.config.name,
        "scenario_version": "v1.0.0",  # Versionar config
        "scenario_config_hash": self._compute_config_hash(),
        "main_variable": self.scenario.config.main_variable,
        "alarm_threshold": self.scenario.config.alarm_threshold,
    }
    
    episode = {
        "episode_id": episode_id,
        "scenario": self.scenario.config.name,  # backward compat
        "scenario_metadata": scenario_metadata,  # nuevo
        "context": {...},
        "result": {...},
    }
```

### 2. `runtime/reality/service.py`
**Cambio**: Ya implementado en P0.3 ✅
```python
# Ya tenemos scenario_name en assessment.details
# Solo verificar que se usa scenario_metadata completo
```

### 3. `runtime/memory/mfm_lite/retrieval.py`
**Cambio**: Agregar filtro de escenario
```python
class MemoryRetrieval:
    def retrieve(
        self,
        *,
        query: str,
        scenario: str | None = None,
        scenario_filter_mode: str = "strict_same_scenario",
        top_k: int = 5,
    ):
        memories = self.storage.list_memories()
        
        if scenario_filter_mode == "strict_same_scenario" and scenario:
            memories = [m for m in memories if m.scenario == scenario]
        
        # ... scoring existente ...
        
        if scenario_filter_mode == "cross_scenario_analogical":
            for m in scored:
                if m.scenario != scenario:
                    m.score *= 0.5  # penalizar cross-scenario
        
        return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]
```

### 4. `runtime/storage/records.py`
**Cambio**: Agregar campo scenario_metadata a schemas
```python
@dataclass
class EpisodeRecord:
    episode_id: str
    run_id: str
    scenario: str  # backward compat
    scenario_metadata: Dict[str, Any]  # nuevo
    context: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: str
```

---

## Timeline Sugerido

```
Semana 1: P0 ✅ [Completado]
├── Día 1-2: Baseline strict + Hook integration
├── Día 3-4: Scenario metrics + Path portability
└── Día 5: Tests + Documentation

Semana 2: P1 🚧 [Próxima]
├── Día 1-2: Scenario metadata en toda la cadena
├── Día 3-4: Memory retrieval con filtro
└── Día 5: Runner unification + Tests

Semana 3: P2 🔬 [Después]
├── Día 1-2: Diseñar + ejecutar benchmark heterogéneo
├── Día 3-4: Análisis de resultados + ajustes
└── Día 5: Documentación + conclusiones
```

---

## Conclusión

**Status actual**: P0 completado, organismo tiene disciplina fisiológica básica.

**Siguiente paso crítico**: P1 para consolidar multi-escenario antes de experimentar con continuidad heterogénea.

**Decisión clave**: No expandir módulos. Estabilizar los existentes primero.

El éxito de esta fase se mide por **estabilidad y continuidad**, no por cantidad de features.
