# Phase 1 Architectural Corrections

**Branch**: `claude/open-clean-branch-1x1-to-5x5`
**Fecha**: 2026-04-19
**Tipo**: Corrección arquitectónica crítica

---

## Problema Identificado

El código de instrumentación de Fase 1 fue implementado asumiendo un **contrato incorrecto del runtime**, lo que habría causado fallos inmediatos al ejecutar benchmarks.

### Síntomas Específicos

1. **Parámetro inexistente**: `benchmark_runner.py` pasaba `max_steps=config.max_steps` a `ScenarioEpisodeRunner.__init__()`, pero este parámetro no existe en el contrato real.

2. **Claves ficticias**: El código asumía que `run_episode()` retorna claves como:
   - `closed` (no existe)
   - `trace` (no existe a nivel root, es reasoning.sequence)
   - `continuity_score` (no existe directamente)
   - `counterfactual` a nivel root (está en episode.context.counterfactual)

3. **Semántica incorrecta**:
   - Asumía "trace" es progresión multi-step del mundo
   - En realidad: cada episodio es **un solo paso cognitivo**
   - El "reasoning trace" es del scheduler, NO del mundo

4. **Outcome ficticio**: Determinaba success/failure desde campo `closed` que no existe, en lugar de usar `certification.verdict`.

---

## Contrato Real del Runtime

### Firma de `ScenarioEpisodeRunner`

```python
class ScenarioEpisodeRunner:
    def __init__(
        self,
        scenario: Any,
        storage: StorageFacade,
        run_id: str,
    ):
        # NO acepta max_steps
```

### Payload Real de `run_episode()`

```python
def run_episode(self, *, external_input: float = 0.04) -> Dict[str, Any]:
    # Retorna:
    {
        "episode": {
            "episode_id": str,
            "scenario": str,
            "context": {
                "observation": {...},
                "counterfactual": {...} or None,
            },
            "result": {
                "updated_world": {...},
            },
        },
        "smg_snapshot": {...},
        "reasoning": {
            "sequence": [...],  # Pasos del scheduler
        },
        "artifact": {
            "abs_path": str,
        },
        "run_id": str,
        "organism_trajectory": {
            "points": [
                {
                    "viability_margin": float,
                    ...
                }
            ],
        },
        "constitutional_validation": {...},
        "viability_assessment": {
            "is_viable": bool,
            "viability_margin": float,
            "distance_to_edge": float,
        },
        "certification": {
            "verdict": "passed" | "certified" | ...,
            "promotion_candidate": bool,
        },
        "eml_shadow": {...},
    }
```

### Determinación de Success/Failure

```python
# CORRECTO:
if certification.verdict in ['passed', 'certified']:
    outcome = 'success'
else:
    outcome = 'failure'

# INCORRECTO (no existe):
if result['closed']:
    outcome = 'success'
```

---

## Solución Implementada

### 1. Adapter Pattern

Creado `adapt_runtime_result_to_benchmark()` en `benchmark_runner.py`:

```python
def adapt_runtime_result_to_benchmark(runtime_result: Dict[str, Any]) -> Dict[str, Any]:
    """Adapta el payload real del runtime a formato benchmark."""
    return {
        'episode_id': runtime_result.get('episode', {}).get('episode_id'),
        'run_id': runtime_result.get('run_id'),
        'scenario_name': runtime_result.get('episode', {}).get('scenario'),

        # Certification
        'certification_verdict': runtime_result.get('certification', {}).get('verdict'),
        'promotion_candidate': runtime_result.get('certification', {}).get('promotion_candidate'),

        # Viability
        'is_viable': runtime_result.get('viability_assessment', {}).get('is_viable'),
        'viability_margin': runtime_result.get('viability_assessment', {}).get('viability_margin'),
        'distance_to_edge': runtime_result.get('viability_assessment', {}).get('distance_to_edge'),

        # Artifact
        'artifact_path': runtime_result.get('artifact', {}).get('abs_path'),

        # Reasoning trace (del scheduler, NO del mundo)
        'reasoning_sequence': runtime_result.get('reasoning', {}).get('sequence', []),

        # Organism trajectory
        'organism_trajectory': runtime_result.get('organism_trajectory'),

        # Episode context
        'observation': runtime_result.get('episode', {}).get('context', {}).get('observation'),
        'updated_world': runtime_result.get('episode', {}).get('result', {}).get('updated_world'),
        'counterfactual_world': runtime_result.get('episode', {}).get('context', {}).get('counterfactual'),

        # Raw payload para métricas que lo necesiten
        'raw_runtime_result': runtime_result,
    }
```

### 2. Corrección de `benchmark_runner.py`

**Antes**:
```python
runner = ScenarioEpisodeRunner(
    scenario=scenario,
    storage=temp_storage,
    run_id=f"bench-{episode_id[:8]}",
    max_steps=config.max_steps,  # ❌ NO EXISTE
)

runtime_result = runner.run_episode(external_input=0.04)

if runtime_result.get('closed'):  # ❌ NO EXISTE
    result.outcome = 'success'
```

**Después**:
```python
# Crear runner SIN max_steps (no existe en el contrato)
runner = ScenarioEpisodeRunner(
    scenario=scenario,
    storage=temp_storage,
    run_id=f"bench-{episode_id[:8]}",
)

# Ejecutar UN SOLO EPISODIO (un paso cognitivo)
runtime_result = runner.run_episode(external_input=0.04)

# Adaptar resultado del runtime a formato benchmark
adapted_result = adapt_runtime_result_to_benchmark(runtime_result)

# Determinar outcome según certificación
cert_verdict = adapted_result['certification_verdict']
if cert_verdict in ['passed', 'certified']:
    result.outcome = 'success'
else:
    result.outcome = 'failure'
```

### 3. Corrección de Métricas Cognitivas

**Antes** (asumía trace multi-step):
```python
def compute_intervention_precision(episode: Dict[str, Any]) -> float:
    trace = episode.get('trace', [])  # ❌ NO EXISTE

    for i in range(len(trace) - 1):
        step = trace[i]
        next_step = trace[i + 1]
        # ...
```

**Después** (usa observation → updated_world):
```python
def compute_intervention_precision(episode: Dict[str, Any]) -> Optional[float]:
    observation = episode.get('observation')  # ✅ Snapshot inicial
    updated_world = episode.get('updated_world')  # ✅ Estado final

    temp_initial = observation.get('world_level') or observation.get('temperature')
    temp_final = updated_world.get('world_level') or updated_world.get('temperature')

    precision = (temp_initial - temp_final) / temp_initial
    return precision
```

### 4. Corrección de Métricas Operativas

**Antes** (inventaba campos):
```python
def compute_counterfactual_overhead_ratio(episode: Dict[str, Any]) -> float:
    factual_time = episode.get('factual_time_ms')  # ❌ NO EXISTE
    cf_time = episode.get('counterfactual_time_ms')  # ❌ NO EXISTE
    return cf_time / factual_time
```

**Después** (solo observables):
```python
def compute_all_operational_cost_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'wall_time_ms': episode.get('wall_time_ms'),  # ✅ Observable
        'artifact_size_bytes': episode.get('artifact_size_bytes'),  # ✅ Observable
        'reasoning_trace_length': episode.get('reasoning_trace_length'),  # ✅ Observable
    }
    # NO: cf_overhead_ratio, memory_pressure_mb, scheduler_cpu_time_ms
```

### 5. Corrección de Taxonomía de Fallos

**Antes**:
```python
class FailureCategory:
    TIMEOUT = 'timeout'  # ❌ No hay timeouts en single-step
    ALARM_PERSISTENT = 'alarm_persistent'  # ❌ Requiere trace multi-step
    OSCILLATION = 'oscillation'  # ❌ Requiere trace multi-step
```

**Después**:
```python
class FailureCategory:
    ERROR = 'error'  # ✅ Exception durante ejecución
    CERTIFICATION_FAILED = 'certification_failed'  # ✅ verdict no es 'passed'/'certified'
    VIABILITY_FAILED = 'viability_failed'  # ✅ is_viable = False
    BOTH_FAILED = 'both_failed'  # ✅ Ambos fallaron

def classify_failure_primary(episode: Dict[str, Any]) -> Optional[str]:
    outcome = episode.get('outcome')

    if outcome == 'error':
        return FailureCategory.ERROR

    if outcome == 'success':
        return None

    cert_verdict = episode.get('certification_verdict')
    is_viable = episode.get('is_viable', True)

    cert_failed = cert_verdict not in ['passed', 'certified']
    viability_failed = not is_viable

    if cert_failed and viability_failed:
        return FailureCategory.BOTH_FAILED
    elif viability_failed:
        return FailureCategory.VIABILITY_FAILED
    elif cert_failed:
        return FailureCategory.CERTIFICATION_FAILED
```

---

## Métricas Removidas

### Grupo 2: Calidad Cognitiva

❌ **Removidas** (requieren trace multi-step del mundo):
- `factual_cf_divergence`: Divergencia entre trayectorias factual vs contrafactual
- `world_level_transitions`: Transiciones entre niveles SAFE/ELEVATED/WARNING/CRITICAL
- `spatial_coherence_index`: Correlación temporal entre celdas vecinas

✅ **Mantenidas** (compatibles con single-step):
- `intervention_precision`: Delta temperatura observation → updated_world
- `proposition_diversity`: Entropía de proposiciones en snapshot
- `spatial_information_usage`: Ratio de proposiciones espaciales

### Grupo 3: Costo Operativo

❌ **Removidas** (no instrumentadas en runtime):
- `counterfactual_overhead_ratio`: No existe timing separado factual/CF
- `memory_pressure_mb`: No hay captura de memoria
- `scheduler_cpu_time_ms`: No hay timing separado del scheduler

✅ **Mantenidas** (observables directos):
- `wall_time_ms`: Tiempo total de ejecución
- `artifact_size_bytes`: Tamaño del artifact serializado
- `reasoning_trace_length`: Longitud del reasoning trace

### Grupo 5: Taxonomía de Fallos

❌ **Removidas** (requieren trace multi-step):
- `TIMEOUT`: No hay concepto de timeout en single-step
- `ALARM_PERSISTENT`: Requiere monitoreo de alarmas en trace
- `OSCILLATION`: Requiere detección de ciclos en trace
- `SCHEDULER_OVERHEAD`: No hay timing separado del scheduler

✅ **Mantenidas** (basadas en certification/viability):
- `ERROR`: Exception durante ejecución
- `CERTIFICATION_FAILED`: Certificación no pasó
- `VIABILITY_FAILED`: Viabilidad fallida
- `BOTH_FAILED`: Certificación y viabilidad fallaron

---

## Principios de Corrección

### 1. **"Desde el runtime hacia el benchmark, NO al revés"**

No modificar el runtime para agregar campos que el benchmark espera.
Adaptar el benchmark para consumir lo que el runtime realmente retorna.

### 2. **No inventar claves ficticias**

Si el campo no existe en el payload real, no asumirlo.
Crear adapter explícito que normaliza el payload.

### 3. **Métricas solo de observables reales**

Si una métrica requiere datos no instrumentados, NO inventar valores.
Documentar explícitamente qué métricas fueron removidas y por qué.

### 4. **Single-step, no multi-step**

Cada episodio es **un paso cognitivo**.
No hay "trace del mundo" con múltiples pasos.
El reasoning trace es del scheduler, no del mundo.

---

## Impacto en el Análisis

### ¿Qué se puede medir?

**Comparación 1x1 vs 5x5**:
- ✅ **Éxito/Fallo**: Certification rate (proporción de 'passed'/'certified')
- ✅ **Precisión**: Reducción térmica lograda (observation → updated_world)
- ✅ **Diversidad**: Entropía de proposiciones
- ✅ **Uso espacial**: Ratio de proposiciones espaciales (5x5 vs 1x1)
- ✅ **Costo temporal**: wall_time_ms
- ✅ **Costo de almacenamiento**: artifact_size_bytes
- ✅ **Costo cognitivo**: reasoning_trace_length

**Ganancia Neta**:
```
Ganancia_Neta = Δ_precision + Δ_spatial_usage - Penalty(Δ_wall_time, Δ_artifact_size)
```

### ¿Qué NO se puede medir (sin instrumentación adicional)?

- ❌ Divergencia entre trayectorias factual/contrafactual (requiere trace multi-step)
- ❌ Transiciones entre niveles del mundo (requiere trace multi-step)
- ❌ Coherencia espacial temporal (requiere snapshots en múltiples pasos)
- ❌ Overhead de contrafactual (no hay timing separado)
- ❌ Presión de memoria (no instrumentada)
- ❌ Timeouts y oscilaciones (no hay trace multi-step)

---

## Conclusión

### Estado Actual

✅ **Fase 1 honestamente completa**:
- Código alineado con contrato real del runtime
- Métricas basadas solo en observables reales
- Adapter explícito para normalización de payload
- Success/failure desde certification.verdict

### Listo para Experimental Validation

El código **ahora sí** puede ejecutar benchmarks sin fallar.

Los benchmarks medirán lo que **realmente se puede medir** con la instrumentación actual.

Si se requieren métricas adicionales (trace multi-step, timing separado CF, memoria), será necesario **instrumentación adicional en el runtime** (fuera del alcance de Fase 1).

---

**Mandato cumplido**:
> "No corrijas la arquitectura desde el benchmark hacia el runtime.
> Corrígela desde el runtime hacia el benchmark."

✅ El runtime permanece **intacto**.
✅ El benchmark fue **adaptado** para consumir el contrato real.
✅ Las métricas fueron **corregidas** para usar solo observables reales.
