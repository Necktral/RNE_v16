# 09 — `runtime/certification/` (vivo) + `runtime/evolution/` (legacy)

## A. `runtime/certification/` (~1.3K LOC) — el integrador central de certificación

`PromotionGate.process_episode` (promotion_gate.py) es el **nudo que une todo el pipeline vivo**:
`evaluate_episode_closure` (reality) → `ContinuityGuard` → `IoCProxy` → `assess_transfer`
(compatibilidad/morfismo) → `ConstitutionalCourtRuntime.ingest_episode` (organismo T5) →
`CertificateBuilder` → ajuste T5 → `promotion_decision` → memoria MFM (micro/meso/macro).

Componentes: `promotion_gate` (317), `transfer_posterior` (358, posterior bayesiano de
transferencia con LCB/UCB beta + scope), `transfer_assessment` (243), `failure_modes` (172),
`certificate_builder` (103), `continuity_guard` (57), `ioc_proxy` (28).

### Hallazgos
**[DISEÑO] Veredicto de certificado acoplado al bug de `list_events`.** `certificate_builder`:
`verdict = "certified" if closure_passed and trace_integrity and not collapse_detected`. Y
`trace_integrity` depende de `evaluate_episode_closure` → `_has_episode_closed_event` →
`storage.list_events(limit=500)`, que en SQLite tiene el bug de filtro post-LIMIT
([02_storage.md](02_storage.md)). Un fallo de storage puede rechazar un episodio válido.

**[DISEÑO] El IoC proxy nunca llega a 1.0 (ioc_proxy.py:21-28).** Pesos positivos
`0.45·continuity + 0.25·closure + 0.20·trace = 0.90` máximo; con penalizaciones
`−0.06·uncertainty − 0.14·collapse`. El techo práctico del IoC es **0.90**, y el umbral de
`promotion_candidate` exige `ioc ≥ 0.72`. Constantes ad-hoc sin normalizar.

**[DISEÑO] `ContinuityGuard` es térmica-céntrica (continuity_guard.py:30-34, 51-57).** Compara
`world_temperature`/`updated_world.temperature`; para escenarios no térmicos cae a `temp_score=0.5`.
Es la **tercera** implementación de "continuidad" del repo (con `reality/continuity` y
`reality/transition_analysis`), y la que usa el gate cuando no hay `reality_assessment` (el caso del
`ScenarioEpisodeRunner`, que llama a `process_episode` sin assessment).

**[DISEÑO] Doble escritura del certificado en la ruta T5 (promotion_gate.py:206).** Con
`T5_MODE=on` (default), si la corte ajusta verdict/promotion/risk, se vuelve a llamar
`write_episode_certificate` con el mismo `certificate_id` → segunda escritura del mismo registro.
La corte puede **degradar** el veredicto a `rejected` si `blocked/quarantine/rollback` o
`max_t4_risk ≥ 0.85`, y gatea `promotion_candidate` a `max_t4_risk < 0.60`. Es decir, el T5
gobierna la certificación por defecto.

**Positivo:** integración coherente y completa; `transfer_posterior` con prior/likelihood + LCB/UCB
beta y determinación de scope (mismo estilo bayesiano que `organism/risk`); memoria multi-escala
solo se escribe si `verdict == "certified"`.

---

## B. `runtime/evolution/` (~1.4K LOC) — neuro-evolución LEGACY (AEON FENIX)

Maquinaria de auto-evolución neuronal del orquestador antiguo. Alcanzable **solo desde el
`module_orchestrator` legacy** (imports relativos `..evolution.*` / `src.evolution.*`).

- `neurogenesis.NeurogenesisManager`: crecimiento **real** de capas (`_create_expanded_layer`
  expande `nn.Linear`, propaga a capas dependientes). Algoritmo genuino.
- `katana_pruner.KatanaPruner`: poda **real** (concrete dropout, KL horseshoe, hybrid score,
  máscara concreta). Algoritmo genuino.
- `auto_mutator.AutoMutator`: orquesta neurogénesis+poda bajo "presión de mutación"; tamaño de paso
  adaptativo; emite payloads que el `QuantumDistributedTrainer.apply_adaptation` aplica.
- `meta_optimizer`: `QuantumExponentialOptimizer` + `QuantumState` + `PhysicsAwareMonitor`. Lógica
  NAS-ish real, pero con la misma **nomenclatura "cuántica" cosmética** que `core/episteme`.
- `predictive_coder.HierarchicalPredictiveCoder`: red de codificación predictiva (`nn.Module`) real.
- `EvolutionaryRehabilitationCenter` (`QuantumNASEngine`, `EpistemicIntelligence` ABC): **sin
  importadores detectados** → candidato a código muerto/orphan.

### Hallazgos
**[MUERTO] `EvolutionaryRehabilitationCenter` parece huérfano** (ningún importador en runtime/
scripts/exocortex/tests/lab).
**[DISEÑO] Acoplado al Orchestrator legacy**: si ese loop es deuda muerta (ver [03_core.md](03_core.md)),
toda `evolution/` lo es por transitividad, salvo que se reutilice suelto.
**[DISEÑO] Nomenclatura "cuántica" cosmética** en `meta_optimizer` (QuantumState/QuantumExponential).

---

## Veredicto
`certification/` es **código vivo de primera línea** (el integrador que produce certificados y
gobierna promoción + memoria), con buena estructura bayesiana pero arrastra el sesgo térmico de la
continuidad, el techo 0.90 del IoC y el acoplamiento al bug de `list_events`. `evolution/` es
**neuro-evolución legacy real pero desconectada del pipeline vivo** (solo el Orchestrator antiguo la
usa); contiene algoritmos genuinos de poda/crecimiento envueltos en naming "cuántico".
