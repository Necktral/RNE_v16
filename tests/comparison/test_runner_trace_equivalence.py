"""B31 — Paridad de TRAZABILIDAD entre runners (canon §4.2).

RUNNER_TRANSITION_POLICY_v1 §4 ("Tests de paridad obligatorios"), 4.2:

    Ambos deben materializar:
    - evento `episode.closed`
    - artifact equivalente
    - secuencia de razonamiento comparable

Este módulo cubre el evento `episode.closed` y la secuencia/traza de
razonamiento. El artifact tiene su propio módulo
(`test_runner_artifact_equivalence.py`), y los gates el suyo
(`test_runner_certification_parity.py`), siguiendo los nombres que el canon
enumera en §10.

Divergencia PERMITIDA (canon §5, documentada): el ScenarioEpisodeRunner emite
`source="scenario_episode_runner"` y agrega `scenario_metadata`; el legacy emite
`source="min_cognitive_episode"` y no lleva metadata de escenario. Eso es
identidad del emisor, no ruptura de trazabilidad.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.world.scenario_runner import ScenarioEpisodeRunner


CANONICAL_SEQUENCE = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "trace_parity.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _run_both(storage, *, external: float = 0.04):
    legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="trace-legacy")
    res_legacy = legacy.run_episode(external_heat=external)
    scenario = ScenarioEpisodeRunner(
        storage=storage, run_id="trace-scenario", scenario="thermal_homeostasis"
    )
    res_scenario = scenario.run_episode(external_input=external)
    return res_legacy, res_scenario


class TestEpisodeClosedEventParity:
    """§4.2 — ambos runners deben materializar el evento `episode.closed`."""

    def test_both_emit_episode_closed(self, tmp_path: Path):
        storage = _storage(tmp_path)
        _run_both(storage)

        closed = [
            e for e in storage.list_events(limit=50) if e.event_type == "episode.closed"
        ]
        sources = {e.source for e in closed}

        assert len(closed) == 2, f"se esperaban 2 eventos episode.closed, hubo {len(closed)}"
        assert sources == {"min_cognitive_episode", "scenario_episode_runner"}
        storage.close()

    def test_both_episode_closed_payloads_carry_the_traceability_core(self, tmp_path: Path):
        """El payload de cierre debe llevar el núcleo trazable en AMBOS runners."""
        storage = _storage(tmp_path)
        _run_both(storage)

        closed = {
            e.source: e.payload
            for e in storage.list_events(limit=50)
            if e.event_type == "episode.closed"
        }

        for source, payload in closed.items():
            assert payload.get("episode_id"), f"{source}: sin episode_id"
            assert payload.get("timestamp"), f"{source}: sin timestamp"
            assert payload.get("closure_profile") == "baseline_fixed", source
            # Contexto: observación, fórmula, intervención y contrafactual.
            context = payload["context"]
            for key in ("observation", "formula", "intervention", "counterfactual"):
                assert key in context, f"{source}: falta context.{key}"
            # Resultado: mundo actualizado, relación semiótica y secuencia.
            result = payload["result"]
            for key in ("updated_world", "relation_kind", "reasoning_sequence"):
                assert key in result, f"{source}: falta result.{key}"
            # Traza materializada.
            assert payload["trace"], f"{source}: traza vacía"

        storage.close()


class TestReasoningSequenceParity:
    """§4.2 — 'secuencia de razonamiento comparable'."""

    def test_both_produce_the_canonical_sequence(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        seq_legacy = res_legacy["episode"]["result"]["reasoning_sequence"]
        seq_scenario = res_scenario["episode"]["result"]["reasoning_sequence"]

        assert seq_legacy == CANONICAL_SEQUENCE
        assert seq_scenario == CANONICAL_SEQUENCE
        # Comparabilidad: la misma secuencia, no sólo dos secuencias válidas.
        assert seq_legacy == seq_scenario
        storage.close()

    def test_traces_have_the_same_shape(self, tmp_path: Path):
        """Misma longitud, mismas familias y mismo contrato de paso."""
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        trace_legacy = res_legacy["episode"]["trace"]
        trace_scenario = res_scenario["episode"]["trace"]

        assert len(trace_legacy) == len(trace_scenario) == len(CANONICAL_SEQUENCE)
        assert [s["family"] for s in trace_legacy] == CANONICAL_SEQUENCE
        assert [s["family"] for s in trace_scenario] == CANONICAL_SEQUENCE

        # El contrato de cada paso de traza es el mismo en ambos runners.
        for step_legacy, step_scenario in zip(trace_legacy, trace_scenario):
            assert set(step_legacy.keys()) == set(step_scenario.keys())
            assert step_legacy["family"] == step_scenario["family"]
            assert step_legacy["status"] == step_scenario["status"]
        storage.close()

    def test_trace_steps_are_persisted_by_both(self, tmp_path: Path):
        """La traza no sólo se devuelve: queda materializada en el episodio cerrado."""
        storage = _storage(tmp_path)
        _run_both(storage)

        closed = [
            e for e in storage.list_events(limit=50) if e.event_type == "episode.closed"
        ]
        for event in closed:
            families = [s["family"] for s in event.payload["trace"]]
            assert families == CANONICAL_SEQUENCE, f"{event.source}: traza {families}"
        storage.close()

    def test_adaptive_profile_keeps_sequence_comparable(self, tmp_path: Path):
        """Ambos aceptan adaptive_min y siguen produciendo la secuencia canónica."""
        storage = _storage(tmp_path)
        legacy = MinimalCognitiveEpisodeRunner(
            storage=storage, run_id="trace-legacy-ad", closure_profile="adaptive_min"
        )
        scenario = ScenarioEpisodeRunner(
            storage=storage,
            run_id="trace-scenario-ad",
            scenario="thermal_homeostasis",
            closure_profile="adaptive_min",
        )
        res_legacy = legacy.run_episode(external_heat=0.04)
        res_scenario = scenario.run_episode(external_input=0.04)

        assert res_legacy["reasoning"]["mode"] == res_scenario["reasoning"]["mode"] == "adaptive"
        assert (
            res_legacy["episode"]["result"]["reasoning_sequence"]
            == res_scenario["episode"]["result"]["reasoning_sequence"]
        )
        storage.close()

