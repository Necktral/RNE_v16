"""B31 — Paridad de ARTIFACT entre runners (canon §4.2).

RUNNER_TRANSITION_POLICY_v1 §4.2 exige que ambos runners materialicen un
"artifact equivalente". Equivalente NO es idéntico: el canon §5 permite
divergencia documentada, y §7.2 exige del ScenarioEpisodeRunner "metadata de
escenario completa". Lo que debe ser equivalente es el CONTRATO del artifact:

- se materializa en disco y es legible;
- mismo `kind` (episode_report);
- misma estructura de tope: episode / smg_snapshot / relation;
- misma disciplina de serialización (JSON determinista: sort_keys, ASCII, indent=2);
- misma metadata mínima (episode_id).

Divergencia PERMITIDA y verificada acá como tal: el scenario runner agrega
`scenario` y `scenario_metadata` a la metadata del artifact. El legacy no.
"""

import json
from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "artifact_parity.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _run_both(storage):
    legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="art-legacy")
    res_legacy = legacy.run_episode(external_heat=0.04)
    scenario = ScenarioEpisodeRunner(
        storage=storage, run_id="art-scenario", scenario="thermal_homeostasis"
    )
    res_scenario = scenario.run_episode(external_input=0.04)
    return res_legacy, res_scenario


def _load(artifact: dict) -> dict:
    return json.loads(Path(artifact["abs_path"]).read_text(encoding="utf-8"))


class TestArtifactMaterialization:
    """§4.2 — ambos materializan un artifact real, no una promesa."""

    def test_both_materialize_a_readable_artifact(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        for res in (res_legacy, res_scenario):
            path = Path(res["artifact"]["abs_path"])
            assert path.exists(), f"artifact no materializado: {path}"
            assert path.stat().st_size > 0, "artifact vacío"
            json.loads(path.read_text(encoding="utf-8"))  # debe parsear
        storage.close()

    def test_both_use_the_same_artifact_kind(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        assert res_legacy["artifact"]["kind"] == "episode_report"
        assert res_scenario["artifact"]["kind"] == "episode_report"
        assert res_legacy["artifact"]["kind"] == res_scenario["artifact"]["kind"]
        storage.close()

    def test_artifact_metadata_carries_episode_id_in_both(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        for res in (res_legacy, res_scenario):
            episode_id = res["episode"]["episode_id"]
            meta = res["artifact"].get("metadata") or {}
            assert meta.get("episode_id") == episode_id
        storage.close()


class TestArtifactStructuralEquivalence:
    """§4.2 — la ESTRUCTURA del artifact debe ser equivalente."""

    def test_same_top_level_structure(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        blob_legacy = _load(res_legacy["artifact"])
        blob_scenario = _load(res_scenario["artifact"])

        assert set(blob_legacy.keys()) == {"episode", "smg_snapshot", "relation"}
        assert set(blob_scenario.keys()) == set(blob_legacy.keys())
        storage.close()

    def test_episode_block_shares_the_common_contract(self, tmp_path: Path):
        """El bloque `episode` del artifact comparte núcleo en ambos runners."""
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        ep_legacy = _load(res_legacy["artifact"])["episode"]
        ep_scenario = _load(res_scenario["artifact"])["episode"]

        common = {"episode_id", "timestamp", "closure_profile", "context", "result", "trace"}
        assert common <= set(ep_legacy.keys())
        assert common <= set(ep_scenario.keys())

        # Núcleo del contexto y del resultado, presente en los dos.
        for ep in (ep_legacy, ep_scenario):
            assert {"observation", "formula", "intervention", "counterfactual"} <= set(
                ep["context"].keys()
            )
            assert {"updated_world", "relation_kind", "reasoning_sequence"} <= set(
                ep["result"].keys()
            )
        storage.close()

    def test_relation_block_is_equivalent(self, tmp_path: Path):
        """La relación semiótica materializada tiene el mismo contrato."""
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        rel_legacy = _load(res_legacy["artifact"])["relation"]
        rel_scenario = _load(res_scenario["artifact"])["relation"]

        assert set(rel_legacy.keys()) == set(rel_scenario.keys())
        for rel in (rel_legacy, rel_scenario):
            assert rel["source_sign_id"]
            assert rel["target_sign_id"]
            assert rel["kind"] in {"support", "contradiction"}
        storage.close()

    def test_serialization_discipline_is_identical(self, tmp_path: Path):
        """JSON determinista en ambos: claves ordenadas, ASCII, indent=2.

        Sin esto los artifacts no son comparables byte a byte entre corridas y se
        rompe la comparabilidad histórica que el canon §9 protege.
        """
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        for res in (res_legacy, res_scenario):
            raw = Path(res["artifact"]["abs_path"]).read_text(encoding="utf-8")
            blob = json.loads(raw)
            # Re-serializar con la misma disciplina debe reproducir el archivo.
            expected = json.dumps(blob, ensure_ascii=True, sort_keys=True, indent=2)
            assert raw == expected, "el artifact no usa JSON determinista"
        storage.close()


class TestDocumentedArtifactDivergence:
    """§5 — divergencia PERMITIDA, explicitada (no silenciosa)."""

    def test_only_the_scenario_runner_adds_scenario_metadata(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        meta_legacy = res_legacy["artifact"].get("metadata") or {}
        meta_scenario = res_scenario["artifact"].get("metadata") or {}

        # Legacy: sin metadata de escenario (canon §1.1, baseline mínimo).
        assert "scenario_metadata" not in meta_legacy
        # Scenario runner: metadata completa (canon §7.2).
        assert meta_scenario["scenario"] == "thermal_homeostasis"
        assert meta_scenario["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        assert meta_scenario["scenario_metadata"]["main_variable"] == "temperature"
        assert meta_scenario["scenario_metadata"]["interventions"] == [
            "activate_cooling",
            "deactivate_cooling",
        ]
        storage.close()

