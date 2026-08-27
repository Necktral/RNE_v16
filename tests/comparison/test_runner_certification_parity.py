"""B31 — Paridad de GATES: cierre y certificación (canon §4.3).

RUNNER_TRANSITION_POLICY_v1 §4 ("Tests de paridad obligatorios"), 4.3:

    La validación de cierre y certificación no deben divergir sin explicación
    documental.

Y §7 (criterios de promoción del ScenarioEpisodeRunner) exige, entre otros:
"no contaminación de certificación" y "no regresión de portabilidad".

Lo que se fija acá:
- ambos runners atraviesan el gate y emiten un certificado persistido;
- el VEREDICTO no diverge en el baseline térmico (ni de certificado ni de decisión);
- el gate de cierre acepta la secuencia canónica en los dos;
- el certificado del scenario runner no queda "contaminado" por escenarios ajenos.

NOTA SOBRE LO QUE ESTE MÓDULO **NO** AFIRMA
--------------------------------------------
El canon no fija umbrales numéricos de IoC/riesgo por runner, así que no se
inventa ninguno: no hay assert sobre "IoC >= X". Afirmar un número que el canon
no manda sería un test decorativo. Se verifica lo que el canon SÍ manda: que los
veredictos no diverjan.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.world.scenario_runner import ScenarioEpisodeRunner


VALID_VERDICTS = {"certified", "PASSED", "CONDITIONALLY_PASSED"}


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "cert_parity.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _run_both(storage, *, episodes: int = 1):
    legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="cert-legacy")
    scenario = ScenarioEpisodeRunner(
        storage=storage, run_id="cert-scenario", scenario="thermal_homeostasis"
    )
    res_legacy = res_scenario = None
    for _ in range(episodes):
        res_legacy = legacy.run_episode(external_heat=0.04)
        res_scenario = scenario.run_episode(external_input=0.04)
    return res_legacy, res_scenario


class TestCertificationGateParity:
    """§4.3 — la certificación no diverge entre runners."""

    def test_both_produce_a_certificate(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        assert res_legacy["certification"]["certificate_id"]
        assert res_scenario["certification"]["certificate_id"]
        storage.close()

    def test_certificate_verdict_does_not_diverge(self, tmp_path: Path):
        """El veredicto del certificado debe COINCIDIR en el baseline térmico."""
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        verdict_legacy = res_legacy["certification"]["verdict"]
        verdict_scenario = res_scenario["certification"]["verdict"]

        assert verdict_legacy in VALID_VERDICTS
        assert verdict_scenario in VALID_VERDICTS
        assert verdict_legacy == verdict_scenario, (
            f"divergencia de certificación sin explicación documental (canon §4.3): "
            f"legacy={verdict_legacy} scenario={verdict_scenario}"
        )
        storage.close()

    def test_decision_verdict_does_not_diverge(self, tmp_path: Path):
        """El veredicto de DECISIÓN (promoción) tampoco debe divergir."""
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        assert (
            res_legacy["certification"]["decision_verdict"]
            == res_scenario["certification"]["decision_verdict"]
        )
        assert (
            res_legacy["certification"]["promotion_candidate"]
            == res_scenario["certification"]["promotion_candidate"]
        )
        storage.close()

    def test_verdicts_stay_aligned_over_several_episodes(self, tmp_path: Path):
        """La paridad de gates no puede ser un accidente del primer episodio."""
        storage = _storage(tmp_path)
        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="cert-legacy-n")
        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="cert-scenario-n", scenario="thermal_homeostasis"
        )
        for episode in range(4):
            res_legacy = legacy.run_episode(external_heat=0.04)
            res_scenario = scenario.run_episode(external_input=0.04)
            assert (
                res_legacy["certification"]["verdict"]
                == res_scenario["certification"]["verdict"]
            ), f"divergencia en el episodio {episode}"
        storage.close()


class TestClosureGateParity:
    """§4.3 — la validación de CIERRE no diverge."""

    def test_both_close_with_the_canonical_sequence(self, tmp_path: Path):
        storage = _storage(tmp_path)
        res_legacy, res_scenario = _run_both(storage)

        canonical = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        assert res_legacy["episode"]["result"]["reasoning_sequence"] == canonical
        assert res_scenario["episode"]["result"]["reasoning_sequence"] == canonical
        # Ambos declaran el mismo perfil de cierre.
        assert (
            res_legacy["episode"]["closure_profile"]
            == res_scenario["episode"]["closure_profile"]
            == "baseline_fixed"
        )
        storage.close()

    def test_both_closure_profiles_are_accepted_by_the_gate(self, tmp_path: Path):
        """adaptive_min no rompe el gate en ninguno de los dos runners."""
        storage = _storage(tmp_path)
        legacy = MinimalCognitiveEpisodeRunner(
            storage=storage, run_id="cert-legacy-ad", closure_profile="adaptive_min"
        )
        scenario = ScenarioEpisodeRunner(
            storage=storage,
            run_id="cert-scenario-ad",
            scenario="thermal_homeostasis",
            closure_profile="adaptive_min",
        )
        res_legacy = legacy.run_episode(external_heat=0.04)
        res_scenario = scenario.run_episode(external_input=0.04)

        assert res_legacy["certification"]["verdict"] in VALID_VERDICTS
        assert res_scenario["certification"]["verdict"] in VALID_VERDICTS
        assert (
            res_legacy["certification"]["verdict"]
            == res_scenario["certification"]["verdict"]
        )
        storage.close()


class TestNoCertificationContamination:
    """§7.5 — 'no contaminación de certificación'."""

    def test_certificates_are_persisted_per_run(self, tmp_path: Path):
        """Cada runner certifica en SU run: los certificados no se mezclan."""
        storage = _storage(tmp_path)
        _run_both(storage, episodes=2)

        certs_legacy = storage.list_episode_certificates(run_id="cert-legacy", limit=10)
        certs_scenario = storage.list_episode_certificates(run_id="cert-scenario", limit=10)

        assert len(certs_legacy) == 2
        assert len(certs_scenario) == 2

        ids_legacy = {c.certificate_id for c in certs_legacy}
        ids_scenario = {c.certificate_id for c in certs_scenario}
        assert ids_legacy.isdisjoint(ids_scenario), "certificados cruzados entre runners"
        storage.close()

    def test_scenario_runner_certificate_is_bound_to_its_scenario(self, tmp_path: Path):
        """El certificado del scenario runner no se contamina con otro escenario."""
        storage = _storage(tmp_path)
        thermal = ScenarioEpisodeRunner(
            storage=storage, run_id="cert-thermal", scenario="thermal_homeostasis"
        )
        resource = ScenarioEpisodeRunner(
            storage=storage, run_id="cert-resource", scenario="resource_management"
        )
        res_thermal = thermal.run_episode(external_input=0.04)
        res_resource = resource.run_episode(external_input=0.04)

        assert (
            res_thermal["episode"]["scenario_metadata"]["scenario_name"]
            == "thermal_homeostasis"
        )
        assert (
            res_resource["episode"]["scenario_metadata"]["scenario_name"]
            == "resource_management"
        )
        # Hashes de configuración distintos: no hay confusión de identidad.
        assert (
            res_thermal["episode"]["scenario_metadata"]["scenario_config_hash"]
            != res_resource["episode"]["scenario_metadata"]["scenario_config_hash"]
        )
        storage.close()

