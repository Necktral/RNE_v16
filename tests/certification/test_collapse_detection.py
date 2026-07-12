"""P9.6 paso 4 — el organismo puede detectar su propio colapso.

Antes (`promotion_gate.py:72-73`)::

    collapse_detected = bool(getattr(reality_assessment, "collapse_detected", False))

y en el camino vivo `reality_assessment` es SIEMPRE None (nadie lo construye: solo lo arma
el bench, `runtime/reality/service.py`). O sea: `collapse_detected` era SIEMPRE False. Y es
el ÚNICO que gatea el veredicto (`certificate_builder.py:55`): la única compuerta que el
organismo tenía estaba ciega.

Estos tests fijan las DOS mitades del contrato, que hay que sostener juntas:

1. El detector DISPARA ante un colapso real (continuidad hundida y SOSTENIDA en racha).
2. El detector NO rechaza episodios sanos — la mitad que puede matar al organismo si el
   detector resulta ruidoso.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.registry import get_scenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "collapse.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def _runner(storage, run_id: str) -> ScenarioEpisodeRunner:
    return ScenarioEpisodeRunner(
        storage=storage,
        run_id=run_id,
        scenario=get_scenario("thermal_homeostasis"),
    )


def _certificate_of(storage, episode_result):
    """El certificado REAL que quedó persistido para ese episodio."""
    return storage.get_episode_certificate(
        certificate_id=episode_result["certification"]["certificate_id"]
    )


def _poison_continuity_history(storage, *, run_id: str, count: int = 3) -> None:
    """Escribe certificados con continuidad HUNDIDA — un organismo realmente roto.

    No es un episodio feo aislado: es la continuidad por el piso, sostenida. La secuencia de
    razonamiento queda deliberadamente incomparable con la del escenario real y la variable
    principal, lejísimos, para que el `ContinuityGuard` mida de verdad una ruptura.
    """
    for i in range(count):
        storage.write_episode_certificate(
            episode_id=f"broken-{i}",
            run_id=run_id,
            trace_id=f"trace-broken-{i}",
            smg_artifacts={},
            lotf_artifacts={},
            world_artifacts={},
            continuity_score=0.0,
            ioc_proxy=0.1,
            risk_score=0.9,
            verdict="rejected",
            rollback_ready=True,
            promotion_candidate=False,
            metadata={
                "reasoning_sequence": ["ZZZ", "YYY", "XXX"],
                "world_main_variable": "temperature",
                "world_main_variable_value": 99.0,
            },
        )


def test_healthy_episode_is_not_rejected_by_collapse_wiring(tmp_path: Path):
    """LA MITAD QUE PUEDE MATAR AL ORGANISMO: cablear el colapso NO rechaza lo sano.

    Si al cablear `collapse_detected` empezaran a rechazarse episodios normales, el detector
    sería ruidoso y el organismo quedaría sin poder certificar nada. Un episodio sano tiene
    continuidad ≈ 1.0: no se acerca ni de lejos al umbral de colapso (0.35).
    """
    storage = _storage(tmp_path)
    runner = _runner(storage, "life-collapse-healthy")

    for _ in range(3):
        result = runner.run_episode(external_input=0.04)
        certificate = _certificate_of(storage, result)
        assert certificate.metadata["collapse_detected"] is False
        assert certificate.verdict == "certified"


def test_sustained_continuity_collapse_is_detected_and_rejects(tmp_path: Path):
    """El organismo detecta su propio colapso: racha de continuidad hundida ⇒ `rejected`.

    Antes esto era IMPOSIBLE: `collapse_detected` era la constante False.
    """
    storage = _storage(tmp_path)
    run_id = "life-collapse-broken"
    runner = _runner(storage, run_id)

    # Control: el primer episodio, sin historia rota, certifica.
    healthy = runner.run_episode(external_input=0.04)
    assert _certificate_of(storage, healthy).verdict == "certified"

    # El organismo se rompe: continuidad por el piso, sostenida (racha de 3).
    _poison_continuity_history(storage, run_id=run_id, count=3)

    broken = runner.run_episode(external_input=0.04)
    certificate = _certificate_of(storage, broken)

    assert certificate.continuity_score < 0.35, "el guardia de continuidad debe ver la ruptura"
    assert certificate.metadata["collapse_detected"] is True, (
        "el organismo tiene que poder detectar su propio colapso"
    )
    assert certificate.verdict == "rejected"
    assert certificate.promotion_candidate is False


def test_isolated_dip_is_not_a_collapse(tmp_path: Path):
    """Un bache AISLADO no es un colapso: el detector exige racha, no histeria.

    Complemento del test anterior: sin `streak`, cualquier episodio feo suelto tumbaría la
    certificación. El detector mide colapso SOSTENIDO.
    """
    storage = _storage(tmp_path)
    run_id = "life-collapse-dip"
    runner = _runner(storage, run_id)
    runner.run_episode(external_input=0.04)

    # Un solo certificado roto: hunde la continuidad del episodio siguiente, pero NO hay racha.
    _poison_continuity_history(storage, run_id=run_id, count=1)

    dip = runner.run_episode(external_input=0.04)
    certificate = _certificate_of(storage, dip)

    assert certificate.continuity_score < 0.35, "la continuidad cae..."
    assert certificate.metadata["collapse_detected"] is False, "...pero un bache aislado NO es colapso"
    assert certificate.verdict == "certified"
