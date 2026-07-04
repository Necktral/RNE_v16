from pathlib import Path

from runtime.reasoning.families.eml_sr import execute as eml_execute
from runtime.storage import StorageConfig, StorageFactory
from runtime.symbolic.eml import EMLRunner, EMLRunnerConfig
from runtime.symbolic.eml.safe_eval import DomainError, safe_eval
from runtime.symbolic.eml.scoring import score_candidate
from runtime.symbolic.eml.search import SearchLimits, generate_candidates
from runtime.symbolic.eml.tree import ExprNode


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "eml.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def test_eml_tree_and_safe_eval():
    expr = ExprNode(
        op="add",
        left=ExprNode(op="var", name="x"),
        right=ExprNode(op="const", value=1.0),
    )
    assert expr.depth() == 2
    value = safe_eval(expr, {"x": 2.0})
    assert value == 3.0
    loaded = ExprNode.from_dict(expr.to_dict())
    assert loaded.op == expr.op


def test_eml_safe_eval_domain_guard():
    expr = ExprNode(op="log1p", left=ExprNode(op="const", value=-2.0))
    try:
        safe_eval(expr, {})
    except DomainError:
        assert True
    else:
        assert False


def test_eml_search_limits_and_scoring():
    limits = SearchLimits(max_depth=3, max_candidates=20, max_evals=60, seed=7)
    candidates = generate_candidates(var_names=["x", "cf"], limits=limits)
    assert 0 < len(candidates) <= 20
    rows = [{"x": 0.8, "cf": 0.85, "y": 0.81}, {"x": 0.82, "cf": 0.88, "y": 0.8}]
    score = score_candidate(candidates[0], rows)
    assert 0.0 <= score.domain_valid_ratio <= 1.0
    assert score.composite_score >= 0.0


def test_eml_family_execute_contract(tmp_path: Path):
    storage = _storage(tmp_path)
    out = eml_execute(
        {
            "run_id": "run-eml-family",
            "episode_id": "episode-eml-family",
            "eml_mode": "shadow",
            "storage": storage,
            "eml_dataset": [
                {"x": 0.8, "cf": 0.85, "y": 0.81},
                {"x": 0.82, "cf": 0.88, "y": 0.8},
                {"x": 0.84, "cf": 0.9, "y": 0.79},
            ],
        }
    )
    assert out["status"] == "ok"
    assert "state_delta" in out
    assert "confidence" in out
    assert "cost" in out
    assert "candidate_count" in out
    events = storage.list_events(run_id="run-eml-family", limit=50)
    assert any(item.event_type == "eml.run.completed" for item in events)
    artifacts = storage.list_artifacts(run_id="run-eml-family", kind="eml_report", limit=10)
    assert artifacts
    storage.close()


def test_eml_runner_shadow_persists_report_and_trace(tmp_path: Path):
    storage = _storage(tmp_path)
    runner = EMLRunner(
        storage=storage,
        config=EMLRunnerConfig(max_depth=2, max_evals=40, max_candidates=12, seed=3, top_k=3),
    )
    out = runner.run_shadow(
        run_id="run-eml-shadow",
        episode_id="episode-eml-shadow",
        rows=[
            {"x": 0.8, "cf": 0.86, "y": 0.81},
            {"x": 0.82, "cf": 0.88, "y": 0.80},
            {"x": 0.85, "cf": 0.90, "y": 0.79},
        ],
    )
    assert out["run"]["candidate_count"] > 0
    assert Path(out["artifacts"]["eml_report"]["abs_path"]).exists()
    assert Path(out["artifacts"]["eml_candidate_trace"]["abs_path"]).exists()
    storage.close()

