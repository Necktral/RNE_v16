from __future__ import annotations

import json
from pathlib import Path

from scripts import benchmark_cognitive_gain_v2
from scripts import benchmark_family_causal_gain
from scripts import run_adaptive_v2_intelligence_campaign
from scripts.intelligence_campaign_lib import run_adaptive_v2_intelligence_campaign as run_campaign_lib


def test_benchmark_cognitive_gain_v2_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "cognitive_smoke.db"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "cognitive_gain"

    exit_code = benchmark_cognitive_gain_v2.main(
        [
            "--campaign-id",
            "cg_smoke",
            "--output-root",
            str(output_root),
            "--db-path",
            str(db_path),
            "--artifact-root",
            str(artifact_root),
            "--blocks",
            "1",
            "--episodes-per-block",
            "1",
            "--bootstrap-samples",
            "64",
        ]
    )

    assert exit_code == 0
    regime_path = output_root / "cg_smoke" / "regime_cognitive_verdicts.json"
    report_path = output_root / "cg_smoke" / "cognitive_gain_report.md"
    assert regime_path.exists()
    assert report_path.exists()
    payload = json.loads(regime_path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "cg_smoke"
    assert len(payload["regime_records"]) == 5
    assert Path(output_root / "cg_smoke" / "block_metrics.jsonl").exists()


def test_benchmark_family_causal_gain_smoke(tmp_path: Path) -> None:
    db_path = tmp_path / "family_smoke.db"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "family_causal_gain"

    exit_code = benchmark_family_causal_gain.main(
        [
            "--campaign-id",
            "fcg_smoke",
            "--output-root",
            str(output_root),
            "--db-path",
            str(db_path),
            "--artifact-root",
            str(artifact_root),
            "--blocks",
            "1",
            "--episodes-per-block",
            "1",
            "--bootstrap-samples",
            "64",
        ]
    )

    assert exit_code == 0
    matrix_path = output_root / "fcg_smoke" / "family_regime_matrix.json"
    report_path = output_root / "fcg_smoke" / "family_regime_matrix.md"
    assert matrix_path.exists()
    assert report_path.exists()
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == "fcg_smoke"
    assert sorted(payload["regime_matrix"].keys()) == [
        "heterogeneous_elevated",
        "heterogeneous_warning",
        "viability_edge",
        "vram_favorable",
    ]


def test_wrapper_sequence_runs_prompt1_and_prompt3_and_skips_prompt2_when_not_needed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_prompt1(**kwargs):
        calls.append("prompt1")
        root_dir = tmp_path / "cognitive" / "campaign_prompt1"
        root_dir.mkdir(parents=True, exist_ok=True)
        verdict_path = root_dir / "regime_cognitive_verdicts.json"
        verdict_path.write_text(
            json.dumps(
                {
                    "primary_verdict": "no hay ganancia cognitiva suficiente",
                    "strong_regimes": [],
                    "conditioned_regimes": [],
                    "closure_regressions": [],
                }
            ),
            encoding="utf-8",
        )
        return {
            "campaign_id": "campaign_prompt1",
            "regime_cognitive_verdicts_path": str(verdict_path),
            "should_run_family_causal": False,
        }

    def fake_prompt2(**kwargs):
        calls.append("prompt2")
        raise AssertionError("prompt2 no debe ejecutarse")

    def fake_prompt3(**kwargs):
        calls.append("prompt3")
        assert kwargs["family_causal_path"] is None
        return {
            "campaign_id": "campaign_prompt3",
            "dictamen_principal": "hubo mejora estructural pero no cognitiva",
            "intelligence_verdict_path": str(tmp_path / "verdict.json"),
            "intelligence_verdict_md_path": str(tmp_path / "verdict.md"),
        }

    monkeypatch.setattr("scripts.intelligence_campaign_lib.run_cognitive_gain_campaign", fake_prompt1)
    monkeypatch.setattr("scripts.intelligence_campaign_lib.run_family_causal_gain_campaign", fake_prompt2)
    monkeypatch.setattr("scripts.intelligence_campaign_lib.render_intelligence_verdict", fake_prompt3)

    result = run_campaign_lib(
        campaign_id="wrapper_smoke",
        cognitive_output_root=tmp_path / "cognitive",
        family_output_root=tmp_path / "family",
        verdict_output_root=tmp_path / "verdicts",
        db_path=tmp_path / "wrapper.db",
        artifact_root=tmp_path / "artifacts",
        blocks=1,
        episodes_per_block=1,
        bootstrap_samples=16,
    )

    assert calls == ["prompt1", "prompt3"]
    assert result["prompt_2"] is None

    cli_exit = run_adaptive_v2_intelligence_campaign.main(
        [
            "--campaign-id",
            "wrapper_cli",
            "--cognitive-output-root",
            str(tmp_path / "cognitive_cli"),
            "--family-output-root",
            str(tmp_path / "family_cli"),
            "--verdict-output-root",
            str(tmp_path / "verdict_cli"),
            "--db-path",
            str(tmp_path / "wrapper_cli.db"),
            "--artifact-root",
            str(tmp_path / "artifacts_cli"),
            "--blocks",
            "1",
            "--episodes-per-block",
            "1",
            "--bootstrap-samples",
            "16",
        ]
    )
    assert cli_exit == 0
