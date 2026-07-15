"""Construye currículo candidato Codex→7B desde una campaña held-out sellada."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 6) if values else 0.0


def select_curriculum_records(
    trials: list[Mapping[str, Any]],
    *,
    campaign_id: str,
    min_support: int = 20,
    min_seeds: int = 10,
    min_perturbations: int = 2,
    min_success_rate: float = 0.90,
) -> list[dict[str, Any]]:
    controls = {
        str(row["evaluation_pair_id"]): row
        for row in trials
        if row.get("variant") == "no_teacher"
    }
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in trials:
        if row.get("variant") == "codex_frontier":
            by_scenario.setdefault(str(row.get("scenario") or "unknown"), []).append(row)

    records = []
    for scenario, rows in sorted(by_scenario.items()):
        comparisons = []
        targets: Counter[str] = Counter()
        target_payloads: dict[str, dict[str, Any]] = {}
        for row in rows:
            pair_id = str(row["evaluation_pair_id"])
            control = controls.get(pair_id)
            if control is None:
                continue
            lesson = dict(row.get("lesson") or {})
            target = {
                "avoid": lesson.get("avoid"),
                "prefer": lesson.get("prefer"),
                "lesson": lesson.get("lesson"),
            }
            target_key = _canonical(target)
            targets[target_key] += 1
            target_payloads[target_key] = target
            reward_delta = round(
                float(row["evaluation"]["cumulative_reward"])
                - float(control["evaluation"]["cumulative_reward"]),
                6,
            )
            severity_delta = round(
                float(row["evaluation"]["mean_severity"])
                - float(control["evaluation"]["mean_severity"]),
                6,
            )
            semantic_pass = row.get("teacher_semantics", {}).get("semantic_pass") is True
            comparisons.append(
                {
                    "evaluation_pair_id": pair_id,
                    "seed": int(row["seed"]),
                    "perturbation_id": str(row.get("perturbation_id") or "pilot"),
                    "external_input": row.get("external_input"),
                    "reward_delta": reward_delta,
                    "severity_delta": severity_delta,
                    "semantic_pass": semantic_pass,
                    "success": semantic_pass and reward_delta > 0.0 and severity_delta <= 0.0,
                    "lesson_id": lesson.get("lesson_id"),
                }
            )
        support = len(comparisons)
        successes = sum(item["success"] for item in comparisons)
        regressions = sum(
            item["reward_delta"] < 0.0 or item["severity_delta"] > 0.0
            for item in comparisons
        )
        semantic_rate = _mean([float(item["semantic_pass"]) for item in comparisons])
        success_rate = _mean([float(item["success"]) for item in comparisons])
        seeds = sorted({item["seed"] for item in comparisons})
        perturbations = sorted({item["perturbation_id"] for item in comparisons})
        dominant_key = targets.most_common(1)[0][0] if targets else _canonical({})
        target = target_payloads.get(dominant_key, {})
        eligible = (
            support >= min_support
            and len(seeds) >= min_seeds
            and len(perturbations) >= min_perturbations
            and success_rate >= min_success_rate
            and semantic_rate == 1.0
            and regressions == 0
            and bool(target)
        )
        record = {
            "schema_version": "rnfe-teacher-curriculum-record-v1",
            "record_id": "curriculum-" + _sha256(
                {"campaign_id": campaign_id, "scenario": scenario, "target": target}
            )[:24],
            "teacher_source": "codex_frontier",
            "student_target": "open-thoughts/OpenThinker3-7B",
            "input": {
                "scenario": scenario,
                "regime": "calm",
                "valid_interventions": sorted(
                    {str(target.get("avoid") or ""), str(target.get("prefer") or "")}
                    - {""}
                ),
                "instruction": "Produce one bounded lesson after an observed harmful action.",
            },
            "target": target,
            "evidence": {
                "campaign_id": campaign_id,
                "support_count": support,
                "success_count": successes,
                "success_rate": success_rate,
                "semantic_pass_rate": semantic_rate,
                "regression_count": regressions,
                "seeds": seeds,
                "perturbations": perturbations,
                "mean_reward_delta": _mean([item["reward_delta"] for item in comparisons]),
                "mean_severity_delta": _mean([item["severity_delta"] for item in comparisons]),
                "source_lesson_ids": sorted(
                    {str(item["lesson_id"]) for item in comparisons if item.get("lesson_id")}
                ),
                "comparison_hash": _sha256(comparisons),
            },
            "training_eligible": eligible,
            "training_authorized": False,
        }
        record["record_hash"] = _sha256(record)
        records.append(record)
    return records


def build_curriculum(
    *,
    campaign_dir: Path,
    output_dir: Path,
    min_support: int = 20,
    min_seeds: int = 10,
    min_perturbations: int = 2,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    trials_payload = json.loads((campaign_dir / "trials.json").read_text(encoding="utf-8"))
    campaign_manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence_manifest = json.loads(
        (campaign_dir / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    campaign_id = str(campaign_manifest["campaign_id"])
    records = select_curriculum_records(
        list(trials_payload["trials"]),
        campaign_id=campaign_id,
        min_support=min_support,
        min_seeds=min_seeds,
        min_perturbations=min_perturbations,
    )
    eligible = [record for record in records if record["training_eligible"]]
    dataset = {
        "schema_version": "rnfe-teacher-curriculum-v1",
        "curriculum_id": output_dir.name,
        "source_campaign_id": campaign_id,
        "source_evidence_manifest_sha256": hashlib.sha256(
            _canonical(evidence_manifest).encode("utf-8")
        ).hexdigest(),
        "record_count": len(records),
        "eligible_record_count": len(eligible),
        "records": records,
    }
    _write_json(output_dir / "curriculum.json", dataset)
    (output_dir / "curriculum.jsonl").write_text(
        "".join(_canonical(record) + "\n" for record in eligible), encoding="utf-8"
    )
    all_scenarios_eligible = len(eligible) == len(records) and bool(records)
    verdict = {
        "schema_version": "rnfe-teacher-curriculum-verdict-v1",
        "curriculum_id": output_dir.name,
        "dataset_candidate_ready": all_scenarios_eligible,
        "training_authorized": False,
        "promotion_authorized": False,
        "verdict": (
            "dataset_candidate_ready_training_requires_separate_approval"
            if all_scenarios_eligible
            else "dataset_rejected_by_heldout_gate"
        ),
        "reason": "post_training_independent_evaluation_required",
    }
    _write_json(output_dir / "verdict.json", verdict)
    report = [
        f"# Currículo candidato {output_dir.name}",
        "",
        f"Fuente: `{campaign_id}`.",
        "",
        f"Registros: {len(records)} · elegibles: {len(eligible)}.",
        "",
        f"Veredicto: **{verdict['verdict']}**.",
        "",
        "Este export no autoriza entrenamiento ni promoción. Requiere una ejecución de fine-tuning separada y evaluación independiente post-entrenamiento.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    artifact_names = ("curriculum.json", "curriculum.jsonl", "verdict.json", "REPORT.md")
    _write_json(
        output_dir / "evidence_manifest.json",
        {
            "schema_version": "rnfe-teacher-curriculum-evidence-v1",
            "curriculum_id": output_dir.name,
            "artifacts": {
                name: {
                    "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
                    "bytes": (output_dir / name).stat().st_size,
                }
                for name in artifact_names
            },
        },
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-seeds", type=int, default=10)
    parser.add_argument("--min-perturbations", type=int, default=2)
    args = parser.parse_args(argv)
    result = build_curriculum(
        campaign_dir=args.campaign_dir,
        output_dir=args.output_dir,
        min_support=args.min_support,
        min_seeds=args.min_seeds,
        min_perturbations=args.min_perturbations,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
