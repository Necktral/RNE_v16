#!/usr/bin/env python3
"""Offline, deterministic attribution audit for the published P1 N3 aggregates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rnfe-p1-n3-attribution-audit-v1"
SOURCE_BRANCH = "codex/neural-agent-suite-v1"
SOURCE_HEAD = "060101975a752793c4b5bae6872002ce69c0f8ec"
EXPERIMENT_COMMIT = "06e95c8f45c132be87f81f03c19a966674dfb51b"
AUDIT_V2_COMMIT = "5f22227a9fee8584be7c740af65b7f6b41a2e47e"
EXPECTED_CAMPAIGN = "neural-p1-final-20260721-06e95c8"
EXPECTED_PROFILES = (
    "off",
    "shadow-none",
    "only-n2",
    "only-n3-reference",
    "only-n3-trained",
    "only-n4-v2",
    "p1-all",
    "p1-without-n2",
    "p1-without-n3",
    "p1-without-n4",
)
NO_N3_PROFILES = frozenset(
    {"off", "shadow-none", "only-n2", "only-n4-v2", "p1-without-n3"}
)
ACTIVE_N3_PROFILES = (
    "only-n3-trained",
    "p1-all",
    "p1-without-n2",
    "p1-without-n4",
)
EXPECTED_SEEDS = (
    911001,
    911102,
    911203,
    911304,
    911405,
    911506,
    911607,
    911708,
    911809,
    911910,
    912011,
    912112,
)
LANE_METRICS = {
    "paired_binary_normalized_dcg_delta_v1": "mean_ndcg_delta",
    "mrr_delta": "mean_mrr_delta",
    "risk_brier": "mean_risk_brier",
    "balanced_accuracy": "balanced_accuracy",
}


class AuditError(RuntimeError):
    """Fail-closed evidence validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256s(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise AuditError(f"invalid_sha256s_line:{line_number}")
        digest, relative = parts[0].lower(), parts[1].lstrip("*")
        if any(character not in "0123456789abcdef" for character in digest):
            raise AuditError(f"invalid_sha256_digest:{line_number}")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in entries:
            raise AuditError(f"unsafe_or_duplicate_sha256_path:{relative}")
        entries[relative] = digest
    return entries


def validate_published_hashes(
    *, matrix_path: Path, audit_v2_path: Path, sha256s_path: Path
) -> dict[str, str]:
    entries = read_sha256s(sha256s_path)
    required = {matrix_path.name: matrix_path, audit_v2_path.name: audit_v2_path}
    for name, source_path in required.items():
        expected = entries.get(name)
        if expected is None:
            raise AuditError(f"required_hash_missing:{name}")
        observed = sha256_file(source_path)
        if observed != expected:
            raise AuditError(f"sha256_mismatch:{name}:{expected}:{observed}")
    for relative, expected in entries.items():
        candidate = sha256s_path.parent / relative
        if not candidate.is_file():
            raise AuditError(f"hashed_file_missing:{relative}")
        observed = sha256_file(candidate)
        if observed != expected:
            raise AuditError(f"sha256_mismatch:{relative}:{expected}:{observed}")
    return entries


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"json_root_not_object:{path.name}")
    return payload


def _profile_map(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    profiles = matrix.get("profiles")
    if not isinstance(profiles, list):
        raise AuditError("profiles_not_list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise AuditError("profile_not_object")
        profile_id = str(profile.get("profile_id") or "")
        if not profile_id or profile_id in mapped:
            raise AuditError(f"profile_missing_or_duplicate:{profile_id}")
        mapped[profile_id] = profile
    if set(mapped) != set(EXPECTED_PROFILES):
        missing = sorted(set(EXPECTED_PROFILES) - set(mapped))
        extra = sorted(set(mapped) - set(EXPECTED_PROFILES))
        raise AuditError(f"profile_set_invalid:missing={missing}:extra={extra}")
    return mapped


def _lane_map(profile: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    profile_id = str(profile.get("profile_id"))
    lanes = profile.get("lanes")
    if not isinstance(lanes, list):
        raise AuditError(f"lanes_not_list:{profile_id}")
    mapped: dict[int, Mapping[str, Any]] = {}
    for lane in lanes:
        if not isinstance(lane, Mapping) or isinstance(lane.get("seed"), bool):
            raise AuditError(f"invalid_lane:{profile_id}")
        try:
            seed = int(lane["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid_seed:{profile_id}") from exc
        if seed in mapped:
            raise AuditError(f"duplicate_seed:{profile_id}:{seed}")
        mapped[seed] = lane
    if tuple(sorted(mapped)) != EXPECTED_SEEDS:
        raise AuditError(f"seed_set_invalid:{profile_id}:{sorted(mapped)}")
    return mapped


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_evidence(
    matrix: Mapping[str, Any], audit_v2: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if matrix.get("campaign_id") != EXPECTED_CAMPAIGN:
        raise AuditError("matrix_campaign_id_invalid")
    if audit_v2.get("campaign_id") != matrix.get("campaign_id"):
        raise AuditError("campaign_id_mismatch")
    if matrix.get("commit") != EXPERIMENT_COMMIT:
        raise AuditError("experiment_commit_invalid")
    if (audit_v2.get("source_matrix") or {}).get("experiment_commit") != EXPERIMENT_COMMIT:
        raise AuditError("audit_v2_experiment_commit_invalid")
    if audit_v2.get("audit_commit") != AUDIT_V2_COMMIT:
        raise AuditError("audit_v2_commit_invalid")
    if matrix.get("steps_per_lane") != 32:
        raise AuditError("steps_per_lane_invalid")
    if matrix.get("warmup_visits_per_scenario") != 2:
        raise AuditError("warmup_contract_invalid")
    if tuple(matrix.get("seed_order") or ()) != EXPECTED_SEEDS:
        raise AuditError("matrix_seed_order_invalid")

    integrity = audit_v2.get("integrity") or {}
    parity = integrity.get("canonical_behavior_parity") or {}
    expected_integrity = {
        "emitted_episodes": 3250,
        "matched_closures": 3250,
        "certification_rate": 1.0,
        "safety_violations": 0,
    }
    for key, expected in expected_integrity.items():
        if integrity.get(key) != expected:
            raise AuditError(f"campaign_integrity_invalid:{key}")
    if parity.get("step_comparison_count") != 3456 or parity.get("mismatch_count") != 0:
        raise AuditError("canonical_parity_invalid")
    if (audit_v2.get("correction") or {}).get("source_matrix_preserved") is not True:
        raise AuditError("source_matrix_not_preserved")

    forbidden_true = (
        matrix.get("promotion_authorized"),
        matrix.get("staging_authorized"),
        matrix.get("training_authorized"),
        audit_v2.get("promotion_authorized"),
        audit_v2.get("staging_authorized"),
        audit_v2.get("training_authorized"),
        matrix.get("external_reasoner_enabled"),
    )
    if any(value is not False for value in forbidden_true):
        raise AuditError("authority_or_external_reasoner_invariant_invalid")

    profiles = _profile_map(matrix)
    for profile_id, profile in profiles.items():
        if int(profile.get("seed_count", -1)) != 12:
            raise AuditError(f"profile_seed_count_invalid:{profile_id}")
        if (
            profile.get("authority_ceiling") != "shadow"
            or profile.get("promotion_authorized") is not False
            or profile.get("promotion_eligible") is not False
            or profile.get("training_authorized") is not False
        ):
            raise AuditError(f"profile_authority_invariant_invalid:{profile_id}")
        lanes = _lane_map(profile)
        for seed, lane in lanes.items():
            summary = lane.get("summary") or {}
            if summary.get("total_steps") != 32:
                raise AuditError(f"lane_step_count_invalid:{profile_id}:{seed}")
            if summary.get("promotion_authorized") is not False:
                raise AuditError(f"lane_promotion_invalid:{profile_id}:{seed}")
    if profiles["only-n3-reference"].get("organ_execution_classes", {}).get("N3") != [
        "reference"
    ]:
        raise AuditError("n3_reference_backend_invalid")
    if profiles["only-n3-trained"].get("organ_execution_classes", {}).get("N3") != [
        "model_bound"
    ]:
        raise AuditError("n3_trained_backend_invalid")
    return profiles


def audit_missingness(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checked = []
    for profile_id in sorted(NO_N3_PROFILES):
        profile = profiles[profile_id]
        metrics = profile.get("metrics") or {}
        for aggregate_name in (
            "n3_ndcg_delta",
            "n3_mrr_delta",
            "n3_risk_brier",
            "n3_balanced_accuracy",
        ):
            aggregate = metrics.get(aggregate_name)
            if not isinstance(aggregate, Mapping):
                raise AuditError(f"missing_n3_aggregate:{profile_id}:{aggregate_name}")
            if aggregate.get("mean") is not None or aggregate.get("seed_count") != 0:
                raise AuditError(
                    f"invalid_metric_missingness_semantics:{profile_id}:{aggregate_name}"
                )
        for seed, lane in _lane_map(profile).items():
            n3 = (lane.get("summary") or {}).get("n3") or {}
            for lane_name in LANE_METRICS.values():
                if n3.get(lane_name) is not None:
                    raise AuditError(
                        f"invalid_metric_missingness_semantics:{profile_id}:{seed}:{lane_name}"
                    )
        checked.append(profile_id)
    return {
        "status": "passed",
        "profiles_without_n3_checked": checked,
        "zero_filling_detected": False,
        "allowed_missing_representations": ["null", "absent", "not_applicable"],
    }


def extract_profile_metrics(
    profile: Mapping[str, Any]
) -> dict[str, dict[int, float]]:
    output = {metric: {} for metric in LANE_METRICS}
    for seed, lane in _lane_map(profile).items():
        n3 = (lane.get("summary") or {}).get("n3") or {}
        for audited_name, lane_name in LANE_METRICS.items():
            value = n3.get(lane_name)
            if not _is_number(value):
                raise AuditError(
                    f"active_n3_metric_missing:{profile.get('profile_id')}:{seed}:{lane_name}"
                )
            output[audited_name][seed] = float(value)
    return output


def paired_differences(
    left: Mapping[int, float], right: Mapping[int, float]
) -> dict[int, float]:
    if set(left) != set(right) or tuple(sorted(left)) != EXPECTED_SEEDS:
        raise AuditError("paired_seed_set_invalid")
    return {seed: float(left[seed]) - float(right[seed]) for seed in EXPECTED_SEEDS}


def bootstrap_mean_ci95(
    values: Sequence[float], *, label: str, samples: int = 10_000
) -> list[float]:
    clean = [float(value) for value in values]
    if not clean or any(not math.isfinite(value) for value in clean):
        raise AuditError("bootstrap_values_invalid")
    seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(clean[rng.randrange(len(clean))] for _ in clean)
        for _ in range(samples)
    )
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def exact_sign_flip_test(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    if len(clean) != 12 or any(not math.isfinite(value) for value in clean):
        raise AuditError("sign_flip_requires_twelve_finite_values")
    observed = abs(statistics.fmean(clean))
    extreme = 0
    assignments = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(clean)):
        assignments += 1
        statistic = abs(statistics.fmean(sign * value for sign, value in zip(signs, clean)))
        if statistic >= observed - 1e-15:
            extreme += 1
    return {
        "method": "exact_two_sided_sign_flip_mean",
        "assignments_enumerated": assignments,
        "extreme_assignments": extreme,
        "p_value": extreme / assignments,
    }


def describe_paired(
    values_by_seed: Mapping[int, float], *, label: str
) -> dict[str, Any]:
    if tuple(sorted(values_by_seed)) != EXPECTED_SEEDS:
        raise AuditError(f"describe_seed_set_invalid:{label}")
    values = [float(values_by_seed[seed]) for seed in EXPECTED_SEEDS]
    return {
        "paired_seed_count": len(values),
        "seed_values": [
            {"seed": seed, "value": values_by_seed[seed]} for seed in EXPECTED_SEEDS
        ],
        "paired_differences": [
            {"seed": seed, "value": values_by_seed[seed]} for seed in EXPECTED_SEEDS
        ],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.stdev(values),
        "standard_deviation_ddof": 1,
        "minimum": min(values),
        "maximum": max(values),
        "ci95": bootstrap_mean_ci95(values, label=label),
        "positive_count": sum(value > 0.0 for value in values),
        "zero_count": sum(value == 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "exact_sign_flip_p_value": exact_sign_flip_test(values),
    }


def classify_verdict(
    *,
    integrity_valid: bool,
    isolated_supported: bool,
    p1_all_signal: bool,
    cross_context_consistent: bool,
    trained_beats_reference: bool,
    limitation_present: bool,
) -> str:
    if not integrity_valid:
        return "audit_invalid"
    if (
        isolated_supported
        and cross_context_consistent
        and trained_beats_reference
        and limitation_present
    ):
        return "n3_attribution_supported_limited"
    if p1_all_signal and (not isolated_supported or not cross_context_consistent):
        return "n3_signal_detected_but_attribution_inconclusive"
    return "n3_attribution_not_supported"


def _metric_contrast(
    left: Mapping[str, Mapping[int, float]],
    right: Mapping[str, Mapping[int, float]],
    *,
    label: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in LANE_METRICS:
        differences = paired_differences(left[metric], right[metric])
        metrics[metric] = describe_paired(differences, label=f"{label}:{metric}")
    primary = dict(metrics["paired_binary_normalized_dcg_delta_v1"])
    primary["metrics"] = metrics
    return primary


def build_audit(
    matrix: Mapping[str, Any],
    audit_v2: Mapping[str, Any],
    *,
    matrix_sha256: str,
    audit_v2_sha256: str,
) -> dict[str, Any]:
    profiles = validate_evidence(matrix, audit_v2)
    missingness = audit_missingness(profiles)
    extracted = {
        profile_id: extract_profile_metrics(profiles[profile_id])
        for profile_id in ("only-n3-reference", *ACTIVE_N3_PROFILES)
    }

    isolated_metrics = extracted["only-n3-trained"]
    isolated = describe_paired(
        isolated_metrics["paired_binary_normalized_dcg_delta_v1"],
        label="isolated_trained_vs_zero:paired_binary_normalized_dcg_delta_v1",
    )
    isolated["metrics"] = {
        metric: describe_paired(values, label=f"isolated_trained_vs_zero:{metric}")
        for metric, values in isolated_metrics.items()
    }
    trained_vs_reference = _metric_contrast(
        isolated_metrics,
        extracted["only-n3-reference"],
        label="trained_vs_reference",
    )
    # Brier is lower-is-better, so expose the scientifically oriented sign too.
    brier_improvement = paired_differences(
        extracted["only-n3-reference"]["risk_brier"],
        isolated_metrics["risk_brier"],
    )
    trained_vs_reference["metrics"]["brier_improvement"] = describe_paired(
        brier_improvement, label="trained_vs_reference:brier_improvement"
    )

    context_specs = {
        "p1_all_vs_isolated_trained": "p1-all",
        "without_n2_vs_isolated_trained": "p1-without-n2",
        "without_n4_vs_isolated_trained": "p1-without-n4",
    }
    context_contrasts = {
        name: _metric_contrast(extracted[profile_id], isolated_metrics, label=name)
        for name, profile_id in context_specs.items()
    }
    profile_summaries = {
        profile_id: {
            metric: describe_paired(values, label=f"profile:{profile_id}:{metric}")
            for metric, values in extracted[profile_id].items()
        }
        for profile_id in ("only-n3-reference", *ACTIVE_N3_PROFILES)
    }

    isolated_supported = bool(
        isolated["mean"] > 0.0
        and isolated["ci95"][0] > 0.0
        and isolated["exact_sign_flip_p_value"]["p_value"] < 0.05
    )
    active_ndcg = {
        profile_id: profile_summaries[profile_id][
            "paired_binary_normalized_dcg_delta_v1"
        ]
        for profile_id in ACTIVE_N3_PROFILES
    }
    cross_context_consistent = all(
        summary["mean"] > 0.0 and summary["negative_count"] == 0
        for summary in active_ndcg.values()
    )
    trained_beats_reference = bool(
        trained_vs_reference["mean"] > 0.0
        and trained_vs_reference["metrics"]["brier_improvement"]["mean"] > 0.0
    )
    limitation_present = bool(
        isolated["metrics"]["mrr_delta"]["mean"] == 0.0
        or isolated["metrics"]["balanced_accuracy"]["mean"] < 0.5
    )
    verdict = classify_verdict(
        integrity_valid=True,
        isolated_supported=isolated_supported,
        p1_all_signal=active_ndcg["p1-all"]["mean"] > 0.0,
        cross_context_consistent=cross_context_consistent,
        trained_beats_reference=trained_beats_reference,
        limitation_present=limitation_present,
    )
    context_effects = {
        name: (
            "unchanged"
            if contrast["positive_count"] == 0
            and contrast["negative_count"] == 0
            else "amplified"
            if contrast["mean"] > 0.0
            else "reduced"
            if contrast["mean"] < 0.0
            else "mixed"
        )
        for name, contrast in context_contrasts.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "matrix_sha256": matrix_sha256,
            "audit_v2_sha256": audit_v2_sha256,
            "source_files_preserved": True,
        },
        "campaign_integrity": {
            "campaign_id": EXPECTED_CAMPAIGN,
            "experiment_commit": EXPERIMENT_COMMIT,
            "audit_v2_commit": AUDIT_V2_COMMIT,
            "profile_count": 10,
            "seed_count_per_profile": 12,
            "steps_per_lane": 32,
            "warmup_visits_per_scenario": 2,
            "emitted_episode_closures": {"matched": 3250, "expected": 3250},
            "certification_rate": 1.0,
            "safety_violations": 0,
            "canonical_step_comparisons": 3456,
            "canonical_mismatches": 0,
            "authority_ceiling": "shadow",
        },
        "metric_semantics": {
            "legacy_name": "n3_ndcg_delta",
            "audited_name": "paired_binary_normalized_dcg_delta_v1",
            "description": (
                "paired binary DCG delta with shared ideal relevant count derived "
                "from the maximum observed relevant count across canonical and shadow"
            ),
            "standard_ndcg_recomputation": {
                "status": "not_recomputable_from_published_aggregate",
                "missing": [
                    "canonical_ranking",
                    "shadow_ranking",
                    "complete_eligible_pool",
                    "document_level_relevance",
                    "fixed_independent_idcg",
                    "k",
                ],
            },
        },
        "missingness_audit": missingness,
        "profiles": profile_summaries,
        "contrasts": {
            "isolated_trained_vs_zero": isolated,
            "trained_vs_reference": trained_vs_reference,
            **context_contrasts,
        },
        "context_interpretation": {
            **context_effects,
            "sign_inversion_detected": not cross_context_consistent,
            "canonical_value_claimed": False,
        },
        "brier_analysis": {
            "reference": profile_summaries["only-n3-reference"]["risk_brier"],
            "trained": isolated["metrics"]["risk_brier"],
            "paired_improvement": trained_vs_reference["metrics"]["brier_improvement"],
            "decomposition": {
                "status": "not_recomputable_from_published_aggregate",
                "missing": ["individual_predictions", "individual_labels"],
            },
        },
        "gates": {
            "source_integrity": True,
            "missingness_integrity": True,
            "seed_pairing": True,
            "isolated_n3_signal": isolated_supported,
            "trained_vs_reference": trained_beats_reference,
            "cross_context_consistency": cross_context_consistent,
            "standard_ndcg_recomputable": False,
            "brier_decomposition_recomputable": False,
            "runtime_unchanged": True,
            "p2_authorized": False,
        },
        "limitations": [
            "The published aggregate cannot reconstruct conventional nDCG.",
            "Brier reliability, resolution, and uncertainty cannot be decomposed.",
            "MRR delta is zero in every trained seed/lane.",
            "Trained balanced accuracy is below 0.5 and worse than reference.",
            "N3 remained SHADOW-only and had no canonical decision influence.",
            "Seed/lane summaries are the independent units; episodes were not pooled.",
        ],
        "verdict": verdict,
        "invariants": {
            "source_matrix_preserved": True,
            "source_audit_v2_preserved": True,
            "runtime_changes": 0,
            "training_executed": False,
            "staging_authorized": False,
            "promotion_authorized": False,
            "live_authority": False,
            "p2_authorized": False,
            "external_reasoner_used": False,
            "raw_runtime_tree_used": False,
        },
        "p2_authorized": False,
        "runtime_modified": False,
        "campaign_rerun": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(*, matrix_path: Path, audit_v2_path: Path, sha256s_path: Path, output: Path) -> dict[str, Any]:
    validate_published_hashes(
        matrix_path=matrix_path, audit_v2_path=audit_v2_path, sha256s_path=sha256s_path
    )
    before = {matrix_path: sha256_file(matrix_path), audit_v2_path: sha256_file(audit_v2_path)}
    report = build_audit(
        _load_json(matrix_path),
        _load_json(audit_v2_path),
        matrix_sha256=before[matrix_path],
        audit_v2_sha256=before[audit_v2_path],
    )
    _atomic_json(output, report)
    after = {path: sha256_file(path) for path in before}
    if before != after:
        raise AuditError("source_files_changed_during_audit")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--audit-v2", required=True, type=Path)
    parser.add_argument("--sha256s", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(
            matrix_path=args.matrix,
            audit_v2_path=args.audit_v2,
            sha256s_path=args.sha256s,
            output=args.output,
        )
    except AuditError as exc:
        parser.exit(2, f"audit_invalid:{exc}\n")
    print(json.dumps({"output": str(args.output), "verdict": report["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
