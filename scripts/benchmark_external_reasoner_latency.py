#!/usr/bin/env python3
"""Microbenchmark lab-only para latencia de EXT_OPEN_THINKER gobernado.

No toca ScenarioEpisodeRunner ni activa el razonador externo en runtime nominal.
Solo compara variantes sobre causal_counterfactual_conflict usando el gate v1,
schema, guard y fallback del benchmark experimental existente.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_external_reasoner_gain import (  # noqa: E402
    GATED_PROFILE,
    _jsonable,
    _mean,
    _p95,
    _safe_float,
    run_episode,
)


REGIME = "causal_counterfactual_conflict"
ALLOWED_INTERVENTIONS = {"activate_cooling", "deactivate_cooling"}
BASELINE = {
    "latency_mean_s": 96.115,
    "latency_p95_s": 98.953,
    "generation_tps_mean": 44.138,
    "corrected_core_failure_rate": 0.875,
    "cost_per_corrected_failure_s": 109.846,
    "ivc_r": 0.231401,
    "precision": 0.067784,
    "viability": 0.029650,
    "success": 0.875,
    "closure": 0.875,
}


@dataclass(frozen=True)
class LatencyVariant:
    name: str
    max_tokens: int
    prompt_style: str = "standard"
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    mlock: bool = False

    def state_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {
            "external_reasoner_max_tokens": self.max_tokens,
            "external_reasoner_prompt_style": self.prompt_style,
        }
        optional = {
            "external_reasoner_ctx_size": self.ctx_size,
            "external_reasoner_batch_size": self.batch_size,
            "external_reasoner_ubatch_size": self.ubatch_size,
            "external_reasoner_threads": self.threads,
            "external_reasoner_threads_batch": self.threads_batch,
        }
        overrides.update({key: value for key, value in optional.items() if value is not None})
        if self.mlock:
            overrides["external_reasoner_mlock"] = True
        return overrides

    def params_label(self) -> str:
        parts = [f"prompt={self.prompt_style}", "no_warmup=true"]
        if self.ctx_size:
            parts.append(f"ctx={self.ctx_size}")
        if self.batch_size:
            parts.append(f"batch={self.batch_size}")
        if self.ubatch_size:
            parts.append(f"ubatch={self.ubatch_size}")
        if self.threads:
            parts.append(f"threads={self.threads}")
        if self.threads_batch:
            parts.append(f"threads_batch={self.threads_batch}")
        if self.mlock:
            parts.append("mlock=true")
        return ",".join(parts)


DEFAULT_VARIANTS = [
    LatencyVariant("tokens_256_standard", max_tokens=256),
    LatencyVariant("tokens_192_standard", max_tokens=192),
    LatencyVariant("tokens_128_standard", max_tokens=128),
    LatencyVariant("tokens_96_standard", max_tokens=96),
    LatencyVariant(
        "tokens_128_compact_ctx1024",
        max_tokens=128,
        prompt_style="compact",
        ctx_size=1024,
        batch_size=128,
        ubatch_size=64,
    ),
    LatencyVariant(
        "tokens_96_compact_ctx1024",
        max_tokens=96,
        prompt_style="compact",
        ctx_size=1024,
        batch_size=128,
        ubatch_size=64,
    ),
]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _select_variants(names: Iterable[str] | None = None) -> List[LatencyVariant]:
    by_name = {variant.name: variant for variant in DEFAULT_VARIANTS}
    if not names:
        return list(DEFAULT_VARIANTS)
    selected: List[LatencyVariant] = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"unknown_latency_variant:{name}")
        selected.append(by_name[name])
    if len(selected) > 6:
        raise ValueError("too_many_latency_variants:max_6")
    return selected


def _called_rows(rows: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in rows if row.get("external_reasoner_called")]


def _summarize_variant(rows: List[Mapping[str, Any]], variant: LatencyVariant) -> Dict[str, Any]:
    called = _called_rows(rows)
    called_count = len(called)
    accepted = [row for row in called if row.get("external_accepted")]
    corrected = [row for row in called if row.get("external_reasoner_corrected_core_failure")]
    invalid_accepted = [
        row
        for row in called
        if row.get("external_accepted") and row.get("selected_intervention") not in ALLOWED_INTERVENTIONS
    ]
    latencies = [_safe_float(row.get("external_reasoner_latency_s"), 0.0) for row in called]
    summary = {
        "variant": variant.name,
        "max_tokens": variant.max_tokens,
        "ctx": variant.ctx_size,
        "params": variant.params_label(),
        "episodes": len(rows),
        "call_rate": called_count / len(rows) if rows else 0.0,
        "latency_mean": _mean(latencies),
        "latency_p95": _p95(latencies),
        "prompt_tps_mean": _mean(
            _safe_float(row.get("external_reasoner_prompt_tps"), 0.0) for row in called
        ),
        "generation_tps_mean": _mean(
            _safe_float(row.get("external_reasoner_generation_tps"), 0.0) for row in called
        ),
        "prompt_bytes_mean": _mean(
            _safe_float(row.get("external_reasoner_prompt_bytes"), 0.0) for row in called
        ),
        "external_reasoner_ok_rate": (
            sum(1 for row in called if row.get("external_reasoner_ok")) / called_count
            if called_count
            else 0.0
        ),
        "schema_validated_rate": (
            sum(1 for row in called if row.get("external_reasoner_schema_validated")) / called_count
            if called_count
            else 0.0
        ),
        "guard_pass_rate": len(accepted) / called_count if called_count else 0.0,
        "guard_reject_rate": (called_count - len(accepted)) / called_count if called_count else 0.0,
        "corrected_core_failure_rate": len(corrected) / called_count if called_count else 0.0,
        "ivc_r": _mean(_safe_float(row.get("ivc_r"), 0.0) for row in rows),
        "precision": _mean(_safe_float(row.get("intervention_precision"), 0.0) for row in rows),
        "viability": _mean(_safe_float(row.get("viability_margin"), 0.0) for row in rows),
        "success": _mean(_safe_float(row.get("success_rate"), 0.0) for row in rows),
        "closure": _mean(1.0 if row.get("closure_stable") else 0.0 for row in rows),
        "delta_ivc_r_vs_core_mean": _mean(
            _safe_float(row.get("external_reasoner_net_ivc_delta"), 0.0) for row in rows
        ),
        "cost_per_corrected_failure": (
            sum(latencies) / len(corrected) if corrected else 0.0
        ),
        "invalid_accepted_count": len(invalid_accepted),
    }
    summary["dictamen"] = _decide_variant(summary)
    return summary


def _unsafe_row_reason(row: Mapping[str, Any]) -> str | None:
    if not row.get("external_reasoner_ok"):
        return "external_reasoner_not_ok"
    if not row.get("external_reasoner_schema_validated"):
        return "schema_not_validated"
    if row.get("guard_reason") != "guard_passed":
        return "guard_rejected:" + str(row.get("guard_reason") or "unknown")
    if not row.get("external_reasoner_corrected_core_failure"):
        return "core_failure_not_corrected"
    if row.get("selected_intervention") not in ALLOWED_INTERVENTIONS:
        return "invalid_intervention_accepted"
    return None


def _decide_variant(summary: Mapping[str, Any]) -> str:
    failures: List[str] = []
    if _safe_float(summary.get("latency_mean"), 0.0) > BASELINE["latency_mean_s"] * 0.80:
        failures.append("latency_reduction_lt_20pct")
    if _safe_float(summary.get("external_reasoner_ok_rate"), 0.0) < 0.95:
        failures.append("ok_rate_lt_0.95")
    if _safe_float(summary.get("schema_validated_rate"), 0.0) < 0.95:
        failures.append("schema_rate_lt_0.95")
    if _safe_float(summary.get("guard_pass_rate"), 0.0) < 0.80:
        failures.append("guard_pass_lt_0.80")
    if _safe_float(summary.get("corrected_core_failure_rate"), 0.0) < 0.80:
        failures.append("corrected_rate_lt_0.80")
    if _safe_float(summary.get("delta_ivc_r_vs_core_mean"), 0.0) <= 0.0:
        failures.append("delta_ivc_r_not_positive")
    if int(summary.get("invalid_accepted_count", 0) or 0) > 0:
        failures.append("invalid_intervention_accepted")
    return "passes" if not failures else "fails:" + ",".join(failures)


def _decide_campaign(variant_summaries: List[Mapping[str, Any]]) -> str:
    passing = [row for row in variant_summaries if row.get("dictamen") == "passes"]
    if passing:
        return "latency_optimized_without_cognitive_loss"

    reduced = [
        row for row in variant_summaries
        if _safe_float(row.get("latency_mean"), 0.0) <= BASELINE["latency_mean_s"] * 0.80
    ]
    if reduced:
        return "latency_reduced_with_cognitive_tradeoff"

    successful_calls = [
        row for row in variant_summaries
        if _safe_float(row.get("external_reasoner_ok_rate"), 0.0) >= 0.95
        and _safe_float(row.get("schema_validated_rate"), 0.0) >= 0.95
    ]
    best_latency = min((_safe_float(row.get("latency_mean"), 0.0) for row in successful_calls), default=0.0)
    if best_latency > 60.0:
        return "server_mode_required"
    return "no_safe_latency_gain"


def _best_variant(variant_summaries: List[Mapping[str, Any]]) -> Dict[str, Any] | None:
    if not variant_summaries:
        return None
    passing = [row for row in variant_summaries if row.get("dictamen") == "passes"]
    candidates = passing or variant_summaries
    return dict(
        min(
            candidates,
            key=lambda row: (
                _safe_float(row.get("cost_per_corrected_failure"), 1e12) or 1e12,
                _safe_float(row.get("latency_mean"), 1e12),
            ),
        )
    )


def _latency_profile(variant_summaries: List[Mapping[str, Any]]) -> Dict[str, Any]:
    best = _best_variant(variant_summaries) or {}
    best_latency = _safe_float(best.get("latency_mean"), 0.0)
    latency_drop = BASELINE["latency_mean_s"] - best_latency
    dominant = "unknown"
    if best_latency > BASELINE["latency_mean_s"] * 0.80:
        dominant = "subprocess_model_load_or_process_start_likely"
    elif _safe_float(best.get("generation_tps_mean"), 0.0) <= 0.0:
        dominant = "generation_or_schema_failure"
    else:
        dominant = "generation_prompt_params_helped"
    return {
        "dominant_cost_inference": dominant,
        "baseline_latency_mean_s": BASELINE["latency_mean_s"],
        "best_latency_mean_s": best_latency,
        "best_latency_drop_s": latency_drop,
        "best_latency_drop_fraction": latency_drop / BASELINE["latency_mean_s"] if best_latency else 0.0,
        "best_prompt_tps_mean": _safe_float(best.get("prompt_tps_mean"), 0.0),
        "best_generation_tps_mean": _safe_float(best.get("generation_tps_mean"), 0.0),
        "best_prompt_bytes_mean": _safe_float(best.get("prompt_bytes_mean"), 0.0),
        "available_timings": ["elapsed", "prompt_tps", "generation_tps", "prompt_bytes"],
    }


def run_latency_campaign(
    *,
    campaign_id: str,
    output_root: Path,
    episodes: int,
    external_input: float,
    backend: str | None,
    allow_cpu_fallback: bool,
    confidence_threshold: float = 0.55,
    variants: List[LatencyVariant] | None = None,
    external_client: Any | None = None,
    abort_unsafe_variant: bool = True,
) -> Dict[str, Any]:
    selected_variants = variants or list(DEFAULT_VARIANTS)
    all_rows: List[Dict[str, Any]] = []
    variant_summaries: List[Dict[str, Any]] = []
    out_dir = output_root / campaign_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latency_variants.jsonl").write_text("", encoding="utf-8")

    for variant in selected_variants:
        rows: List[Dict[str, Any]] = []
        abort_reason: str | None = None
        print(
            json.dumps(
                {
                    "event": "variant_start",
                    "campaign_id": campaign_id,
                    "variant": variant.name,
                    "episodes": episodes,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for episode_index in range(episodes):
            row = run_episode(
                profile=GATED_PROFILE,
                regime=REGIME,
                episode_index=episode_index,
                external_input=external_input,
                backend=backend,
                allow_cpu_fallback=allow_cpu_fallback,
                confidence_threshold=confidence_threshold,
                external_client=external_client,
                external_state_overrides=variant.state_overrides(),
            )
            row["latency_variant"] = variant.name
            row["latency_variant_params"] = variant.params_label()
            unsafe_reason = _unsafe_row_reason(row)
            if unsafe_reason:
                row["latency_variant_unsafe_reason"] = unsafe_reason
            rows.append(row)
            all_rows.append(row)
            _write_rows_jsonl(rows=all_rows, out_dir=out_dir)
            print(
                json.dumps(
                    {
                        "event": "episode_complete",
                        "variant": variant.name,
                        "episode_index": episode_index,
                        "latency_s": row.get("external_reasoner_latency_s"),
                        "ok": row.get("external_reasoner_ok"),
                        "schema": row.get("external_reasoner_schema_validated"),
                        "guard": row.get("guard_reason"),
                        "selected": row.get("selected_intervention"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if abort_unsafe_variant and unsafe_reason:
                abort_reason = unsafe_reason
                print(
                    json.dumps(
                        {
                            "event": "variant_abort",
                            "variant": variant.name,
                            "episode_index": episode_index,
                            "reason": abort_reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
        variant_summary = _summarize_variant(rows, variant)
        variant_summary["aborted"] = abort_reason is not None
        variant_summary["abort_reason"] = abort_reason
        variant_summaries.append(variant_summary)
        partial_summary = _build_summary(
            campaign_id=campaign_id,
            output_dir=out_dir,
            selected_variants=selected_variants,
            variant_summaries=variant_summaries,
            all_rows=all_rows,
            episodes=episodes,
            external_input=external_input,
            backend=backend,
            allow_cpu_fallback=allow_cpu_fallback,
            confidence_threshold=confidence_threshold,
            abort_unsafe_variant=abort_unsafe_variant,
        )
        write_summary_outputs(summary=partial_summary, out_dir=out_dir)

    summary = _build_summary(
        campaign_id=campaign_id,
        output_dir=out_dir,
        selected_variants=selected_variants,
        variant_summaries=variant_summaries,
        all_rows=all_rows,
        episodes=episodes,
        external_input=external_input,
        backend=backend,
        allow_cpu_fallback=allow_cpu_fallback,
        confidence_threshold=confidence_threshold,
        abort_unsafe_variant=abort_unsafe_variant,
    )
    write_summary_outputs(summary=summary, out_dir=out_dir)
    return summary


def _build_summary(
    *,
    campaign_id: str,
    output_dir: Path,
    selected_variants: List[LatencyVariant],
    variant_summaries: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    episodes: int,
    external_input: float,
    backend: str | None,
    allow_cpu_fallback: bool,
    confidence_threshold: float,
    abort_unsafe_variant: bool,
) -> Dict[str, Any]:
    summary = {
        "campaign_id": campaign_id,
        "profile": GATED_PROFILE,
        "regime": REGIME,
        "episodes_per_variant": episodes,
        "episodes_total": len(all_rows),
        "external_input": external_input,
        "backend": backend,
        "allow_cpu_fallback": allow_cpu_fallback,
        "confidence_threshold": confidence_threshold,
        "abort_unsafe_variant": abort_unsafe_variant,
        "baseline": BASELINE,
        "variants": [variant.__dict__ for variant in selected_variants],
        "variant_summaries": variant_summaries,
        "best_variant": _best_variant(variant_summaries),
        "latency_profile": _latency_profile(variant_summaries),
    }
    summary["dictamen"] = _decide_campaign(variant_summaries)
    summary["output_dir"] = str(output_dir)
    summary["partial"] = len(variant_summaries) < len(selected_variants)
    summary["completed_variants"] = [row["variant"] for row in variant_summaries]
    return summary


def _write_rows_jsonl(*, rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "latency_variants.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def write_outputs(*, rows: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    _write_rows_jsonl(rows=rows, out_dir=out_dir)
    write_summary_outputs(summary=summary, out_dir=out_dir)


def write_summary_outputs(*, summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "external_reasoner_latency_report.md").write_text(
        render_report(summary),
        encoding="utf-8",
    )
    verdict = {
        "campaign_id": summary.get("campaign_id"),
        "dictamen": summary.get("dictamen"),
        "best_variant": summary.get("best_variant"),
        "latency_profile": summary.get("latency_profile"),
        "baseline": summary.get("baseline"),
    }
    (out_dir / "external_reasoner_latency_verdict.json").write_text(
        json.dumps(_jsonable(verdict), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# External Reasoner Latency Report",
        "",
        f"- campaign_id: `{summary.get('campaign_id')}`",
        f"- profile: `{summary.get('profile')}`",
        f"- regime: `{summary.get('regime')}`",
        f"- dictamen: `{summary.get('dictamen')}`",
        f"- baseline_latency_mean_s: `{BASELINE['latency_mean_s']}`",
        f"- baseline_corrected_core_failure_rate: `{BASELINE['corrected_core_failure_rate']}`",
        "",
        "| Variante | max_tokens | ctx | params | latency_mean | ok_rate | schema_rate | guard_pass | corrected_rate | ivc_r | dictamen |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.get("variant_summaries", []):
        lines.append(
            "| {variant} | {tokens} | {ctx} | {params} | {latency:.3f} | {ok:.3f} | "
            "{schema:.3f} | {guard:.3f} | {corrected:.3f} | {ivc:.6f} | {dictamen} |".format(
                variant=row.get("variant"),
                tokens=int(row.get("max_tokens", 0) or 0),
                ctx=row.get("ctx") or "",
                params=row.get("params"),
                latency=_safe_float(row.get("latency_mean"), 0.0),
                ok=_safe_float(row.get("external_reasoner_ok_rate"), 0.0),
                schema=_safe_float(row.get("schema_validated_rate"), 0.0),
                guard=_safe_float(row.get("guard_pass_rate"), 0.0),
                corrected=_safe_float(row.get("corrected_core_failure_rate"), 0.0),
                ivc=_safe_float(row.get("ivc_r"), 0.0),
                dictamen=row.get("dictamen"),
            )
        )
    lines.extend(
        [
            "",
            "## Latency Profile",
            "",
            "```json",
            json.dumps(_jsonable(summary.get("latency_profile", {})), indent=2, sort_keys=True),
            "```",
            "",
            "## Best Variant",
            "",
            "```json",
            json.dumps(_jsonable(summary.get("best_variant", {})), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=f"external_reasoner_latency-{_now_stamp()}")
    parser.add_argument("--output-root", default="data/benchmarks/external_reasoner_latency")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--external-input", type=float, default=0.04)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument(
        "--no-abort-unsafe-variant",
        action="store_true",
        help="Continue a variant after schema/guard/correction failure.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    variants = _select_variants(args.variants)
    summary = run_latency_campaign(
        campaign_id=args.campaign_id,
        output_root=Path(args.output_root),
        episodes=max(1, args.episodes),
        external_input=args.external_input,
        backend=args.backend,
        allow_cpu_fallback=bool(args.allow_cpu_fallback),
        confidence_threshold=args.confidence_threshold,
        variants=variants,
        abort_unsafe_variant=not bool(args.no_abort_unsafe_variant),
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
