"""Orquestador EML-SR en modo shadow (eventos + artifacts)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from runtime.storage import get_storage

from .scoring import score_candidate
from .search import SearchLimits, generate_candidates


@dataclass(frozen=True, slots=True)
class EMLRunnerConfig:
    max_depth: int = 3
    max_evals: int = 512
    max_candidates: int = 64
    seed: int = 0
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "EMLRunnerConfig":
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
            max_depth=max(1, _int("RNFE_EML_MAX_DEPTH", 3)),
            max_evals=max(16, _int("RNFE_EML_MAX_EVALS", 512)),
            max_candidates=max(8, _int("RNFE_EML_MAX_CANDIDATES", 64)),
            seed=_int("RNFE_EML_SEED", 0),
            top_k=max(1, _int("RNFE_EML_TOP_K", 5)),
        )


class EMLRunner:
    def __init__(self, *, storage=None, config: EMLRunnerConfig | None = None):
        self.storage = storage or get_storage()
        self.config = config or EMLRunnerConfig.from_env()

    def run_shadow(
        self,
        *,
        run_id: str,
        episode_id: str,
        rows: Iterable[dict[str, Any]],
        target_key: str = "y",
    ) -> Dict[str, Any]:
        rows_list = [dict(row) for row in rows]
        var_names = set()
        for row in rows_list:
            for key, value in row.items():
                if key == target_key:
                    continue
                if isinstance(value, (int, float)):
                    var_names.add(key)
        candidates = generate_candidates(
            var_names=sorted(var_names),
            limits=SearchLimits(
                max_depth=self.config.max_depth,
                max_candidates=self.config.max_candidates,
                max_evals=self.config.max_evals,
                seed=self.config.seed,
            ),
        )

        scored: List[Dict[str, Any]] = []
        for expr in candidates:
            score = score_candidate(expr, rows_list, target_key=target_key)
            scored.append(
                {
                    "expr": expr.to_dict(),
                    "depth": expr.depth(),
                    "fit_score": score.fit_score,
                    "stability_score": score.stability_score,
                    "domain_valid_ratio": score.domain_valid_ratio,
                    "composite_score": score.composite_score,
                }
            )
        scored.sort(key=lambda item: item["composite_score"], reverse=True)
        top = scored[: self.config.top_k]

        eml_run = {
            "eml_run_id": f"eml-run-{uuid4()}",
            "run_id": run_id,
            "episode_id": episode_id,
            "config": asdict(self.config),
            "rows_count": len(rows_list),
            "candidate_count": len(scored),
            "top_candidates": top,
        }
        self.storage.append_event(
            event_type="eml.run.completed",
            run_id=run_id,
            source="eml_runner",
            payload=eml_run,
        )
        for candidate in top:
            self.storage.append_event(
                event_type="eml.candidate.generated",
                run_id=run_id,
                source="eml_runner",
                payload={
                    "eml_run_id": eml_run["eml_run_id"],
                    "episode_id": episode_id,
                    "candidate": candidate,
                },
            )

        report_artifact = self.storage.materialize_artifact(
            run_id=run_id,
            kind="eml_report",
            filename=f"{eml_run['eml_run_id']}.json",
            content=json.dumps(eml_run, ensure_ascii=True, sort_keys=True, indent=2),
            metadata={"episode_id": episode_id, "eml_run_id": eml_run["eml_run_id"]},
        )
        trace_artifact = self.storage.materialize_artifact(
            run_id=run_id,
            kind="eml_candidate_trace",
            filename=f"{eml_run['eml_run_id']}_candidates.json",
            content=json.dumps(scored, ensure_ascii=True, sort_keys=True, indent=2),
            metadata={"episode_id": episode_id, "eml_run_id": eml_run["eml_run_id"]},
        )
        return {
            "run": eml_run,
            "artifacts": {
                "eml_report": {
                    "artifact_id": report_artifact.artifact_id,
                    "abs_path": report_artifact.abs_path,
                },
                "eml_candidate_trace": {
                    "artifact_id": trace_artifact.artifact_id,
                    "abs_path": trace_artifact.abs_path,
                },
            },
        }

