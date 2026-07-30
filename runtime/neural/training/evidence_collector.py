"""Persistencia incremental, reanudable y canónica de evidencia causal."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from runtime.symbolic.mci.causal_learning import TransitionEvidence


class EvidenceJSONLCollector:
    """Observador callable para ``CausalLearningEngine.add_evidence_observer``."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        scenario_id: str,
        seed: int,
        transition_spec_hash: str,
        scenario_version: str = "1.0",
        source: str = "factual_transition",
    ) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.scenario_id = str(scenario_id)
        self.scenario_version = str(scenario_version)
        self.seed = int(seed)
        self.transition_spec_hash = str(transition_spec_hash)
        self.source = str(source)
        self.episode = 0
        self.regime_id = "default"
        self.oracle_fields: Mapping[str, Any] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen = self._read_existing_ids()

    def set_context(
        self,
        *,
        episode: int,
        regime_id: str = "default",
        oracle_fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.episode = int(episode)
        self.regime_id = str(regime_id)
        self.oracle_fields = dict(oracle_fields or {})

    def __call__(self, evidence: TransitionEvidence) -> None:
        record_id = f"{self.run_id}|{evidence.evidence_id}"
        if record_id in self._seen:
            return
        record = {
            "schema_version": "n4-transition-evidence.v1",
            "evidence_id": evidence.evidence_id,
            "record_id": record_id,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "seed": self.seed,
            "episode": self.episode,
            "logical_time": int(evidence.logical_time),
            "action": evidence.action,
            "state_before": _plain(evidence.state),
            "state_after": _plain(evidence.observed),
            "external_input": float(evidence.external_input),
            "predicted_effect": _plain(evidence.predicted),
            "observed_effect": _plain(evidence.observed),
            "success": _matches(evidence.predicted, evidence.observed),
            "regime_id": self.regime_id,
            "source": self.source,
            "oracle_fields": _plain(self.oracle_fields),
            "transition_spec_hash": self.transition_spec_hash,
        }
        encoded = _canonical_bytes(record)
        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
        with self.path.open("ab") as handle:
            handle.write(_canonical_bytes(record) + b"\n")
            handle.flush()
        self._seen.add(record_id)

    def _read_existing_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        seen: set[str] = set()
        with self.path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    seen.add(
                        str(
                            payload.get("record_id")
                            or f"{payload['run_id']}|{payload['evidence_id']}"
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid_evidence_jsonl_line:{line_number}"
                    ) from exc
        return seen


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _matches(predicted: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    common = set(predicted) & set(observed)
    if not common:
        return False
    for key in common:
        left, right = predicted[key], observed[key]
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                return False
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9):
                return False
        elif left != right:
            return False
    return True
