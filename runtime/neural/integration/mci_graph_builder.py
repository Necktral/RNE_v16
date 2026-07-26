"""Construcción preacción de grafos causales desde evidencia MCI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from runtime.neural.contracts import canonical_sha256
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.contracts import TransitionSpec


@dataclass(frozen=True)
class N4GraphBundle:
    graph: Mapping[str, Any]
    graph_sha256: str


class MCIToN4GraphBuilder:
    """No acepta outcomes del episodio actual: solo el buffer ya observado."""

    def build(
        self,
        *,
        spec: TransitionSpec,
        evidence: Sequence[TransitionEvidence],
        state: Mapping[str, Any],
        candidate_action: str,
        logical_time: int,
    ) -> N4GraphBundle:
        variable_names = tuple(sorted(item.name for item in spec.variables))
        if any(item.logical_time >= logical_time for item in evidence):
            raise ValueError("n4_preaction_graph_rejects_current_or_future_evidence")
        nodes = [
            {"id": f"variable/{name}", "kind": "world_variable"}
            for name in variable_names
        ]
        nodes.append({"id": f"action/{candidate_action}", "kind": "intervention"})
        edges: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        for row in sorted(evidence, key=lambda item: (item.logical_time, item.evidence_id)):
            observed = float(row.observed[spec.main_variable])
            predicted = float(row.predicted[spec.main_variable])
            evidence_rows.append(
                {
                    "evidence_id": row.evidence_id,
                    "logical_time": row.logical_time,
                    "action": row.action,
                    "prediction_error": round(observed - predicted, 12),
                    "state": {
                        key: row.state[key]
                        for key in sorted(row.state)
                        if key in variable_names
                    },
                }
            )
        for equation in sorted(spec.equations, key=lambda item: item.target):
            for term in equation.terms:
                source = term.source.split(".", 1)[-1]
                if source in variable_names:
                    edges.append(
                        {
                            "source": f"variable/{source}",
                            "target": f"variable/{equation.target}",
                            "kind": "declared_term",
                            "condition": None,
                        }
                    )
            for effect in equation.effects:
                edges.append(
                    {
                        "source": f"action/{effect.action}",
                        "target": f"variable/{equation.target}",
                        "kind": "declared_effect",
                        "condition": effect.precondition,
                    }
                )
        graph = {
            "schema": "n4-mci-graph.v1",
            "spec_id": spec.spec_id,
            "spec_sha256": spec.sha256,
            "logical_time": int(logical_time),
            "state": {key: state[key] for key in sorted(state) if key in variable_names},
            "nodes": sorted(nodes, key=lambda item: item["id"]),
            "edges": sorted(
                edges,
                key=lambda item: (
                    item["source"],
                    item["target"],
                    item["kind"],
                    str(item["condition"]),
                ),
            ),
            "evidence": evidence_rows,
        }
        return N4GraphBundle(graph=graph, graph_sha256=canonical_sha256(graph))
