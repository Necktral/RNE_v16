"""Agente de conectómica: audita topología y conectividad observada."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from runtime.neural.connectome import ConnectomeRuntime
from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
    OrganTrace,
    SymbiosisIdentity,
)

from .contracts import (
    AgentFinding,
    AgentReport,
    AgentRole,
    AgentState,
    FindingSeverity,
)


class ConnectomicsAgent:
    """Contrasta el conectoma declarado con actividad y recibos reales."""

    agent_id = "agent-connectomics-v1"

    def __init__(self, *, connectome: ConnectomeRuntime) -> None:
        self.connectome = connectome

    def analyze(
        self,
        *,
        identity: SymbiosisIdentity,
        activity: Mapping[str, Any] | None,
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
    ) -> AgentReport:
        topology = self.connectome.topology
        activity = dict(activity or {})
        findings: list[AgentFinding] = []
        declared_nodes = {node.node_id for node in topology.nodes}
        declared_edges = {edge.edge_id for edge in topology.edges}
        active_nodes = set(activity.get("active_nodes") or ())
        active_edges = {
            str(item.get("edge_id") or "")
            for item in activity.get("active_connections") or ()
            if isinstance(item, Mapping)
        }

        if activity and activity.get("topology_hash") != topology.topology_hash:
            findings.append(
                AgentFinding(
                    "connectome_topology_hash_mismatch",
                    FindingSeverity.CRITICAL,
                    "La actividad no pertenece a la topología canónica cargada.",
                    evidence_refs=(topology.topology_hash,),
                )
            )
        for node in sorted(active_nodes - declared_nodes):
            findings.append(
                AgentFinding(
                    "connectome_unknown_active_node",
                    FindingSeverity.CRITICAL,
                    "La actividad contiene un nodo no declarado.",
                    subject=node,
                )
            )
        for edge in sorted(active_edges - declared_edges):
            findings.append(
                AgentFinding(
                    "connectome_unknown_active_edge",
                    FindingSeverity.CRITICAL,
                    "La actividad contiene una conexión no declarada.",
                    subject=edge,
                )
            )

        candidate_organs = {
            organ.organ
            for organ in organs
            if organ.candidate_hash and organ.effective_mode != "off"
        }
        consumed_organs = {receipt.organ for receipt in receipts}
        isolated_organs = tuple(sorted(candidate_organs - consumed_organs))
        for organ in isolated_organs:
            findings.append(
                AgentFinding(
                    "connectome_candidate_without_consumer_receipt",
                    FindingSeverity.WARNING,
                    "El órgano produjo candidato pero aún no existe consumo tipado.",
                    subject=organ,
                )
            )

        critical = any(item.severity is FindingSeverity.CRITICAL for item in findings)
        if critical:
            state = AgentState.BLOCKED
        elif isolated_organs:
            state = AgentState.DEGRADED
        elif not activity:
            state = AgentState.ABSTAINED
        else:
            state = AgentState.OBSERVED

        plasticity = [
            dict(item)
            for item in activity.get("plasticity_proposals") or ()
            if isinstance(item, Mapping)
        ]
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.CONNECTOMICS,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "declared_node_count": len(declared_nodes),
                "declared_edge_count": len(declared_edges),
                "active_node_count": len(active_nodes),
                "active_edge_count": len(active_edges),
                "candidate_organ_count": len(candidate_organs),
                "consumed_organ_count": len(candidate_organs & consumed_organs),
            },
            findings=findings,
            outputs={
                "topology_hash": topology.topology_hash,
                "isolated_organs": list(isolated_organs),
                "plasticity_proposals": plasticity,
                "graph_mutated": False,
                "apply_authorized": False,
            },
        )
