"""Pipeline de auto-modificación certificada del organismo.

RNFE debe poder proponerse cambios a sí mismo sin romperse.

Etapas:
1. proposal_generation
2. constitutional_precheck
3. sandbox_simulation
4. edge_stress_test
5. posterior_estimation
6. accept / quarantine / reject
7. rollback_preparation
8. lineage_update

Componentes mutables (primera fase):
  - transport_parameters
  - selection_policy
  - benchmark_policy_experimental
  - reasoning_activation_policy
  - memory_scoring_secondary_weights
  - analogical_lab_parameters

Componentes inmutables (requieren aprobación superior):
  - baseline_semantics
  - constitutional_invariants
  - baseline_fixed
  - constitutional_purity_minimum
  - lineage_identity_anchor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple
from uuid import uuid4

from .state import ModificationProposal, ModificationState, OrganismState
from .constitution import OrganismConstitution, ConstitutionalValidation
from .viability import ViabilityKernel, ViabilityAssessment
from .risk import compute_constitutional_posterior, ConstitutionalPosterior


ModificationVerdict = Literal["accepted", "quarantined", "rejected"]


@dataclass(frozen=True)
class SandboxResult:
    """Resultado de una simulación en sandbox.

    Attributes:
        proposal_id: ID de la propuesta evaluada.
        simulated_state: Estado simulado post-modificación.
        constitutional_check: Resultado de validación constitucional.
        viability_check: Resultado de evaluación de viabilidad.
        posterior: Posterior constitucional estimado.
        verdict: Veredicto de la sandbox.
        risk_level: Nivel de riesgo ('low', 'medium', 'high', 'critical').
    """

    proposal_id: str
    simulated_state: OrganismState
    constitutional_check: ConstitutionalValidation
    viability_check: ViabilityAssessment
    posterior: ConstitutionalPosterior
    verdict: ModificationVerdict
    risk_level: Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class ModificationDecision:
    """Decisión final sobre una propuesta de modificación.

    Attributes:
        proposal_id: ID de la propuesta.
        verdict: Veredicto final.
        sandbox_result: Resultado de sandbox.
        rollback_plan: Plan de rollback.
        lineage_delta: Cambio de lineage que aplicar.
    """

    proposal_id: str
    verdict: ModificationVerdict
    sandbox_result: SandboxResult
    rollback_plan: str
    lineage_delta: Dict[str, Any]


# ── Self-modification pipeline ───────────────────────────────────────────────

class SelfModificationPipeline:
    """Pipeline de auto-modificación certificada.

    Evalúa propuestas de cambio en sandbox constitucional
    antes de aplicarlas al organismo.
    """

    def __init__(
        self,
        *,
        constitution: OrganismConstitution | None = None,
        viability_kernel: ViabilityKernel | None = None,
        acceptance_threshold: float = 0.60,
    ):
        self.constitution = constitution or OrganismConstitution()
        self.viability_kernel = viability_kernel or ViabilityKernel(
            constitution=self.constitution,
        )
        self.acceptance_threshold = acceptance_threshold

    def generate_proposal(
        self,
        *,
        target: str,
        description: str,
        parameters: Dict[str, Any] | None = None,
    ) -> ModificationProposal:
        """Genera una propuesta de auto-modificación.

        Args:
            target: Componente a modificar.
            description: Descripción del cambio.
            parameters: Parámetros del cambio.

        Returns:
            ModificationProposal.

        Raises:
            ValueError: Si el componente es inmutable.
        """
        if self.constitution.is_immutable(target):
            raise ValueError(
                f"Component '{target}' is constitutionally immutable. "
                f"Cannot propose modification without superior approval."
            )

        return ModificationProposal(
            proposal_id=f"mod-{uuid4().hex[:8]}",
            target=target,
            description=description,
            risk_posterior=0.5,
            sandbox_verdict="pending",
        )

    def constitutional_precheck(
        self,
        proposal: ModificationProposal,
        current_state: OrganismState,
    ) -> bool:
        """Verificación constitucional previa.

        Returns True si la propuesta es elegible para sandbox.
        """
        if self.constitution.is_immutable(proposal.target):
            return False

        validation = self.constitution.validate(current_state)
        # Cannot modify if already in rollback/quarantine
        if validation.verdict == "rollback":
            return False

        return True

    def sandbox_simulate(
        self,
        *,
        proposal: ModificationProposal,
        current_state: OrganismState,
        apply_fn: Any | None = None,
        n_historical: int = 0,
        historical_success_rate: float | None = None,
    ) -> SandboxResult:
        """Simula la modificación en sandbox constitucional.

        Args:
            proposal: Propuesta a evaluar.
            current_state: Estado actual del organismo.
            apply_fn: Función que aplica la modificación al estado
                      (opcional; si None, simula con estado actual).
            n_historical: Episodios históricos observados (evidencia E del
                posterior; sin evidencia el LCB nunca alcanza el umbral de
                aceptación — el organismo debe ganarse el derecho a mutar).
            historical_success_rate: Tasa de éxito histórica (certificación).

        Returns:
            SandboxResult con verificación constitucional y viabilidad.
        """
        # Simulate the modification
        if apply_fn is not None:
            simulated = apply_fn(current_state, proposal)
        else:
            # Default: simulate as identity (no actual change)
            simulated = current_state

        # Constitutional check
        const_check = self.constitution.validate(simulated)

        # Viability check
        viab_check = self.viability_kernel.assess(simulated, previous_state=current_state)

        # Posterior estimation
        posterior = compute_constitutional_posterior(
            state=simulated,
            constitutional_validation=const_check,
            viability_assessment=viab_check,
            n_historical=n_historical,
            historical_success_rate=historical_success_rate,
        )

        # Determine verdict
        if posterior.rollback_required or not const_check.is_valid:
            verdict: ModificationVerdict = "rejected"
            risk = "critical"
        elif posterior.quarantine_required:
            verdict = "quarantined"
            risk = "high"
        elif posterior.lower_confidence_bound >= self.acceptance_threshold:
            verdict = "accepted"
            risk = "low" if posterior.constitutional_posterior >= 0.80 else "medium"
        else:
            verdict = "quarantined"
            risk = "medium"

        return SandboxResult(
            proposal_id=proposal.proposal_id,
            simulated_state=simulated,
            constitutional_check=const_check,
            viability_check=viab_check,
            posterior=posterior,
            verdict=verdict,
            risk_level=risk,
        )

    def evaluate_proposal(
        self,
        *,
        proposal: ModificationProposal,
        current_state: OrganismState,
        apply_fn: Any | None = None,
        n_historical: int = 0,
        historical_success_rate: float | None = None,
    ) -> ModificationDecision:
        """Pipeline completo: precheck → sandbox → decision.

        Args:
            proposal: Propuesta a evaluar.
            current_state: Estado actual.
            apply_fn: Función de aplicación.
            n_historical: Evidencia histórica (episodios observados).
            historical_success_rate: Tasa de éxito histórica.

        Returns:
            ModificationDecision.
        """
        # Step 1: Constitutional precheck
        if not self.constitutional_precheck(proposal, current_state):
            return ModificationDecision(
                proposal_id=proposal.proposal_id,
                verdict="rejected",
                sandbox_result=SandboxResult(
                    proposal_id=proposal.proposal_id,
                    simulated_state=current_state,
                    constitutional_check=self.constitution.validate(current_state),
                    viability_check=self.viability_kernel.assess(current_state),
                    posterior=compute_constitutional_posterior(
                        state=current_state,
                        constitutional_validation=self.constitution.validate(current_state),
                        viability_assessment=self.viability_kernel.assess(current_state),
                    ),
                    verdict="rejected",
                    risk_level="critical",
                ),
                rollback_plan="no_modification_applied",
                lineage_delta={},
            )

        # Step 2: Sandbox simulation
        sandbox = self.sandbox_simulate(
            proposal=proposal,
            current_state=current_state,
            apply_fn=apply_fn,
            n_historical=n_historical,
            historical_success_rate=historical_success_rate,
        )

        # Step 3: Build decision
        rollback_plan = (
            "revert_to_previous_state"
            if sandbox.verdict == "accepted"
            else "no_modification_applied"
        )

        lineage_delta = {}
        if sandbox.verdict == "accepted":
            lineage_delta = {
                "modification_id": proposal.proposal_id,
                "target": proposal.target,
                "description": proposal.description,
                "posterior": sandbox.posterior.constitutional_posterior,
            }

        return ModificationDecision(
            proposal_id=proposal.proposal_id,
            verdict=sandbox.verdict,
            sandbox_result=sandbox,
            rollback_plan=rollback_plan,
            lineage_delta=lineage_delta,
        )
