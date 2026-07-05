# Operational Conjunction Layer

## Summary

`runtime/conjunction/` is the operational control layer that joins evidence,
causal support, reasoning routes, agent policy, validators, compensations, and
traceability. It is not a model and does not replace `LifeKernel`; it gates each
life-cycle decision before the organism acts.

The hierarchy is:

```text
operational truth > evidence > causality > constraints > reasoning > generation
```

## Runtime Integration

- `LifeKernel.step()` asks `AutonomySupervisor` for an action, then sends that
  action through `OperationalConjunctionLayer.evaluate_life_cycle()`.
- The layer builds an `OperationContext` with vital signs, scenario causal
  signature, memory/RAG hits when available, checkpoint evidence, agent policy,
  risk, resource pressure, uncertainty, and compute constraints.
- `ComputeRouter` selects the cheapest sufficient tier:
  `tier_0_deterministic`, `tier_1_local_light`, `tier_2_specialized`, or
  `tier_3_external`.
- `OperationalValidatorStack` checks schema, evidence, causal support,
  constraints, risk, and agent execution policy.
- `CompensationMatrix` converts failures into explicit operational behavior:
  evidence recovery, conflict marking, causal downgrade, local degradation,
  action blocking, plan validation, rollback requirement, or resource
  conservation.
- The result is appended as `operational.conjunction.evaluated` and copied into
  `life.step.completed` plus the episode result under `operational_conjunction`.

## Critical Action Policy

Critical actions are not free-running:

- `self_modify` is permitted only with `validated_plan` and `rollback_plan`
  evidence. Without both, the kernel degrades it to `act` in `recovery` mode.
- `consult_external` is permitted only when external reasoning is enabled and
  the plan is validated.
- `rollback` requires healthy checkpoint evidence.
- `shutdown` remains allowed as a safe terminal action.

This keeps self-modification and agents policy-bound instead of model-bound.

## Extension Points

- Add new evidence sources by producing `EvidenceItem` values in
  `OperationalConjunctionLayer._life_evidence()`.
- Add a new compute route by extending `ComputeRouter.route()` and keeping it
  bounded by `OperationalConstraints.max_compute_tier`.
- Add a new validator in `OperationalValidatorStack`; every failure should map
  to a compensation in `CompensationMatrix`.
- Add an executor/agent only after its role, allowed actions, prohibited
  actions, stop conditions, budget, validators, and rollback behavior are
  represented in `AgentPolicy` or equivalent evidence.

## Verification

Minimum evidence that the layer is alive:

```bash
.venv/bin/python -m pytest tests/conjunction/test_operational_conjunction.py -q
.venv/bin/python -m pytest tests/life/test_life_kernel.py tests/miniworlds/test_scenario_runner.py tests/organism/test_autoevolution.py -q
PYTHONPATH=. .venv/bin/python scripts/life_kernel.py --run-id rne16-postgres-smoke --max-steps 3 --no-restore
```

The tests cover cheap deterministic routing, missing evidence compensation,
contradiction blocking, unsupported causal claim downgrade, critical action
blocking, and persistence of operational trace inside the living cycle.
