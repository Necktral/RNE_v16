# P1 experimental evidence — N2/N3/N4

This package publishes the reviewable evidence for campaign
`neural-p1-final-20260721-06e95c8` on the experimental branch only.

## Contents

- `matrix.json`: immutable matrix emitted by experiment commit `06e95c8`.
- `matrix.audit-v2.json`: closure-denominator and canonical-parity audit emitted
  after fix commit `5f22227`.
- `n4_preaction_v2/manifest.json`: trained N4-v2 manifest.
- `n4_preaction_v2/artifact.json`: trained N4-v2 parameters and split evidence.
- `n3-attribution.audit-v1.json`: deterministic seed/lane-level attribution audit
  derived exclusively from the two published matrices. Its verdict is
  `n3_attribution_supported_limited`; it does not authorize P2.
- `p1-closure.audit-v1.json`: machine-readable canonical closure with
  `P1_STATUS=CLOSED`, `N2=FAILED`, `N3=SUPPORTED_LIMITED` and `N4=FAILED`.
- `P1_CLOSURE.md`: human-readable definitive P1 closure. The ambiguous source gate
  `trained_vs_reference` is replaced by separate ranking, Brier, MRR, balanced
  accuracy and global-superiority subgates.
- `SHA256SUMS`: hashes for byte-level verification.

## N3 attribution audit

The derived audit preserves seed pairing and enumerates all 4,096 sign-flip
assignments. It names the published ranking statistic
`paired_binary_normalized_dcg_delta_v1`, because the aggregate lacks the rankings,
eligible pool and fixed independent IDCG required to reconstruct conventional
nDCG. Individual predictions and labels are also absent, so Brier decomposition is
not recomputable. N3 remains SHADOW-only; P2 remains subject to human review.

## Canonical P1 closure

P1 is closed as a SHADOW cognitive-attribution experiment. N3 has limited support,
while N2 and N4 failed their endpoints. Trained N3 global superiority over reference
was not demonstrated. P2, live authority, staging, promotion and merge to `main`
remain unauthorized. Verify the complete package from this directory with:

```bash
sha256sum -c SHA256SUMS
```

The original runtime tree is approximately 17 GB and contains PostgreSQL/runtime
material plus 120 verbose lane reports. It is intentionally not committed to Git.
The aggregate matrices, model evidence, source code, tests and exact hashes needed
to review the reported gates are included here. This package grants no live,
staging or promotion authority and must not be merged into `main` before review.
