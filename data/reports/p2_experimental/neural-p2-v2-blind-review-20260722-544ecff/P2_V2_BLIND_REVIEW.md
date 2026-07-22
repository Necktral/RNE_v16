# P2-v2 Independent Blind Review

## 1. Identity and lineage

This independent review was computed at `e2061ef487aea37fb718ea0c6867086a2e638b14` from the frozen P2-v2 package `neural-p2-v2-n3-causal-20260722-71257a6`. It did not rerun the campaign, retrieval, N3 adapters, temporal backends, counterfactual simulation, or any outcome-producing path.

## 2. Evidence hierarchy

The calculation used decision receipts first, followed by preregistration and manifest. Frozen execution code and v1/v2 artifacts were used only for post-computation comparison and implementation-bias review. Human predictions and narrative expectations were not reviewer inputs.

## 3. Primary integrity

`matrix.audit-v3.json.primary_evidence_integrity` reports `valid=true`, no integrity errors, and zero derived-field mismatches. Dimensions were derived as 12 seeds, 4 scenarios, 16 episodes per scenario, 3 arms, `k_exposed=4`, 768 paired units, and 2304 receipts.

## 4. Ordinal treatment

`matrix.audit-v3.json.observable_treatment` records extensive ordinal influence. The reference arm changed full order in 713 units and top-k sequence in 579; the trained arm changed full order in 766 units and top-k sequence in 675. These observations establish ranking sensitivity, not cognitive benefit.

## 5. Membership treatment

Top-k membership changed in zero units for both N3 arms. Consequently, neither arm delivered the preregistered membership treatment at the `0.10` gate. This is the decisive gate behind `verdict.v3.json.P2_V2_REVIEW_RESULT=TREATMENT_NOT_DELIVERED`.

## 6. Action and regret changes

Reference changed action and regret in 56 of 768 units; trained changed both in 64. Because membership did not change, these observations are compatible with sensitivity to ordering within the exposed set. They do not establish improvement or identify its mechanism.

## 7. Independent contrasts

The seed-level contrasts were independently recomputed from validated receipts. Reference minus canonical had mean `0.0017447916666666666` and exact sign-flip p-value `0.099609375`; trained minus canonical had mean `0.0016666666666666663` and p-value `0.1748046875`; trained minus reference had mean `-0.000078125` and p-value `0.9375`. No confirmatory contrast gate passed.

## 8. V3 versus v2

The shared v1 contrast fields and v2 numeric membership metrics reproduce. V3 independently reaches `TREATMENT_NOT_DELIVERED`, no backend preference, and no operational authority. V2 is classified `CONFIRMED_WITH_IMPLEMENTATION_BIAS`, not because its methodology was neutral, but because the independently derived observed disposition was unchanged.

## 9. Implementation bias

`BIAS_FINDINGS.json` documents hardcoded top-k, treatment gate, validity, closure status, result, and backend preference, a non-neutral delivered branch, and dimensions not fully derived. These defects can affect metrics or disposition in other inputs. The review found no evidence that they changed the observed P2-v2 result. Self-confirming input dependency was not found.

## 10. Authorized causal conclusions

The primary evidence is internally valid; v2 metrics are reproducible; N3 arms exerted observable ordinal influence; and the preregistered membership treatment was absent. These claims reference `primary_evidence_integrity`, `observable_treatment`, and `comparison_with_v1` in `matrix.audit-v3.json`.

## 11. Unauthorized conclusions

The evidence does not authorize a causal claim of improved decisional gain, a preference between reference and trained backends, generalization beyond P2-v2, training, staging, promotion, live activation, or operational mutation. `verdict.v3.json` and all authority maps retain these values as false or `NONE`.

## 12. Knowledge gain

Resolved: primary integrity, v2 metric reproducibility, existence of ordinal influence, absence of membership treatment, and implementation bias in the v2 builder. These are represented by `primary_evidence_integrity`, `comparison_with_v1`, `observable_treatment`, and `BIAS_FINDINGS.json.findings`.

Partially resolved: the relation between ordering and action change. Action changes coexist with unchanged membership, but the available evidence does not isolate causality.

## 13. Remaining uncertainties

Unresolved are the exact per-candidate reranking mechanism, causality between membership change and regret, the utility or fragility of ordinal sensitivity, and generalization outside the frozen scenarios. The missing instrumentation includes observed N3 signals, applied multipliers, and adjusted candidate scores, as recorded in `closure.review.json.knowledge_gain.instrumentation_gaps`.

## 14. Highest-information causal question

Does a preregistered N3 intervention that changes top-k membership reduce seed-level decisional regret relative to canonical retrieval?

## 15. Global QA

`FULL_SUITE_REVIEW_SHARDS.json` binds QA to HEAD and records 42 complete invocations: 1794 passed, 22 skipped, 32 xfailed, one known historical XPASS, 6 warnings, zero failures, zero timeouts, and zero new XPASS node IDs. The final focused confirmation added a fresh result of 37 passed.

## 16. Authority

The review is `READY_TO_CLOSE` as an epistemic record. It grants no design, training, staging, promotion, merge, live, or mutation authority. Codex/frontier remains the reference teacher; no local model receives authority through this review.
