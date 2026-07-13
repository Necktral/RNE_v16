# Mamba vendored provenance

- Upstream: `https://github.com/state-spaces/mamba`
- Version: `2.2.5`
- Revision/tag target: `e0761ece1db07e0949dd88b4f4cd440420a19fd9`
- License: Apache-2.0; the upstream license is vendored in `LICENSE`.
- Verified: 2026-07-12; `mamba_ssm/` matches upstream v2.2.5.

RNFE additionally retains `patches/rocm6_0.patch` and a lazy top-level import
bridge outside this directory. N3 uses the differentiable SSD-minimal
factorization for compact CPU training/serving; the fused vendor layer remains
available for verified inference experiments.
