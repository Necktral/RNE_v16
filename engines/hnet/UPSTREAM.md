# H-Net vendored provenance

- Upstream: `https://github.com/goombalab/hnet`
- Imported revision: `3ae01de79e560234776d06ceb1153ab76a5aad32`
- License: MIT; the upstream license is vendored in `LICENSE`.
- Verified: 2026-07-12 by comparison against the upstream Git history.

RNFE carries local compatibility changes for execution without compiled
FlashAttention/causal-conv extensions. Those changes do not imply that local
weights are pretrained H-Net weights. Every runtime artifact must provide its
own training provenance and SHA-256 manifest.
