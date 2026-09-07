# Accelerated C2/C3 Posterior-Input GPU Score V5

V5 is the narrow successor to the V4 device-attestation failure.  V4 stopped
before checkpoint or target access because PyTorch renders the same GPU UUID
without the `GPU-` prefix used by `nvidia-smi`.  V5 binds the exact immutable
V4 attempt/failure graph, canonicalizes only that representation, and retains
the complete V4 science, parity, batch fallback, device, input, score, gate,
closure and lifecycle contract unchanged.

Canonical result root:

`tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_c3_posterior_gpu_score_v5`

Execute once on physical GPU1.  V3 remains an independent running CPU
reference and is neither stopped nor treated as a predecessor.  Review-only
work-order/test drift remains accepted non-numerical drift; numerical closure
drift remains fail-closed.
