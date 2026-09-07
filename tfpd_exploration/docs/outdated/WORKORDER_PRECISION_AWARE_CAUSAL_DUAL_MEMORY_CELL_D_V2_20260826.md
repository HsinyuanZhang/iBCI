# Precision-Aware Causal Dual-Memory Cell-D V2

## Boundary

V2 is an additive, source-only, no-data candidate that supersedes the V1
candidate's precision *math*, not any accepted historical CDM-D evidence.
V1 remains frozen evidence. V2 may import the accepted independent-activity
state machine but must not change shared core behavior, Cell-D, the sealed
checkpoint, ordinary OLS normalizer, decoder token, source/target boundary,
or any historical receipt. This work order authorizes only static code and
synthetic CPU tests: no source, result root, checkpoint, CUDA/GPU, capability,
or launch action.

## Exact transition statistic

For M4 and M10 only, use the exact support-only fixed-ridge-by-trial table in
float64. With `X=[cos(theta), sin(theta), 1]`, `M` support rows, and
`lambda=0.1`, freeze:

```
A = X'X + diag(M*0.1, M*0.1, 0)
H = X A^-1 X'
df = M - tr(H)
sigma_j^2 = max(SSE_j / df, 1e-12)
Sigma_j = sigma_j^2 A^-1
```

The retained two-dimensional conditional Gaussian-ridge **posterior** block
is `Sigma_j[:2,:2]`. It is not the ridge sampling/sandwich covariance. It is
fitted once from sealed support rows, retained for valid units only, never
refit from pseudo labels, and never supplied to the decoder.

For `N` valid units, pre-register the Bonferroni family-wise 95% threshold:

```
t_N = -2 * log(0.05 / N)
q_j = (candidate_ac_j - frozen_support_ac_j)' Sigma_ac,j^-1
      (candidate_ac_j - frozen_support_ac_j)
```

The precision gate accepts only when every valid `q_j <= t_N`. The reference
is the immutable support-only initial fixed-ridge carrier captured before the
first query, never the current active carrier; therefore a sequence of small
accepted updates cannot ratchet beyond the frozen support region. Replacing a
covariance with a smaller positive-definite covariance is monotonically at
least as strict for an identical proposed delta.

Existing B8, pseudo-direction, design, finite, and departure checks execute
first through `IndependentActivityCausalDualMemory`. If they accept a carrier
proposal but V2 rejects it on precision, the typed carrier reason is
`precision_credible_region` while the independent valid B3S activity commit
still occurs. A preceding core rejection remains its original reason and is
not relabelled. M30 is literal deployment parity: no proposal, activity
transition, carrier transition, state, or prediction change is attempted.

## Required no-data gates

1. Reconstruct the exact conditional posterior formula and reject the V1
   sandwich covariance as a substitute.
2. Prove `t_N=-2*log(0.05/N)` depends on the exact valid-unit count.
3. Prove high precision is stricter for an otherwise identical delta.
4. Reproduce V1's current-relative per-step ratchet algebra and prove V2
   rejects the same cumulative candidate against frozen support.
5. Preserve invalid rows, channel/unit permutation equivariance, and the
   complementary group topology.
6. Prove M30 exact no-op, pre-transition/next-transition causality, and
   independent activity commit on a V2 precision rejection.
7. Bind an explicit successor closure and keep the public dry CLI torch-free
   and unable to execute.

## Deferred physical seam

The route-owned adapter is dependency injected and can only bind typed sealed
Cell-D/SWA/ordinary-OLS authority, V5 independent-activity evidence, exact
support rows, and post-first30 query chronology. It has no parser, evaluator,
checkpoint loader, normalizer refit, or target operation. A future physical
successor requires separate root review; V2 grants no execution capability.
