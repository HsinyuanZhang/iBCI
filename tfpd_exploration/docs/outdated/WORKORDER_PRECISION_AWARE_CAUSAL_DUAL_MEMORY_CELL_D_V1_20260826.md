# Precision-Aware Causal Dual-Memory Cell-D V1

## Purpose and boundary

This is a source-only, additive performance mechanism for the accepted
independent-activity CDM-D state machine.  It must not alter the sealed
Cell-D graph, checkpoint, OLS normalizer, B3S activity path, decoder token,
parameter set, target boundary, or any historical CDM-D receipt.  It is a
candidate only: this work order authorizes synthetic CPU/no-data code and
tests, not source access, result-root creation, CUDA, GPU execution, or a
launch.

The route name is `PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_V1`.  It keeps
the sealed M4/M10 support rows and post-first-30 causal query surface.  M30
is a deployment no-op: no carrier proposal or commit is attempted and its
state and prediction inputs remain byte-identical to the sealed baseline.

## One mechanism only

For M4 and M10, fit the existing support-only fixed-ridge-by-trial carrier
with normalized ridge literal `0.1`.  In float64, form

```
X = [cos(theta), sin(theta), 1]
A = X.T @ X + diag(M*0.1, M*0.1, 0)
beta = solve(A, X.T @ rates)
Cov(beta_j) = sigma_j^2 * A^-1 @ X.T @ X @ A^-1
sigma_j^2 = max(sum(residual_j^2)/(M - trace(X @ A^-1 @ X.T)), 1e-12)
```

Only `Cov([a,c])` is retained.  It is support-only, frozen before the first
query, and is never reconstructed from pseudo labels.  Valid units alone are
eligible; invalid rows remain represented in the exact mask topology and are
excluded from the statistic.

After the existing independent-activity machine has accepted all of its
typed B8, pseudo-direction, design, non-finite, and departure checks, compare
the proposed carrier delta `d_j=[a'-a,c'-c]` against its frozen support
covariance:

```
q_j = d_j.T @ inv(Cov_ac,j) @ d_j
accept iff max_valid(q_j) <= 5.991464547107979
```

The literal is the pre-registered 95% chi-square(2) credible-region bound.
This rule is monotone: replacing a covariance by a smaller positive-definite
covariance cannot make an otherwise rejected identical proposal become
accepted.  A precision rejection uses the new typed reason
`precision_credible_region`; it preserves the carrier exactly while the
accepted independent B3S activity transition still commits.

The covariance and statistic are state-transition evidence only.  They are
never passed to Cell-D as a side feature, attention term, decoder token, or
normalizer input.

## Required no-data gates

1. Exact fixed-ridge coefficient parity against the shared production helper.
2. Float64 covariance symmetry/finite topology and invalid-mask exclusion.
3. High precision is at least as strict as low precision for the same delta.
4. M30 does not call the proposal machine and leaves state/prediction input
   digests exact.
5. M4/M10 preserve causal ordering: each query reads the pre-transition
   state, then an accepted carrier changes only the next query state; a
   precision rejection preserves carrier while independent activity advances.
6. Channel/unit permutations, complementary-group topology, and invalid masks
   preserve the statistic after the corresponding permutation.
7. No model, checkpoint, OLS normalizer, decoder parameter, target label, or
   target gradient is accessed or changed by this route.
8. The old core enum values and default V1/V3/V5/V8 behavior remain unchanged
   unless this additive wrapper explicitly chooses `precision_credible_region`.

Historical current-byte closures legitimately reject the shared-core byte
after the additive enum extension.  Their immutable receipt graphs remain
historical evidence; this successor records a new explicit closure instead of
claiming any historical closure remains current.

## Physical seam and execution boundary

The route-owned physical adapter is deferred and dependency injected.  It
only validates sealed authorities and support/query chronology, and it wraps
the exact `IndependentActivityCausalDualMemory` object.  It does not copy a
parser, evaluator, model loader, normalizer, or optimizer loop.  A future
root-reviewed physical implementation must bind the sealed Cell-D terminal,
SWA, OLS normalizer, V5 independent-activity state-machine evidence, and the
same post-first-30 query rows before it can issue a capability.  This work
order grants no such capability.
