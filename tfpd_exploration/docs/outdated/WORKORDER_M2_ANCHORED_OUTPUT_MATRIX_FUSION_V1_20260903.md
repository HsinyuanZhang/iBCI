# Work order — M2 Anchored Output Matrix Fusion V1

Implement and execute the frozen development-only source OOF design in
`DESIGN_ANCHORED_OUTPUT_MATRIX_FUSION_V1_20260903.md` as a thin additive
successor to the completed AOF V1 source screen.

## Authorized science

- Exact AOF V1 model, source data, PIT-cubic activity-30, D-opt4 ridge T4,
  normalizer, scale-5 decoder, batch law, and CPU/GPU authority bridge.
- Exactly one residual matrix operator:
  `y = native + (post - native) @ B`, `B` shape `[2,2]`.
- Exactly seven lexical leave-one-source-session-out folds.
- Exactly one matched scalar-AOF closed-form control inside each fold.
- Equal-session float64 MSE-surrogate solves; no ridge or pseudoinverse.
- Conditional all-seven matrix refit only after the complete OOF gate.

No model/backbone/identity/gate training, optimizer, epoch, seed sweep,
intercept, diagonal alternative, nonlinear combiner, target-domain data,
EvalAI data, query-memory update, or trial-boundary state is authorized.

## Required predecessor and parity

- Held descriptor validation of the completed AOF V1 seven-body/fourteen-leaf
  terminal graph, including terminal SHA
  `ab7f864e8922dc0afe7ce445f09e043f206a4eb55b70ce7620e9373ecc8c71cd`
  and closure
  `28f54813f90563eb779cf76af931b3c9cef4dd1bc1eb34955c372874bfabdd5f`.
- The exact seven predecessor body digests are frozen as follows; the validator
  must not discover, substitute, or infer any of them from a mutable path:

  ```text
  attempt.json                 a624183bcd6d1e21dddbb03320aab19a0b156c22e7a350f49d7adb60e5c6518c
  launch.json                  d3b6240526b81946d0e43541bbdc960e3248d19ea28c59023371df1f757f877d
  source_authority.json        ac3e7fb3158f2fecddeb27eeec39dabf6532be4699380a2c8cc3653d9b96ccf8
  paired_output_authority.json dc988ed139c89f07c2e43388402f4670eb7c74d62b308f9cc1f3bd8f567fc17a
  fit.json                     b0aba60f798e5364252b7da2b573b02064912f3e20baf0577fb85f5af4f6044a
  validation.json              b0f196cfd554c7c0cb2ead44ca249fc13e4167ec99b5501812448dbb210aab8d
  terminal.json                ab7f864e8922dc0afe7ce445f09e043f206a4eb55b70ce7620e9373ecc8c71cd
  ```
- The only authorized source roster and declared fold order is exactly:

  ```text
  ses-2020-10-19-Run1
  ses-2020-10-19-Run2
  ses-2020-10-20-Run1
  ses-2020-10-20-Run2
  ses-2020-10-27-Run1
  ses-2020-10-27-Run2
  ses-2020-10-28-Run1
  ```

  For fold `k`, roster entry `k` is held out and the other six entries retain
  this declared relative order during float64 accumulation.
- The canonical predecessor root is exactly
  `tfpd_exploration/results/m2_anchored_output_fusion_v1/source_screen`. Open it
  through a held directory FD with `O_DIRECTORY|O_NOFOLLOW`. Its topology is
  exactly the seven named JSON bodies above plus their seven basename-matched
  `.sha256` sidecars: every leaf is a regular non-symlink file with mode `0444`
  and `nlink=1`; no `failure.json`, additional body, or additional sidecar is
  permitted. Each sidecar and the terminal's published mapping must point to
  the canonical body digest, and terminal/failure XOR must resolve to terminal.
- Attempt must precede predecessor, checkpoint, data, and CUDA access.
- One PIT source prepare and one strict frozen model load on physical GPU0.
- Source sessions are materialized once in exact lexical order. Every native,
  post, target, start, identity, activity, support, T4, state, and bridge
  witness must equal the predecessor authority before fitting.
- Raw paired arrays are process-local only; receipts contain digests and
  sufficient statistics, not reusable target arrays.
- Reapply the predecessor bridges per session with the exact limits
  `identity_maxabs <= 2e-6`, `prediction_maxabs <= 2e-6`, and
  `R2_abs_difference <= 2e-7`. These bridges verify authority only; the official
  CPU cached identity or its decoder output must not enter `G`, `H`, the scalar
  fit, or any source OOF prediction.
- Physical GPU1 must not be queried, initialized, inspected, signaled, or used.

## Numerical contract

For each fold, accumulate `G` and `H` in declared-order float64, record the
finite antisymmetric residue, set `G_sym=0.5*(G+G.T)`, and use Cholesky plus a
direct 2-by-2 solve. Require `lambda_max(G_sym)>=1e-12`,
`lambda_min/lambda_max>=1e-8`, `cond_2(G_sym)<=1e8`, and

```text
||G_sym B-H||F /
max(||G_sym||F ||B||F + ||H||F, float64_tiny) <= 1e-12.
```

All input arrays, sufficient statistics, eigenvalues, `B`, predictions, R2
values, and deltas must be finite. At the all-positive zero matrix, return the
exact native object without matrix multiplication; any negative-zero entry is
not the sentinel.

The matched scalar control uses the AOF V1 equal-session closed form on the
same six fold-training sessions. Its denominator must be finite and strictly
positive and its beta/prediction/R2/delta finite. Failure of either fit fails
the complete run; no fold may be dropped, repaired, or replaced. No arm or fold
selection is allowed.

The fit API accepts explicit `held_session` and ordered
`train_sessions_exactly_six`; it rejects held membership, duplication,
omission, reordering, or any mismatch to the frozen seven-session roster. The
matrix and scalar fits must complete before the held target is supplied to the
scoring call. Each fold receipt records held/train IDs, all six row counts,
matrix and scalar sufficient statistics (or their canonical digests), `G`,
`H`, `B`, scalar numerator/denominator/beta, conditioning, and solve residual.
Tests must demonstrate that attempting to include or access held target during
fit fails closed.

For each held session `k`, compute governing predecessor variance-weighted,
last-bin/mask/scale-5 R2 values and then
`delta_native[k]=R2_k(matrix)-R2_k(native)` and
`delta_scalar[k]=R2_k(matrix)-R2_k(scalar)`. Gate means are exactly the
arithmetic mean of the seven per-session deltas, not a pooled-window score. All
seven canonical rows must exist and be finite before aggregation.

## Decision gate

```text
mean OOF(AOF-M - native) >= +0.005
positive sessions >= 5/7
worst session delta >= -0.005
mean OOF(AOF-M - scalar-AOF) >= +0.003
all 7 folds finite and condition-valid
all exact authority and zero-anchor checks pass
```

Failure terminates PF output fusion. Success permits only the deterministic
all-seven matrix refit and a separately reviewed static packaging work order.
It does not authorize external/hidden target scoring or EvalAI submission.
