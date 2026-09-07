# Design — Anchored Output Matrix Fusion V1

Date: 2026-09-03  
Status: **development-only source OOF candidate; official target untouched**  
Working name: **AOF-M** (Anchored Output Fusion, Matrix residual)  
Resource scope: physical GPU0 only; physical GPU1 is out of scope

## 1. Motivation

AOF V1 moved the Post-Fusion residual after the nonlinear decoder but a single
global scalar still failed its source gate:

```text
beta = -0.4234267943
validation deltas = {-0.0022238381, +0.0067978962}
mean = +0.0022870291, positive = 1/2
```

The sign disagreement is compatible with a residual whose two output
coordinates are misoriented rather than uniformly over- or under-scaled. M2 has
exactly two behavior outputs. The smallest operator that can rotate and rescale
the decoded PF residual while retaining an exact native anchor is a 2-by-2
matrix.

## 2. Frozen operator

For row-vector predictions:

```text
D = y_post - y_native
y_AOF-M = y_native + D B
B in R^(2x2)
```

At an IEEE all-`+0.0` matrix, the public path returns `y_native` directly and
does not evaluate a residual multiply-add. There is no intercept. The four
matrix entries are the only fitted quantities. The backbone, identity encoder,
decoder, PIT-cubic activity-30 pool, D-opt4 T4 carrier, and normalizer remain
frozen.

AOF-M is not a new neural network and has no optimizer, epoch, seed sweep,
ridge, clamp, nonlinear gate, session parameter, pseudo-label, or online state.

## 3. Closed-form fit

For a set of source sessions `S`, define `E_s = target_s - native_s` and
`D_s = post_s - native_s`. Use equal-session weighting:

```text
G = sum_s (D_s^T D_s / n_s)
H = sum_s (D_s^T E_s / n_s)
B = solve(G, H)
```

All accumulation and solve operations use declared-order float64. After those
ordered additions, define `G_sym = 0.5 * (G + G.T)`; record the finite
antisymmetric residue and use `G_sym` for Cholesky and the direct solve. No ridge
or pseudoinverse is permitted. Fail closed unless every input is finite,
Cholesky succeeds, `lambda_max(G_sym) >= 1e-12`,
`lambda_min(G_sym) / lambda_max(G_sym) >= 1e-8`, `cond_2(G_sym) <= 1e8`, and

```text
||G_sym B - H||_F /
max(||G_sym||_F ||B||_F + ||H||_F, float64_tiny) <= 1e-12.
```

All four entries of `B`, every fused prediction, every R2, and every delta must
also be finite. The objective is an equal-session MSE surrogate, not a direct
optimizer of variance-weighted R2.

## 4. Source grouped-OOF decision

The two-session AOF V1 validation split has already been observed. It cannot be
reused as a clean selection split for a richer operator. AOF-M therefore uses
one fixed seven-fold leave-one-source-session-out calculation:

1. materialize each of the seven source sessions exactly once;
2. for each lexical held session, fit `B` on the other six sessions only;
3. score native, scalar-AOF control, and AOF-M on that held session;
4. aggregate the seven paired deltas with equal session weight;
5. do not select a fold, seed, threshold, or matrix variant.

The scalar control is refitted independently inside every fold using the same
six training sessions. It is diagnostic and cannot be selected instead of the
matrix after seeing results. Its denominator must be finite and strictly
positive, and its beta, predictions, R2 values, and deltas must all be finite.
Failure of either solver in any fold fails the complete run; no fold may be
dropped, repaired, or replaced.

The fold solver receives an explicit `(held_session,
train_sessions_exactly_six)` contract. It must prove that held is absent from
the ordered training tuple and that held plus training is exactly the frozen
seven-session roster. Each fold receipt records held/train IDs, each training
session row count, `G`, `H`, `B`, scalar numerator/denominator/beta, and the
digests of the six ordered sufficient-statistic records. Held-session target
data may be used only after both matrix and scalar fits for that fold return.

For held session `k`, define the two paired quantities exactly as

```text
delta_native[k] = R2_k(matrix) - R2_k(native)
delta_scalar[k] = R2_k(matrix) - R2_k(matched scalar)
```

The two reported means are the arithmetic means of their seven session deltas,
never pooled-window R2 values. Masking, last-bin selection, target scaling, and
variance-weighted R2 are byte-semantically inherited from AOF V1. All seven
per-session rows are mandatory and published before aggregation.

AOF-M passes the development gate only if all conditions hold:

```text
mean OOF(AOF-M - native) >= +0.005 R2
positive sessions >= 5/7
worst session delta >= -0.005 R2
mean OOF(AOF-M - scalar-AOF) >= +0.003 R2
all seven matrix fits satisfy the conditioning contract
all native/zero/authority checks pass
```

These seven source sessions and the AOF-M hypothesis are development evidence,
not untouched confirmation. A pass permits one all-seven closed-form matrix fit
and a separate packaging audit. Only a later untouched official EvalAI result
can confirm transfer.

## 5. Matched authority and computation

AOF-M must descriptor-bind the completed AOF V1 terminal and reuse its exact
science contract:

- selected-T4 checkpoint/state;
- official PIT-cubic first-30 activity;
- D-opt4 selected-support ridge T4;
- frozen normalizer and behavior scale 5.0;
- GPU native identity and exact sealed source prediction anchor;
- SHA-bound official CPU cached identity plus the declared numerical bridge;
- independent `B` native/post production decodes;
- no query commit, online update, optimizer, or hidden/external target.

The successor reruns each source forward once because AOF V1 intentionally
stored only immutable digests, not raw predictions. Every rematerialized
native/post/target/start digest must equal the AOF V1 paired-output authority
before any matrix fit. The seven raw arrays may then remain process-local for
all seven CPU OOF solves; no decoder forward is repeated per fold.

## 6. Decision after OOF

- **Gate fails:** close the PF output-fusion family. Do not add an intercept,
  nonlinear combiner, per-session matrix, or larger output head.
- **Gate passes:** fit one all-seven matrix with the same closed form. Preserve
  the seven OOF matrices and the final deployment matrix in immutable receipts.
  A separate work order must validate a two-identity static EvalAI package.
- **Official delta < +0.010:** bounded diagnostic only, not a headline method.
- **Official delta >= +0.010 over exact matched `act30_dopt4`:** meaningful PF
  improvement candidate, subject to container and repeated-run audit.

## 7. Scope and claim

AOF-M is M2-specific because its matrix dimension follows the two-dimensional
M2 output. It should not be generalized to M1/H1 without a new parameter-count,
conditioning, and source-OOF analysis.

The intended claim, if it passes, is that Post-Fusion preserves complementary
behavior signal whose output-space orientation is wrong; a four-parameter
source-fitted residual map recovers it while exactly nesting the strong native
decoder. If it fails, the honest conclusion is that PF's apparent complementarity
is too small or too session-specific to exploit without target labels.
