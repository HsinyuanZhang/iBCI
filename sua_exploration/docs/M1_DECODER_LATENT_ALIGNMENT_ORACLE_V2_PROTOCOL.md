# M1 decoder-latent alignment oracle v2: non-circular formal protocol

**Frozen:** 2026-08-02 (Asia/Hong_Kong)  
**Status:** protocol plus CPU identity-base addendum only. This file does **not** authorize an
adapter fit, model training, GPU launch, method-specific report evaluation, held-out access, or
EvalAI action.

## 1. The object being tested

This is not a second way of reproducing the teacher identity.  Two distinct, frozen identities
are used on the *same chronological support* `C = trials[0:10]`:

```text
E_teacher = teacher.fc_id_out(mean_m teacher.fc_id_in(C_m))
E0        = F0/B3 student.id_encoder.forward_batch(C)
Delta*    = E_teacher - E0
E_hat     = E0 + Delta_hat
```

The target of a proposed support-only low-rank adapter is therefore `Delta*`, not `E_teacher`
alone. `Delta*=0` means no residual correction is needed and the proposed residual task
degenerates; it is not evidence for an adapter. The CPU addendum records whether this occurs.
All identities have shape `[1, N=64, W=100]` and are added before the same decoder
read-in, `decoder.fc_in(x + E)`.

The frozen source objects are:

| Object | Exact artifact | SHA-256 |
| --- | --- | --- |
| Teacher target | `SPINT-main/logs/train/runs/2026-07-21-19-11-01/checkpoints/best_ckpt/epoch_019.ckpt` | `c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2` |
| Base `E0` and query decoder | `streaming_calibration_exp/outputs/streaming_calibration/m1_clean_selection_v1_f0_m1_f1_s42_20260801_192017/checkpoints/best.ckpt` | `1ec318f81cfaa9f6eb5e998a2b34135e2bde9941c48fb47bc46f47632f0d6cd8` |

`E0` is specifically the F0 checkpoint's `student.id_encoder`, whose saved variant is `B3`,
with `trial_length=1024`, `hidden_dim=64`, `window_size=100`, `side_dim=0`, and
`identity_mode=calibrated`. It is not reconstructed from the teacher.  Query inference uses the
F0 checkpoint's `student.decoder`, frozen in `eval()` mode. Before any formal run, all 31 decoder
state tensors must have identical key, dtype, shape, and `torch.equal` value to the teacher
checkpoint's corresponding `net.*` tensors. The addendum records this bit-exact relation. If it
does not hold, this protocol is invalid rather than silently switching decoders.

## 2. Two formally different experiments — do not merge their names or claims

Only `C`, its trial lengths, and (where an arm uses them) `obj_id[0:10]` from the calibration NWB
may enter an adapter. No query-trial condition, target, behavior, neural activity, hidden data, or
EvalAI datum may construct `Delta_hat`, fit its normalization, or select its hyperparameters.

For a support feature map `F(C)`, fit a source-session-shared map producing a decoder-coordinate
residual of rank `r`:

```text
Delta_hat = A(F(C)) B^T,  A in R^[N,r], B in R^[W,r], r in {1,2,3}.
```

The only allowed regularization grid is `{1e-4, 1e-2, 1, 100}`.  Rank, regularization,
normalization, and the rate-residualization transform are selected solely by **inner LOSO among
the outer-train source sessions**. The outer-left-out source session is not used in any of those
steps. Once that inner procedure locks `(rank, lambda, transform)`, the outer-left-out selection
window is evaluated exactly once as a gate; it may never choose or revise a parameter.

The support feature map is mechanically fixed. For each unit, let `q_m` be its support-trial
spike sum divided by that trial's positive exposure, `ell_m` the exposure, and `g_m=obj_id_m`:

```text
r_base       = [log1p(mean_m q_m), log(mean_m ell_m)]
r_obj,k      = mean_{m:g_m=k} q_m, or 0 if no such support trial
mask_obj,k   = 1[g_m=k occurs in support],  k in {1,2,3,4}
F_full(C)    = [r_base, r_obj,1:4, mask_obj,1:4]       # 10 dimensions
F_rate(C)    = r_base                                  #  2 dimensions
```

`F_shuffle` is `F_full` after a deterministic, session-and-seed-keyed nonidentity permutation of
the ten `g_m`; it preserves the label multiset. `F_rate_resid` is the eight-dimensional
`[r_obj,mask_obj]` residual after a ridge linear prediction from `r_base`, fitted on inner-train
unit rows only. Each arm's feature mean/std is then fitted only on those same inner-train rows;
zero-variance dimensions use scale one. No session-specific or outer-left-out statistic is
substituted for these transforms.

For each candidate fit, concatenate only outer/inner-train unit rows of `Delta*`, take its thin
SVD, and fix the shared decoder-coordinate basis `B_r=V[:,0:r]`. Project only train rows to
`A*=Delta* B_r`, and fit a ridge map (unpenalized intercept) from source-normalized `F` to `A*`.
The validation/outer session receives only this already fitted basis, map, normalization, and its
own support feature. This row-level construction shares `B_r` across source sessions and forbids
left-out `E_teacher` from entering a deployable fit.

The inner-LOSO criterion is **not** latent error: for each inner-validation source, construct
`E_hat` from its support using the inner-train-fitted map, run the frozen bit-exact F0 decoder on
that source's `[10,210)` query neural activity, and score its paired behavior `R2` difference
against the same decoder with `E0`. Candidate rank/lambda is the mean of those inner-validation
paired behavior `R2` differences (deterministic tie-break: lower rank, then lower lambda).

### A. Deployable shared reduced-rank map (the only deployable claim candidate)

The map parameters are fitted **only** on outer-train session pairs
`{F(C_s), Delta*_s}`. On an outer-left-out session, it receives only `F(C_leftout)` and `E0`; it
must not compute, inspect, or fit to `E_teacher_leftout` or `Delta*_leftout`. Its candidate
identity is `E0_leftout + Delta_hat(F(C_leftout))`. Teacher targets are offline source-training
supervision, not a left-out calibration input.

The calibration label budget is at most ten `obj_id` values when a condition arm is used; there
are no behavior labels. Online calibration state must be reported separately as (i) the existing
F0/B3 support state, (ii) the feature summaries required by `F`, and (iii) the `N*r` residual
coefficients. The report must also give map parameter count, M10 calibration MACs, and the number
of floating-point state values. This arm is eligible for the behavior-R2 endpoint in §5.

### B. Target-support teacher oracle (teacher-assisted, neural-only upper bound)

This distinct experiment may compute `E_teacher_leftout` from the same outer-left-out support and
then form `Delta*_leftout`, including a closed-form rank-restricted fit. It is a **teacher-assisted
target-support oracle**, not a deployable shared map: availability of the teacher identity path
at left-out calibration is precisely the additional high-calibration-compute assumption. Because
directly using `E_teacher_leftout` would already eliminate the alignment problem, a closed-form
fit here is useful only as a neural/latent compatibility upper bound under the declared rank and
feature restrictions; it may not be presented as a low-cost calibration improvement.

Its mandatory accounting is the teacher `fc_id_in/fc_id_out` support compute and its intermediate
`[N,1024]` identity-state requirement, plus the rank-fit workspace/state. Its label budget is
zero unless the particular feature restriction deliberately consumes `obj_id[0:10]`, in which
case that must be stated separately. Its primary measurements are `Delta*` latent MSE/cosine and
rank-reconstruction error only; it cannot pass a behavioral or deployment gate. It may not be
combined with A in an aggregate, named “oracle improvement,” or used to select A's rank/lambda.

## 3. Controls and terminology

Every arm uses the same frozen F0 query decoder, support `[0,10)`, source/session splits,
rank/lambda candidate grid, and adapter fitting budget **within experiment A**. Experiment B is
reported separately under §2 and has no query behavior comparison.

1. **identity/no-alignment:** `E_hat=E0`.
2. **rate-only:** support rate/exposure summaries only; no `obj_id` assignment.
3. **condition-label shuffle:** a deterministic, nonidentity permutation of the ten support
   labels, preserving their multiset; all condition features are then built from shuffled labels.
4. **rate-residualized-condition-only:** first regress condition features on rate/exposure
   features using *inner-train unit rows only*, then use the held-out residual. This tests
   condition information not linearly recoverable from rate/exposure.

The fourth control was formerly called “orthogonal-only” in the v1 feasibility note. It is **not
orthogonal Procrustes** and no rotation/reflection alignment is permitted by this protocol. The
new name is mandatory in all results and figures.

## 4. Development mechanism endpoint, split, and no-leakage rule

The formal unit for experiment A is the already frozen clean-selection M1 cell: fold 1, seed 42,
outer-left-out source session `ses-20120926`. It is a one-cell mechanism screen, not a six-session
or benchmark-wide claim. Experiment B has the same support object but no query behavior endpoint.

| Purpose | Chronological trials | Permitted use |
| --- | --- | --- |
| Support, inner/outer-train | `[0,10)` | Construct `E0`, `E_teacher`/`Delta*` supervision, and features. |
| Support, outer-left-out A | `[0,10)` | Construct only `E0` and `F(C)`; computing/reading `E_teacher` or `Delta*` is forbidden. |
| Support, outer-left-out B | `[0,10)` | May construct `E_teacher` only for the separately named teacher-assisted latent upper bound. |
| Inner selection | `[10,210)` of inner validation sources | Select only rank/lambda/normalization/residualizer using inner LOSO. |
| Outer selection gate | `[10,210)` of outer-left-out source | One locked, no-retuning gate after inner selection. |
| Method-specific locked report | `[210,end)` | One DLA comparison after choices are locked; see historical-window limitation below. |

The **primary development mechanism endpoint for A only** is paired behavior `R2` on the
method-specific locked report window:

```text
Delta R2_primary = R2(F0 decoder with E_hat_DLA) - R2(F0 decoder with E0).
```

Both terms run the identical frozen F0 decoder and identical query neural tensor; only identity
changes. Query behavior is used only to score predictions after `E_hat` has been formed from
support. Query labels never construct the adapter. Latent MSE/cosine to `Delta*` are secondary
diagnostics for A and cannot by themselves pass A. For B they are the only allowed endpoint;
opening or scoring B on this report window is prohibited.

This is **not** a globally sealed or confirmatory window: M1 `[210,end)` was previously read by
the `m1_clean_selection_v1` F0/T4/TS4 report evaluations and later D4 work. It is therefore only
a *method-specific locked report window / held-in development mechanism screen*. DLA must not
read it before its own selection choices are locked, nor tune after reading it, but any resulting
number is not external held-out confirmation. Do not cut a new post-hoc window to restore a
confirmation claim; a positive screen instead requires a separately reviewed new held-out/EvalAI
protocol.

## 5. Predeclared uncertainty, gates, and stopping rule

Before opening the method-specific locked report, estimate a **selection-only paired uncertainty**
from trials `[10,210)`: partition the locked outer-left-out query trials into predeclared
contiguous trial blocks; for each bootstrap replicate resample blocks with replacement, concatenate
all bins/outputs of the chosen trials, recompute aggregate variance-weighted behavior `R2` for
F0 and DLA on the identical resample, then store their difference. Report the resulting paired
standard error, percentile 95% interval, and a two-sided detectable-effect sensitivity (MDE) for
the locked blocks. There is no “per-trial R2” claim. This outer value is a gate only: it may not
trigger another inner fit, rank/lambda sweep, feature change, or normalization refit. Do not claim
that an observed effect below the MDE is evidence of absence.

Experiment A may open the sealed report only when all are true:

- F0 identity and decoder bit-equality addendum passes, `Delta*` is finite and nonzero on every
  audited support session, and the four support `obj_id` levels are present;
- inner fit, inner candidate selection, normalization, and rate-residualizer use no
  outer-left-out source or method-specific report data; the locked outer selection gate is the
  sole permitted use of outer-left-out `[10,210)`;
- selection paired MDE is no larger than `0.015 R2`; otherwise the screen is **non-identifiable**;
- the locked DLA arm improves the selection aggregate over identity by at least `+0.015 R2`, and
  exceeds both rate-only and condition-label-shuffle by at least `+0.010 R2`.

After A's single method-specific locked report is opened, report the paired block-bootstrap
interval and the session-level aggregate value. The mechanism screen passes only if
`Delta R2_primary >= +0.015`, its paired 95% interval excludes zero, and neither rate-only nor
label-shuffle reaches the DLA report value within `0.010 R2`. The
rate-residualized-condition-only arm is a diagnostic control only: it cannot independently pass
or defeat the primary gate, but its result must be reported to qualify the full-arm interpretation.
If a prerequisite fails, record the arm as **ineffective** (clear negative) or
**non-identifiable** (MDE/uncertainty failure), do not try larger rank, a wider lambda grid,
another decoder, or another report split. A pass is still only a one-cell held-in development
mechanism result requiring an independently reviewed held-out/deployment protocol.

## 6. Explicit exclusions

No future-rate target; no decoder/core modification; no FiLM/attention/value residual; no
hyperparameter expansion; no post-hoc window change; no reuse for D4 rescue; no official held-out
selection; no EvalAI access or submission. The predecessor v1 target-existence receipt remains a
historical CPU feasibility artifact; v2 supersedes it for any proposed formal oracle. No result in
this document may be called a formal confirmation or external held-out result.

## 7. Execution outcome (2026-08-02)

The independently audited selection runner completed only the `full` and `rate_only` arms on the
locked `[10,210)` outer-left-out window. `full` improved over identity by `+0.000426 R2`
(`95% CI [-0.000352,+0.001220]`, paired MDE `0.000749`) and rate-only improved by
`+0.000681 R2`. Thus `full-rate_only=-0.000254`, while the required margins were
`full-identity >= +0.015` and `full-rate_only >= +0.010`. The effect was identifiable at the
predeclared practical scale but ineffective.

Because the primary conjunction had already failed, the label-shuffle and diagnostic
rate-residualized arms were not executed. This is a prospective early stop, not missing positive
evidence: neither remaining arm can make the already observed full-vs-identity or full-vs-rate
conditions true. The method-specific report, held-out, EvalAI, hyperparameter expansion, and
quantization remain unopened and unauthorized. The machine-readable decision is
`results/m1_decoder_latent_alignment_oracle_v2/selection_early_stop.json`.
