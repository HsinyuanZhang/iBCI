# Design — M2 AOF Scalar OOF Confirmation V1

Date: 2026-09-03  
Status: **static-package candidate; no official submission authorized**  
Working name: **AOF-S** (Anchored Output Fusion — Scalar)

## 1. Why this successor exists

AOF-M did not beat its predeclared matched scalar control. The matrix therefore
failed and no richer output-fusion operator is allowed. However, the scalar
control itself produced `+0.0063584` mean source OOF improvement, `6/7`
positive sessions, and a worst delta of `-0.002107`. It passes the exact
native-stability thresholds used in the matrix screen.

AOF-S preserves that simpler result and asks one remaining question: does the
source-fitted decoded-output correction transfer to the untouched official M2
surface? It introduces no new fit family, model training, hyperparameter,
target adaptation, or online state.

This result has a deliberately narrow architectural scope. It does **not**
show that pooling after the per-trial MLP is intrinsically inferior when a
network is trained from scratch for that operator. It shows that, starting
from the strong encoder/decoder solution pretrained for native early pooling,
moving the pooling position and retraining the student identity path for 12
epochs while keeping the decoder frozen did not recover the native solution,
let alone surpass it. The likely mechanism is operator-distribution shift: the
frozen nonlinear decoder was trained for one pooling location, so replacing
that location presents an out-of-distribution identity even though the
late-pooled path is not information-free. Full decoder adaptation and a
from-scratch late-pooling model remain untested; this result cannot close them.

The positive residual evidence is correspondingly modest but real. PF-R1 was
about `+0.05 R2` better than PF-MEAN on the matched external screen, and the
small-pool scalar output residual is positive in source OOF. Thus late-pooled
features carry complementary signal; the evidence supports using them only as
an anchored residual around the native prediction, not replacing the native
identity path.

This successor is explicitly post hoc with respect to choosing the scalar as
the surviving method. The scalar arm itself was predeclared and evaluated in
every OOF fold before results were observed. Source OOF is development evidence;
only an untouched official score can confirm transfer.

## 2. Frozen source fit

The only deployment scalar is reconstructed from the immutable AOF-M OOF
sufficient statistics in the frozen seven-session order:

```text
numerator   = -2.643848775861138e-05
denominator =  6.958658366347930e-05
beta        = -0.3799365677508742
```

The successor must prove that every session's numerator, denominator, and row
count is repeated identically across the six OOF fold receipts that contain
that session. It then adds the seven unique terms in declared order and exact
float64. No source arrays are reopened and no fit is rerun.

There is no clamp, ridge, intercept, seed, fold selection, beta sweep, or
target-side update.

## 3. Static deployment operator

For every dataset tag, the build-time payload contains two `[96,50]` float32
identities under the exact `act30_dopt4` calibration contract:

```text
h_native = official activity30 / D-opt4 native identity
h_post   = frozen per-trial post-pool identity averaged over the same first30
```

The post identity is frozen to the source-screen operator, not described only
by shape. Given PIT-cubic activity `A` with shape `[1,30,100,96]` and normalized
D-opt4 side features `S` with shape `[1,96,4]`, both contiguous float32:

```text
V = native_encoder.pre_pool(A.permute(0,1,3,2))
S30 = S.unsqueeze(1).expand(-1, 30, -1, -1)
P = native_encoder.post_pool(concat(V, S30, dim=-1))
total = P[:,0]
for i = 1,...,29 in chronological arrival order:
    total = total + P[:,i]
h_post = total / 30
```

Every intermediate and the final contiguous `[1,96,50]`/`[96,50]` identity is
float32. Tree reduction, float64 accumulation, reordering, `mean` with a
different reduction kernel, or a different activity/side representation is a
different operator and is forbidden.

At every `predict` call, with one shared neural history window `x`:

```text
y_native = D(x, h_native)
y_post   = D(x, h_post)
y_AOF-S  = y_native + beta * (y_post - y_native)
```

The two decoder evaluations use the same frozen weights, same window, same
behavior scale `5.0`, and no update. They remain two independent batch-`B`
calls in the governing path; a concatenated `2B` optimization is diagnostic
only unless separately proven within the inherited tolerance.

At exact IEEE `beta=+0.0`, the public zero-control path returns the native
prediction object directly and does not evaluate `y_post` or a multiply-add.
Negative zero is not the sentinel.

For nonzero beta, each decoder call takes the last output bin, transfers it to
a contiguous NumPy float32 array, and divides by `np.float32(5.0)`. Fusion then
uses the literal NumPy expression above in that order with the frozen Python
float beta, and the public result must remain contiguous float32. Algebraically
equivalent reassociation or a fused kernel is not governing unless it first
reproduces the reference prediction within max-abs `2e-6` and R2 absolute
`2e-7` on all local rows.

## 4. Build-time calibration authority

Reuse the exact official `act30_dopt4` package:

- selected checkpoint/state;
- 13 dataset tags and native identity payload bytes;
- first-30 PIT-cubic activity;
- D-opt4 support and ridge T4;
- frozen normalizer and scale-5 decoder;
- no hidden-query label, optimizer, gradient, online memory, or trial boundary.

The post identities may be computed only during package construction from the
same public calibration inputs. D-opt4 T4 consumes the same four public
calibration labels as the comparator; activity30 remains label-free. Runtime
receives cached identities only.

The governing deployment authority is the exact package scored as submission
`581361`, not the later reconstruction:

- export receipt
  `sua_exploration/evalai_t4_m2_activity_budget/artifacts/t4_m2_seed42_ridge_m4_activity30_identity.receipt.json`,
  SHA `a3c304106ce56105c3bde59b3237604e95388608d740800da4ff2da0fa7186db`;
- native payload SHA
  `c51b71167d81490927fee8a552785ee27ddff7e90085ed3ff4b18b857afbac40`;
- scored image/manifest
  `sha256:378bbd4868e515a3527418e098db2989e0fc8fbccc929ee029a237ce5e9f291b`;
- sealed local score
  `tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json`, SHA
  `6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce`;
- checkpoint SHA
  `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`;
- student-state SHA
  `2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20`.

All 13 session records must preserve the receipt's dataset tag, activity,
selected-support, raw T4, normalized T4, side, native-identity, normalizer, and
checkpoint links. Source-screen native/post prediction parity exists only for
the seven source sessions; the six external identities receive constructibility
and local-validation checks but must not be described as source-parity rows.

The later local reconstruction remains a required numerical-bridge authority,
not the deployed native payload: receipt SHA
`6f90230f9f8f330edeec970ea0108defde24cd43ca3195fea264083cac6fa583`,
payload SHA
`e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261`,
and image/manifest
`sha256:79ac29ca71a84eb97be59411fc80d2a567127e95fd96b63dfa042d22b5a4421f`.
Its CPU identities and the source-screen GPU identities are known to differ at
the floating-point level. AOF-S construction must therefore retain the exact
`c51b...` decoder/native identities and separately prove the declared source
bridges; it must never replace `c51b...` native identities with `e4ff...` ones.

The source AOF/AOF-M GPU operator and official CPU cached identities are not
byte-identical. Preserve the established authority split: exact payload bytes
on CPU, exact source-screen native prediction on GPU, and only the declared
numerical bridges (`identity <=2e-6`, prediction `<=2e-6`, R2 `<=2e-7`). Never
claim cross-device SHA equality.

## 5. Required local validation

Before any network action, the static package must prove:

1. exact descriptor validation of the valid-terminal-but-gate-failed AOF-M
   terminal/OOF and its AOF V1 predecessor, including `passed=false`,
   `all7_refit_performed=false`, no `failure.json`, and no all-seven matrix;
2. exact reconstruction of the seven scalar OOF rows and all-seven beta;
3. 13/13 native cached identities equal the exact `c51b...` payload used by
   submission `581361`, and all 13 activity/support/T4/side/input digests match
   that frozen export receipt;
4. 13/13 post identities have the declared shape, dtype, finite values, and
   build-time input digests;
5. exact `+0` native prediction control for every local validation row;
6. the seven available source rows match the source AOF operator when the same
   native/post identities and inputs are used, within max-abs `2e-6` and R2
   absolute `2e-7`; no equivalent source-parity claim is made for the six
   external tags;
7. native and post calls share identical input-window digests;
8. decoder weights and both identity maps are unchanged before/after replay;
9. `on_done` remains a no-op; no optimizer/gradient/parameter update exists;
10. no hidden/EvalAI target file is opened during build or validation.

Local R2 is descriptive and cannot select or alter beta.

## 6. Official interpretation

The exact scored comparator is private EvalAI submission `581361`, cue-budgeted
M4 T4 plus activity30. Its immutable terminal receipt is
`sua_exploration/evalai_t4_m2_activity_budget/artifacts/evalai_submission_581361_terminal_receipt_v1.json`
with SHA `68da5427e2d27169b8b6e1e08a487113b65b734a7e7644a5d1778f993febbb97`.
Its held-out R2 is `0.2897439880338965`, held-in R2 is
`0.5627788928403332`, and normalized latency is `0.04701159050299251`.

AOF-S must reuse that exact `c51b...` decoder and native-identity map. The
later `e4ff...` / `79ac...` reconstruction has the same calibration law but is
bound only as numerical bridge evidence and is never the governing native
payload. Therefore a future AOF-S submission changes only the added post
identity map and literal output-fusion operator relative to the exact scored
native package. It remains one untouched confirmation point, not another
development trial.

- `delta >= +0.010`: meaningful PF improvement candidate;
- `+0.003 <= delta < +0.010`: transferable but small diagnostic effect;
- `|delta| < +0.003`: practical tie; close PF output fusion;
- `delta <= -0.003`: harmful transfer; close PF output fusion.

No official submission is authorized by this design. Submission requires a
separate operator decision, available quota, an immutable locally validated
image, and exact comparison against the matched official result.

## 7. Stop rule

Regardless of outcome, do not create a matrix, per-output beta, intercept,
nonlinear combiner, or target-fitted successor. AOF-S is the final PF
output-fusion candidate.
