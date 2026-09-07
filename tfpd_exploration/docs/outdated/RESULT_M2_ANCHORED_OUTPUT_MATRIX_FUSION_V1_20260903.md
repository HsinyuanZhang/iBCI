# Result — M2 Anchored Output Matrix Fusion V1

Date: 2026-09-03  
Status: **valid terminal; matrix gate failed; no all-seven matrix refit**  
Route: **AOF-M** (Anchored Output Fusion, Matrix residual)  
Scope: source-only development OOF; official/EvalAI target untouched

## 1. Immutable result authority

Canonical root:

```text
tfpd_exploration/results/m2_anchored_output_matrix_fusion_v1/source_oof
```

The root contains exactly six immutable body/sidecar pairs:

```text
attempt.json
predecessor_authority.json
launch.json
source_authority.json
oof.json
terminal.json
```

There is no `failure.json` and no `all7_matrix.json`. Every body and sidecar is
mode `0444`, `nlink=1`, and the sidecars verify. The terminal body SHA-256 is

```text
3bc012339f73ab207d18fcddd701adf1fb884c73bb34512cffb0376ba32f1a6e
```

Reviewed, initial, and final execution closure are identical:

```text
eb74bd47cd0e97f10474d1472919f3b591ef2f0b70feededee44dc77329462e8
```

The run used physical GPU0 only. Source target access is disclosed; hidden,
external, and EvalAI target access are false. One PIT source materialization
was used, no optimizer was constructed, and model/target parameter updates are
zero.

## 2. Governing AOF-M result

| Held source session | Matrix − native ΔR² | Matrix − matched scalar ΔR² |
|---|---:|---:|
| 2020-10-19 Run1 | +0.0088788832 | −0.0012244221 |
| 2020-10-19 Run2 | +0.0138199943 | −0.0016417736 |
| 2020-10-20 Run1 | +0.0070527542 | −0.0009796173 |
| 2020-10-20 Run2 | +0.0064970742 | +0.0003280245 |
| 2020-10-27 Run1 | −0.0002123324 | −0.0002695095 |
| 2020-10-27 Run2 | −0.0030363838 | −0.0009293986 |
| 2020-10-28 Run1 | +0.0063195335 | −0.0004723838 |

The matrix passed every native-stability component:

```text
mean OOF(matrix - native) = +0.005617074728110476  [pass: >= +0.005]
positive sessions         = 5/7                    [pass: >= 5/7]
worst session             = -0.003036383848561308  [pass: >= -0.005]
```

It failed the incremental-capacity component:

```text
mean OOF(matrix - matched scalar) = -0.0007412972195724851
required                              >= +0.003
```

Thus the terminal correctly records `passed=false` and
`all7_refit_performed=false`.

All seven matrix systems were well conditioned: condition numbers are about
`2.34–2.78`, eigenvalue ratios about `0.359–0.427`, antisymmetric Gram residue
is zero, and relative solve residual is at most approximately `9.19e-17`.
The negative incremental result is not a numerical-identifiability failure.

This does not establish that post-MLP pooling is intrinsically inferior for a
network trained from scratch around that operator. The experiment starts from
a strong solution pretrained for native early pooling, changes the pooling
location, and gives the trainable identity path 12 epochs of retraining while
the decoder remains frozen. The supported conclusion is narrower: under that
transfer contract, late pooling did not recover the early-pooled solution and
did not surpass it. A frozen nonlinear decoder trained for one pooling
location sees the other location as an out-of-distribution identity operator.
Full decoder adaptation and a from-scratch late-pooling model were not tested
and are not ruled out by this result.

Nor is the late-pooled path devoid of information. In the matched corrected
screen PF-R1 exceeded PF-MEAN by about `+0.0515 R2` on the external surface,
and the small-pool scalar output residual below is positive in source OOF.
These observations support a small complementary residual around the native
anchor; they do not support replacing the native identity path. The residual
also weakens as the pool grows, which is why capacity expansion is closed.

## 3. The predeclared matched-scalar diagnostic

The matched scalar was not chosen after looking at an arm grid. It was the
single predeclared control fit on the exact same six sessions inside every
fold. Its held-session deltas are:

| Held source session | Scalar − native ΔR² | Fold beta |
|---|---:|---:|
| 2020-10-19 Run1 | +0.0101033052 | −0.3929074277 |
| 2020-10-19 Run2 | +0.0154617679 | −0.3451611020 |
| 2020-10-20 Run1 | +0.0080323715 | −0.3581291032 |
| 2020-10-20 Run2 | +0.0061690497 | −0.3713094390 |
| 2020-10-27 Run1 | +0.0000571772 | −0.3991453662 |
| 2020-10-27 Run2 | −0.0021069852 | −0.4196983404 |
| 2020-10-28 Run1 | +0.0067919174 | −0.3763730480 |

Its independently recomputed summary is:

```text
mean OOF(scalar - native) = +0.006358371947682961
positive sessions         = 6/7
worst session             = -0.0021069852144761647
```

Therefore the scalar passes the same three native-stability thresholds that
the matrix passed, while the matrix is on average `0.0007413` worse. This is a
clean simplicity result: the PF decoded residual contains a small transferable
correction, but a four-parameter rotation/rescaling does not improve its use.

The seven unique per-session sufficient-statistic records in the immutable OOF
receipt reconstruct, in frozen roster order:

```text
all-seven numerator   = -2.643848775861138e-05
all-seven denominator =  6.958658366347930e-05
all-seven beta        = -0.3799365677508742
```

This reconstruction requires no new source-array or target access. A successor
must descriptor-bind the OOF graph and independently verify that all repeated
per-session terms agree before accepting these values.

## 4. Decision

The AOF-M gate remains failed. It must not be rewritten as a matrix success,
and it closes intercepts, diagonal/full-matrix variants, nonlinear output
combiners, session-specific fusion, and further PF capacity searches.

One narrow confirmation remains scientifically justified: package the
predeclared matched scalar as **AOF-S**, using the all-seven beta above, two
static cached identities, and two frozen decoder evaluations. This is not a
continuation of matrix-capacity exploration. It is a test of the simpler
control that won the source OOF comparison.

The AOF-S hypothesis is still development-selected and its source effect is
small. Local/source results cannot promote it. A local package must first prove
exact native anchoring, model/identity immutability, matched calibration law,
and numerical agreement with the AOF source operator. A later untouched
EvalAI point may be requested separately; only a delta of at least `+0.010 R2`
over the exact `act30_dopt4` official comparator is a meaningful PF result.
