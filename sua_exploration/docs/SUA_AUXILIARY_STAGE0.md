# SUA auxiliary Stage-0: source separation, not raw concat

**Status (2026-07-30 07:00 +08:00):** the read-only eligibility audit,
end-to-end relation integration, strict train/validation isolation, CPU
mechanism tests and the complete validation-only `4 arms × 3 seeds` matrix
have passed their integrity checks. The final strict group verdict is
**ineffective**: the real same-electrode relation is worse than T4 on all
three seeds and does not beat the parameter-matched no-group arm by the
pre-registered margin. The route stops here; relative amplitude and extra
seeds are not launched. No formal-test session was constructed or opened.

This note implements the scope in
[T4_SUA_AUXILIARY_EXPERIMENT_PROGRAM.md](T4_SUA_AUXILIARY_EXPERIMENT_PROGRAM.md).
It supersedes any interpretation that a generic static waveform/SNR gate is the
next SUA mechanism.  The prior F1 screen did not support static SNR concat
(`F1-F0=-0.0324`, `F1-FS1=-0.0457`); no raw waveform concat is repeated here.

## Boundary, cache, and provenance

The audit used exactly `sub-C / CO / units < 100 / 27-6-6`, 50 rewarded
calibration trials, and created only the versioned cache namespace
`cache/sua_auxiliary_stage0_v1/sessions`.  The cache key contains the Stage-0
feature version, source fingerprint, signal view, pool size, bin/window
parameters, and reward filter.  The full receipt, including the 33 opened
train/validation session names and the six test names that were *not* opened,
is in [`audit.json`](../results/sua_auxiliary_stage0/audit.json).

The formal test files `20151113/16/17/19/20` and `20151201` were not opened for
neural, behavioral, waveform, or trial data.  No formal-test receipt was
created or changed.

Waveform columns are deliberately bounded to the first 256 calibration-pool
waveforms/unit.  This makes the diagnostic reproducible and prevents a
high-rate session from silently becoming a data-scale waveform experiment.
They are not inputs to the proposed model.

## Reproducible merge-sensitivity diagnostic

For an electrode group `e` with sorted units `i`, define the unit-weighted
within-electrode dispersion

```text
H_x = sum_e sum_{i in e} ||x_i - mean_{j in e} x_j||² / number_of_terms,
```

only over groups of size >1.  Stage-0 calculates it separately for:

- `x=T4`: train-normalized `[a,c,m,b]` fit on the same first 50 rewarded trials;
- `x=activity`: the per-unit 50-trial `log(1 + rate)` trajectory; and
- secondary relative-amplitude diagnostic: `log(1+p2p)` (never absolute
  electrode ID and only meaningful against an electrode-group mean).

It also retains calibration T4 relative residual, design rank/condition, and
spike exposure.  These form the only candidate *source-separation context*:
`[T4 residual, log(1+condition), rank-valid, log(1+exposure)]`.

Static SNR, waveform residual CV, and template drift are recorded only as
read-only negative diagnostics.  They cannot be sent as a default FiLM input,
cannot be headline results, and cannot rescue F1's failed static SNR premise.

To connect the diagnostic to existing evidence without inventing
per-electrode R², the audit reports the six session-level F0 and T4
`R²_SUA-R²_pMUA` gaps from the frozen bridge artifact and their descriptive
Pearson correlations with coverage/heterogeneity.  There is no claim of
causality or statistical inference at `n=6`; in particular, T4-gap correlation
with multi-unit share is `+0.036` and with T4 dispersion is `-0.047`.  The
current artifacts do not contain per-electrode decoder R², so an alleged
stratum decomposition of those R²s would be nonidentified.  The defined
diagnostic is instead a pre-training, source-level merge-sensitivity measure.

## Audit outcome

| Validation session | units/electrodes | multi-unit electrode share | multi-unit unit share | max group | `H_T4` | `H_activity` |
|---|---:|---:|---:|---:|---:|---:|
| 20151103 | 38/25 | 36.0% | 57.9% | 4 | 0.377 | 0.267 |
| 20151104 | 59/40 | 35.0% | 55.9% | 4 | 0.620 | 0.199 |
| 20151106 | 60/39 | 35.9% | 58.3% | 3 | 0.384 | 0.167 |
| 20151109 | 65/41 | 39.0% | 61.5% | 4 | 0.552 | 0.324 |
| 20151110 | 61/39 | 38.5% | 60.7% | 3 | 0.621 | 0.288 |
| 20151112 | 42/33 | 21.2% | 38.1% | 3 | 0.459 | 0.248 |

Every validation T4 design has rank 3 and condition number `sqrt(2)`.  Five of
six sessions have a multi-unit unit share >=50%, and all six have nonzero T4
and activity heterogeneity.  Thus the narrow **relation eligibility** rule
passes.  It does **not** establish a performance gain and does not make
generic quality gating eligible.

## Minimal interfaces and controls implemented

[`mc_maze/sua_auxiliary_stage0.py`](../mc_maze/sua_auxiliary_stage0.py) adds
the isolated, CPU-testable building blocks. The accepted relation path is now
wired into the shared encoder and DANDI training/evaluation pipeline under
fresh variant tokens; the historical T4 artifacts and implementation are not
modified.

- `ZeroInitLowRankFiLM`: an activity-conditioned, low-rank FiLM interface.
  Scale and shift heads are zero initialized, so it is exactly the selected
  baseline at step 0.  Its input is intentionally called `context`, not
  `quality`; the allowed primary context is source separation above.
- `ParameterMatchedConcatMLP`: nonlinear `[activity, context]` residual control
  with exact initialization equivalence and the nearest real hidden-width
  parameter count (for `D=64,Q=9,r=8`, 1168 versus FiLM 1232, `−5.2%`).
  No dummy parameters are used. It has no multiplicative activity-context
  interaction.
- `SegmentedMeanResidual`: uses segmented sums/counts (linear in units; no
  `N²` attention and no absolute electrode table).  Its output depends on
  `u_i-mean(u_e)` and `log(group size)`.  Hence all-singleton membership is
  **exactly** the baseline, including after relation-head weights move.
- `ParameterMatchedNoGroupMLP`, deterministic row shuffle/mismatch helper, and
  membership shuffle preserving each session's exact group-size histogram.

The integrated variants are:

- `B3S/t4`: fresh T4 substrate control;
- `B3SER/t4rel`: equality-only same-electrode relation, without an absolute
  electrode lookup;
- `B3SER/t4rel_membership_shuffled`: deterministic non-identity membership
  shuffle preserving the exact per-session group-size histogram;
- `B3SERN/t4rel_nogroup`: the same relation-unit/output widths but no
  membership and no group statistic.

`B3SER` adds about 890 parameters over T4 and uses linear segmented reductions,
not `N²` attention. Both relation and no-group residual output layers are
bias-free and zero initialized. Tests verify that all-singleton membership
remains exactly equivalent to the T4 substrate even after an optimizer step.

The required eventual relation comparison is: baseline versus relation,
membership shuffle, no-group MLP, and singleton pseudo-MUA boundary.  Relative
amplitude may be appended only as a group-relative secondary scalar and must
receive the same membership-shuffle test.  No absolute ID embedding/gate/table
is revived; design D was already ineffective.

## Tests and smoke status

CPU contracts pass. The expanded isolation/data-path suite reports 16 passing
tests and the targeted relation-encoder suite reports 2 passing tests; an
independent maintainer selection reports 9 passing relation/manifest tests:

```text
/home/xinyuan/miniconda3/envs/spint/bin/python -m pytest -q \
  sua_exploration/tests/test_sua_auxiliary_stage0.py \
  sua_exploration/tests/test_pseudomua_t4_bridge.py
# expanded suite: 16 passed; targeted encoder: 2 passed
```

They cover finite/degenerate confidence fields, row/membership marginal
preservation, zero-init FiLM and concat equivalence, explicitly bounded
near-parameter matching, unit permutation equivariance, and exact singleton
degeneracy.
The completed real-data cache smoke uses train-only normalization over all 27
train sessions and fixed validation loading over all six validation sessions.
In addition, [`component_smoke.json`](../results/sua_auxiliary_stage0/component_smoke.json)
records one CPU optimizer step on the frozen first train session (20131003) and
one fixed validation forward (20151103).  It verifies exact baseline equality
at initialization and a finite post-step validation forward, using only
`[T4 residual, condition, rank, exposure]`; it does not use behavior labels,
does not report R², and does not open a test cache/NWB file.

The integration now uses a frozen manifest,
`configs/subc_co_27_6_strict_train_val_manifest.json`. Only the 27 train and
six validation paths are resolved. The six formal-test entries remain names in
a receipt; `session_files.test=[]`. Training records the manifest SHA-256, and
the epoch-window evaluator re-hashes the exact bytes before scoring. The
relation contract separately fixes training activity calibration to 10 trials
and evaluation forward calibration to 30 trials.

The matrix used 12 epochs, no early stopping, per-epoch checkpoints and the
fixed epoch 5--12 validation mean. Seed 42 ran as the first complete block;
GPU0 then ran seed 43 and seed-44 REL-MS/REL-NG, while GPU1 ran seed-44
T4/REL. The two seed-44 halves were strictly aggregated together. This
partition kept both GPUs productive without selecting arms or seeds from
intermediate results. Runtime logs and the final aggregate are under
`results/sua_electrode_relation_full_v1_scheduler/`.

### First two strict paired results

At 23:02 and 23:22 HKT, the first two fixed epoch-window REL artifacts
completed and passed the fail-closed single-arm validator:

| Seed | T4 | REL | REL−T4 | positive validation sessions |
|---:|---:|---:|---:|---:|
| 42 | 0.572816744 | 0.572654321 | **−0.000162423** | 2/6 |
| 43 | 0.554434118 | 0.553392850 | **−0.001041267** | 2/6 |

For seed 42, the six session deltas are
`+0.061261/−0.004633/+0.002535/−0.015507/−0.010466/−0.034164`;
their descriptive session-paired SE is `0.013288967`. For seed 43 they are
`+0.059098/−0.009084/−0.023087/−0.023768/+0.008585/−0.017992`,
with session-paired SE `0.012992166`.

The two seed means average `−0.000601845`; both are negative. The paired SE
across those two seed means is `0.000439422`. This is replicated evidence that
the current relation form does not provide a large net gain over T4. It remains
an interim two-seed decision rather than the pre-registered three-seed final
artifact.

All arms use the same
strict manifest SHA-256, train activity calibration `n=10`, evaluation forward
calibration `n=30`, epochs 5--12, and report
`no_test_files_evaluated=true`.

All four arms are now complete for seeds 42 and 43. Seed 44 remains required
for completion of the frozen matrix, not because the first two seeds justify
selective continuation.

The complete seed-42 four-arm block subsequently passed strict aggregation:

| Arm | epoch-window validation R² |
|---|---:|
| T4 | 0.572816744 |
| REL | 0.572654321 |
| REL-MS | 0.569188251 |
| REL-NG | **0.579877558** |

| Required seed-42 pair | mean ΔR² | positive sessions |
|---|---:|---:|
| REL−T4 | −0.000162423 | 2/6 |
| REL−REL-MS | +0.003466069 | 1/6 |
| REL−REL-NG | **−0.007223237** | 1/6 |

Thus none of the three necessary comparisons supports effectiveness in seed
42, and the parameter-matched no-group arm is numerically better than the real
relation. This remains a single-seed block, not the final three-seed verdict.
Its strict aggregate is
`results/sua_electrode_relation_pilot_v2/seed42_strict_aggregate.json`.

The complete seed-43 block independently passed the same raw-artifact and
manifest checks:

| Arm | epoch-window validation R² |
|---|---:|
| T4 | 0.554434118 |
| REL | 0.553392850 |
| REL-MS | 0.530710293 |
| REL-NG | 0.546593416 |

| Required seed-43 pair | mean ΔR² | positive sessions |
|---|---:|---:|
| REL−T4 | **−0.001041267** | 2/6 |
| REL−REL-MS | +0.022682557 | 5/6 |
| REL−REL-NG | +0.006799435 | 4/6 |

The complete seed-44 block also passed the same raw-artifact and manifest
checks. Its fresh T4 artifact is exactly identical to the historical
`e3_tuning_ablation/t4_s44.json`, including every epoch score.

| Arm | seed-44 epoch-window validation R² |
|---|---:|
| T4 | 0.572995040 |
| REL | 0.569878852 |
| REL-MS | 0.545060549 |
| REL-NG | 0.549966383 |

| Required seed-44 pair | mean ΔR² | positive sessions |
|---|---:|---:|
| REL−T4 | **−0.003116188** | 2/6 |
| REL−REL-MS | +0.024818303 | 5/6 |
| REL−REL-NG | +0.019912469 | 4/6 |

The final strict cross-seed aggregate reopens and revalidates all 12 raw arm
artifacts. The three-seed arm means are T4 `0.566748634`, REL `0.565308674`,
REL-MS `0.548319698` and REL-NG `0.558812452`. The three required comparisons
are:

| Final paired comparison | mean ΔR² ± paired SE | positive seeds | positive sessions | verdict |
|---|---:|---:|---:|---|
| REL−T4 | **−0.001439959 ± 0.000875671** | 0/3 | 1/6 | **ineffective** |
| REL−REL-MS | +0.016988976 ± 0.006789505 | 3/3 | 5/6 | indeterminate |
| REL−REL-NG | +0.006496222 ± 0.007834871 | 2/3 | 3/6 | **ineffective** |

For REL−T4, the upper two-SE bound is only `+0.000311382`; for REL−REL-NG it
is `+0.022165963`. Both are below the pre-registered `+0.03` threshold, and
REL−T4 is negative for seeds 42, 43 and 44 individually. Therefore the final
group verdict is `ineffective`. No extra seeds and no relative-amplitude stage
are justified. The authority is
`results/sua_electrode_relation_full_v1_scheduler/multiseed_strict_aggregate.json`;
the two-seed interim artifact remains only as an audit trail.

The first seed-42 REL-NG launch fail-closed before any training because
`base_feature_group()` did not canonicalize `t4rel_nogroup` to the T4 feature
registry. The missing token mapping was added and covered by a targeted test;
11 Stage-0 tests passed. Only the missing REL-NG cell was recovered under a
fresh run token, after which all four referenced raw artifacts were
revalidated. No successful cell was rerun and no formal-test file was opened.
Completion of this recovered block automatically released GPU1 to the
pre-registered seed-44 T4/REL queue.

The first REL training run also exposed an implementation overhead: its epoch
time is about twice T4 despite only about 890 extra parameters. Inspection
localizes this to a Python loop over batch rows and a per-row
`labels.max().item()` GPU-to-CPU synchronization in segmented grouping, not to
an `N²` relation. The frozen matrix was left unchanged for comparability.
Vectorized batch-offset `scatter_add` remains an algebraically equivalent
implementation improvement if another future mechanism reuses this operator,
but the failed efficacy gate gives no reason to optimize or deploy this
relation path itself.

## Decision

The complete matrix resolves the same-electrode route as **ineffective**.
Eligibility was real—multi-unit groups and within-electrode heterogeneity
exist—but the proposed membership relation did not convert that structure
into higher SUA R². REL was below T4 on every seed, while its apparent gain
over shuffled membership was too small and did not establish an advantage
over the parameter-matched no-group model.

Stop the same-electrode relation route. Do not add seeds 45--47 and do not
enter the relative-amplitude stage. SNR, waveform residual CV and template
drift remain read-only diagnostics, not model inputs. The result is
validation development evidence; no formal held-out conclusion is claimed.
