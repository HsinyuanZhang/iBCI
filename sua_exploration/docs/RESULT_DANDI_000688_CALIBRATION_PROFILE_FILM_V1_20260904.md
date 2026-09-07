# Result: DANDI 000688 Calibration-Profile FiLM V1

Status: `SOURCE_ONLY_TERMINAL__PROFILE_NULL__NO_FORMAL_TEST__NO_EVALAI`

## 1. Decision

The low/high-speed calibration profile did not provide a reproducible semantic
benefit on the frozen DANDI 000688 M30/T4@30 early-pooling substrate.

Across three paired seeds and six validation sessions per seed:

- `CP10@M10 - NATIVE = +0.000179` mean R2;
- `CP30@M30 - NATIVE = +0.000887` mean R2;
- `EMPTY@ZERO - NATIVE = +0.003680` mean R2;
- `SHUFFLE10@M10 - NATIVE = +0.000112` mean R2.

The real profiles therefore did not beat the matched capacity control:

- `CP10@M10 - EMPTY@ZERO = -0.003502`;
- `CP30@M30 - EMPTY@ZERO = -0.002793`.

This is a profile-semantic null, not evidence that adding a small trainable
head is beneficial. It also does not establish deployment noninferiority:
CP10's worst seed-session delta was `-0.034603`, below the frozen `-0.030`
single-seed bound. No formal-test data were opened and no EvalAI submission was
made.

## 2. Frozen experiment

Governing inputs:

- design SHA256:
  `5aa3d8a6abb7a627bb710e49154673caa2e0e829d66f6c867cdb595fae63ba42`;
- workorder SHA256:
  `2002c974793e836af1e34bd610ad715015bf51842817f2e1faa6dcb1ff23df93`;
- strict source manifest SHA256:
  `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`;
- seeds: 42, 43, and 44;
- source split: 27 train / 6 validation / 6 formal-test names only;
- substrate: each seed's final-epoch M30/T4@30 B3S checkpoint;
- activity/T4 support: first 30 chronological rewarded trials;
- profile horizons: first 10 or first 30 rewarded trials;
- profile: per-unit low/high-speed mean contrast and log-mean contrast,
  robust-z normalized; standard-deviation columns were receipted but masked;
- query windows: rewarded trials 50 onward;
- base B3S encoder, T4, post-pool network, and decoder frozen;
- trainable head: zero-initialized `8 -> 8 -> 128` FiLM MLP, 1,224
  parameters;
- training: 12 epochs, Adam `3e-4`, batch 32, at most 1,024 deterministic
  windows per source session per epoch;
- four paired arms on GPU0: `CP10`, `CP30`, `SHUFFLE10`, and `EMPTY`.

All four arms shared each resident input batch. Their zero-initialized identity
and prediction were bitwise equal to native before the first update. GPU1 was
not used.

## 3. Per-seed primary results

All values are equal-session mean R2 deltas against the seed-matched native
checkpoint on identical Q50 windows.

| Seed | CP10@M10 | CP30@M30 | EMPTY@ZERO | SHUFFLE10@M10 | CP10 NI | CP30 NI |
|---:|---:|---:|---:|---:|:---:|:---:|
| 42 | -0.004932 | -0.008930 | -0.004430 | -0.007583 | yes | no |
| 43 | -0.002058 | +0.002678 | +0.002450 | -0.000868 | no | yes |
| 44 | +0.007526 | +0.008913 | +0.013021 | +0.008787 | yes | yes |
| three-seed mean | +0.000179 | +0.000887 | +0.003680 | +0.000112 | descriptive | descriptive |

The `NI` columns reproduce the frozen per-seed compute-allocation predicate:
mean delta at least `-0.005` and worst session at least `-0.030`. `PASS` in a
terminal receipt means that the run completed and the seed-expansion decision
was evaluated; it is not a scientific-positive label.

## 4. Paired 18-cell evidence

The 18 cells are the crossed set of three seeds and six validation sessions.
They are useful descriptively, but they are not treated as 18 independent
biological sessions.

| Cell | Mean delta | Median delta | Positive | Worst | Crossed seed/session bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| CP10@M10 | +0.000179 | -0.001761 | 8/18 | -0.034603 | [-0.01115, +0.01303] |
| CP30@M30 | +0.000887 | -0.003379 | 8/18 | -0.024530 | [-0.01090, +0.01586] |
| EMPTY@ZERO | +0.003680 | +0.000249 | 9/18 | -0.026430 | [-0.00920, +0.01877] |
| SHUFFLE10@M10 | +0.000112 | -0.003108 | 8/18 | -0.025690 | [-0.01208, +0.01438] |

The bootstrap resamples both the three seed indices and six session indices
with replacement using a frozen diagnostic seed. It was computed after the
three terminal runs and is descriptive, not a new acceptance test.

The most discriminating paired comparisons are:

| Comparison | Mean delta | Positive seed-session cells |
|---|---:|---:|
| CP10@M10 minus EMPTY@ZERO | -0.003502 | 3/18 |
| CP30@M30 minus EMPTY@ZERO | -0.002793 | 6/18 |
| CP10@M10 minus SHUFFLE10@M10 | +0.000066 | 9/18 |

At the session level after averaging the three seeds, CP10 beat EMPTY in only
one of six sessions. CP30 also beat EMPTY in only one of six sessions. This is
the main evidence that the profile contents are not useful under this
operator, even though the trainable head itself can move predictions.

## 5. Interpretation

### 5.1 What is supported

The FiLM parameterization is structurally portable: all three runs trained and
scored without changing the base encoder or decoder, and the average native
degradation of the real-profile arms was close to zero. A compact M10 profile
was no worse than an M30 profile overall.

The data do not support a DANDI performance claim. The sign varies strongly by
seed, the crossed intervals contain zero, and the empty-profile capacity
control is better than both real-profile arms on average.

### 5.2 Mechanism boundary

The result is consistent with the pre-registered boundary condition. DANDI's
T4 carrier already includes preferred-direction coefficients, modulation
depth, and baseline. The frozen activity identity also pools the complete
trial waveform. A low/high-speed firing-rate contrast may therefore repeat
information already present in T4 and the pooled activity representation.

This differs from M2, where the T4 carrier lacks an explicit baseline term,
and from the H1 local positive instance, whose carrier lacks a speed/rest
term. The correct cross-dataset statement is conditional:

> Calibration-profile FiLM is useful when the frozen carrier/identity leaves a
> task-state calibration gap; it is not expected to help when that information
> is already encoded.

This statement remains a hypothesis until the profile-redundancy diagnostic
is completed. It must not be strengthened into a causal claim from this screen
alone.

### 5.3 Why the final-epoch loss spikes do not rescue the claim

All four arms saw identical batch contents, so their epoch-mean losses move
together. The large epoch-to-epoch changes mainly measure the heavy-tailed
window sample selected for that epoch. The final optimizer state is cumulative;
one high-loss final epoch does not by itself prove that the checkpoint was
corrupted. Only final head checkpoints were frozen, so selecting an earlier
epoch after seeing validation R2 would be post-hoc selection. A future
successor may pre-register Polyak averaging or a decaying learning rate, but
it must retain an EMPTY control and cannot reinterpret this V1 result.

## 6. Completed post-result diagnostics

### 6.1 Profile redundancy

A source-only grouped ridge diagnostic selected its regularization on the 27
training sessions and evaluated the six validation sessions. It used seed42's
frozen B3S activity representation and made no model update.

| Target | Feature | Validation equal-session R2 | Residual variance fraction |
|---|---|---:|---:|
| M10 profile | T4 | -0.0129 | 1.0129 |
| M10 profile | frozen h | 0.5183 | 0.4817 |
| M10 profile | T4 + frozen h | 0.5198 | 0.4802 |
| M30 profile | T4 | -0.0066 | 1.0066 |
| M30 profile | frozen h | 0.4572 | 0.5428 |
| M30 profile | T4 + frozen h | 0.4597 | 0.5403 |

This rejects the strongest redundancy claim: T4 does not linearly reconstruct
the profile, and roughly half of the profile variance remains unexplained by
`[T4,h]`. It does not prove that the residual profile variance is relevant to
behavior prediction.

The diagnostic implementation is
`sua_exploration/scripts/audit_dandi688_cp_profile_redundancy_v1.py`, SHA256
`82510bb541ad74a64513ebfb8f86c7ee1bcca38ac5acb7b910ac7ab6acd8ed63`.

### 6.1.1 Profile split-half reliability

A separate source-only, zero-model diagnostic split the first M10 or M30
rewarded trials into chronological odd/even halves. Each half independently
recomputed the speed quartiles, the two active profile columns, and their
within-session robust-z normalization. Correlations were then computed across
the unit rows of each session and profile column.

| Horizon | Mean r over 33 sessions x 2 columns | Median r | Positive cells | Spearman--Brown from mean r |
|---|---:|---:|---:|---:|
| M10 | 0.7765 | 0.8185 | 64/66 | 0.8742 |
| M30 | 0.9163 | 0.9281 | 66/66 | 0.9563 |

The M30 profile is therefore highly repeatable under disjoint calibration
trials. The approximately 54% variance not linearly explained by `[T4,h]`
cannot be dismissed as split-half estimator noise. The combined evidence says
that the profile is stable but has not shown incremental task value under the
tested full-T4 FiLM operator; it does not say that the profile is absent.

The diagnostic implementation is
`sua_exploration/scripts/audit_dandi688_cp_profile_split_half_v1.py`, SHA256
`e83f96a044b00f05b7b997bb77255402fcd3e3e3764905e6fdc38a89f7e86f40`.

### 6.2 Alternative attack surfaces

A second zero-update diagnostic used labeled rewarded trials 30--49 as a guard
calibration block and scored rewarded trials 50 onward. This is a Tier-2 use of
dense calibration behavior and is not FiLM.

| Method | Seed42 | Seed43 | Seed44 | Three-seed mean |
|---|---:|---:|---:|---:|
| one shared output scale | +0.00263 | -0.00002 | +0.00208 | +0.00156 |
| per-output scale + offset (4 parameters) | +0.00347 | +0.00462 | +0.00698 | +0.00502 |
| 16 prediction-averaged K10 activity subsets | -0.03037 | -0.01520 | -0.02439 | -0.02332 |

The four-parameter output affine was positive in 16/18 seed-session cells. It
is a small but reproducible performance lead for a separate supervised
calibration route. The K10 activity-subset ensemble is decisively negative for
an M30-trained checkpoint, but that comparison is cardinality-out-of-
distribution. It therefore shows that the K30-to-K10 cardinality mismatch
dominates this particular diagnostic; it does **not** test or close the
same-cardinality identity-estimation-variance hypothesis. A matched test would
require K30 subsets drawn from a candidate pool larger than 30 (for example,
K30-of-C50) while preserving the checkpoint's trained support cardinality.

The diagnostic implementation is
`sua_exploration/scripts/audit_dandi688_cp_alternative_surfaces_v1.py`, SHA256
`c43eaf44d0e366b280da92edd15c4eda5466bea091232add5f76a24618eb6b52`.

## 7. Post-pool co-adaptation successor

The first co-adaptation implementation completed but accidentally preserved
`requires_grad=False` on the copied post-pool parameters. Its immutable root is
retained and the error is disclosed in
`INCIDENT_DANDI_000688_CP_FILM_POSTPOOL_V1_FROZEN_POSTPOOL_20260904.md`. That
run is not evidence about post-pool co-adaptation.

An additive V2 repaired the trainability contract and ran seed42 under the
same data/operator law. Each arm had exactly 13,050 trainable parameters: four
FiLM tensors and six post-pool tensors. Every final and averaged post-pool
digest differed from native, while the complete frozen native substrate digest
was identical before and after.

| Governing epoch-9--12 average | Delta vs native | Delta vs EMPTY | Positive vs EMPTY |
|---|---:|---:|---:|
| CP10@M10 | -0.00472 | +0.00122 | 3/6 |
| CP30@M30 | -0.00615 | -0.00020 | 2/6 |
| EMPTY@ZERO | -0.00594 | -- | -- |
| SHUFFLE10@M10 | -0.01086 | -0.00492 | 0/6 |

Neither real-profile arm met the pre-registered profile-utility law of positive
gain over native plus at least `+0.002` and 4/6 over EMPTY. Seeds43/44 were
therefore not run. The epoch-12 diagnostic was substantially worse (CP10
`-0.02266`, CP30 `-0.02893` against native), showing that parameter averaging
reduced late drift but did not reveal profile utility.

This closes the narrow explanation that the V1 null was caused only by a
frozen post-pool MLP. It does not test a newly represented two-state activity
pool or a new task descriptor.

V2 immutable body hashes:

- attempt: `eb10c9962309eb58795ebe3fedc7181f85ba738ad7d2ede87f30876619e77d91`;
- source authority: `04e129b3fc158209557ad61d417e521085121a1a426858e269293c46ea3dd6e8`;
- training: `2d6563cecc8d672444fa1db2f32e8311f875af5cba85652d04741fdca3706170`;
- score: `b88a92eb811cfa56543bf7ffb800db54a6dc99a1a0e533506b0421d0825fba06`;
- terminal: `28928bb0b4c82a73ebd4a1261882dafaed61e37c23a5beda80e215962b8007a4`.

## 8. Immutable CP-FiLM V1 evidence

| Seed | score.json SHA256 | terminal.json SHA256 |
|---:|---|---|
| 42 | `4df7f750695630d19d5d5faaa45da0ef03c09f6b72c48cd045547fa854ed9f4b` | `0e3ab27f33527dd268be9f80e38efa63b8bd5de97ad7932bee3dc1619725983b` |
| 43 | `3407a08083803e171430dab4785981e53e9723d7253ff2f5d52a0243f6f0809d` | `a64c89f3c1d87a65dc835ecef352d6764e9f41e32adbdff4627fe11225b07f2b` |
| 44 | `0300a1ce389c01bd8f5dcdb3242afa3e381c250e1d8402a303c7b26bbdbd5242` | `5c6e6b94241e664e2eedc0fe60a739368384a3fe55ef68e617b03a273b43169e` |

Every body and checkpoint sidecar in the three result roots passed
`sha256sum -c`. The receipts state `formal_test_files_opened=false`,
`target_session_updates=0`, and `evalai_push=false`.
