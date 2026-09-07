# Identity-token content probe protocol (cross-session phase content, item A4)

**Dated:** 2026-08-12 (Asia/Hong_Kong)  
**Status:** root-audited v2 forward-only CPU probe; three-seed replication complete. The
executable content gate passed, but this prose file was finalized after the first and multi-seed
receipts and is therefore **not an independent pre-registration record**. Authorizes no GPU run
and no network training. The v1
within-session unit-fold estimator and its M50 hard-code are superseded by section 0.  
**Relation:** implements HANDOFF item **A4** and formalizes the section **1.3** argument in
`HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md`. It does not reopen any closed fusion lever.

## 0. Root-audit v2 override

The first scaffold mixed a checkpoint trained with a 30-trial side-feature pool with a hard-coded
50-trial probe target, fitted behavior normalization on validation sessions, and fit a separate
unit-level map inside every target session. Those choices do not test whether the token uses a
common, cross-session phase representation.

The executable v2 contract is therefore:

1. Use the paired component-attribution v10 **AC4 and Z4 checkpoints of the same seed and epoch
   rule**. `run_metadata.json` is authoritative for `pool_size` (expected `30`), train/validation
   roster, descriptor contract, and normalizer SHA. Both arms must match these fields.
2. Construct both identity input and raw carrier target from the same checkpoint-authoritative
   `M=30` support. No validation-session behavior/side normalizer may be fitted.
3. Use **session-LOSO probing** across the six development validation sessions. In each fold,
   fit feature standardization and ridge weights using units from five sessions, then score the
   sixth session. A separate within-session unit-fold result may be emitted only as a secondary
   diagnostic.
4. The primary estimand is held-out-session phase recovery:

   ```text
   delta_phase(session) = mean_cosine_phase(AC4) - mean_cosine_phase(Z4)
   ```

5. The pairing-permutation null is fit by permuting carrier-target rows **within each training
   session** before fitting the global probe; the held-out session remains untouched. Null seeds
   and row digests must match across AC4/Z4.
6. `b` and `m` probes are descriptive. They cannot veto an otherwise valid phase comparison.
7. Ridge and null settings are frozen: standardized-feature ridge `lambda = 1.0`, base null
   seed `42`, modulation exclusion `m <= 1e-6`, and at least `15` units in every held-out
   session. The command exposes no scientific hyperparameter override; the runner and aggregate
   both reject a receipt that differs from these exact values.

The executable v2 content rule recorded in the immutable receipts requires mean
`delta_phase >= 0.10`, at least `5/6` held-out sessions positive, and AC4's mean advantage over
its pairing-permutation null `>= 0.10`. These thresholds were embedded in the executable before
the receipts, but the later prose file must not be cited as proof of independent pre-registration.
No exact p-value is claimed from six sessions.

This probe can show that AC4 training preserves a linearly recoverable, cross-session phase
coordinate that Z4 lacks. It cannot prove that every nonlinear activity-only representation is
mathematically phase-blind, and it cannot by itself explain decoder R-squared.

## 1. Hypothesis and predictions

### 1.1 Structural hypothesis

A center-out calibration block has an approximately direction-balanced trial set, so
`E[cos theta] = E[sin theta] = 0`. SPINT forms the per-unit identity token `E_i` by
**averaging** per-trial projections over `M` calibration trials, and an individual calibration
trial carries no direction label. This motivates the hypothesis that trial-averaged activity
pooling attenuates tuning **phase**, while preserving baseline rate `b_i` and possibly modulation
depth `m_i`. It is not a structural theorem because the learned per-trial map is nonlinear and
finite direction sets are only approximately balanced.

Existing supporting evidence (not re-derived here): mean firing rate vs T4 `b` Pearson
`r = 0.9960089736` with residual `R^2 = 0.002697`; production carrier regressed on pooled
activity has superseding same-pipeline residual `R^2 = 0.8048187892`; `B4 = 0.287273` is below `Z4 = 0.326008`;
median angle between activity-driven and carrier-driven components of `E_i` is `83.483` degrees.

### 1.2 Probe predictions (per arm, before any run)

| Target | Definition | Activity-only arm (`z4`) | Carrier arm (`t4`) |
|---|---|---|---|
| `b_i` | baseline rate from cosine fit | **HIGH** probe `R^2` | **HIGH** probe `R^2` |
| `m_i` | modulation depth `m_i = hypot(a_i,c_i)` | **INTERMEDIATE** probe `R^2` | **INTERMEDIATE** probe `R^2` |
| phase pair `[a_i,c_i]/m_i` | unit direction in tuning plane | **NEAR CHANCE** mean cosine | **HIGH** mean cosine |
| `[a_i,c_i]` unnormalized | raw directional pair | intermediate `R^2` (reported, not gated) | intermediate `R^2` (reported, not gated) |

**HIGH** and **INTERMEDIATE** are descriptive predictions. The only inferential read rule is the
frozen paired phase comparison in section 4. **NEAR CHANCE** is interpreted relative to the
matched within-training-session target-pairing permutation baseline.

## 2. Probe definition

### 2.1 Inputs (per session, per arm)

- Learned identity token `E_i in R^{window_size}` for every unit `i`, shape `[N, 50]`, computed
  by a **forward-only** call to the frozen identity encoder on the first `M = 30` chronological
  rewarded calibration trials. The activity support is exactly `[M, T, N] = [30, 100, N]` before
  its batch dimension is added. No backward pass, no optimizer, no weight update.
- Ground-truth carrier `[a_i, c_i, m_i, b_i]` for the same units from the same session's first
  `30` rewarded calibration-pool trials, using the closed-form cosine tuning fit in
  `unit_side_features.py::_fit_cosine_tuning` (raw Hz units, **before** train-only z-scoring).

### 2.2 Estimator

For each scalar or vector target, fit one **session-LOSO global ridge probe** per held-out
development session: concatenate units from the other five sessions, fit feature
standardization and an unpenalized-intercept ridge only on those training units, and score the
untouched sixth session. The fixed penalty is `lambda ||W||^2` with `lambda = 1.0` on
standardized features. Equivalently, its normal equations use `X'X / n + lambda I` and
`X'Y / n` (with the intercept unpenalized), so duplicating every training unit leaves the fitted
probe unchanged.

Phase is evaluated as a **circular** quantity:

- target: unit vector `[a_i, c_i] / m_i` (units with `m_i <= 1e-6` excluded);
- primary score: mean cosine between held-out-session predicted and true unit vectors;
- secondary score: held-out-session `R^2` on the two phase components (reported for completeness
  only).

### 2.3 Mandatory null baseline (same estimator, fixed seed)

1. **Within-training-session target-pairing permutation:** for every held-out session, permute
   carrier target rows independently within each of the five training sessions while keeping
   identity-token rows fixed. The held-out session remains untouched. Permutation indices, source
   row digests, and fixed base seed `42` are bound into the receipt and must match AC4/Z4.

A reported carrier-arm phase advantage is not admissible unless it exceeds this matched null by
the pre-declared amount in section 4.

### 2.4 Minimum unit count (fail-closed)

Let `N` be the number of units in any held-out development session. Refuse to fit when

```text
N < 15
```

No silent downgrade to a smaller unit threshold, a hold-out fraction, or an optimistic
within-session split score.

## 3. Scope, checkpoints, and sealed sessions

### 3.1 Dataset and sessions in scope

- DANDI 000688, subject C, center-out sorted SUA.
- **Development / validation sessions only:** the six non-sealed `val` rows of the strict split
  manifest (`MANIFEST_SHA256 = 4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`).
- Activity calibration budget: `M = 30` rewarded trials `[0, 30)`.
- T4 carrier support pool: the same `M = 30` rewarded trials `[0, 30)`.
- `window_size = 50`, `trial_length = 100`, bin size `20 ms` (SUA M30 substrate).

### 3.2 Arms (paired per session)

Run exactly two matched arms on the **same** session list and probe hyperparameters:

1. **Carrier arm (`ac4`):** B3S identity encoder with standardized T4 side input.
2. **Activity-only arm (`z4`):** width-matched B3S encoder with standardized side masked to zero
   (`mask_standardized_t4`, arm `z4`).

Checkpoints are sealed artifacts named on the command line. Each receipt records the checkpoint
SHA-256, arm label, and `sealed_test_sessions_opened: false`.

### 3.3 Sealed sessions (must never be opened)

The following six formal-test sessions are **out of scope** and must be rejected at ingest:

```text
sub-C_ses-CO-20151113
sub-C_ses-CO-20151116
sub-C_ses-CO-20151117
sub-C_ses-CO-20151119
sub-C_ses-CO-20151120
sub-C_ses-CO-20151201
```

Any receipt whose `sessions` list intersects this set is invalid.

### 3.4 Explicit non-authorizations

This protocol does **not** authorize:

- any GPU run (all runners force `torch.device("cpu")` and `CUDA_VISIBLE_DEVICES=""`);
- any training, fine-tuning, or checkpoint mutation;
- opening sealed formal-test NWBs;
- changing frozen probe hyperparameters;
- reporting decoder `R^2` as evidence for the structural claim (the probe is about token content,
  not downstream gain).

## 4. Recorded content gate and post-result interpretation

The immutable receipts preserve the executable thresholds and results. This section corrects the
interpretation after an independent review; it does not alter or regenerate those receipts.

### 4.1 Executable paired content gate (primary estimand)

The scientifically usable part of `cross_session_phase_content_gate_pass = true` requires:

1. mean held-out-session `AC4 - Z4` phase cosine is `>= 0.10`;
2. at least `5/6` held-out-session phase-cosine deltas are positive;
3. AC4's pooled phase-cosine advantage over its matched pairing-permutation null is `>= 0.10`;

The historical aggregate evaluated these criteria only after rejecting incomplete matrices, mismatched
permutations, support shapes, or frozen hyperparameter drift. `b`, `m`, and unnormalized
`[a,c]` scores remain descriptive and cannot veto this paired phase comparison.

The receipts also contain a legacy fourth conjunct and the alias
`phase_blindness_gate_pass`. They must not be used as a falsification result: that conjunct made
Z4's positive null-relative advantage consequential only when `AC4-Z4 < 0.03`, so it could not
falsify phase-freeness in the same setting where AC4 carried a useful differential signal.

### 4.2 Provenance correction and direct falsification of phase-freeness

The protocol file timestamp (`22:10:26`) postdates the seed-42 aggregate (`22:02:29`) and the
multi-seed summary (`22:07:11`). An independent reviewer reports that an earlier draft used an
absolute raw-cosine rule (`Z4 <= 0.25` near chance, `Z4 > 0.25` falsifying), but no immutable copy
of that draft exists in the repository; it therefore cannot be presented as an auditable frozen
rule. Seed 42's raw Z4 cosine `0.312414` would have crossed that reported threshold.

The matched-null result answers the scientific question without relying on either post-result
rule. Across three seeds, Z4's raw/null/advantage values are
`0.319513/0.229369/+0.090144`, whereas AC4's are
`0.749669/0.232369/+0.517299`. Thus activity-only pooling is **phase-poor, not phase-free**;
AC4's null-relative advantage is `5.74x` larger. The valid positive result is differential
cross-session linear recoverability, not phase blindness.

## 5. Required artifacts

Per arm:

- `probe_identity_token_content.py` JSON receipt with schema version, arm identity, checkpoint
  SHA-256, session list, per-session and pooled probe scores, the matched pairing-permutation
  null, exact `[M,T,N]` input shapes, `N` per session, frozen probe hyperparameters, resolved
  random seed, and `sealed_test_sessions_opened: false`. The receipt is created with exclusive
  create semantics, set read-only (`0444`), and accompanied by an exclusive-create full-file
  SHA-256 sidecar. Its canonical body SHA is embedded in the JSON.

Paired:

- `aggregate_identity_token_content_probe.py` output with the recorded paired gate verdict and
  explicit rejection reasons for any incomplete, sealed, hyperparameter-mismatched, or
  integrity-unbound input matrix. It recomputes and verifies a present canonical body hash and/or
  full-file SHA sidecar before it evaluates a gate.

## 6. What this probe can and cannot establish

**Can establish:** AC4 preserves substantially more cross-session linearly recoverable tuning
phase than its paired Z4 arm under the stated calibration budget. Against matched pairing nulls,
the three-seed advantages are `+0.517299` for AC4 and `+0.090144` for Z4. The result must be
paraphrased as “phase-poor, not phase-free,” not “activity contains no phase.”

**Cannot establish:** that phase information is useless to the decoder, that no nonlinear probe
would recover phase from the activity-only token, or that the small native decoder deltas are
explained by this mechanism alone. The superseding overlap-residual statistic `0.804819` remains a fact about
linear recoverability, not usefulness (HANDOFF section 1.3 caution).

## 7. Three-seed result

The repaired v2r2 seed-42 receipt and independently generated aggregate are immutable and their
sidecars verify:

```text
receipt SHA-256  = eee7a9514b16659018ad35aead94e4d00d4b156bcc2625ecc17164dc28545118
aggregate SHA-256 = 063f98de27c9dff4d097f89d052942e04ab4e6fc92653c305047866b6395ba16
```

Seed 42 passed all frozen gates: AC4 phase cosine `0.740706`, Z4 `0.312414`, paired delta
`+0.428292`, AC4-minus-pairing-null `+0.505071`, and all `6/6` held-out sessions had positive
AC4-Z4 deltas. The predeclared seed-43/44 replication also passed every gate. Across all three
seeds, mean AC4/Z4 phase cosine was `0.749669/0.319513`, mean paired delta was `+0.430155`, mean
AC4-minus-pairing-null was `+0.517299`, and all `18/18` seed-session deltas were positive.

The immutable multi-seed summary is
`results/a4_identity_token_content_v2/a4_v2r2_multiseed_summary.json`, SHA-256
`6e0ead189ba7deb8aaf42cb5cc2a345e1ecd77f48c5d0fc7af39e878bcaa3e96`. This supports a
cross-session linearly recoverable token-content claim only; it does not authorize a decoder
redesign or a claim of nonlinear structural necessity.
