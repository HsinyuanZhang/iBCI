# HANDOFF: Next-round directions for the functional-carrier program

**Date:** 2026-08-12
**Status:** ROOT CLAIM/IMPLEMENTATION AUDIT COMPLETE; GATED EXECUTION IN PROGRESS. The original
queue below is preserved for provenance, but it is **not executable as written**. The audit
override in section -1 governs until each affected contract and real-data adapter is replaced and
re-reviewed. This document authorizes no GPU run by itself.
**Scope:** what to do next after the fusion/decoder search returned flat, and the gain over SPINT stayed small on native benchmarks.
**Relation to other docs:** does not override `HANDOFF_MAINLINE_CLOSURE_20260811.md`. It adds a diagnosis and a ranked menu. Every closed lever listed in that handoff stays closed unless this document names a new estimand.

---

## Round boundary (2026-08-13)

The validation round opened by this handoff is now closed around the terminal A2 result.  Its authoritative
endpoint is `a2_matched_subject_shift_v2/terminal_aggregate.json` (SHA-256 `5b1459df...`): the same three
source-trained T4/Z4 checkpoint pairs were scored within sub-C development and on external sub-M, all frozen
interaction gates passed, and the six formal sub-C test sessions remained sealed.  A4 and A11 are supporting
CPU diagnostics from that closed round; they do not enlarge the A2 claim.

The next experimental round covers three separate scopes: B1 carrier-by-distillation Stage P, A12 descriptive
attention auditing, and the A1 hidden-space carrier interface. Partial B1 cells are not A2 replications, and A12
is non-causal. This boundary prevents evidence, gates, and checkpoint-reuse rules from being silently carried
between rounds. C1 teacher-domain and C2 sampling experiments remain held and are not part of the active GPU
queue; terminal A1/B1 routing outcomes are recorded once in the audited execution ledger below.

---

## -1. Root audit override (2026-08-12)

The handoff and its same-evening scaffolding were independently reviewed before execution. The
review found several **estimand-level and implementation-level blockers**. Passing synthetic unit
tests is not evidence that a real experiment is runnable: the combined synthetic suite passed
`102` tests with `3` real-data tests skipped, while multiple real paths remained absent or wrong.

### -1.1 Material corrections

| Item | Audit verdict | Required correction before use |
|---|---|---|
| Section 1.2 / A2 | **Framing and gate are inverted.** The historical numbers predict a larger carrier gain on external subject M, not within subject C. The A2 contract tests the opposite sign. It also proposes independently retraining identical source arms for the two scoring domains, doubling cost and weakening matching. Its exact two-sided Wilcoxon gate on three seed-level interactions can never attain `p <= 0.05`. | Rename the factor from “unit correspondence” to **target-subject distribution shift**. Train one T4 and one Z4 source checkpoint per seed, score each unchanged checkpoint on C, M, and (if admissible) J, and test `gain_external - gain_within`. Do not claim general cross-subject invariance from one external subject. Remove mathematically unattainable seed-level p-value gates. |
| A3 / B4 | **Not correspondence breakage.** Applying the same unit permutation to neural, calibration, and carrier tensors is an exact null for a permutation-invariant shared-unit model. Consistent subsetting/pooling changes population size or signal view but preserves attachment. | Keep only as a **channel-loss / population-degradation robustness** study. A permutation arm is an implementation null, not an active augmentation or mechanism test. |
| B1 | **Current contract cannot identify the claimed interaction.** A T4-only loss sweep can show which loss trains T4 better, but not whether identity distillation specifically suppresses carrier content. Its exact two-sided Wilcoxon gate over four session means also cannot attain `p <= 0.05`. B2/B3 do not logically depend on B1 passing. | Use a carrier-content x distillation design (at minimum `{T4,Z4} x {with-E-loss,no-E-loss}`), with the interaction as primary. Treat `task_only` as a later secondary arm. Use effect-size, sign-consistency, and a cluster-aware interval; do not use impossible exact-p gates. |
| A12 | **The original scaffold was not runnable or correctly bound.** It used the wrong model/data paths, transposed session batches, mislabeled whole-identity zeroing, and took K as V. Those defects were repaired before the real audit; a later review also froze and recorded CPU batch size because BLAS partitioning can change raw float32 digests. | The bounded partial aggregate now covers seed 42, epochs 5--7 (`3/24` seed-epoch pairs; SHA `2bf287...`). Across the six development sessions, T4-minus-Z4 normalized entropy, effective attended units, and pairwise head cosine are each negative in `6/6` sessions. These are descriptive frozen-checkpoint observations only: the matrix is incomplete, no inference or causal intervention was performed, and A12 gates nothing. |
| A4 | **Original scaffold was not runnable; the blocker is resolved.** The repaired probe is checkpoint-bound and uses M30, train-only authorities, fixed epochs 5--12, and session-LOSO validation. | Three seeds are complete and positive. Use it only as evidence that AC4 preserves substantially more cross-session linearly recoverable tuning phase than Z4; it is not a theorem about nonlinear information and not a decoder-gain result. |
| A13 | **Inferential design invalid.** Window-level bootstrap/sign flips treat autocorrelated windows as independent; the runner accepts externally prepared predictions without proving they came from the named checkpoints; most session-level coverage cells cannot have two sessions; and “poorer design should yield larger carrier gain” conflicts with estimator-noise logic. | Drop the proposed 30-stratum falsification gate. If retained, first build a checkpoint-bound exporter and use trial/session-blocked resampling. Report a small descriptive decomposition only; do not use uniform gain as a falsifier of the carrier mechanism. |
| A5 | **SUA date-gap interpretation is invalid without stable unit identity.** Prefix row index is not a biological match across sorted SUA sessions, so a date-gap trend mixes arbitrary attachment with drift. | Do not run the SUA aged-carrier claim. If this question remains useful, move it to a fixed-channel native-MUA substrate with explicit channel correspondence; otherwise existing row/label-shuffle controls already answer the wrong-attachment question. |
| B9 | **Primary score is ill-defined.** `median cosine - median RMSE` mixes a dimensionless similarity with a scale-dependent error. Selecting M trials from the first 50 also preserves a 50-trial waiting time unless target order is controllable. | Separate phase cosine and held-out encoding error, normalize errors, use a disjoint reference fit, and state whether the experiment tests label count, waiting time, or pre-scheduled target geometry. Synthetic receipts do not authorize a real replay. |
| B8 | **No real gate exists.** The runner explicitly stops on real data and only generates synthetic correct/shuffled labels. Its static same-M refit is not the proposed prefix-to-future RLS deployment. | Require real frozen-decoder pseudo-labels on post-prefix trials, compare against true post-prefix labels only for retrospective audit, and measure freeze frequency/drift. Defer until that path exists. |
| A10 | **“Same labels” is underspecified.** A readout/fine-tune trained on dense per-bin velocity uses much more supervision than T4’s one direction label per trial. | Split the study into (i) an explicitly dense-label supervised upper bound and (ii) an equal-annotation sparse target if a valid loss can be defined. Do not call their difference the isolated cost of backpropagation. |
| B14 / B15 | **Structural arguments are wrong as written.** Softmax attention can produce signed output contributions through signed V and output projections. Extra latent queries cannot affect behavioural queries in a layer containing only independent cross-attention plus per-token FFN. | Remove B14. B15 requires an explicit query-mixing/self-attention mechanism and then becomes a different, higher-cost hypothesis; it is not authorized by this handoff. |

Two scope corrections also govern all presentation work:

1. The H1 oracle swap measures local sensitivity of **one frozen H1 consumer**. It is not an
   estimator ceiling and does not globally demote estimator work on SUA, M2, or RT.
2. H1 H-SE5 is not a positive cross-date sparse-label result: date 1 was `+0.028469` versus an
   independently trained Zero5, while date 2 was `-0.022590`. H1 remains the dense H-C compact-
   consumer result; no H1 sparse-label claim may be made.

### -1.2 Audited execution order

1. **A4 is complete.** Preserve the immutable three-seed result and do not expand it into a GPU
   branch.
2. **Repair A12 as a descriptive diagnostic only.** It does not gate A10 and is not a causal
   head-collapse test.
3. **Rewrite A2 as a target-subject-shift interaction.** Reuse each source-trained checkpoint
   across scoring domains; expected fresh training cost is two arms x three seeds, not twelve
   independently trained cells.
4. **Rewrite B1 as carrier-content x distillation.** B2 and estimator-noise augmentation remain
   independent hypotheses; neither is foreclosed by a B1 null.
5. **Retain the A11 baseline schedule caveat**; the completed replay does not support resumed B0 training.
6. **Redesign A10's supervision accounting** before any gradient-based upper bound is launched.

A13, SUA A5, B8, B9, B14, and B15 are held. A14 remains an independent low-priority efficiency
branch: three fresh matched cells would only reconstruct a one-seed pilot, while a three-seed
publication matrix requires nine. It is not on the critical scientific path. No GPU job may be
launched from the superseded section 5 table until the relevant rewritten contract and real-data
preflight pass independent review.

### -1.3 Execution ledger after independent code audit

| Item | Audited state | Live decision |
|---|---|---|
| A4 token-content probe | Rewritten to paired v10 AC4/Z4, M30, fixed epochs 5--12, six-session LOSO, within-training-session pairing null, mean-normalized ridge, fixed executable hyperparameters, and immutable verified receipts. Focused suite: `17 passed`. Across seeds 42/43/44, AC4 raw/null/advantage is `0.749669/0.232369/+0.517299`; Z4 is `0.319513/0.229369/+0.090144`; paired AC4-Z4 delta is `+0.430155`, with `18/18` positive seed-session signs. Multi-seed summary SHA: `6e0ead18...`. The prose protocol postdates the receipts, so it is not cited as an independent pre-registration. | **Complete positive diagnostic.** Activity-only tokens are phase-poor, not phase-free; AC4 preserves `5.74x` more null-relative cross-session linear phase signal. This is not nonlinear structural necessity or decoder gain. No further A4 GPU/training experiment is implied. |
| A11 B0 convergence | Independent authoritative CPU replay **complete**. All 36 pinned checkpoints (3 seeds x 12 epochs) were replayed on exactly six allowed development sessions with gradients disabled, unchanged model-state hashes, and no training/formal-test NWB access. Epoch-5--12 mean is `0.236417`; per-seed late-minus-early is `-0.01095/-0.01595/+0.00187`. The first stored checkpoint has three-seed mean `0.269176`, `+0.032759` above the fixed-window mean, but the curve is strongly non-monotone. Immutable full-receipt SHA: `955ebaf8...`. | **Closed: no late-curve improvement signal.** The frozen continuation flag is false (mean `-0.00834`, not `>=+0.03`, only 1/3 positive), so A11 does not support resumed B0 training. The early-checkpoint observation makes the architectural decomposition schedule-dependent, but post-hoc checkpoint selection is not admissible. |
| A2 target-subject shift | v2 is **terminal positive**.  The same source-trained checkpoint bundle was scored on sub-C development and external sub-M for each of three seeds.  Mean `T4-Z4` is `+0.248968` on sub-C and `+0.484766` on sub-M; the subject-shift interaction is `+0.235799`, with seed interactions `+0.230085/+0.187631/+0.289681` and crossed seed-by-session bootstrap interval `[+0.100852,+0.371768]`.  The four absolute means are within `Z4=+0.326008`, `T4=+0.574976`; external `Z4=-0.143399`, `T4=+0.341367`.  Immutable aggregate SHA: `5b1459df...`; the six formal sub-C test sessions remained sealed. | **Complete.**  The result supports increased *relative carrier value* under the observed C-to-M subject shift.  It is not an absolute `+0.235799` T4 lift and not a biological unit-correspondence test; the four absolute arm/domain means must accompany the interaction. |
| B1 carrier-content x distillation | Stage-P routing result | Mean interaction `+0.0137`; mixed signs; below `+0.03`: **`STOP_B1_NO_STAGE_F`**. |
| A1 hidden-space adapter | Seed-42 routing pilot | Primary interaction `+0.0128 R²`; below prereg `+0.03`: **`PILOT_ROUTING_STOP`**. Attachment control confirms use but no useful lift. |

---

## 0. Why this document exists

Two facts drive it.

1. The gain over a properly trained SPINT is small where it has been measured natively: `+0.089796` on matched M2, `+0.013447` on the H1 organizer-held set, `-0.00652` on M1.
2. A heterogeneous set of architecture/fusion interventions produced mostly small matched-control
   deltas, while two backbone-replacing interventions failed badly. They are **not** one eight-arm
   pre-registered lattice: one logit row is duplicated under two names, B15 is activity-only, and
   CI64 is slightly outside `+-0.02`.

The natural reaction is to try a ninth fusion mechanism. The evidence says not to. This document explains what the evidence actually implies, then splits the remaining work into two categories:

- **Category A — proof, validation, and presentation.** No new mechanism. Supplementary experiments on the existing design plus changes to how results are described.
- **Category B — redesign.** New network interface, new training method, new estimator.

---

## 1. Diagnosis

### 1.1 Tested fusion/interface variants have not improved their matched controls

| Intervention | Dataset | Result | Gate | Source |
|---|---|---|---|---|
| Concat carrier at `post_pool` input (selected) | all | reference | - | `streaming_encoders.py::SideFeatureEarlyPoolEncoder` |
| Confidence-FiLM on pooled `h` | SUA | `+0.003399` vs T4 | fail | `DECODER_SIDE_DESIGN_SPACE_20260809.md` |
| Multiplicative gain on live activity (L-D option 1) | RT | `G-Full - A0 = -0.002068` | `+0.03`, fail | `HANDOFF_MAINLINE_CLOSURE_20260811.md` |
| T4 carrier-biased logit residual, rank 8, 48 parameters | SUA | `-0.003142` vs T4 continuation, 3/6 sessions | fail | `results/sua_t4_factorized_logit_residual_v1/aggregate.json`; the design-space document cites this same run |
| Electrode gate `t4gate` | SUA, 3 seeds | `-0.010817` vs T4, 1/6 sessions | `ineffective` | `results/t4_gate_screen/aggregate.json` |
| Same-electrode relation `t4rel` | SUA, 3 seeds | `-0.001440` vs T4, 1/6 sessions | `ineffective` | `results/sua_electrode_relation_full_v1_scheduler/` |
| Activity-only relational cross-neuron attention B15 | M2 | `B15-B3 = -0.014218`, 0/3 cells | fail | `results/attention_arch_screen_v3/aggregate.json`; not a carrier-fusion arm |
| Same activity-only mechanism, capacity control | SUA | `B15-B15P = +0.006354`, 2/6 sessions | fail | same |
| Joint carrier interface width 32 to 64 (CI64) | H1 | `-0.020130`, 2/5 dates | fail | `HANDOFF_MAINLINE_CLOSURE_20260811.md` |
| **Decoupled cross-attention `K(E,T4), V(x)` v1** | SUA | **`-0.444658`** vs coupled, 0/6 sessions | fail | `results/sua_t4_decoupled_kv_v1/aggregate_seed42.json` |
| **Fixed slot router, K=32 soft** | SUA, 2 seeds | **`-0.177935`** vs B15P | fail | `results/fixed_slot_router_pilot_v1/aggregate.json` |

All four gates in `attention_arch_screen_v3` are `false`. On SUA the apparent `B15-B3 = +0.039770` collapses to `+0.006354` against the parameter-matched control `B15P`, so the gain was capacity, not cross-neuron communication.

**Reading — two distinct classes, not one.**

1. **Carrier-path additions that keep the backbone have not beaten their matched controls.** FiLM,
   live-activity gain, logit residual, electrode gate, relation, and CI64 are small (CI64 is
   `-0.020130`, so “inside `+-0.02`” is only approximate). B15 is not evidence in this class
   because it is an activity-only attention intervention. The results justify stopping this
   implementation family, but do not prove that concat expresses every useful carrier function.
2. **Interventions that REPLACE part of the backbone fail catastrophically.** Decoupled K/V v1 at `-0.444658` removed the pretrained `fc_in: 50->512->512` read-in and substituted a random `50->32` value projection, so it does not test key/value factorization at all. The fixed-slot router at `-0.177935` is a failed compression intervention, but its mechanism is unresolved: the ordinary-temperature run had entropy `0.856824` (about `23.61` effective slots), while the predeclared low-temperature follow-up was never launched. Do not promote "rate-distortion" or "uniform routing collapse" to a finding.

**The recurring signature, with limits.** The listed candidates compared against a
*parameter-matched but mechanism-free* control lost or tied: FiLM versus `nofilm_match`
`+0.000698`; relation versus `no_group` `+0.006496`; logit residual versus
`additive_control` `+0.003756`; L-D versus `G-XLS` `+0.006606`. Several content controls are
clearly worse, but not all by a large margin: the membership-shuffle contrast is only
`+0.016989` and was classified indeterminate. The table is a heterogeneous audit, not one
pre-registered fusion lattice, so it supports “no tested matched mechanism has improved the
selected interface” rather than the stronger claim that all content controls or all fusion
families are settled.

**Never actually run, so unmeasured rather than refuted:** residual-FiLM, the full-64-head oracle, the T4 key-residual adapter, and the electrode anchor and embedding designs. All four have green CPU suites and no numeric outcome. Decoupled K/V v2 was killed inside epoch 0 with a held-in reading of about `0.06` against `0.58` for coupled T4; that is a kill signal, never a formal comparison. The closure board forbids all of them. Do not revive one merely because the code exists.

### 1.2 Unmatched hints suggest larger carrier value under target-subject shift

| Regime | Historical comparator (not uniform) | Carrier system | Historical delta / hint |
|---|---:|---:|---|
| Within-subject cross-session (sub-C validation, B3) | `0.326479` | - | - |
| **Cross-subject (subject-M, Zero4 / T4)** | **`-0.057766`** | **`0.356828`** | **unusable to usable** |
| Native M2, same subject across days (matched SPINT) | `0.293110` | `0.382906` | `+0.089796` |
| H1, same subject across dates (organizer-held) | `0.261492` | `0.274939` | `+0.013447` |
| M1, same subject | `0.648591` | `0.644766` | `-0.003825` |

Zero4 itself is not a “zero-content floor”; it is the width-matched **activity-only identity
encoder** with standardized carrier masked to zero (`mask_standardized_t4`, arm `z4`). However,
only the subject-M row above is a T4/Zero4 comparison. The M2 row uses SPINT/B0-style comparators,
the H1 row uses organizer system submissions, and M1 uses another system-level contrast. The table
therefore does not yet show that activity-only identity works within subject and collapses across
subjects.

M2, H1, and M1 are all same-subject, same-array, across-day, whereas the large subject-M contrast
uses an external subject. Because the protocols and comparators differ, the table is a
**hypothesis-generating pattern**, not evidence that the small delta is a property of the benchmark
or that the carrier repairs unit correspondence. The audited A2 estimand is target-subject
distribution shift.

**Caveats that must be respected.**

*Protocol mismatch.* `0.326479` comes from the 20-epoch `attention_arch_screen_v3` sub-C protocol; `-0.057766` and `0.356828` come from the 12-epoch V9 subject-M protocol. These are not matched runs. The table above is a **hypothesis**. Item A2 is the experiment that would make it printable. Note the better within-subject comparator is the SUA lattice `Z4 = 0.326008`, which is literally the same arm on the sub-C M30 substrate, not `B3 = 0.326479`.

*A large slice of the SUA headline is architectural, not informational.* On the M30 substrate, `Z4 = 0.326008` versus `B0 = 0.236417`, where B0 is a trainable non-aliased copy of SPINT's original `fc_id_in/fc_id_out` with **side width zero**. Therefore:

```text
T4 - B0 = +0.338559      headline
Z4 - B0 = +0.089591      the side pathway carrying an all-zero vector
T4 - Z4 = +0.248968      actual carrier content
```

**26.5% of the headline SUA gap is delivered by a four-wide side path containing zeros.** Any
statement of the form "the carrier beats SPINT by +0.34" must be replaced by the T4-minus-Z4
figure. A11 has now replayed all 36 stored B0 checkpoints. It finds no late upward trend, but the
first stored checkpoint has a three-seed mean `0.269176`, `+0.032759` above the fixed epoch-5--12
mean `0.236417`, on a strongly non-monotone curve. Therefore the decomposition is valid under the
declared epoch rule, not an architecture-invariant quantity; post-hoc early-checkpoint selection is
not admissible. See item A11.

*The M2 delta shrinks further under matched epochs.* The official SPINT image packages an epoch-27 decoder while the T4 submission packages epoch-34. Under matched epoch-34 replay, `T4 - B0 = +0.06420` (3/4 sessions) and `T4 - TS4 = +0.09558` (4/4), against the official system-level `+0.116765`. The matched-epoch carrier gain is roughly half the headline.

*On H1, correct attachment accounts for about two thirds of a small decomposition.* `H-C =
0.525511`, `H-LS = 0.499895` (a label-rotated carrier), `H-S = 0.496833` (the
5,965,500-parameter SPINT identity MLP), and `H-C0 = 0.486616`. Thus `H-C - H-LS = +0.0256`,
while `H-LS - H-C0 = +0.0133`. H-LS still uses behavioural labels before rotating their
attachment, so the latter term must **not** be called label-free; it is nonspecific carrier-path
content under wrong attachment. The compact rotated-label model nevertheless matches H-S, which
motivates separating compression from correct functional content.

### 1.3 Why concat may be enough, and the phase-loss hypothesis

In an ideal linear cosine encoding model with an exactly balanced direction set,
`E[cos theta] = E[sin theta] = 0`, so the **raw mean response** removes the first-harmonic phase.
This motivates a phase-loss hypothesis for an activity-only identity that averages unlabeled
trials. It is not a structural theorem about SPINT: the network applies a learned nonlinear
per-trial projection before averaging, the finite direction set is only approximately balanced,
and temporal response structure can preserve phase-correlated statistics. A4 therefore measures
cross-session linear recoverability instead of assuming phase is absent.

This hypothesis is compatible with several results above, but it does not establish that phase is
the only missing information or that concat is universally sufficient.

**Three existing measurements motivate it, but they are not one matched proof.** They mix M2
baseline redundancy, an H1 non-T4 carrier residual, and an H1 token-space angle; none was collected
to establish center-out phase loss.

1. **The baseline coordinate is redundant, while carrier content is only partly linearly recoverable.** Mean firing rate versus T4's `b`: Pearson `r = 0.9960089736` on M2 native, residual R-squared `0.002697` (`results/n4_cpu_precheck_v1/audit.json`). The superseding same-pipeline H1 calibration gives carrier residual R-squared `0.8048187892`, so about `19.5%` is linearly explained by the pooled activity probe (`SPINT-main/src/data/h1_overlap_gate_decoder_calibration_receipts/h1_overlap_gate_decoder_calibration_root_review_v2.json`). This is a cross-dataset motivation, not proof about T4 phase.
2. **Returning the redundant coordinate is actively harmful.** `B4 = 0.287273` sits `0.038735` **below** the zero vector `Z4 = 0.326008`. Feeding the encoder a scalar it already holds consumes side-path capacity and delivers nothing. The CPU screen predicted this sign in advance from the `0.001` residual.
3. **The two paths write to near-orthogonal directions.** In the H1 identity-token audit the median angle between the activity-driven and carrier-driven components of `E_i` is `83.483` degrees, with the carrier component at about one third the amplitude (RMS ratio `0.332486`).

The joint pattern is consistent with redundant baseline-rate information and complementary carrier
content, but it does not prove that `[a,c]` is absent from a nonlinear activity token. A4 is the
direct empirical check.

**Direct token measurement.** A4 now reads tuning phase out of the learned identity token under
six-session LOSO. Across seeds 42/43/44, AC4 raw/null/advantage is
`0.749669/0.232369/+0.517299`, while Z4 is `0.319513/0.229369/+0.090144`; the paired
AC4-Z4 delta is `+0.430155` and all `18/18` seed-session signs are positive. Activity-only
pooling is therefore phase-poor, not phase-free, while AC4 preserves `5.74x` more null-relative
linear phase signal. This does not establish decoder gain or nonlinear absence in Z4.

**One caution.** The overlap-residual statistic was explicitly calibrated against decoder gain and **failed**: Spearman `rho = -0.5`, exact `p = 1.0` over the three H1 arms. Treat `0.804819` as a fact about linear recoverability under that calibrated probe, not as evidence of usefulness.

### 1.4 Unmatched diagnostics suggest possible headroom; matched headroom is unmeasured

From `P3_CROSS_SESSION_ANALYSIS.md`:

| Quantity | Value |
|---|---:|
| Single-session end-to-end ceiling (B3, 80/20 chronological within one session) | `0.6937` |
| POYO single-session CO reference (13M parameters, 5-10 ms bins) | `~0.935` |
| Labelled encoder-only finetune oracle, K=20 calibration trials | `+0.692` |
| Same oracle, K=10 / K=5 | `+0.614` / `+0.408` |
| Zero-shot in the same setting | `-0.122` |
| Best gradient-free cross-session with carrier (subject-M) | `0.356828` |

**These numbers are not comparable to the mainline and must never be quoted as if they were.** The oracle crossed a 4x unit-count regime jump (train `~60` units, test `~245` units), used held-out behaviour labels and backward gradients, and used a different split from V9.

Even so, they are the only existing hint about the cost of the no-backprop constraint, and the hint is that the
cost may be large. A10 v1 does not supply the missing matched measurement: its gradient-free arm uses one
direction annotation per trial while its adaptation arms consume behaviour labels for weight updates, and its
non-gradient-free entry point is intentionally inert. This blocks claims about the isolated cost of
backpropagation and target-time adaptation branches; it does **not** upper-bound a new source-trained interface
that remains gradient-free at deployment.

### 1.5 One frozen H1 consumer is locally insensitive to a query-fitted leakage perturbation

`pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUERY_ORACLE_LEAKAGE_DIAGNOSTIC_v1.json`:

| Condition on the frozen H-C consumer | Pooled R-squared |
|---|---:|
| Honest support-fitted carrier | `0.5255107931` |
| **Query-fitted oracle carrier** | **`0.5226522069`** |
| Oracle minus support | **`-0.0028585862`** |

The swap genuinely changed the carrier — relative Frobenius delta `0.827` / `1.051`, row-cosine mean `0.537` / `0.359` — and the consumer barely moved. The receipt records `consumer_local_insensitivity_supported: true` and `consumer_response_locally_attenuated: true`, with prediction-displacement RMS at only `0.0606` of the target-centred RMS.

This is explicitly a `LEAKAGE_DIAGNOSTIC_ONLY` perturbation, not a strict oracle ceiling. About
`4716/8965` evaluated outputs (`52.6%`) overlap trials used to fit the query-local carrier, the
replacement was not established to be a better estimator, and the sealed consumer was trained on
a different carrier distribution. The result supports **local insensitivity of this one frozen H1
consumer to this perturbation**; it does not prove carrier-estimation saturation.

In this same fold/seed receipt, same-checkpoint zeroing gives `0.103871`, versus `0.038895` for the
separately trained H-C0 contrast, a `2.67x` ratio. That shows same-checkpoint zeroing exaggerates
the carrier effect in this receipt; the factor must not be generalized.

This has a direct consequence for the menu below, and it is the main reason this document's priorities differ from an intuitive reading:

- **On this frozen H1 consumer only, carrier-estimator improvements need matched retraining before
  they can be judged.** The oracle swap does not globally demote D-optimal design, temporal
  kernels, population covariance, or shrinkage on SUA/M2/RT. Separately, the SUA M15 W3 arm was
  ineffective: `T4W3@15 - ordinary T4@15 = +0.000753`, even though its train-only proxy improved
  27/27 sessions. The ordinary low-budget deficit `T4@15 - T4@50 = -0.058842` is not itself a
  shrinkage contrast.
- **Consumer-side and estimator-side ideas remain separate hypotheses.** This diagnostic alone
  neither promotes every consumer redesign nor demotes estimator work.

The receipt names its own required follow-up: train a source consumer from scratch on the oracle-carrier distribution, then evaluate with a matched query-local carrier. It also states `estimator_saturation_proven: false`. Until that runs, "carrier estimation is not the bottleneck" holds only for a model that was never given a chance to use better carriers.

---

## 2. Category A — proof, validation, and presentation

No new mechanism. Ordered by value per unit of effort.

### A1. Rename Zero4 to "activity-only identity"

**What.** Change the description, not the experiment. State that the Zero4 arm is architecturally identical to the carrier arm with the standardized carrier masked to zero.
**Why.** It prevents a zero-vector control from being confused with a zero-capacity model. The
subject-M score is still protocol-specific and is not, by itself, a general statement that the
baseline fails.
**Cost.** Writing only.
**Rests on.** `unit_side_features.py::mask_standardized_t4`, arm `z4` returns `zeros_like`.
**Kill criterion.** None. This is a factual correction of an under-claim.

### A2. Matched target-subject-shift × carrier-content experiment

**What.** Train `{activity-only, carrier}` once per seed on sub-C, then score each unchanged
checkpoint on sub-C development sessions and external subject M. Primary statistic is the
external-minus-within interaction in `T4-Z4` gain.
**Why.** It replaces the mismatched historical hints with one target-subject-shift estimand. It
does not test stable unit-index correspondence, which neither B3S nor T4 assumes.
**Cost.** `2 arms × 3 seeds = 6` fresh source-training cells, plus two score domains per checkpoint.
**Kill criterion.** If the interaction gate in `A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md`
fails, drop the subject-shift narrative; make no correspondence claim.

### A3. Robustness-under-unit-availability changes (held pending redesign)

The original consistent-permutation arm is an exact null for a permutation-invariant model when
activity and carrier rows move together. Cross-date row-prefix matching also does not establish
biological unit correspondence. No A3 experiment is authorized until it is rewritten around a
well-defined deployment perturbation such as missing-unit robustness, with no sealed-session use.

### A4. Cross-session phase-content probe

**What.** Train a linear probe to recover the direction of `[a_i, c_i]` from learned identity token
`E_i`, separately for paired Z4 and AC4 checkpoints, using six-session LOSO and a within-training-
session pairing null.
**Why.** It directly tests whether phase is linearly recoverable from the learned token across
sessions. It cannot prove nonlinear absence or structural necessity.
**Cost.** Forward-only on sealed checkpoints, CPU.
**Result.** Complete across seeds 42/43/44: AC4 raw/null/advantage
`0.749669/0.232369/+0.517299`, Z4 `0.319513/0.229369/+0.090144`, paired delta
`+0.430155`, and `18/18` positive seed-session deltas. Activity-only is phase-poor rather than
phase-free; this supports differential linear recoverability only. The prose protocol was
finalized after the receipts and is not presented as independent pre-registration.

### A5. Transferred/aged carrier control — held as non-identifiable

**Original proposal.** Score session B with session A's per-unit carrier and sweep the date gap.
**Why it is invalid now.** The sessions do not provide a verified biological unit correspondence,
and B3S/T4 itself is permutation invariant. Attaching carrier row `i` from A to activity row `i`
from B would manufacture an arbitrary row-prefix/electrode correspondence; any degradation would
mix carrier aging with wrong attachment. This is the same defect that closes the original A3
cross-date proposal.
**Disposition.** Do not run A5. Delete or soften any paper sentence that claims an experimentally
measured drift advantage. A future stale-carrier experiment requires an independently validated
unit-tracking map or a correspondence-free estimand; neither exists here.

### A6. Within-pipeline applicability diagnostic

**What.** Within one fixed carrier/decoder/comparator protocol, relate paired session-level carrier
gain across calibration budgets to train-support design rank, condition number, and directional
balance. Fit any predictive rule only on source/development cells and compare it by held-out-session
cross-validation with a simpler budget-only (`M`) baseline.
**Why the old cross-task rule is invalid.** SUA, pseudo-MUA, M2, RT, H1, and M1 use different
carriers, outputs, supervision densities, consumers, and comparators. Task identity jointly changes
both the proposed predictor and measured gain, so a six-task correlation cannot identify an
applicability law or turn M1 into confirmatory evidence.
**Cost.** CPU after an exact matched-cell inventory. `design_rank` and `design_condition` already
exist for the cosine-fit path; non-cosine carriers require separately defined statistics.
**Kill criterion.** If the within-pipeline rule does not outperform the budget-only baseline under
held-out-session evaluation, publish no predictive rule. Cross-task plots may remain descriptive
only and cannot set deployment thresholds.

### A7. Use the free assets already identified but unused

Organizer-measured latency `0.113919` versus `0.129075` for paper-LR SPINT; the dense H1 H-C organizer-held result; compact-consumer compression; and M1 as a volunteered negative boundary. H-SE5 is a sparse-label boundary rather than a positive cross-date result, and CI64 only rules out that tested widening configuration, not consumer capacity in general. Writing only. See `PAPER_FRAMING_ANALYSIS_20260812.md` part 2.

### A8. Present the label-budget curve as empirical sample complexity

**What.** `M10 = 0.304264`, `M15 = 0.338115`, `M20 = 0.351767`, `M30 = 0.358154`,
`M50 = 0.356828`. Present the matched budget curve with session dispersion/sign counts and the
fixed evaluation boundary.
**Why.** It supports empirical saturation near M20--M30 under that protocol. Do not attribute the
shape to design conditioning unless A6 independently predicts held-out sessions better than a
budget-only baseline.
**Cost.** Writing; A6 is optional mechanism evidence, not a prerequisite for reporting the curve.

### A9. Publish the tested fusion slice as a historical ablation family

**What.** Report the section 1.1 table as a historical map of the tested fusion slice, keeping the
two failure classes separate: matched additive pathways over a kept backbone are flat, while
backbone-replacing variants fail under different documented confounds. Report the never-run items
of section 4b as never-run; do not let them read as nulls or as covered design space.
**Why.** It converts spent GPU into an honest coverage map: the matched additive variants tested on
their selected interfaces were flat, while two backbone replacements failed under different,
documented confounds. This narrows the observed design space but does **not** prove that the entire
fusion space is flat or that descriptor content is the unique causal factor.
**Cost.** Writing only.
**Risk.** A reviewer may read it as a fishing expedition. Report it as a heterogeneous historical
ablation family with matched contrasts where they exist; do not call the collection one
pre-registered lattice or use it to pre-empt untested alternatives.

### A10. Design a matched audit of the no-backprop constraint

**Question.** How much target-session performance is forgone by forbidding weight updates?
**Current design blocker.** The existing proposal is not matched. T4 consumes one direction target
per calibration trial, whereas ordinary decoder fine-tuning consumes dense neural--velocity pairs.
Changing update rule, label density, and optimization target together cannot identify the cost of
backpropagation. “Same calibration prefix” is not “same supervision.”
**Required redesign before any GPU.** Freeze a supervision ledger listing every target value visible
to each arm, its temporal multiplicity/weighting, trainable parameters, optimizer steps, saved state,
and calibration/query windows. The primary contrast must hold the target observations and objective
fixed while changing only whether a closed-form/readout update or backward gradient is used. If no
scientifically meaningful sparse-label gradient objective can be made identical to the carrier arm,
report dense-label fine-tuning only as an explicitly label-richer upper bound, not as the cost of the
no-backprop constraint.
**Status and cost.** Design-only and **GPU-blocked**. The unmatched section 1.4 oracle cannot set a
numeric gate or imply `0.3 R2` headroom. A10 does not currently gate B1 or establish consumer
saturation; any future gate must be frozen after the supervision contract is auditable.

### A11. Convergence diagnostic for B0

**What.** Replay all 36 stored B0 checkpoints and test whether the late part of the fixed 12-epoch
curve was still improving. Continuation beyond epoch 12 is not part of this diagnostic.
**Why.** `Z4 - B0 = +0.089591` means a quarter of the headline SUA gap is the side pathway rather
than the carrier. The stored curve can test whether a late upward trend motivates any new training.
**Cost.** Forward-only replay; no new training.
**Authoritative result.** The immutable CPU full replay (`SHA-256 955ebaf8...`) covers all 36
pinned checkpoints and gives epoch-5--12 mean `0.236417` and late-minus-early
`-0.01095/-0.01595/+0.00187` for seeds 42/43/44. The aggregate late-minus-early is `-0.008342`;
the frozen continuation flag is false, so A11 is closed with no evidence-based reason to extend
B0 training. The maximum CPU-smoke/full-replay numerical difference is below `3e-8` and all
checkpoint, query, normalizer, model-state, and scope bindings passed.

### A12. Descriptive attention-path audit

**What.** On the canonical SUA v10 M30 paired T4/Z4 checkpoints, intercept the exact Q/K/V tensors
and attention weights from the frozen decoder. Report per-head normalized entropy, effective support,
cross-window/covariate variability, and paired T4-Z4 changes. Bind every dump to checkpoint,
run metadata, normalizer, support/query policy, session roster, and arm. Do not substitute whole-token
zeroing for a carrier intervention.
**Why.** The consumer uses a single cross-attention layer with 64 heads and two output-query tokens
on SUA. A checkpoint-bound dump can show whether explicit carrier content measurably changes the
attention geometry and whether many heads are descriptively redundant. It cannot by itself identify
why accuracy saturates: high or low entropy is not causal head importance, and no pruning or
intervention is included.
**Cost.** Forward-only CPU after a metadata-only preflight and independent root review.
**Status.** v4 preflight SHA `ce9fcfc3...` binds `B_cpu=512`. A bounded partial aggregate now covers
seed 42, epochs 5--7 (`3/24` planned seed-epoch pairs), with immutable aggregate SHA
`2bf287307595e5b58e10b3ea00ff082de9873d50421ccd67c9b099764d95eb30`. Across the six
development sessions, mean T4-minus-Z4 deltas are `-0.041562` for normalized entropy, `-2.612026`
effective attended units, and `-0.035522` pairwise head cosine; all three deltas are negative in
`6/6` sessions. The remaining `21/24` pairs are absent.
**Interpretation.** No pass/fail gate and no causal “head collapse” claim. A12 does not gate A10,
does not authorize a decoder redesign, and should not be generalized to M2/H1 until an exact
dataset-specific checkpoint path is separately bound. The partial aggregate is a descriptive
frozen-checkpoint observation, not an inferential or causal result.

### A13. Error-structure decomposition — held pending provenance/statistical redesign

**Original question.** Does paired T4-Z4 gain vary with movement direction, speed, magnitude, or
calibration coverage?
**Why the current scaffold is invalid.** It consumes external prediction JSON without proving that
the predictions came from the named checkpoint, query windows, normalizer, or arm. Its bootstrap
and sign-flip treat correlated windows as IID. Calibration coverage is a session/support-level
quantity, while speed/magnitude/direction are trial/window-level quantities; pooling them into one
stratification gate creates pseudoreplication and level mixing. The H1 recording spread and A2b
ridge instability come from different carriers/protocols and cannot be “explained” by this SUA
analysis.
**Required redesign.** First build a checkpoint-bound exporter that writes paired T4/Z4 predictions,
targets, trial IDs, session IDs, window hashes, normalizer, support policy, and arm provenance from
the same forward pass. Freeze strata before export. Estimate within-session contrasts with trial or
contiguous-block resampling, then aggregate at session level; coverage effects require session-level
replication and cannot use windows as sample size.
**Disposition.** Held; no current numeric gate and no mechanism claim. A future valid result would
be descriptive heterogeneity unless the interaction and resampling hierarchy were independently
predeclared.

### A14. Finish the B3TStream + T4 efficiency branch

**What.** `results/sua_b3t_t4_efficiency_v1/` has `t4_s42 = 0.585301` and `b3t_t4_s42 = 0.597073`, a delta of `+0.011773` — inside the `-0.03` non-inferiority margin — with a recorded 30.79% parameter reduction, 65.29% session-MAC reduction, and 88% transient-state reduction. The required same-seed `B3TStream + TS4` content control was never run and no aggregate exists.
**Why.** A favourable efficiency result that was cancelled as queue position 5 by the 2026-07-31 narrowing, not because it failed.
**Cost — corrected 2026-08-13.** A read-only inspection found **zero checkpoints**: only
`t4_s42.json`, `b3t_t4_s42.json`, and three logs, one of which is `b3t_ts4_s42.log`; the control
was killed before a receipt. Without the original checkpoints, the historical `0.585301/0.597073`
numbers cannot be paired with a new control. Reconstructing seed 42 is **three fresh cells**, but
that is only a pilot. A paper-level three-seed matrix is **nine fresh cells**. The preflight must
fail closed until checkpoints are restored or the intended complete matrix is explicitly frozen.
**Kill criterion.** At pilot scope, stop if B3TStream+T4 does not beat its TS4 content control or
falls outside the frozen non-inferiority margin to T4. A positive one-seed pilot authorizes at most
the remaining six cells; it is not itself a publication claim.

---

## 3. Category B — redesign

### 3a. Training method (never tried; highest expected value in this category)

### B1. Remove or retarget the identity-distillation term

**What.** On the M2 and streaming line the default is `loss_mode = task_plus_y_plus_E` with
`lambda_y = 1.0` and `lambda_E = 0.1`. The `lambda_E` term pulls the student's identity token
toward an **activity-only teacher**. A4 shows that this activity-only representation is phase-poor,
not phase-free, while AC4 retains substantially more null-relative linear phase signal. The
remaining hypothesis is therefore narrower: identity distillation may selectively suppress the
extra carrier-specific content. B1 must identify that carrier-content x distillation interaction.
**Why it matters.** The SUA line already defaults to `task_only` (`train_variant_dandi688.py`, default `task_only`), and SUA is where the carrier looks strongest. M2 uses `task_plus_y_plus_E`, and M2 is where the delta is small. That co-occurrence has never been tested as a cause.
**Experiment.** A primary `2 x 2` factorial on M2:

```text
carrier content:       T4 versus Z4
identity distillation: task_plus_y versus task_plus_y_plus_E
```

Hold `lambda_y`, architecture, seed, split, calibration/query policy, epoch rule, and all other
training choices fixed. The primary estimand is the difference-in-differences
`(T4-Z4)_without_E - (T4-Z4)_with_E`. A T4-only three-loss sweep cannot identify this interaction
and is withdrawn. `task_only` may be reported only in a separately frozen secondary experiment.
**Cost and staged scope.** The current M2 `loso_fold=0` contract has exactly one validation session.
Therefore `4 arms x 3 seeds = 12` fresh GPU cells are a **single-session Stage-P routing pilot**,
not a cross-session result. Before Stage P, freeze folds 1--3 as the only possible expansion. If and
only if the practical interaction threshold and 3/3 seed-sign gate pass, Stage F adds `3 folds x 4
arms x 3 seeds = 36` fresh cells. **Only Stage F is the confirmatory primary:** its terminal rule is
applied to the three fresh predeclared LOSO sessions × three seeds. P+F is allowed only as a labeled
descriptive sensitivity; it can never receive the terminal rule. A full 4-session claim cannot be
made from the 12-cell pilot.
**Kill criterion.** Freeze the practical interaction threshold, the Stage-P seed rule, and the
**Stage-F-only** terminal session/seed rule before launch. If Stage P fails, stop; if the confirmatory
Stage F does not
increase `T4-Z4` under the terminal rule, drop the selective-suppression hypothesis. Do not rescue it
with a T4 main effect, a post-hoc loss mode, or a different fold.
**Provenance, now checked (2026-08-12).** The original R1 loss-mode selection was made on a **carrier-free** student. The Gate-2 R1 ablation configs (`b3_d64_anchor`, `b3_d64_task_only`, `b3_d64_task_plus_y`) all resolve to `streaming_b3`, i.e. variant **B3 with `side_dim = 0`**, and `gate2_matrix.py::evaluate_r1` / `choose_winning_loss` select from exactly those rows. No receipt records a loss-mode selection on a B3S or T4 carrier-present student.

Two consequences. First, the M2 carrier arm inherited `task_plus_y_plus_E` from a decision taken
**without a carrier present**, so the interaction between `lambda_E` and carrier content has never
been tested. Second, the matched `2 x 2` factorial above is a **new estimand**, not a replay of a
sealed decision. It requires its own source-only selection and cannot inherit a gate from the
carrier-free R1 sweep.

### B2. Session-consistent carrier-reliance corruption

**What.** Keep the architecture and deployment path unchanged. During source training, assign each
source session one carrier state per logical epoch using an exact twelve-epoch schedule
`T4 x 6 / RS4 x 3 / Z4 x 3`. All windows from one session share that state throughout the epoch;
RS4 uses one fixed complete row derangement per run seed/session. This tests whether the consumer can
fall back toward activity-derived identity when carrier content is wrong or absent, without erasing
the benefit of correct T4.
**Why.** Historical P3 never trained a model: its sampler was not wired into the datamodule, and its
only receipt replayed old RS4/LS4/Z4 numbers. Activity-path dropout is the opposite intervention—it
forces greater carrier dependence—and is removed from this B2 estimand. Per-window random carrier
noise is also rejected because T4 is a session identity, not a rapidly changing input.
**Primary estimand.** Compare clean versus corruption-trained consumers under matched forward views
`{T4,Z4,RS4,LS4}`. The difference-in-differences measures whether RS4/LS4 move toward Z4; a common
regularization lift cancels. Mandatory anti-triviality gates require corruption-trained `T4-Z4 >=
+0.03` and clean-T4 deployment non-inferiority.
**Staging and status.** CPU contract v2 is complete (`31` focused tests) but no runtime hook or GPU
result exists. Stage P is `2 arms x 3 folds x seed42 = 6` cells. Only after its frozen mechanism,
content-retention, and deployment gates pass may Stage F add seeds 43/44 (`12` cells). Stage F alone
is confirmatory; P+F is descriptive only. The mechanism is logically distinct from B1, but runtime
integration waits for B1 because B1's selected loss defines the B2 substrate and B1's streaming tree
is already hash-locked.
**Kill criterion.** Stop after Stage P if wrong-content penalties do not move toward Z4, if correct
T4 content disappears, or if clean-T4 deployment degrades beyond `-0.03`. No probability, M, loss,
fold, or epoch rescue is allowed.

### B3. Estimator-noise augmentation

**What.** During source training, add noise to the carrier drawn from the analytically known small-M estimator covariance `sigma^2 (X'X)^-1`, which `tuning_fit_confidence_descriptor` already computes.
**Why.** It trains the consumer to be robust to exactly the estimation error it will meet at deployment. It targets the `M10 = 0.304264` to `M50 = 0.356828` gap directly.
**Cost.** Small code change, one sweep over noise scale.
**Kill criterion.** No recovery of the low-M budget points.
**Note.** This is distinct from the failed `t4c` / confidence-FiLM route. That route fed uncertainty to the model as an extra input. This one uses uncertainty to shape the training distribution and adds no input width. It remains a lower-priority held hypothesis, not a queued GPU experiment: any eventual test must retrain the consumer and include a matched T4/Z4 sibling interaction. The frozen H1 query-oracle diagnostic does not authorize it.

### B4. Population-degradation augmentation — held pending a non-null contract

**What.** A future source-training robustness experiment may use fixed-severity unit dropout,
unit subsetting, or pseudo-electrode pooling, paired with the same explicitly defined degradation at
evaluation. A simultaneous permutation of activity, calibration, and carrier rows is excluded: for
the shared-unit permutation-invariant model it is an exact implementation null, not an augmentation.
**Why.** The defensible target is robustness to fewer or coarser population observations, not repair
of biological unit correspondence. Subsetting and pooling preserve activity-carrier attachment while
changing signal availability; they therefore cannot isolate correspondence.
**Status and cost.** Held. Pseudo-MUA pooling exists as an evaluation view, but no matched A3
degradation curve, frozen severity grid, T4/Z4 anti-triviality control, or source-only selection rule
currently exists. Creating those would be a moderate new experiment, not free A2 checkpoint reuse.
**Kill criterion.** Before any launch, freeze a clean-performance non-inferiority margin and a
population-degradation endpoint. Stop if augmentation does not improve the latter or harms clean T4
beyond the frozen margin. Do not interpret a generic clean-score lift as correspondence evidence.

### B5. Episodic, deployment-matched objective

**What.** Optimize post-calibration performance on a held-out query split of a simulated session, instead of per-batch MSE with `random_calibration: true`.
**Why.** The current objective is only a weak approximation of the deployment metric. An explicitly episodic (calibration prefix, then query) objective optimizes what is actually scored.
**Cost.** Training-loop rewrite. Highest engineering cost in section 3a.
**Kill criterion.** Run only after B1-B3, and only if at least one of them moves.

### B6. Multi-subject and multi-dataset joint source training

**What.** Train one consumer on sub-C plus M2 plus RT with the carrier as the shared interface.
**Why.** If the carrier is a subject-invariant coordinate system, joint training should improve every domain. Proposed as "option B" in `ARCHITECTURE_ANALYSIS.md` and never run.
**Cost.** Data plumbing plus one large training run.
**Kill criterion.** Any domain degrades beyond the frozen margin.

### B7. Use the unsupervised output bins

**What.** `decode_last_timestep_only = True` means the decoder emits `W` bins and only the last one is supervised — 49 of 50 output bins carry no gradient. Add an auxiliary loss over the remaining bins or a carrier-reconstruction head on `E`.
**Cost.** Small.
**Priority.** Low. Listed for completeness.

### 3b. Estimator and carrier content

> **Scope-limited by section 1.5, not globally downgraded.** A query-fitted oracle carrier moves
> one frozen H1 consumer by `-0.002859`, but that receipt is a leakage diagnostic, not a strict
> upper bound, and the consumer was not trained on the query-carrier distribution. Standalone H1
> estimator swaps are therefore low-value; SUA/M2/RT estimator work must be judged on its own
> matched substrate. The relevant W3 warning is `T4W3@15 - ordinary T4@15 = +0.000753`; the
> `-0.058842` number is the ordinary M15-to-M50 budget gap, not the shrinkage effect.
>
> **Therefore no item in 3b should be launched as a standalone accuracy arm.** Each must be paired with a consumer retrained on the improved carrier distribution, which is the follow-up the oracle receipt itself names. B8 is the exception: it changes *when* the carrier is estimated, not how accurate it is, so it is not subject to this argument.

### B8. Self-consistent carrier via recursive least squares

**What.** Fit the carrier from M labelled trials as today. Then, on subsequent **unlabelled** trials, use the decoder's own output as a pseudo-target and update `[a, c, b]` by recursive least squares with a forgetting factor. Closed form, `O(1)` state per unit, no backward pass, no optimizer.
**Why.** It converts the carrier from a static descriptor into an adaptive state. It pays for labels once per session instead of once per session per recalibration, and it addresses chronic drift, which is the actual clinical problem. It directly answers the standing objection that calibration is not label-free.
**Why the N4 failure does not close it.** N4 was a label-free **static** descriptor (rate, Fano, autocorrelation, population coupling) and scored `+0.001588` with 3/6 sessions positive. The RLS carrier inherits task alignment from the labelled prefix; it is a different mechanism.
**Cheap CPU gate before any GPU.** Using a sealed checkpoint, refit the carrier from decoder-output pseudo-labels and measure `cos([a,c]_pseudo, [a,c]_true)` against a shuffle baseline. This mirrors the RT split-half constructibility gate (`0.787119`), so the tooling pattern already exists.
**Strongest objection.** Self-training amplifies its own error. **Response.** The update touches three parameters per unit and no network weight; add a forgetting factor and a fail-closed rule that freezes the carrier if it departs from the initial fit by more than a fixed threshold. A5 supplies the no-update decay baseline that RLS must flatten.

### B9. D-optimal calibration design

**What.** Choose which target directions to instruct during calibration to minimize the carrier's covariance, instead of taking the chronological first M rewarded trials.
**Why.** The estimator is OLS with a known design matrix, so the design is controllable. The `M10 = 0.304264` deficit may be mostly a conditioning problem rather than a data-quantity problem, because chronological-first-M gives random direction coverage.
**Cost.** CPU, retrospective. Recompute the budget curve with D-optimal selection over the same trials.
**Payoff if it works.** "Eight labelled trials suffice" is a much sharper claim than "fifty".
**Kill criterion.** No recovery of M50 performance at low M.
**Current correctness block.** The scaffold privately copies the canonical-direction and cosine-fit
estimator instead of importing the mainline implementation, and no equivalence test binds the two.
B9 remains held until those primitives are extracted into one framework-free shared module or a
parameterized randomized test proves bit-level agreement. No B9 number produced before this gate
is admissible.

### B10. Temporal-kernel carrier

**What.** Replace the single source-selected latency `tau` with a per-unit temporal kernel: `r_i(t) = sum_tau w_i(tau) y(t - tau)`. The carrier becomes `[N, d_y x n_lags]`.
**Why.** It is a strict generalization of the paper's own general form, still closed form, and it is the only content extension that unambiguously adds information rather than rearranging it. It is also the natural fit for H1 and RT, where one latency is unlikely to serve every phase of the movement.
**Cost.** CPU constructibility gate first, then one GPU arm.
**Risk.** Widening the descriptor worsens conditioning at `M = 30-50`. Pair it with B9 and with the shrinkage machinery that already exists (`uncertainty_wiener_shrink_t4`).

### B11. Population carrier: add a label-free covariance summary

**What.** The per-unit carrier supplies `W` and `b` but discards the noise covariance `Sigma`. Optimal linear decoding needs both. Add a compact, closed-form summary of `Sigma` (for example its top-k eigenbasis) as a session-level side input.
**Why.** `Sigma` needs **no labels**. This is the only route that adds information without adding supervision, which keeps the supervision-density claim intact. The H1 estimator already does project, regress, back-project, shrink, so the precedent exists.
**Cost.** CPU constructibility gate, then one GPU arm.

### B12. Procrustes alignment of the carrier population

**What.** Align the target session's carrier population to a source-defined canonical frame by closed-form SVD.
**Why.** Across subjects or arrays, the whole tuning distribution may be rotated. Per-column z-scoring with source statistics does not fix a rotation. Electrode and spatial priors have been tried; carrier-population Procrustes has not.
**Cost.** CPU, cheap.
**Link.** If A2 shows a cross-subject interaction, this is the first thing to try to widen it.

### 3c. Network interface and target-time adaptation (separate boundaries)

**Boundary for the 2026-08-13 hidden-space adapter proposal.** A1 is a source-training intervention; target
deployment still performs only the ordinary analytic T4 fit and a forward pass. It is therefore **not blocked
by A10**, whose valid scope is target-session weight-update headroom. Its reviewed implementation is additive,
with the adapter held in a separate path from the sealed shared backbone.

The attributable H-add path retains the same activity-only identity rather than silently deleting it:

`h_i = fc_in(x_i + E_i^A) + P(T4_i)`,

where `E_i^A` is produced by the matched Z4/activity route, `P` is bias-free, and `P(0)=0`. H-add/Z4 is an exact
structural alias of W-add/Z4. The minimal logical 2x2 is sealed W/Z4 and W/T4, aliased H/Z4, and one freshly
trained H/T4 cell. The uncorrected bare expression
`fc_in(x_i)+P(T4_i)` is an identity-route replacement and cannot isolate the add site. The contract must also
freeze zero-initialization parity, unchanged teacher/decoder state, identical support/query provenance,
parameter/MAC/state accounting, and an attainable synthetic pass/fail test. Its primary estimand removes
generic adapter gain:

`(H-add(T4)-H-add(Z4)) - (W-add(T4)-W-add(Z4))`.

Existing A2 W-add checkpoints may be reused only after exact source roster, normalizer, M30, loss, schedule,
epoch-bundle, teacher, query, and implementation bindings pass. Those checks and the optimizer/parity proofs
passed under preflight SHA `812d426ef73464ce4f6cb933856ad39035ffbe95c95114e38fc8596f125cd7b1`.
The terminal A1 routing outcome is recorded in the execution ledger above.

### B13. Closed-form last-layer adaptation

**What.** Instead of adapting only four numbers per unit in feature space, adapt **parameters** in closed form: ridge-solve the final linear layer of `psi` (or the decoder readout) on the calibration prefix.
**Why.** It is still backprop-free and optimizer-free in exactly the sense the paper already claims, but it adapts thousands of parameters instead of `4N` inputs. Linear probing is known to capture much of what fine-tuning buys, and section 1.4 suggests fine-tuning buys a lot. **This is the largest untried capacity increase that respects the deployment constraint.**
**Cost.** Moderate. Needs a careful statement of what closed-form parameter adaptation costs on device (one ridge solve, plus storing the solved layer).
**Kill criterion.** Run only if A10 shows meaningful headroom.

### B14. Signed readout / linear attention

> **Withdrawn by the root audit.** Non-negative attention probabilities do not imply non-negative
> unit contributions because both the value and output projections are signed. The structural
> premise is false as written; this item is not executable and no experiment is queued.

### B15. Latent query tokens

> **Withdrawn by the root audit.** In the current decoder, cross-attention and the FFN operate on
> each query token independently. Unscored latent queries cannot influence behavioural queries
> without an additional query-mixing/self-attention path, which would be a different hypothesis.
> No experiment is queued.

### B16. Carrier-driven rank-1 modification of the read-in

**What.** Let the carrier produce a per-unit rank-1 modification of the effective `fc_in` row for that unit.
**Why.** Additive `x_i + E_i` and multiplicative `x_i (1 + g)` can shift and scale a unit's contribution but structurally cannot **rotate** its direction in output space. This is the mechanism the classical encoding model implies and the only one not yet expressible.
**Kill criterion.** If implemented as target-session parameter adaptation or justified by adaptation
headroom, it requires a repaired A10-style supervision contract. A distinct source-trained forward-only
variant would require its own T4/Z4 factorial and is not automatically A10-blocked. Highest risk item in this
document.

---

## 4. Do not repeat

Two different statuses here. Keep them distinct.

### 4a. Measured and refuted

- Any further fusion mechanism that adds a carrier pathway to the existing backbone (section 1.1, class 1).
- Confidence or posterior information as an extra **input** to the consumer: `t4c`, confidence-FiLM `+0.003399`.
- Waveform and SNR side features: `F1 - F0 = -0.0324`, `F2 - F0 = -0.0012`, both indeterminate against their dimension-matched controls.
- Static electrode gate `-0.010817` and same-electrode relation `-0.001440`, both `ineffective`.
- Widening the identity interface: CI64 `-0.020130` is terminal; H64 is prohibited.
- Label-free **static** descriptors of the N4 family: `N4 - NS4 = +0.001588`, 3/6 sessions.
- Fixed-K temporal prototypes: Gate A2 lost to a simpler order-invariant marginal baseline, `P20 - B20 = -0.041963`, 0/4 sessions.
- Wiener shrinkage at low label budget: `T4W3@15 - ordinary T4@15 = +0.000753`, effectively
  zero despite a train-only proxy that improved 27/27 sessions. Ordinary `T4@15 - T4@50 =
  -0.058842` records the budget deficit, not the shrinkage effect. **Treat W3 as the warning that a
  train-only proxy need not transfer.**
- Second-order or gain-field carrier terms: never run, but they add estimator parameters and worsen conditioning at `M = 30-50`, and section 1.5 says a frozen consumer would not spend the improvement anyway.

### 4b. Never measured, but forbidden by the closure board

These have implemented code and green CPU suites and **no numeric outcome at all**. They are unmeasured, not refuted. `ACTIVE_EXPERIMENT_CONTROL_BOARD.md` section 7 forbids them. Do not revive one merely because the code exists.

- Residual-FiLM (`B3SCFR` / `B3SCFRS` / `B3SCFRA`) — cancelled at queue position 4.
- The full-64-head oracle — never launched; its results directory holds only a watcher log.
- The T4 key-residual adapter — no runner was ever written. Note the closely related logit-residual variant *was* run and failed at `-0.003142`, and the two are described in the design notes as equivalent lower-state forms of the same idea.
- Electrode anchor (`t4anchor`) and electrode embedding (`t4e`) — never scheduled after design D reported `ineffective`.
- Decoupled K/V v2 — killed inside epoch 0; the `~0.06` versus `~0.58` reading is a kill signal, never a formal comparison.

### 4c. Unresolved, not failed — and worth finishing

`B3TStream + T4` (item A14) has favourable historical seed-42 receipts but no retained checkpoints,
so the apparent missing-control-only completion is unavailable. It is a low-priority three-cell
reconstruction pilot or nine-cell three-seed matrix, not the cheapest unclaimed result.

---

## 5. Recommended order

Root-audited order. The old broad queue is withdrawn: several proposed diagnostics are exact
nulls, under-specified, synthetic-only, or statistically incapable of passing their stated gate.

| Step | Items | Cost | Blocks what |
|---|---|---|---|
| Done | A4 under the v10 M30 AC4/Z4, fixed epoch-5-to-12, session-LOSO protocol | CPU/forward-only | Positive linear token-content result; no automatic follow-up |
| Done | A11: authoritative CPU replay of all stored B0 checkpoints | CPU/forward-only | No late continuation signal; continuation false; schedule caveat added to SUA decomposition |
| Done | A2 v2 target-subject-shift interaction | 6 fresh source-training GPU cells | Terminal positive interaction `+0.235799`; report absolute external T4/Z4 means as well |
| Descriptive partial | A12 v4 attention audit | CPU/forward-only; `3/24` pairs, aggregate SHA `2bf287...`; three attention-geometry deltas are negative in 6/6 sessions | No causal saturation claim and no gate |
| 4 | Preserve B2 v2 as a separate training-side contract | CPU contract complete; B1 ended without selecting a substrate; 6-cell routing + 12-cell confirmation | No automatic launch; whether training can reduce blind reliance without discarding correct T4 |
| 5 | Redesign A10 with explicit supervision accounting | design first | Any matched no-backprop-cost statement |
| 6 | A14 only as an independent efficiency branch | 3-cell pilot; 9 cells for three-seed evidence | B3TStream efficiency hypothesis |

**Held:** A13, SUA A5, B8, B9, B14, and B15. A3 requires a new non-null robustness
intervention. B3/B4 and downstream decoder redesigns are not queued by this handoff; B2 is
contracted but operationally downstream of B1. Negative
or weak experiments are closed concisely rather than expanded into new branches.

**Current strategy boundary.** A1 and B1 terminal routing outcomes are recorded in the execution ledger above.
C1 `{MC-Maze,CO-native} x {T4,Z4}` with W-add fixed, C2 `{legacy,equal-session} x {T4,Z4}`
sampling, and CF1 activity-path dropout remain held candidates; C1 and C2 are not automatically promoted.
CF1 uses
`{p=0,p>0} x {T4,Z4}`.  C2 initially means ordinary session-balanced MSE, not an R2-native loss and
not the later B5 episodic calibration-prefix/query objective.  Only the
`streaming_calibration_exp` M2 sampler currently implements `balance_sessions`; the ordinary M2
config leaves it off and the switch applies only to the source-training sampler, not validation sampling;
fixed per-session window budget is not exposed through that DataModule, and
the SUA/A2 and legacy SPINT samplers do not provide the same switch.  CF1 asks the network to rely
more on T4 when activity identity is unreliable; B2 carrier corruption asks it to fall back to
activity when the carrier is wrong.  They are opposite interventions and must not be described as
substitutes.  Each held candidate needs its own CPU contract and source-only selection before any
GPU launch.

---

## 6. Honest statement of what is hypothesis

To keep this document usable as evidence later, the status of each claim:

| Claim | Status |
|---|---|
| Tested matched add-on mechanisms have not improved the selected interfaces | **Measured with heterogeneous protocols.** Keep only the matched contrasts in section 1.1; B15 is activity-only and the duplicate logit row is one run. |
| Two backbone-replacing variants fail catastrophically | **Measured effect, unresolved cause.** Decoupled K/V v1 `-0.444658`, fixed-slot router `-0.177935`; both have documented confounds. |
| Zero4 is the activity-only identity encoder | **Verified in code.** `mask_standardized_t4`, arm `z4`. |
| A quarter of the SUA headline is the side pathway, not the carrier | **Measured.** `Z4 - B0 = +0.089591`; `T4 - Z4 = +0.248968`. |
| B0 showed a late upward trend at epoch 12 | **Not supported by the authoritative A11 CPU full replay.** Late-minus-early is negative on two seeds and `+0.00187` on the third; mean `-0.00834`; the frozen continuation flag is false. |
| T4's `b` is redundant with activity | **Measured on M2.** `r = 0.996`. A4 separately shows that AC4 tokens contain substantially more cross-session linearly recoverable `[a,c]` phase than Z4; the H1 overlap residual `0.804819` is not a gain predictor. |
| AC4 preserves more linearly recoverable tuning phase than Z4 | **Measured by A4.** Null-relative advantage `+0.517299` versus `+0.090144` (`5.74x`), raw delta `+0.430155`, `18/18` positive. Z4 is phase-poor, not phase-free; this does not prove decoder benefit. |
| Relative carrier value increases under the observed C-to-M subject shift | **Measured by A2 v2.** Interaction `+0.235799`, three positive seed interactions, crossed seed-by-session interval `[+0.100852,+0.371768]`. Absolute external `T4=+0.341367`, `Z4=-0.143399`; therefore the interaction is not an absolute T4 lift and is not a correspondence test. |
| One frozen H1 consumer is locally insensitive to a query-fitted leakage perturbation | **Measured locally.** Delta `-0.002859`; not a strict ceiling, with fit/evaluation overlap and distribution mismatch; `estimator_saturation_proven: false`. |
| Same-checkpoint carrier zeroing overstates the carrier | **Measured on one H1 fold/seed receipt.** `0.103871` zeroed versus `0.038895` retrained, a factor of `2.67`; do not universalize. |
| Headroom above the gradient-free result is large | **Unmatched hint only.** Regime-confounded, label-using oracle. A10 measures it. |
| The `lambda_E` distillation term caps the carrier on M2 | **Not established; the B1 routing outcome is recorded in section -1.3.** |
| Head concentration explains the local insensitivity | **Not established.** A12's `3/24` descriptive partial aggregate has three 6/6-negative geometry deltas, but no intervention, inference, or causal gate. |
| Hidden-space carrier addition improves content use | **Routing pilot stopped below the preregistered gate; see section -1.3.** |
| Current MC-Maze teacher creates a domain-mismatch penalty | **Plausible but not isolated.** C1 must vary teacher domain while fixing the add site and include T4/Z4 siblings. |
| Equal-session sampling improves the carrier interaction | **Untested.** Only part of the M2 sampler infrastructure exists; C2 must not be conflated with R2-native or episodic training. |
| Softmax non-negativity limits signed population readout | **Withdrawn.** Signed value/output projections invalidate the premise. |
| The carrier's gain lands on the strata theory predicts | **Untested and current A13 scaffold invalid.** It lacks checkpoint/query provenance and cluster-aware resampling; no mechanism claim is available. |

Nothing in this document authorizes a GPU run, reopens a closed lever, or changes any sealed result.

## 7. Revision note

This document was revised on 2026-08-12 after three independent read-only audits of the repository. The revision added sections 1.5, A11 to A14, and 4a/4b/4c; corrected the section 1.1 characterization from "eight flat nulls" to two distinct failure classes; corrected the section 1.2 headline arithmetic for the `Z4 - B0` side-pathway confound and the M2 epoch mismatch; changed A4 reporting from raw cosine alone to matched-null-relative evidence and recorded the protocol-timestamp limitation; added the B9 estimator-equivalence blocker; and reordered section 5 so that consumer diagnostics precede carrier improvements. The Chinese translation `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812_zh.md` tracks this revision.

**2026-08-13 addendum.** Added the terminal A2 three-seed interaction and its four absolute means; recorded
the A12 `3/24` descriptive partial aggregate; and summarized the terminal A1/B1 routing outcomes in the
execution ledger above. C1/C2 remain held.
Separately, the cold archive at `/mnt/data/SPINT_cold_archive/2026-08-13/` now contains the initial `8.90 GiB`
plus a sealed-batch source-unique `81.9967 GiB` migration; hard-link deduplication added `74.5878 GiB` of new
archive storage. The `49.940 GiB` `streaming_calibration_exp/logs/train` tree was subsequently archived after
path-compatibility smoke tests, with its original absolute path retained as a symlink. Root free space is
approximately `208 GB`; pointer-plus-SHA manifests preserve recovery without changing authoritative artifacts.
