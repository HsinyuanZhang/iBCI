> **SUBORDINATE — not authoritative on priority.** This file was written in parallel with
> `HANDOFF_FOUR_LANE_BRAINSTORM_SYNTHESIS_20260813.md` and `HANDOFF_NEXT_CONTRIBUTION_STRATEGY_20260812.md`, and it
> re-derived a ranking those two already contained. **Those two are authoritative**; the closure ledger is
> FOUR_LANE §3 and its dated appendix §3.A, and the execution queue is STRATEGY §5 with FOUR_LANE §5.3. Retain this
> file only for its derivations — the label-density decomposition in §1.2, the linear-decoder pricing in §1.3, and
> the measurement write-ups. Do not read its ranking sections as instructions.

# Structural proposals and the corrected claim

**Date:** 2026-08-13, revision 6
**Status:** two brainstorm rounds (six agents) then two adversarial audits and two ideation agents across two model
families. Every number below re-verified by me against code, receipts or raw NWBs. **Authorizes nothing.**

**Premise.** Primary goal is structural innovation that raises accuracy. Deployment stays BP-free (no optimizer, no
loss, no backward on the target session). **Target labels stay trial-level** — discrete direction plus task-event
timestamps; the per-bin `cursor_vel`/`cursor_pos`/`cursor_acc` streams are inadmissible **in any role, including as
a mask**. Ablations and controls are preconditions, never deliverables. Paper space is limited.

---

## 1. The central claim was wrong, and the correct one is already measured

### 1.1 Retract: "carrier value equals correspondence-problem severity"

That claim rested on a four-dataset ordering (H1 `+0.0389` → M2 `+0.089796` → RT `+0.268905` → SUA-external
`+0.414594`) and cannot be identified, for three reasons.

**It is perfectly collinear with a rival cause that this same document asserts is real.** The ordering is monotone in
`1/num_covariates`: M1 `C=16`, H1 `C=7`, M2/RT/SUA `C=2`. So "correspondence severity" and "the 4-D descriptor is
nearly sufficient for a 2-D directional estimand and insufficient for a high-dimensional one" order these four
points identically. No ranking over them separates the two.

**M1 is not a low-correspondence datapoint; it is a descriptor–estimand mismatch.** M1's behaviour is **EMG** —
`acquisition["preprocessed_emg"]`, verified in `audit_m1_t4_mechanism.py:409` as `"behavior_source":
"preprocessed_emg via load_nwb (16 channels)"` — while M1's carrier is a cosine fit to target azimuth. The arm asks a
per-unit reach-direction tuning vector to predict 16 muscle activations. A reviewer with the FALCON datasheet
demolishes this in one sentence.

**Two of the three supports do not support it.** The A4 phase probe (`+0.430155`, 18/18) is measured on **SUA**, the
high-severity end, so it is a constant across the gradient rather than a discriminator. And `H-LS − H-S = +0.003062`
is below this document's own declarability floor, and compares a 4-wide analytic input against a 5.97M-parameter MLP
— a statement about the MLP, not about label semantics. **I asserted in conversation that "what H1 consumes is not
the label-response association"; that is withdrawn** — the five-date `H-C − H-LS` CI is `[−0.000686, +0.069720]` and
includes zero.

**Cheap disconfounder, no new supervision:** rescore M1 or H1 on a 2-dimensional subset of its own covariates and
re-measure the carrier delta. If M1's delta stays ≈0 at `C=2`, the correspondence account survives a real test for
the first time; if it jumps, the claim is dead, and better dead now than in review.

### 1.2 The correct claim is supervision efficiency, and it is already measured with large effects

`sua_exploration/results/trial_level_ridge_v1/priority_a_label_density_receipt.json` holds features, calibration
trials, query windows and regularization fixed and changes **only the target**:

| endpoint | dense-label ridge | ridge held to trial-level direction labels | T4 | T4 − sparse | T4 − dense |
|---|---:|---:|---:|---:|---:|
| SUA M50 | `0.417922` | **`−0.122027`** | `0.356828` | **`+0.478855`** (15/15) | `−0.061093` (3/15) |
| SUA M30 | — | **`−0.208580`** | `0.358154` | ≈ `+0.567` | — |
| pseudo-MUA M30 | — | — | — | `+0.468830` (14/15) | `−0.036153` (4/15) |
| pseudo-MUA M15 | — | **`−0.424391`** | `0.288234` | **`+0.712625`** (15/15) | **`+0.159127`** (7/15) |

`dense − sparse` is `+0.539948` (SUA M50, 15/15, bootstrap `[+0.4299, +0.6584]`) and `+0.553498` (pseudo-MUA M15,
14/15). **Held to the same supervision the classical baseline is negative**, and at pseudo-MUA M15 the method beats
even the dense-label ridge.

So the abstract should say: *a frozen cross-session decoder calibrated from roughly 750 trial-level direction
scalars — no optimizer, no loss, no backward pass on the new session, and no per-bin kinematics in any role — comes
within `0.061` R² (SUA) and `0.104` (pseudo-MUA) of a per-session closed-form ridge trained on `199.633x` more target
supervision, and beats that same ridge by `+0.47` to `+0.71` R² (14–15 of 15 sessions) when the ridge is held to the
same trial-level supervision.* That turns the ridge from a defeat into the axis of the claim and the `199.633x` from
a defensive footnote into the headline. Correspondence severity becomes one scoped mechanism sentence with its
confound named, not the lead.

### 1.3 Where the ridge gap actually lives — and it is not population geometry

`sua_exploration/results/linear_decoder_control.json`, same SUA protocol, 6 validation sessions:

| decoder | R² | Gram inverse | per-unit temporal taps |
|---|---:|---|---|
| `population_vector` (cosine tuning, T4's math) | `0.0870177` | no | no |
| `ridge_pooled_rate` (N-dim pooled window rate) | `0.0887223` | **yes** | no |
| `ridge_raw_window` (flattened 50×N) | `0.3077721` | yes | **yes** |
| F0 (SPINT, activity only) | `0.3139872` | — | — |
| T4 (SPINT + carrier) | `0.5667486` | — | — |

**Population/Gram whitening is worth `+0.0017046`.** Inverting the cross-unit Gram, with dense labels, buys nothing
over the tuning-based population vector. **Per-unit temporal resolution is worth `+0.2190498`** — the entire
classical gain is one unit getting its own 50-tap filter instead of one scalar gain on a summed window. Also
`gap_best_minus_F0 = −0.006215`: the best linear decoder is slightly *below* activity-only SPINT.

This kills the population-statistic family, including the whitened decoding row. Its motivating statistic
(`corr(‖w‖,‖d‖) = 0.076 / −0.111 / −0.185`) is real and its claim that a per-unit path cannot construct
`(WᵀΣ⁻¹W)⁻¹` is correct — but the quantity it cannot construct is priced at `+0.0017`. Two uncorrelated things need
not differ in value. Relatedly, `self_attn`, `TransformerEncoder`, `encoder_layer` and `unit_self` occur **zero**
times in the repository, so there is no unit-to-unit path at any capacity; that restriction is not what binds.

---

## 2. What died this round, with the reason

| item | why it died |
|---|---|
| **RW1, response-window alignment** | Dead by two independent routes. (a) I reproduced whole-trial versus go-cue-anchored pooling on `sub-C_ses-CO-20131003`: `corr(a) = 0.9764`, `corr(c) = 0.9432`, `corr(m) = 0.9745`, z-scored `[a,c]` per-unit cosine median `0.9887`. Median `m_go/m_whole = 2.2993`, so the dilution is a nearly **uniform** scale factor, and the pipeline's per-column z-scoring removes uniform scale. (b) The rebuilt case — `target_on_time` shows 59.4% of the window precedes target onset, giving ~`1.7x` SNR ≈ `3x` effective trials — routes into a quantity already measured at zero: M30 − M50 is `+0.001326`. |
| **DR1, un-shared readout row** | The freeze argument holds only in the FALCON/M2 YAML world: `_streaming_base.yaml` defaults `freeze_decoder: true` (145 of 172 YAML configs) and `freeze_decoder()` iterates `self.decoder.parameters()`, which includes `fc_out` — but the **SUA** path is argparse, `--freeze_decoder` is `store_true`, and `train_variant_dandi688.py` callers do not pass it, so `fc_out` is trainable there. DR1 is a no-op on the FALCON side only. It still dies on the rest: "row 49" is wrong on both intended hosts (`window_size` is `700` on H1, `100` on M1); the quoted `−49,152` MAC is the `model_dim=512, C=2, W=50` configuration. The dose-response is measured **inverted**: covariate participation ratio M1 `3.84–3.93`, H1 `2.93–3.06`, M2 `1.82–1.91` against **64 heads**, and per-covariate independent dimensions run M2 ≈`0.95` > H1 ≈`0.43` > M1 ≈`0.24`. And my §4 claim that DR1 would convert M1's `−0.00652` to positive is a non-sequitur: that is a carrier-versus-SPINT contrast and DR1 raises both arms. |
| **TV1, supply estimator variation** | Premise weak (M30 ≈ M50) and decidable far more cheaply — see §3.3. Its motivating `−0.0588423` is a comparison of two separately-trained checkpoints from different experiment lineages (`t4_m15` fresh; `t4_m50` borrowed from `sua_t4_confidence_film_v1`), not a target-time budget measurement. The target-time curve is M15 `−0.018713`, M30 `+0.001326`. |
| **DQ1 / DQ2, iterated re-query** | Its own row conceded "almost nothing" if flat, DQ2 exists only to interpret DQ1, so item 4 cost two runs. Under "flat arms are not written up" that is the definition of a bad buy. |
| **Whitened decoding row / any population statistic** | Priced at `+0.0017046` (§1.3). |
| **The `+0.013` band as a claim** | Not a truncation artifact from below — the gates are **upper** thresholds and this document itself lists a dozen measured effects underneath (`+0.00969`, `+0.00557`, `+0.006354`, `+0.003979`, `+0.002987`, `−0.002068`, `−0.003142`, `−0.004432`, `−0.014218`, `−0.020130`, …), centred near zero with spread ≈`±0.02`. The four "members" are that distribution's **upper tail**, selected on outcome. They also have different gates and incommensurable precision: a 13/13-positive `+0.0161408` is a sign test at `p = 2⁻¹²`; a 2/3-positive `+0.013748` with spread `0.081` is unresolved. Grouping them merges an established small effect with a null. |
| **The seed-power inference** | Overreach. B1 is a four-cell interaction on **one** validation session; interaction variance is about twice the simple contrast's, so `σ_cell ≈ 0.021`, and from `n=3` the 95% interval for `σ` is roughly `[0.022, 0.26]`. The defensible claim is only that B1's interaction was underpowered for its `+0.03` gate. For reference, `linear_decoder_control.json` gives real seed sigmas on this protocol: T4 `0.010665`, F0 `0.023207`. |
| **A1, hidden-space fusion** | Closed at `+0.012833`, 4/6, `PILOT_ROUTING_STOP`. A valid test — `optimizer_tensor_count: 40` versus 39 for the A2 W cells, and the same-checkpoint attachment control at `+0.5793`. Note the airtight proof is optimizer coverage plus TS4, **not** my "otherwise H/T4 would equal H/Z4" argument, since H/T4 is a fresh run and H/Z4 is a sealed alias. |
| **B1, loss composition** | Closed as a high-variance null: `+0.059153 / +0.004307 / −0.022215`, `STOP_B1_NO_STAGE_F`. |

---

## 2b. The six-candidate round, and its closure

Root raised the objection that a paper whose only change is a small encoder modification is too thin, and asked
whether the decoder, training-method and data-processing axes were really empty. Six candidates were assembled and
audited by two independent agents (prior-art/cost, and mechanism/effect-size). **All six close.**

| # | candidate | outcome |
|---|---|---|
| 1 | Encoder–decoder co-design (unfreeze factorial) | **Closed by a receipt I had not used as one.** `checkpoints/a2_matched_subject_shift_v2_source_t4_dandi688_co_s42/run_metadata.json` records `training/freeze_decoder = False`, `side_features/group = t4`, `variant = B3S`. `--freeze_decoder` is `store_true` and A2's launcher never passes it, so **the SUA mainline already jointly trains encoder and decoder with T4 present**. My claim that "the decoder was never trained to consume an analytic descriptor" is false on SUA; the 145/172 freeze count is the FALCON YAML world and I conflated the two. What is genuinely unrun is decoder-from-random-init (every unfrozen run still loads the teacher state dict first), and that would forfeit the transfer property rather than test co-design. On M2, joint T4 versus joint Zero4 is already `+0.018389`, inside the floor. Filling the frozen half of `{frozen, unfrozen} x {T4, Z4}` on SUA is an ablation. |
| 2 | **T3K, direction x time tuning kernel** | **Closed by measurement, before any GPU spend.** The prior-art agent kept it conditional on a CPU rank check; that check has now been run twice independently. My own reproduction on three CO sessions: `[2,K]` rank-1 fraction `0.860 / 0.854 / 0.839` at M50 and `0.864 / 0.908 / 0.875` on all rewarded trials; variance captured by the T4 outer product `0.793 / 0.763 / 0.778` at M50 and `0.816 / 0.862 / 0.832` on all trials. **Both rise with more trials**, so the M50 residual is sampling noise rather than recoverable interaction. The pre-registered abandon criterion was rank-1 below `~0.6` and capture below `~0.5`, not increasing. The descriptor is ~80% an outer product of T4's direction with a single shared time shape, which per-column z-scoring partly removes anyway. The raw 8x100 PSTH interaction is *not* rank-1 (`0.30–0.33`), but that residual structure lives in higher angular harmonics, already priced at `T8 − T4 = +0.003979`. |
| 3 | All-bins auxiliary loss | **Demoted to a screen.** The batch fact is right but I cited the wrong file copy: `covariate_window = ...  # W x C` is at `streaming_calibration_exp/src/data/falcon_datamodule.py:572`, not the `SPINT-main` copy. `fc_out` already emits all `W` bins so no new head is needed, and the untrained H1 count is `715,776` **weights** (the 700 biases are extra). But the teacher never trained those rows either, and on the FALCON side they stay dead under `freeze_decoder: true`. Keep it as a cheap source-training screen for whether the single live readout row binds; it is not a paper section. |
| 4 | TV1 restated as source-side descriptor diversity | **Restate accepted, with two corrections.** M30 ≈ M50 genuinely does not close source-side hull coverage, and nothing has trained on out-of-hull descriptors — the noise path is real and unwired (`set_carrier_noise_cholesky` defined once, never called; `b3_carrier_noise_scale1_m2_t4.yaml` tagged UNLAUNCHED). But the consumer sees **27 distinct per-session T4 clouds** per epoch, not one global vector, so my framing oversold the starvation; and SSC-T4 (`−0.004432`, 2/6) is the *wrong* neighbour — it splits activity M24 into even/odd and holds the full-M24 T4 fixed. Still a training-method item, not an estimand, and only after a CPU wiring proof that `scale=0` is bitwise T4. |
| 5 | Trial time-warping | **Closed.** T4 integrates spike times over `[start_time, stop_time)` and the FALCON path deliberately uses un-interpolated trial sums, so T4 is **invariant** to the cubic warp; the warp only reaches `pre_pool`. My implication that it distorts T4 is false. It mattered only conditional on item 2, which is dead. |
| 6 | Consistent unit-subset augmentation | **Closed as a contribution.** The masking gap is real (`model_step` masks `calib` and `neural`, never `side_features`) and `neuron_dropout_mode: none` everywhere, so it never fires. But the proposal itself already exists as `correspondence_breaking` mode **`subset`** — same index on all three tensors — and it is both UNLAUNCHED and **broken**: `_subset_last_dim` is called three times and defined zero times. The 2026-08-12 audit already classified consistent subsetting as population-size robustness, not a mechanism. Worth a small code fix; not a paper section. |

**Both agents independently concluded that only T3K was ever structural**, and T3K is now closed by measurement. The
honest reading is that the accuracy axis under the trial-level-label and BP-free constraints is exhausted, and the
remaining value is in the claim (§1.2) rather than in a new arm.

---

## 3. What survives, ranked

### 3.0 PRI-T composition — the one live method-level candidate

Wilson et al., *Nature Biomed. Eng.* 2025, "Long-term unsupervised recalibration of cursor-based intracortical BCIs
using a hidden Markov model" (`s41551-025-01536-z`; code at `github.com/guyhwilson/PRI-T`, closed-loop data at
`10.5061/dryad.1jwstqk6g`). PRI-T runs an HMM over **decoder outputs** to infer which discrete target the user is
moving toward, yielding pseudo-labels plus confidence, then refits by weighted least squares. It uses **zero** labels.

**This is first a threat to §1.2 and must be answered in the paper.** The answer rests on four limitations the paper
states itself: PRI-T needs a **human in the closed loop** (it explains its own offline-versus-closed-loop gap by the
user's visual corrections making outputs informative about the target); it needs **discrete-target 2D cursor task
structure**, so it does not transfer to 7-DoF H1, 16-channel EMG M1, or continuous-target RT; it **fails above ~90°
subspace rotation** in the single-pair setting and its thesis is that you need a **chain of daily** recalibrations;
and it **updates decoder weights** per session. So the axes are orthogonal: PRI-T trades task structure, a closed
loop and a session chain for labels, while T4 trades ~750 trial-level scalars for none of those and a frozen decoder.

**Then it is an opportunity.** PRI-T infers exactly the quantity T4 consumes — a direction label per trial. Composing
them makes the pipeline **label-free and still BP-free**: Viterbi plus weighted least squares on one side, OLS on the
other, no optimizer at deployment. The falsifiable prediction is sharp and uses existing numbers: bootstrapping needs
a usable starting decoder, and within-subject cross-session `F0 = 0.3139872` plausibly supplies one while external
cross-subject `Zero4 = −0.057766` does not. So it should work within-subject and fail cross-subject, and both branches
are reportable.

**Do not reproduce the paper.** There is no closed-loop rig and no clinical participant here, the FA-stabilization and
ADAN comparator numbers are already published, and this repo already closed the alignment family
(`generic_cca_or_procrustes` is `FAIL_CLOSED_NO_PAIRED_OBJECT_OR_ANCHOR`). Clone the compact repo to lift the HMM
observation model and Viterbi step; skip the Dryad download.

**Two free framing gains.** The paper is independent external evidence that the **alignment** family accumulates
compounding error and diverges over long horizons while label-based self-training can drive error back down — which
is why a label-based descriptor is the right family, and it converts T4 from "a small encoder tweak" into the
BP-free single-shot member of the family the field's own long-horizon evidence favours. And its Fig. 1b fits
**per-unit cosine tuning** within each session, with tuning-weight cosine similarity falling from `0.75`
within-session to `~0.33` at 1–2 weeks, which establishes T4's estimand as the field-standard characterization of
the nonstationarity this method is built to absorb.

### 3.1 T3K — CLOSED, retained for provenance

**The change.** Replace the estimand. Today `_pool_trial_rate_matrix` (`unit_side_features.py:970-998`) integrates
whole-trial Hz, `_unit_tuning_features` (`:882-928`) collapses to ≤8 per-direction means, and `_fit_cosine_tuning`
(`:728-752`) fits 3 parameters from ≤8 rows. Instead take the calibration tensor `[M, 100, N]` that
`_build_calib_trials` (`multisession_datamodule.py:573-603`) **already materializes** by cubic-interpolating each
trial onto a normalized 100-bin trial-time axis and already passes in as `calib_trialized_neural_features`, project
time onto the existing `_raised_cosine_temporal_basis` (`streaming_encoders.py:864-884`) to get `[M, K, N]`, and per
unit solve `c_i[k,m] ≈ β0_i[k] + βc_i[k] cos θ_m + βs_i[k] sin θ_m`. The design matrix `[M,3]` is identical for every
unit and basis index, so one `[3,M]` pseudo-inverse per session serves all units via a single einsum. Drop `β0` (the
direction-marginal profile is what `pre_pool` already pools) and the descriptor is `[2,K]` — 12 wide at `K=6`. It
enters the existing `side_features` port with `side_dim=12`; the zero-init at `streaming_encoders.py:397-399` means
the arm starts exactly at activity-only B3, the same starting point T4 had.

**Magnitude.** The measured price of per-unit temporal resolution on this exact dataset is `+0.2190498` (§1.3), and
the descriptor currently carries **zero** of it — whole-trial Hz is temporally constant by construction. Capturing
15% is `+0.033`. The one richer-descriptor experiment on record, `T8 − T4 = +0.003979`, added *angular* resolution,
a different and now-priced-worthless axis. Note the estimator objection runs backwards: T4 fits 3 parameters from ≤8
rows, T3K fits 12 from `M × 100 ≈ 3000`, and §1.2's saturation by M30 says the label budget has spare capacity.

**What structurally prevents it today.** Both paths destroy, before pooling, exactly what the other needs.
`push_trial` accumulates `state["sum_feat"] += pre_pool(trial)` (`streaming_encoders.py:404-409`), summing over
trials and destroying the trial→direction association; `_pool_trial_rate_matrix` integrates over `[start_time,
stop_time)`, destroying time. So `h_i` is a direction-marginal temporal profile, `T4_i` is a time-marginal
directional fit, and `post_pool` receives their concatenation. **No MLP of two marginals can recover an interaction
integrated out upstream.** That is an information-theoretic absence, unlike the nine flat fusion arms, every one of
which added a path to information already present.

**Cost.** Reads the already-materialized `[M,100,N]` tensor and one direction scalar per calibration trial — both
already read. No per-bin kinematics. One `[3,M]` pseudo-inverse plus one einsum. Cached state unchanged at `[N,50]`.
No optimizer, no loss, no backward.

**Required sibling.** `T4 ⊗ basis`: the outer product of the existing 4-vector with the K basis coefficients of the
direction-marginal profile. Same width, same estimator cost, **no interaction term** — the width-matched
mechanism-free control, so the contrast isolates the interaction rather than the width. Primary null stays LS4.

**Most likely failure.** CO velocity within a reach is `s(t)·u_θ` over 8 directions, so the direction-dependence of
the time course may be pure amplitude scaling of a common profile, making `[2,K]` rank-1 and collapsing to T4 times a
fixed shape. Closest measured negative: fixed-K temporal prototypes, `P20 − B20 = −0.041963`, 0/4 — narrow defence
being that P20 was strictly **label-free**, and the repo has now twice priced unlabeled per-unit descriptors at
nothing (`N4 − NS4 = +0.001588`).

**If flat.** A negative with teeth: the per-unit direction×time interaction, estimated at ~375x the design rows of
the whole-trial fit, is inert for this consumer while per-unit temporal resolution is worth `+0.219` to a linear
readout on the same data. That localizes the ridge gap to supervision density alone and retires the estimand axis.

### 3.2 All-bins auxiliary loss — free, and it screens the readout premise

I killed this last revision by bundling it with "localize identity to the scored bin" and then rebutting the latter.
That was a category error. "Windows slide, so every timepoint is already used once as some window's bin 49" answers
*are the labels wasted?*; the proposal's claim is *is the representation under-constrained?* Deep supervision needs
the same labels in different contexts, not fresh labels. "The 49 extra tasks predict from their own future" is a
valid deployment objection and a non-sequitur as a training objection — the auxiliary head is dropped at deployment
and the input window is unchanged, so there is no leakage.

`covariate_window = self.covariate_data[session_name][start_idx:end_idx]  # W x C` (`falcon_datamodule.py:572`) means
the dense per-bin targets are **already in the batch and already discarded**. Zero new labels, zero target-session
supervision, zero deployment change, so the `199.633x` differentiator is untouched. It is a training-method change
with structure fixed, explicitly in scope. It revives 715,776 `fc_out` parameters on H1 that currently receive no
gradient in any config. And it is the cheap screen for the readout premise: if an all-bins auxiliary loss is flat,
the single-live-row readout is not a bottleneck.

### 3.3 Forward-only carrier-sensitivity curve — the gate on everything noise-related

Perturb the carrier **at deployment** using the analytic OLS covariance that `carrier_noise_augmentation.py` already
implements and already tests to 15% relative, and measure R² degradation versus perturbation scale. Forward-only,
BP-free, hours, no cache-key change, no new artifacts. A flat curve delivers exactly the claim TV1 was supposed to
produce — the consumer is insensitive to descriptor perturbations of the size M=15 estimation noise induces — at
roughly 1% of TV1's cost, and kills TV1. A steep curve motivates TV1 and names the scale to sweep.

Positioning must be honest: this is a **gate**, not a paper item. It is admissible under the premise only because it
prevents a cache-invalidating build.

### 3.4 The `[a,c]` z-scoring defect — a correctness fix worth finding ourselves

`fit_side_feature_stats` z-scores every side-feature column independently, so `a` and `c` receive different scales,
and the consumer is handed an **elliptically distorted angle**. The comment at `unit_side_features.py:916-925`
explains at length why emitting bare `cos φ, sin φ` would be wrong because independent per-column z-scoring "would
rescale it into an ellipse and destroy the 'this pair is a single angle' geometry" — and the code then does exactly
that to `a` and `c`, which are the same angle scaled by `m`. Fix: shared scale across the pair. `m = hypot(a,c)` is
already exactly rotation-invariant, so `[a,c,m,b]` splits cleanly into an equivariant pair and two invariants.

The optional structural extension is an SO(2) orbit during source training — sample `ψ` per batch, apply
`(a,c) → R_ψ(a,c)` and `y → R_ψ y` — which removes function-class volume rather than adding a path, so it cannot be
"another redundant fusion". Applies only where the covariates are a plane (`C=2`: SUA, M2). Closest measured
negative: `SSC-T4 = −0.004432`, 2/6.

### 3.5 Held, in order: Template-Ridge; `H-U`; the `C=2` disconfounder

**Template-Ridge** instantiates the ridge's *object* — a session-specific 50×N causal map — with a synthesized target
`Y_synth(t) = s(t − go_cue) · [cos θ, sin θ]`, where `s` is a **source-only** speed profile. The target session never
opens `cursor_vel`. Its magnitude anchor is the within-trial target-variation dose (`+0.289`/`+0.328`). Two honest
caveats: the solver still sees thousands of *constructed* window rows, which are not measured kinematic labels and
must be described that way; and §1.2 now shows the dense-versus-sparse gap is precisely a supervision effect, so
synthesizing supervision is either the whole answer or a stereotyped approximation to it.

**`H-U`, one arm only.** Cut the `H-S(p)` capacity sweep entirely — it will separate nothing on a `+0.0389` budget.
Keep a single zero-learned-parameter, label-free 4-vector built from **rotation-invariant** per-unit statistics
(mean rate, log ISI CV, Fano factor, autocorrelation timescale) — **not PCs**, whose per-session sign and rotation
ambiguity would test the registration thesis with a broken instrument. Frame its positive branch as a headline, not
a scope note: "on fixed-array within-subject data a label-free zero-parameter descriptor matches a 5,965,500-parameter
learned identity encoder." And disclose voluntarily that if it holds, the `102.6x` compression is *also* not
attributable to the functional carrier on H1.

**The `C=2` disconfounder** for §1.1, which costs no new supervision.

---

## 4. Order

Every arm this document previously ranked is now closed — RW1 by measurement, DR1 on the remainder of its case, TV1's
premise, DQ1 on payoff, T3K by measurement, and all six of §2b. What remains is one method-level candidate and three
riders.

**PRI-T composition first, and it is the only remaining item worth a GPU queue.** It is method-level rather than
detail-level, it directly answers root's objection that the contribution is a small encoder change, and it has to be
engaged with regardless because it is a published zero-label competitor to §1.2's claim. Start CPU-only: lift the HMM
target-inference step, run it on SUA validation sessions against the existing frozen source decoder's outputs, and
measure inferred-versus-true direction agreement per session. **That agreement number is the gate.** If inferred
labels are no better than chance on within-subject sessions, the composition cannot work and the paper carries the
threat analysis only.

**Three riders, none of them a cell.** Wire the carrier-noise Cholesky and prove `scale=0` is bitwise T4 (§2b item 4).
Fix `_subset_last_dim`, called three times and defined zero times. Fix the `[a,c]` z-scoring so the pair shares a
scale instead of being rescaled into an ellipse (§3.4). All three are correctness work better found by us than by a
reviewer.

**Then write.** §1.2 is measured, large and near-unanimous in sign; §1.3's pricing table plus the closures in §2 and
§2b form a bounded scope statement few method papers can offer. Further ideation now has negative expected value:
today's ten agents net *removed* candidates, which is the signature of an exhausted search.

GPU is free — A1 and B1 both completed and no training process remains.

---

## 5. Standing preconditions, not deliverables

Primary null is **LS4**; Z4 is a floor reference and is vacuous for any `f(0)=0` design; row permutations are vacuous
for any permutation-invariant set function of the carrier. Every gate needs one synthetic must-pass and one synthetic
must-fail input asserted in the aggregator suite before it is frozen — five gates in this program have turned out
incapable of acting, including one of mine.

One rule to add: the "if flat, the paper gets" column may only cite scope statements that survive the
no-negative-arms rule. "A refuted dose-response" is a negative arm, so it does not count as a payout.
