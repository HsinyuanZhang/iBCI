# HANDOFF: where accuracy can still come from on top of T4

**Date:** 2026-08-13
**Status:** root-audited design record plus live execution ledger. Section 3 lists closed ideas so they are not
proposed again. A2 is terminal; A1 and B1 Stage P are terminal routing outcomes; C1 and C2 are held.
**Authorizes:** nothing new by itself. The A1/B1 terminal values and decisions are recorded once in section 5.3.
The A12 CPU audit has a non-causal `3/24` partial aggregate (SHA `2bf287...`), not a gate.
**Scale expectation:** A2's frozen aggregate shows that the relative `T4-Z4` carrier margin grows under the
observed subject shift (`+0.235799`, not an absolute T4 lift). What follows targets the native accuracy
remainder — a few points, not another `+0.25`.

**Conflict rule:** Sections 1--4 preserve the brainstorm and its audit history.  Where an earlier priority,
gate, or execution statement conflicts with section 5, the root-audited section 5 is authoritative.  It does
not itself authorize a GPU launch.

---

## 1. The three directions worth pursuing

### A. Change what identity *is*

Today `post_pool` maps `[mean_h, T4]` into `R^W`, and the decoder computes `src = x + E` before `fc_in`. So the
identity token lives in the same space as binned spike counts and is read by a filter bank trained on spikes.
T4 is four tuning coefficients; `post_pool` has to paint them into a 50-bin waveform and `fc_in` has to read
them back out.

Nothing requires this. The alternative keeps the pretrained read-in — the lesson of decoupled K/V's
`-0.444658`, which failed because it deleted `fc_in` — and drops only the waveform contract:
`h_i = fc_in(x_i + E_i^A) + P(e_i)`, with `e_i` a short code and `E_i^A` the matched
activity-only identity. The bare route without `E_i^A` would also remove activity identity and is therefore a
broader identity-route replacement, not a pure add-site comparison.

**Why it is still standing after four audits.** None of the nine flat or negative arms tested it: six kept
`src = x + E` and added another route, decoupled K/V destroyed the read-in, slot routing compressed units, B15
mixed neurons. The audits killed the cheap premise-check I proposed for it (an identity-rank diagnostic, which
cannot work because `E` mixes activity and carrier), not the idea.

This is the largest structural change available and the only one that changes the *type* of the identity.

### B. Make source training practise the condition that actually pays

A2 now measures that the carrier's relative value concentrates under the observed target-subject shift:
three-seed interaction `+0.235799` (bootstrap interval `[+0.100852,+0.371768]`), with activity-only identity
going negative on sub-M while T4 remains positive.  This is a relative interaction, not an absolute lift of
the same size.

Source training never creates that condition. Every batch is a single session with matching calibration
activity and matching T4, so the student can spend the activity path, get a good last-bin MSE, and never
practise decoding when activity identity is unreliable.

**Why it is still standing.** The audits killed my *design* — `p = 1` leaves the activity path untrained but
live at evaluation because the mask is `if self.training`; two cells cannot satisfy the Z4-sibling rule; one
seed cannot support an interval — but not the mechanism. All three are fixable.

### D. Synthesize supervision density instead of re-parameterizing the descriptor — Template-Ridge

**Added 2026-08-13 evening. This is the strongest remaining structural direction, and the only one that attacks the
measured cause of the classical gap rather than working around it.**

**Why it is worth doing.** §1.2 of the subordinate proposals file decomposes the Ridge50 gap and finds it is
*entirely* supervision density: holding features, calibration trials and query windows fixed and changing only the
target, a per-session ridge held to trial-level direction labels goes **negative** (SUA M50 `−0.122027`,
pseudo-MUA M15 `−0.424391`), while `dense − sparse = +0.539948` (15/15). Every attempt this program made to reach
that gap by re-parameterizing the 4-vector is now closed and priced — angular resolution `+0.003979`, population
whitening `+0.0017046`, precision/budget `+0.001326`, response-window realignment closed by measurement, and the
direction×time kernel closed by a rank-1 collapse. The gap is not in the descriptor. **Template-Ridge is the only
admissible route left to the supervision itself.**

**What it is.** Fit the ridge's *object* — a session-specific `50×N → C` causal map — at the target session, but
against a **synthesized** target `Y_synth(t) = s(t − go_cue) · [cos θ, sin θ]`, where `θ` is the trial's discrete
direction label and `s` is a speed profile learned **on source sessions only**. The target session never opens
`cursor_vel`, `cursor_pos` or `cursor_acc` in any role.

Two versions, and they are not interchangeable:

- **D-a, as a decoder.** The fitted map decodes directly. This replaces SPINT at the target session and therefore
  **forfeits the identity-path compression claim** (`102.6x`, `58,140` against `5,965,500`) and the deployed-object
  accounting built on it. Do not run D-a without deciding first that the paper is willing to pay that.
- **D-b, as a descriptor source.** The map is fitted, then a fixed-width per-unit reduction of it is fed into `E`,
  and deployment stays SPINT. Compression and the BP-free contract survive. **D-b is the version this entry
  recommends.**

**Cost, stated explicitly, because two of the three costs are much larger than T4's and one of them is invisible
if you only count labels.**

1. **Label cost — unchanged, and this is the point.** Per calibration trial it reads `target_dir` plus task-event
   timestamps (`go_cue_time`, `start_time`, `stop_time`, `result`). `go_cue_time` is finite on 99.89% of rewarded
   trials (13,929 of 13,945 across 53 CO sessions). So the claimable ratio stays `199.633x` fewer *measured*
   kinematic target rows than Ridge50.
2. **Design-size cost — as large as Ridge50's, and it must be disclosed.** The solver sees thousands of
   **constructed** rows. From the label-density receipt's own conditioning block, a design of this shape is
   `rows 9744 / features 4600 / ratio 2.118` at SUA M50, `5958 / 3100 / 1.922` at M30, and `3568 / 3100 / 1.151`
   at M15, with `gram_regularized_condition 63.98` and `trace_hat 1944.09` at M50. Those rows are *generated from*
   sparse labels; they are not measured kinematics. **Say so in the paper before a reviewer says it.** The honest
   sentence is "the same design size as the dense comparator, built from `199.633x` fewer measured target rows",
   not "the same supervision".
3. **Calibration compute cost — four to five orders of magnitude above T4, and it scales badly.** T4 is one shared
   `[3, n_dir]` pseudo-inverse, microseconds. Template-Ridge at SUA M50 forms a `4600 × 4600` Gram from a
   `9744 × 4600` design and factorizes it: roughly `2×10¹¹` flops to build the Gram and `3×10¹⁰` to factorize,
   with a `359 MB` design matrix and a `169 MB` Gram in float64. It scales as `O(N²W²)` in features, so a wider
   window or higher unit count grows it quadratically. On a workstation this is seconds; **for the on-device story
   it is not free and must not be reported as "one closed-form solve" without the size**.
4. **Cached-state cost at deployment.** D-b caches a per-unit reduction, so state stays comparable to today's
   `[N, 4]` descriptor plus the `[N, 50]` identity. D-a would cache a `[50N, C]` map — `6400` floats at `N=64`
   against `256` — which is the other half of why D-a threatens the hardware accounting.

**The failure mode is already measured and it is not a flat.** If the synthesized target is too stereotyped, the
arm degenerates toward the sparse-direction target, which measured **`−0.122027`** at SUA M50. So a crude template
does not merely fail to help, it can be a large negative. The quantity that decides this is how much real velocity
variance the template explains: a 0.6 s minimum-jerk template reaches affine-R² `0.371` and correlation `0.609`
against real velocity, and a source-empirical go-cue-aligned profile reaches `0.411` and `0.641`. **Freeze the
template family before running**, or the arm becomes a search over templates with the outcome as the selector.

**Pre-conditions.** Choose D-a or D-b in writing first. Freeze `s` from source sessions with its fitting procedure
recorded. State the constructed-row disclosure in the receipt, not only in the paper. And note that the one
inference-based alternative source of labels is now closed: PRI-T composition measured `0.119` against 8-class
chance `0.125` at the operative cold-start condition (§3.A), so synthesis, not inference, is the remaining route.

### C. Align a recipe that was inherited rather than chosen

This direction is underrated because each piece looks like a settled default. None of them was selected for
this problem:

- **The teacher is MC_Maze `hand_vel` with fixed `N`.** `task_only` removed the distillation terms but not the
  warm start; A2's checkpoints still load it. P3 already blamed this domain mismatch and nobody has ablated it.
- **The loss is not the estimand.** Training MSE is window-count weighted — `SessionBatchSampler` drops
  remainders, so long sessions get more gradient steps — while scoring is an unweighted session-mean R².
- **`lambda_E` was selected on a carrier-free student.** The original R1 selection ran on `streaming_b3` with
  `side_dim = 0`, so the M2 carrier arm inherited a decision taken with no carrier present, and that term pulls
  the student identity toward an activity-only teacher identity that A4 measured to be phase-poor.

They are independent and untested on the current system, but not equally cheap: changing the loss or the
M2 sampler is bounded, whereas changing teacher domain requires a new compatible teacher/source-training
lineage.

---

## 2. Priority

> **Historical brainstorm table, superseded by section 5.**  It records candidate generation, not current
> gates or an execution queue.  Section 5 separates A1 from teacher domain, holds R2-native training, and uses
> B1's frozen gate with a descriptive bootstrap interval.

| # | Change | Cost | Gate |
|---|---|---|---|
| **1** | **B1 carrier-by-distillation factorial.** `{T4, Z4} x loss_mode` on M2. | GPU, staged under the frozen contract | Stage-P routing: mean interaction `>=+0.03` and 3/3 seeds positive. Stage-F terminal: mean `>=+0.03`, 3/3 session means and 3/3 seed means positive; interval descriptive only. |
| **2** | **A1 hidden-space carrier addition.** `{W-add,H-add} x {T4,Z4}` with TS4 attachment controls. | CPU contract and independent review first; not A10-gated | Primary interface-by-content interaction `>=+0.03`; preserve matched activity identity, require `P(0)=0`, zero-init parity, unchanged teacher/decoder and identical provenance. |
| **3** | **C1 teacher-domain ablation.** `{MC-Maze,CO-native} x {T4,Z4}`, W-add fixed. | GPU, new teacher | Source-only compatibility first; external T4 lift plus carrier interaction, without changing add site or sampling. |
| **4** | **CF1 carrier forcing.** Partial activity-path dropout, `{T4,Z4} x {p=0,p>0}`. | GPU, held | Absolute external T4 lift `>=+0.03`, within-C T4 non-inferior, and Z4 must not reproduce the lift. |
| **5** | **C2 equal-session M2 sampling.** `{legacy,equal-session} x {T4,Z4}`. | GPU, held | Improve unweighted session-mean R2 by `>=+0.03` without a generic Z4 lift; ordinary MSE first, not R2-native training. |
| **6** | **B2 carrier-corruption safety.** Clean versus session-consistent corruption. | GPU, downstream of B1 | Wrong-content RS4/LS4 moves toward Z4 while clean T4 content and deployment performance are retained. |

**Historical status, superseded by section 5.3.** B1's frozen scientific gate and terminal Stage-P decision are
recorded in section 5.3. B1 remains an M2-line hypothesis and does not bear on the SUA subject-shift estimand,
so do not read its result as evidence about A2.

**A1 and C1 are deliberately separated.**  A1 changes the add site while retaining the compatible teacher and
decoder state wherever the contract permits; C1 changes teacher domain while W-add stays fixed.  Combining
`CO-native + H-add` in the first arm would make a positive result unattributable.

**One measurement rule for every arm scored across domains.** The primary endpoint is **absolute external T4**,
never `T4 - Z4`. The difference inflates when Z4 crashes, which is not a win. Every such arm needs a Z4 sibling
that does not show the same lift, and a within-domain non-inferiority floor of `-0.03`.

**Running in parallel, gating nothing:** the behavior-normalizer decomposition on sub-M (note it is *not*
label-free — behavior statistics are fitted from `cursor_vel`, the decoding target); a B2-style safety arm,
which needs its own SUA contract because the existing B2 is an M2 experiment and cannot safety-test a SUA
checkpoint.

---

## 3. Analysed and found ineffective — do not re-propose

**Measured negative or flat.**

| Intervention | Result |
|---|---|
| Confidence-FiLM | `+0.003399` |
| Live-activity gain (L-D) | `-0.002068` |
| Rank-8 attention-logit residual | `-0.003142` |
| Electrode gate | `-0.010817` |
| Same-electrode relation | `-0.001440` |
| Interface width 32 to 64 (CI64) | `-0.020130`, terminal; `H64` prohibited |
| Decoupled K/V v1 | `-0.444658` — but this deleted the pretrained read-in, so it is a read-in ablation |
| Fixed slot router, K=32 | `-0.177935` |
| Cross-neuron encoder attention (B15) | gain explained by capacity: `B15 - B15P = +0.006354` |
| **T8, raw per-direction means** | **`T8 - T4 = +0.003979`** — no gain for this model/dataset/budget/schedule; not proof that the cosine fit is universally sufficient |
| N4 label-free static descriptor | `+0.001588`, 3/6 sessions |
| Fixed-K temporal prototypes | lost to a simpler order-invariant baseline, `-0.041963`, 0/4 |
| Wiener shrinkage at low budget | `+0.000753` decoding, despite a train-only proxy winning 27/27 |
| Query-fitted oracle carrier on one frozen consumer | `-0.002859` — local leakage-diagnostic insensitivity, not a strict estimator ceiling |

**Killed on reasoning, not measurement.**

- **Procrustes alignment of T4 clouds.** Needs paired rows that unordered unit populations do not provide, and
  would rotate `[a,c]` out of the shared task frame the decoder was trained in.
- **Rank-1 `fc_in` modification on its stated motivation.** The tested L-D gain is `Linear(4, 50, bias=False)`,
  an explicitly per-bin diagonal map that already rotates the `W`-vector.
- **Signed readout and linear attention.** Attention weights are non-negative but values are signed, so a unit's
  contribution can already be negative.
- **Identity-rank diagnostic as a decision fork.** `E` mixes activity-derived identity with the carrier, so its
  spectrum cannot isolate the carrier's contribution and both competing hypotheses survive either outcome.
- **Joint training on sub-M.** Destroys the A2 estimand.
- **Unmixed latent query tokens.** Structurally inert without a query-mixing path.

**Four numbers that keep being misquoted.**

- `0.693663` is **B3**, one session, 80/20 split, decoder unfrozen. It is not a T4 ceiling.
- `0.357220` was the **A2 external sub-M seed-43** interim T4 value.  It has been superseded by the terminal
  three-seed aggregate (`T4=0.341367`, `Z4=-0.143399`, interaction `+0.235799`) and must not be quoted as the
  final result by itself.
- `m2_joint_t4_upperbound_v2`'s `+0.018389` is a contrast among independently joint-trained arms, **not** an
  unfreeze factorial.
- `Z4 - B0 = +0.089591` is a four-wide **all-zero** side port beating no port. Port width alone has an effect,
  so any width-changing comparison needs a padded control.

### 3.A Appendix, 2026-08-13 evening: closures added after the A1/B1 receipts

Fourteen agents ran between 15:00 and 19:30. They **net removed** candidates. Everything below was closed with a
number, and every number was re-verified against code, receipts or raw NWBs before being written here.

**Measured negative, flat, or closed by direct measurement.**

| Intervention | Result |
|---|---|
| A1 hidden-space fusion (the tenth fusion arm) | `+0.012833`, 4/6 sessions, median `+0.0175`, `PILOT_ROUTING_STOP` against `+0.03`. Valid test: `optimizer_tensor_count 40` vs 39 for A2's W cells, attachment control `+0.5793` |
| B1 carrier x distillation | `+0.059153 / +0.004307 / −0.022215`, mean `+0.013748`, `STOP_B1_NO_STAGE_F`. Spread `0.081` is six times the mean: a high-variance null, not a small positive |
| **Response-window realignment (go-cue anchoring)** | Closed by measurement on three CO sessions: whole-trial versus go-cue-anchored pooling gives `corr(a) = 0.9764 / 0.9432 / 0.9745`, z-scored `[a,c]` per-unit cosine median `0.9887`, and median `m_go/m_whole = 2.2993`. The `0.7031` hold dilution is a nearly **uniform** scale factor, which per-column z-scoring removes |
| **T3K, direction x time tuning kernel** | Closed before any GPU spend. `[2,K]` rank-1 fraction `0.860 / 0.854 / 0.839` at M50 rising to `0.864 / 0.908 / 0.875` on all trials; variance captured by the T4 outer product `0.793 / 0.763 / 0.778` rising to `0.816 / 0.862 / 0.832`. **Both rise with more trials**, so the M50 residual was sampling noise. The descriptor is ~80% T4's direction times a shared time shape |
| **PRI-T composition (external HMM target inference feeding T4)** | Dead at the operative condition. Deploying the real T4 checkpoint with identity zeroed gives `0.119` against 8-class chance `0.125`; a purpose-built Z4 checkpoint gives `0.295` at the authors' defaults and `0.438` as best-of-276 tuned on the eval sessions. The encouraging `0.723` exists only after true labels have already built the identity. Separately: the HMM adds only `+0.016` over snapping mean decoded velocity to the nearest of 8 directions (`0.7075` → `0.7232`), and that `0.723` required adding a speed gate because PRI-T's emission model discards velocity magnitude |
| Cross-unit Gram / population whitening | `+0.0017046` (`population_vector 0.0870177` → `ridge_pooled_rate 0.0887223`). This prices the whole population-statistic family, including a whitened decoding row, at essentially zero |
| Un-shared `fc_out` readout row across covariates | Dose-response measured **inverted**: covariate participation ratio M1 `3.84–3.93`, H1 `2.93–3.06`, M2 `1.82–1.91` against `num_heads: 64`, and per-covariate independent dimensions run M2 `≈0.95` > H1 `≈0.43` > M1 `≈0.24`. Also "row 49" is wrong on both intended hosts (`window_size` is `700` on H1, `100` on M1) |

**Killed on reasoning, not measurement.**

- **Encoder–decoder co-design as a new design.** `checkpoints/a2_matched_subject_shift_v2_source_t4_dandi688_co_s42/run_metadata.json` records `training/freeze_decoder = False` with `side_features/group = t4`, and `--freeze_decoder` is `store_true` which A2's launcher never passes. The SUA mainline already jointly trains encoder and decoder with T4 present. What is unrun is decoder-from-random-init, which would forfeit the transfer property rather than test co-design.
- **Session-conditioned residual on the query token.** Decomposes into a rank-≤`C` perturbation of the `C×N` score matrix — a constrained special case of the rank-8 logit residual already measured at `−0.003142` — plus a per-covariate output bias. LayerNorm makes leg one *stronger*, not weaker.
- **Scoring all 50 output bins, and localizing identity to the scored bin.** Windows slide, so every behavioural timepoint is already used once as some window's bin 49; and the extra tasks predict bins from windows containing their own future.
- **"Carrier value equals correspondence-problem severity" as a central claim.** The four-dataset ordering is perfectly collinear with `1/num_covariates`, and its low anchor is confounded: M1's behaviour is 16 EMG channels (`acquisition["preprocessed_emg"]`) while M1's carrier is a target-azimuth cosine, so M1's `−0.00652` is a descriptor–estimand mismatch, not a correspondence datapoint.
- **`+0.013` as a property of the system.** Gates are *upper* thresholds, and a dozen measured effects sit below the band (`+0.00969`, `+0.006354`, `+0.003979`, `+0.002987`, `−0.002068`, `−0.003142`, `−0.014218`, `−0.020130`, …), centred near zero with spread `≈±0.02`. The band members are that distribution's upper tail, selected on outcome, and grouping them merges a `p = 2⁻¹²` effect with an unresolved null.

**Numbers that keep being misquoted — six more.**

- **`+0.296802` is Full versus B2**, a `side_dim=0` activity-only LatePool identity, **not** velocity versus direction. On the same 15-fold nested LOSO the direction-based T4d scores `0.448176` against dense velocity's `0.445189`, a difference of `+0.002987`, and the receipt explicitly declines any superiority, equivalence or non-inferiority claim.
- **`+0.2190498` is 50 taps at 20 ms versus one pooled scalar**, not 100 taps versus 50. The same file records `gap_best_minus_F0 = −0.006215` against a 2σ tolerance of `0.0213`, so activity-only SPINT does not *exceed* the 50-tap ridge — a path that extracts nothing nonlinear from 50 taps is not fixed by giving it 100.
- **The H1 `H-C / H-LS / H-S / H-C0` set is fold-0, single date `19250101`, seed 42, epoch-49.** It is not the five-date protocol, so `H-C − H-S = +0.056287` from `CURRENT_RESULTS.md` must not be combined with it. And **H-S is a different init lineage** (`f4f876d5…`) from H-C and H-C0 (`8208b6eb…`), so any H-S comparison is system-level, not controlled.
- **Two comparator families differ by three to five times, and the interventions measure different things.** Separately trained: `H-C − H-C0 = +0.038895`, `H-C − H-LS = +0.025616`. Same-checkpoint forward-only on frozen H-C: `zero +0.103872`, `row +0.153079`, `label +0.131571`. `row` permutes the 176 unit rows (whole 4-vectors together) and is the **unit**-attachment number; `label` rolls each support trial's velocity **in time** and refits, so it is the **temporal**-attachment number. Both families are valid answers to different questions — the separately-trained family lets the net re-optimize around missing content, the interventions lesion a net that already learned to rely on it. Quoting only the conservative family understates what the trained decoder uses.
- **PRI-T's Fig. 1f is population readout subspace drift**, defined over flattened ridge *decoder weight matrices*, not per-unit tuning similarity. The paper prints `0.75` (41°) within session and 71° at 1–2 weeks; the `0.33` that circulated here was our own `cos 71°` arithmetic. Quote the R² fall from `0.39` to `0.21` instead. Fig. 1b *is* citable for the estimand — Methods give `x_t = b₀ + b₁cos(θ_t) + b₂sin(θ_t)` fitted by standard least squares, identical to T4 — with one honest difference in our favour: they regress against the *instantaneous* angle and so need dense kinematics, where T4 uses one scalar per trial.
- **"BP-free" is not a differentiator against PRI-T and must stop being claimed as one.** Its refit is `decoder.fit(neural_flattened, inferredPosErr, maxProb**2)` with `from sklearn.linear_model import LinearRegression` — closed-form weighted least squares, no optimizer, no backward pass. The surviving, narrower claim: PRI-T's closed form exists *because its decoder is linear*, whereas T4 performs BP-free adaptation of a **nonlinear cross-attention decoder**, for which PRI-T has no analogue.

**Code paths that silently do nothing, or crash if used.**

- `set_carrier_noise_cholesky` occurs **once** repo-wide, its own definition, so `_carrier_noise_cholesky` is always `None` and `b3_carrier_noise_scale1_m2_t4.yaml` would produce a bitwise-identical baseline.
- `_subset_last_dim` is called three times and defined zero times, so `correspondence_breaking` mode **`subset`** — which is the consistent-unit-subset proposal — raises `NameError` if invoked.
- `activity_path_dropout` occurs **zero** times in `sua_exploration/scripts/train_variant_dandi688.py`, so CF1 is not runnable on SUA today; the `0.0` in A2's hparams is a constructor default no CLI flag reaches. Only M2 is wired.
- A10's preflight appends `SUPERSEDED_A10_V1` **unconditionally** and exits 2, and the named reference scorer contains zero occurrences of `adaptation_mode` / `closed_form`. The gate that would have foreclosed closed-form last-layer adaptation never fired, so that item is **unmeasured, not foreclosed**.
- A sweep of the model tree found **no further** silent-failure paths beyond those two. `set_activity_path_dropout` and `perturb_t4_carrier_numpy` are unused entry points whose primary paths work.

**Two corrections to items still in the live queue.**

- **CF1's Z4 sibling is not a valid generic-regularization control.** Under Z4 the identity is `post_pool([mean_feat, 0])`, so zeroing `mean_feat` leaves the T4 arm with an informative identity and the Z4 arm with `post_pool([0, 0])`, a constant. A positive `T4 − Z4` interaction is predicted by that asymmetry alone. Replace it with an information-matched control — dropout on the concatenated `[mean_feat, T4]`, or matched-magnitude noise on `mean_feat`.
- **CF1's mask is `[B, 1]`, i.e. per window, while `mean_feat` is a session identity.** Per-window dropout trains on an identity that flickers between windows of one recording, which is the exact construction B2 candidate v1 was rejected for. Adopt B2's session-epoch schedule before launching.

**One risk closed rather than opened.** No teacher leakage. The MC-Maze teacher is DANDI 000128 sub-Jenkins (2009, Shenoy/Stanford, `hand_vel`); SUA validation is DANDI 000688 sub-C (2015, Miller/Northwestern, `cursor_vel`). No shared animal, lab, year or dandiset. `clean_teacher.yaml` guards the FALCON M2 path and never guarded SUA. `fresh_common_teacher_fit = False` is decoder-initialization domain mismatch, not leakage.

**Subordinate document.** `JOINT_ENC_DEC_PROPOSALS_20260813.md` was written in parallel with this ledger and duplicates parts of it. This file and `HANDOFF_NEXT_CONTRIBUTION_STRATEGY_20260812.md` §5 are authoritative on priority; that document is retained for its derivations only. Note also an unresolved conflict between the two authoritative files: exposure reweighting is "cheapest CPU gate", rank 2, in the strategy table and "Held D, last" in §5.3 here.

---

## 4. Standing discipline

Four gates in this program have turned out to be mathematically incapable of doing their job: a `det(X'X)` gate
that could only pass, a falsification rule that could never fire, a Wilcoxon gate unattainable at `n = 3`, and
my own rank diagnostic that could not discriminate. **Before freezing any gate, construct one synthetic input
that must pass it and one that must fail it, and assert they give different verdicts.** Current aggregator
tests prove gates compute correctly; none proves they can act.

---

## 5. Root review and corrected execution route (2026-08-13)

**Status:** reviewed as an ideation record; **does not authorize a new GPU arm**.  The structural pivot is useful,
but several factual statements and several proposed gates must be corrected before this document can become an
execution contract.  The already frozen A2 and B1 contracts remain authoritative.

### 5.1 What survives the review

The strongest surviving idea is the **hidden-space carrier adapter**.  The current coupled path is

```text
E_i = post_pool([mean_h_i, T4_i]) in R^W
h_i = fc_in(x_i + E_i),
```

whereas the proposed path is

```text
h_i = fc_in(x_i + E_i^A) + P(T4_i),
```

where `E_i^A` is the matched Z4/activity identity. This correction is necessary: the bare proposal
`fc_in(x_i)+P(T4_i)` would delete activity identity while the W-add control retains it, confounding add site with
identity-route replacement.

Because `fc_in` is nonlinear, these are not algebraically equivalent.  This proposal also does **not** delete
the pretrained read-in, so the failed decoupled-K/V arm does not test it.  Importantly, a hidden adapter can be
implemented in the streaming student around the existing decoder; it requires a new source-training contract,
but it does **not** intrinsically require a new teacher checkpoint.  A CO-native teacher is a separate factor
and must not be introduced in the same first comparison.

The training-recipe concern also survives, but it is not one intervention:

1. the carrier-by-identity-distillation interaction is exactly the already frozen **B1** factorial;
2. MC-Maze versus CO-native teacher initialization is a separate teacher-domain experiment;
3. legacy window-count weighting versus equal-session sampling is a separate objective-distribution experiment.

These factors must be isolated.  A positive run that changes teacher, add site, and sampling together would be
an accuracy result with no defensible mechanism attribution.

### 5.2 Factual and inferential corrections

1. **A2 is now terminal.**  The three subject-shift interactions are `+0.230085`, `+0.187631`, and
   `+0.289681`; their mean is `+0.235799` with the frozen crossed seed-by-session bootstrap interval
   `[+0.100852,+0.371768]`.  All interaction and secondary gates passed in immutable aggregate SHA-256
   `5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc`.  This is an increase in the relative
   `T4-Z4` margin under the observed C-to-M shift, not an absolute external-performance improvement of the same
   magnitude.  The full 2x2 means are within `T4=0.574976`, `Z4=0.326008`; external `T4=0.341367`,
   `Z4=-0.143399`.  Quote the interaction together with all four means, never as an absolute T4 lift.
2. **Do not replace B1's frozen gate.**  Stage P is fold 0 and passes only if mean interaction is at least
   `+0.03` and all three seed interactions are positive.  Stage F uses the three predeclared fresh folds and
   passes only if its mean is at least `+0.03`, all three session means are positive, and all three seed means
   are positive.  Its crossed bootstrap interval is descriptive; an interval excluding zero is not a frozen
   terminal requirement.
3. **The teacher statement needs narrower wording.**  A2 loads the MC-Maze-derived decoder initialization even
   under `task_only`, but it does not use an explicit selected-T4 encoder warm-start.  The decoder accepts
   variable unit counts even though its teacher was trained in a fixed-unit regime.  Teacher/target mismatch is
   therefore a plausible contributor, not an already isolated cause.
4. **The session-weighting mismatch is real, but the existing support is path-specific.**  The M2 streaming
   implementation `streaming_calibration_exp/src/data/falcon_datamodule.py` defines `balance_sessions` at lines
   751--904, exposes it as `balance_session_batches` at line 1001, and passes it into the training sampler at
   lines 1497--1503; the ordinary M2 config currently fixes it to `false`.  It is an existing lever for the
   **M2 streaming source-training sampler only**, not validation sampling.  The same sampler core also has
   `window_budget_per_session`, but the ordinary M2 DataModule does not currently route that argument from its
   config.  More importantly, the SUA/A2
   path `sua_exploration/mc_maze/multisession_datamodule.py:805--848` has neither switch.  Therefore do not
   invent a new M2 sampler merely to test equal-session interpolation, but do not claim that fixed-budget or
   SUA/A2 balancing is already wired; either would need an explicit implementation and provenance audit.
5. **The H1 query-fitted carrier result is local, not a ceiling.**  The `-0.002859` intervention is explicitly a
   `LEAKAGE_DIAGNOSTIC_ONLY`, `not_a_strict_upper_bound` substitution into one frozen consumer.  It supports
   local insensitivity of that consumer to that substitution.  It does not prove estimator saturation or that
   better estimates cannot help a jointly retrained consumer.
6. **The T8 conclusion is bounded.**  `T8-T4=+0.003979` says that raw per-direction means did not improve the
   tested model, dataset, budget, and schedule.  It does not prove that the cosine fit is a universal sufficient
   statistic.

For cross-domain interventions, report both quantities rather than declaring either one universally primary:

- absolute external T4 performance and its change versus the matched parent, so a collapsing Z4 cannot create
  a false deployment win; and
- the matched T4/Z4 interaction, so a generic optimization improvement cannot be misreported as greater use of
  carrier content.

### 5.3 Canonical execution queue and held candidate ranking

B1, B2, and A1 now have settled scientific contracts. B1 Stage P and A1's one-cell development pilot are terminal;
C1/CF1/C2 remain held candidates whose ordering expresses information value, not intervening GPU queue entries.

| Order | Experiment | Minimal attributable design | Decision |
|---|---|---|---|
| 0 | Finish A2 | completed three-seed aggregate SHA `5b1459df...` | Terminal positive interaction; preserve its exact interpretation and receipts. |
| Terminal B1 | B1 distillation interaction | frozen `{T4,Z4} x {task_plus_y,task_plus_y_plus_E}` Stage P | Mean interaction `+0.0137`; mixed signs; below `+0.03`: `STOP_B1_NO_STAGE_F`. |
| 2, contracted | B2 robustness/safety | clean versus session-consistent carrier corruption, evaluated on T4/Z4/RS4/LS4 | Separate contract after B1's terminal stop; no substrate or GPU launch is authorized automatically. Opposite hypothesis to CF1. |
| Terminal A | A1 hidden-space adapter | logical `{W-add,H-add} x {T4,Z4}`, with sealed W arms, exact H/Z4 alias, and one fresh H/T4; TS4 is same-checkpoint attachment | Primary interaction `+0.0128 R²`; below preregistered `+0.03`: `PILOT_ROUTING_STOP`. Attachment control confirms use but no useful lift. |
| Held B | C1 teacher-domain ablation | teacher `{MC-Maze,CO-native}` x carrier `{T4,Z4}`, with W-add fixed | Requires source-only teacher compatibility; do not combine with A1. |
| Held C | CF1 carrier forcing | activity-path dropout `{p=0,p>0}` x carrier `{T4,Z4}` | Distinct accuracy hypothesis; requires an external-T4 endpoint and Z4 interaction control. |
| Held D | C2 sampling objective | M2 source-training sampling `{legacy,equal-session}` x carrier `{T4,Z4}` | No validation/SUA claim and no batch-level differentiable R2 loss. |

The A1 primary estimand is the interface-by-content interaction

```text
(H-add(T4) - H-add(Z4)) - (W-add(T4) - W-add(Z4)),
```

not only `H-add(T4)-W-add(T4)`. In the corrected minimal design H/Z4 is an exact structural alias of W/Z4, so
the numerical contrast reduces to H/T4-minus-W/T4 without deleting the full interaction interpretation. TS4
then asks whether correct unit attachment is necessary. The CPU contract froze `P(0)=0`, H/Z4 versus W/Z4
exact parity, unchanged shared base/teacher semantics, identical provenance, and immutable artifact handling.

A10 does not authorize or foreclose A1: it concerns target-session weight-update headroom, while A1 changes
source training and retains zero-backprop deployment. The A1 pilot is terminal routing evidence; see the section
5.3 ledger for its decision.

### 5.4 Routes not promoted to the next GPU queue

- **Activity-path dropout is not replaced by carrier corruption.**  They apply opposite training signals.
  Activity-path dropout asks the model to continue using the carrier when activity-derived identity is
  unreliable; session-consistent carrier corruption asks it to ignore a bad carrier and fall back to activity.
  The latter is a safety/robustness question and, given A2's negative external Z4 means, must not be presented as
  a cleaner route to the former accuracy objective.  The current dropout is training-only, which is ordinary
  stochastic regularization rather than automatically an invalid design, but it does not literally reproduce
  the deployment condition.  A valid CF1 experiment must therefore use the matched
  `{T4,Z4} x {p=0,p>0}` factorial: the Z4 sibling tests generic regularization, while success additionally
  requires an absolute external-T4 improvement and within-subject T4 non-inferiority.  Keep this route queued
  held after B1's terminal stop rather than declaring it either solved or superseded.
- **R2-native training is held.**  Batch-level R2 has unstable and batch-dependent denominators and is not the
  same object as the final unweighted session-mean R2.  Equal-session sampling with ordinary MSE is the clean
  first test.
- **Estimator-noise augmentation remains lower priority.**  It should be revisited only with a retrained
  consumer and a T4/Z4 sibling, not justified by the frozen H1 oracle substitution.
- **Depth, 10-ms/W100, and a CO-native teacher plus H-add combined arm remain high-cost holds.**  They change too
  many contracts to serve as the next diagnostic.

The main conceptual result of this review is therefore narrower and more useful than the original synthesis:
the next plausible accuracy gain is not another descriptor column. With B1 terminal, it is either the held CF1
carrier-specific training interaction or a separately attributable teacher/sampling correction. B2 remains a distinct safety
route, while C1, C2, and CF1 remain held.

**Workspace cold archive.** `/mnt/data/SPINT_cold_archive/2026-08-13/` now contains the initial `8.90 GiB` plus
a sealed-batch source-unique `81.9967 GiB` migration; hard-link deduplication added `74.5878 GiB` of new archive
storage. The `49.940 GiB` `streaming_calibration_exp/logs/train` tree was subsequently archived after
path-compatibility smoke tests, with its original absolute path retained as a symlink. Root free space is
approximately `208 GB`; pointer-plus-SHA manifests preserve recovery without changing authoritative artifacts.
