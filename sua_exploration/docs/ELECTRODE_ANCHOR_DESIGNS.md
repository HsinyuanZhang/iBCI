# Electrode-identity designs on a T4 substrate (A / D / C)

**Status: design D screen completed and judged `ineffective`; C and A remain
implemented-but-unrun (updated 2026-07-29).**

This document is maintained by the implementing agent, not the project maintainer. It does
not modify and is not a substitute for `E3_E4_ENCODER_PROGRAM.md`, `CURRENT_RESULTS.md`, or
`MEASUREMENT_PROTOCOL_V4.md`.

## 0. Why the premise changed

The original electrode-identity hypothesis (`UNIT_SIDE_FEATURE_ABLATION.md` section 6, F3/FS3)
was motivated as "provide a cross-session-stable anchor" for the streaming identity, on the
assumption that nothing else in the encoder carried a stable, session-independent signal.

E3 changed that assumption. Measured directional tuning fitted from the first 50 rewarded
calibration-pool trials (`side_feature_pool_size=50`;
`E3_E4_ENCODER_PROGRAM.md` section 1) gives a large gain over the no-side-feature baseline:

```
F0 (B3 baseline)                    0.3140
T4 (cosine tuning, 4 dims)           0.5667   T4 - F0 = +0.2528 (6/6 sessions, 3/3 seeds)
```

with shuffled controls (TS4/TS8) sitting exactly at baseline, i.e. the T4 gain is content, not
architecture width. A follow-up classical-linear-decoder control
(`scripts/linear_decoder_control_dandi688.py`, `results/linear_decoder_control.json`) further
shows the T4/T8 gain is **not** simply "linearly decodable tuning information" -- the best
closed-form linear/population-vector decoder fit on the 30 B3-selected calibration trials
reaches only ~0.31 (essentially F0), far below T4's 0.5667. This comparison is not
information-matched: T4 uses direction/rate information from the 50-trial pool, whereas the
classical decoder uses 30 trials. So T4 already supplies a live, per-session, richly-used
functional identity signal, but the linear control cannot by itself attribute the gain to the
network rather than the larger supervised information budget.

**The question this document's designs answer is therefore not "does electrode identity beat
B3" (that measures something no longer of interest) but "given we already have measured
tuning, does electrode identity add anything on top of T4?"** Every arm below is built ON a T4
substrate; the baseline for every comparison is **T4**, not B3/F0. The primary pairs are
`T4gate - T4` and `T4gate - T4gate_shuffled` (design D's own dimension-matched control), never
`* - F0`.

## 1. Three designs, all layered on T4

All three must be, and are (see section 3), **exactly equivalent to plain T4 at
initialization** -- the comparison must start from an identical function, so any measured
difference is attributable to what the electrode mechanism *learns*, not to a different
function at step 0.

Each design reuses `mc_maze.multisession_datamodule.electrode_ids_from_units()` for the raw
per-unit electrode index (never reimplemented) and requires its own dimension/parameter-matched
shuffled control (electrode ids permuted along the unit axis within each session, fixed
recorded seed via `permute_electrode_ids`) -- otherwise the "width vs content" confound
`UNIT_SIDE_FEATURE_ABLATION.md` section 6 already documents for the original F1-vs-FS defect
would simply reappear here.

### Design D -- per-electrode reliability gate (`B3SEG`, implemented and runnable first)

```
E_i  <-  E_i * (1 + tanh(g[electrode(i)]))     g: num_electrodes learnable scalars, init 0
```

`E_i` is T4's own `post_pool` output (`SideFeatureEarlyPoolEncoder` with `side_dim=4`, i.e.
plain T4 -- `electrode_embed_dim` is always 0 for this design; nothing is concatenated).
`g=0` at init makes the gate factor exactly 1 for every unit, so this encoder is functionally
identical to plain T4 until training moves `g` away from zero.

The mechanism is judged *least likely to be subsumed by tuning*: it is orthogonal to tuning
content by construction -- it encodes "how much should this recording SITE be trusted"
(chronic noise, impedance drift, dead channels), a property of the electrode that a 30-trial
cosine-tuning fit cannot express on its own, however good that fit is.

**Parameters:** `num_electrodes` (measured 94 for this dataset's 27-session train split via
`compute_electrode_vocab_size` -- see section 4 for why this is not the nominal 96-channel
figure). Measured total: T4 = 18,290 params; T4+gate = 18,384 params (+94, i.e. exactly
`num_electrodes`).

Implementation: `ElectrodeGateEarlyPoolEncoder` in
`streaming_calibration_exp/src/models/components/streaming_encoders.py`, variant string
`B3SEG`. Overrides only `finalize_identity`'s return value (multiplies T4's own output by the
gate); `reset_stream`/`push_trial` are inherited unchanged from `SideFeatureEarlyPoolEncoder`.

### Design C -- electrode-anchored identity prior (`B3SEA`, implemented, not yet run)

```
E_i  <-  psi(pooled_i, tuning_i) + alpha * M[electrode(i)]     M: [num_electrodes, W] learnable; alpha: learnable scalar, init 0
```

`psi(pooled_i, tuning_i)` is again plain T4's `post_pool` output. `M` is a
`[num_electrodes, window_size]` learnable table (`nn.Embedding`, zero-initialized for a clean
start), `alpha` a single learnable scalar initialized to 0. `alpha=0` alone guarantees
`E_i` equals plain T4 at step 0, independent of `M`'s own values.

This is the design that literally implements the original "anchor the drifting units to a
stable substrate" hypothesis. Unlike design A's concat embedding (which the frozen decoder's
attention weights must learn to interpret from scratch, with no guaranteed relationship to
T4's own scale), design C's anchor is *additive in the same space E_i already lives in* -- a
small `alpha` acts as a direct regularizer that pulls a noisy 30-trial tuning estimate toward
a stable, session-independent electrode prior without discarding T4's own per-session read.

**Parameters:** `num_electrodes * window_size + 1` (measured 94*50+1 = 4,701 on top of T4;
T4+anchor = 22,991 total).

Implementation: `ElectrodeAnchorEarlyPoolEncoder`, variant string `B3SEA`.

### Design A -- learned electrode embedding (`t4e` on plain `B3S`, degenerate control for C)

`nn.Embedding(num_electrodes, ELECTRODE_EMBED_DIM=8)` concatenated at the psi (`post_pool`)
input alongside T4's own 4 features (`post_pool_side_dim("t4e") = 4 + 8 = 12`), with the
corresponding `post_pool[0].weight` columns zero-initialized -- exactly F3's mechanism, with
T4 playing the role F2's waveform scalars played for F3. No new encoder class: this is plain
`SideFeatureEarlyPoolEncoder` (variant `B3S`) with `side_dim=4, electrode_embed_dim=8`,
reached via `--side_features t4e`.

This is the simplest of the three and is judged *most likely to be subsumed by T4* --
concatenation hands the frozen decoder raw material it must learn to use, with none of design
C's built-in "pull toward a stable prior" structure or design D's orthogonal reliability
semantics. It is kept implemented as design C's **degenerate control**, not as a headline arm
of its own: if C's gain (if any) is not distinguishable from A's, the additive/regularizing
structure of C is not what is doing the work, plain concatenation would suffice.

**Parameters:** electrode embedding `num_electrodes * 8` + extra `post_pool[0]` input columns
`8 * hidden_dim` = measured 1,264 on top of T4 (T4+embed = 19,554 total).

## 2. Naming / token map

| Design | `--variant` | `--side_features` (real) | shuffled control |
|---|---|---|---|
| D (gate) | `B3SEG` | `t4gate` | `t4gate_shuffled` |
| C (anchor) | `B3SEA` | `t4anchor` | `t4anchor_shuffled` |
| A (embed) | `B3S` | `t4e` | `t4e_shuffled` |

All six tokens resolve `base_feature_group(token) == "t4"` (`mc_maze/unit_side_features.py`):
every design reuses T4's own cosine-tuning fit unchanged; only the electrode mechanism (or
absence of one) differs between them. The three `*_shuffled` tokens are **electrode-shuffle**
controls (`is_electrode_shuffle_control` -- ids permuted, exactly like FS3), never
**feature-shuffle** controls (`is_feature_shuffle_control` -- T4's own tuning values are never
permuted for any of these six tokens).

`train_variant_dandi688.py` enforces the variant/side_features pairing structurally: `B3SEG`
requires `side_features in {t4gate, t4gate_shuffled}`, `B3SEA` requires
`side_features in {t4anchor, t4anchor_shuffled}`, and `B3S` refuses the gate/anchor tokens (a
plain B3S build would silently ignore their mechanism rather than raising, which is worse than
refusing outright).

`mc_maze/unit_side_features.py` distinguishes two electrode gates, deliberately split from the
single, narrower F3-era `uses_electrode_embedding`:

- `uses_electrode_embedding(group)` -- narrow: True only for groups whose mechanism is the
  concat-at-input embedding (`f3`, `fs3`, `t4e`, `t4e_shuffled`). Still gates
  `post_pool_side_dim()`'s `+ELECTRODE_EMBED_DIM` and `train_variant_dandi688.py`'s
  `electrode_embed_dim` -- unchanged behavior for every pre-existing caller.
- `uses_electrode_ids(group)` -- broad superset: True for every group above **plus**
  `t4gate`/`t4gate_shuffled`/`t4anchor`/`t4anchor_shuffled`. This is the gate data-loading call
  sites (`Dandi688MultiSessionDataModule.setup`, `eval_adaptation_dandi688.attach_side_features`,
  and `train_variant_dandi688.py`'s `num_electrodes` computation) now use, so design D/C's own
  tables receive electrode ids and a correctly-sized vocabulary even though they never build an
  `nn.Embedding` at the psi input.

## 3. Zero-init-equivalence guarantee (tested)

`streaming_calibration_exp/tests/test_side_feature_encoder.py` (parametrized over both
`ElectrodeGateEarlyPoolEncoder` and `ElectrodeAnchorEarlyPoolEncoder`) asserts, among other
properties:

- **zero-init equivalence to plain T4** -- loading a plain-T4 state dict into either encoder
  and running `forward_batch` produces byte-identical output to T4 itself;
- **shape independence from N** (unit count);
- **permutation invariance** when the unit axis of `calib`/`side_features`/`electrode_ids` is
  permuted together (units carry their own electrode id with them, like F3);
- **output actually depends on electrode assignment** once the gate/anchor parameters are
  non-zero (permuting only `electrode_ids`, with unit order and T4 content held fixed, changes
  the output) -- otherwise the mechanism would not be reading electrode identity at all;
- **out-of-range electrode id raises** (`ValueError`, `[0, num_electrodes-1]` bound) for every
  value outside range, including a large out-of-range id -- never silently clamped, and
  checked explicitly before indexing rather than relying on backend-dependent (CPU raises,
  CUDA does not) out-of-bounds indexing behavior;
- **measured parameter counts** match the formulas in section 1 exactly (`num_electrodes` for
  D; `num_electrodes*window_size + 1` for C).

## 4. Array-specificity -- do not bury this

**All three designs make the trained model array-specific.** `g` (design D), `M` (design C),
and the electrode embedding table (design A) are indexed by electrode id, a property of one
physical implant. None of these tables can transfer to a different implant or subject without
retraining from scratch -- there is no notion of "electrode 37 means the same thing" across two
different arrays, even of the same nominal channel count.

This is an acceptable cost for a per-patient-calibrated chip (the deployment story this project
targets), but it is a real cost, and it partially cuts against the "no fixed index lookup"
framing this project's non-electrode encoders (B3/B3T/B3A/B3S-with-tuning-only) otherwise get
to claim. The distinction that actually matters, and that must not be elided:

> **Electrode index is stable across sessions of the same array; sorted-unit index is not.**

That asymmetry -- not "indices are fine" or "indices are bad" in the abstract -- is the entire
reason an electrode-indexed table is a coherent design at all (a fixed table keyed by something
that drifts session to session, like a spike-sorted unit id, would be meaningless), and it is
also exactly the boundary of what that table can promise: stability within one array's
lifetime, nothing more.

`compute_electrode_vocab_size` (used by all three designs, unchanged from F3) computes
`num_electrodes` **dynamically** from the training sessions' own NWB electrode tables --
measured 94 for this dataset's 27-session train split (`train+val` also gives 94), not the
nominal 96-channel Utah array figure the initiating design brief used for illustration. No
code in this document's designs hardcodes 96; every out-of-range check, embedding table, and
gate vector is sized from the real, per-split `num_electrodes` at construction time.

## 5. Staged execution -- D only, run by the maintainer

Per instruction, **only design D is wired into a runnable screen**; C and A stay
implemented-but-unrun until D reports.

```
scripts/run_t4_gate_screen.sh --max_epochs N --seeds S1,S2,...
```

is a work-queue scheduler (claim-next-job-from-a-shared-queue over 2 GPUs, modeled on
`run_b3t_confirmation.sh`'s scheduler -- **not** the lockstep-pairs pattern
`run_e3_tuning_ablation.sh`/`run_electrode_ablation_f3.sh` use, which idles a GPU whenever one
job in a pair finishes first). `--max_epochs`/`--seeds` are required with no defaults, same
discipline as every other screen in this project (`E3_E4_ENCODER_PROGRAM.md` section 0: do not
silently guess an epoch budget or seed count).

Matrix: `T4` / `T4+gate` (`t4gate`) / `T4+gate_shuffled` (`t4gate_shuffled`) x
`--seeds`. All three groups are trained fresh by this screen (not reused from
`e3_tuning_ablation`'s T4 artifacts) so the screen is fully self-contained -- three seeds at
`--seeds 42,43,44` is 9 runs total.

The maintainer subsequently launched the frozen 9-run matrix on 2026-07-27. All 9 runs
completed with `eval_rc=0`; the authoritative post-hoc aggregate was written on 2026-07-29
to `results/t4_gate_screen/aggregate.json`.

Aggregation: `scripts/aggregate_t4_gate_screen.py --seeds ... --effective_mean_delta ...`,
built on the same shared four-state gate (`aggregate_side_feature_ablation_v2.classify_pair_
verdict`/`classify_group_verdict`, `MEASUREMENT_PROTOCOL_V4.md` section 4.2b/4.2c: paired
sigma, "ineffective" means `mean_delta + 2*sigma_delta_paired < threshold`, group verdict is
OR'd across a group's pairs for "ineffective") every other aggregator in this repo uses.
Primary pairs: `T4GATE_minus_T4` and `T4GATE_minus_T4GATE_SHUFFLED`, dimension-matched only
(never `T4GATE - F0`).

### 5.1 Final design-D result

| Group | mean R² |
|---|---:|
| T4 | **0.566749** |
| T4GATE | 0.555931 |
| T4GATE_SHUFFLED | 0.561363 |

| Pair | mean delta | paired SE | positive sessions | verdict |
|---|---:|---:|---:|---|
| T4GATE − T4 | **−0.010817** | 0.004873 | 1/6 | **`ineffective`** |
| T4GATE − T4GATE_SHUFFLED | **−0.005432** | 0.005779 | 1/6 | **`ineffective`** |

Both pairs have `mean+2SE < +0.03`; the group verdict is therefore
**`ineffective`**. A fixed, learned per-electrode scalar does not add useful reliability
information on top of per-session functional T4, and slightly worsens the observed mean.

This result changes the staged decision:

- do **not** run C or A merely because their code exists;
- only revisit an electrode-indexed table if a new diagnostic identifies a quantity that is
  stable within an array but absent from T4 (the failed static scalar is not such evidence);
- prioritize T4 component attribution, fit uncertainty, and activity–T4 interaction, which
  remain session-functional rather than array-lookup mechanisms.

## 6. Data isolation (unchanged from every other screen in this project)

- Only train + validation sessions are ever loaded for spikes/behavior/trials;
- the 6 held-out test sessions
  (`sub-C_ses-CO-20151113/-20151116/-20151117/-20151119/-20151120/-20151201`) are only ever
  touched for their names and NWB unit-table row counts (`nwb_unit_count`), never spike,
  behavior, or trial data;
- no formal-test receipt is created, modified, or deleted;
- `results/e3_tuning_ablation/`, `results/b3t_confirmation/`, `results/convergence_swa_v1/`,
  `results/e4_encoder_variants/`, `eval_epoch_window_dandi688.py`,
  `eval_epoch_window_generic_dandi688.py`, `swa_utils_dandi688.py`, `run_b3t_confirmation.sh`
  are all unmodified by this work.
