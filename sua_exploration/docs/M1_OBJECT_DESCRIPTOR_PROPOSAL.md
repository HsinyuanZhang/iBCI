# M1 object-conditioned per-channel descriptor (pre-registered proposal)

**Status:** **superseded — do not implement this O4/object formulation.**

The CPU semantic audit showed that every deployment M=10 calibration block has only
`tgt_obj=Sphere`, while `obj_id` is deterministically locked to four target locations.  O4 is
therefore not identifiable as an object descriptor.  The repaired audit nevertheless found a
strong categorical-profile signal, so the surviving candidate has been renamed **D4** and is
governed by:

- `docs/M1_D4_SEMANTIC_AUDIT_PROTOCOL.md`;
- `results/m1_d4_semantics_v2/audit.json` (SHA-256
  `5d56e893ce7f9695766fd2d01a8786abf4be62aca0532eb3916105d2b034d160`);
- `docs/M1_D4_MINIMAL_GPU_PILOT_PROTOCOL.md`.

The remainder of this file is retained as historical proposal text.  Its O4 naming, object
interpretation, and label-shuffled OS4 control are not authorized.  D4 uses a categorical
calibration profile, and DS4 shuffles complete fitted feature rows across channels.

## Motivation

The read-only mechanism audit in `sua_exploration/results/m1_t4_mechanism_v1/` shows that M1
trial-mean EMG and movement-window neural rates are better explained by `obj_id` than by a
direction cosine design, while T4 fits direction.  `obj_id` is present in every M1 deployment
calibration NWB and all four levels appear within the frozen first-10 chronological prefix.

## Descriptor definition

**Name (working):** O4 — object-conditioned per-channel baseline descriptor.

**Calibration legality:** identical to T4.  Read `obj_id` only from the held-in (fit) or
held-out-calib (explicit test replay) calibration NWB trials table.  No query/hidden labels.

**Support prefix:** frozen chronological first 10 calibration trials (`calibration_n_trials=10`,
`random_calibration=false`), matching native-MUA T4/M1.

**Per-channel fit:** for each neural channel, pool spike counts over the same valid
calibration-trial prefix T4 uses (un-interpolated in-trial counts, no partial exposure beyond
the T4 exposure rule).  For each trial, use the trial's `obj_id` label.  Estimate four object
baselines by one-way ANOVA / cell-mean coding:

\[
r_{i,k} = \frac{1}{|T_k|}\sum_{t \in T_k} \frac{\text{spike\_count}_{i,t}}{\text{trial\_length}_t}
\]

where \(T_k\) is the set of support-prefix trials with `obj_id = k`.

**Side-feature vector per channel:** `[μ_1, μ_2, μ_3, μ_4]`, the four object-conditioned mean
rates (Hz) for the `obj_id` levels observed in M1 calibration data (`{1,2,3,4}`).

**Dimensionality:** `4 × n_channels` raw descriptor values, collapsed to `4` side-feature
dimensions per channel before batching (same layout contract as T4's four scalars per channel:
here the four entries are object means, not `[a, c, m, b]`).

**Normalization:** reuse the existing train-fold calibration z-score path used for T4 side
features (`fit_train_t4_stats` analogue for O4 statistics on the training sessions only).

## Fair comparison arms

| Arm | Side feature | Purpose |
|-----|--------------|---------|
| F0 | none | baseline decoder |
| T4 | direction cosine `[a, c, m, b]` | current production descriptor |
| O4 | object means `[μ_1, μ_2, μ_3, μ_4]` | proposed factor-aligned descriptor |
| OS4 | O4 with per-channel object labels permuted within the support prefix | shuffled control analogous to TS4 |

OS4 must permute labels using the same deterministic seed plumbing as TS4
(`side_feature_shuffle_seed`, row permutation scoped to calibration trials only).

## Pre-registered endpoint

**Primary (deployment-faithful):** clean held-in-calib post-support replay already receipted
under `sua_exploration/results/m1_heldin_disjoint_replay_v1/` — same LOSO plumbing,
`query_start_trial=10`, native MUA, frozen first-10 support.  Report `O4−F0`, `O4−T4`, and
`O4−OS4` with the existing aggregate script conventions (no new pass/fail gate language).

**Secondary (support-budget ablation, held-in-calib only):** sweep `calibration_n_trials ∈
{10, 40, 90, 200}` on the constructible post-support endpoint to separate "wrong factor" from
"too noisy."  At 10 trials with four objects there are only ~2.5 trials per object, so O4
estimates will be high-variance; larger budgets on held-in-calib are explicitly for
identifiability, not for changing the deployment 10-trial contract.

## Expected interpretation boundaries

- O4 is a linear trial-mean object baseline, not a claim about maximal decodable EMG information.
- A positive `O4−T4` on the 10-trial endpoint supports factor alignment, not optimality.
- Failure of O4 at 10 trials with success at 90/200 would implicate noise/support, not object
  irrelevance.

## Non-goals

- No change to `streaming_calibration_exp/src/data/falcon_datamodule.py` in this proposal pass.
- No GPU training or hidden EvalAI query in the proposal phase.
- No modification of existing scored trees under `m1_heldin_disjoint_replay_v1/`.
