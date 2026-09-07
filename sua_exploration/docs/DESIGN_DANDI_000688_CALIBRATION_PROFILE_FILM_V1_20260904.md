# DANDI 000688 Calibration-Profile FiLM V1

Status: `FROZEN_SOURCE_ONLY_SCREEN__GPU_AUTHORIZED__FORMAL_TEST_FORBIDDEN`

## 1. Question

Can the calibration-profile FiLM mechanism that improved M2 officially and
H1 locally transfer to DANDI 000688 without changing the decoder, activity
encoder, carrier, or query inputs?

This is a mechanism screen on the frozen 27 train / 6 validation split. It is
not an EvalAI submission and it must not open the six formal-test sessions.

The older DANDI `confidence_film` result does not answer this question. That
route conditioned on T4 fit residual variance and direction-design condition.
This route conditions on a per-unit contrast between two task states, matching
the semantic role used by M2 and H1.

## 2. Frozen parent and scope

- Data: DANDI 000688, `sub-C`, center-out, sorted SUA.
- Roster: `subc_co_27_6_strict_train_val_manifest.json`, exact 27/6/6.
- Formal-test session names remain strings only; their NWB paths must not be
  resolved or opened.
- Parent models: the available three-seed M30/T4@30 B3S mainline final-epoch
  checkpoints. They are strong source-validation anchors, but they are not the
  later selected M30/T4@50 V9 checkpoints, whose exact bytes are absent in this
  workspace.
- Activity support is chronological rewarded trials `[0:30]`.
- T4 is fit on the same chronological first 30 rewarded trials.
- Every training and validation query window is contained in rewarded trials
  `[50:]`. Trials 30--49 are an unscored guard interval.
- No target-session update, no test-time adaptation, and no continual memory.

This parent choice makes the first run a clean cross-dataset mechanism screen.
A positive result licenses a later M30/T4@50 product successor; it does not
retroactively claim to improve the missing V9 checkpoint.

## 3. Calibration profile

For profile horizon `M in {10,30}`, use only bins contained in the first M
chronological rewarded trials. Neural values are the native 20 ms sorted-SUA
count bins used by the DataModule. Velocity is the DataModule's source-frozen
standardized 2-D cursor velocity.

Let speed be the Euclidean norm of the two velocity coordinates. Within a
session prefix, define low state as bins at or below the 25th percentile and
high state as bins at or above the 75th percentile. For every unit compute:

1. high-state mean minus low-state mean;
2. `log1p(high mean) - log1p(low mean)`;
3. low-state population standard deviation;
4. high-state population standard deviation.

Robust-z each column across units with median and
`max(1.4826 * MAD, 1e-6)`. The governing mask is `[1,1,0,0]`, matching the
mean-only M2/H1 result. The standard-deviation columns are receipted but zeroed.

The low/high-speed definition is deliberate. A preliminary source-only audit
found that `target_on_time -> go_cue_time` is about 1 ms in multiple older
source sessions, so a literal delay/reach split is not stable across this
dataset. The speed-state definition is already the H1 instance of the common
"two calibration regimes" abstraction.

Source-only constructibility audit on all 33 allowed sessions found:

- all profiles finite;
- units per session 38--91;
- minimum robust scales for the four columns approximately
  `[0.0207, 0.0182, 0.1016, 0.1067]`;
- median maximum absolute within-session correlation between either active
  profile column and T4 is 0.326, maximum 0.597.

Thus the profile is neither degenerate nor a literal copy of T4, although T4
already contains modulation/baseline information and a null result remains a
plausible boundary condition.

## 4. Frozen operator

Let `phi` be the frozen B3S pre-pool map, `psi` its frozen post-pool map,
`c` the frozen normalized T4 carrier, and `q` the masked profile.

```
h = mean_i phi(activity_i)
[gamma, beta] = Linear_2( ReLU(Linear_1([c, q])) )
h_film = (1 + gamma) * h + beta
identity = psi([h_film, c])
prediction = frozen_decoder(query_neural, identity)
```

`Linear_1: 8 -> 8`; `Linear_2: 8 -> 128`. The final layer is initialized to
exact positive zero. Total trainable parameters are 1,224. B3S `phi`, `psi`,
T4, and the entire decoder are frozen. Before the first optimizer step, every
FiLM arm must reproduce the native identity and prediction bitwise.

Only the FiLM MLP is optimized with last-bin task MSE. Frozen modules remain in
evaluation mode so there is no dropout. Learning rate is `3e-4`, Adam has no
weight decay, batch size is 32, and training lasts exactly 12 epochs.

## 5. Same-GPU arms

All arms are byte-identical copies of one zero-initialized FiLM template and
reuse one frozen B3S/decoder, one DataModule, one query batch transfer, and one
cached frozen pre-pool mean per session.

- `CP10`: train with the real first-10 profile.
- `CP30`: train with the real first-30 profile.
- `SHUFFLE10`: train with a frozen nonidentity row permutation of the first-10
  profile while keeping T4/activity/query rows aligned.
- `EMPTY`: train with the profile set to exact positive zero; T4 remains in the
  FiLM context. This is the parameter/capacity control.

Each source session contributes at most 1,024 deterministic Q50 windows per
epoch. Indices are sampled without replacement from a route-owned seed domain,
then sorted. The same resident batch is presented to all four arms before the
next batch is loaded.

## 6. Evaluation matrix

On the six validation sessions, score every Q50 window and report native plus
each trained head under:

- real first-10 profile;
- real first-30 profile;
- exact-zero profile;
- the frozen row-shuffled profile at its matching horizon.

The primary cells are `CP10@M10`, `CP30@M30`, `SHUFFLE10@M10`, and
`EMPTY@ZERO`. Cross-horizon `CP10@M30` and `CP30@M10` are diagnostics for the
small-versus-large profile boundary.

Primary estimands use paired per-session R2 deltas against the exact native
zero-init anchor on identical windows.

## 7. Decision law

Seed 42 is the first gate.

- Noninferior: mean delta >= -0.005 and worst session >= -0.030.
- Positive candidate: mean delta > 0, at least 4/6 sessions positive, and mean
  delta exceeds both `EMPTY` and `SHUFFLE10` at the matching M.
- Solid source signal: mean delta >= +0.010, at least 4/6 positive, and both
  controls have mean delta <= 0.

Seeds 43 and 44 run only if either CP10 or CP30 is noninferior at seed 42.
This gate is for compute allocation, not a scientific significance claim.

Interpretation:

- CP10 positive and CP30 null/noninferior supports the proposed small-sample
  boundary.
- Both null but noninferior supports safe structural portability, not efficacy.
- EMPTY or SHUFFLE matching FULL means extra head capacity, not profile
  semantics, explains the result.
- Any material negative closes DANDI CP-FiLM under this frozen substrate.

## 8. Forbidden claims and actions

- Do not call this a T4@50 or V9 champion improvement.
- Do not use the formal-test six or external-15 sessions.
- Do not tune the profile definition, mask, rank, learning rate, epoch count,
  gates, or query horizon after seeing validation R2.
- Do not update the base encoder, T4 normalizer, decoder, or target session.
- Do not submit to EvalAI from this route.

