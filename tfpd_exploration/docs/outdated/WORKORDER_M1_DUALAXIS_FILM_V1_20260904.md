# Work Order: M1 Dual-Axis FiLM V1 (hold/reach × direction, S1 structure)

Date: 2026-09-04
Status: design authorized; implementation + local validation; **no submission without user confirmation**
Rev3 (post second external review, 09-04): carrier dims ENTER the FiLM context MLP
(581801's t4_plus_contrast structure — it beat contrast-only by +0.008 with a healthier
shuffle gap) AND stay raw-concat into post_pool; static-anchor gate restored as a
precondition (docker-recovered weights + zero-init FiLM must bit-exactly reproduce the
carrier_k_later_day_v1 chrono-k4 baseline); training config pinned (Adam 1e-4, 12 epochs,
256 windows/session, batch 32, seed 42 — explicitly not 3e-4); new Task 0.5 (phase features
from RAW 20ms counts + trials table — never from the interpolated [trials,1024,N] encoder
array); Surface B third mismatch documented (h-distribution shift is the main pressure
source); training/compute moved to CPU (M1 plan.py GPU0_ALLOWED=False); 3/3-pass wording
softened to "consistent with generalization"; official-transfer estimate 60–70% → ~50%.
Rev2 (post external review, same day): validation surface fixed (later-day M4/6 proxy + held-in
sanity; no M10-matched local held-out exists); main arm = **phase-only**, direction demoted to
pre-registered ablation; three zero-cost null arms (zero / session-constant / row-shuffle) added
to the gates; "carrier-T4" renamed **rSyn3 carrier** with direction-redundancy warning; one
confirmation seed allowed; "held-in non-negative" demoted to sanity disclosure.
Lineage: ports the officially validated M2 recipe (581801, HO 0.3203: per-unit task
contrast → 1.2k-param FiLM → cached identity) to M1, chosen for its metadata-rich
calibration recordings (audited this session: 414 trials/session with full phase
timing + target locations).

## 1. Baselines (official, fetched 09-04)

| Submission | Config | HO R² | HI R² |
|---|---|---:|---:|
| **581727** | freeze rSyn3 top4 (**M1 champion**) | **0.6396** | 0.7635 |
| 581736 | acyc rSyn3 top4 | 0.6238 | 0.7611 |
| 581665 | B3S rSyn3 M10 | 0.6225 | 0.7622 |
| 581660 | B3 M10 | 0.6087 | 0.7585 |

M1 R² base (0.64) is far above M2's (0.29): expected absolute increments are
smaller; gates are set accordingly.  Primary comparison anchor = **581727**.

## 2. What is being built

One additive package `tfpd_exploration/submissions/…` + training cell that:

1. Takes the M1 champion model (581727's weights, B3S id encoder + coupled
   decoder, everything frozen except the new module);
2. Adds a **dual-axis contrast context** and an **S1 feature-FiLM** on the
   pooled B3S features, zero-initialized (identity at init, the M2 anchoring
   philosophy);
3. Trains ONLY the FiLM (≈1.5k params) on the 4 held-in sessions;
4. Exports a cached-identity payload — deployment identical to 581727 except
   identities pass through the trained FiLM.

`IsTestTimeAdaptive=false`, label budget unchanged (M10 D-opt-k4 carrier law
kept as-is — do not change two variables).

## 3. Dual-axis contrast (the recipe's input, per unit n)

**Axis 1 — phase (MAIN ARM's only contrast)**: using the trials table's own
timestamps, per trial t: hold bins = `[start_time, gocue_time)`, reach bins =
`[gocue_time, contact_time]`.  Per-unit hold-mean / reach-mean rate over the
calibration horizon → `phase_delta = reach_mean − hold_mean`,
`phase_logratio = log1p(reach_mean) − log1p(hold_mean)`.  This axis is
reliable: every trial contributes both phases with many bins.

**Axis 2 — direction (ABLATION ONLY)**: target location `tgt_loc` in degrees,
estimated from the same 10 trials the deployment carrier uses.  Per unit:
cosine regression (3 params) for preferred angle φ_n, then
`dir_delta = mean rate (|wrap(θ_t − φ_n)| < 90°) − mean rate (≥ 90°)` and
`dir_depth = a_n`.  NOISE DISCLOSURE: ~5 trials per group per unit — this
axis's estimation noise likely exceeds M2's std features (which were already
rejected as noise there); it is an ablation, never the main conclusion.

**Terminology fixed**: the champion's carrier is the **rSyn3 carrier** (EMG
relu + NMF3 + ridge; trial SELECTION uses dopt_tgt_loc k=4).  It is not T4
and not neural direction tuning.  REDUNDANCY WARNING: rSyn3 already encodes
unit↔EMG synergy relations that depend strongly on movement direction — the
direction axis risks re-entering the same pathway twice (the same reason
"hold into T4" was rejected on M2).

FiLM input (main arm) = `[rSyn3 carrier (4) ‖ phase contrast (2)] = [N,6]`
per unit; ablation arm = `[N,8]` (+ direction dims).  **Both context paths,
the 581801 layout**: the full 6-dim (carrier + phase) goes through the context
MLP to become γ/β (on M2, t4_plus_contrast beat contrast-only by +0.008 with
a healthier shuffle gap), AND the carrier dims additionally pass raw into the
post_pool concat — dual-path, not either/or.

## 4. Network structure (S1, frozen-substrate FiLM)

```
seed/trial rows → pre_pool (frozen) → per-trial feats → mean over trials → h [N,H]
c = [rSyn3-carrier4 ‖ phase2] [N,6]
γ, β = Linear(6→8)+ReLU → Linear(8→2H), zero-init        # the ONLY trainable module
h' = (1+γ)⊙h + β
E = post_pool( cat[h' ‖ carrier4] )   # carrier: in the context MLP AND raw concat
ŷ = decoder(x, E) / 5.0-equivalent    # frozen
```

**Pooling position (explicit)**: this design does NOT use post-MLP (late)
pooling.  The trial-mean happens on pre_pool features — BEFORE the frozen
post_pool — identical to the champion's pooling site; the FiLM only modulates
the already-pooled feature.  Every late-pooling / row-shape variant tested on
M2 (J-MEAN −0.032; LP_cross; trial-shaped TTA −1.69) changed the pooling
position or the row definition, and all failed; S1 deliberately stays in the
proven regime.  The only MLP touching the contrast is the confidence context
(6/8 → γ/β), which acts on the side vector, never on activity rows.

Structural anchors: zero-init γ/β ⇒ before training the pipeline is bitwise the
champion; calibration rows are never evicted/modified; no query-time updates
(`IsTestTimeAdaptive=false`).

## 5. Task 0 — champion checkpoint recovery (blocking, must resolve first)

The M1 run dirs' `checkpoints/` are EMPTY; the weights live inside the pushed
docker images.  Recover in this order:

1. `docker create spint-b3s-rsyn3-freeze-top4-m1:allsource-s42-7899ae17` →
   `docker cp` the baked model artifact (payload pkl and/or full ckpt; inspect
   the image for the baked path — the M1 images followed the
   `spint-b3-m1:allsource-*` build pattern with the checkpoint inside);
2. Verify recovered state digest against any receipt in the 581727 lineage
   (`results/` receipts of the M1 cell that pushed it);
3. Record SHA-256 + seal 0444+sidecar in the new package's artifacts.

If the full model (encoder+decoder) is not recoverable from the image, fall
back: rebuild from `source_manifest.json` + the training recipe in
`m1_b3_allsource_v1` (the training root contains resolved_config.yaml) — cost:
one M1 training run (the line's own pipeline).

**Task 0.5 — phase-feature extraction (new loader path, blocking for training).**
The phase contrast MUST be computed from the RAW 20 ms counts + the trials
table (`start_time/gocue_time/contact_time`), NEVER from the encoder-facing
calib feature array: the M1 rsyn3 calib features are `[trials, 1024, N]` after
pad/interpolate (`interpolate_trials`), which destroys the bin↔time mapping.
The M1 rsyn3 datamodule currently exposes neither `calib_trial_spike_sums` nor
a phase segmentation — a new loader path is required.  CPU test: for every
calibration trial, hold-bin count + reach-bin count == the trial's valid
length.

## 6. Training and validation contract

- Train: FiLM only; frozen decoder/encoder/carrier/normalizer; data = 4
  held-in sessions, their scored windows.  **Training config pinned** (§7
  forbids sweeps, so the single allowed set is written here): Adam, lr 1e-4,
  12 epochs, 256 windows/session, batch 32, seed 42 — the M2
  `means_ep12_lr1e4` cell.  Explicitly NOT lr 3e-4: on M2 it had a higher
  mean but shuffle≈real — adverse selection for a mechanism-generality
  experiment.
- Validate on TWO surfaces, honestly labeled:
  - **Surface A (sanity, held-in)**: the 4 training sessions.  FiLM was
    trained here, so this CANNOT be a gate — it only checks no corruption
    ("held-in non-negative" is a disclosure, not a constraint: the FiLM is
    trained on this data and will trivially pass).
  - **Surface B (decision, later-day proxy)**: the 3 later-day sessions under
    the `carrier_k_later_day` convention — **M4-support carrier / last-6-trial
    query** (later-day public files have only 10 trials; with the deployed
    M10 carrier there is no remaining query — CARRIER_K_NOTE).  THREE
    disclosed mismatches: (1) carrier M4 ≠ deployed M10; (2) contrast
    estimation noise is higher on the 4-trial support; (3) **h-distribution
    shift — the deepest one**: the FiLM is trained on M10-pool features h and
    evaluated on M4-pool features h; the static champion is evaluated under
    the same M4 regime so the comparison stays fair, but the FiLM has never
    seen the M4 regime.  This is the main source of the 15–25% gate-pass
    estimate.
- **Null arms (zero cost, same trained weights, inference-input only)** —
  without these a +0.005 on 3 sessions is indistinguishable from an extra
  session-independent affine bias (M2 analysis: ~40% of the FiLM gain was the
  f(0) bias):
  1. zero-contrast: contrast dims set to 0 (only f(0) acts);
  2. session-constant: every unit gets the session-mean contrast (per-unit
     structure killed);
  3. row-shuffle: contrast rows shuffled across units (M2 protocol).
- **Static-anchor precondition (RESTORED from Rev1 — Task 0's only correctness
  proof)**: before ANY FiLM number is read, the docker-recovered champion
  weights with a zero-init FiLM, evaluated on Surface B, must **bit-exactly
  reproduce** the `carrier_k_later_day_v1` `b3s_rsyn3_freeze` chrono-k4
  baseline R².  Without this, "3/3 positive" may only mean the recovered
  weights are not 581727's.
- **Gates (pre-registered, on Surface B, after the anchor precondition)**:
  - full FiLM beats static champion with **3/3 sessions positive** and mean
    ≥ +0.005;
  - **zero-contrast arm's delta ≤ 80% of the full arm's delta** (else the
    gain is a bias, not contrast — reject);
  - session-constant and row-shuffle arms must each fall below full;
  - one confirmation seed allowed (disclosed as confirmation, not a sweep).
- If S1 fails on Surface B: run S3 (additive-only) as the pre-registered
  ablation before closing; S2 (unit gating) is the pre-declared second
  structure, not a replacement for a failed S1.

## 7. Boundaries

- No EvalAI submission from this cell; packaging is a separate gated cell.
- Compute: CPU (the 1.5k-param FiLM training is CPU-scale anyway).  Note the
  M1 line's own plan.py sets `GPU0_ALLOWED = False` (GPU0 is the display
  adapter); GPU1 may be used only after its ownership check.  Coordination
  with the other agent's in-flight work still applies.
- Luna's AJPF roots and the M2 roots remain read-only.
- Single training run; no LR/epoch/axis-mask sweeps; the contrast axes are
  frozen as specified (phase + direction only — no extra axes without a
  successor).

## 8. Expected outcome — probability re-evaluation (Rev2, honest)

Supporting factors: the mechanism is officially proven on M2 (+0.025 over
act30_full; local→official offset +0.001); the M1 phase axis is better
instrumented than M2's (explicit start/gocue/contact timestamps; every trial
contributes both phases); zero-init anchoring makes the downside exactly zero
(no submission without a passed gate).

Pressure factors: 0.64 base R² leaves little headroom; Surface B is thin
(3 sessions × 6 query trials) and doubly mismatched (M4 carrier ≠ the M10 the
FiLM was trained with; contrast noise on a 4-trial support); the rSyn3
carrier may partially subsume the phase contrast.

Tiered estimates:

| Event | Estimate | Basis |
|---|---:|---|
| Surface B gate passes (3/3 + ≥+0.005 + null checks) | **15–25%** | thin statistics + M4/M10 context shift; the phase axis itself is well-estimated |
| Official HO improves over 0.6396, given the gate passed | ~50% | static identity constructions transferred cleanly on M2, but M2's offset experience is M33-matched while this proxy is M4-vs-deployed-M10 — the offsets are not comparable, so 60–70% was too optimistic |
| **Official improvement overall (unconditional)** | **~10–17%** | product of the above |

Cost side: FiLM training is ~1.5k params on 4 sessions (CPU, minutes);
Task 0 recovery ≈ 1 hour; no submission without a passed gate — so the
expected cost of a negative outcome is confined to diagnostic effort.

Information value (independent of the gate): 3/3 pass → **consistent with**
the M2 recipe generalizing across datasets (a 3-session M4 proxy cannot prove
generality); zero-contrast
≈ full → the "gain" is a session-independent bias (misassignment caught);
full gate fails but nulls behave → the phase contrast axis is M2-task-
specific.  All three endings are attributable and publishable as analysis.

Calibrated expectation: this cell is a **low-cost, attributable, pre-
registered mechanism-generality experiment** — not a path to a second 0.32.
Submission is only discussed if Surface B passes its full gate.
