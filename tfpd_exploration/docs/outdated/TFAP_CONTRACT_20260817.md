# TFAP Contract: Task-Frame Aligned whole-Model Pretraining

Date frozen: 2026-08-17
Status: frozen next-round contract. Stage 0 (CPU preflight) must pass completely
before any GPU work. This document supersedes nothing in the Gate 1–4 receipts;
all sealed artifacts remain read-only.

Authority: this file is the single authority for the TFAP round. It freezes the
user's design as stated on 2026-08-17 and borrows the still-binding discipline of
`HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md` (matched scorer, 0444 receipts,
closure equality, source-only normalizers, no formal/organizer-held data) by
reference.

## 0. Hypothesis and arms

Hypothesis (narrow, staged-pretraining only): pretraining the WHOLE teacher-free
spintshape model (encoder + decoder, standard initialization, no teacher
checkpoint anywhere) on an aligned task-frame supervision source — DANDI 000128
MC-Maze (sub-Jenkins), the data the SPINT teacher lineage trained on — with
task-frame-aligned T4 side features, then fine-tuning on the strict-27 DANDI
000688 source roster, improves external subject-M transfer over the scratch
48-epoch baseline.

Three arms:

1. **Arm A (frozen baseline, no new training)**: the Gate-2 `direct_t4_48` SWA,
   external native R² **0.1610** (receipt
   `results/gate4_arm_external_v1/armA__external_subject_M.json`, sha
   `941949b2…`), within 0.516273 (Gate-3 receipt `96bdccb6…`). Sealed, read-only.
2. **P-T4 (main TFAP arm)**: whole-model pretraining on 000128 with canonical
   aligned T4 side features (closed-form fit on 000128's own trial directions,
   normalized by 000128's own source-only stats), then strict-27 fine-tuning
   identical to arm A's recipe.
3. **P-Z4 (mechanism control)**: identical data, budget, steps, schedule, and
   code path as P-T4, with the visible side masked to exact zero AFTER T4
   normalization (`zeros_like(standardized_T4)`, never `0 × raw_T4`). Any P-T4
   gain that P-Z4 also shows is not task-frame information; it is generic
   pretraining or warm-start value.

## 1. Stage 0 — CPU preflight (all gates must pass before any GPU)

- **(a) Data accessibility.** Local DANDI 000128, sub-Jenkins:
  `sua_exploration/data/000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`
  (the only file TFAP opens; `desc-test` is the NLB held-out set with no behavior
  and is never opened). Loader lineage: `mc_maze/datamodule.py`
  (`MCMazeDataModule`, the SPINT-teacher MC-Maze loader family) for binning and
  behavior interpolation discipline; TFAP consumes it through a 688-contract
  adapter (M30 chronological calibration, T=100 interpolated trial length,
  window 50, bin 20 ms, units = non-heldout only).
- **(b) Interface consistency (test + 0444 receipt).** ONE model builder —
  `src/tfpd/spintshape_module.py:build_spintshape_model(seed=42)` — serves both
  datasets. Proven by tests: state-dict keys and shapes are IDENTICAL across
  128- and 688-shaped consumption (the set decoder is unit-count agnostic); the
  builder never touches a teacher checkpoint (no teacher architecture, no
  teacher logits); whole-model transfer is `load_state_dict(strict=True)` with a
  bitwise state-SHA roundtrip; a perturbed or partial state raises (no silent
  partial load); a forward on a 688-shaped batch and a 128-shaped batch with the
  same loaded state is finite and leaves the parameter set unchanged.
- **(c) T4 on 000128 (own closed form, own normalizer).** Per-trial movement
  direction `theta = atan2(target_pos[active_target])` from the trials table
  (000128 has no `target_dir` column; targets are maze positions relative to the
  screen center). Pool = the first 30 chronological successful train trials (M30
  discipline). Per unit (non-heldout only): per-distinct-direction mean firing
  rate over pool trials, then the SAME closed-form least-squares cosine fit used
  on 000688 (`rate(theta) = b + a·cos(theta) + c·sin(theta)`), emitting
  `[a, c, m, b]` with `m = hypot(a, c)`. Normalizer: per-column mean/std over
  the 000128 units only — 000688 statistics are never reused. The fitted matrix,
  normalizer semantic SHA, and degeneracy counts are bound in the Stage-0
  receipt before any GPU.
- **(d) Budget freeze.** 000128 pretraining budget is EXACTLY 48 epochs on the
  000128 epoch definition (all train-split windows, batch 32, single-session
  drop-partial sampler), optimizer Adam
  betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False, no clipping.
  **Revision (2026-08-17, coordinator GO, before the first GPU step):** the
  Stage-1 LR schedule is arm A's recipe applied phase-locally to the 48
  pretraining epochs — linear warmup 1e-5 -> 1e-4 over the first two epochs'
  steps, then cosine decay to 1e-6 at the final step of epoch 47 (the same
  step-level function `arm_common.lr_at_step`, steps_per_epoch measured in
  Stage 0). This replaces the earlier "constant 1e-4" draft; the revision is
  sealed here before any pretraining gradient step. P-T4 and P-Z4 run
  bit-identical schedules.
- **(e) P-Z4 mask semantics.** Standardize T4 with the 000128 normalizer, then
  mask to exact zero: `zeros_like(standardized_t4)` (bitwise +0.0), identical to
  the Gate-2 `admit_side(..., "z4")` semantics. Never `0 × raw_T4`; admission
  strictly after normalization.

## 2. Stage 1 — 000128 pretraining (GPU, only after Stage 0 passes)

- Both arms load the SAME canonical initial state as arms A/B/C
  (`results/admission_arms_v1/canonical_initial_state.pt`, state sha
  `65bacb85…`), strict=True, no rebuild, and consume EXACTLY the sealed
  Stage-0 payload (`results/tfap_stage0_v1/jenkins_derived_payload.npz`,
  sidecar-verified).
- P-T4 visible side: standardized 000128 T4. P-Z4 visible side:
  `zeros_like` of the same tensor. Everything else identical: data, sampler,
  batch order, steps, optimizer, schedule, seed.
- Behavior normalization for the 000128 phase is 000128's own hand-velocity
  stats (self-fit, never 000688's). No validation, no early stopping, no
  checkpoint selection; final-four SWA window predeclared (epochs 44–47).
- 0444 launch/terminal receipts per arm with full disclosure, closure equality,
  and per-epoch §9-style diagnostics (loss, steps, LR, grad norms, W_side norms,
  state/optimizer SHAs, finiteness).

## 3. Stage 2 — strict-27 fine-tune (GPU)

- Whole-model `load_state_dict(pretrained_state, strict=True)` — encoder AND
  decoder; no partial load, no reset, no freeze.
- Optimizer: fresh Adam (all parameters), then EXACTLY arm A's recipe: linear
  warmup 1e-5 → 1e-4 over the first two epochs, cosine decay to 1e-6 at the
  final step of epoch 47, 48 total epochs on the strict-27 full window set
  (33,925 steps/epoch), same sampler (batch 32, seed 42).
- Visible side: canonical normalized T4 under the strict-27 source-only
  normalizer (the deployment contract). Clear optimizer state at the phase
  boundary (no carryover from Stage 1).
- Predeclared final-four SWA (epochs 44–47) + final checkpoint, immutable.
- Same receipts/diagnostics discipline as Gate 2.

## 4. Stage 3 — one-shot scoring matrix and adoption gates

One scorer everywhere (`src/tfpd_lane/matched_scorer.session_r2`, per-session
variance-weighted, equal weight per session), scored once, no checkpoint
selection, no target adaptation:

- within-dev 6 sessions (Gate-3 surface) and external sub-M 15 sessions
  (Gate-4 surface, authorization-gated), four diagnostics each (native / zero /
  wrong-pair / destroyed-activity, frozen seeds 1234 / 4321);
- paired contrasts with the full §10 statistics: mean, median, n-positive,
  min/max, all deltas, fixed-seed bootstrap 95% interval, exact sign pattern.

**Adoption gates (all six must hold for P-T4 to be adopted over arm A):**

1. **Δexternal ≥ +0.03**: P-T4 SWA minus arm A SWA, paired mean over the 15
   external sessions, ≥ +0.03;
2. **≥ 10/15 positive**: at least 10 of 15 external sessions with positive
   paired delta;
3. **within ≥ −0.03**: P-T4 within-dev mean no more than 0.03 BELOW arm A's
   (0.516273) — pretraining must not trade within away;
4. **native > zero / wrong-pair / destroyed**: the P-T4 external native score
   exceeds each diagnostic arm's mean (no carrier-only or activity-destroyed
   shortcut);
5. **zero target update**: no target-session weight updates, gradients, or
   optimizer steps anywhere in scoring — receipts must show the counter at 0;
6. **formal unopened**: no formal or organizer-held data opened at any point.

P-Z4 is read as mechanism evidence: if P-Z4 also clears gate 1, the gain is
attributed to generic pretraining, not task-frame alignment, and the TFAP claim
fails even if P-T4 is adopted as an engineering recipe.

## 5. Discipline carried over (binding)

- No teacher checkpoint, teacher logits, or distillation loss anywhere in TFAP.
- All receipts 0444 with sidecar SHAs; fresh output roots; launch/final
  implementation closure equality; PYTHONNOUSERSITE=1 mandatory.
- Source-only normalizers per phase; never refit on target sessions; formal and
  organizer-held data sealed.
- Development data (within/external) influences NOTHING before Stage 3: no
  pretraining-phase or fine-tune-phase decisions may read it.
- Stop conditions of the 2026-08-16 handoff apply mutatis mutandis (fail closed
  on any invariant drift, closure drift, or nonfresh output root).
