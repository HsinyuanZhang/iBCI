# Work Order — CAL-AUG V1: matched T0/C1 B3S-calibration-prefix robustness training

Date: 2026-08-29
Authority: `docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md` §6 (queue items 3–4), §13.
Authorization: operator goal directive 2026-08-29. Smoke → throughput probe → full pair, one launch each.

## 1. Hypothesis (single axis)

Make the B3S activity identity less sensitive to the exact calibration prefix,
leaving query activity, T4, decoder, loss, optimizer, batch order, and the
dropout-p stream byte-identical to matched T0.

- C1 operator: **deterministic chronological prefix-length cycle**
  `M_step = (30, 10, 4)[global_training_forward % 3]`, applied as
  `calib_trials[:, :M_step]` before the model call. Pure integer arithmetic on
  the forward counter — no Python/NumPy/Torch RNG consumption, no change to the
  dynamic-dropout p sequence (proved by digest + count invariant).
- T0: the SAME successor runner, operator disabled (hook registered, schedule
  returns the input unchanged). Never the old sealed checkpoint as the comparator.

## 2. Seam (frozen bytes stay authoritative)

Successor `scripts/run_cal_aug_cell_v1.py` importlib-loads the sealed
`run_admission_arm.py` + `run_pop_robust_cell.py` after verifying their pinned
SHA-256 literals (guidance §2.1 table; both still match the working tree). The
operator is a `forward_pre_hook (with_kwargs=True)` on the `StreamingSpintModel`
that rewrites `kwargs["calib_trials"] = kwargs["calib_trials"][:, :M]`, gated on
`module.training` so every eval/diagnostic forward is untouched. One hook
invocation per training forward (the sealed p-count invariant already proves
one forward per optimizer step and stays as a hard check).

Arms share: seed 42, canonical initial tensor state (strict load + bitwise
proof), model graph/parameter count, strict-27 roster (manifest
`4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`), batch 32,
33,925 steps/epoch × 48 epochs, Adam(1e-4, 0.9/0.999, eps 1e-8) + warmup/cosine,
final-four SWA, identical sampler batch order, identical dropout-p stream,
normalized ordinary M30 T4 (`t4_authority_fingerprint` byte binding), dense
valid-bin MSE, no target selection/update.

## 3. Smoke (feasibility + no-harm gate only; cannot select the method)

Short-N-step run of BOTH arms on the training GPU producing:
exact initial-state equality (canonical artifact + `state_sha256`), optimizer
and schedule equality (`schedule_params` + constructor receipts), first-N batch
indices + session order digest equality (hash of `sampler.batched_indices[:N]`
+ per-sample session names), first-N dropout-p sequence digest equality (hash of
`recorder.sampled_p[:N]`), C1 prefix length sequence + support-row digests
(which calib rows are visible per step), finite loss/model/grad/Adam state, B3S
consumes the declared variable prefix (assert `trial_count == M` per step via
encoder state or equivalent), T4 bytes unchanged (fingerprint), query neural
bytes before dropout unchanged (T0 vs C1 first-N input digests), and no target
path resolved (within/external rosters never opened; val roster dropped before
array open — assert flags).

## 4. Throughput probe + device binding (§6.4)

Same physical GPU for both arms, serially: **GPU 0 = `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`** (the sealed run's own card, `CUDA_VISIBLE_DEVICES=0`). Probe: in-run 100-step throughput after preparation on the Cell-D graph, publishing steps/s, projected 1,628,400-step seconds per arm, pair GPU-hours, scoring time estimate. Planning ceiling **12 GPU-hours** for the two arms; exceeding requires revised explicit authorization. Hard wall-clock timeout per arm: 8 h (sealed anchor 5 h 08 m 32 s; 3090), with atomic `CELL_FAILED` receipt on breach — new surface, no precedent lane has one. No other GPU job may share the card during a training arm.

## 5. Mechanism readout (§6.6, source roster, per arm X ∈ {T0, C1})

New additive source-27 materialization: `arm_runner.build_datamodule` →
`dm.train_dataset.sessions[name]` per source session (neural, behavior,
calib_trials [30,100,N], valid_starts, side_features = the TRAINING M30 T4,
byte-bound by `t4_authority_fingerprint`). Windows = the training window law
(valid starts inside rewarded trials), last-bin scoring, valid mask
`(behavior != -1).all(-1)`, house `session_r2`, equal-session mean over 27.

For M ∈ {30, 10, 4} (chronological prefix `calib[:M]`, T4 = the SAME training
side for every M within an arm):

```text
prefix_degradation_X(M) = R2_X(B3S=M, T4=M30) − R2_X(B3S=M30, T4=M30)
prefix_robustness_recovery(M) = prefix_degradation_C1(M) − prefix_degradation_T0(M)
```

Paired per-session deltas everywhere; B3S identity-vector distance between M30
and each shortened prefix reported as descriptive only (collapse caveat).
Source mechanism registration (NOT a deployment claim):

```text
≥1 of M4/M10 recovery ≥ +0.01 R2 AND positive source sessions ≥ 18/27
AND C1−T0 at M30 ≥ −0.01
```

## 6. Deployment scoring (§6.7, frozen matched-scorer recipe)

After a fixed source-only choice, score T0 and C1 once each on the frozen
M4/M10/M30 within-6 + external-15 inputs through the sealed deployment recipe
(D-opt-first-30 @M4, chronological @M10/M30; ridge-T4 λ=0.1 per budget),
governing last-bin variance-weighted R², equal-session means, paired deltas,
positive counts, fixed-seed session bootstrap CIs, M30 safety, full prediction
and state digests, target optimizer/backward/update counts = 0.

Lower continuation gate:
`≥1 of external M4/M10: mean Δ ≥ +0.015, ≥10/15 positive, bootstrap LB ≥ 0;
the other: Δ ≥ 0; external M30 Δ ≥ −0.02; within every budget Δ ≥ −0.02.`
Primary performance claim: `≥1 of external M4/M10: mean Δ ≥ +0.03 and ≥10/15
positive` with all safety conditions. Mechanism-positive + deployment-failed →
`MECHANISM_POSITIVE__DEPLOYMENT_INCONCLUSIVE`, stop C1 expansion. Never average
budgets to hide an M30 regression.

## 7. Process

Fresh roots `results/cal_aug_v1/{smoke,probe,t0,c1,mechanism,deployment}/`;
attempt receipt before any source/model access; sealed-file SHA pinning at
import; `source_closure` at launch and terminal with launch==final equality;
per-epoch diagnostics + SWA manifest with strict reload; atomic
terminal-or-failure; checkpoints `seal_file` 0444. Tests (no data, no CUDA):
schedule arithmetic, hook gating (training-only, kwargs rewrite, eval
untouched), T0/C1 stream-equality laws on synthetic modules, smoke digest
logic, mechanism/deployment gate boundary semantics (epsilon 1e-12), timeout
receipt path. Focused tests green before any GPU process.

## 8. Out of scope

T4 perturbation (separate later cell, gated on C1 passing §6); combined
calibration+T4 arms; query-unit dropout variants (already covered by dynamic
U(0,1)); prefix operators other than the declared cycle (resampling, rate
scaling, noise); any target-session fitting; M1/H1.
