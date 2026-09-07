# H1 QueryAge: one authorized continuation to cumulative epoch 24

Date: 2026-09-06. Scope: the user's explicit request to continue this H1 pair to
24 epochs after observing continued improvement through epoch12. All other
new experiments and network changes remain paused. This document defines the
continuation; a separate exact code/input/output authorization and successful
admission are required before launch. It does not authorize a submission,
image push, new architecture, sweep, or automatic successor after epoch24.

## Parent and preservation

The parent is the completed `formal_prefix_split12_v1` under
`results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/`.
Require its original `COMPLETE_FIXED_FORMAL_NO_PROMOTION` receipt, both full
epoch1–12 histories, original external authorization/current source closure,
selection freeze, selected/epoch12 native archives and plain exports, and
complete owned manifest. Do not change any parent file, parent executable,
parent protocol or original result. The parent is an actual12-epoch result,
not retrospectively described as initially preregistered for24 epochs.

Resume only the per-arm `checkpoints/{arm}_epoch_012.pt` full training states.
Parent receipt SHA:
`32736060d23962c42d8c7e7cd73059cfba08b9785d66a47b409a0e8fbf411e82`.
Parent epoch12 checkpoints: FLAT
`ddc7df11b0487af317e427ef0a31787825b5d973c735ae820ce5377e4af1312a`;
ROUTE `484f2f3086897729608fe6bd47b9c173f621a7c088c91085e8b5be08c6819fd8`.
No source-capacity checkpoint or plain EMA export is a resume state.

## Only changed factor: additional training budget

- Absolute epochs13 through24, exactly731 effective updates per epoch and
  23,212 source windows per epoch. Final global update is17,544; the first new
  update is8,773. No extra epochs or early best-epoch stopping.
- Unchanged H1 FLAT/ROUTE QueryAge16 model factory, spatial preset, parameter
  topology, raw input W700×176, seven output channels, and immutable M3 banks.
- Unchanged seed42, absolute-epoch/session sampler and batch ordering;
  p=.5 left-zero cold prefix before stateless p=.1 whole-unit dropout and bank
  mask. Use epochs13–24 in all seeds; do not replay epochs1–12 augmentation.
- Effective batch32, microbatch8, native target×20 in raw MSE, the same trusted
  AdamW groups/weight decay, gradient clip1, and constant post-warmup LR1e-4.
  Do not restart epoch-one warmup or add a new scheduler.
- Strictly restore RAW parameters/buffers, every AdamW param group and saved
  moment/step state, EMA shadow/decay=.9995/update count8,772, and torch CPU,
  visible CUDA, NumPy and Python RNG. Construction/probes must not consume RNG
  after the final resume restoration before training begins.
- Preserve epoch checkpoint-before-selection ordering, recursive disk
  round-trip validation, RAW restoration after EMA scoring, finite/resource
  guards, and fresh input/code/external-authority checks. Continuation
  checkpoints must retain the same complete state information plus new
  continuation binding and parent-checkpoint identity.

The new runner must not mutate the old module's EPOCHS or monkeypatch its
functions. Absolute-epoch helpers with expanded bounds belong only to the
additive continuation implementation. Test agreement with the parent law for
epochs1–12 and correct absolute13–24 behavior before production admission.

## Paired order, selection and reporting

The parent supervisor must have finished and both GPUs must be free. Start
FLAT on physical GPU0 and ROUTE on physical GPU1, one CPU intra-/inter-op thread
per worker and the same CPU affinity as the parent (FLAT12–15, ROUTE8–11).
Both workers must validate their actual restored state and precommit matching
epoch13–24 sampler/prefix/dropout identities before one shared START barrier.

After every new epoch, score only the same2,908 frozen minival endpoints with
the unchanged native FP64 EMA scorer. Select each arm once, after both workers
finish, using the highest EMA pooled R² over all ordered epochs1–24, earliest
tie. Epochs1–12 come from immutable parent records, not re-evaluation or RAW.
No complete-surface score changes selection or the continuation recipe.

After the paired selection freeze, strictly reload/export the selected EMA and
fixed endpoint24 EMA for each arm and score the same20,325 complete points.
Keep exact prediction/target/session/end NPZ plus plain EMA states and checksums.
If selected equals endpoint24 (or is an old parent epoch), identify it explicitly;
duplicate labels are not independent replications. Keep parent12 quality and
all negative results visible. No quality/non-inferiority promotion is automatic.

## Resource and execution boundaries

No fresh architecture/resource probe is needed for this unchanged-step budget
extension: the actual completed parent is direct resource evidence for the
same two-arm, same-data, same-shape12-epoch update/evaluation workload. Its
measured whole-job duration was6,756.502721746 seconds. The conservative new-job
envelope is `1.5 × parent elapsed + 1800 = 11934.754082619 seconds`, below the
unchanged21,600-second hard supervisor limit. The1800-second allowance covers
additional continuation admission, state checks, schedule preparation and IO.
This is a bound/forecast, not a promised completion time.

Before START require restored-state checks and actual per-worker thread,
resident-memory and GPU-allocation guards. Retain22GiB per-worker peak GPU/RSS
limits and a6-hour whole-job wall limit. No external task may be terminated to
obtain a GPU. Any owned child failure/timeout stops only this supervisor's
children, preserves failure evidence, and does not auto-retry.

An external root authorization must bind the completed parent, actual current
immutable source/code closure, both resume checkpoints, this protocol, both
new executable modules, exact fresh output and resource/selection rules.
Output/authorization must not overlap parent or any frozen input. Admission,
pre-START and post-finalization checks cannot be replaced by an existing PASS
label. Test-only fixture controls have no production CLI entry.

## Interpretation and stopping point

This is a user-authorized adaptive extension after inspecting the12-epoch
selection trajectory. It is not an untouched confirmatory test, a clean
QueryAge-versus-prefix ablation, or evidence of improvement until quality is
compared on the appropriate same surface. Original and C2 remain separately
identified references; runtime correctness or training completion is not
non-inferiority. After cumulative24-epoch closeout, stop and hand results to
the user for review. No epoch25+, new network, new dataset, extra diagnostic
forward, latency run or official action is included.
