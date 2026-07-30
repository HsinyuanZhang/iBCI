# Native-FALCON MUA T4: M1/M2 feasibility, implementation, and frozen launch gate

## Local held-out-calibration test-only replay (2026-07-30)

This separate replay evaluates the same three frozen internal-LOSO cells
(`fold1/seed42`, `fold1/seed43`, `fold2/seed42`) on local FALCON
`held-out-calib` files, matching the repository's existing SPINT/B3 local
held-out path. It is **not** a public benchmark submission or a hidden EvalAI
query/test-set result. Target labels are a deployment privilege available in
each session's calibration NWB (`trials.tgt_loc`); they are not taken from
velocity/EMG evaluation covariates. The local calibration NWB necessarily
contains the neural and behavioural arrays used for this replay, so the result
must not be described as evaluation on an unopened hidden query file. Each cell
reloads its frozen source `best.ckpt`; `train=false`, `test=true`, no
optimizer/backward step, and checkpoint selection remains held-in only. T4/TS4
normalization is fit only on the corresponding held-in fold-train sessions;
TS4 is a stable non-identity channel-row permutation.

Strict artifact aggregation requires 3 M1 and 6 M2 held-out sessions per cell,
raw/filtered label-to-trial-boundary alignment, prefix/rank audit (M1 first 10,
M2 first 33), source-checkpoint SHA and frozen aggregate mapping. Results:

| task | F0 | T4 | TS4 | T4−F0 | T4−TS4 |
|---|---:|---:|---:|---:|---:|
| M1 (9 cell×session observations) | .62725 | .62505 | .62879 | −.00221 (1/3 cells positive) | −.00374 (0/3) |
| M2 (18 cell×session observations) | .22160 | .29139 | .22938 | +.06979 (3/3; 16/18 sessions) | +.06201 (3/3; 17/18) |

The auditable aggregate is `results/native_mua_heldout_t4_v1/aggregate_heldout.json`.

F0 is the B3 calibration baseline and does not consume target labels. T4 and
TS4 both consume the same target-labeled support and use the same B3S width;
TS4 differs only by a stable channel-row permutation. Consequently, `T4−F0`
measures the deployable benefit of adding supervised target-tuning information
but is not label-information matched, whereas `T4−TS4` isolates whether the
correct channel-attached T4 content matters. M2 passes both comparisons; M1
passes neither.

This matrix establishes only the tested budgets: M1 chronological first 10 and
M2 first 33 (16 directional trials in each M2 held-out prefix). It does not
establish that 10 or 33 is minimal. A legal T4 support must preserve exact
trial-label alignment and a rank-3 `[1, cos(theta), sin(theta)]` design; the
current path therefore disables random support. A separate label-budget curve
is required before making a low-label or label-efficiency claim.

**Status (2026-07-30):** the frozen M1 and M2 blocks are both
complete (9/9 each) and passed strict aggregation. The local
`held-out-calib` NWBs were opened and scored; no hidden EvalAI query/test NWB
was available, opened, or evaluated. The result rejects a universal "T4 stably
improves native MUA" statement: M2 has a consistent T4 gain over F0 and TS4,
whereas M1 does not.

M2 means are `F0=0.527791`, `T4=0.624124`, and `TS4=0.588357`.
`T4-F0=+0.096334` is positive in 3/3 cells
(`+0.073464/+0.094278/+0.121259`), but `T4-TS4=+0.035767` is positive in only
2/3 cells (`-0.057007/+0.028527/+0.135781`). Thus the side-feature network path
is useful on M2 relative to F0, while the evidence that the correctly attached
T4 content itself beats the matched shuffled-content control remains
heterogeneous.

M1 means are `F0=0.591662`, `T4=0.599500`, and `TS4=0.574703`.
`T4-F0=+0.007837`, with only 1/3 cells positive
(`+0.039790/-0.003559/-0.012720`). `T4-TS4=+0.024797` is positive in 3/3
cells (`+0.049554/+0.024130/+0.000707`), although the fold-2 margin is
negligible. Thus correct T4 content is distinguishable from shuffled content
on M1, but it does not yield a stable net gain over F0.

## 1. Feasibility decision

**Decision: feasible on held-in calibration data, with a deterministic-support
constraint.** Both datasets are native FALCON MUA, not the DANDI pseudo-MUA bridge.

| Task | Native neural evidence | Legal calibration target source | T4 direction construction | Fixed support gate |
|---|---|---|---|---|
| M1 / 000941 | 64 channels in NWB `units`; FALCON M1 spike-count path | held-in-calibration NWB `/intervals/trials/tgt_loc`, scalar target azimuth (degrees) | `theta=deg2rad(tgt_loc)` | first 10 trials: all four calibration files have rank-3 `[1,cos,sin]`, condition numbers 5.20--5.75 |
| M2 / 000953 | 96 channels; NWB electrode metadata says `250Hz HPF, -4.5RMS threshold` | held-in-calibration NWB `/intervals/trials/tgt_loc`, 2-D target coordinate | angle of `tgt_loc-(0.5,0.5)`; centre/rest target is explicitly unlabeled | first 33 trials: 16 directional trials, rank 3, condition numbers 1.58--2.09 |

The M1 labels also include `condition_id`, `obj_id`, and target object, but T4 uses only
`tgt_loc`; it does not silently mix task/object identity into a directional feature.
M2 centre targets have no direction and are omitted from the cosine fit, never assigned an
arbitrary direction. The neural feature is fit in the correct native channel axis
(`N=64` M1; `N=96` M2).

### Data isolation and alignment contract

T4 reads labels only from the held-in **calibration** file corresponding to the session.
It never reads `tgt_loc`, velocities, EMG, or behaviour from the held-in-minival query
file to create side features. Query/minival covariates continue to be used only as the
existing internal-LOSO evaluation target. `include_heldout_in_fit=false` and
`include_heldout_in_test=false` are asserted in the runner, artifact validator, and data
module. Native T4 refuses held-out loading even if a caller attempts to enable it.

The FALCON loader creates trial boundaries at neural/behaviour timestamp indices. The
implementation reads NWB trial labels, checks their count against the raw `trial_change`,
then rechecks after `use_calib_intertrials=false` filtering. This caught a real M2 edge
case: filtering can remove the first calibration trial boundary; its target label is removed
with it before T4 fitting. Any count mismatch raises rather than attaching a shifted label.

### Why random calibration is disabled for this screen

Existing FALCON training configurations use random calibration windows. This is not legal
to retain unchanged for M1 T4: a read-only audit found rank-2 target designs for some
arbitrary contiguous 10-trial windows (for example M1 session `20120926[93:103]`). A
minimum-norm fit would be an unmeasured, direction-dependent intervention. The native-T4
path therefore fail-closes when `random_calibration=true`.

The frozen fair comparison instead uses deterministic first support for **all** F0, T4,
and TS4 cells: M1 first 10 / M2 first 33 calibration trials. This changes support sampling
relative to historical B3 runs, so F0 is retrained; historical F0 artifacts must not be
reused. It preserves the existing FALCON internal-LOSO split, native signal view, decoder,
loss, fold/seed cells, and held-out exclusion.

## 2. Frozen mechanism and controls

For each native channel, use valid-prefix calibration spike sums and lengths, not cubic
interpolated/padded neural tensors. On directional trials fit

```text
rate_i(theta) = b_i + a_i cos(theta) + c_i sin(theta)
T4_i = [a_i, c_i, sqrt(a_i^2+c_i^2), b_i]
```

All train-only calibration support windows selected by the deterministic policy provide
the training feature mean/std. Validation T4 is normalized by those statistics only.
Feature extraction creates no formal-test cache and does not draw labels from evaluation
behaviour.

| Group | Encoder | Content | Required matched facts |
|---|---|---|---|
| F0 | B3 | no side feature | fixed first support, 12 epochs, frozen decoder, internal LOSO |
| T4 | B3S (`side_dim=4`) | correct channel-attached normalized T4 | same support / fold / seed / budget as F0 |
| TS4 | B3S (`side_dim=4`) | stable non-identity permutation of T4 rows along the native channel axis | preserves values/normalization/dimensions; only channel-to-row assignment changes |

The T4/TS4 first post-pool columns are zero initialized, as in the existing B3S path, so
the architecture-width comparison is controlled. The primary contrasts are `T4-F0` and
`T4-TS4`, never T4 alone. This screen tests whether calibration target tuning transfers to
native MUA; it does not establish a new real-MUA external formal-test result.

## 3. Implemented path

New/changed components:

| Component | Role and safety property |
|---|---|
| `streaming_calibration_exp/src/data/falcon_t4_features.py` | task-specific target extraction, rank-3 cosine fit, train stats, deterministic TS4 permutation; raises on missing/nonfinite/misaligned/rank-deficient labels |
| `src/data/falcon_datamodule.py` | carries calibration-only labels with trial boundaries, calculates valid-prefix channel sums, lazy per-session T4/TS4, passes side features as a five-tensor batch, refuses held-out access and random calibration for T4 |
| `configs/model/streaming_b3s_t4{,_m1}.yaml` | B3S with exactly four T4 inputs |
| `configs/experiment/b3_native_mua_f0_*`, `b3s_t4_*`, `b3s_ts4_*` | matched M1/M2 F0/T4/TS4, `E=12`, no early stopping, deterministic support, internal LOSO, held-out false |
| `tests/test_falcon_t4_features.py` | cosine recovery, rank rejection, deterministic nonidentity TS4, real held-in M1/M2 data-module routes, held-out refusal |
| `scripts/run_native_mua_t4_screen.sh` | two-GPU safe queue, dry-run, preflight, no-overwrite, per-cell logs/PIDs/manifest |
| `scripts/aggregate_native_mua_t4_screen.py` | requires complete artifacts and validates task/fold/seed, B3/B3S, side group, fixed support, 12 epochs, LOSO and held-out exclusion before reading `test_heldin` scores |

## 4. Completed gates

| Gate | Evidence | Status |
|---|---|---|
| target labels exist without evaluation leakage | direct held-in calibration NWB inspection | pass |
| native-MUA identity | FALCON M1/M2 units and M2 threshold metadata | pass |
| fixed support direction rank | read-only all-session audit above | pass |
| trial-label alignment | raw and post-calibration-filter checks | pass |
| T4 math and TS4 control | synthetic cosine/rank/permutation tests | pass |
| real M1/M2 data route | `tests/test_falcon_t4_features.py`: **6 passed** | pass |
| model integration | CPU one-batch smoke: M1 `[2,10,1024,64] + [2,64,4] -> [2,1,16]`; M2 `[2,33,100,96] + [2,96,4] -> [2,1,2]`; `val_heldout_dataset=None` | pass |
| runner/aggregator syntax | runner dry-run emits all M2 F0/T4/TS4 commands; aggregator compiles | pass |
| GPU forward/backward | M1-T4 and M2-T4 each completed 1 epoch with two train and two validation batches; exit status 0 | pass |
| two-GPU utilization | GPU0/GPU1 observed at 95%/91% utilization after persistent launch | running |

## 5. Frozen launch matrix and stop rules

For each task, the minimum matrix is all three groups over cells
`(fold1,seed42)`, `(fold1,seed43)`, `(fold2,seed42)`: nine cells/task, 18 cells for
M1+M2. With both tasks launched, GPU0 runs M1 and GPU1 runs M2. Each queue begins
with the matched diagnostic triplet `F0 -> T4 -> TS4` at `(fold1,seed42)`, before
any replication cell. This produces the earliest paired result but is not a selection
or stopping analysis; the complete frozen block still determines the development result.

Before `--launch`, require:

1. `nvidia-smi` succeeds and both devices have sufficient free memory; no existing native-MUA
   T4 training process exists.
2. A GPU single-fold/single-seed smoke completes for one T4 cell on each task and
   exercises setup, teacher loading, B3S/T4 forward, backward, and validation.
3. Formal cells—not the deliberately truncated smoke—must produce complete
   `resolved_config.yaml`, `split_manifest.json`, `run_metadata.json`, and
   `metrics_summary.csv`. The strict aggregate waits for every frozen cell.

If any calibration label/rank/alignment assertion fails, stop the affected task and report
the exact session/window; never use query behaviour to replace it. If F0/T4/TS4 complete,
report per-task absolute score and paired `T4-F0` and `T4-TS4` deltas. Do not compare M1 and
M2 absolute R², and do not claim formal held-out or cross-modal equivalence.

## 6. Current operational handoff

The matrix started under screen ID `native_mua_t4_v1`. The persistent M2 supervisor is
PID `623885`, with GPU1 worker PID `623894`; the recovered M1 supervisor is PID `624408`,
with GPU0 worker PID `624416`. Runtime state and per-cell logs are under
`sua_exploration/results/native_mua_t4_v1/`.

A detached 24-hour watcher (PID `627126`) writes `monitor_progress.json` every
60 seconds and runs the strict aggregator as soon as either complete nine-cell task
block is available (`aggregate_m1.json` / `aggregate_m2.json`), followed by the combined
`aggregate.json`. At 14:21 HKT, the first formal artifact—M2 F0
`(fold1,seed42)`, held-in R² `0.58708197`—had passed the strict per-cell contract;
M2 T4 was running and M1 F0 was in epoch 5/12. This isolated F0 value is only a
paired baseline and is not an effectiveness result.

The first fire-and-forget launch was reaped when its short-lived launcher exited, before
Python initialized; the runner now keeps its supervisor alive and records terminal worker
status. The first persistent M1 F0 attempt then correctly failed a legacy-baseline guard:
the default baseline manifest belongs to the M2 teacher and its hash cannot validate the M1
teacher. This historical B0 comparison is unrelated to the new matched
`F0/T4/TS4` contrast, so the native-MUA runner now sets
`require_baseline_validation=false` for every group and task. It does not relax target
alignment, held-out exclusion, fold purity, fixed support, epoch budget, or artifact
validation. M1 was relaunched on GPU0 while the valid M2 queue continued on GPU1.

B3TStream hardware work remains paused until this native-MUA T4 screen yields a
controlled result.

### 6.1 Completed M1 and M2 blocks

The strict M2 artifact is
`sua_exploration/results/native_mua_t4_v1/aggregate_m2.json`. It contains all
nine expected artifacts, validates the frozen task/fold/seed, architecture,
support, epoch and held-out-exclusion contracts, and records
`no_formal_test_sessions_evaluated=true`.

At 19:38 HKT the monitor reported M1 9/9 and M2 9/9, wrote
`aggregate_m1.json`, `aggregate_m2.json`, and the combined `aggregate.json`,
and marked the screen complete. GPU0 was then reassigned to seed 43 of the
validation-only SUA same-electrode relation pilot; GPU1 was already running
seed 42. This resource reassignment does not mix data, checkpoints, or
evaluation scopes.

## 7. 24-hour interpretation discipline

The earliest `(fold1,seed42)` triplet is a smoke/diagnostic only. It may establish that F0,
T4, and TS4 execute and are paired, but it cannot establish effectiveness, M1/M2 transfer, or
a modality claim. Do not add a seed, change a fold, or alter the T4 feature after seeing it.

After the complete nine-cell task block, report each task separately: absolute F0/T4/TS4
`test_heldin` R², `T4-F0`, `T4-TS4`, signs across the three cells, and the calibration-label
contract. A positive T4 result can support only: “target-conditioned calibration tuning is
usable on this native FALCON MUA internal-LOSO protocol.” It cannot support pseudo-MUA
equivalence, M1-to-M2 generalization, formal held-out performance, or any SUA-specific claim.
A null/negative result can reject this exact target-angle cosine feature under the fixed
first-support budget; it cannot establish that native MUA lacks functional calibration
information. If M1 and M2 differ, report the difference descriptively and do not compare
their absolute R² scales or choose a task as the winner post hoc.
