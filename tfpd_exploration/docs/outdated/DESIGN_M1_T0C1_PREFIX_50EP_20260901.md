# M1 matched T0/C1 calibration-prefix pair — 50-epoch extension

Status: SPEC FROZEN 2026-09-01, pre-implementation.
Owner lane: `m1_t0c1_prefix_v1_50ep` (new package, new result root, fully additive).
Predecessor lane: `m1_t0c1_prefix_v1` (20-epoch, sealed, **read-only — never edited**).

This document is the single source of frozen literals. The implementation, the
independent audit, and the receipts all bind to the values written here. Any
number that appears in code and not here is drift.

---

## 0. Isolation contract (hard, non-negotiable)

Another agent is training concurrently on this host under the codename
`paired_anchored_calibration_dropout`, on **GPU 0**, with its own result roots.

| Rule | Binding |
| --- | --- |
| GPU | `CUDA_VISIBLE_DEVICES=1` only. GPU 0 is never used, never queried, never killed. |
| GPU queries | Only ever `nvidia-smi ... --id 1` / `-i 1`. Never a bare `nvidia-smi`. |
| Foreign roots | Any `results/` root this lane did not create is untouched (not even read). |
| Frozen code | No edit to `src/m1_t0c1_prefix_v1/`, to any `cross_session_worst_group*`, to `m1_heldin_heldout_gap_v1`, to `m1_h1_activity_headroom_v1`, or to `tfpd_lane/`. Import only. |
| New files | Only under `src/m1_t0c1_prefix_v1_50ep/`, `results/m1_t0c1_prefix_v1_50ep/`, `scripts/run_m1_t0c1_prefix_v1_50ep.py`, `tests/test_m1_t0c1_prefix_v1_50ep.py`, `docs/`. |
| CPU | Dataloader / worker count `<= 4`. No process-wide thread saturation. |
| Abort rule | If GPU 1 shows a foreign compute process or the host is anomalous, **stop and write a status report**. Never contend for the card. |

Resource-interaction log target: **zero** interactions with GPU 0 or with the
peer lane's roots.

---

## 1. What changes relative to the sealed 20-epoch pair

The 20-epoch pair trained under a **constant** LR: `Adam(lr=1e-5, weight_decay=0.0)`,
`"scheduler": "None"`, `swa=False`. It is therefore **not** the case that the
50-epoch run is a horizon extension of the same recipe.

Operator decision (2026-09-01): adopt the frozen `warmup_then_cosine` law,
recomputed over the **whole** 50-epoch horizon. Consequences, disclosed rather
than discovered later:

- Peak LR becomes `1e-4`, **10x** the anchor pair's constant `1e-5`.
- The epoch-20 checkpoint of this lane will **not** reproduce the sealed
  20-epoch terminal state digest. The nesting property is deliberately given up.
- The 20-epoch numbers are consequently a **different-recipe reference line**,
  not a nested baseline. Every comparison table must label them as such.

Everything else is held identical to the sealed pair: seed, steps/epoch, batch,
objective, prefix cycle, dropout law, source preparation law, eval law, scorer.

---

## 1.1 Resource parity against the original SPINT M1 recipe

Operator requirement (2026-09-01): this lane's training resources must not be
below SPINT's original. SPINT's released M1 recipe is documented in
`SPINT-main/README.md:48`:

```
python src/train.py data=falcon_m1 model=falcon_m1 trainer=gpu \
    model.optimizer.lr=1e-4 trainer.max_epochs=50 <...>
```

with `configs/model/falcon_m1.yaml` (`Adam`, `weight_decay=0.0`,
`scheduler: null`), `configs/data/falcon_m1.yaml` (`batch_size: 32`,
`window_size: 100`, `calibration_n_trials: 10`, `heldin_session_names: ['']`
which matches **all** held-in-calib files), `configs/callbacks/default.yaml`
(`monitor: val_heldout/r2_mean`, `mode: max`, `periodic_checkpoint`
`every_n_epochs: 10`, `save_top_k: -1`), and `seed: 42`. This repo's own SPINT
reproductions (`configs/experiment/h1_baseline_{paper,released_code}_lr.yaml`)
likewise pin `max_epochs: 50`.

| axis | SPINT M1 released | sealed 20-epoch pair | this lane (50-epoch) | verdict |
| --- | --- | --- | --- | --- |
| epochs | 50 | 20 | **50** | parity |
| batch size | 32 | 32 | 32 | parity |
| window | 100 | 100 | 100 | parity |
| calibration trials | 10 | 10 | 10 | parity |
| seed | 42 | 42 | 42 | parity |
| optimizer | Adam, wd 0.0 | Adam, wd 0.0 | Adam, wd 0.0 | parity |
| peak LR | 1e-4 constant | 1e-5 constant | 1e-5→**1e-4**→1e-6 | peak parity |
| checkpoint cadence | every 10 epochs | best + last only | every 10 epochs | parity |
| selection criterion | `val_heldout/r2_mean` max | train-loss argmin | `val_heldout` max | parity |
| training sessions | 4 (all held-in-calib) | 3 (LODO) | 3 (LODO) | **deficit, mandated** |
| steps/epoch | ~6667 | 4951 | 4951 | **deficit, mandated** |
| total optimizer steps | ~333 350 | 99 020 | **247 550** | **~26% below** |

Two disclosures follow.

**The step deficit is structural, not a budget choice.** SPINT trains on all
four held-in-calib sessions, including 20120924. This lane holds 20120924 out as
the leave-one-session-out fold target, so it trains on three sessions and sees
`4951` instead of `~6667` steps per epoch. Adding the fourth session would
destroy the held-out fold and with it the entire point of the experiment. The
deficit could be compensated by raising the horizon to 70 epochs (`346 570`
steps, strictly above SPINT), which was costed at 9.6 h / 8.4 h per arm and
fits the 12 h bound; the operator elected to keep **50 epochs** and accept the
~26% step deficit as a disclosed cost of the LODO design.

**Mean LR is below SPINT's constant.** SPINT runs a constant `1e-4`. The frozen
warmup+cosine reaches `1e-4` only at step 9902 and decays to `1e-6`, so its
mean LR over the run is materially lower than SPINT's, even though the peak
matches. The operator elected to keep warmup+cosine. Note also, correcting the
framing of §1: the sealed pair's constant `1e-5` was SPINT's *config default*,
i.e. 10x below SPINT's documented M1 command — so relative to the released
recipe the sealed 20-epoch pair was undertrained on both axes at once.

Consequently the only intentional deviations from SPINT's original M1 method in
this lane are: (i) the 3-vs-4 session LODO split, (ii) warmup+cosine in place of
a constant LR, and (iii) the C1 calibration-prefix operator, which is the
experimental variable under test.

---

## 2. Frozen literals

### 2.1 Identity and roots

```
CELL                  = gap_plan.CELL          (CROSS_SESSION_WORST_GROUP_SPINT_M1_V1)
PHASE                 = "m1_t0c1_prefix_v1_50ep"
RESULT_ROOT_RELATIVE  = "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep"
  smoke   -> {root}/smoke
  t0      -> {root}/t0
  c1      -> {root}/c1
  probe   -> {root}/probe
  phase3  -> {root}/phase3_table
```

Stage order: `smoke -> {t0 || c1} -> probe -> phase3`. The two arms run
**concurrently**; every other stage requires an exclusive card.

### 2.1.1 Arm concurrency law (amended 2026-09-01 on measurement)

The lane originally ran arms strictly serially. Measurement during an aborted
t0 killed that plan: with the frozen envelope (`*_NUM_THREADS=1`, mandatory for
step determinism) the arm is single-core CPU-bound, and GPU 1 sat at **~11%
utilization** (0-16% over a 30 s sample), `1201 MiB` of `24576`, `151 W` of
`420 W`, SM `1800 MHz`, throttle flags `0x0`, while **30 of 32 cores were idle**
(load average `2.08`). Serial arms cost 11.7 h; concurrent arms cost ~7 h.

Concurrency is safe for the numbers, and the reason is specific rather than
hopeful: neither trainer enables `cudnn.benchmark` (PyTorch's default is
`False`), so kernel selection does not vary with card load, and the per-step
batch, RNG and LR streams are all CPU-side and per-process deterministic. A
second process on the card cannot reach them.

```
concurrency permitted for   = {t0, c1}, and only with each other
exclusive card required for = {smoke, probe, phase3}
sibling admitted iff        = cmdline invokes this launcher
                              AND --stage in {t0, c1}
                              AND stage differs from the one starting
refuse (fail closed) if     = pid unresolvable to a cmdline
                              OR sibling runs the SAME stage (double-launch)
                              OR more than one sibling present
                              OR GPU 1 used memory > 8192 MiB before start
```

**What concurrency costs, disclosed rather than discovered.** Per-epoch wall
times are inflated by contention and are therefore **not comparable** to the
sealed 20-epoch reference timings. They remain honest receipts of what happened;
they simply lose cross-recipe comparability. Every t0/c1 attempt, launch and
terminal binds `arm_concurrency = "concurrent_pair_on_gpu1"`, the
`concurrent_sibling_stage`, and that disclosure in words, so no reader can
mistake a contended epoch time for the reference. One thing improves: run
conditions are now symmetric across arms, where serial execution would have let
the peer GPU-0 lane load the machine during one arm and not the other.

The epoch-1 abort heuristic is **not** relaxed to accommodate contention. The
bound stays `40 000 s`; concurrent arms project to ~25 200 s, and if contention
were ever bad enough to breach the bound we genuinely want the abort.

### 2.2 Budget

```
SEED                        = 42
EPOCHS                      = 50
STEPS_PER_EPOCH             = 4951        (asserted against prepared.paired_steps_per_epoch)
TOTAL_OPTIMIZER_STEPS       = 247_550
TRAIN_BATCH_SIZE            = 32
OBJECTIVE_LAMBDA            = 0.0
OBJECTIVE_TAU               = 0.01
NO_SWA                      = True
RECORD_STEPS                = 40
SMOKE_STEPS                 = 12
HARD_TIMEOUT_SECONDS_PER_ARM= 43_200      (12 h, enforced at epoch boundary)
```

Measured 20-epoch cost on this GPU: T0 `9919.430263 s` (2.756 h), C1
`8685.517042 s` (2.413 h). Dividing by total steps gives `0.10018` / `0.08771`
s per step, but those are **full-run means and must not be called
post-memoization rates**: epoch 0 fills the episode memo and costs `1851 s`
against roughly `425 s` for later epochs. The true post-memo rates are
`0.08577056604618638` / `0.07313887172792367` s per step (audit, 2026-09-01).

Correct projection = `epoch_0 + 49 x mean(epochs 1..19)`:

| arm | projected 50-epoch | worst-observed-epoch projection |
| --- | --- | --- |
| t0 | **6.29 h** | 7.11 h |
| c1 | **5.43 h** | — |

Both fit the 43 200 s bound with >40% headroom, and even the worst observed
per-epoch cost leaves ~4.9 h of margin. No epoch reduction is authorized.

**Runtime abort heuristic (fail early, not at epoch 47).** Observed per-epoch
cost on t0 jittered `374-484 s` after the memo filled. A sustained ~2x
regression against the worst observed epoch would breach the bound. Therefore,
immediately after epoch 1 completes, the trainer computes
`epoch_0_seconds + 49 * (epoch_1_seconds)` and aborts the arm with an honest
`failure.json` if that estimate exceeds `40 000 s`. This spends ~40 minutes to
learn the run is infeasible instead of 11 hours.

`epoch_i_seconds` is frozen as the **per-epoch wall duration**, obtained by
differencing the cumulative `elapsed_seconds` the epoch rows already carry
(`epoch_0_seconds = elapsed[0]`, `epoch_1_seconds = elapsed[1] - elapsed[0]`).
The comparison is strict: an estimate of exactly `40 000` does not abort.

### 2.3 LR schedule law (the one substantive change)

Reused **verbatim by import** from `tfpd_exploration/src/tfpd_lane/arm_common.py`
(`lr_at_step`, `warmup_steps`, `schedule_params`) — not reimplemented, not copied.

```
kind             = "warmup_then_cosine"
n_epochs         = 50
steps_per_epoch  = 4951
total_steps      = 247_550
warmup_epochs    = 2
warmup_steps     = 9_902
lr_warmup_start  = 1e-5
lr_warmup_end    = 1e-4
lr_final         = 1e-6
phase_local_steps= True   (step is 0-based over the whole 50-epoch run)
```

Application: `Adam` is constructed at `lr = lr_at_step(0, 50, 4951) == 1e-5`
with `weight_decay=0.0` and library-default `betas=(0.9,0.999)`, `eps=1e-8`,
`amsgrad=False` — byte-identical construction to the sealed pair. Before every
`optimizer.step()`, every param group's `lr` is set to `lr_at_step(global_step, 50, 4951)`.

"Full-horizon rearrangement" means the cosine denominator is
`total_steps - warmup_steps = 237_648`, computed once over all 50 epochs. There
is no resumption, no restart, and no per-epoch cosine restart.

The schedule consumes **no RNG**, so the matched-pair equality contract
(§2.6) is unaffected by it.

### 2.4 Arms and operator

Unchanged from the sealed pair, imported from `m1_t0c1_prefix_v1.hook`:

```
ARMS  = ("t0", "c1")
CYCLE = (10, 5, 2)        full=10, half=5, quarter=floor(10/4)=2 (disclosed floor)
t0: operator registered but disabled; effective prefix is the full 10 every step
c1: kwargs["calib_trialized_neural_features"][:, :M], M = CYCLE[step % 3]
eval-mode forwards are never touched by the operator
```

### 2.5 Dropout proof (bound into every attempt, before any data access)

```
block_sha256_both_trees = eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884
m1 line block           = streaming_calibration_exp/src/models/components/spint.py:449-455
SPINT original block    = SPINT-main/src/models/components/spint.py:131-137
byte_identical          = True
law                     = whole-unit Bernoulli F.dropout(ones(B,N), p) after the
                          learnable-ID addition, before fc_in; p = random.uniform(0,1)
                          drawn once per TRAINING forward; inactive in eval
```

### 2.6 Smoke equality contract (must be re-run and all-green for the 50-epoch pair before training)

12 steps per arm, both arms in one process, asserting equality of:

1. `initial_model_state_sha256`
2. per-step batch digest (episode row sample-id stream) and its stream digest
3. per-step Python-RNG state digest (the dropout-p stream) and its stream digest
4. optimizer-step count, one forward per step
5. **new for this lane:** the per-step LR sequence is identical across arms and
   equals `[lr_at_step(i, 50, 4951) for i in range(12)]`
6. eval-mode dropout inactivity: two eval forwards are bit-identical, per arm
7. `t0` effective prefixes are all 10; `c1` recorded prefix sequence is the cycle

### 2.7 Source preparation law (frozen, identical to the sealed full runs)

V6-bound full source preparation: sealed-metadata read, stratum authority fit,
deterministic `common ∩ min-count-2` pruned pools, audit-to-full rebind.

```
raw distinct strata      = {20120926: 47, 20120927: 51, 20120928: 50}
eligible common min2     = 43
min rows per stratum     = 2
accepted_v6_graph_sha256 = 32750a6a9f4a726ee63a59508a210d9f184d224f27cbc8c80354faca021618bf
```

**Stale-field overlay law.** The inherited `prepare()` fragment is reused
verbatim, and it carries the 20-epoch `epoch_budget`, `scheduler` and `adam_lr`
inside nested historical spec objects. Those nested numbers are **not** rewritten
in place — rewriting inherited bodies would break the predecessor digests that
bind this lane. Instead `source_authority.json` publishes a top-level
`superseded_by_50ep_law` block that names each stale key, its inherited value,
and its true 50-epoch value. A reader who trusts the nested numbers without
reading the overlay is wrong, and the overlay exists to say so loudly.

### 2.8 Predecessor digests bound into every attempt

```
20-epoch t0 terminal sha256 = 4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9
20-epoch c1 terminal sha256 = cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99
20-epoch smoke terminal     = ed15417eec1a4a10aafd9e67b13514ebd43f04b0f15cea12bb615185078ceb6f
20-epoch phase3 terminal    = 821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096
```

These are bound as **provenance**, read-only. The 20-epoch roots are never
written to and their sidecars are only verified, never replaced.

### 2.9 Launch envelope (frozen; bound into every attempt and launch)

Promoted into the frozen contract on audit finding: the default interpreter on
this host cannot run this driver at all, so the envelope is load-bearing, not
hygiene. Full rationale and the three-way torch comparison are in
`RUNBOOK_M1_T0C1_PREFIX_50EP_20260901.md` §1.

```
PYTHON                 = /home/xinyuan/miniconda3/envs/spint/bin/python
CUDA_VISIBLE_DEVICES   = 1
PYTHONNOUSERSITE       = 1     # REQUIRED: user-site torch 2.12.0+cu130 shadows
                               # the env and fails on driver 535.309.01
PYTHONPATH             = /home/xinyuan/Work_host/SPINT
OMP_NUM_THREADS        = 1
MKL_NUM_THREADS        = 1
NUMEXPR_NUM_THREADS    = 1
OPENBLAS_NUM_THREADS   = 1
cwd                    = /home/xinyuan/Work_host/SPINT
expected torch         = 2.5.1.post303 / cuda 11.8 / cudnn 90300
expected device        = NVIDIA GeForce RTX 3090, total_memory_bytes 25438126080,
                         capability (8, 6)
```

### 2.10 Device identity law (GPU 1 only)

The frozen `m1_t0c1_prefix_v1.trainer.live_device_profile` must **not** be
called: it runs `nvidia-smi --id 0`, which is physical GPU 0 (NVML indices
ignore `CUDA_VISIBLE_DEVICES`), and it writes GPU 0's uuid/PCI into the profile,
which `SelectedCudaRuntime` then re-queries. This lane supplies its own
profiler:

```
query                  = nvidia-smi --id ${CUDA_VISIBLE_DEVICES}   (i.e. --id 1)
required uuid          = GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86
required pci.bus_id    = 00000000:03:00.0
refused uuid (GPU 0)   = GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
```

The name check alone cannot catch a mix-up: both cards are RTX 3090 24576 MiB.

---

## 3. Checkpoint policy (task 1)

Five checkpoints per arm, at 1-based epochs **10, 20, 30, 40, 50** =
0-based `CHECKPOINT_EPOCH_INDICES = (9, 19, 29, 39, 49)`. Epoch 50 is the
terminal state.

Each checkpoint is `torch.save(model.state_dict(), ...)` bytes published
through the immutable artifact root, therefore `0444` + `<name>.sha256`
sidecar, and entered in a single `checkpoint_manifest.json` carrying, per
entry: filename, file sha256, `state_sha256`, epoch index (0- and 1-based),
`epoch_mean_source_train_loss` at that epoch, the LR at that epoch's final
step, and `strict_reload: True`.

`swa_enabled: False`, `swa_artifact_forbidden: True` retained.

Note: the sealed pair additionally published a
`checkpoint_best_source_train_loss.pt`. This lane keeps that leaf too (the
train-loss-argmin checkpoint) so the manifest schema stays comparable, but the
probe and Phase 3 select **only** from the five sealed epoch checkpoints.

---

## 4. Overfitting probe (task 2)

For each arm, for each of the 5 checkpoints, for each of the 7 sessions:
strict-reload the checkpoint, `model.eval()`, and score under the **frozen
static forward law** reused verbatim from `m1_t0c1_prefix_v1.phase3.score_static`
(Phase-1 exact forward: repeated per-row M10 calibration, batch 128, `no_grad`,
dropout inactive). Metric: variance-weighted last-bin R2 via
`m1_heldin_heldout_gap_v1.physical.variance_weighted_last_bin_r2`.

```
train-fit surface = equal-session mean over (20120926, 20120927, 20120928)   [held-in-calib]
val_heldout       = equal-session mean over (20121004, 20121017, 20121024)   [held-out-calib]
test-fold surface = 20120924                                                  [held-in-calib LODO target]
grid              = 2 arms x 5 checkpoints x 7 sessions = 70 scored cells
deployment        = static_m10 ONLY (operator decision; the anchors were measured
                    under this law). CDM-FIFO is reserved for task 3.
```

### 4.0 Why three surfaces (operator revision 2026-09-01)

The first version of this spec selected on the held-in surface. That was wrong,
and the sealed receipts prove it quantitatively: the training pool retains
`54467/54476`, `49189/49228`, `54766/54783` windows on 20120926/27/28, i.e.
**99.97%** of every held-in window is a training row. Scoring held-in is a
training-fit measurement, so it cannot detect overfitting and cannot legitimately
select a checkpoint.

The replacement follows **SPINT's original protocol** rather than inventing one.
SPINT selects checkpoints on `val_heldout/r2_mean` with `mode: max`
(`SPINT-main/configs/callbacks/default.yaml`), where `val_heldout` is built from
`*held-out-calib*.nwb` (`SPINT-main/src/data/falcon_datamodule.py:503,592-597`),
and it saves a `periodic_checkpoint` every 10 epochs with `save_top_k: -1` —
which is exactly this lane's 10/20/30/40/50 cadence.

Measured feasibility of that surface on M1 (metadata-only read, no pipeline, no
receipts — a design-time feasibility probe, disclosed here):

| session | split | trials | units | eval-mask bins |
| --- | --- | --- | --- | --- |
| 20121004 | held-out-calib | 10 | 64 | 3442 |
| 20121017 | held-out-calib | 10 | 64 | 3342 |
| 20121024 | held-out-calib | 10 | 64 | 6671 |

Exactly 10 trials per session — identical to this lane's M10 calibration
budget — 64 units matching `MODEL_SHAPE`, and roughly 3.3k scoreable windows per
session. Following SPINT, calibration support for these sessions is the session's
**own** 10 trials (`calib_sessions_dict = val_calib_heldout_sessions`).

Stated plainly, because it is the one soft spot in this surface: the calibration
support trials and the scored windows come from the same 10 trials, so the
identity token has seen the *activity* of the windows being scored. It has never
seen their labels, and no gradient ever touches them, so the surface stays
label-clean and training-clean — and it is precisely the FALCON deployment
protocol (calibrate on the session's 10 available trials, then decode). It is
nonetheless a mildly optimistic absolute number, and must not be quoted as a
benchmark result. Its role here is *ranking 5 checkpoints*, for which a shared
constant optimism is harmless.

Two alternatives were measured and rejected:

- **minival of the training sessions** (`*held-in-minival*`, SPINT's
  `val_heldin`): only **2 trials** per session (652/1039/727 bins). Too thin to
  select on, and direction coverage is negligible.
- **carving a chronological tail out of 20120926/27/28**: a genuine validation
  set, but it shrinks the training pool, so `STEPS_PER_EPOCH` leaves the frozen
  4951, the common-stratum pool must be re-derived, and
  `rebind_v3_audit_prepared_to_full`'s `accepted_v6_graph_sha256` binding breaks.
  Rejected as a large change inside frozen-code boundaries for no gain over the
  SPINT-original surface.

The global benchmark held-out is **not** touched and is not even present on
disk: there is no `held-out-minival`, `evalai`, `test` or `formal` split under
`SPINT-main/data/000941/`. The true FALCON evaluation is server-side.

### 4.0.1 Evaluation-only access law for `val_heldout`

The sealed metadata authority
(`sua_exploration/manifests/m1_b20_source_characterization_v1_manifest.json`)
covers only the four held-in-calib sessions and declares
`excluded_path_tokens = ["held-out", "minival", "test", "evalai", "formal"]`.
That exclusion governs the **source/training** authority: held-out data must
never enter training. This lane's use is evaluation-and-selection only, so it
requires its own narrowly-scoped access law, frozen here:

```
schema             = m1_t0c1_50ep_val_heldout_access_law_v1
purpose            = checkpoint selection only, mirroring SPINT val_heldout/r2_mean
training_use       = False
gradient_updates   = 0
optimizer_steps    = 0
labels_used_for    = metric only
opened_paths       = SPINT-main/data/000941/sub-MonkeyL-held-out-calib/
                     sub-MonkeyL-held-out-calib_ses-{20121004,20121017,20121024}_behavior+ecephys.nwb
forbidden_tokens   = ["minival", "test", "evalai", "formal", "held-out-minival"]
fold_target_excluded_from_selection = 20120924
```

Frozen body digests (computed 2026-09-01, bound into every probe attempt):

```
20121004 = 782c1fd090facfb3c50b6a85da8209ca3aea71f66bd4edc13d61fe4a55a3429d
20121017 = 8dd22c67500445ec1e0c11980475badbba9e960a05db16b78aaaad5502b3c652
20121024 = bbeb6c7d66e2c2e9b76506021a6cf8800bc19021b6d1fd03c4eb6de71490bafb
```

The probe must fail closed if any opened body digest differs, if any forbidden
token appears in an opened path, or if 20120924 appears in the selection input.

The law itself is hashed for binding: `law_sha256` is the digest of the
canonical law body **excluding** the `law_sha256` field, so the field can be
published inside the same object it describes. Frozen expected value:

```
law_sha256 = 24205e652628aaf428e4d5bc27cad9c6386589c719db02613e9e3870c7fd341a
```

Verdict rule, frozen before the probe runs. It is evaluated on the
**`val_heldout`** curve (the SPINT selection surface), not on the test fold:

- `OVERFITTING_PRESENT` iff the `val_heldout` curve has an interior maximum,
  i.e. `argmax` of the 5 means is **not** epoch 50 **and** the drop from that
  maximum to epoch 50 exceeds `0.005` R2.
- `OVERFITTING_ABSENT_WITHIN_50EP` iff the `val_heldout` `argmax` is epoch 50.
- `OVERFITTING_INCONCLUSIVE` otherwise (interior max but drop `<= 0.005`).

The same three-way verdict is also computed on the test-fold (20120924) curve
and reported side by side, as a *diagnostic only*. Agreement between the two is
evidence the selection surface is doing its job; disagreement is itself a
finding and must be reported, not reconciled.

**The verdict is descriptive, not inferential.** The audit established that
`0.005` R2 sits *inside* the relevant noise: the sealed held-in per-session SD
is `0.007-0.011`, and the Phase-1 gap CI is `0.020` wide. Moving the verdict to
`val_heldout` improves this — it averages 3 sessions rather than the fold's
n=1 — but no seed-variance receipt exists for this pair, so the threshold cannot
support an inferential claim. Consequences, binding on the report:

- the verdict word (`OVERFITTING_PRESENT` etc.) is a **label on a curve shape**,
  never a statistical test result;
- every probe row must publish per-session values and the across-session SD
  alongside the mean, so a reader can see the drop against the spread;
- if the observed drop is smaller than the `val_heldout` across-session SD at
  the argmax epoch, the receipt must additionally set
  `drop_within_session_spread: true`, and the report must say so in words.

**Spread comparison, frozen.** The across-session SD is population SD,
`numpy.std` with `ddof=0`, matching the predecessor's aggregation. The spread
yardstick is always `val_heldout`'s across-session SD, because the test fold is
a single session and has no across-session spread of its own. The epoch at which
that SD is read is the **argmax of the curve being judged**: for the
`val_heldout` verdict that is `val_heldout`'s own argmax; for the test-fold
diagnostic it is the test fold's argmax, with `val_heldout`'s SD read at that
same epoch. Both verdict payloads must name the epoch they used.

### 4.1 Selection rule (frozen BEFORE any probe number is read)

```
selected_epoch = argmax over CHECKPOINT_EPOCH_INDICES of the val_heldout
                 equal-session-mean static_m10 R2
                 (val_heldout = 20121004, 20121017, 20121024)
tie-break      = smallest epoch index
mode           = max            (mirrors SPINT monitor "val_heldout/r2_mean")
FORBIDDEN as selection input: 20120924 (test fold) and the held-in train-fit
                 surface. The rule must raise if either is passed to it.
```

This is SPINT's original checkpoint-selection criterion, applied to this lane's
5 periodic checkpoints. Unlike the rule it replaces, it is a true validation
criterion: `val_heldout` receives zero gradient updates and shares no windows
with the training pool, so it *can* select an interior epoch and *can* detect
overfitting. The test fold 20120924 remains untouched by selection, so the
reported fold numbers stay honest.

---

## 5. Phase-3 re-readout (task 3)

Rebuild the `{T0,C1} x {static_m10, cdm_activity_fifo_m10} x {held-in, held-out}`
2x2x2 table, twice:

1. at the **epoch-50** checkpoint of each arm;
2. at the **selected epoch** of each arm (§4.1). If the selection is epoch 50
   for both arms, this table is recorded as an alias of (1) with an explicit
   `alias_of_epoch50: true` flag rather than being silently skipped.

Scorers, laws and aggregation reused verbatim from `m1_t0c1_prefix_v1.phase3`
(`score_static`, `score_cdm_fifo`, `open_session_dataset`, `build_table`).
CDM-FIFO law unchanged: `ROLLING_FIXED_M`, support 10, causal, label-free.

### 5.1 Reference line — the sealed 20-epoch table (different recipe, see §1)

| arm | deployment | surface | 20-epoch equal-session mean |
| --- | --- | --- | --- |
| t0 | static_m10 | held-in | 0.7270238995552063 |
| t0 | static_m10 | held-out | 0.5707439184188843 |
| t0 | cdm_activity_fifo_m10 | held-in | 0.6594383120536804 |
| t0 | cdm_activity_fifo_m10 | held-out | 0.5842786431312561 |
| c1 | static_m10 | held-in | 0.7306439876556396 |
| c1 | static_m10 | held-out | 0.5848761796951294 |
| c1 | cdm_activity_fifo_m10 | held-in | 0.6623618404070536 |
| c1 | cdm_activity_fifo_m10 | held-out | 0.5947345495223999 |

Per-session detail (held-in): t0 `{26: 0.7142195701599121, 27: 0.7335554957389832,
28: 0.7332966327667236}`, c1 `{26: 0.7211775779724121, 27: 0.7341418266296387,
28: 0.7366125583648682}`.

20-epoch C1−T0: static held-in `+0.0036200881`, static held-out `+0.0141322613`,
CDM-FIFO held-in `+0.0029235284`, CDM-FIFO held-out `+0.0104559064`.
20-epoch CDM-FIFO−static: t0 held-in `−0.0675855875`, held-out `+0.0135347247`;
c1 held-in `−0.0682821472`, held-out `+0.0098583698`.

Train-loss context: both sealed arms had `best_epoch_index = 19`, i.e. the last
epoch was the best — source train loss had **not** bottomed out at 20 epochs
(t0 final epoch mean `0.042958451470428254`, c1 `0.043094195018068535`). That
is the empirical motivation for extending the horizon.

---

## 6. Receipt discipline

Every stage: fresh root, `attempt.json` published **before** any NWB open,
checkpoint load, model construction, or CUDA call; then `launch.json`; then
bodies; then `terminal.json`. All leaves `0444` + `.sha256` sidecar. Any
exception after the attempt writes an honest `failure.json` and the failure
root is **kept, never deleted**.

`attempt.json` binds, for every stage: pair-spec sha, implementation closure
sha, dropout proof, cycle law, LR schedule law, source preparation law, and the
two 20-epoch terminal digests as predecessors.

Stage gating: `t0`/`c1` refuse to run unless this lane's own 50-epoch smoke
terminal is present, `0444`, sidecar-consistent, and status-OK. `probe`/`phase3`
refuse unless both 50-epoch arm terminals are present and sidecar-consistent.

Test gating: the no-data, no-CUDA test module must be green **before** any GPU
work begins.

---

## 7. Deliverables

1. One arm's smoke equality result (all channels, incl. the new LR channel).
2. Per-arm per-epoch receipt summary (50 rows) + 5-checkpoint manifest digests.
3. Probe curve table: 2 arms x 5 checkpoints x {train-fit mean, val_heldout mean,
   test-fold}, with the two overfitting verdicts (val_heldout and test-fold).
4. Overfitting verdict and the selected epoch under the frozen §4.1 rule.
5. 2x2x2 table at epoch 50 (and at the selected epoch), with the delta against
   the §5.1 reference line, labelled as a cross-recipe delta.
6. Resource-interaction log with the peer lane — target zero.
