# Independent pre-GPU audit: `DESIGN_M1_T0C1_PREFIX_50EP_20260901.md`

Auditor: independent audit agent, 2026-09-01. Read-only except this file.
Predecessor sealed: `tfpd_exploration/src/m1_t0c1_prefix_v1/` + `tfpd_exploration/results/m1_t0c1_prefix_v1/`.
Peer lane `paired_anchored_calibration_dropout*` was not listed, read, or touched.
GPU queries: only `nvidia-smi -i 1`. No bare `nvidia-smi`. GPU 0 not queried.

Verdict up front: the scientific literals in spec §2/§5.1 mostly match the sealed bytes. The 12 h per-arm timeout **fits** under a properly memo-aware cost model. Three things will waste or contaminate the GPU run if the implementation follows the spec naively: (1) reusing `live_device_profile` (queries physical GPU 0), (2) launching without the sealed interpreter envelope (`PYTHONNOUSERSITE=1` + `spint` env torch `2.5.1.post303`), (3) treating the sealed 8-module package as a 50-epoch drop-in.

---

## A. Frozen literals vs actual bytes

### A.1 20-epoch terminal digests, sidecars, mode — VERIFIED

Computed `sha256(body)` of each `terminal.json`; sidecar text is exactly `f"{digest}  terminal.json\n"`; both body and sidecar are regular files mode `0o444`.

| leaf | digest (64 hex) | sidecar consistent | mode |
| --- | --- | --- | --- |
| `t0/terminal.json` | `4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9` | yes | `0o444` |
| `c1/terminal.json` | `cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99` | yes | `0o444` |
| `smoke/terminal.json` | `ed15417eec1a4a10aafd9e67b13514ebd43f04b0f15cea12bb615185078ceb6f` | yes | `0o444` |
| `phase3_table/terminal.json` | `821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096` | yes | `0o444` |

Spec §2.8 truncates the phase3 digest to `821818b1...`. Full value is the 64-hex row above. That truncation is a spec defect (bind-time "read the full digest" is not a frozen literal).

### A.2 §5.1 reference table means and per-session detail — VERIFIED (exact)

Every equal-session mean and every held-in per-session static R2 in spec §5.1 equals `phase3_table/table.json` bit-for-bit.

Held-in session SD (not in the spec table; present in `table.json`):

| arm | deployment | surface | `equal_session_sd` |
| --- | --- | --- | --- |
| t0 | static_m10 | heldin | `0.009054644883778266` |
| t0 | cdm_activity_fifo_m10 | heldin | `0.011741198186956571` |
| c1 | static_m10 | heldin | `0.006769333450441317` |
| c1 | cdm_activity_fifo_m10 | heldin | `0.0076868238326458635` |
| * | * | heldout | `0.0` (n=1 session) |

### A.3 §5.1 eight deltas — MISMATCH (truncated to 10 d.p.)

`round(full, 10)` matches the spec. The spec is **not** the `table.json` value. Correct full-precision values from `table.json`:

| quantity | spec (10 d.p.) | actual |
| --- | --- | --- |
| C1−T0 static held-in | `+0.0036200881` | `0.0036200881004333496` |
| C1−T0 static held-out | `+0.0141322613` | `0.014132261276245117` |
| C1−T0 fifo held-in | `+0.0029235284` | `0.0029235283533731726` |
| C1−T0 fifo held-out | `+0.0104559064` | `0.010455906391143799` |
| t0 fifo−static held-in | `−0.0675855875` | `-0.06758558750152588` |
| t0 fifo−static held-out | `+0.0135347247` | `0.013534724712371826` |
| c1 fifo−static held-in | `−0.0682821472` | `-0.06828214724858606` |
| c1 fifo−static held-out | `+0.0098583698` | `0.009858369827270508` |

`phase3.build_table` also **duplicates** the fifo−static pair once per deployment (8 JSON delta entries, 4 unique fifo−static payloads). Scientific content is the 8 values above.

### A.4 `STEPS_PER_EPOCH` / `TOTAL_OPTIMIZER_STEPS` / sealed `optimizer_steps` — VERIFIED

- `50 * 4951 = 247550`. Spec `TOTAL_OPTIMIZER_STEPS = 247_550` is arithmetically consistent.
- Sealed `plan.STEPS_PER_EPOCH = 4951`, `plan.EPOCHS = 20`, `plan.TOTAL_OPTIMIZER_STEPS = 99020`.
- Both sealed `training.json` files: `optimizer_steps == 99020 == 20 * 4951`. Last epoch `global_optimizer_steps_completed == 99020`.
- Provenance of 4951: `PreparedSourceFold.paired_steps_per_epoch` ← `source_lifecycle.paired_epoch_step_count` = `sum(valid_windows[s] // 32)` (`source_lifecycle.py:439-447`). Sealed windows `54476, 49228, 54783` → `1702+1538+1711 = 4951`. Driver asserts equality (`driver.py:193-195`).

### A.5 Elapsed seconds and 50-epoch projection — VERIFIED numbers, MISMATCH of cost model (highest-value item)

Sealed `training.json` totals (spec 6 d.p. is correct rounding):

| arm | actual `elapsed_seconds` | spec |
| --- | --- | --- |
| t0 | `9919.43026295118` | `9919.430263` |
| c1 | `8685.517041608226` | `8685.517042` |

Spec's `0.1002 / 0.0877 s/step post-memoization` is **not** post-memo. It is `total / (20*4951)`:

- t0 full-run mean: `0.10017602770098143` s/step
- c1 full-run mean: `0.08771477521317134` s/step

Each epoch row's `elapsed_seconds` is **cumulative** from `run()` start (`trainer.py:141, 219, 232`). Per-epoch cost = difference of consecutive rows.

**t0 per-epoch wall (s):**
`1850.791312, 384.530790, 387.277677, 472.855443, 479.694842, 473.187604, 390.525137, 409.279885, 383.991143, 373.990654, 384.143644, 377.094684, 421.481407, 405.109379, 466.504637, 484.230261, 451.000351, 479.191684, 463.549946, 380.712209`

**c1 per-epoch wall (s):**
`1805.120545, 361.465131, 361.793387, 360.845190, 362.633986, 361.934982, 364.049607, 363.919152, 363.648542, 361.919190, 360.117497, 362.051356, 361.305980, 362.364193, 361.466660, 360.957207, 360.612606, 362.277287, 363.714056, 363.024517`

| | t0 | c1 |
| --- | --- | --- |
| epoch 0 (memo fill) | `1850.7913120612502` | `1805.1205445630476` |
| epochs 1–19 mean | `424.6500724946688` | `362.1105539249501` |
| epochs 1–19 min / max | `373.990654` / `484.230261` | `360.117497` / `364.049607` |
| true post-memo s/step | `0.08577056604618638` | `0.07313887172792367` |
| spec linear 50ep (`total/20*50`) | `24798.575657 s = 6.8885 h` | `21713.792604 s = 6.0316 h` |
| proper 50ep (`e0 + 49 * mean_post`) | `22658.644864 s = 6.2941 h` | `19548.537687 s = 5.4301 h` |
| conservative 50ep (`e0 + 49 * max_post`) | `25578.074108 s = 7.1050 h` | `19643.551266 s = 5.4565 h` |
| margin vs 43200 s (proper / cons.) | `20541 s / 17622 s` | `23651 s / 23556 s` |

Linear projection is **conservative** (overestimates) because it smears epoch-0 cost across all 50 epochs. Memoization makes epoch 0 ~4–5× a later epoch; epochs 1..N-1 are the ones that repeat 49 times.

t0 post-memo is **not stable** (100 s range, last-5 mean `451.7` vs first-5 `439.5`). c1 is rock-stable (~362 s). t0's jitter is consistent with host contention (peer on GPU 0) plus full-M10 vs cycling prefixes; it is not a memo-cache miss (cache is filled in epoch 0).

**Timeout verdict: 50 epochs fits inside 43200 s per arm with ~5 h margin even at t0's worst observed epoch.** It will not blow at epoch 47 under costs we have actually seen. A 2× blow-up vs t0-max (~968 s/epoch) *would* breach 12 h (`1851+49*968 = 49282 s`). Spec should tell the implementation to abort after epoch 1 if `e0 + 49*(epoch1_delta)` exceeds ~40000 s, rather than discovering this at epoch 47.

Extra 5× `torch.save` inside the timeout window is seconds, not hours. `runtime.close()` runs in `run()`'s `finally` before receipt JSON publish, so epoch-json I/O is outside the 12 h training timer.

### A.6 Dropout block sha `eacb0402...` — VERIFIED (convention recovered)

Target: `eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884`.

Both cited ranges are **byte-identical** 7-line blocks:

- `streaming_calibration_exp/src/models/components/spint.py` lines 449–455 (1-based) = `lines[448:455]`
- `SPINT-main/src/models/components/spint.py` lines 131–137 (1-based) = `lines[130:137]`

**Exact hashing convention that reproduces the digest:**

1. Read the file as UTF-8.
2. `splitlines()` (strips the `\n` from each line; does not keep a trailing newline).
3. Take the 7 cited lines.
4. Join with `"\n"` — **no trailing newline after the last line**.
5. `hashlib.sha256(block.encode("utf-8")).hexdigest()`.

The joined payload is 508 bytes. Equivalent: concatenate the 7 lines *with* `keepends=True`, then `.rstrip("\n")` before hashing.

Hashing the same 7 lines *with* a trailing newline yields `31ed62f79bb28005a0635cc9b75c34412edfb3b8bfd96076df736e27df36a729` — **not** the sealed value. The sealed `plan.py:171` asserts the no-trailing-newline digest and never computes it at runtime.

### A.7 `ACCEPTED_V6_GRAPH_SHA256` and stratum counts — VERIFIED

- `plan.ACCEPTED_V6_GRAPH_SHA256 = 32750a6a9f4a726ee63a59508a210d9f184d224f27cbc8c80354faca021618bf` (`plan.py:250`).
- Both sealed `source_authority.json` files: `accepted_v6_smoke_graph_sha256` is that same 64 hex.
- `deterministic_common_stratum_fallback.original_stratum_counts` lengths: `{20120926: 47, 20120927: 51, 20120928: 50}`.
- `eligible_common_min2_count == 43`, `common_stratum_count == 43`, `minimum_per_session_count == 2`.
- t0 and c1 fallback payloads have identical sha.

### A.8 `best_epoch_index == 19` and final-epoch train losses — VERIFIED

Both `terminal.json` and `training.json`:

| arm | `best_epoch_index` | `best_source_train_loss` == epoch-19 mean |
| --- | --- | --- |
| t0 | `19` | `0.042958451470428254` |
| c1 | `19` | `0.043094195018068535` |

Best checkpoint bytes equal last checkpoint bytes on both arms (last epoch was the argmin). Motivation for a longer horizon is real **for the constant-LR recipe**. It does not automatically transfer to warmup+cosine at 10× peak LR.

### A.9 Other §2 identity literals — VERIFIED against sealed `plan.py`

`CELL = CROSS_SESSION_WORST_GROUP_SPINT_M1_V1`, `SEED=42`, `TRAIN_BATCH_SIZE=32`, `OBJECTIVE_LAMBDA=0.0`, `OBJECTIVE_TAU=0.01`, `NO_SWA=True`, `RECORD_STEPS=40`, `SMOKE_STEPS=12`, `HARD_TIMEOUT_SECONDS_PER_ARM=43200`, `CYCLE=(10,5,2)`, `ARMS=("t0","c1")`. Spec's new `PHASE` / `RESULT_ROOT_RELATIVE` are new-lane values (not in the sealed plan) — that is intended.

---

## B. LR schedule vs `arm_common.py`

Loaded `arm_common.py` via `importlib.util.spec_from_file_location` because `import tfpd_exploration.src.tfpd_lane.arm_common` **crashes** (see C / Blocker 3). Function body: `arm_common.py:54-72`.

### B.1 `lr_at_step(s, 50, 4951)` full precision — VERIFIED claims, exact final is NOT 1e-6

`warmup_steps(4951) = 9902` — VERIFIED (`2 * 4951`).
`total_steps - warmup_steps = 247550 - 9902 = 237648` — VERIFIED.

| s | `lr_at_step(s, 50, 4951)` | spec claim |
| --- | --- | --- |
| 0 | `1e-05` (exact `== 1e-5`) | VERIFIED |
| 1 | `1.0009089072914563e-05` | — |
| 9901 | `9.999091092708544e-05` | last warmup; not 1e-4 |
| 9902 | `0.0001` (exact `== 1e-4`) | VERIFIED |
| 9903 | `9.99999999956748e-05` | first decrease |
| 123775 | `5.3737454896892084e-05` | midpoint-ish of cosine |
| 247548 | `1.0000000173008152e-06` | — |
| 247549 | `1.0000000043252051e-06` | spec `≈ 1e-6` — VERIFIED as approx only |

**Off-by-one: the docstring at `arm_common.py:57-59` ("decays by cosine to exactly 1e-6 at the final step") is false.** Cosine parameterisation hits `progress=1` at `step == total`, which `lr_at_step` rejects (`step >= total` raises). The last legal step is `total-1 = 247549`, `progress = 237647/237648 = 0.9999957920958729`. Residual vs `1e-6` is `4.325205185597581e-15`. Training-irrelevant. Do not write a test that asserts `== 1e-6`.

`schedule_params(...)["lr_at_final_step"]` is the same `1.0000000043252051e-06`.

### B.2 Monotonicity — VERIFIED

Scanned every step in `[0, 247550)`.

- Strictly increasing on `[0, 9902)` (no plateaus).
- Peak at `s=9902` (`== 1e-4`).
- Strictly decreasing on `[9902, 247550)` (no plateaus, including no float64 ties in the cosine tail).
- Seam: `lr(9901) < lr(9902) > lr(9903)`. No plateau at the warmup/cosine join.

Spec's interval language matches this if `[9902, 247550)` is read as "from the peak onward".

### B.3 RNG / side effects of `lr_at_step` — VERIFIED

Source uses only `int` arithmetic and `math.cos`. No `random`, no numpy, no torch. Empirically: `random.getstate()` and `np.random.get_state()` unchanged across many calls. **The module file, however, imports `numpy` and `torch` at import time** (`arm_common.py:28-29`). Importing the module is not RNG-free at process level if torch init has side effects; calling the function after import is.

### B.4 `ADAM_CONSTRUCTOR` vs sealed Adam vs spec "byte-identical" claim — MISMATCH if they use the dict; VERIFIED if they construct like the sealed pair

`arm_common.ADAM_CONSTRUCTOR` (`arm_common.py:32-39`): `{cls: torch.optim.Adam, lr: 1e-4, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.0, amsgrad: False}`.

Sealed trainer (`trainer.py:156-158`):

```
optimizer = torch.optim.Adam(
    model.parameters(), lr=plan.ADAM_LR, weight_decay=plan.ADAM_WEIGHT_DECAY,
)
```

with `ADAM_LR=1.0e-5`, `ADAM_WEIGHT_DECAY=0.0`. Only two kwargs. No betas/eps/amsgrad.

Sealed-run interpreter (`PYTHONNOUSERSITE=1` + `miniconda3/envs/spint/bin/python`): **torch `2.5.1.post303`, CUDA 11.8**. `torch.optim.Adam` defaults on that install:

`lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None`.

Constructing `Adam(..., lr=1e-5, weight_decay=0.0)` therefore does use library-default `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`. Spec's "byte-identical construction to the sealed pair" is **true for that call**. It is **false** if anyone splats `ADAM_CONSTRUCTOR` (that dict's `lr` is `1e-4`, 10× the sealed constructor lr). Setting `param_group["lr"]` before the first `step()` would hide the constructor mismatch for updates, but `state_dict()["param_groups"]` would still differ until mutation, and it is not "byte-identical construction".

Do **not** construct at `ADAM_CONSTRUCTOR["lr"]=1e-4`. Construct at `lr_at_step(0)=1e-5` with `weight_decay=0.0` only.

### B.5 Per-step `param_group["lr"]` mutation on this torch — no evidence of unsafety

2.5.1 Adam `fused` default is `None` (not forced fused). This tree already mutates `param_group["lr"]` as a scheduler. Foreach/fused kernels read `group["lr"]` each step; this is the documented custom-schedule pattern and is equivalent to a LambdaLR that writes the same value. No 2.5.1 behavior found that would make the spec's "before every `optimizer.step()`" mutation non-equivalent to a scheduler.

**Torch-version trap (not in DESIGN §2):** default `python3` / `spint` without `PYTHONNOUSERSITE` resolves to `torch 2.12.0+cu130` from `~/.local`, which this host's driver (535 / CUDA 12.2) cannot run, and whose Adam gained `decoupled_weight_decay=False`. That is a third silent recipe axis. See Blocker 2.

---

## C. Reuse surface — what the sealed 20-epoch package cannot give the 50-epoch sibling

The sealed 8 modules are `plan.py, schedule.py, hook.py, trainer.py, receipts.py, phase3.py, driver.py, __init__.py`. Implementation must import, not edit, them.

### C.1 `receipts.arm_stage_names()` — BLOCKS 50-epoch reuse

`receipts.py:102-107`: builds `epoch_{index:02d}.json` for `range(plan.EPOCHS)` with **`plan.EPOCHS = 20`**. Also only names `checkpoint_best_source_train_loss.pt` and `checkpoint_last.pt` — not five sealed epoch checkpoints. Using this as `expected_terminal_names` after a 50-epoch publish will fail `validate_live` (wrong leaf set). Fork in the new package: 50 epoch jsons + 5 epoch checkpoints + best + last + manifest.

### C.2 `phase3.load_arm_binding` — BLOCKS reuse

`phase3.py:53-81` pins:

- `plan.ARM_ROOT_RELATIVE[arm]` (the **20-epoch** result paths),
- `plan.SEALED_ARM_TERMINAL_SHA256[arm]` (the **20-epoch** terminal digests),
- `manifest["checkpoints"]["best_source_train_loss"]` only.

A 50-epoch phase3 that calls this will either refuse (digest mismatch vs new terminals) or score the old 20-epoch checkpoints. New binding must pin the **new** arm terminals and a chosen epoch checkpoint, not the 20-epoch `SEALED_ARM_TERMINAL_SHA256`.

### C.3 `phase3.strict_load_arm_model` — BLOCKS reuse for epoch checkpoints; primitives underneath are reusable

`phase3.py:238-258` hardcodes filename `checkpoint_best_source_train_loss.pt`. Probe/phase3 must load `checkpoint_epoch_{10,20,30,40,50}` (or whatever names the new manifest uses), **not** the train-loss argmin.

Reusable primitives in `source_physical.py`:

```
def load_exact_m1_spint_model(root: Path) -> Any:                    # :815
def materialize_exact_m1_model(model: Any, *, device: str = "cpu") -> dict[str, object]:  # :838
def strict_reload_checkpoint_bytes(
    body: bytes, *, expected_state_sha256: str, model_factory: Any, device: str,
) -> str:                                                           # :951
```

Write a thin loader that reads an arbitrary checkpoint path + expected state sha and calls those three. Keep `model.eval()` after reload.

`score_static` / `score_cdm_fifo` / `open_session_dataset` have no epoch/root coupling. Reuse those.

### C.4 `trainer.ArmedM1Trainer.prepare()` — operationally reusable; receipt coupling remains

`prepare()` (`trainer.py:80-129`) does sealed-metadata read, V6-bound common-stratum fallback, audit-to-full rebind. No training loop, no LR. Safe to call for the 50-epoch arms.

Caveat: `matched_erm_fold20120924_route_spec()` (`trainer.py:39-55`) builds `CSWGRunSpec` whose `__post_init__` **requires** `epoch_budget == M1_EPOCH_BUDGET == 20`, `adam_lr == 1e-5`, `lr_schedule_literal == "None"` (`cross_session_worst_group_v1/plan.py:403-436` and `source_lifecycle.py:310-311`). You cannot construct a 50-epoch / warmup-cosine `CSWGRunSpec`. `source_authority.json` will still stamp `epoch_budget: 20`, `adam_lr: 1e-05`, `lr_schedule_literal: None`. Overlay the true 50-epoch recipe in the new-lane attempt/launch/training receipts and never read those three fields back as the 50-epoch law.

`prepared.paired_steps_per_epoch`: see A.4. Driver must keep the `== 4951` assert.

### C.5 `trainer.ArmedM1Trainer.run()` — no extension point; subclass must reimplement the loop

Confirmed: one `for global_step in range(total_steps)` with Adam constructed at constant `plan.ADAM_LR`, no `param_group["lr"]` write, checkpoints only serialized at the end (`best`/`last`). No hook for extra epoch checkpoints or an LR schedule.

A reimplementation **must** preserve, in this order of importance:

1. `ProcessRngSnapshot.capture_and_seed(torch, seed=plan.SEED)` at start; `restore()` in `finally` (`trainer.py:144, 273`; snapshot class `source_physical.py:991-1057`). Seed is frozen 42.
2. `operator.attach` after materialize; `operator.detach()` both after eval-proof and in `finally` (idempotent if handle is None).
3. Per-named-parameter finite-gradient assert before `optimizer.step()` (`trainer.py:192-197`).
4. `stream.record_step` on **every** step (epoch-mean coverage invariant: exactly `steps_per_epoch` loss rows; `epoch_mean_loss` at `trainer.py:281-286`).
5. Timeout check **at epoch boundary only**, using cumulative `time.monotonic() - started` vs `HARD_TIMEOUT_SECONDS_PER_ARM` (`trainer.py:215-221`).
6. Eval-dropout proof ordering: `model.eval()` → `eval_dropout_inactive_proof` → `model.train(False)` (`trainer.py:241-243`). Proof itself requires `model.training is False` (`hook.py:155`).
7. Episode memo keyed by **epoch-local** `epoch_step`, populated on miss via `prepared.episode(epoch_step)` (`trainer.py:179-185`). Do not rebuild per epoch.
8. `runtime.close()` in `finally` (`trainer.py:274`). GPU released before heavy JSON publish.
9. New, required: before every `optimizer.step()`, set every param group's `lr` to `lr_at_step(global_step, 50, 4951)`. Construct Adam at `lr=lr_at_step(0)=1e-5`, `weight_decay=0.0` only.
10. New, required: at 0-based epochs `{9,19,29,39,49}`, `torch.save(model.state_dict(), ...)` **without** toggling `model.eval()`. Eval-mode in the middle of training would break dropout/RNG.

### C.6 `receipts.stage_attempt_payload` — BLOCKS `"probe"`

`receipts.py:120-122`: `stage in {"smoke", "t0", "c1", "phase3"}`. New stage `"probe"` fails closed. Fork the allow-list. Also the sealed payload binds `plan.SEALED_SMOKE_TERMINAL_SHA256` / `SEALED_ARM_TERMINAL_SHA256` of the **20-epoch** lane (`receipts.py:134-137`); the 50-epoch attempt must bind **this lane's** smoke/arm terminals as predecessors, and the 20-epoch digests only as provenance.

### C.7 `driver._verify_sealed_smoke` — BLOCKS reuse

`driver.py:67-84` requires `plan.SMOKE_ROOT_RELATIVE` (20-epoch smoke path) and `plan.SEALED_SMOKE_TERMINAL_SHA256` (the 20-epoch smoke digest `ed15417e...`). The 50-epoch t0/c1 stages must gate on **this lane's** 50-epoch smoke terminal, newly computed, not the old literal. Spec §6 already says this; the sealed helper cannot be called as-is.

### C.8 `phase3.build_table` — aggregation reusable; provenance stamp is not

Key format: `f"{arm}_{deployment}_{session_id}"` (`phase3.py:272`). Sessions/deployments/arms in the sealed plan **are** the same tuples the new lane wants (`ARMS`, `PHASE3_DEPLOYMENTS`, `HELDIN_TRAINING_SESSIONS`, `HELDOUT_FOLD_SESSIONS`, `SCORE_ORDER`, `PHASE3_SURFACES`).

If you call the sealed `build_table` it will stamp `phase: "m1_t0c1_prefix_v1"` and the old `PHASE1_MOTIVATION` (`phase3.py:304-317`). Wrap and restamp `phase` / cell identity, or copy the aggregation. Do not ship a 50-epoch table that claims to be the 20-epoch phase.

### C.9 `hook.M1CalPrefixOperator` and `schedule.*` — reusable

No epoch coupling. Cycle frozen to `(10,5,2)` via `schedule.validate_cycle` → `plan.CYCLE`. Eval forwards untouched. Import from the sealed package.

### C.10 `trainer.live_device_profile` — does **not** enforce GPU-1 isolation; it queries physical GPU 0

See E. This function must **not** be called by the new lane.

Assertions it *does* make (`trainer.py:305-327`):

- `CUDA_VISIBLE_DEVICES == "1"` (`plan.BOUND_GPU["cuda_visible_devices"]`)
- `torch.cuda.is_available() and torch.cuda.device_count() == 1`
- `profile.name == "NVIDIA GeForce RTX 3090"`

Those are necessary and insufficient. Both cards are 3090s, so the name check cannot distinguish GPU 0 from GPU 1.

### C.11 Complete reuse-blocker list (implementation handoff)

| sealed symbol | 20-epoch coupling | 50-epoch action |
| --- | --- | --- |
| `plan.EPOCHS`, `TOTAL_OPTIMIZER_STEPS`, `ADAM_LR`, `PHASE`, `RESULT_ROOT_*`, `SEALED_*_SHA256` | 20, 99020, 1e-5, old roots/digests | new `plan.py`; do not import these as the run law |
| `receipts.arm_stage_names` | 20 epoch leaves; 2 ckpts | rewrite |
| `receipts.stage_attempt_payload` | no `"probe"`; binds old terminals | rewrite allow-list + predecessor fields |
| `receipts.phase3_stage_names` | old `plan.PHASE` unused in names; session set OK | reusable for the 16-cell table **if** SCORE_ORDER unchanged; probe needs its own names |
| `driver._verify_sealed_smoke` | old smoke digest + path | new verifier |
| `driver.execute_arm` | `epoch_count=plan.EPOCHS` (20); old smoke gate; old manifest schema | rewrite |
| `driver.execute_phase3` | `load_arm_binding` + best.pt | rewrite; call `score_*` / `open_session_dataset` / wrapped `build_table` |
| `driver.execute_smoke` | no LR-equality channel | rewrite; keep `_validate_smoke_equality` ideas and add LR channel |
| `phase3.load_arm_binding` | 20-epoch terminals + best.pt | rewrite |
| `phase3.strict_load_arm_model` | hardcoded best.pt | wrap primitives |
| `phase3.build_table` | stamps old `PHASE` | wrap/restamp |
| `trainer.run` | constant LR; 2 ckpts at end; 20-shaped via caller | reimplement loop (preserve invariants C.5) |
| `trainer.prepare` | stamps epoch=20 / scheduler None / lr=1e-5 in authority | call it; overlay true recipe elsewhere |
| `trainer.live_device_profile` | `nvidia-smi --id 0` → GPU 0 | **do not call**; new GPU-1 profiler |
| `hook.*`, `schedule.*` | none (cycle only) | import |
| `score_static`, `score_cdm_fifo`, `open_session_dataset` | none | import |
| `base_physical.load_exact_m1_spint_model` / `materialize_exact_m1_model` / `strict_reload_checkpoint_bytes` | none | import |
| `arm_common.lr_at_step` / `warmup_steps` / `schedule_params` | no epoch default; takes `n_epochs` | import **without** going through `tfpd_lane/__init__.py` (Blocker 3) |
| `arm_common.ADAM_CONSTRUCTOR` | `lr=1e-4` | do not splat |
| `arm_common.build_arm_plan` | asserts 48-epoch Gate-2 budget | do not call |

---

## D. Experimental logic (adversarial)

### D.1 §4.1 selection rule — disclosure is honest; task-3 "selected epoch" table is nearly vacuous

Held-in sessions **are** the training sessions (`HELDIN_TRAINING_SESSIONS = (20120926, 20120927, 20120928)`). Argmax of held-in static R2 is a training-fit criterion. With a monotone-enough fit it will pick epoch 50 (or near it). Spec §4.1 already says this.

That does **not** make the *probe curve* vacuous — the held-out curve is the overfitting detector. It **does** make the second 2×2×2 table ("at the selected epoch") scientifically redundant in the expected case, which the spec handles with `alias_of_epoch50: true`. Adequate as a **process** description ("what a held-out-blind operator would have shipped"). Inadequate as a **model-selection** claim.

Minimal fix that still forbids 20120924 leakage, without new data:

- Drop "selected epoch" as a scientific table. Ship epoch 50. Use the probe only as a report.
- If a selection rule is required: among the 5 sealed checkpoints, pick train-loss argmin (already recorded, no extra forwards, still in-sample but at least not R2-on-train). Or split **windows within the three training sessions** (e.g. last 10% of windows per session, never used for gradients) and select on that split — still zero held-out leakage, actually able to see train-window overfitting.

Do not implement a "fix" that peeks at 20120924.

### D.2 §4 verdict rule: 0.005 R2 on one held-out session — threshold is inside noise. Say this loudly.

Held-out surface is a **single** session (`20120924`, n=1). `equal_session_sd = 0` by construction. You cannot declare overfitting present/absent on a population of sessions with n=1.

Existing receipts that quantify variance on this metric:

1. **This pair's own held-in session SD** (`table.json`): t0 static `0.00905`, c1 static `0.00677`. The 0.005 drop threshold is **smaller than the same model's cross-session spread on training days**.
2. **Phase-1 gap bootstrap** (`m1_heldin_heldout_gap_v1/plan.py:101-102, 213-222`; `RESULT_M1_HELDIN_HELDOUT_GAP_V1_20260831.md` §1 and §4): 10 000 session-resamples, seed 42. Gap CI `[-0.133009, -0.112670]` — width `0.020`. The workorder and plan both state the held-out arm is **degenerate** (`degenerate_heldout_arm: true`); the CI is training-session resampling, not held-out-session uncertainty.
3. **No multi-seed training of this T0/C1 pair exists.** Seed variance on 20120924 last-bin R2 is unmeasured. Do not invent one.

The 20-epoch C1−T0 held-out delta is `0.01413` — 0.005 is 35% of the treatment effect you care about, on n=1. An `OVERFITTING_PRESENT` / `ABSENT` / `INCONCLUSIVE` trichotomy at 0.005 will be dominated by that one day's idiosyncrasy. Keep the rule (it is frozen) but **do not let it license a scientific sentence stronger than "descriptive, n=1, threshold inside held-in session SD"**. `formal_benchmark_verdict` must stay false.

### D.3 Probe wall-clock estimate — ~1 hour, not free

Sealed phase3: attempt→terminal mtime delta **677.8 s ≈ 11.3 min** for 16 cells (8 static + 8 fifo).

Static forwards dominate. Per-session static wall from score mtimes (t0): `82.1 s, 74.3 s, 83.1 s` for 26/27/28 after 24. FIFO for all four sessions was ~10 s total (cached identity).

| session | n_windows | static `forward_batches` | fifo `forward_batches` |
| --- | --- | --- | --- |
| 20120924 | 54849 | 429 | 692 |
| 20120926 | 54476 | 426 | 696 |
| 20120927 | 49228 | 385 | 590 |
| 20120928 | 54783 | 428 | 692 |
| **per arm static total** | | **1668** | |

Probe grid = 40 static cells = `5 * 2 * 1668 = 16680` forward batches. Scale from ~78 s/session × 40 = **~3120 s ≈ 52 min**, plus ~10 checkpoint reloads (phase3 loaded the model once per arm; probe cannot). Budget **60–90 min** for probe. Phase3 2×2×2 at epoch 50 is another ~11 min; a non-aliased selected-epoch table another ~11 min.

### D.4 Cross-recipe 50-vs-20 comparison — `cross_recipe: true` is necessary, not sufficient

Peak LR is 10× the sealed constant `1e-5`. The 50-vs-20 delta **cannot** be attributed to horizon. It is a joint (horizon × schedule × peak-LR) contrast. It **cannot** support "we under-trained at 20 epochs" as a causal claim. It **can** support "this new recipe, at 50 epochs, scores X vs the old recipe at 20 epochs" as a descriptive delta.

`cross_recipe: true` on a table is not enough if a reader can miss the 10× LR. Stamp the table with `peak_lr_20ep=1e-5, peak_lr_50ep=1e-4, schedule_20ep=None, schedule_50ep=warmup_then_cosine`. Spec §1 already says this in prose; the receipts must repeat it where the numbers live.

The sealed pair's `best_epoch_index=19` motivates *some* longer run under **constant 1e-5**. It does not motivate this particular 10× schedule. If the scientific question is "does train loss still descend after epoch 20 at the original LR", this lane does not answer it.

### D.5 Other ways to waste ~13 GPU-hours

- Call sealed `driver.execute_arm` → trains **20** epochs at constant 1e-5, writes into a 50-epoch-shaped story. Wrong experiment, full GPU cost.
- Call sealed `live_device_profile` → isolation violation (E) and GPU-0 UUID in the 50-epoch receipts, repeating the sealed pair's false attestation.
- Launch with default `python3` / `spint` without `PYTHONNOUSERSITE=1` → `torch 2.12.0+cu130`, driver too old, or a silent third recipe axis.
- Forget per-step LR writes → 50 epochs at constant 1e-5 while receipts claim warmup-cosine. Uninterpretable.
- Toggle `model.eval()` when sealing intermediate checkpoints → corrupts subsequent training.
- Use `ADAM_CONSTRUCTOR["lr"]=1e-4` as the constructor lr and forget to mutate before step 0 → first update at 10× intended start.
- Let `build_table` stamp `phase: m1_t0c1_prefix_v1` on 50-epoch numbers.
- Treat probe `OVERFITTING_*` as a confirmatory result on n=1.

---

## E. Isolation verification

### E.1 GPU 1 currently free — VERIFIED (query was `nvidia-smi -i 1` only)

```
index, uuid, name, pci.bus_id, utilization.gpu, memory.used, memory.total
1, GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86, NVIDIA GeForce RTX 3090, 00000000:03:00.0, 0 %, 23 MiB, 24576 MiB
```

`--query-compute-apps` on GPU 1: empty. No foreign compute process. GPU 0 was not queried.

### E.2 Git status of sealed files — VERIFIED no `M` on the sealed 8 modules or either `spint.py`

`tfpd_exploration/src/m1_t0c1_prefix_v1/`, its test, its script, and `tfpd_lane/arm_common.py` are **untracked** (`??`), not modified. `streaming_calibration_exp/src/models/components/spint.py` and `SPINT-main/src/models/components/spint.py` are tracked and clean. The working tree is dirty elsewhere (unrelated `M` files). No edits were made to sealed lane files during this audit.

### E.3 `live_device_profile`'s `nvidia-smi --id 0` **does violate** "never query GPU 0"

Quote, `trainer.py:305-327`:

```
visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
_require(visible == plan.BOUND_GPU["cuda_visible_devices"], ...)  # must be "1"
...
completed = _subprocess.run(
    ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader", "--id", "0"],
    ...
)
...
_require(profile.name == "NVIDIA GeForce RTX 3090",
         "armed trainer device is not the bound RTX 3090 GPU 1")
```

`nvidia-smi --id` is an **NVML physical index**. It does **not** honor `CUDA_VISIBLE_DEVICES`. CVD remaps the CUDA runtime only (`torch.cuda` device 0 → physical GPU 1 when CVD=`1`).

Empirical proof **without querying GPU 0**:

| source | uuid | pci |
| --- | --- | --- |
| `nvidia-smi -i 1` (this audit) | `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` | `00000000:03:00.0` |
| sealed t0/c1 `runtime_environment.profile` (from `nvidia-smi --id 0` + torch props) | `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` | `00000000:01:00.0` |

Those are two different cards. Torch-derived fields in the same profile (`name`, `total_memory_bytes=25438126080`, `compute_capability=(8,6)`, `cuda_visible_devices="1"`) are GPU 1. The nvidia-smi-derived `uuid`/`pci_bus_id` are GPU 0. Both cards are 3090 24576 MiB, so `SelectedCudaRuntime`'s memory cross-check (`source_physical.py:1096-1113`) cannot catch the swap.

Then `SelectedCudaRuntime.__enter__` re-queries:

```
["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.total",
 "--format=csv,noheader,nounits", "--id", self.profile.uuid]
```

(`source_physical.py:1104-1107`). If `profile.uuid` is GPU 0's, this is a **second** GPU-0 query per stage.

**Definitive: reusing `live_device_profile` queries physical GPU 0 twice per stage (profile `--id 0`, then runtime `--id <gpu0-uuid>`). That violates spec §0 and the operator isolation contract.** It is a read-only NVML query — it will not knock the peer off GPU 0 — but it is still a query of GPU 0, and it writes GPU 0's identity into this lane's receipts.

Sealed path that performs this: `driver.execute_*` → `trainer.live_device_profile` → `ArmedM1Trainer.run` → `SelectedCudaRuntime`. `hook`/`schedule`/`phase3.score_*` do not call nvidia-smi. No bare `nvidia-smi` in the sealed 8 modules.

Fix (new package only; do not edit frozen `trainer.py`): a GPU-1 profiler that queries `--id 1` / `-i 1` (or `--id ${CUDA_VISIBLE_DEVICES}` after asserting CVD=`1`), asserts uuid `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` and bus `00000000:03:00.0`, and asserts uuid is **not** `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`. `SelectedCudaRuntime` will then re-query GPU 1 by uuid. Zero GPU-0 interactions.

---

## Orchestrator handoff

### 1. BLOCKERS (fix before any GPU work)

1. **Do not call `trainer.live_device_profile`.** It runs `nvidia-smi --id 0` (physical GPU 0) and poisons the profile UUID. New-lane `live_device_profile` must query `-i 1`, pin GPU 1 uuid `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` / `00000000:03:00.0`, refuse GPU 0 uuid `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`. Then `SelectedCudaRuntime`'s uuid re-query stays on GPU 1. DESIGN §0 forbids the query; DESIGN never tells the implementer to replace this function — that hole is the bug.

2. **Freeze the launch envelope. DESIGN §2 does not.** Sealed pair ran `torch 2.5.1.post303 / cuda 11.8 / cudnn 90300`. That is `miniconda3/envs/spint/bin/python` **with `PYTHONNOUSERSITE=1`**. Without it, `python3.10` user-site `torch 2.12.0+cu130` shadows the env; this driver's 535/CUDA 12.2 cannot run cu130. Bind: `CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=<repo> OMP/MKL/NUMEXPR/OPENBLAS_NUM_THREADS=1 PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python`. Put those strings in every attempt.

3. **`from tfpd_exploration.src.tfpd_lane.arm_common import lr_at_step` is dead.** `tfpd_lane/__init__.py:13` does `from src.tfpd_lane import mech_diag, pregate, receipt`, which requires `tfpd_exploration` on `sys.path` as `src`. Frozen: cannot edit `tfpd_lane/`. Load `arm_common.py` by file location, or `sys.path.insert(tfpd_exploration)` and `from src.tfpd_lane.arm_common import lr_at_step, warmup_steps, schedule_params`. Do not import `build_arm_plan` (48-epoch assert). Do not splat `ADAM_CONSTRUCTOR`.

4. **Do not drive the 50-epoch run through sealed `driver.execute_*` / `receipts.arm_stage_names` / `phase3.load_arm_binding`.** Those will train 20 epochs, expect 20 epoch leaves, or pin the old terminals. See C.11. Missing this wastes 7 h producing the wrong experiment or failing `validate_live` after the 7 h.

### 2. WARNINGS (results harder to interpret)

1. Held-in argmax selection is training-fit. Task-3 "selected epoch" table will almost certainly alias epoch 50. Frozen, disclosed, still nearly vacuous. Probe held-out curve is the actual overfitting instrument.
2. Overfitting threshold `0.005` R2 on n=1 session 20120924 is inside held-in session SD (`0.007–0.011`) and inside Phase-1 gap-CI width (`0.020`, degenerate held-out arm). No seed-variance receipt for this pair. Descriptive only.
3. 50-vs-20 delta is not a horizon effect. Peak LR 10×. Stamp `peak_lr` and `schedule` on every comparison row, not just `cross_recipe: true`.
4. Sealed `best_epoch_index=19` motivates longer **constant-1e-5** training. This recipe does not test that claim.
5. t0 post-memo epoch cost jittered 374–484 s. Under observed max, 50ep still has 4.9 h margin. A 2× contention spike vs t0-max would breach 12 h. After epoch 1, compute `e0+49*delta_1` and abort if >40000 s.
6. `source_authority` will still say `epoch_budget=20`, `scheduler=None`, `adam_lr=1e-5`. Overlay the true law or readers will cite the wrong recipe.
7. `build_table` will stamp `phase=m1_t0c1_prefix_v1` if reused raw.
8. `arm_common` docstring "exactly 1e-6 at the final step" is false (`lr_at(247549)=1.0000000043252051e-06`). Do not assert exact equality.
9. Probe is ~1 hour of GPU 1 (40 static forwards). Budget it; it is not "a few minutes".
10. Working tree is dirty on unrelated files. Sealed lane files are clean/untracked. Do not train against an accidental dirty M1 datamodule; the two `spint.py` dropout sources are currently clean.

### 3. Spec literals that MISMATCHED (correct value)

| spec location | spec wrote | actual |
| --- | --- | --- |
| §2.8 phase3 digest | `821818b1...` (truncated) | `821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096` |
| §2.2 "post-memoization" s/step | `0.1002 / 0.0877` | those are **full-run** means; true post-memo `0.08577056604618638` / `0.07313887172792367` |
| §5.1 eight deltas | 10 decimal places | full values in A.3 |
| §2.3 / arm_common docstring | cosine "exactly 1e-6 at the final step" | `1.0000000043252051e-06` at `s=247549`; exact 1e-6 is at illegal `s=247550` |
| §2.3 "byte-identical Adam" vs `ADAM_CONSTRUCTOR` | implied reuse of arm_common constructor | `ADAM_CONSTRUCTOR["lr"]=1e-4`; sealed call is `Adam(..., lr=1e-5, weight_decay=0.0)` |
| §C implication that `live_device_profile` enforces GPU 1 | CVD=`1` + name 3090 | also queries physical GPU 0; sealed receipts carry GPU 0 uuid |

Elapsed totals, means, per-session R2, terminal digests, steps, V6 sha, strata 47/51/50/43, dropout sha (under the recovered convention), `best_epoch_index=19`, final losses: **match**.

### 4. 12 h timeout verdict (proper cost model)

Fits. Proper projection `e0 + 49 * mean(epochs 1–19)`: t0 **6.29 h**, c1 **5.43 h**. Conservative (t0 max post-memo): **7.11 h**. Linear spec numbers 6.89 / 6.03 h are safe **upper** bounds because they overweight epoch 0. Margin at t0-max: **4.89 h**. Not a blocker. Add a live abort after epoch 1 if the projection exceeds ~40000 s.

### 5. `nvidia-smi --id 0` isolation verdict

**Violates the rule.** NVML index 0 is physical GPU 0 regardless of `CUDA_VISIBLE_DEVICES=1`. Sealed receipts prove it: attested uuid/PCI are GPU 0's; `nvidia-smi -i 1` returns a different uuid/PCI for GPU 1. `SelectedCudaRuntime` then queries that GPU-0 uuid again. New lane must not call the frozen profiler.

### 6. Reuse-blocker list

The table in C.11 is the implementation checklist. Short form: reuse `hook`, `schedule`, `prepare()` (with recipe overlay), `score_static`/`score_cdm_fifo`/`open_session_dataset`, and the three `base_physical` load primitives. Reimplement plan, receipts names/attempt allow-list, driver stages, `run()` loop, arm binding, checkpoint loader, device profiler, smoke LR channel, probe stage. Import `lr_at_step` without `tfpd_lane/__init__.py`.

GPU 1 is idle. Sealed files are unmodified. Spec science literals are mostly right. Spec reuse/isolation/launch-envelope holes will waste the run if ignored.
