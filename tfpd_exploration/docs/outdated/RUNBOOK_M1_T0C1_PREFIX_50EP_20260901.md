# Runbook — `m1_t0c1_prefix_v1_50ep` execution environment and isolation findings

Companion to `DESIGN_M1_T0C1_PREFIX_50EP_20260901.md`. Written by the
orchestrator before any GPU work. Everything here was verified empirically on
2026-09-01 against the sealed 20-epoch receipts.

---

## 1. Launch environment (must match the sealed 20-epoch run exactly)

The sealed 20-epoch receipts record `torch 2.5.1.post303 / cuda 11.8 /
cudnn 90300`, `total_memory_bytes 25438126080`, capability `(8,6)`. Reproducing
that is **not** the default on this host. There are three torch installs and
two of them are wrong:

| Interpreter | torch | Usable? |
| --- | --- | --- |
| `/usr/bin/python3` (→ `~/.local/.../site-packages`) | `2.12.0+cu130` | **NO** — `RuntimeError: NVIDIA driver too old (found 12020)`. Driver is 535.309.01 (CUDA 12.2 max); cu130 needs newer. |
| `miniconda3/envs/spint/bin/python` **without** `PYTHONNOUSERSITE` | `2.12.0+cu130` | **NO** — the env is python 3.10.15, so `~/.local/lib/python3.10/site-packages` shadows the env's own torch. |
| `miniconda3/envs/spint/bin/python` **with** `PYTHONNOUSERSITE=1` | `2.5.1.post303` | **YES** — exact sealed match. |
| `miniconda3/envs/snn_cuda/bin/python` | `2.3.1+cu121` | Works, but wrong version — would break the runtime-identity attestation. |

`PYTHONNOUSERSITE=1` is therefore **load-bearing**, not hygiene. Verified:

```
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python -c "..."
-> torch 2.5.1.post303  cuda 11.8  cudnn 90300
-> avail True  count 1  name NVIDIA GeForce RTX 3090  mem 25438126080  cap (8,6)
```

### Frozen launch envelope

```
CUDA_VISIBLE_DEVICES=1          # physical GPU 1 only
PYTHONNOUSERSITE=1              # required to reach torch 2.5.1.post303
PYTHONPATH=/home/xinyuan/Work_host/SPINT
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
PYTHON=/home/xinyuan/miniconda3/envs/spint/bin/python
cwd=/home/xinyuan/Work_host/SPINT
```

The four thread caps are the values the sealed run recorded in
`runtime_environment.profile.threads`, so replicating them is both
reproducibility and the "gentle CPU" constraint (one BLAS thread per process,
no dataloader worker pool in this code path at all — the trainer assembles
episodes in-process).

---

## 2. FINDING 1 — the frozen device profiler queries physical GPU 0

`m1_t0c1_prefix_v1/trainer.py:311-314` (frozen, cannot be edited):

```python
completed = _subprocess.run(
    ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader", "--id", "0"],
    check=True, text=True, capture_output=True, timeout=15,
)
```

`nvidia-smi --id` indexes **physical NVML devices and ignores
`CUDA_VISIBLE_DEVICES`**. Proven without ever querying GPU 0:

```
$ nvidia-smi --query-gpu=index,uuid,... --id 1
1, GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86, 00000000:03:00.0, NVIDIA GeForce RTX 3090
$ CUDA_VISIBLE_DEVICES=1 nvidia-smi --query-gpu=index,uuid,... --id 1
1, GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86, 00000000:03:00.0, NVIDIA GeForce RTX 3090
```

`--id 1` returns physical GPU 1 under both settings, i.e. the flag is not
remapped. Therefore `--id 0` in the frozen profiler reads **physical GPU 0** —
the peer lane's card.

Two consequences.

**(a) Isolation.** Running the frozen profiler performs a read-only NVML query
against GPU 0. It consumes no GPU memory, launches no kernel, and cannot
disturb the peer's training — but it does literally violate the operator's
"never query GPU 0" rule. It happens **twice** per stage, because
`SelectedCudaRuntime.__enter__`
(`cross_session_worst_group_v1/source_physical.py:1104-1113`) re-queries
`nvidia-smi --id <profile.uuid>`, and the profile's uuid is GPU 0's.

**(b) Provenance.** The sealed 20-epoch receipts are internally contradictory.
`results/m1_t0c1_prefix_v1/t0/training.json` records:

```
cuda_visible_devices : "1"          <- GPU 1 (correct)
torch_device         : "cuda:0"
total_memory_bytes   : 25438126080  <- from torch, i.e. GPU 1 (correct)
uuid                 : GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9   <- GPU 0 (WRONG)
pci_bus_id           : 00000000:01:00.0                            <- GPU 0 (WRONG)
nvidia_smi_command   : [... "--id", "GPU-ac7388a5-..."]            <- queried GPU 0
```

The *training* unambiguously ran on physical GPU 1: with `CUDA_VISIBLE_DEVICES=1`
torch can only see GPU 1, and every torch-derived field (`name`,
`total_memory_bytes`, `compute_capability`) matches GPU 1. Only the two
nvidia-smi-derived identity fields are wrong. The bug went undetected because
both cards are RTX 3090 24576 MiB, so the memory cross-check in
`SelectedCudaRuntime` passes either way.

### Required fix for this lane

This lane must **not** call the frozen `trainer.live_device_profile`. It gets
its own `device.py` exposing `live_device_profile_gpu1(torch)` that is identical
except:

- it queries `--id ${CUDA_VISIBLE_DEVICES}` (i.e. `--id 1`), so the recorded
  `uuid`/`pci_bus_id` describe the device actually being trained on;
- it asserts the returned uuid is **not** the known GPU 0 uuid
  `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`, as an explicit isolation guard;
- it asserts the returned uuid **is** `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`
  and bus `00000000:03:00.0`.

This is safe downstream: `SelectedCudaRuntime` only requires the profile be
*self-consistent* (`fields[0] == profile.uuid and fields[1] == profile.pci_bus_id`)
and that torch's `name`/`total_memory`/versions match the profile's — all of
which hold for a correct GPU-1 profile. Its own nvidia-smi re-query then also
targets GPU 1, so the whole stage becomes a genuine zero-GPU-0-interaction run
and the receipts finally carry truthful device identity.

Physical identities, for the record:

```
physical GPU 0 : GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9  00000000:01:00.0  (peer lane)
physical GPU 1 : GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86  00000000:03:00.0  (this lane)
```

---

## 3. FINDING 2 — the sealed pair had no LR scheduler

Restating the §1 disclosure of the design doc, because it is the single most
misreadable thing about this lane: the 20-epoch pair ran
`Adam(lr=1e-5, weight_decay=0.0)` at **constant** LR (`plan.py` pins
`ADAM_LR = 1.0e-5` and the pair spec records `"scheduler": "None"`). Adopting
warmup+cosine over 50 epochs raises peak LR to `1e-4`, **10x**. The 50-epoch
run is therefore a *new recipe*, not a horizon extension, and its epoch-20
checkpoint will not reproduce the sealed 20-epoch state digest. Every table
comparing the two must carry `cross_recipe: true`.

---

## 4. Cost model and the 12 h per-arm bound

Measured 20-epoch wall clock, from the sealed `training.json`:

| arm | elapsed | s/step | 50-epoch linear projection |
| --- | --- | --- | --- |
| t0 | 9919.430263 s (2.756 h) | 0.10018 | 6.89 h |
| c1 | 8685.517042 s (2.413 h) | 0.08771 | 6.03 h |

Both projections sit inside the 43 200 s bound with >40% headroom. The
projection is *not* exactly linear, because the episode memo cache is filled
during epoch 0 and reused thereafter, making epoch 0 more expensive than
epochs 1..N-1. That skew makes the linear projection **conservative** (it
over-weights the expensive first epoch across all 50), so it is a safe upper
bound. The audit agent recomputes the per-epoch cost model from the cumulative
`elapsed_seconds` in each epoch row to confirm.

The timeout is enforced at the epoch boundary
(`trainer.py:220`, `elapsed <= HARD_TIMEOUT_SECONDS_PER_ARM`), so a breach
raises inside the stage and produces an honest `failure.json` with the
completed-epoch progress preserved — checkpoints already published up to that
point remain sealed and usable.

---

## 5. Stage order and gating

```
1. tests (CPU, no CUDA)        must be green before any GPU work
2. smoke   -> {root}/smoke     12 steps/arm, LR-equality channel included
3. t0      -> {root}/t0        50 epochs, gated on smoke terminal
4. c1      -> {root}/c1        50 epochs, gated on smoke terminal
5. probe   -> {root}/probe     gated on both arm terminals
6. phase3  -> {root}/phase3_table  gated on both arm terminals
```

Strictly serial on GPU 1, one detached process per stage, `attempt.json`
published before any NWB open / checkpoint load / model construction / CUDA
call. Failure roots are renamed-and-kept, never deleted.

Cost expectation per stage: smoke a few minutes; `t0` ~6.29 h and `c1` ~5.43 h
(corrected cost model, DESIGN §2.2); `probe` roughly **1 hour** for the 70-cell
three-surface grid; `phase3` ~11-22 min. Serial total ~14 h.

### 5.1 Launcher

All stages go through `tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py`,
which refuses to start unless the full envelope is present, the resolved device
is physical GPU 1 by UUID, GPU 1 has **no** foreign compute process, and torch
matches the sealed identity. It queries `nvidia-smi --id 1` only.

```bash
cd /home/xinyuan/Work_host/SPINT
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/xinyuan/Work_host/SPINT \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py \
  --stage <smoke|t0|c1|probe|phase3> --i-am-root-reviewer
```

`--preflight-only` runs every check and exits without reserving a result root.
Verified green 2026-09-01: GPU 1 idle, uuid `GPU-2220ed5d-…`, bus
`00000000:03:00.0`, torch `2.5.1.post303` / cuda `11.8` / cudnn `90300`,
`total_memory_bytes 25438126080`.

---

## 6. Pre-flight checklist

- [ ] `nvidia-smi -i 1` shows no foreign compute process (never a bare `nvidia-smi`)
- [ ] `git status --short` shows no modification to any sealed file
- [ ] CPU-only test module green under the `spint` env with `PYTHONNOUSERSITE=1`
- [ ] `device.py` GPU-1 profiler in place and asserted against the GPU 1 uuid
- [ ] Launch envelope exported exactly as §1
- [ ] Peer-lane roots untouched; GPU 0 never queried
