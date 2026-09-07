# Submission 581763 — M2 Ce-NAT chunk100e continual TTA (decoder-fix replacement)

Date: 2026-09-03
Status: **finished.** Official held-out R2 **0.2333** (below team-best static 0.295). Local
continual gain did not transfer. Replaces 581749 (crash).
Replaces: **581749** (permanent wave-2 `IndexError` + batch `np.roll` wrap). Payload
bytes are unchanged (`33181fa3…`); only `cenat_chunk_decoder.py` was rebuilt into
a new image.

## Submission facts

| Field | Value |
|---|---|
| Submission ID | **581763** (challenge 2319, phase few-shot-test-2319 / 4599, team HKU-ECE, private) |
| Arm | `cenat_chunk100e_tta` — **Ce-NAT / chunk100e / epoch 12** |
| Image | `spint-t4-m2:cenat-chunk100e-s42-33181fa3`, id `sha256:28139d5d173b4ee84e4cc5916b9d043e0608fb06f9b24fc8dfff14246da5dc79` |
| Payload | `33181fa3…` (frozen decoder + id-encoder state dict + 13 per-tag seed pools/sides) |
| ECR uuid tag | `bd1cf674-cef2-4b88-ada1-a3c0cd44d5e9` |
| Attributes | **IsTestTimeAdaptive=true**, IsHeldOutZeroShot=false, IsPretrained=false |
| Chain | training receipt `fe225086…` → checkpoint `c5672a2b…` → V4 score `b7e9ab89…` → payload `33181fa3…` → image `28139d5d…` |
| Submitted at | 2026-09-03T09:39:13.910749Z |

Deployment: seed pool = session's public first-30 calibration block; during the eval
stream every 100-bin window passing the label-free running-median rate gate commits
(capacity 30, chronological prefix-4 protected) and the cached identity recomputes.
Decode-before-commit; no labels, boundaries, or backprop online.

## Fixes versus 581749

1. `predict` iterates `len(self.slots)` (official waves are 7 then 6), not
   `batch_size`.
2. Shared `observation_buffer` is rolled once per step, then all rows are written.
   Per-slot `np.roll` had wrapped every other slot and produced session R2 ≈ −0.02.

## Pre-push evidence (post-fix)

- Wave-2 and batch-isolation pytest: 2 passed.
- Parity sentinel: online decoder vs V4 scorer — worst |Δpred| 5.2e-08,
  worst |ΔR2| 1.4e-08, all 13 sessions.
- Full-stream batch-7/6 simulation vs V4: worst |ΔR2| 1.43e-08, wall 271 s.
- Host minival: Held In R2 **0.66085** (was −0.032 under the roll bug).
- Offline container minival (`--network none`): identical 0.66085, latency 0.037,
  on_done 0.

## Official result

| Split | R2 Mean | R2 Std |
|---|---:|---:|
| Held Out | **0.23331** | 0.09492 |
| Held In | 0.51203 | 0.05043 |

Normalized latency 0.190. Worker marked `finished` at 18:08 (execution_time field 0.10 s is the
platform scoring tick, not decode wall). Predicted ~0.368–0.370 did not appear; 0.233 is below
the static team best 0.295 (act30_full). Local +0.0785 continual gain did not transfer.

## Prediction and interpretation

Predicted official held-out R2 ≈ **0.368–0.370** (local 0.37022, historical
local→official offset −0.0013; team best 0.295, act30_full). Reading on the
official number:
- ≥ 0.35: deployment bundle confirmed on hidden data; TTA-track result.
- 0.30–0.35: direction confirmed, magnitude partially surface-dependent.
- < 0.295 (below team best static): the local gain did not transfer; bundle
  closed regardless of the honest tier framing.

## Quota after this push

Platform at preflight: day 3/6, month 9/50, total 22/100, active 0. After
registration: day 4/6, month 10/50, total 23/100.
