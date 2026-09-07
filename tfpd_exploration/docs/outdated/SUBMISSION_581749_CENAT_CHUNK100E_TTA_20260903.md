# Submission 581749 — M2 Ce-NAT chunk100e continual TTA

Date: 2026-09-03
Status: **PERMANENT FAILURE.** Official evaluator crashed deterministically at
wave 2 (`IndexError` from looping `range(batch_size=7)` over 6 slots; execution
351.9 s, empty stdout/stderr). Do not wait for a flip. Replacement is
**581763** after the wave-2 + `np.roll` decoder fixes
(`HANDOFF_M2_CENAT_TTA_ONLINE_DECODER_FIX_20260903.md`,
`SUBMISSION_581763_CENAT_CHUNK100E_TTA_20260903.md`).
Decision chain: AJPF-C result → external audit (conditional GO, Ce-NAT recommended)
→ three implementation contracts implemented → equivalence sentinels passed →
user GO → push → official crash.

## Submission facts

| Field | Value |
|---|---|
| Submission ID | **581749** (challenge 2319, phase few-shot-test-2319 / 4599, team HKU-ECE, private) |
| Arm | `cenat_chunk100e_tta` — **Ce-NAT / chunk100e / epoch 12** |
| Image | `spint-t4-m2:cenat-chunk100e-s42-33181fa3`, id `sha256:2bf66a720a229dc902d66ec82ec0f2889c474bf5bddb1d373e9f37011da896b4` |
| Payload | `33181fa3…` (frozen decoder + id-encoder state dict + 13 per-tag seed pools/sides) |
| ECR uuid tag | `f4c13844-7132-4bea-a94f-9211a2196ddb` |
| Attributes | **IsTestTimeAdaptive=true**, IsHeldOutZeroShot=false, IsPretrained=false |
| Chain | training receipt `fe225086…` → checkpoint `c5672a2b…` → V4 score `b7e9ab89…` → payload `33181fa3…` → image `2bf66a72…` |

Deployment: seed pool = session's public first-30 calibration block; during the eval
stream every 100-bin window passing the label-free running-median rate gate commits
(capacity 30, chronological prefix-4 protected) and the cached identity recomputes.
Decode-before-commit; no labels, boundaries, or backprop online.

## Pre-push evidence

- Parity sentinel: online decoder vs V4 scorer — worst |Δpred| 5.2e-08,
  worst |ΔR2| 1.4e-08, commit sequences identical (405 commits), all 13 sessions.
- Offline container minival: completed, latency 0.037, on_done 0.
- Disclosed at push time: held-in minival R2 −0.032. **That number was the
  batch `np.roll` wrap, not a first-chunk effect.** Post-fix minival is 0.66085.
- Audit: external review conditional GO; tier discipline respected
  (deployment evidence, not paper confirmation).

## Prediction and interpretation

Predicted official held-out R2 ≈ **0.368–0.370** (local 0.37022, historical
local→official offset −0.0013; team best 0.295, act30_full). Reading on the
official number:
- ≥ 0.35: deployment bundle confirmed on hidden data; TTA-track result.
- 0.30–0.35: direction confirmed, magnitude partially surface-dependent.
- < 0.295 (below team best static): the local gain did not transfer; bundle
  closed regardless of the honest tier framing.

## Quota after this push

User-rationed submissions: **2 of 2 spent** (581713, 581749). Platform: day
3/6, month 9/50, total 22/100 at push time. Replacement 581763 is a third
push of the same scientific candidate.
