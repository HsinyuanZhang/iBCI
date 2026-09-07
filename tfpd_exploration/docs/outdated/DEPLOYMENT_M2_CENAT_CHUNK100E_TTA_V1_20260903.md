# Deployment: M2 Ce-NAT chunk100e continual TTA V1 (post-audit)

Date: 2026-09-03
Status: **replacement submission 581763 pushed** (581749 is a permanent decoder-crash failure). Payload unchanged.
Audit verdict incorporated: external review (conditional GO) + its three implementation
contracts and evidence-chain fixes are implemented verbatim.

## Submitted configuration (auditor-recommended)

- **Ce-NAT / chunk100e / epoch 12** — checkpoint SHA `c5672a2b…` (auditor choice:
  best external 0.37022, no extra gate scalar, smallest online failure surface).
- `IsTestTimeAdaptive=true` (all other attributes false). Label budget 4 (unchanged).

## Package

`tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1/`
- payload `t4_m2_seed42_cenat_chunk100e_tta.pkl` SHA `33181fa3…` (frozen decoder +
  id-encoder state dict + per-tag seed pools/sides; container-side `PayloadIdEncoder`
  reconstructs the encoder op-for-op, strict state-dict load);
- image `spint-t4-m2:cenat-chunk100e-s42-33181fa3`, id
  `sha256:28139d5d173b4ee84e4cc5916b9d043e0608fb06f9b24fc8dfff14246da5dc79`
  (rebuild after the two runtime bugs below; payload bytes unchanged);
- submit helper `submit_evalai_cenat_chunk100e.py` (guarded: preflight → execute with
  confirm literals; challenge 2319 / phase few-shot-test-2319 / team HKU-ECE / private).

## Two runtime bugs found after 581749 (fixed, re-validated)

581749 crashed deterministically at official wave 2 (~351 s) and will never flip
to finished. Root causes in `cenat_chunk_decoder.py`, both fixed before 581763:

1. **Wave-2 IndexError.** Official test loads 13 eval NWBs in waves of 7 then 6.
   `predict` looped `range(self.batch_size)` (=7) after `reset` created only 6
   slots. Fix: loop `n = len(self.slots)`; raise on row/slot mismatch; return `n`
   rows. Regression: `test_wave2_six_file_batch_does_not_indexerror`.
2. **`np.roll` wrap in batch mode.** Per-slot roll of the shared observation
   buffer wrapped every other slot's 50-bin window → near-constant predictions
   (session R2 ≈ −0.02). Batch-1 parity could not catch this. Fix: one roll per
   predict step, then write all rows, decode, then update continual states.
   Regression: `test_batch_roll_matches_independent_batch1_streams`.

The previously disclosed minival Held In R2 = −0.032 was this roll corruption,
not a first-chunk identity poison. After the fix, host and container minival
both return Held In R2 = **0.6608538031578064**.

## Audit contracts implemented

1. **49-bin zero pre-history**: `_SlotState` seeds the candidate buffer with exactly
   49 zero rows at reset; the first chunk completes after real bin 50, matching the
   local padded-stream law bin-for-bin.
2. **Protection semantics renamed honestly**: the pool protects the **chronological
   seed prefix-4** (rows 0–3), NOT the D-opt4 rows; D-opt4 is used only for carrier
   support selection.  Package metadata and docs say so.
3. **Decode-before-commit order** in `predict`: one batch roll → decode with current
   identity → append bin to candidate → on completion gate (median of previous
   candidates; every candidate enters the history) → commit/evict → identity
   recompute effective next bin.
4. Evidence chain sealed: `training.json` sidecar → checkpoint bytes sealed
   0444+sidecar (`artifacts/Ce-NAT_epoch12.sealed.pt`) → V4 score SHA → payload SHA
   (`33181fa3…`) → image id (`28139d5d…`) → validation receipt.
5. Audit-brief wording fixes applied (prefix-4 not D-opt4; median not O(1); encoder
   forward cost to be measured — Docker minival measured total wall ~5 s host /
   seconds in container; 0444 claim scoped to published JSON; PASS wording split into
   deployment-candidate PASS vs paper-claim NOT ELIGIBLE; +0.079 attributed to the
   chunk100e-matched checkpoint + causal energy-gated deployment bundle).

## Local validation results (artifacts/local_validation_receipt.json, post-fix)

| Sentinel | Result |
|---|---|
| Payload audit (schema/arm/chain/13 sessions) | PASS; SHA `33181fa3…` unchanged |
| **Per-bin parity, online decoder vs V4 scorer semantics** | worst \|Δpred\| = **5.2e-08**; worst \|ΔR2 vs V4 rows\| = **1.4e-08**; commit-bin sequences identical |
| **Full-stream batch-7/6 two-wave simulation vs V4** | worst \|ΔR2\| = **1.43e-08**; wall 271 s; 13/13 sessions match (external 0.5356/0.5044/0.3778/0.2440/0.3662/0.1934) |
| Contract (on_done no-op, weights frozen, no gradients) | PASS |
| Host FalconEvaluator minival (batch 7) | completed; `on_done` calls 0; Held In R2 **0.66085**; latency 0.114 |
| **Offline container minival** (`--network none`) | completed; identical Held In R2 **0.66085**; latency 0.037 |
| Batch-slot isolation | batch-2 matches two independent batch-1 streams (max abs ≤ 1e-6 over 400 bins) |

## Predicted official outcome

Local→official historical offset −0.0013 → predicted official held-out R2
≈ **0.368–0.370** (team best 0.295). Class: TTA track. Confirmation value:
deployment evidence on the hidden stream for a bundle selected on public surfaces;
by tier discipline this is NOT a paper-confirmatory claim.

## Push procedure (already executed for 581763)

```text
cd tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1
/tmp/spint-e8-evalai-py38/bin/python submit_evalai_cenat_chunk100e.py --execute \
  --pin-image-id sha256:28139d5d173b4ee84e4cc5916b9d043e0608fb06f9b24fc8dfff14246da5dc79 \
  --confirm-image-id sha256:28139d5d173b4ee84e4cc5916b9d043e0608fb06f9b24fc8dfff14246da5dc79 \
  --confirm-payload-sha256 33181fa3c1543b9af777a17571d9c98d427b7e65d49375716d624e80c99603f3
```
