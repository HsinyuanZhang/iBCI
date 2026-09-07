# Work Order: M2 CAMR Diagnostics V1 (zero-training, cross-record)

Date: 2026-09-03
Status: authorized zero-training diagnostics only (CPU); no GPU, no training, no submission
Input: official result of 581763 (HO 0.233 / HI 0.512 — below static champions) and the CAMR proposal (calibration-anchored dual memory)

## Question

Is the official regression caused by (a) fine-tune damage to cross-recording
static generalization, (b) the continual update law (cold start / poisoning),
or (c) absence of any cross-record query information?  Per the CAMR plan's
three zero-training diagnostics, run on **cross-record episodes** (seed from
one recording, query from the other recording of the same date).

## Episodes

Same-date opposite-run pairs (seed recording → query recording):

- within (6): 1019 R1↔R2, 1020 R1↔R2, 1027 R1↔R2
- external (4): 1030 R1↔R2, 1124 R1↔R2

Query surface: external = all official windows of the query session; within =
post-30 common windows.  Seed = partner run's first-30 calib block +
D-opt4/ridge T4 (public calib labels, unchanged law).

## Arms

| Arm | Checkpoint | Seed | Memory |
|---|---|---|---|
| S0-same | pretrained 25d7 | same record | none (sealed act30 anchor; must match ≤1e-7) |
| S0-cross | pretrained 25d7 | cross record | none |
| S1-same | Ce-NAT epoch12 | same record | none |
| S1-cross | Ce-NAT epoch12 | cross record | none |
| S2-cross | Ce-NAT epoch12 | cross record | G0 (first candidate unconditional; running median after) |
| S3-cross | Ce-NAT epoch12 | cross record | G1 (τ0 = median energy of the calib-30 rows; threshold = causal median over calib energies ∪ past candidates) |
| S4-cross | Ce-NAT epoch12 | cross record | G2 (warm-up K=4: history only; running-median commits from candidate 5) |
| LP-cross | pretrained 25d7 | cross record | none, late-pooled identity (per-trial post_pool then mean) — headroom probe |

Decode-before-commit, capacity 30, chronological prefix-4 protected (frozen
laws from the shipped decoder).  Chunk origin = query-stream bin 0 with the
49-row zero pre-history (deployment convention).

## Pre-registered reading rules (no tuning)

- S1-cross ≈ S0-cross → fine-tuning did not break cross-record statics;
  S1-cross ≪ S0-cross → fine-tune damage (CAMR's frozen-anchor is mandatory).
- S2 ≪ S1-cross → update law is the main damage source; G1/G2 recovery points
  to the cold start; no recovery with S3/S4 either → close the query-memory law.
- LP-cross vs S0-cross: any late-pooling identity headroom on cross-record.
- Exp2: commit-indexed R² bands (0, 1, 2–4, 5–10, 10+) on S2 — early drop =
  cold start; late drop = stale memory; never positive = no cross-record
  query information (stop).
- All numbers descriptive diagnostics; no promotion, no submission from this
  cell.  CAMR training requires: LP/C-NAT cross-record residual headroom
  ≥ +0.01 AND S3-or-S4 ≥ S1-cross.

## Output

`tfpd_exploration/results/m2_camr_diag_v1/diag.json` (0444+sidecar) + RESULT doc.
