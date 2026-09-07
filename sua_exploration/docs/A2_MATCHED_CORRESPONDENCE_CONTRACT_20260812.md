# A2 — matched target-subject-shift × carrier-content experiment contract

**Date:** 2026-08-12  
**Status:** root-audited v2 contract. **Authorizes no GPU run.** The former correspondence-v1
estimand and its 12-run matrix are withdrawn before launch.  
**Screen ID:** `a2_matched_subject_shift_v2`  
**Contract document SHA-256:** bind at launch via `contract_sha256` in the inert runner; recompute with `sha256sum` on this file before any authorization.

---

## Claim under test (must be able to fail)

Under one frozen source-training protocol, the **carrier marginal gain** `(T4 - Z4)` is larger
when the unchanged sub-C-trained decoder is scored on external subject M than when it is scored
on held-out sub-C development sessions. This is a test of robustness to a target-subject
distribution shift, **not** a test of stable unit correspondence: neither B3S nor T4 requires
unit-index correspondence across sessions.

If the pre-registered interaction gate fails, section 1.2 correspondence framing in `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` is **not printable** and must be dropped or rewritten as unmatched historical hints.

The primary estimand is the **external-minus-within interaction on carrier gain**, not either
main effect alone. A positive result is scoped to the observed C-to-M transfer; it does not prove
population-wide cross-subject invariance from one external subject.

---

## Historical hints (not baselines — protocol-mismatched)

| Quantity | Value | Source | Why not a baseline |
|---|---:|---|---|
| Within-subject activity-only Z4 | `0.326008` | SUA lattice M30 sub-C | Different screen substrate; not this matrix |
| Cross-subject activity-only Zero4 | `−0.057766` | subject-M V9 12-epoch | M50 calibration / trial-50 query |
| Cross-subject carrier T4 | `0.356828` | subject-M V9 M50 | Same V9 protocol mismatch |

This experiment produces the **matched** replacement for those numbers.

---

## Frozen matrix

| Factor | Levels |
|---|---|
| Scoring domain | `within_subject` (sub-C 6 validation sessions) · `external_subject_M` (15 admissible sub-M sessions) |
| Arm | `z4` activity-only (`mask_standardized_t4`) · `t4` carrier |

| Training cell | Arm | Variant | Side feature group | Training sessions | Scoring sessions from the same checkpoint | Seeds |
|---|---|---|---|---|---|---|
| `source_z4` | z4 | B3S | z4 | sub-C 27 train | sub-C 6 val **and** sub-M 15 external | 42, 43, 44 |
| `source_t4` | t4 | B3S | t4 | sub-C 27 train | sub-C 6 val **and** sub-M 15 external | 42, 43, 44 |

**Total GPU cells:** `2 arms x 3 seeds = 6` fresh source-training runs. Re-training a
checkpoint separately for the two scoring domains is forbidden. Each checkpoint SHA must appear
unchanged in both domain receipts. Subject J may be added only as a separately named external
replication after a CPU eligibility audit; it is not pooled silently with M.

### Frozen protocol (identical across all 6 training cells and 12 domain-specific evaluations)

| Parameter | Frozen value |
|---|---|
| Activity calibration trials | chronological first `30` rewarded trials |
| T4 label / side-feature pool | chronological first `30` rewarded trials |
| Evaluation start | trial index `30` (windows strictly after trial 30) |
| Epoch budget | `12` epochs, **no early stopping** |
| Checkpoint rule | arithmetic mean of validation R² over epochs `5–12` |
| Loss mode | `task_only` |
| Identity mode | `calibrated` |
| Split manifest | `configs/subc_co_27_6_strict_train_val_manifest.json` |
| Signal view | SUA |
| Task | CO |
| `max_units_exclusive` | `100` |
| Teacher checkpoint | `checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt` |
| Data (train) | `data/dandi_000688/sub-C` |
| Cache | `cache/dandi688_subc_co_v1` |
| Cross-subject scoring data | `data/dandi_000688/sub-M` (score-only; no sub-M train split) |
| Batch size | `32` |
| Learning rate | `1e-4` |
| Window / trial | `50` / `100` bins, `20 ms` |
| Decoder | coupled, trainable |
| Formal test NWBs | **never opened** |

Cross-subject cells use the **same M30 / trial-30 boundary** as within-subject cells. The legacy subject-M V9 trial-50 policy is **prohibited** in this matrix.

### Target-session carrier provenance

For every scored session, both arms construct the session-local T4 carrier
from the chronological first 30 rewarded **target-direction** labels: the T4
arm uses it directly and Z4 constructs the same carrier before applying its
standardized T4 mask. Domain receipts therefore record
`target_session_carrier_fit_performed=true` and
`target_direction_labels_used_for_carrier=true`. This is not target-domain
decoder adaptation: source-train behavior/T4 normalizers remain unchanged,
target velocity labels are not used for weight updates, and no backward
gradients or decoder-weight updates occur. The corresponding receipt fields
are `target_velocity_labels_used_for_weight_updates=false`,
`backward_gradients=false`, and `decoder_weight_updates=false`.

---

## Interaction estimand and power honesty

### Primary interaction (carrier gain contrast)

For each seed `s` and scoring domain `d`:

```text
carrier_gain[d, s] = mean_session_R2(t4[d,s]) - mean_session_R2(z4[d,s])
```

Primary interaction statistic:

```text
interaction[s] = carrier_gain[external_subject_M, s] - carrier_gain[within_subject, s]
mean_interaction = mean(interaction[s])   over s ∈ {42,43,44}
σ_interaction_paired = stdev(interaction[s], ddof=1) / √3
```

### Formal interaction gate (pre-registered)

All must pass for `subject_shift_interaction_effective`:

1. `mean_interaction ≥ +0.03`;
2. all three seed-level `interaction[s]` values are strictly positive;
3. a crossed seed-by-session bootstrap has a 95% CI lower bound `> 0`: the paired seed resample is
   shared across arms/domains, and each scoring domain draws one session roster shared across the
   sampled seeds so repeated sessions are not pseudoreplicated;
4. every receipt pair has identical checkpoint SHA, query policy, normalizer authority, and
   per-domain session roster across T4/Z4.

**Power assessment (honest).** Three seeds and one external subject do not support a population-
level subject-generalization p-value. The former exact two-sided Wilcoxon gate was removed because
with three observations its minimum attainable two-sided p-value exceeds `0.05`. If the interval
is unresolved, report the descriptive table without a subject-generalization claim.

### Descriptive statistics (always reported)

| Output | Definition |
|---|---|
| 2×2 cell means | grand mean R² per `(regime, arm)` over seeds and sessions |
| Carrier gain per regime | `t4 − z4` with per-session and per-seed breakdown |
| `mean_interaction` | as above |
| Historical hint comparison | table only; no claim of replication until gates pass |

---

## Frozen acceptance rule (FP32 mainline format — secondary contrasts)

Secondary gates (descriptive, not sufficient alone):

| Contrast | Rule |
|---|---|
| Within-subject `t4 − z4` | FP32 mainline five-gate rule on 6 sub-C sessions |
| Cross-subject `t4 − z4` | same five-gate rule on 15 sub-M sessions |

Each contrast requires: mean paired delta `≥ +0.03`, all seed means positive, all session means positive, bootstrap 95% CI lower `> 0`, exact Wilcoxon `p ≤ 0.05`.

---

## Outcome authorization map

| Outcome | Authorizes | Forecloses |
|---|---|---|
| `subject_shift_interaction_effective` | Printable C-to-M target-shift robustness result; optional J replication | No correspondence claim |
| `interaction_indeterminate` | Descriptive 2×2 only; no mechanism or generalization claim | Strong subject-shift narrative in paper |
| `interaction_ineffective` (`mean + 2σ < +0.03`) | Drop section 1.2 subject-shift explanation | B12 and B4 as primary story |
| Within `t4−z4` fails but cross succeeds | Cross-subject carrier utility only | Within-subject correspondence repair claim |
| Within succeeds, interaction fails | Within-subject benefit only | Cross-subject repair narrative |

---

## Data isolation boundary

**Sealed formal-test sessions (never load, score, or reference):**

- `sub-C_ses-CO-20151113`
- `sub-C_ses-CO-20151116`
- `sub-C_ses-CO-20151117`
- `sub-C_ses-CO-20151119`
- `sub-C_ses-CO-20151120`
- `sub-C_ses-CO-20151201`

Train + validation development sessions only. No formal-test receipt creation or modification.

---

## GPU cost

| Item | Count |
|---|---:|
| Fresh training cells | 6 |
| Estimated wall time | ~6–9 GPU-hours (1h/cell order-of-magnitude on RTX 3090 class) |

---

## Implementation bindings

| Artifact | Exact path / status |
|---|---|
| Frozen v2 configuration | `configs/a2_matched_subject_shift_v2.json` — active, source-hashed by the official preflight |
| Contract core | `mc_maze/a2_matched_subject_shift_v2_core.py` — active; owns topology, source-only normalizer, immutable-artifact, implementation-SHA, and isolated-runtime validators |
| One official CPU preflight | `scripts/a2_matched_subject_shift_v2_preflight.py` → `results/a2_matched_subject_shift_v2/official_cpu_preflight.json` and `.sha256` — active; `O_EXCL`, `fsync`, mode `0444`, and non-authorizing `CPU_PREFLIGHT_PASSED_AWAITING_ROOT_GO` status required. **Official receipt SHA-256: unminted until a fresh empty-matrix CPU audit is run; record the emitted SHA here/on the run log before any GO.** |
| Per-cell runner | `scripts/run_a2_matched_subject_shift_v2_one_cell.sh` — active; verifies the one official preflight and all implementation bindings, then checks only its candidate `(arm, seed)` freshness so cell 2–6 remain launchable after earlier cells finish |
| Domain scorer | `scripts/a2_matched_subject_shift_v2_score.py` — active; source checkpoint bundle is reused unchanged on both domains; source-cache normalizers must be bitwise equal to an uncached strict-27 recomputation before target scoring |
| Aggregator | `scripts/aggregate_a2_matched_subject_shift_v2.py` — active; requires the official preflight and verifies every domain body/sidecar integrity pair before aggregation |
| Bound implementation surface | config, this contract, core, preflight, scorer, runner, aggregator, trainer/evaluators (including the frozen-model loader), both SUA datamodules, side-feature and gradient-free protocol code, plus the directly imported streaming-calibration model/component/run-artifact modules — all explicit path/SHA bindings are sealed by the official preflight and rechecked at launch/score/aggregate |
| Freshness surface | candidate checkpoint directory, both domain receipts, per-cell launch receipt, and the trainer's global `p3_<out_name>_seed<seed>.json` summary must all be absent before a cell starts; previous completed cells do not block later cells |
| Python/Torch isolation | every v2 command forces and receipt-binds `PYTHONNOUSERSITE=1`; launch additionally verifies active-env Torch `2.5.1.post303` / CUDA `11.8`, in-env import path, and RTX 3090-class visible device before trainer code |
| Mandatory invariant | each trained checkpoint is scored unchanged on both domains; every domain receipt and aggregate is immutable (`O_EXCL` + `fsync` + `0444`) with a verified SHA-256 sidecar |

**This document authorizes nothing.** The top status remains non-authorizing. GPU execution requires both an explicit root GO flag and the runner authorization flag, after a newly minted immutable official CPU preflight has passed.
