# A14 — B3TStream + T4 efficiency branch completion contract

**Date:** 2026-08-12  
**Status:** pre-registered contract. **Authorizes no GPU run.** A separate explicit authorization decision is required.  
**Screen ID:** `sua_b3t_t4_efficiency_v1` (existing partial matrix)  
**Contract document SHA-256:** bind at launch via `contract_sha256` in the inert runner.

---

## Claim under test (must be able to fail)

The existing partial matrix shows B3T+T4 non-inferior to fresh T4 at seed 42 (`+0.011773` mean delta, inside `−0.03` margin) with measured efficiency gains. The branch remains **incomplete** without the same-seed `B3TStream + TS4` content control. This contract pins the missing cell and forbids silent configuration drift from the two existing receipts.

---

## Existing evidence (read-only audit 2026-08-12)

| Artifact | `variant_score` | Notes |
|---|---:|---|
| `results/sua_b3t_t4_efficiency_v1/t4_s42.json` | `0.585301` | fresh B3S/T4 |
| `results/sua_b3t_t4_efficiency_v1/b3t_t4_s42.json` | `0.597073` | fresh B3TS/T4 |
| Delta B3T+T4 − T4 | `+0.011773` | inside `−0.03` NI margin |
| Parameter reduction | `30.79%` | from existing cost profiles |
| Session-MAC reduction | `65.29%` | from existing cost profiles |
| Transient-state reduction | `88%` | peak live state bytes |

**Missing cell:** `b3t_ts4_s42.json` — `B3TS` variant, `ts4` side features, seed `42`.

No aggregate exists for the full three-arm matrix at any seed count.

---

## Frozen matrix

| Arm | Variant | Side | Seed | Status |
|---|---|---|---|---|
| `t4` | B3S | t4 | 42 | **complete** (immutable) |
| `b3t_t4` | B3TS | t4 | 42 | **complete** (immutable) |
| `b3t_ts4` | B3TS | ts4 | 42 | **missing — this contract** |

**New GPU cells required:** `1` (only `b3t_ts4_s42`).

Aligned-first gate: `b3t_ts4` may launch only after read-only `aggregate_sua_b3t_t4_efficiency.py --aligned-only --seeds 42` returns `control_permitted=true` (already satisfied by existing receipts).

### Matched configuration (must byte-match existing cells except arm fields)

Every field below is extracted from `t4_s42.json` / `b3t_t4_s42.json` metadata and enforced by the aggregator:

| Parameter | Frozen value |
|---|---|
| `M_activity` | `30` |
| `M_T4` / side pool | `30` |
| Evaluation start trial | `30` |
| Epochs | `12`, burn-in `4`, scored window `5–12` |
| Loss mode | `task_only` |
| Identity mode | `calibrated` |
| Split | `27,6,6` via strict manifest |
| Signal | SUA CO |
| `max_units_exclusive` | `100` |
| Teacher SHA | `9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d` |
| Manifest SHA | `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9` |
| Validation sessions | six sub-C val sessions (see A2 contract) |
| `b3t_ts4` permutation seed | `42` (unit shuffle for TS4) |
| Fresh training | required (no warm-start) |
| Formal test | never opened |

**Prohibited:** any change to `M=50`, trial-50 query, alternate teacher, or warm-start from existing checkpoints.

---

## Frozen acceptance rules

Uses `aggregate_sua_b3t_t4_efficiency.py` logic on the completed seed-42 three-arm matrix.

### Content control (primary)

Contrast `b3t_t4 − b3t_ts4` must pass **strict superiority** gates:

1. mean paired delta `≥ +0.03`;
2. all seed means positive (one seed → one seed mean);
3. all six session means positive;
4. hierarchical bootstrap 95% CI lower `> 0`;
5. exact Wilcoxon `p ≤ 0.05`.

At one seed, gates 2–5 collapse to single-seed/session checks; the aggregator reports `passes_strict_superiority` only when all six session deltas are positive and mean delta `≥ 0.03`.

### Efficiency non-inferiority (secondary — already satisfied)

`b3t_t4 − t4` mean delta `≥ −0.03` with parameter/MAC reductions ≥ `25%` and no support-state increase.

### Branch closure rule

| Outcome | Verdict |
|---|---|
| Content gate passes | Branch **claimable** for efficiency + T4 content |
| Content gate fails | Branch **closes cleanly** — efficiency result cannot claim T4-specific content |
| NI fails | Branch closed regardless of content |

---

## Outcome authorization map

| Outcome | Authorizes | Forecloses |
|---|---|---|
| Content + NI pass | Publish B3TStream efficiency branch; INT8/efficiency follow-on review | — |
| NI pass, content fail | Efficiency-only deployment claim (no T4 mechanism) | T4-content efficiency narrative |
| Content pass, NI fail | Mechanism only; no efficiency claim | Deployment efficiency table |
| Both fail | Close branch | Further B3T efficiency work |

---

## Data isolation boundary

**Sealed formal-test sessions:** same six sub-C sessions listed in A2 contract — never opened.

---

## GPU cost

| Item | Count |
|---|---:|
| Missing cells | 1 |
| Aggregator run | CPU only |
| Estimated wall time | ~1–2 GPU-hours |

---

## Implementation bindings

| Artifact | Path |
|---|---|
| Preflight | `scripts/a14_b3t_t4_efficiency_completion_preflight.py` |
| Inert runner | `scripts/run_a14_b3t_t4_efficiency_completion_one_cell.sh` |
| Aggregator | `scripts/aggregate_a14_b3t_t4_efficiency_completion.py` |
| Existing partial runner | `scripts/run_sua_b3t_t4_one_cell.sh` (reference only) |
| Authorization env | `A14_GPU_AUTHORIZATION=I_AUTHORIZE_A14_B3T_EFFICIENCY_COMPLETION_GPU` |

**This document authorizes nothing.**
