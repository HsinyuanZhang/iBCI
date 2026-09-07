# Carrier transfer and correspondence-breakage protocol

**Date:** 2026-08-12  
**Status:** pre-registered scaffolding only. Authorizes no GPU run and no training.  
**Scope:** items **A5** (transferred and aged carrier) and **A3** (correspondence-breakage dose response) from `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md`, motivated by `PAPER_FRAMING_ANALYSIS_20260812.md` sections 3.1–3.2.

---

## 0. Authorization boundary

| Allowed | Forbidden |
|---|---|
| Forward-only scoring on **development validation** sessions with a **sealed** B3S/T4 checkpoint | Any GPU use (`CUDA_VISIBLE_DEVICES` must be empty; `torch.device("cpu")` only) |
| Writing JSON receipts from `carrier_transfer_and_breakage_screen.py` | Training, fine-tuning, background launchers, watchers, auto-launchers |
| CPU `pytest` on synthetic fixtures (<60 s) | Opening sealed formal-test sessions (see §5) |
| Aggregating complete receipt sets with `aggregate_carrier_transfer_and_breakage.py` | Modifying existing results, receipts, checkpoints, or logs |

---

## 1. Hypotheses (pre-registered)

### A5 — transferred and aged carrier

**Hypothesis.** A correctly fitted carrier from session A, transferred to session B and scored with B's neural stream on a frozen checkpoint, performs **worse** than B's own refitted carrier when the **same zero-fill pattern** is applied to both arms. As the calendar gap between donor and recipient increases, the primary statistic should decay.

**Zero-fill confound (pre-registered, 2026-08-12 revision).** When `N_donor < N_recipient`, the transfer arm leaves `N_recipient − min(N_d, N_r)` units on standardized-zero carrier. That alone can cost on the order of `T4 − Z4` (≈ `−0.249` on the M30 substrate). A low `transferred − own_full` score is therefore **uninterpretable** without separating truncation cost from wrong-session content.

**Three-way decomposition (required on every pair).**

```text
transferred − own_full        = total effect (confounded; secondary statistic)
transferred − own_truncated   = pure wrong-session carrier effect (PRIMARY)
own_truncated − own_full      = pure truncation/zero-fill nuisance (measured)
```

**Reference arms (required on every recipient session).**

| Arm | Carrier source | Role |
|---|---|---|
| `own_carrier` | Recipient T4, full `N_r` rows | Upper reference (no truncation) |
| `own_truncated` | Recipient T4 with **identical** zero-fill pattern as transfer | Matched truncation control |
| `transferred_carrier` | Donor T4 after shared zero-fill pattern | Test arm |
| `zero_carrier` | Standardized z4 mask | Lower reference |

The zero-fill pattern is derived **once per pair** via `build_transfer_arm_carriers()` and applied identically to `transferred_carrier` and `own_truncated`. Code asserts byte-identical zero-fill tails.

**Primary statistic (frozen gate).** `transferred_minus_own_truncated` per pair.  
**Secondary statistic.** `transferred_minus_own_full` (reported, not gated alone).  
**Nuisance statistic.** `own_truncated_minus_own_full` (truncation cost).

**Falsification.** If `transferred_minus_own_truncated` is not materially below zero on secondary-subset pairs (and does not decay with calendar gap on aged sweeps), the method behaves partly as a fixed prior. Delete the paper's drift sentence. Report honestly.

**Pair subsets (must not be pooled).**

| Subset | Criterion | Count | Interpretation |
|---|---|---:|---|
| **Primary** | `N_donor ≥ N_recipient` | **15** / 30 | No zero-fill; `own_truncated == own_carrier`; primary ≈ secondary |
| **Secondary** | `N_donor < N_recipient` | **15** / 30 | Zero-fill present; **requires** `own_truncated` control |

Verified against `SESSION_UNIT_COUNTS` in `carrier_transfer_and_breakage.py` (manifest audit 2026-08-12).

### A3 — correspondence-breakage dose response

**Hypothesis.** Progressively breaking unit correspondence degrades the **activity-only** arm faster than the **carrier** arm. The carrier repairs broken correspondence.

**Primary statistic.** The **interaction**: change in `(carrier_score − control_score)` from the least-broken to the most-broken rung of the pre-registered ladder (not the level at any single point).

**Control arm.** `z4` (activity-only identity encoder with standardized zero carrier), with **identical** unit/electrode breakage applied to query neural stream, calibration trials, and identity path.

**Falsification.** If the two arms degrade in parallel (interaction ≈ 0), the carrier is not a correspondence-repair mechanism and A2/A4 correspondence framing must be reinterpreted.

---

## 2. Sessions and admissible pairs

### 2.1 Development validation sessions (only these may be opened)

From `configs/subc_co_27_6_strict_train_val_manifest.json`:

| Session | NWB units-table rows |
|---|---:|
| `sub-C_ses-CO-20151103` | 38 |
| `sub-C_ses-CO-20151104` | 59 |
| `sub-C_ses-CO-20151106` | 60 |
| `sub-C_ses-CO-20151109` | 65 |
| `sub-C_ses-CO-20151110` | 61 |
| `sub-C_ses-CO-20151112` | 42 |

**Critical fact (pre-registered).** No validation pair has equal `N`. A naive `[N_A,4] → [N_B,4]` paste is **inadmissible**.

### 2.2 Admissible donor→recipient pairs

All **30** ordered pairs `(donor, recipient)` with `donor ≠ recipient` and both sessions in §2.1 (`ADMISSIBLE_TRANSFER_PAIRS`).

**Primary subset (`PRIMARY_TRANSFER_PAIRS`):** `N_donor ≥ N_recipient` → **15** pairs, zero-fill fraction = 0.  
**Secondary subset (`SECONDARY_TRANSFER_PAIRS`):** `N_donor < N_recipient` → **15** pairs, zero-fill fraction > 0.

The aggregator **refuses** to pool primary and secondary subsets into one headline. Use `--subset primary` or `--subset secondary`.

**Aged sweep subset:** **15** pairs with `calendar_gap_days(donor, recipient) > 0` (`AGED_TRANSFER_PAIRS`).

### 2.3 Sealed formal-test boundary (never open)

```
sub-C_ses-CO-20151113
sub-C_ses-CO-20151116
sub-C_ses-CO-20151117
sub-C_ses-CO-20151119
sub-C_ses-CO-20151120
sub-C_ses-CO-20151201
```

Code must raise if any of these names are requested.

---

## 3. Row-matching rule (pre-registered before any result)

**Rule ID:** `nwb_unit_index_prefix_v1`  
**Version:** 1

Given donor carrier `C_d ∈ R^{N_d × 4}` in fixed NWB units-table row order and recipient requiring `N_r` units:

1. `K = min(N_d, N_r)`.
2. `output[i] = C_d[i]` for `i = 0 … K−1`.
3. `output[i] = 0` (standardized z4 coordinate) for `i = K … N_r−1`.
4. Record `matched_rows = K`, `donor_rows_dropped = max(0, N_d − K)`, `recipient_rows_zero_filled = max(0, N_r − K)`.

**Refusal conditions.**

- Attempting to attach `C_d` with `N_d ≠ N_r` **without** this rule → raise (no silent truncation).
- `K = 0` while `N_r > 0` → raise.

**Limitations (stated honestly).**

- Row index is **not** unit identity across sessions; it is a pre-registered **positional** convention chosen because no validation pair shares `N`.
- Unmatched recipient tail units receive **zero carrier**, not donor padding or interpolation.
- Electrode-aware or Procrustes alignment is **out of scope** for this screen (future work).

**Design against reviewer criticism.** The rule is frozen in source (`ROW_MATCHING_RULE`), named in every receipt, checked by the aggregator, and bracketed by four arms on the **same** recipient session and query window. The **matched-truncation arm** (`own_truncated`) holds the zero-fill pattern fixed so the primary estimand isolates wrong-session carrier content from truncation nuisance. Primary and secondary pair subsets are reported separately; the aggregator refuses to pool them.

---

## 4. Breakage ladders and seed discipline

**Base seed:** `20260812` (`BASE_RANDOMIZATION_SEED`).  
**Derived seed:** `derived_seed(family=..., base_seed=..., **parts)` — SHA-256 salted, 32-bit little-endian (same pattern as `ac4rs4_derived_seed`).

### 4.1 Unit dropout (`unit_dropout`)

Dropout probabilities `p ∈ {0, 0.1, 0.25, 0.5}`.  
Drop the **same** unit indices from neural bins, calibration tensor, and carrier rows.  
Seed: `derived_seed(family="breakage:unit_dropout", session=..., level=p)`.

### 4.2 Electrode pooling (`electrode_pooling`)

Deterministic sorted-unit → electrode aggregation via `pool_trial_rates_by_electrode` on neural/calib streams; carrier rows averaged within electrode.  
Two rungs: full units (`level=0`) vs pooled (`level=1`).  
Re-sorting with different sorter parameters is **explicitly out of scope** (needs new caches).

### 4.3 Unit subsetting (`unit_subset`)

Subset sizes `{N_full, 32, 16, 8}` (skip sizes larger than `N_full`).  
Seed: `derived_seed(family="breakage:unit_subset", session=..., subset_size=...)`.

---

## 5. Frozen evaluation contract

| Field | Value |
|---|---:|
| Activity calibration trials | 30 |
| T4 label pool | 30 |
| Query evaluation start | trial index 50 (pool_size) |
| Checkpoint | sealed B3S/T4 development checkpoint (SHA-256 in receipt) |
| Behavior normalization | train-session stats from Step-1 manifest |
| Side-feature normalization | train-only T4 stats (no per-session refit of normalizer) |
| Backprop / weight updates | none |

Both experiments share `evaluate_session_configs` / `eval_r2` from `eval_adaptation_dandi688.py` so arms are directly comparable.

---

## 6. Receipt schema (screen output)

**Schema version:** 2

Every transfer receipt includes:

- `schema_version`, `protocol_id`, `checkpoint_sha256`
- `donor_session`, `recipient_session`, `pair_subset` (`primary` | `secondary`)
- `row_matching_rule`, `zero_fill_pattern_digest`, `matched_rows`, `donor_rows_dropped`, `recipient_rows_zero_filled`
- `N_donor`, `N_recipient`, `zero_fill_fraction`
- per-arm scores: `own_carrier`, `own_truncated`, `transferred_carrier`, `zero_carrier`
- `deltas`: `transferred_minus_own_truncated` (primary), `transferred_minus_own_full` (secondary), `own_truncated_minus_own_full` (nuisance)
- `query_window`, `sealed_test_sessions_opened: false`

---

## 7. Aggregator fail-closed checks

`aggregate_carrier_transfer_and_breakage.py` rejects:

1. Incomplete breakage ladders (missing rung).
2. Mismatched `query_window` across compared receipts.
3. Sealed-session contamination.
4. Row-matching rule drift between transfer receipts.
5. Donor/recipient pairs not on `ADMISSIBLE_TRANSFER_PAIRS`.
6. `unit_dropout` ladders not exactly `{0, 0.1, 0.25, 0.5}`.
7. **Pooling primary and secondary subsets into one headline** (mixed `pair_subset` in a single aggregate).
8. Missing `own_truncated` arm or inconsistent `zero_fill_fraction` vs row counts.

Transfer aggregates emit `primary_subset` and `secondary_subset` separately. Each per-pair row includes `zero_fill_fraction` alongside every delta (`deltas_with_confound`). Use `--subset primary` or `--subset secondary` when aggregating one subset only.

---

## 8. Implementation map

| Artifact | Path |
|---|---|
| Pure contracts | `mc_maze/carrier_transfer_and_breakage.py` |
| Screen runner | `scripts/carrier_transfer_and_breakage_screen.py` |
| Aggregator | `scripts/aggregate_carrier_transfer_and_breakage.py` |
| Unit tests | `tests/test_carrier_transfer_and_breakage.py` |

---

## 9. What this protocol does not claim

- No electrode-identity matching across sessions.
- No re-sorting / alternate sorter breakage (A3 future work).
- No formal-test numbers until a separate, explicitly authorized lock exists.
- No guarantee that index-prefix transfer is biologically meaningful — only that it is **pre-registered, fail-closed, and bracketed**.
