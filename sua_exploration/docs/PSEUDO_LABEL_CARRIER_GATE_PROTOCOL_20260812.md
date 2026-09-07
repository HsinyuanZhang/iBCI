# Pseudo-label carrier constructibility gate protocol (B8)

**Frozen:** 2026-08-12 (Asia/Hong_Kong)  
**Status:** CPU scaffolding and pre-registered gate only. Authorizes no GPU run and no
training-loop wiring of recursive least squares by itself.  
**Parent:** `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` item B8.

## 1. Hypothesis

After an initial labelled prefix of `M` calibration trials, subsequent **unlabelled**
trials could update `[a, c, b]` by recursive least squares (RLS) using the decoder's
own output direction as a pseudo-target — closed form, `O(1)` state per unit, no backward
pass. Before any GPU work, this protocol screens whether pseudo-labels are **constructible**:
if refitting the carrier from decoder-output directions cannot recover the true `[a,c]`
block on held labelled trials, the online update route is not credible.

Reference: RT sparse-endpoint Stage-1 split-half constructibility gate — median `[a,c]`
cosine `0.787119` (`HANDOFF_MAINLINE_CLOSURE_20260811.md`).

## 2. Why the N4 failure does not close this route

N4 was a label-free **static** descriptor (rate, Fano, autocorrelation, population
coupling). It scored `N4 - NS4 = +0.001588` with 3/6 sessions positive. That tests
whether unsupervised scalars carry directional information. The RLS carrier inherits
**task alignment from the labelled prefix** and then propagates it via pseudo-labels — a
different mechanism. N4's near-null result is therefore not a direct falsification of B8.

## 3. Standing objection and mitigations

**Objection (self-training is circular):** the decoder's pseudo-targets are correlated with
its own errors; updating the carrier from them can amplify bias rather than correct drift.

**Mitigations pre-registered here (not optional rhetoric):**

1. RLS touches only three scalars per unit; no network weights move.
2. Forgetting factor `λ ∈ (0, 1]` limits memory of bad pseudo-labels.
3. **Fail-closed freeze:** if `[a,c]` departs from the initial batch fit by more than a
   frozen relative threshold (`departure_threshold = 2.0` in
   `carrier_recursive_estimator.py`), that unit's carrier stops updating.
4. Constructibility gate (this document) must pass before any online deployment test.
5. Transferred/aged-carrier controls (handoff A5) supply the no-update decay baseline RLS
   must beat.

The gate does not prove online RLS works; it only proves pseudo-label directions carry
enough signal for a **static** refit to be non-chance.

## 4. CPU gate algorithm

### 4.1 Inputs (retrospective screen)

On each validation session:

1. Fit **true** carrier from `M` chronological labelled trials (production rule).
2. Obtain **pseudo** direction labels — in full deployment these come from a sealed
   checkpoint's decoder outputs on the same trials; scaffolding uses synthetic pseudo-labels
   with an explicit `correct` vs `shuffled` mode.
3. Refit carrier from pseudo directions on the same `M` trials (static refit, not RLS).
4. Per unit, `cos([a,c]_true, [a,c]_pseudo)`.
5. **Shuffle baseline:** deterministically permute pseudo labels (`shuffle_seed = 42`,
   namespace `pseudo-label-carrier-gate-v1`) and repeat refit.

### 4.2 RLS component (standalone, not wired to training)

`mc_maze/carrier_recursive_estimator.py` implements batch-consistent RLS with forgetting
and departure freeze. Unit tests require convergence to batch OLS on stationary data.
**No training loop imports or calls it** in this scaffolding phase.

### 4.3 Leakage boundary

Pseudo-labels must come from decoder outputs or an explicit synthetic stand-in documented
in the receipt. True-label fitting uses only the labelled prefix. Query-window fitting or
using future trial rates to choose pseudo-labels is forbidden.

## 5. Frozen pass/fail gates

Per session, after static pseudo-label refit at `M = 30` (default):

| Gate | Threshold |
|---|---|
| `median_cosine_ge_050` | median `cos([a,c])` ≥ `0.50` |
| `correct_minus_shuffle_ge_001` | median correct − shuffle ≥ `0.01` |
| `fraction_ge_040_ge_050` | fraction of units with cosine ≥ `0.40` ≥ `0.50` |

`all_predeclared_gates` must be true for a session to count as passing. Pooled screen
reports `sessions_passing_all_gates`.

Synthetic CI expectation: cosine ≈ `1.0` when pseudo equals true; ≈ chance when pseudo
labels are permuted.

## 6. What would falsify the route

- Median cosine does not exceed the shuffle baseline (pseudo-labels carry no directional
  signal beyond chance).
- Gates pass on synthetic correct labels but fail on validation sessions with real
  decoder outputs — pseudo-labels too noisy for carrier refit.
- RLS departure-freeze fires on most units during synthetic tracking tests — online path
  would freeze immediately under realistic error.

A failure **closes the B8 GPU branch** until a new estimand is proposed.

## 7. Sessions and sealed-test isolation

Same validation session list as B9. Sealed formal-test sessions are refused in code.
Receipts: `sealed_test_sessions_opened: false`.

## 8. Scope limits

- CPU only; no GPU; no training runs.
- No wiring of RLS into `train_variant_dandi688.py` or any optimizer loop.
- Decoder R² is out of scope for this gate.
- Real-data checkpoint paths behind pytest marker `real_data` (skipped by default).

## 9. Artifacts

| Artifact | Path |
|---|---|
| Protocol (this file) | `sua_exploration/docs/PSEUDO_LABEL_CARRIER_GATE_PROTOCOL_20260812.md` |
| Runner | `sua_exploration/scripts/pseudo_label_carrier_gate.py` |
| Gate logic | `sua_exploration/mc_maze/pseudo_label_carrier_gate.py` |
| RLS | `sua_exploration/mc_maze/carrier_recursive_estimator.py` |
| Aggregator | `sua_exploration/scripts/aggregate_pseudo_label_carrier_gate.py` |
| Tests | `sua_exploration/tests/test_pseudo_label_carrier_gate.py` |

Receipt schema: `pseudo_label_carrier_gate_v1`.
