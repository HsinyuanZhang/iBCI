# Carrier Reliability vs Outcome — V2 Recomputation Protocol

**Frozen:** 2026-08-14, before any correlation was computed
**Scope:** the 13 public held-in-calibration H1 recordings, CPU only
**Reason:** a prior split-half reliability statistic (Spearman `rho = 0.808`, `p = 0.0008`,
`n = 13`) was computed with the **V1** estimator (`CARRIER_DIM=4`, `RIDGE_LAMBDA=0.1`, q=3
slopes) while the arm it claims to explain, **H-SE5**, is **V2** (`CARRIER_DIM=5`,
`RIDGE_LAMBDA=3.0`, q=4 slopes). The prior provenance check bound `source_audit_v2r2.json`,
which contains V2 carriers, and therefore did not cover the V1 statistic at all.

## Budget

H-SE5's development decoder result (`+0.0285` on date 1, `-0.0226` on `19250108`) is the
**four-trial** check, and `H1_SE5_M4_FOLD0_TERMINAL_v1.json` binds
`event_estimator_sha256 = 05ab4735...f819e`, the V2 module. All analysis is therefore at **M=4**.

M=3 is not analysable: the parity split puts one trial in the odd half (4-6 events in every
session), below the eight-event floor that both estimators enforce. This is a property of the
sealed code, not a choice, and it is reported rather than worked around.

## Frozen definitions

- **Reliability statistic** `S_v2`: partition the M=4 support block by trial parity, even
  `trial_index` against odd. Project both halves through the V2 date-LODO basis
  (`v2.fit_source_all_event_basis`), fit each half with the sealed `v2.fit_carrier_arrays`, take
  the per-channel cosine between the two `q=4` slope vectors, and report the median over channels
  with a defined (nonzero-norm) cosine. Intercepts are excluded, matching the V1 definition.
- **Reproduction control** `S_v1`: identical construction using `v1.fit_endpoint_basis` and
  `v1.fit_carrier_from_arrays` (q=3), i.e. the sealed `v1.coefficient_split_stability`.
- **Outcome** `Y`: `median_delta_intercept` at M=4 from `v2.forward_transfer`, recomputed and
  required to equal the sealed `source_audit_v2r2.json` value.
- **Competing explanation** `C`: M=4 support-event count.
- **Test**: two-sided Spearman `rho`, `n = 13`. The reported `p` is an exact-as-feasible Monte
  Carlo permutation p-value over 200,000 seeded permutations, because at `n = 13` the
  t-approximation is not trustworthy. The asymptotic p is reported alongside it.

## Gate order

Provenance is checked **first**. The 6 V2 `basis_sha256` and 13 V2 `carrier_sha256` values must
match `source_audit_v2r2.json` for all 13 sessions. If any hash differs the run stops and reports;
no correlation is computed.

## Predeclared decision rule

`S_v2` is **supported** as a reliability gate only if all four hold:

| | Condition |
|---|---|
| P1 | `rho(S_v2, Y) >= 0.60` |
| P2 | two-sided permutation `p < 0.05` |
| P3 | minimum leave-one-session-out `rho >= 0.40` |
| P4 | `rho(S_v2, Y) - abs(rho(C, Y)) >= 0.30` |

Any failure means **not supported**: the self-assessment proposal does not have evidence at the
deployed estimator, and the paper's conclusion sentence is not replaced. A negative result is a
successful outcome of this protocol.

## Predeclared validity control

The reproduction arm `S_v1` must land within `+/-0.05` of the reported `0.808`. If it does not,
the discrepancy is in this reimplementation rather than in the estimator version, and the run is
reported as **inconclusive**, not as a negative result.

## Known limitations, declared in advance

1. Each half is roughly 10 events against 5 free parameters. This understates the deployed
   carrier's reliability, which is fit on ~20 events. A Spearman-Brown step-up is reported as an
   approximate correction; it assumes parallel halves and a correlation-like metric, and median
   cosine is neither exactly, so it is an indication of direction and rough size only. A direct
   empirical n-scaling curve is **not** available: the sealed fitter refuses fewer than 8 events
   and `SUPPORT_BUDGETS` stops at 4 trials, so neither 5-event quarters nor a 40-event support
   block can be fit without modifying sealed code, which this protocol forbids.
2. `S_v2` and `Y` overlap within each session: both are computed from the same recording, so part
   of any correlation is "session SNR predicts session SNR". Acceptable for a ranking gate,
   **not** a causal claim, and no causal language is permitted in the write-up.
3. Trial-parity is one arbitrary partition. A seeded 50-draw random event-level split is reported
   as a robustness check. Event-level splits break trial grouping and are therefore *less*
   conservative than parity; they bound split-choice sensitivity, they do not replace the primary.
