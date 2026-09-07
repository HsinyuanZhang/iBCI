# Carrier Reliability vs Outcome — V2 Recomputation

**Date:** 2026-08-14
**Protocol:** `sua_exploration/docs/CARRIER_RELIABILITY_V2_PROTOCOL_20260814.md`
(`sha256 1eec7e1db65f4b30da09116e7d2fe69915c57e6e7f7c092c48340a0d73d22df2`, frozen before any
correlation was computed)
**Receipt:** `sua_exploration/results/carrier_reliability_v2_20260814/receipt.json`
(`sha256 35413b8a4e2982259ee4a34b37515637c7b0be1baa2dff85d0a69dc1979e460e`, immutable)
**Runner:** `sua_exploration/scripts/run_carrier_reliability_v2_recompute.py`
**Tests:** `sua_exploration/tests/test_carrier_reliability_v2_recompute.py` (28 passed)
**Compute:** CPU only, single-threaded, `nice -n 15`. No CUDA. Total runtime under 30 s.

## Headline

The correlation **survives**, and it survives because it was never a V1 number in the first place.
Recomputed with the sealed V2 estimator, split-half carrier stability predicts
`median_delta_intercept` at Spearman **rho = 0.8077, permutation p = 0.0013, n = 13**, against the
disputed V1-attributed **0.808 / 0.0008**. Support-event count reproduces its null exactly at
**0.1747 / 0.57**. Leave-one-session-out rho stays in **[0.755, 0.874]**.

The recomputation nevertheless does **not** rescue the abstention proposal, for reasons that are
new rather than inherited. The relationship is not a property of the V2 carrier: every estimator
configuration tried lands in 0.75-0.81. The absolute reliability level is near zero (median
split-half cosine 0.060, 4 of 13 sessions negative). And the outcome being predicted is positive
in all 13 sessions, so it is not the quantity that flipped sign and prompted the proposal.

## 1. Provenance — 13/13

Recomputed V2 carriers were hash-checked against the sealed `source_audit_v2r2.json` before any
correlation was computed, as the protocol required. All 13 sessions matched on every field:
`input_sha256`, `input_path`, `basis_sha256`, `carrier_sha256`, `design_rank`, `support_events`,
and `median_delta_intercept` (bit-exact float equality, not a tolerance). The sealed audit was
itself verified against its own `.sha256` sidecar first.

File discovery went through `h1_sparse_event_endpoint.index_heldin_calib` only. A test asserts the
runner source contains no `.nwb` literal and no `glob(`, and that `reject_path_scope` fails closed
on both `held-out` and `minival`.

## 2. Budget: M=4, and M=3 is structurally unavailable

H-SE5's development decoder result is the four-trial check, and
`H1_SE5_M4_FOLD0_TERMINAL_v1.json` binds `event_estimator_sha256 = 05ab4735...f819e`, the V2
module. So M=4 is the arm's budget and the analysis follows it.

M=3 cannot be analysed at all. The parity split puts a single trial in the odd half, which leaves
4-6 events in **every one of the 13 sessions**, below the eight-event floor that both estimators
enforce. This is a property of the sealed code, not a choice. It matters because the V2 entrance
gate selected M=3 as `selected_gpu_budget`: at the budget the gate prefers, the proposed
reliability statistic cannot be formed.

## 3. Per-session values

`S_v2` is the recomputed V2 split-half median slope cosine; `S_v1` is the V1 construction;
`S_v2 rand` is the mean over 50 seeded random event-level splits.

| session | date | support | `median_delta_intercept` | `S_v2` | `S_v1` | `S_v2` rand |
|---|---|---|---|---|---|---|
| ses-19250101T111740 | 19250101 | 20 | 0.01798 | 0.0656 | 0.1964 | 0.0967 |
| ses-19250101T112404 | 19250101 | 21 | 0.01263 | 0.0598 | 0.0966 | 0.0534 |
| ses-19250108T110520 | 19250108 | 19 | 0.00347 | −0.0358 | −0.1104 | 0.0529 |
| ses-19250108T111022 | 19250108 | 23 | 0.00675 | −0.0175 | 0.1767 | 0.1025 |
| ses-19250108T111455 | 19250108 | 22 | 0.02360 | 0.2000 | 0.3823 | 0.1994 |
| ses-19250113T120811 | 19250113 | 19 | 0.01151 | 0.0008 | 0.1534 | 0.1012 |
| ses-19250113T121303 | 19250113 | 19 | 0.01099 | 0.1233 | 0.0908 | 0.1076 |
| ses-19250115T110633 | 19250115 | 20 | 0.01695 | 0.2051 | 0.3184 | 0.1306 |
| ses-19250115T111328 | 19250115 | 20 | 0.01238 | 0.2293 | 0.3235 | 0.1628 |
| ses-19250119T113543 | 19250119 | 21 | 0.01019 | 0.0350 | 0.0278 | 0.0766 |
| ses-19250119T114045 | 19250119 | 20 | 0.01239 | 0.1359 | 0.2277 | 0.1415 |
| ses-19250120T115044 | 19250120 | 21 | 0.00864 | −0.0084 | −0.0227 | 0.0461 |
| ses-19250120T115537 | 19250120 | 17 | 0.01016 | −0.0030 | −0.1390 | 0.0975 |

## 4. Correlations

`p_perm` is a seeded 200,000-draw two-sided permutation p; at n = 13 the t-approximation
(`p_asym`) is not trustworthy, so the permutation value is primary. Both are reported.

| relation | rho | p_perm | p_asym | n | prior reported |
|---|---|---|---|---|---|
| **`S_v2` vs `median_delta_intercept`** | **0.8077** | **0.0013** | 0.00084 | 13 | 0.808 / 0.0008 |
| `S_v1` vs `median_delta_intercept` | 0.7527 | 0.0040 | 0.0030 | 13 | — |
| **support count vs `median_delta_intercept`** | **0.1747** | **0.5654** | 0.5681 | 13 | 0.175 / 0.57 |
| `S_v2` random splits vs outcome | 0.5220 | 0.0697 | 0.0673 | 13 | — |
| `S_v2` vs `S_v1` | 0.7857 | 0.0023 | 0.0015 | 13 | — |
| `S_v2` vs support count | 0.0535 | 0.8652 | 0.8621 | 13 | — |

Leave-one-session-out, n = 12 each:

| relation | min rho | (dropping) | max rho |
|---|---|---|---|
| `S_v2` vs outcome | **0.7552** | ses-19250108T110520 | 0.8741 |
| `S_v1` vs outcome | 0.6853 | ses-19250108T111455 | 0.8531 |
| support count vs outcome | 0.0036 | ses-19250108T110520 | 0.4334 |

The relationship is **not** driven by one or two sessions: the worst single removal still leaves
rho = 0.755. All four predeclared criteria (P1 rho >= 0.60, P2 p < 0.05, P3 leave-one-out min
>= 0.40, P4 margin over support count >= 0.30) pass.

## 5. The estimator version was never the load-bearing variable

Crossing the two things that differ between V1 and V2 — latent rank and basis scope — shows the
statistic is close to indifferent to both:

| configuration | rho vs M4 outcome | rho vs M3 outcome | median cosine | negative |
|---|---|---|---|---|
| V2: q=4, ridge 3.0, all-event basis | 0.8077 | 0.4670 | 0.0598 | 4/13 |
| V1: q=3, ridge 0.1, V1 M4-support basis | 0.7527 | 0.6044 | 0.1534 | 3/13 |
| V1: q=3, ridge 0.1, V1 M3-support basis | 0.8022 | 0.5220 | 0.1403 | 3/13 |
| V1: q=3, ridge 0.1, q=3 all-event basis | 0.8077 | 0.5769 | 0.1387 | 3/13 |

Every configuration lands in 0.75-0.81. Spearman is a rank statistic on 13 points, and it is
simply not sensitive to whether the carrier has 4 or 5 free parameters per channel. The last row
reaches the same 0.8077 as V2 while ranking the sessions differently — both happen to give a
sum of squared rank differences of 70. So the *number* does not identify the estimator that
produced it, and correcting the estimator version does not add information.

### Attribution of the disputed 0.808

The reviewer's objection was raised from citation inspection, not recomputation: the prior report
described "a 5-parameter V2 split-half" while naming `coefficient_split_stability`, which lives in
V1 and calls V1's 4-wide fitter. Those two descriptions are indeed incompatible.

The recomputation resolves which one was real. A genuine V2 split-half gives 0.8077 with
asymptotic p 0.00084, matching the reported 0.808 / 0.0008 to every reported digit. The actual V1
code path does not: with V1's own M4 basis it gives 0.7527 / 0.0030, and with its M3 basis
0.8022 / 0.00097. The only other configuration reaching 0.8077 is a V1-rank/V2-basis-scope hybrid
that exists nowhere in the tree.

The prior statistic was therefore, on the evidence, **computed with V2 and cited with the wrong
function name**. The defect was in the citation, not the arithmetic. This does not make the
provenance objection wrong to raise — an uncheckable citation is a real defect, and the prior work
published no receipt, so nothing in committed state could have settled it. It does mean the
recomputation changes no number.

## 6. Why this still does not support the abstention proposal

Four findings, three of them new, block the "compute reliability, abstain when low" gate.

**The absolute reliability level is approximately zero.** Median split-half cosine is **0.0598** on
a scale where 1.0 is a perfectly reproduced weight vector, and **4 of 13 sessions are negative**,
meaning the two halves of the calibration block disagree about the sign of the encoding direction.
Spearman-Brown step-up to the deployed ~20-event carrier gives **0.113**, still negligible. A gate
must compare a statistic to an absolute threshold in order to abstain. A statistic whose typical
value is 0.06 and whose best session is 0.23 cannot supply a principled threshold. It can rank
sessions; it cannot certify any of them.

**The predicted outcome never goes negative.** All 13 `median_delta_intercept` values are positive,
from 0.0035 to 0.0236. The V2 carrier beats an intercept-only predictor in every session. But the
proposal exists to handle the *decoder* result that went `+0.0285` on one date and `−0.0226` on
`19250108`. The reliability statistic has never been related to that quantity. Predicting a proxy
that is favourable everywhere cannot justify abstaining anywhere — there is no session in this
evidence where the carrier hurt. Bridging that gap needs the decoder outcome, which exists for
2 dates, far too few for a correlation, and would need GPU work not authorised here.

**The result depends on the arbitrary partition.** Replacing trial parity with 50 seeded random
event-level splits drops rho to **0.522, p = 0.070** — not significant at the level the proposal
would need. Event-level splits mix trials and are the *less* conservative choice, so this is not a
harsher test; it shows the 0.81 is partly a property of splitting by trial rather than a stable
property of the data.

**It does not transfer across budget.** Against the M=3 outcome, the same V2 statistic gives
rho = 0.467, p = 0.108. Since M=3 is the budget the V2 entrance gate selected and the one at which
the statistic cannot even be formed, there is no budget at which the gate is both computable and
demonstrated.

## 7. Carried-forward caveats

Both caveats the prior work raised against itself still apply.

**Half-length understatement — partially quantified.** Each half is ~10 events against 5 free
parameters, which understates the deployed ~20-event carrier's reliability. Spearman-Brown steps
the median 0.0598 up to 0.113. This is an approximation: it assumes parallel halves and a
correlation-like metric, and median cosine is neither exactly, so treat it as direction and rough
magnitude only. A direct empirical n-scaling curve is **not obtainable** without modifying sealed
code: `v2.fit_carrier_arrays` refuses fewer than 8 events, ruling out 5-event quarters, and
`SUPPORT_BUDGETS` stops at 4 trials, ruling out a 40-event support block. The correction moves the
reliability from negligible to negligible, so it does not change any conclusion.

**Within-session overlap — restated, not resolved.** `S_v2` and `median_delta_intercept` are
computed from the same recording, so part of the correlation is session SNR predicting session
SNR. This is admissible for a ranking gate and is **not** a causal claim. Nothing here shows that
low carrier stability *causes* poor transfer. The fact that four unrelated estimator
configurations all produce rho ≈ 0.8 is consistent with a shared session-level nuisance driving
both sides, and makes the shared-SNR reading more plausible than it was before, not less.

## 8. Verdict

The ranking relationship is real, reproduces under the correct estimator, and is robust to
leave-one-out. The reliability statistic does **not** have support as a deployable abstention gate:
its absolute scale cannot supply a threshold, it has never been related to the decoder outcome
that actually changed sign, and it is partition- and budget-dependent.

The paper's conclusion sentence should not be replaced on this basis. The two development dates
remain the honest statement of what is known.

## 9. What could not be verified

- **The prior agent's exact code path.** No receipt for the 0.808 measurement exists in committed
  state; it survives only in a chat transcript. The attribution in section 5 is inference from
  reproduced digits, strong but not a provenance check.
- **The link to the decoder outcome.** `median_delta_intercept` is a CPU forward-transfer proxy.
  The `+0.0285 / −0.0226` decoder deltas are n = 2 dates and were not recomputed; no GPU work was
  run.
- **The size of the half-length understatement.** Bounded only by the Spearman-Brown
  approximation; the direct measurement is blocked by sealed-code limits.
- **The contribution of within-session overlap.** Not decomposable — both quantities come from the
  same recording by construction. No estimate of how much of the 0.81 it accounts for.
- **Whether the gate would work at M=3.** Structurally uncomputable; not tested.
