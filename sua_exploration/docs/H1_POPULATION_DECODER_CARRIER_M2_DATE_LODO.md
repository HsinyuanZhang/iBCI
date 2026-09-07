# H1 population decoder-direction carrier: M=2 date-LODO CPU feasibility

## Decision

**STOP — CPU H1 population-decoder-carrier feasibility failed; no GPU work is authorized.** This is a new, independent directional hypothesis and does not alter the sealed AFC4/LFMC4 per-channel encoding conclusions or the existing SPINT handoff.

The immutable machine receipt is [`H1_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json`](../results/h1_population_decoder_carrier_m2_date_lodo_v1/H1_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json). It was generated with the existing CPU `spint` environment, with `CUDA_VISIBLE_DEVICES` unset, from exactly the 13 public `held-in-calib` H1 NWB files. It does not open minival/query/formal held-out data and does not construct a GPU model.

## Hypothesis and independence boundary

This audit asks a different question from the sealed routes. It fits a population decoder, `y <- X`, after a source-only neural low-rank projection. For a target recording, the pooled decoder coefficient matrix has one 7-D row per neural channel. That row is represented in a source-only 4-D coefficient row basis. Thus the proposed object is a **population decoder-direction carrier**, not T4 and not a per-channel encoding estimate `x_i <- y`.

Passing this audit would only have justified further investigation of that carrier. It would not have established stable M=2 encoding, reopened AFC4 or LFMC4, or authorized attachment to SPINT. The result below fails the stricter attachment-stability gate anyway.

## Frozen protocol

For each of the six calendar dates, all recordings from that date are excluded from all of the following:

- neural mean/scale normalization;
- population PCA and the selected feature dimension `q`;
- ridge-grid selection; and
- the 4-D coefficient-row carrier basis.

Those objects are fitted on the other five dates only. The source-only grid is `q in {2, 4, 8, 16}` and `lambda in {0.1, 1, 10, 100}`, ranked only by source-recording M=2-support-to-later-trial R2. No outer-date query labels or outer-date targets are used in selection.

For each target recording, the first two chronological **eval-valid** TrialNum values supply the only target fit data. The signed H1 authority then converts 20-ms bins to finite, trial-bounded 100-ms blocks. The target ridge decoder is fit on the pooled blocks of those two trials. Every subsequent chronological trial is query-only. The result reports R2 as `1 - sum(SSE) / sum(TSS)`, where TSS is relative to the pooled two-trial target mean; this is variance-weighted across later trials and recordings rather than an unweighted mean of session R2 values. Two recordings begin at eval-valid TrialNum 3/4 because their raw earlier TrialNum bins do not provide legal eval-valid blocks; no invalid data were substituted.

The null has 31 deterministic replicates. It independently rotates labels within each of the two support trials before fitting. All later query rates, labels, and the correct pooled fit-side baseline remain untouched. This makes the null test an association test at the target support boundary, rather than a label corruption of evaluation data.

The predeclared gate requires all six dates defined, at least four dates with correct R2 above zero, at least four with correct R2 above the date-level rotation-null q95, and—deliberately stricter—at least four with mean trial-1/2 raw coefficient-row median cosine at least 0.5.

## Held-in results

| Outer date | Source-selected `(q, lambda)` | Correct held-trial R2 | Rotation null q95 | Correct − q95 | Attachment stability | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 19250101 | (8, 100) | 0.03043 | -0.01770 | 0.04812 | 0.41150 | stability fail |
| 19250108 | (8, 100) | 0.02636 | -0.01562 | 0.04197 | 0.33962 | stability fail |
| 19250113 | (16, 100) | 0.00677 | -0.05686 | 0.06364 | 0.11820 | stability fail |
| 19250115 | (8, 100) | -0.00862 | -0.02421 | 0.01559 | 0.39473 | gain/stability fail |
| 19250119 | (8, 100) | 0.00665 | -0.01257 | 0.01922 | 0.13556 | stability fail |
| 19250120 | (8, 100) | -0.01456 | -0.01672 | 0.00216 | -0.06134 | gain/stability fail |

All six date estimates are defined. The correct decoder exceeds the rotation null q95 on 6/6 dates, and its R2 is positive on exactly 4/6 dates. However, the coefficient-row attachment statistic reaches the required 0.5 on **0/6** dates. This is decisive under the declared gate: modest decoder-direction signal is not enough when the channel-wise 4-D carrier to be attached later is not stable between the two fit trials.

## Consequence

This program is sealed as STOP at this frozen grid, timebase, source-only selection procedure, and strict coefficient-stability criterion. Do not tune `q`, lambda, the null, the timebase, or the stability threshold to rescue it; do not train a GPU model; and do not attach this carrier to original SPINT. The AFC4/LFMC4 sealed decisions and the baseline/terminal package handoff remain unchanged.
