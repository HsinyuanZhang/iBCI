# Current results: analytic functional carriers for streaming neural decoding

**Updated:** 2026-08-11 HKT
**Status:** concise result ledger; RT sparse Stage-2 remains nonterminal
**Full project interpretation:**
[`HANDOFF_MAINLINE_CLOSURE_20260811.md`](HANDOFF_MAINLINE_CLOSURE_20260811.md)

This file contains result-level facts, scope, and evidence pointers. It replaces the former chronological
PID/tmux/epoch/launch diary. Historical implementation incidents remain recoverable from Git history, immutable
receipts, and the dated protocol documents still referenced by scripts and tests.

## 1. Result summary

| dataset / endpoint | principal R² result | evidence status |
|---|---:|---|
| external subject-M SUA | T4/Zero4/TS4 `0.356828/−0.057766/−0.115319`; T4−Zero4 `+0.414594` | terminal 270-cell matrix; 3/3 seed means and 15/15 session means positive |
| external subject-M pseudo-MUA | T4/Zero4/TS4 `0.306073/−0.086878/−0.164314`; T4−Zero4 `+0.392951` | terminal controlled signal-view bridge |
| native M2 matched seed42 | SPINT/T4 `0.293110/0.382906`; delta `+0.089796` | 14/14 development cells; 7/7 session deltas positive; not the missing fresh three-seed gate |
| RT dense carrier | Full/B2 `0.441950/0.145148`; delta `+0.296802` | terminal 15-fold development matrix |
| RT sparse endpoint T4d | at 13/15 folds, T4d/Full/Zero4 `0.430038/0.423912/0.152172` | nonterminal early evidence; 45-cell verifier and B2 exact-query pass remain |
| H1 compact CarrierID | five-date H-C−H-S `+0.056287`; organizer-held `0.274939` vs paper-LR `0.261492` | dense-covariate compact-consumer evidence, not sparse-label evidence |
| M1 | Original/T4/D4 `0.648591/0.644766/0.643993`; matched carrier content `−0.00652` | carrier-content negative boundary |

## 2. Selected method and evidence boundary

The selected carrier is a fixed-width analytic session descriptor. For the canonical direction-tuning case,

```text
T4_i = [a_i, c_i, m_i, b_i]
m_i = sqrt(a_i^2 + c_i^2)
rate_i(theta) ~= b_i + a_i cos(theta) + c_i sin(theta)
```

Target-session calibration is closed form. It does not create an optimizer, execute backward, or update the
decoder. B3/B3T supplies temporal activity; T4 supplies cached functional identity. Offline source training and
query scoring still use dense behavior targets, so no-backpropagation and sparse-label claims apply only to
target-session calibration.

Evidence labels used below:

- **terminal:** complete frozen matrix plus terminal aggregate and required integrity checks;
- **organizer-held system result:** official aggregate without matched per-session attribution;
- **development:** complete local matrix on held-out development sessions;
- **nonterminal:** partial matrix, constructibility audit, or diagnostic; never a paper headline.

## 3. External subject-M SUA and pseudo-MUA

### 3.1 Terminal V9 three-arm matrix

The matrix contains 15 sessions × 2 views × 3 separately source-trained arms × 3 seeds = 270 cells. Activity
uses the first 30 rewarded trials, carrier fitting uses the first 50 trial-direction labels, and query windows are
strictly after trial 50. Target-session optimizer/backward/weight update counts are zero.

| view | T4 | Zero4 | TS4 | T4−Zero4 | T4−TS4 |
|---|---:|---:|---:|---:|---:|
| SUA | `0.356828` | `−0.057766` | `−0.115319` | **`+0.414594`** | **`+0.472147`** |
| pseudo-MUA | `0.306073` | `−0.086878` | `−0.164314` | **`+0.392951`** | **`+0.470387`** |

All four contrasts are positive for 3/3 seed means and 15/15 session means; exact sign p for each is
`6.1035e-5`. Hierarchical-bootstrap 95% intervals are:

- SUA T4−Zero4 `[0.340273,0.488060]`;
- SUA T4−TS4 `[0.395148,0.549113]`;
- pseudo-MUA T4−Zero4 `[0.314165,0.472539]`;
- pseudo-MUA T4−TS4 `[0.388574,0.553836]`.

The pseudo-MUA view is a deterministic electrode-level sum of the same sorted SUA data. It demonstrates robustness
to the signal representation, not an independent native threshold-crossing replication.

### 3.2 Label budget

The activity support remains fixed at 30 trials and the evaluation window remains after trial 50.

| T4 labels | pooled fraction | SUA R² | Δ vs M50 | pseudo-MUA R² | Δ vs M50 |
|---:|---:|---:|---:|---:|---:|
| 10 | `3.49%` | `0.304264` | `−0.052565` | `0.259185` | `−0.046888` |
| 15 | `5.24%` | `0.338115` | `−0.018713` | `0.288234` | `−0.017839` |
| 20 | `6.98%` | `0.351767` | `−0.005061` | `0.298447` | `−0.007626` |
| 30 | `10.47%` | `0.358154` | `+0.001326` | `0.305291` | `−0.000782` |
| 40 | `13.96%` | `0.351648` | `−0.005180` | `0.301075` | `−0.004997` |
| 50 | `17.45%` | `0.356828` | `0` | `0.306073` | `0` |

M15 passes the frozen point-estimate and three-seed `−0.03` adequacy gate but not a CI-based non-inferiority
claim. M30 is the stronger deployment point: the SUA/pseudo intervals `[-0.00978,+0.01257]` and
`[-0.01288,+0.00966]` lie within the `−0.03` margin. A separate true-early-start replay verifies positive decoding
immediately after trial 30 rather than only after trial 50.

### 3.3 Classical comparators and supervision density

| view | historical F0-B3 | PV50 | T4 | Ridge50 |
|---|---:|---:|---:|---:|
| SUA | `−0.196077` | `0.115374` | `0.356828` | **`0.417922`** |
| pseudo-MUA | `−0.204725` | `0.104219` | `0.306073` | **`0.410193`** |

T4 exceeds PV50 by `+0.241454/+0.201854` but is below Ridge50 by `−0.061094/−0.104120`. Do not claim that
T4 beats dense-label ridge. Across the 15 sessions, T4 uses 750 direction scalars; the evaluated Ridge50 uses
149,725 finite 2-D target rows or 299,450 scalar coordinates. The ratios are `199.633x` by rows and `399.267x`
by coordinates. They measure algorithmic target-supervision consumption, not independent samples, manual effort,
compute, latency, memory, or energy.

Evidence:

- V9 terminal aggregate under `results/dandi_000688_subm_v9_*`;
- label-budget aggregate SHA `12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2`;
- true-early-start aggregate SHA `2e858403ae01da06c15448b6c3f8685f88eee9a9dc7a9da63acda332b926b12b`;
- F0/PV/Ridge aggregate SHA `ffccd91fc128edb7ad6199671f2e32d0c6c450cdff4f2b3d734a1815b176ebc0`;
- supervision audit SHA `43582628a86e80d08d91c60ce3f283502076bac951b06499649042e61feeea03`.

## 4. Native M2

### 4.1 Matched Stage-A development result

| arm | seven-session equal-fold mean R² |
|---|---:|
| matched SPINT | `0.293110` |
| matched-decoder T4 | **`0.382906`** |
| T4−SPINT | **`+0.089796`** |

Seven session deltas are
`+0.115231/+0.059998/+0.187562/+0.029894/+0.068974/+0.016567/+0.150345`.
This is exact 14/14 seed42 development evidence. It cannot be combined with separately trained seeds43/44 to
invent the missing fresh r10 three-seed lineage.

The organizer-held T4 system exceeds the original system by `+0.116765`, but the comparator is not a matched
retraining; treat this as system-level evidence only.

### 4.2 M24 ridge diagnostic

On the strict local M24/q24 six-session endpoint:

| system | mean R² |
|---|---:|
| fixed-lambda Ridge24-W50 | `0.113916` |
| legacy F0 | `0.173901` |
| T4 | `0.226786` |
| K4 | `0.245774` |

T4−Ridge is `+0.112870`; K4−T4 is only `+0.018987`, below the frozen `+0.03` gate. Ridge24 is a per-session
closed-form fit without backpropagation. It is one fixed implementation, not a ridge-family upper bound.

## 5. RT

### 5.1 Dense carrier versus SPINT-structured reference: terminal

| arm | mean R² |
|---|---:|
| Full dense-velocity carrier | **`0.4419498241`** |
| B2-D1024 | `0.1451479751` |
| paired Full−B2 | **`+0.2968018490`** |

All 15 folds are positive; median `+0.3168286171`, exact sign p `0.00006103515625`, and paired-fold bootstrap
95% interval `[+0.2377831355,+0.3524398339]`. B2 is a same-pipeline SPINT-structured identity reference, not a
claim of a bit-exact released-code RT port.

Terminal aggregate:
`results/k4_rt_loso_v1/RT_STAGE_R_D1024_FULL15_AGGREGATE_v1.json`, SHA
`95bca578cd9ac412c88eb29b96e22c3eda5968ecb39b8f21dfc8c3fff5b536b8`.

### 5.2 Sparse endpoint carrier: nonterminal

RT has no non-degenerate native per-reach target field. The production method derives one direction per reach from
go-cue-bounded endpoint coordinates. Therefore it can support an algorithmic supervision-density claim, but not a
native annotation-cost claim.

Constructibility evidence:

- endpoint coverage min/median `0.964286/1.0`;
- 60--88 derived reach angles per session;
- rank-3 designs with condition `1.437--1.603`;
- endpoint versus dense-integrated direction cosine `0.999991--0.999997`;
- split-half `[a,c]` cosine median `0.787119`;
- correct-minus-shuffle forward-transfer mean/median `+0.035910/+0.031461`, 15/15 sessions positive.

The estimator consumes 1,103 reach-direction scalars versus 7,855 dense 2-D velocity rows / 15,710 scalar
coordinates: `7.121x/14.243x`. The interpolation implementation actually reads 5,502 coordinate scalars, so its raw
coordinate-I/O ratio is `2.855x`.

At the current 13/15 complete folds:

| arm / contrast | result |
|---|---:|
| T4d mean | `0.430038` |
| dense Full mean | `0.423912` |
| Zero4 mean | `0.152172` |
| T4d−Zero4 mean / median / drop-largest | **`+0.277865/+0.350521/+0.267222`** |
| T4d−Zero4 signs | **13/13 positive** |
| T4d−Full mean / median / drop-largest | `+0.006126/+0.004358/+0.000314` |
| T4d−Full signs | 7/13 positive |

These are early numbers, not a terminal claim or equivalence test. Execution is at 39/45 cells: fold12 is fully
closed, fold13 T4d is running, and every incomplete fold13 score remains excluded from the paired aggregate. A clearly
marked 13-fold nonterminal placeholder may be mirrored into the paper, but an independent verifier must pass before terminal
promotion or a scientific claim. Then all 15 sealed B2 checkpoints will be evaluated forward-only on the
exact Stage-2 query windows without retraining or reselection.

Key receipt SHAs:

- Stage0B constructibility `b88c91ab5cfb30b4a9ef978622e00488193c4ad18b09498c84ba76e10b9943b1`;
- supervision closure `b826aa49e1d045da22068f5a9c12e2672d700d073549134105029051327b5f16`;
- Stage1 `9b69eaaa2339610116a5db8efa23ba20ad61459373293d98e80ceec294c3d0e9`;
- Stage2 readiness `183ae60c314d1213d0c5429545d54f5c34d71a8a170df52c381480de157ba87d`;
- matrix manifest `93a1aa3549b844c399ab1cc2b9bddb1d93ee2070b51c91e80d60875cee4b3ca4`.

## 6. H1 compact consumer

H1 uses dense behavior covariates and is scientifically separate from sparse-label T4.

Five-date development decomposition:

| contrast | equal-date mean | positive dates | date-bootstrap 95% interval |
|---|---:|---:|---:|
| H-C−H-S | **`+0.056287`** | 4/5 | `[+0.005965,+0.097195]` |
| H-C−H-C0 | `+0.032557` | 4/5 | `[−0.004809,+0.079771]` |
| H-C0−H-S | `+0.023730` | 3/5 | `[−0.004455,+0.052217]` |
| H-C−H-LS | `+0.030367` | 4/5 | `[−0.000686,+0.069720]` |

The decomposition exactly satisfies `0.032557+0.023730=0.056287`: carrier content and compact consumer both have
positive mean contributions, but the content-only interval touches zero.

Organizer-held result:

| system | held-out mean ± std | held-in mean ± std | normalized latency |
|---|---:|---:|---:|
| all-source CarrierID `578689` | **`0.274939±0.127206`** | **`0.473125±0.039341`** | **`0.113919`** |
| paper-LR SPINT `578474` | `0.261492±0.148717` | `0.470423±0.048025` | `0.129075` |
| released-LR SPINT `578473` | `0.209917±0.114232` | `0.439027±0.056906` | `0.129293` |

The official endpoint has no per-session scores, so do not claim paired significance or session-uniform superiority.
The compact identity path is about 58k parameters versus 5.97M for H-S, approximately `102.6x` fewer.

The CI64 five-date matrix is secondary and nonterminal. At the current update, 19/25 source checkpoints are
complete; target evaluation has not opened. It cannot delay the SUA/M2/RT mainline or authorize H64 escalation.

Official terminal receipt:
`SPINT-main/pilot_artifacts/h1_carrierid_all_source_official_v1/H1_CARRIERID_ALL_SOURCE_EVALAI_TERMINAL_RESULT_RECEIPT_v5.json`,
SHA `b7f76450494f3c0d0ad169de9e89f662c17f435ef2c08bb3d7062908515b488e`.

## 7. M1 and other tested boundaries

| route | result | disposition |
|---|---|---|
| M1 official Original/T4/D4 | `0.648591/0.644766/0.643993` | T4/D4 do not improve Original |
| M1 matched EMG-AFC4 content | `B-C−B-C0=−0.00652` | carrier-content negative; stop |
| M1 matched compact consumer | deltas `+0.095977/−0.033281/+0.047269`; mean `+0.036655` | only 2/3 pass `−0.03`; strict 3/3 non-inferiority closes |
| RT L-D live gain | G-Full−A0 `−0.002068`; G-Full−G-XLS `+0.006606` | fails both `+0.03` gates; stop |
| N4 neural-only carrier on M2 | N4−NS4 `+0.001588`, 3/6 positive | static label-free statistics do not reproduce T4 |
| T4GATE static reliability | T4GATE−T4 `−0.0108±0.0049 SE` | ineffective |
| waveform/SNR/electrode relations | no reliable held-out improvement | do not continue |
| H1 H-PCF8/native-phase precursors | source constructibility gates fail | no GPU |

These close the tested implementations, not every possible self-supervised representation or adaptation method.

## 8. Additional completed attribution

The SUA component lattice reports T4/AC4/PH4/MB4/Z4/B4/TS4/LS4
`0.574976/0.562753/0.516961/0.356620/0.326008/0.287273/0.284528/0.265626`.
AC4−T4 is `−0.012222`, inside the `−0.03` tolerance, while T4−Z4 is `+0.248968`. This supports the signed
`[a,c]` functional component; rate-scale alone does not recover the full benefit.

Paired SUA/pseudo co-training is non-inferior to separate T4 training in the tested matrix:

- shared-T4−separate-T4 `+0.008351/+0.012543` for SUA/pseudo-MUA;
- shared-T4−shared-TS4 `+0.285474/+0.369468`.

These are supporting architecture results, not the main sparse-supervision comparison.

## 9. Paper-safe interpretation

Supported after RT terminal closure:

1. functional carrier content can adapt a frozen streaming decoder without target-session backpropagation;
2. correct content and row attachment matter;
3. useful performance transfers across SUA, pseudo-MUA, native M2, and RT, with H1 as a separate compact-consumer
   module;
4. the evaluated analytic carrier can use substantially fewer target-supervision values than the evaluated dense
   ridge/carrier implementations;
5. the selected carrier-aware identity path is compatible with large parameter compression.

Not supported:

- universal superiority over ridge or SPINT;
- a universal native annotation-cost, compute, latency, energy, or memory advantage;
- a fully sparse-label source-training/scoring pipeline;
- RT sparse superiority or equivalence before terminal paired analysis;
- H1 sparse-label evidence;
- positive M1 carrier content;
- label-free calibration.

## 10. Remaining result-bearing work

1. RT folds12--14 and 45-cell terminal verifier;
2. 15-fold exact-query B2 forward-only companion;
3. terminal updates to this ledger, the authoritative handoff, and Overleaf;
4. reference-aware document/checkpoint cleanup and narrow GitHub synchronization.

All other proposed carrier, decoder-fusion, M1/H1 rescue, and quantization experiments are outside this closure.
