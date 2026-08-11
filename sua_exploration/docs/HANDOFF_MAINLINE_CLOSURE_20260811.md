# HANDOFF: T4 / analytic functional-carrier mainline closure

**Status:** authoritative project handoff for current conclusions, remaining work, and paper wording
**Updated:** 2026-08-11 HKT
**Scope:** SUA, pseudo-MUA, native M2, RT, H1, and the M1 boundary
**Supersession rule:** use this file for project status and scientific decisions. Older `HANDOFF_*`, dated
reviews, launch notes, and `AGENT_BRIEF_*` files are historical evidence only; they cannot reopen experiments
or override the receipts and frozen protocols cited here.

## 1. One-paragraph conclusion

The main result is not that T4 always beats a densely supervised decoder or that its arithmetic is always
cheaper than ridge. The defensible result is that a small analytic functional carrier, fitted once from a
chronological calibration prefix without target-session backpropagation, supplies session-specific neural
identity that a frozen streaming decoder can use. On external-subject SUA and pseudo-MUA the carrier gives a
large, session-consistent gain over matched zero-content and wrong-attachment controls; on native M2 it has
positive matched development and organizer-held system evidence; and on RT a per-reach endpoint-direction
variant is currently matching a carrier fitted from dense per-bin velocity while strongly beating Zero4.
The benefit is therefore task-aligned carrier content plus a pretrained consumer, not static electrode/SNR/
waveform metadata. H1 supports a separate compact-consumer claim under dense covariates. M1 is the tested
negative boundary for the present carrier content.

## 2. What T4 is

For unit or channel `i`, the calibration block fits an analytic tuning model and emits a fixed-width side
descriptor. The canonical four-value form is

```text
T4_i = [a_i, c_i, m_i, b_i]
m_i = sqrt(a_i^2 + c_i^2)
rate_i(theta) ~= b_i + a_i cos(theta) + c_i sin(theta)
```

The target-session operation is a closed-form fit followed by forward inference. It does not construct an
optimizer, call backward, or update decoder weights. The activity path still uses chronological neural support,
and offline source training and final scoring still use dense behavior targets. Therefore “sparse labels” refers
only to the target-session carrier estimator, not to the entire source-training or evaluation pipeline.

T4 is complementary to B3/B3T streaming activity calibration:

- B3/B3T provides the temporal neural-activity representation and streaming state;
- T4 provides a session-specific functional identity carrier;
- the selected deployment path caches T4 after calibration and does not refit it on every bin.

## 3. Current evidence ledger

### 3.1 External-subject SUA and pseudo-MUA: terminal positive

The strongest controlled result is the external subject-M V9 matrix: 15 sessions, two views, three arms, and
three seeds (`270/270` cells). The pseudo-MUA view is a deterministic electrode-level aggregation of the SUA
view, so it is a controlled signal-view bridge rather than an independent threshold-crossing dataset.

| view | T4 R² | Zero4 R² | TS4 R² | T4−Zero4 | T4−TS4 |
|---|---:|---:|---:|---:|---:|
| SUA | `0.356828` | `−0.057766` | `−0.115319` | **`+0.414594`** | **`+0.472147`** |
| pseudo-MUA | `0.306073` | `−0.086878` | `−0.164314` | **`+0.392951`** | **`+0.470387`** |

All four contrasts are positive for 3/3 seed means and 15/15 session means. This establishes that correct
functional content and correct row attachment matter; it is not a width-only effect.

The label-budget replay kept the activity support at 30 trials and the evaluation window fixed after trial 50.

| T4 calibration trials | SUA R² | Δ vs M50 | pseudo-MUA R² | Δ vs M50 |
|---:|---:|---:|---:|---:|
| 10 | `0.304264` | `−0.052565` | `0.259185` | `−0.046888` |
| 15 | `0.338115` | `−0.018713` | `0.288234` | `−0.017839` |
| 20 | `0.351767` | `−0.005061` | `0.298447` | `−0.007626` |
| 30 | `0.358154` | `+0.001326` | `0.305291` | `−0.000782` |
| 50 | `0.356828` | `0` | `0.306073` | `0` |

M15 passes the frozen point-estimate and three-seed `−0.03` adequacy gate but not a CI-based non-inferiority
claim. M30 is the stronger deployment point: both bootstrap intervals remain inside the `−0.03` margin, and
the true-early-start replay shows positive decoding immediately after trial 30.

Against classical comparators at M50, T4 is much stronger than PV50 but below dense-label Ridge50 on mean R²:

| view | PV50 | T4 | Ridge50 |
|---|---:|---:|---:|
| SUA | `0.115374` | `0.356828` | `0.417922` |
| pseudo-MUA | `0.104219` | `0.306073` | `0.410193` |

This is an accuracy/supervision trade-off, not evidence that T4 dominates the ridge family. Across the 15
sessions, T4 consumes 750 trial-direction scalars; the evaluated Ridge50 implementation consumes 149,725
two-dimensional velocity rows, or 299,450 scalar target coordinates. The pooled ratios are `199.633x` by rows
and `399.267x` by scalar coordinates. These are algorithmic supervision counts, not independent-sample,
manual-annotation, compute, energy, or latency ratios.

### 3.2 Native M2: positive but scope-limited

The fresh matched seed-42 Stage-A development matrix has exact 14/14 cells:

| arm | seven-session equal-fold mean R² |
|---|---:|
| matched SPINT | `0.293110` |
| matched-decoder T4 | **`0.382906`** |
| T4−SPINT | **`+0.089796`** |

All seven session deltas are positive. This is strong matched development evidence, but it is not the missing
fresh three-seed r10 formal gate. An organizer-held native-M2 system comparison also reports
`T4−original=+0.11677`, but that original comparator is not a matched retraining and the difference cannot be
attributed solely to T4.

The strict local M24/q24 comparison gives T4 `0.226786` and fixed-lambda Ridge24-W50 `0.113916`, a difference
of `+0.112870`. This is one fixed classical implementation on six development sessions, not a ridge-family
frontier. The tested ridge performs a per-session closed-form solve without backpropagation.

### 3.3 RT: dense carrier terminal; sparse endpoint carrier in final matrix

The sealed dense-velocity RT comparison is terminal:

| system | mean R² |
|---|---:|
| dense Full carrier | **`0.4419498241`** |
| matched SPINT-structured B2-D1024 | `0.1451479751` |
| paired Full−B2 | **`+0.2968018490`** |

The difference is positive in 15/15 folds; paired median `+0.3168286171`, exact two-sided sign
`p=0.00006103515625`, and paired-fold bootstrap 95% interval
`[+0.2377831355,+0.3524398339]`. B2 is a same-pipeline SPINT-structured reference, not a claim to reproduce
every released-code RT detail.

The sparse RT route derives one direction per reach from go-cue-bounded endpoint coordinates. RT has no
non-degenerate native per-reach target-direction field, so this result cannot be called a native annotation-cost
comparison. The carrier estimator nevertheless consumes 1,103 reach-direction scalars versus 7,855 dense
two-dimensional velocity rows / 15,710 scalar coordinates for dense AFC4: `7.121x` by rows or `14.243x` by
coordinates. The production interpolation path actually reads 5,502 coordinate scalars, so its raw coordinate-I/O
ratio is only `2.855x`.

The CPU constructibility gate passed: split-half `[a,c]` cosine median `0.787119`; correct-minus-shuffle neural
forward-transfer mean/median `+0.035910/+0.031461`, 15/15 sessions positive.

The fresh GPU Stage-2 matrix is fixed at 15 folds × `T4d/Full/Zero4`, seed 42, M24/q24/W50, 35 epochs. At the
last authoritative update, folds 0--12 are complete:

| paired contrast, 13 complete folds | mean | median | drop-largest mean | positive folds |
|---|---:|---:|---:|---:|
| T4d−Zero4 | **`+0.277865`** | **`+0.350521`** | **`+0.267222`** | **13/13** |
| T4d−dense Full | `+0.006126` | `+0.004358` | `+0.000314` | 7/13 |

This is early evidence, not the terminal claim. At the current execution update, 39/45 cells are closed; fold12 is
fully closed and fold13 T4d is running. All fold13 scores remain excluded until its three-arm pair is complete. If the
final result remains similar, the scientific reading is that
one endpoint-direction label per reach preserves dense-carrier accuracy while the correct carrier content strongly
beats Zero4. It is not evidence that T4d is superior or equivalent to Full. Fold13 is currently running.

After 45/45 cells, the independent terminal verifier must pass. Then the 15 sealed B2 checkpoints will be evaluated
forward-only on the exact Stage-2 query windows. B2 will not be retrained, reselected, or updated.

### 3.4 H1: separate compact-consumer evidence

H1 uses dense behavior covariates and therefore does not support the sparse-label claim. It provides two separate
pieces of evidence:

- five-date development: `H-C−H-S=+0.056287`, 4/5 dates positive, paired date-bootstrap 95% interval
  `[+0.005965,+0.097195]`;
- matched compact-consumer content attribution: `H-C−H-C0=+0.032557`, 4/5 dates positive, but its date-bootstrap
  interval `[-0.004809,+0.079771]` touches zero;
- organizer-held system result: all-source CarrierID `0.274939±0.127206` versus paper-LR SPINT
  `0.261492±0.148717`, with normalized latency `0.113919` versus `0.129075`.

The compact H-C identity path has about 58k parameters versus 5.97M for H-S, roughly `102.6x` fewer. The running
CI64 experiment only tests whether widening the compact joint consumer improves H1; it is not a blocker for the
SUA/M2/RT mainline and must not be used to support sparse supervision.

### 3.5 M1: current negative boundary

The official M1 Original/T4/D4 held-out R² values are `0.648591/0.644766/0.643993`; neither carrier improves the
original system. The matched EMG-AFC4 content contrast is also negative (`B-C−B-C0=−0.00652`). M1 shows that
identity can be load-bearing while the tested task-aligned carrier still fails to add useful content. Do not group
this with H1: H1 has small positive content and organizer-held system evidence; M1 does not.

## 4. What did not work

The following routes are closed unless a new dataset or genuinely different estimand is declared:

- waveform, SNR, and static electrode lookup features: no reliable held-out carrier gain;
- same-electrode relation models and static T4GATE: ineffective;
- N4 label-free rate/Fano/autocorrelation/population-coupling descriptor: N4−NS4 `+0.001588` on M24 held-out,
  only 3/6 sessions positive;
- fixed-K temporal prototype and closed-form alignment pilots tested so far: no stable mainline improvement;
- RT live-activity gain L-D: `G-Full−A0=−0.002068`, below the frozen `+0.03` gate;
- M1 EMG-AFC4, D4, and the tested Version-B carrier: no carrier-content improvement;
- H1 H-PCF8 and native-phase precursors: failed source-only constructibility gates before GPU;
- deeper FiLM, carrier-biased attention, or additional fusion paths: not authorized by existing evidence.

These negative results do not show that every self-supervised representation or every decoder adaptation is
impossible. They close the tested implementations and prevent post-hoc rescue on already opened endpoints.

## 5. Paper claims

### Claims supported after RT terminal closure

1. An analytic functional carrier can adapt a frozen streaming neural decoder to new sessions without
   target-session backpropagation.
2. Correct functional content and unit/channel attachment matter; the effect is not reproduced by zero content,
   wrong rows, or static neural statistics.
3. The carrier transfers across SUA, deterministic pseudo-MUA, native M2, and RT movement decoding, with H1 as a
   separate dense-covariate compact-consumer module.
4. Useful accuracy can be obtained with substantially lower target-supervision density than the evaluated direct
   ridge/dense-carrier implementations.
5. A compact carrier-aware identity consumer can replace a much larger SPINT identity path on H1 with a small or
   positive system-level accuracy change.

### Claims not supported

- T4 is universally more accurate, faster, or cheaper than ridge;
- every dataset supplies native low-cost annotations;
- the entire source-training and scoring pipeline is sparse-label;
- RT T4d is better than dense Full before a valid superiority test;
- H1 proves sparse-label adaptation;
- M1 carrier content is positive;
- organizer-held mean differences are traditionally significant or session-consistent without per-session scores;
- calibration is label-free.

## 6. Remaining work before mainline closure

Only the following work can change the main paper:

1. finish RT folds 11--14 so the fixed 45-cell matrix is complete;
2. run the independent RT terminal verifier and freeze the 15-fold mean/median/sign/drop-largest statistics;
3. run the uniform 15-fold B2 forward-only exact-query companion and report both all-fold and prospective-fold
   summaries;
4. update the main results ledger, paper, and this handoff with terminal-only numbers;
5. run reference-aware document cleanup, recoverable checkpoint cleanup, focused tests, and narrow Git commits.

H1 CI64 may finish in parallel or be handed to another maintainer. It cannot delay items 1--5 or reopen the main
design search. Quantization is deferred and is not part of this closure.

## 7. Golden references

The following are the only status/protocol entry points a new maintainer should need:

- this handoff: `sua_exploration/docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`;
- concise result ledger: `sua_exploration/docs/CURRENT_RESULTS.md`;
- live execution board: `sua_exploration/docs/ACTIVE_EXPERIMENT_CONTROL_BOARD.md`;
- selected FP32 method contract: `sua_exploration/docs/FP32_T4_MAINLINE_PROTOCOL.md`;
- measurement rules: `sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md`;
- RT sparse frozen contract: `sua_exploration/docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md`;
- RT B2 companion protocol: `sua_exploration/docs/RT_SPARSE_T4D_VS_B2_D1024_COMPANION_PROTOCOL_20260810.md`;
- executable RT Stage-2 and B2 scripts/tests under `sua_exploration/scripts/` and `sua_exploration/tests/`;
- immutable JSON receipts and aggregates under ignored `sua_exploration/results/` and
  `SPINT-main/pilot_artifacts/`.

Older handoffs explain how individual hypotheses were proposed or rejected, but they are not current instructions.
Generated checkpoints, predictions, logs, and receipts remain outside Git; published numbers must cite their SHA
and terminal verifier.

## 8. Workspace and handoff policy

- Do not use `git add .` in this dirty multi-agent worktree.
- Do not delete active RT/H1 run roots, immutable receipts, terminal aggregates, or the best checkpoint retained for
  each completed run.
- Byte-identical periodic checkpoint aliases may be moved to a recoverable quarantine only after all relevant
  processes and watchers have exited and their SHA/retained-peer paths are recorded.
- Delete an old document or program only after confirming that no active script, test, receipt, current document,
  or paper source references it. If it is needed only for provenance, keep it outside the active navigation rather
  than presenting it as current guidance.
- The final GitHub update must be path-scoped and followed by remote-SHA verification.
