# M2 and SUA Complete Experiment Audit

Date: 2026-08-28

Status: complete audited M2/SUA experiment ledger. All predeclared low-cost
local comparisons and all three endpoint-compatible official M2 budget
submissions are terminal. Invalid endpoint combinations and stopped seed
expansions are recorded explicitly rather than treated as missing results.

## 1. Scope and common reporting rule

The immediate scope is:

- DANDI 000688 sorted SUA, including the deterministic pseudo-MUA electrode-sum
  view;
- FALCON native M2 threshold-crossing MUA;
- T4 at short calibration budgets and the causal dual-memory family where a
  post-support trial stream is available.

Every new behavior result must use a chronological support/query split, no
target-session backpropagation, the governing last-bin variance-weighted R2,
and equal-session aggregation. Paired per-session differences are primary.
SUA within-subject development, SUA external-subject, M2 local development,
and M2 organizer-hidden results are separate surfaces and must not be pooled.

## 2. Accepted results that must not be rerun

### 2.1 Static T4 on external-subject SUA and pseudo-MUA

The accepted 270-cell V9 result uses 15 sub-M sessions, three seeds, three
systems, and two neural views. Its authoritative TorchMetrics 1.5.1 aggregate
is:

| View | T4 mean R2 | Zero4 mean R2 | TS4 mean R2 | T4-Zero4 | T4-TS4 |
|---|---:|---:|---:|---:|---:|
| SUA | 0.356828 | -0.057766 | -0.115319 | +0.414594 | +0.472147 |
| pseudo-MUA | 0.306073 | -0.086878 | -0.164314 | +0.392951 | +0.470387 |

All four contrasts have 3/3 positive seed means and 15/15 positive session
means. This proves strong system-level value for source-trained T4 on the same
DANDI dataset and a second animal. It does not prove that a fixed SUA decoder
or a CDM state machine transfers unchanged to native FALCON M2.

Authority:

- `sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805/aggregate/endpoint_aggregate_torchmetrics151.json`
- status `FULL_270_LOCAL_TORCHMETRICS_1_5_1_FINALIZED`
- exactly 270 verified cells and 708,795 query windows per view.

### 2.2 Static T4 and CDM on DANDI 000688 SUA

The accepted sealed Cell-D, CDM-D V8, and Precision-CDM V2 results are:

| Budget | Surface | Sealed Cell-D | CDM-D V8 | Precision-CDM V2 | Best minus sealed |
|---|---|---:|---:|---:|---:|
| M30 | within-6 | 0.566518 | 0.558798 | not run | -0.007720 |
| M30 | external-15 | 0.428593 | 0.369501 | not run | -0.059092 |
| M10 | within-6 | 0.467718 | 0.530380 | 0.531656 | +0.063938 |
| M10 | external-15 | 0.295456 | 0.340083 | 0.351002 | +0.055546 |
| M4 | within-6 | 0.308920 | 0.493972 | 0.468633 | +0.185053 |
| M4 | external-15 | 0.119684 | 0.191126 | 0.222630 | +0.102946 |

Interpretation:

- M4 is the strongest CDM result. Precision gating improves the V8 external
  mean by +0.031504 and gives 14/15 positive sessions versus sealed. This is
  a repair relative to ordinary carrier adaptation, not evidence that carrier
  adaptation is better than freezing the carrier: activity-only is still
  0.003865 higher on the exact same external sessions.
- M10 is useful but not saturated. Precision gives only +0.010919 over V8 on
  external sessions, while the total gain over sealed is +0.055546. However,
  activity-only is 0.037807 higher than Precision on the exact same M10
  external input.
- ordinary CDM carrier updates are unsafe at M30. The deployment rule is a
  frozen support-only carrier at M30; no adaptive-carrier M30 result may be
  presented as a robustness win.
- all of these CDM results are seed 42 matched screens. They are not yet a
  multi-seed claim.

Accepted V8 bodies:

- closure `62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939`
- attempt `557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376`
- input `ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07`
- score `98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e`
- terminal `80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096`

### 2.3 FALCON M2 accepted anchors

The organizer-hidden system comparison is already complete:

| Official submission | Held-out R2 | Held-in R2 | Normalized latency |
|---|---:|---:|---:|
| original SPINT, 578218 | 0.186479 | 0.568243 | 0.112361 |
| submitted T4, 578221 | 0.303244 | 0.587608 | 0.042903 |
| M30 static T4 + activity30, 581362 | 0.295161 | 0.575440 | 0.044834 |
| M10 T4 + activity30, 581359 | 0.263833 | 0.558446 | 0.044338 |
| cue-budgeted M4 T4 + activity30, 581361 | 0.289744 | 0.562779 | 0.047012 |
| T4 578221 minus original | +0.116765 | +0.019366 | -0.069458 |
| M10 activity30 minus original | +0.077354 | -0.009797 | -0.068023 |
| M10 activity30 minus T4 578221 | -0.039411 | -0.029162 | +0.001435 |

This is an end-to-end system comparison, not a matched-decoder T4-only effect.

The legal local post-support anchors are:

| Endpoint | Sessions | F0 | T4 | Other | T4-F0 |
|---|---:|---:|---:|---:|---:|
| M33/q33 corrected eligible subset | 4 | 0.205823 | 0.277803 | TS4 0.211319 | +0.071980 |
| M24/q24 chronological disjoint | 6 | 0.173901 | 0.226786 | K4 0.245774; KS4 0.202252 | +0.052885 |

For M24, fixed dense-label Ridge-W50 is 0.113916. T4 and K4 are above it,
but K4 uses denser velocity supervision and K4-T4 is only +0.018987.

The fresh matched Phase-C seed-42 stage has SPINT 0.293110, T4 0.382906,
delta +0.089796, with 7/7 outer sessions positive. Its old r9 lineage is not
spliced into a new aggregate. Instead, Section 2.12 reports the cheaper
three-checkpoint frozen-inference replication; Section 3.3 records why new
full training seeds are stopped for this program.

### 2.4 SUA activity-only CDM quick screen

The successor V2 quick screen completed successfully on the exact V8 inputs.
It freezes the support-only fixed-ridge T4 carrier and changes only the B3S
activity FIFO. It makes zero carrier proposals and performs zero target
optimizer/backward/update operations.

| Budget | Surface | Sealed | Full CDM V8 | Activity-only | Activity-only minus sealed | Positive sessions |
|---|---|---:|---:|---:|---:|---:|
| M10 | within-6 | 0.467718 | 0.530380 | 0.541487 | +0.073769 | 6/6 |
| M10 | external-15 | 0.295456 | 0.340083 | 0.388809 | +0.093352 | 15/15 |
| M4 | within-6 | 0.308920 | 0.493972 | 0.465590 | +0.156670 | 6/6 |
| M4 | external-15 | 0.119684 | 0.191126 | 0.226495 | +0.106811 | 14/15 |

The external paired bootstrap 95% intervals versus sealed are
`[+0.070178,+0.117767]` at M10 and `[+0.064139,+0.149633]` at M4. All
predeclared gates passed and the verdict is `ADVANCE_ACTIVITY_ONLY`.

Activity-only also has a higher external equal-session mean than full CDM V8:
+0.048726 at M10 and +0.035369 at M4. These latter contrasts are descriptive,
not a clean superiority claim: their bootstrap intervals include zero and one
V8 session is an extreme negative outlier. The robust conclusion is narrower
and important: activity memory alone carries a large short-budget gain, while
carrier adaptation is not required to beat the sealed system and can subtract
from that gain in some sessions.

#### Exact Precision-CDM V2 versus activity-only pairing

The two accepted receipts contain the same 42 `(budget, surface, session)`
keys. For every key, `input_record_sha256`, `support_trial_ids_sha256`,
`target_last_bin_sha256`, `valid_mask_sha256`, and `n_windows` match exactly.
The following is therefore a direct paired comparison. Confidence intervals
are a deterministic 10,000-resample paired bootstrap of the equal-session
mean with seed 42.

| Budget | Surface | Activity-only | Precision-CDM V2 | Precision minus activity | Positive sessions | Paired 95% CI |
|---|---|---:|---:|---:|---:|---:|
| M10 | within-6 | 0.541487 | 0.531656 | -0.009832 | 2/6 | [-0.037092, +0.013280] |
| M10 | external-15 | 0.388809 | 0.351002 | -0.037807 | 4/15 | [-0.101635, +0.008249] |
| M4 | within-6 | 0.465590 | 0.468633 | +0.003042 | 2/6 | [-0.014301, +0.024484] |
| M4 | external-15 | 0.226495 | 0.222630 | -0.003865 | 3/15 | [-0.012738, +0.003481] |

No completed surface demonstrates Precision-CDM V2 superiority over
activity-only. Precision does improve ordinary CDM on external M10/M4, so the
precision gate has diagnostic value as a partial repair for unsafe carrier
updates. It is not currently the selected deployment system. Activity-only
remains the simplest and strongest accepted SUA route. Any broader precision
claim must beat activity-only on same-input pseudo-MUA or native-M2 cells; a
positive precision-minus-ordinary contrast alone is no longer sufficient.

Immutable V2 result graph:

- attempt `9c43b0ee5f67cdd8300856569939124ecfccd3213a3028bf07fa07de99417ec6`
- input authority `484aa399fa15179b74ad9ed2a485a99ace5047a6ba0cf55b319487d42302f133`
- result `48d658c866b0bcb45a9f03c6e0c84204ecb87286e940d850c846a46f67b33bb8`
- terminal `707a33f6561d680e92454e4d59aaf0ff31618a8fcfcbfb681845ff56548f63fb`
- topology: exactly attempt/result/terminal body+sidecar pairs, all 0444,
  terminal root 0555, no failure.

### 2.5 Native M2 frozen-weight activity-budget screen

The first native-M2 activity-axis screen is complete on the selected seed-42
B3S+T4 checkpoint. It uses fixed ridge-T4 lambda 0.1, no training, no query
labels, no gradients, and no parameter updates. M4 uses causal D-optimal
selection among non-centre directional trials inside the first-30 calibration
pool; M10/M30 use chronological support. The activity-expanded cells retain
only M carrier labels but give B3S the first-30 calibration activity.

| Budget | Surface | Static ridge-T4 | Activity30 + same carrier | Delta | Positive sessions |
|---|---|---:|---:|---:|---:|
| M30 | within post-30, 7 | 0.689261 | same definition | 0 | 7 sessions |
| M10 | within post-30, 7 | 0.555846 | 0.666265 | +0.110420 | 7/7 |
| M4 | within post-30, 7 | 0.451784 | 0.677220 | +0.225436 | 7/7 |
| M30 | external official-query, 6 | 0.295220 | same definition | 0 | 6 sessions |
| M10 | external official-query, 6 | 0.223252 | 0.272322 | +0.049070 | 6/6 |
| M4 | external official-query, 6 | 0.222720 | 0.290992 | +0.068272 | 5/6 |

This is the first independent-task replication of the CDM activity mechanism.
The external M4 activity row is only 0.004227 below the M30 safety anchor;
within-post30 M4 is 0.012041 below M30. The result therefore supports activity
coverage, not carrier adaptation, as the next main route.

The M4 support accounting must remain explicit: D-opt selection reads the
target cue/direction metadata of the first-30 candidate pool, although only
four trials enter the T4 fit. This is a cue-budgeted M4 result, not a pure
four-metadata-observation chronological result. M10/M30 do not have this
distinction.

Result authority:

- checkpoint `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`
- normalization `d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e`
- score `6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce`
- 65 session-cell rows, batch 1024, GPU1, 23.04 seconds, result pair 0444
  under a 0555 root.

### 2.6 Existing three-seed SUA/pseudo-MUA label-budget and classical controls

The completed 450-cell label-budget curve must be treated as accepted evidence,
not rerun. It uses the same 15 external sub-M sessions, both neural views,
seeds 42/43/44, a fixed first-30 B3S activity identity, a chronological T4
label prefix, and one common query strictly after rewarded trial 50.

| View | T4 M10 | T4 M15 | T4 M20 | T4 M30 | T4 M40 | T4 M50 | Smallest all-seed budget within 0.03 of M50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUA | 0.304264 | 0.338115 | 0.351767 | 0.358154 | 0.351648 | 0.356828 | M15 |
| pseudo-MUA | 0.259185 | 0.288234 | 0.298447 | 0.305291 | 0.301075 | 0.306073 | M15 |

At M10 the deltas from M50 are -0.052565 for SUA and -0.046888 for
pseudo-MUA. At M30 they are +0.001326 and -0.000782. These are already
activity30/short-carrier-label results: they establish the short-label curve,
but do not isolate activity30 from activityM because the matched activityM arm
was not included. M4 is also absent.

Authority:

- `sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json`
- SHA `12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2`
- status `FULL_450_LOCAL_TORCHMETRICS_1_5_1_FINALIZED`.

The separate finalized 150-cell classical-control result uses the same
external cohort and query endpoint:

| View | Historical F0 means across seeds | PV50 | Ridge50 | T4 M50 |
|---|---|---:|---:|---:|
| SUA | -0.215048 / -0.227371 / -0.145813 | 0.115374 | 0.417922 | 0.356828 |
| pseudo-MUA | -0.202339 / -0.162278 / -0.249558 | 0.104219 | 0.410193 | 0.306073 |

PV50 and Ridge50 use dense behavior samples from the first 50 trials and are
stronger-label performance comparators, not equal-information T4 controls.
Their aggregate SHA is
`ffccd91fc128edb7ad6199671f2e32d0c6c450cdff4f2b3d734a1815b176ebc0`
with status `FULL_150_LOCAL_TORCHMETRICS_1_5_1_FINALIZED`.

### 2.7 Paired three-seed SUA/pseudo-MUA activity-budget screen

The paired successor has now terminalized.  It uses exactly the same 15
external sessions, query targets, two signal views, and shared-T4 checkpoints
for seeds 42/43/44.  M10 compares first-10 versus first-30 B3S activity with
the same M10 OLS carrier.  M4 compares D-optimal four-trial versus first-30
B3S activity with the same fixed-ridge M4 carrier.

| View | Budget | Activity M | Activity 30 | Delta | Positive seed means | Positive sessions after seed averaging | Hierarchical 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUA | M10 | 0.273699 | 0.304264 | +0.030565 | 3/3 | 11/15 | [+0.010569,+0.052887] |
| SUA | M4 | 0.114616 | 0.160935 | +0.046319 | 3/3 | 9/15 | [-0.003844,+0.099463] |
| pseudo-MUA | M10 | 0.223314 | 0.259185 | +0.035871 | 3/3 | 12/15 | [+0.012704,+0.061353] |
| pseudo-MUA | M4 | 0.059368 | 0.107111 | +0.047743 | 3/3 | 11/15 | [-0.008986,+0.107568] |

The M10 activity effect is replicated with a positive interval on both signal
views.  M4 has a positive three-seed mean on both views but its hierarchical
interval crosses zero, so it is supportive rather than a completed breadth
claim.  The SUA-minus-pseudo difference in the activity gain is only
-0.005306 at M10 and -0.001425 at M4; activity coverage is therefore not
visibly dependent on spike sorting in this cohort.

Authority: score SHA
`487d3b18be104ae9a6d40c517de2637162a773a97baeb1e872c44684cb170853`;
360 aligned rows = 270 new + 90 immutable references; exactly two views x
three seeds x 15 sessions x four cells; all R2 finite; target gradients,
backward calls, and updates zero.

### 2.8 Existing chronological SUA classical budget comparators

The accepted `calibration_budget_comparators_v1` receipt already supplies
chronological-first-M classical controls on six within and 15 external sorted
SUA sessions.  It is useful accepted evidence and must not be rerun, but its
M4 support is chronological first-4 rather than the D-optimal four-trial
support in Section 2.7.

| System | Within M4/M10/M30 | External M4/M10/M30 |
|---|---|---|
| Trial-rate Ridge | -0.036194 / 0.033909 / 0.083517 | -0.079130 / 0.020522 / 0.093295 |
| Dense W50 Ridge | 0.085337 / 0.182903 / 0.306302 | -0.668431 / -0.144636 / 0.330755 |
| Population Vector | 0.045151 / 0.064246 / 0.091484 | 0.022425 / 0.072417 / 0.110392 |
| Cell-D fixed-ridge T4, activity30 | 0.453729 / 0.536870 / 0.566518 | 0.166311 / 0.370259 / 0.428593 |

This table shows that classical dense supervision is not automatically a
strong short-budget baseline.  It does not close the remaining pseudo-MUA
classical comparison and cannot replace a D-opt matched-M4 control.

### 2.9 Pseudo-MUA Precision-CDM V2 four-arm screen

The pseudo-MUA seed-42 screen terminalized successfully. It contains exactly
150 finite rows: 15 external sessions times ten predeclared cells. For every
`(budget, session)` group, all arms bind the same asset, query starts, target,
window count, raw-M30 T4, raw validity mask, and support directions. All target
gradient, backward, state-use, and parameter-update counters are zero. The M30
Precision cell is a literal reuse of the static prediction: every session has
the same prediction SHA and exactly equal R2.

| Budget | Static T4 | Activity-only | Ordinary CDM | Precision V2 |
|---:|---:|---:|---:|---:|
| M30 | 0.271245 | n/a | n/a | 0.271245, exact no-op |
| M10 | 0.177178 | **0.238078** | 0.196937 | 0.210377 |
| M4 | 0.055270 | 0.101885 | 0.027076 | **0.102542** |

Same-input paired contrasts use 10,000 deterministic equal-session bootstrap
resamples with seed 42:

| Budget | Contrast | Mean delta | Positive sessions | Paired 95% CI |
|---:|---|---:|---:|---:|
| M10 | activity-only minus static | +0.060901 | 13/15 | [+0.034099, +0.089051] |
| M10 | Precision minus ordinary | +0.013441 | 2/15 | [-0.003338, +0.043580] |
| M10 | Precision minus activity-only | **-0.027701** | 5/15 | [-0.079294, +0.023216] |
| M4 | activity-only minus static | +0.046615 | 10/15 | [-0.012429, +0.105064] |
| M4 | Precision minus ordinary | +0.075466 | 11/15 | [+0.020127, +0.133725] |
| M4 | Precision minus activity-only | **+0.000658** | 5/15 | [-0.035197, +0.036626] |

Precision rejected 295 proposed carrier transitions at M10 and 721 at M4;
it accepted 781 and 115, respectively. Ordinary CDM accepted 1,071 at M10
and 1,370 at M4. This is strong evidence that the precision gate suppresses
unsafe carrier adaptation, especially at M4, but it still does not add
performance beyond activity memory. M10 activity-only is clearly best; M4
Precision and activity-only are practically tied. No pseudo-MUA seeds 43/44
are warranted for a Precision-superiority claim.

Immutable engineering-result evidence:

- result root `results/pseudo_mua_precision_cdm_v2_screen_v1`, mode 0555;
- score body and sidecar mode 0444;
- score SHA `97b2aa457785970cd2df6ad61bcca54136c4cc5352e9fcd5314b49b0396582b2`;
- the running process loaded the pre-expanded summary helper, so its embedded
  summary contains Precision-minus-ordinary only. The raw 150 rows are the
  authoritative evidence for the post-terminal activity contrasts above; no
  prediction or transition code changed during execution.

### 2.10 Native M2 same-query 21-cell comparator

The native M2 comparator terminalized with 273 finite rows: 13 sessions times
21 cells, split into seven within-post30 and six external-official-query
sessions. It uses one fixed seed-42 model family, zero target gradients and
updates, and reuses the accepted fixed-ridge/activity rows by exact parent
score SHA. The following table gives equal-session means; slash-separated
values are M30/M10/M4.

| System | Within post-30 | External official-query |
|---|---:|---:|
| Original SPINT, chronological activity | 0.696579 / 0.508843 / 0.289672 | 0.239235 / 0.181411 / 0.056392 |
| Original SPINT, D-opt matched M4 only | M4 0.390109 | M4 0.157962 |
| T4 population-mean side intervention | 0.595122 / 0.464028 / 0.346876 | 0.203885 / 0.175649 / 0.161850 |
| T4 production OLS | 0.695066 / 0.571886 / 0.463499 | 0.300196 / 0.194780 / 0.223550 |
| T4 fixed ridge, static | 0.689261 / 0.555846 / 0.451784 | 0.295220 / 0.223252 / 0.222720 |
| T4 fixed ridge, activity30 | M10 0.666265 / M4 0.677220 | M10 0.272322 / M4 0.290992 |
| Direct dense W50 ridge | 0.008973 / 0.007048 / -0.001631 | 0.032859 / 0.009781 / -0.000640 |
| Population vector | 0.001433 / -0.002914 / -0.006009 | -0.000925 / -0.006227 / -0.022753 |

The result resolves several comparison questions. Production OLS T4 beats
chronological SPINT on external M30/M10/M4 by +0.060961, +0.013370, and
+0.167158, respectively. Ridge and OLS are nearly tied on external M4
(`ridge-OLS=-0.000830`), while dense direct ridge and population vector fail
badly despite consuming behavior supervision. Most importantly, giving B3S
the first-30 unlabelled activity raises fixed-ridge T4 by +0.049070 at M10
(6/6 positive) and +0.068272 at M4 (5/6 positive) on the official-query
surface. The large M4 gap between chronological and D-opt matched SPINT
(+0.101569 external) is retained as a selection-control effect, not attributed
to the decoder architecture.

Immutable result evidence:

- result root `results/m2_same_query_comparator_v1`, mode 0555;
- score and sidecar mode 0444;
- score SHA `5efa738ba536ee7cfc624a9dd79fd4ceedb36ae54f482cdcd48e59e355dc3d8b`;
- 273 rows = 13 sessions x 21 predeclared cells; target gradients and
  parameter updates are zero.

### 2.11 Native M2 Precision-CDM V2 four-arm screen

The native seed-42 local trial-aware screen terminalized with exactly 130
finite rows: seven within-post30 and six external-post30-local sessions times
ten cells. All arms within a `(budget, session)` bind the same query starts,
target, window count, checkpoint, normalizer, raw-M30 T4 and validity mask;
M4/M10 arms additionally bind the same selected support indices. All target
gradient, backward, state-use, and parameter-update counters are zero. M30 is
an exact static/Precision no-op in prediction SHA and R2 for every session.

| Budget | Surface | Static T4 | Activity-only | Ordinary CDM | Precision V2 | Activity minus static |
|---:|---|---:|---:|---:|---:|---:|
| M30 | within-post30 | 0.689639 | n/a | n/a | 0.689639 | exact no-op |
| M10 | within-post30 | 0.573516 | 0.665390 | 0.665390 | 0.665390 | +0.091874, 7/7 |
| M4 | within-post30 | 0.458196 | 0.654086 | 0.654086 | 0.654086 | +0.195890, 7/7 |
| M30 | external-post30-local | 0.250363 | n/a | n/a | 0.250363 | exact no-op |
| M10 | external-post30-local | 0.193573 | 0.209687 | 0.209687 | 0.209687 | +0.016115, 5/6 |
| M4 | external-post30-local | 0.250413 | 0.299057 | 0.299057 | 0.299057 | +0.048645, 5/6 |

The paired 10,000-resample seed-42 bootstrap interval for activity-minus-static
is `[+0.051300,+0.138060]` within M10, `[+0.160093,+0.235361]`
within M4, `[-0.015763,+0.045534]` external M10, and
`[+0.004329,+0.086020]` external M4.

This run does **not** validate the Precision gate on native M2. Activity-only,
ordinary CDM, and Precision produced the exact same prediction SHA and R2 in
every M4/M10 session. Ordinary and Precision each recorded zero accepted
carrier updates and 1,767 typed non-Precision carrier rejections at each
budget; Precision-specific rejection count was zero. Thus the earlier core
carrier/pseudo gates stopped every proposal before the precision credible-region
decision could matter. The result positively replicates trial-structured
activity memory, but makes carrier adaptation and Precision observationally
inert. Seeds 43/44 for native Precision are therefore not warranted.

The local adapter records 167 completed query trials shorter than one 50-bin
decoder window. Those trials advance activity and receive typed
`velocity_shape` carrier rejection, with no padding or cross-trial history.
Absolute values are local CDM-adapter results built from complete raw-trial Hz
carriers and must not be substituted for the production filtered-calibration
T4 baseline in Section 2.10.

Immutable result evidence:

- result root `results/m2_precision_cdm_v2_screen_v1`, mode 0555;
- score and sidecar mode 0444;
- score SHA `455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6`;
- 130 rows = 13 sessions x 10 predeclared cells.

### 2.12 Native M2 three-seed activity-memory replication

The frozen-inference replication terminalized with 195 rows: 65 exact seed-42
parent rows plus 130 new seed43/44 forwards. All three checkpoints have distinct
bound body SHAs but share the exact frozen T4 normalizer. Each seed/session has
the same query starts, targets, and window count across its five cells; all
R2 values are finite and all target-gradient/update counters are zero.

| Surface | Budget | Mean activity30-minus-static across seeds | Seed means 42 / 43 / 44 | Positive seed means | Positive sessions after seed averaging | Hierarchical 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| within-post30 | M10 | +0.106510 | +0.110420 / +0.095775 / +0.113335 | 3/3 | 7/7 | [+0.088598, +0.124870] |
| within-post30 | M4 | +0.219581 | +0.225436 / +0.192217 / +0.241091 | 3/3 | 7/7 | [+0.190821, +0.248368] |
| external official-query | M10 | +0.040569 | +0.049070 / +0.034418 / +0.038220 | 3/3 | 6/6 | [+0.026935, +0.053333] |
| external official-query | M4 | +0.062467 | +0.068272 / +0.040230 / +0.078898 | 3/3 | 5/6 | [+0.028942, +0.097589] |

The intervals are a deterministic seed-42 two-level bootstrap that resamples
the three frozen checkpoints and then sessions within each sampled seed. This
is strong evidence that first-30 unlabelled calibration activity, not a
particular model seed, drives the native-M2 short-budget improvement. M10 is
the cleanest official candidate because its ten support labels and cue rows
are chronological. M4 has the larger effect but remains explicitly
cue-budgeted: four labels enter the T4 fit while D-opt selection reads finite
direction metadata across the first-30 candidate pool.

Immutable result evidence:

- result root `results/m2_t4_activity_budget_seed_replication_v1`, mode 0555;
- score and sidecar mode 0444;
- score SHA `33446390408a610c552cccfac2b892f7cf7add5ab72dc42161c4d83c5027b4c9`;
- 195 rows, 65 reused and 130 newly forwarded, all zero-update.

### 2.13 Official-compatible activity30 payloads and public minival

The selected frozen inference route has now been exported into three independent
official-compatible cached-identity payloads. All three use the same seed-42
checkpoint, decoder, source normalizer, 13-session coverage, and first-30 B3S
activity. M10 uses the chronological first ten labelled/cue rows. M4 uses four
labels selected D-optimally after reading finite direction metadata in the
first-30 candidate rows, so it is explicitly **cue-budgeted M4**, not strict
four-cue M4. M30 is the chronological static safety reference: its T4 and B3S
inputs both use first30 and it has no adaptive state.

Both payload exporters proved direct identity computation, cached-identity
forward, and decoder-only forward bit-exact (`max_abs=0`) for all 13 sessions.
The images then loaded the payloads under the historical official namespace,
constructed the frozen `SpintModel`, and kept CUDA unavailable and
uninitialized. The public-minival comparison uses the same seven NWBs and the
same GT pickle SHA
`546e5a4a0b2260cdd4c7124d4d668fe0f136f84a81f7aa472e7dbe39a3115530`.

| Public M2 minival image | Held-in R2 mean | Held-in R2 std. | Delta vs E8 anchor | CPU-container normalized latency |
|---|---:|---:|---:|---:|
| E8 epoch27 anchor | 0.523848601 | 0.130990970 | reference | 0.456569539 |
| ridge T4 M10 + activity30 | **0.635452852** | 0.066619214 | **+0.111604251** | 0.125665280 |
| D-opt ridge T4 M4 + activity30 | **0.636385188** | 0.077477336 | **+0.112536587** | 0.129248598 |
| ridge T4 M30 + activity30 | **0.637800992** | 0.061056325 | **+0.113952391** | 0.111545488 |

The M4-minus-M10 public-minival difference is only `+0.000932336`. The
M30-minus-M10 difference is only `+0.002348140`, and M30-minus-M4 is only
`+0.001415804`. Thus both short-label systems are within 0.0024 R2 of the
30-label safety reference on this public surface. The date-reduced R2 values
also show that no short-budget arm dominates every date.

| Date | M10 + activity30 | M4 + activity30 | M30 static |
|---|---:|---:|---:|
| 20201019 | 0.734357 | 0.750347 | 0.730465 |
| 20201020 | 0.657316 | 0.664145 | 0.654959 |
| 20201027 | 0.585073 | 0.555262 | 0.582898 |
| 20201028 | 0.565066 | 0.575787 | 0.582881 |

The latency values above are CPU-container diagnostics, not hidden EvalAI
latency. All three images also passed the exact remote filesystem simulation:
`/dataset/evaluation_data/m2` was mounted read-only, each image found all seven
NWB files, wrote an 11,435-byte `/submission/submission.csv`, and entered the
expected 300-second server polling wait. No network or GPU was used in these
local validations.

Immutable local evidence:

- M10 payload SHA `e371541f27cce2d3b4fd418f8ab54663d66010e31da882658b9c39d0df9e339e`;
- M10 image ID `sha256:381adc2b4ba8bd741db1b6e9c6aaaeff17bd3c0ee88de5b79c3ca9c0b6fc784b`;
- M4 payload SHA `c51b71167d81490927fee8a552785ee27ddff7e90085ed3ff4b18b857afbac40`;
- M4 image ID `sha256:378bbd4868e515a3527418e098db2989e0fc8fbccc929ee029a237ce5e9f291b`;
- M30 payload SHA `097b0fdf83ac1ac0846b36fc02c6e4f0a755f477514c83c4de8fcd08572797ae`;
- M30 image ID `sha256:5bf9682c4c96aa27d51b7db8584f634411fe7b7b24bdb1d23052f73c452c86eb`;
- combined local-validation receipt SHA
  `ddd865ac25f195326fcd124d747933f24456a6ee0aa3e7f242cca092ae49f53b`;
- all payload, receipt, prediction, GT, and remote-path output bodies are mode
  0444.

Authenticated read-only preflight confirmed phase 4599
`few-shot-test-2319`, team 41975, private submission, active/unpaused state,
image-size compliance, and available quota. The strict M10 candidate was
submitted first as private submission `581359` at
`2026-08-28T11:37:18.661264Z`; its uploaded ECR manifest digest exactly equals
the local image ID. It reached the official private `finished` state without a
retry. The official result is:

| Official submission | Held-out R2 mean | Held-out std. | Held-in R2 mean | Held-in std. | Normalized latency |
|---|---:|---:|---:|---:|---:|
| original SPINT anchor 578218 | 0.186478720 | 0.162144616 | 0.568242670 | 0.031164882 | 0.112361031 |
| historical T4 anchor 578221 | 0.303243951 | 0.108234005 | 0.587608272 | 0.025462358 | 0.042902605 |
| **M30 static + activity30, 581362** | **0.295160713** | 0.108884101 | **0.575440441** | 0.026353585 | 0.044834056 |
| **M10 + activity30, 581359** | **0.263832896** | 0.099040158 | **0.558445783** | 0.028171075 | 0.044338009 |
| **cue-budgeted M4 + activity30, 581361** | **0.289743988** | 0.103868803 | **0.562778893** | 0.017415237 | 0.047011591 |
| delta, 581359 minus original SPINT | **+0.077354175** | n/a | -0.009796887 | n/a | -0.068023021 |
| delta, 581359 minus 578221 | **-0.039411056** | n/a | **-0.029162489** | n/a | +0.001435405 |

The official held-out ordering is therefore historical T4 (`0.303244`) > M30
(`0.295161`) > cue-budgeted M4 (`0.289744`) > M10 (`0.263833`) > original
SPINT (`0.186479`). M4 is only `0.005417` below M30 despite using four target
labels, and it is `0.025911` above M10. M30 is `0.008083` below the historical
T4 anchor; M4 is `0.013500` below it. Thus the short-budget result is useful,
but none of the three new candidates is a new official best. M4's result must
remain labelled cue-budgeted because its D-opt selector reads finite direction
metadata from the first-30 public calibration pool.

This is also a decisive official-surface correction to the strong
public-minival result. M10 activity30 improves held-out R2 substantially over
the original SPINT submission, but the public near-equality of M4/M10/M30 does
not persist on organizer-hidden data. All three official prediction payloads
are finite, have the same keys/shapes/dtypes as successful submission 578221,
and match their official stdout metrics exactly. Terminal receipt SHAs are:

- M10/581359: `a747ece42f8b7b467767e4d86debc0368dcc3df0db81195f0849967b35f78c07`;
- M4/581361: `68da5427e2d27169b8b6e1e08a487113b65b734a7e7644a5d1778f993febbb97`;
- M30/581362: `ea7d4185943602b964accd98e4c8a90758350fb689227efe0fa8ff64fbbe9c5e`.
EvalAI briefly exposed an intermediate `failed` state after the prediction job
had written a valid 1,258,463-byte submission but before the comparison worker
ran. The same submission then advanced to `finished` about five minutes later;
no retry or second submission was created. Accordingly, M4/M30 monitoring
requires a stable failure across at least two comparison polling windows
rather than treating this transient state as a scientific failure.

M4 and M30 each passed fresh authenticated preflight and were submitted
exactly once as private submissions `581361` and `581362`. Their uploaded
manifest digests exactly equal their local image IDs. A transient connection
failure occurred before the successful M30 preflight; it created no state,
push, registration, or quota use. No failed candidate was reported as a
performance result.

## 3. Coverage matrix and explicit stops

### 3.1 Highest-priority low-cost SUA cells

| Experiment | M30 | M10 | M4 | SUA within | SUA external | pseudo-MUA | Status |
|---|---|---|---|---|---|---|---|
| Activity-only CDM, frozen support carrier | exact sealed no-op | complete | complete | complete | complete | paired M10 and M4 activity-axis complete | accepted SUA V2 plus three-seed paired signal-view screen |
| Precision-CDM seeds 43 and 44 | reference only | seed42 complete | seed42 complete | complete seed42 | complete seed42 | seed42 complete | stopped: no Precision-over-activity signal |
| V8 ordinary CDM pseudo-MUA | safety reference | complete | complete | not primary | reference only | complete | completed inside four-arm pseudo-MUA screen |
| Precision-CDM pseudo-MUA | exact no-op complete | complete | complete | not primary | reference only | complete | seed42 complete; no superiority over activity-only, no seed expansion |

Activity-only is first because it changes no model or training rule and directly
tests whether V8's short-budget gain came from B3S activity memory or risky
pseudo-label carrier updates. M30 is not a new cell: zero activity capacity plus
a frozen carrier is exactly the sealed support-only system.

### 3.2 FALCON M2 T4+CDM cells

FALCON M2 needs its own post-support evaluator. DANDI session identifiers,
checkpoints, normalizers, canonical direction bins, and exact trial schema may
not be reused.

The local trial-aware M2 matrix is complete for the requested M4/M10/M30
budgets. The immutable coverage is:

| Budget | Support/query rule | Completed systems |
|---|---|---|
| M4 | D-opt 4 from first-30 cue pool; query strictly after the fixed comparison boundary | F0, chronological and D-opt SPINT, mean-side/OLS/fixed-ridge T4, Dense Ridge-W50, PV, static T4, activity-only, ordinary independent-activity CDM, Precision-CDM V2 |
| M10 | first 10 trials; same fixed query boundary | F0, chronological SPINT, mean-side/OLS/fixed-ridge T4, Dense Ridge-W50, PV, static T4, activity-only, ordinary independent-activity CDM, Precision-CDM V2 |
| M30 | first 30; query after 30 | F0, chronological SPINT, mean-side/OLS/fixed-ridge T4, Dense Ridge-W50, PV, static T4, and exact activity/Precision no-op safety references |

For cross-budget comparisons, the query boundary must be common and no earlier
than trial 30. A second within-budget endpoint may use query after M, but it
must be reported separately because its scored windows differ.

Before any M2 CDM behavior result, the adapter must prove:

1. exact native 96-channel order and threshold-crossing identity;
2. continuous M2 target direction about `(0.5, 0.5)`, with centre trials
   excluded from the T4 fit;
3. support-only closed-form carrier and rank/conditioning evidence;
4. variable-length B3S support at M4/M10/M24/M30;
5. completed-trial boundaries and next-trial-only memory updates;
6. no target-session gradients or parameter updates;
7. exact last-bin R2 and common-query input digests.

The organizer-hidden M2 endpoint does not expose the same causal post-support
trial-update protocol. This is the precise endpoint-based impossibility for an
official CDM submission: without a separately validated online trial/change
detector, the official runtime cannot implement the tested state transition.
The official submissions therefore contain only endpoint-compatible frozen
T4/activity identities and no adaptive CDM state.

### 3.3 Three-seed M2 matched closure decision

A new 7-fold x 3-seed training successor is stopped for this program. The
cheaper frozen-inference replication already uses three independently trained
T4 checkpoints and shows the activity effect with the same sign for all three
seeds on both M4 and M10. More importantly, the official M10 result improves
over original SPINT but remains below the existing official T4 anchor. Fresh
training seeds would quantify another variance component without changing the
method that failed to set a new official best. They become justified only
after a materially new endpoint-compatible mechanism is proposed; historical
r7-r9 cells remain audit evidence and are not spliced into a new aggregate.

### 3.4 Active 2026-08-28 successor queue

Three bounded successors now make the remaining low-cost matrix executable
without retraining or changing a network:

1. `sua_paired_activity_budget_screen_v1` is complete and independently
   audited.  Its 360 aligned rows, result SHA, and contrasts are recorded in
   Section 2.7.
2. `m2_same_query_comparator_v1` is complete with 273 rows and score SHA
   `5efa738b...`. It adds original SPINT with honest chronological
   M30/M10/M4 activity, a separately named D-opt M4 selection control,
   normalized-mean-side and production OLS T4, dense-label direct Ridge-W50,
   and PV. It reuses the five accepted ridge/activity rows by exact parent
   score SHA. Its principal numbers are recorded in Section 2.10.
3. `m2_t4_activity_budget_seed_replication_v1` is complete. It binds all three already-trained
   T4 checkpoints (42/43/44) and the common frozen normalizer. It reuses the 65
   seed42 rows and adds 130 seed43/44 forwards for the same five
   M30/M10/M4 cells. Its score SHA is `33446390...`; results are in Section
   2.12. This is a cheap
   frozen-inference seed replication, not the more expensive fresh 7-fold
   training closure described in Section 3.3.

The two M2 successors share one queue but produce distinct evidence: the first
answers baseline and supervision comparisons at seed42; the second answers
whether the activity-only gain is stable across the three existing T4
training seeds.

### 3.5 Precision-CDM V2 cross-dataset validation

Precision-CDM V2 is a candidate carrier-adaptation repair, not the selected
deployment mechanism. On external sorted SUA it improved ordinary CDM by
+0.031504 at M4 and +0.010919 at M10, but it did not beat activity-only on
either budget. Its proposed broader claim must therefore be tested against
activity-only, not merely against ordinary CDM, on two orthogonal
replications:

1. pseudo-MUA on DANDI 000688, which changes neural signal granularity while
   retaining task/session chronology;
2. native FALCON M2 local trial-aware evaluation, which changes source data,
   model artifact family, and parser/interface.

Both seed-42 replications are now complete. Pseudo-MUA Precision is tied with
activity-only at M4 and worse at M10; native M2 Precision is exactly identical
to activity-only because all carrier proposals fail before the precision gate.
The predeclared seed-expansion criterion is not met, so seeds43/44 are stopped.

For each replication, M4 and M10 must compare the same support/query rows for
`static T4`, `activity-only`, `ordinary independent-activity CDM`, and
`Precision-CDM V2`.  M30 is a frozen-carrier no-op safety reference, not an
adaptive precision cell. Seed42 is the mechanism screen; seeds43/44 are run
only after at least one external M4/M10 precision-minus-activity contrast is
positive, paired breadth is credible, and M30 does not regress. The precision receipt must report the
support-only conditional Gaussian-ridge covariance, Bonferroni familywise
threshold, acceptance/rejection counts, and per-session paired R2.

This mechanism is not forced onto M1, which has no homologous T4 carrier, or
onto H1 without a new carrier with demonstrated headroom.  The official M2
endpoint also remains ineligible for online CDM until trial boundaries are
available or separately detected.

## 4. Execution order

1. Treat the accepted activity-only V2 result as the selected SUA route.
2. Treat the completed paired SUA/pseudo-MUA activity successor as accepted;
   do not rerun the accepted 450-cell M10-M50 curve or 150-cell M50 controls.
3. The M2 same-query 21-cell comparator is complete. Chronological
   original-SPINT M4 remains separately named from the D-opt selection-control
   row.
4. The seed43/44 frozen-inference M2 activity replication is complete; all
   three existing checkpoints agree on the sign of the paired activity gain.
5. The pseudo-MUA and native-M2 Precision-CDM V2 matched successors are
   complete. Their Precision-minus-activity criterion failed, so seeds43/44
   Precision expansion is stopped rather than run selectively.
6. Fresh M2 matched Phase-C training is stopped under Section 3.3; it is not a
   missing low-cost comparison.
7. Only endpoint-compatible frozen systems were packaged for official
   evaluation. Trial-aware CDM remains local development evidence and was not
   forced into an invalid official submission.

### 4.1 Activity-only quick V1 execution record

The first quick-V1 physical attempt was reserved once and failed closed during
`materialize_inputs`, before opening a source NWB, loading a checkpoint,
initializing CUDA, or executing a forward. The inherited held-data reader
required explicit `SUBC_DATA_ROOT` and `SUBM_DATA_ROOT` launch variables; the
quick-V1 CLI did not bind them. The immutable failed root is:

- `results/causal_dual_memory_cell_d_activity_only_quick_v1`
- attempt SHA `be00e6346383d90808d4772d7a46589a07c3765fb0317113e6dbb201581e48b7`
- failure SHA `42210c7eba20af2123e5d586338efe5f40c6459148efdf3e12065d510028cb25`
- stage `materialize_inputs`, `input_authority_sha256_or_null=null`

Do not edit, overwrite, or retry this root. A successor must bind the failed
pair and require the canonical environment values before reserving its own
fresh root.

### 4.2 Activity-only quick V2 terminal record

V2 bound the V1 failed graph and the canonical sub-C/sub-M data roots before
reserving its fresh root. It completed in 860.1 seconds with peak process RSS
about 14.03 GB. GPU1 was used; GPU0 and unrelated workloads were untouched.
The result has four cells only: within/external x M10/M4. M30 is definitionally
the sealed no-op because activity capacity is zero. The run is a non-governing
matched engineering screen, not a multi-seed terminal claim.

## 5. Completion criteria

This program is complete because:

- every required row above has an immutable result or a precise endpoint-based
  impossibility receipt;
- SUA and pseudo-MUA are reported separately;
- M2 local and official surfaces are reported separately;
- M4/M10/M30 use explicit support and query boundaries;
- static T4, sealed baseline, activity-only, full CDM, Precision-CDM where
  applicable, and fixed classical comparators appear in one aligned table;
- seed and session uncertainty are shown without pooling nonexchangeable
  surfaces; and
- no pending or failed root is reported as a performance result.

All conditions above are satisfied. In particular:

- every result named in Sections 2.4, 2.7, 2.9, 2.10, 2.11, and 2.12 was
  independently rehashed from its immutable body and canonical sidecar;
- official submissions 581359, 581361, and 581362 are all private `finished`
  jobs with finite prediction payloads and exact terminal receipts;
- the official M2 endpoint limitation for trial-aware CDM is an interface
  impossibility, not an unexecuted valid cell;
- Precision seed expansion and fresh matched Phase-C training have explicit
  evidence-based stop decisions; and
- the only failed CDM root retained in the ledger is labelled as a failed
  preparation attempt and is never used as performance evidence.
