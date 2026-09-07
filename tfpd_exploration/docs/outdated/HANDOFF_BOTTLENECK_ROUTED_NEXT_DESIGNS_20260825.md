# Handoff: Performance-First Next Designs After Bottleneck Review

Date: 2026-08-28
Status: CDM-D V8 completed a clean 12-cell matched score and passed the predeclared M4/M10 performance gates; M30 is negative; the Precision V2 successor completed a clean four-cell M10/M4 matched screen and improved external transfer over both sealed Cell D and CDM-D V8, especially at M4, while sacrificing some M4 within performance; CS-WG M1 V3 completed and independently passed the physical CPU/source-only common-stratum audit; the single V4 GPU smoke completed all 100 optimizer steps but failed closed while validating the returned smoke evidence; the V5 typed numerical successor passed no-data review but its one authorized GPU1 smoke failed closed during source preparation because its provider passed the smoke spec into the audit-only V3 builder; V5 never constructed the model or initialized CUDA; additive V6 restored the accepted audit-spec-to-smoke rebind and its sole GPU1 source-only smoke completed a clean 100-step terminal with full 31/31 gradient coverage and strict best/last checkpoint reload; its additive no-SWA fold-20120924 full successor completed all 20 epochs and 99,020 source updates, and its metric-only held-in replay scored variance-weighted last-bin R2 `0.5679166316986084` on 54,849 windows; the exact same-fold matched-ERM control also completed 20 epochs and scored `0.5707439184188843` on the identical input authority, so paired CS-WG minus ERM is `-0.0028272867202759` and CS-WG expansion stops; a subsequent frozen-weight, label-free activity-headroom experiment produced a positive development signal on the matched-ERM M1 fold (`+0.013535` causal rolling; `+0.022569` causal growing) and H1 H-C fold0 (`+0.032292` governing equal-recording for causal growing), but the three available official M1 checkpoints do not support a broad causal gain: rolling is negative on all three folds and growing averages only `+0.000979`; M1 activity-memory training is therefore not authorized by the breadth result, while H1 remains a single-fold variable-cardinality candidate
Authorization: **V2 SOURCE GATE IS HISTORICAL EVIDENCE; V3 SMOKE FAILED BEFORE DATA; V4 M10 SMOKE PASSED; V4 STRICT-27 GATE FAILED CLOSED; V5 STRICT-27 SOURCE GATE PASSED; V5/V6/V7 SCORE ROOTS ARE IMMUTABLE FAILURE EVIDENCE; V8 SCORE TERMINAL IS ACCEPTED; PRECISION MATCHED-SCORE V2 TERMINAL IS ACCEPTED AS A NON-FORMAL PERFORMANCE SCREEN; CS-WG SOURCE-AUDIT V1/V2 ARE IMMUTABLE FAILURE EVIDENCE; CS-WG SOURCE-AUDIT V3 TERMINAL IS ACCEPTED; CS-WG V4 GPU SMOKE IS IMMUTABLE POST-100-STEP FAILURE EVIDENCE AND MUST NOT BE RETRIED; CS-WG V5 GPU SMOKE IS IMMUTABLE PRE-CUDA SOURCE-PREPARATION FAILURE EVIDENCE AND MUST NOT BE RETRIED; CS-WG V6 SOURCE-ONLY SMOKE TERMINAL IS ACCEPTED; CS-WG FULL V1 NO-SWA FOLD-20120924 TERMINAL IS ACCEPTED; THE SAME-FOLD MATCHED-ERM FULL AND HELD-IN SCORE TERMINALS ARE ACCEPTED; THE STRICT PAIRED CELL IS NEGATIVE FOR CS-WG AND DOES NOT AUTHORIZE MORE CS-WG M1/H1 FOLDS; CDM PRECISION TRANSITION V1 IS NO-GO; PRECISION TRANSITION V2 IS STATICALLY ACCEPTED; PRECISION MATCHED-SCORE V1 IS IMMUTABLE PRE-FORWARD FAILURE EVIDENCE AND MUST NOT BE RETRIED**

H1 authorization update: the variable-cardinality candidate named in the
status line has now completed training and matched scoring. It is accepted as
negative evidence with verdict `STOP_VARIABLE_ACTIVITY_SUCCESSOR`; more epochs,
seeds, or a retry of the same recipe are not authorized. This update
supersedes the earlier status clause that described H1 as a future candidate.

H1 breadth update: all five original date-LODO H-C checkpoint tensors were
recovered byte-for-byte from the 5070Ti host and matched their immutable
terminal receipts. The frozen causal-growing activity system improves 5/5
dates and 11/11 recordings, with equal-date mean gain `+0.048937` and exact
date-bootstrap 95% interval `[+0.032665, +0.073602]`. This is accepted as a
positive H1 internal-protocol result and supersedes the earlier single-fold
breadth limitation. It does not rehabilitate the negative variable-exposure
training recipe.

This revision supersedes the earlier bytes of this file.

## Paper-facing result ledger — 2026-08-28

This section is the compact authoritative ledger for deciding what belongs in
the paper. It separates positive governing performance, budget-specific safety
failures, and informative negative routes. Detailed lifecycle and receipt
history remains below. Numbers from different scoring surfaces are never
treated as paired comparisons.

### A. Primary positive result: precision-gated causal dual memory

The strongest new-design result is Precision V2 on the short-label M2/SUA
protocol. It keeps the Cell-D decoder sealed, uses no target gradient or new
target labels, advances causal activity memory from completed trials, and
uses frozen-support precision to gate carrier-state transitions. The following
cells all use paired per-session last-bin, variance-weighted R2 on the same
within-6 or zero-shot cross-subject external-15 input authority:

| Budget / surface | Sealed Cell D | CDM-D V8 | Precision V2 | Precision minus sealed | Precision minus V8 | Positive vs sealed | Paired bootstrap 95% CI vs sealed |
|---|---:|---:|---:|---:|---:|---:|---:|
| M10 within | 0.467718 | 0.530380 | **0.531656** | **+0.063938** | +0.001276 | 4/6 | [+0.00842, +0.11407] |
| M10 external | 0.295456 | 0.340083 | **0.351002** | **+0.055546** | +0.010919 | 12/15 | [-0.02120, +0.11982] |
| M4 within | 0.308920 | **0.493972** | 0.468633 | **+0.159713** | -0.025340 | 6/6 | [+0.10188, +0.21211] |
| M4 external | 0.119684 | 0.191126 | **0.222630** | **+0.102946** | **+0.031504** | 14/15 | **[+0.06319, +0.14393]** |

The paper-primary cell is M4 external: absolute R2 improves from `0.119684`
to `0.222630`, a paired gain of `+0.102946`; 14/15 sessions improve and the
bootstrap interval is fully above zero. M4 within is also strong, but
Precision V2 gives back `0.025340` relative to CDM-D V8 there. M10 external is
directionally broad but its interval crosses zero. Therefore the defensible
claim is short-budget cross-subject improvement led by M4, not universal
dominance on every surface.

Precision V2 completed as a matched non-formal performance screen with
terminal verdict `SCREEN_COMPLETE_NO_FORMAL_VERDICT`. Its accepted body SHAs
are:

```text
official_preflight  6c08320054b24ea9281ab521719462c7a03ca023d8e214730f5b78a723b56536
root_authorization  e6bb2ea84806867ba95622ce6265f281e33df9e87c5f5056acd4ea242d81ce80
attempt             71897d2026b11c7e7ae18b760c189e8eda9c255094d46350f459b84e201e9061
input_authority     5cbd7376f3fed117de02ce532be20ef09cb314972ce02c2e9e05415b851278df
score               4e06ffc544210fbb7cba68c21fd86831c7d0a0407d43ef379a33b9a05bb2bcb1
terminal            dfff5ebb6dd1cbab6e40547d86e6db25715e7a34262054e0406dd144eb8e441c
```

### B. First positive system result: CDM-D V8

CDM-D V8 established that the causal dual-memory mechanism itself is useful at
short budgets before the precision-gate repair:

| Budget / surface | Sealed Cell D | CDM-D V8 | Paired delta | Positive sessions | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| M30 within | 0.566518 | 0.558798 | -0.007720 | 1/6 | [-0.021873, +0.005657] |
| M30 external | 0.428593 | 0.369501 | **-0.059092** | 3/15 | **[-0.130382, -0.010437]** |
| M10 within | 0.467718 | 0.530380 | **+0.062662** | 4/6 | **[+0.004171, +0.117576]** |
| M10 external | 0.295456 | 0.340083 | **+0.044627** | 12/15 | [-0.068487, +0.127681] |
| M4 within | 0.308920 | 0.493972 | **+0.185053** | 6/6 | **[+0.125384, +0.237386]** |
| M4 external | 0.119684 | 0.191126 | **+0.071442** | 13/15 | [-0.067941, +0.180296] |

This is a real M4/M10 win and an equally real M30 safety failure. The accepted
V8 receipt reports carrier-update acceptance rates of `49.26%` at M30,
`31.45%` at M10, and `32.24%` at M4. In the audited session
`sub-M_ses-CO-20140307`, M30 accepted 33 carrier updates while M4 accepted only
9. The uncertainty-blind gate therefore updated the already reliable M30
carrier most aggressively. The deployment rule is consequently explicit:
M30 carrier memory stays frozen and falls back to sealed Cell D; no paper claim
may average the short-budget gains together with M30.

The protocol references help size the remaining opportunity but are not exact
causal ceilings. CDM-D external M4 `0.191126` approximately reaches the prior
M30-activity/M4-label-limited reference `0.1902`; Precision V2 reaches
`0.222630`. Precision V2 external M10 reaches `0.351002`, still `0.019298`
below the M30-activity/M10-label-limited reference `0.3703`.

### C. M1/H1: positive frozen-weight headroom, but learned successors remain null

The earlier conclusion that M1/H1 had no positive new-design result is now
superseded on the two development surfaces below. A four-arm frozen-weight
experiment isolated the activity-identity path while leaving the decoder and,
for H1, the analytic H-C carrier unchanged:

- `STATIC_SUPPORT`: the accepted support-only activity state;
- `ROLLING_FIXED_M`: a causal FIFO with the original support cardinality;
- `CAUSAL_GROWING_CAP30`: causal accumulation up to 30 trials;
- `FULL_SESSION_ORACLE`: all-session activity, label-free but noncausal.

No arm used target labels, target gradients, backward calls, optimizer steps,
or parameter updates. The full-session arm is an oracle diagnostic and not a
deployable result. The growing arm is causal but outside the training-time
activity-cardinality support, so it is evidence for a variable-cardinality
successor rather than a finished deployment claim.

#### M1 fold-20120924, frozen matched-ERM decoder

All four arms use the identical 54,849 valid windows and the accepted
variance-weighted last-bin metric:

| Activity arm | Activity cardinality | R2 | Delta vs static | Fraction of full-oracle headroom | Interpretation |
|---|---:|---:|---:|---:|---|
| Static M10 support | 10 | 0.5707439184 | — | — | accepted matched-ERM replay |
| Causal rolling M10 | 10 | **0.5842786431** | **+0.0135347247** | 46.40% | deployable/cardinality-matched |
| Causal growing M10→M30 | 10–30 | **0.5933129191** | **+0.0225690007** | 77.37% | causal, cardinality-OOD |
| Full-session activity oracle | 414 | **0.5999125242** | **+0.0291686058** | 100% | label-free, noncausal upper diagnostic |

The accepted static prediction digest is reproduced exactly by the eager
path. The cached state-once path differs by at most
`1.6689300537109375e-06` in prediction and `1.1920928955078125e-07` in R2,
inside the frozen parity tolerances. Model state before/after is identical.

This result says the M1 migration gap is not wholly decoder-side: roughly
`0.0292` R2 on this fold is attributable to target activity identity, and a
strictly causal, cardinality-matched FIFO already recovers `0.0135` without
training. It does **not** explain the entire source-to-target gap and does not
yet establish cross-fold breadth.

That breadth check has now been run on all three available official M1
source-only checkpoints. Each row is internally paired to its own static arm;
these checkpoints are distinct from the matched-ERM checkpoint above, so the
absolute levels are not pooled with that cell:

| Official outer fold / target | Static M10 | Rolling M10 | Delta | Growing M10→M30 | Delta | Full-session oracle | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold0 / 20120924 | 0.6503151655 | 0.6423621178 | -0.0079530478 | 0.6560776234 | +0.0057624578 | 0.6641705632 | +0.0138553977 |
| fold1 / 20120926 | 0.6865432262 | 0.6632360220 | -0.0233072042 | 0.6829933524 | -0.0035498738 | 0.6932289600 | +0.0066857338 |
| fold2 / 20120927 | 0.6803454757 | 0.6581065655 | -0.0222389102 | 0.6810708046 | +0.0007253289 | 0.6950112581 | +0.0146657825 |
| Equal-fold mean | 0.6724012891 | 0.6545682351 | **-0.0178330541** | 0.6733805935 | **+0.0009793043** | 0.6841369271 | **+0.0117356380** |

The noncausal oracle is positive on 3/3 folds, confirming a small activity
identity ceiling. The causal rolling arm is negative on 0/3 positive folds,
and causal growing is positive on only 2/3 with an equal-fold gain below
`0.001`. Therefore the strong matched-ERM fold0 development signal is
checkpoint/fold-sensitive and does **not** authorize an M1 GPU training job.
This is exactly why the frozen-weight breadth test precedes training.

Immutable official-checkpoint breadth result SHAs:

```text
fold0  4bc0eb040de23c0ffe5ee934ee1e3c817f82674cda23cad80bb89bfe9b9df085
fold1  a090b9d5ab2cb7991223c9c24bfe4502baf3dd928f125239b8652625e64844fc
fold2  dedf8b4523069247c66b1523c210bd09516ca09ba522f268ef8570e420f2a724
```

#### H1 fold0 H-C, frozen carrier and decoder

H1 is governed by equal-recording weighting because the two recordings have
6,735 and 2,230 valid windows. Pooled values are retained only as a secondary
view:

| Activity arm | Equal-recording R2 | Delta | Pooled R2 | Pooled delta | Per-recording deltas | Interpretation |
|---|---:|---:|---:|---:|---|---|
| Static M4 support | 0.4564882149 | — | 0.5255107801 | — | — | accepted H-C replay |
| Causal rolling M4 | 0.4706254846 | +0.0141372697 | 0.5229039090 | -0.0026068711 | -0.020327 / +0.048602 | mixed sign; not a breadth claim |
| Causal growing M4→available | **0.4887798017** | **+0.0322915868** | **0.5531940374** | **+0.0276832574** | +0.022828 / +0.041755 | both recordings positive |
| Full-session activity oracle | **0.5116554512** | **+0.0551672364** | **0.5681889353** | **+0.0426781552** | +0.029496 / +0.080839 | label-free, noncausal upper diagnostic |

The growing arm recovers 58.53% of the full equal-recording activity
headroom. H-C carrier tensors remain exact in all arms. Static cached-path
parity is `1.4901161193847656e-08` maximum prediction error and
`7.793745648854156e-10` absolute R2 error. Model state is immutable and target
updates are zero.

The H1 result conditionally justified one variable-cardinality
activity-exposure successor. It did not reopen the q=4 carrier-refinement
route: the carrier stayed frozen and the diagnostic gain came from activity
identity. The rolling M4 result is not stable enough to claim independently
because its two recording deltas have opposite signs.

That successor has now completed. It warm-started the sealed fold0 H-C model,
used only the eleven non-target source recordings, and replayed a deterministic
50/50 mixture of M4 and variable M5-through-available activity prefixes while
keeping the analytic H-C carrier fixed. All 4,555 source steps had finite,
nonzero gradients. Mean source loss decreased from `2.729533e-6` in epoch 0
to `2.346027e-6` in epoch 4. Matched target evaluation used the identical two
recordings, 8,965 windows, targets, four arms, and equal-recording metric as
the sealed diagnostic:

| Activity arm | Sealed H-C R2 | Variable-exposure R2 | Paired delta | Per-recording delta | Decision role |
|---|---:|---:|---:|---|---|
| Static M4 support | 0.4564882149 | 0.4556959065 | -0.0007923084 | -0.001967 / +0.000383 | static safety passes |
| Causal rolling M4 | 0.4706254846 | 0.4578327135 | -0.0127927711 | -0.011156 / -0.014430 | worse on both recordings |
| Causal growing M4→available | 0.4887798017 | 0.4833430372 | **-0.0054367645** | -0.007615 / -0.003259 | governing improvement gate fails |
| Full-session activity oracle | 0.5116554512 | 0.5018032129 | -0.0098522384 | -0.006993 / -0.012712 | diagnostic only; also lower |

The successor therefore learned to preserve the static model but did not
learn to exploit additional activity better than sealed H-C. Its own growing
arm remains above its own static arm by `+0.027647`, but the sealed model's
matched gain was larger (`+0.032292`). More epochs or seed replication do not
address this mechanism failure and are not authorized.

The five original date-LODO H-C checkpoint tensors were subsequently recovered
from the 5070Ti host. Every checkpoint/config SHA matches its original
immutable terminal-evaluation receipt, so no model was retrained or selected
using target performance. Replaying the same four frozen activity arms gives:

| Target date | Static M4 | Rolling M4 | Delta | Causal growing | Delta | Full oracle | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.545514 | 0.578829 | +0.033315 | 0.585001 | **+0.039487** | 0.614536 | +0.069021 |
| 19250113 | 0.339962 | 0.354775 | +0.014813 | 0.375285 | **+0.035324** | 0.405708 | +0.065746 |
| 19250115 | 0.521320 | 0.518955 | -0.002365 | 0.548556 | **+0.027236** | 0.566421 | +0.045100 |
| 19250119 | 0.317487 | 0.359273 | +0.041786 | 0.363780 | **+0.046293** | 0.404250 | +0.086763 |
| 19250120 | 0.349828 | 0.432503 | +0.082675 | 0.446173 | **+0.096346** | 0.483661 | +0.133833 |
| Equal-date mean | 0.414822 | 0.448867 | **+0.034045** | **0.463759** | **+0.048937** | 0.494915 | +0.080093 |

Causal growing is positive on all 5 dates and all 11 recordings. Its minimum
recording-level gain is `+0.021306`; the exact five-date bootstrap interval for
the equal-date paired mean is `[+0.032665, +0.073602]`. Rolling is positive on
4/5 dates and 10/11 recordings, with equal-date gain `+0.034045`. The full
oracle remains positive on 5/5 dates but is noncausal.

The original H-S checkpoints were then replayed on the identical windows to
separate the activity-memory effect from the H-C carrier system:

| Target date | H-S static | H-S growing | H-S gain | H-C static | H-C growing | H-C gain | Growing H-C minus H-S |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.447211 | 0.506159 | +0.058948 | 0.545514 | 0.585001 | +0.039487 | +0.078842 |
| 19250113 | 0.257016 | 0.302111 | +0.045095 | 0.339962 | 0.375285 | +0.035324 | +0.073175 |
| 19250115 | 0.439425 | 0.468149 | +0.028724 | 0.521320 | 0.548556 | +0.027236 | +0.080407 |
| 19250119 | 0.294756 | 0.340699 | +0.045943 | 0.317487 | 0.363780 | +0.046293 | +0.023081 |
| 19250120 | 0.375374 | 0.473520 | +0.098146 | 0.349828 | 0.446173 | +0.096346 | -0.027347 |
| Equal-date mean | 0.362756 | 0.418128 | **+0.055371** | 0.414822 | 0.463759 | **+0.048937** | **+0.045632** |

Both systems benefit on all 5/5 dates. The exact five-date bootstrap intervals
are `[+0.038213,+0.077265]` for the H-S activity gain and
`[+0.032665,+0.073602]` for the H-C activity gain. The correct carrier-by-
activity interaction is the difference in differences, not the final growing
system gap:

```text
(H-C growing - H-C static) - (H-S growing - H-S static)
= +0.048937 - +0.055371
= -0.006434
exact five-date bootstrap 95% interval [-0.013647, -0.000815]
```

Therefore there is no positive carrier-by-activity synergy. Activity-state
accumulation is a general H1 mechanism and its incremental gain is slightly
larger for H-S. H-C nevertheless retains the higher final equal-date level by
`+0.045632` (4/5 dates; exact bootstrap `[+0.003976,+0.078335]`) because its
static baseline starts `+0.052066` higher. The practical frozen system remains
H-C plus causal growing activity, but the scientific explanation is additive
activity memory on top of a stronger H-C baseline, not a synergistic
interaction. The immutable V1 raw matrix remains valid; a V2 interpretation
receipt explicitly supersedes only its first descriptive verdict label.

A final low-cost causal memory-policy test checked whether the remaining
growing-to-full-oracle gap could be recovered by replacing the early-history
memory with either recent FIFO30 or uniformly representative history. It could
not: every recording has only 8--15 trials, below the cap of 30. Consequently
the existing causal-growing arm already contains every completed past trial.
FIFO30, coverage-cap30, and causal-all-past produced the exact same prediction
SHA and R2 as causal growing on every date:

| New causal policy minus existing growing | 19250108 | 19250113 | 19250115 | 19250119 | 19250120 | Equal-date |
|---|---:|---:|---:|---:|---:|---:|
| FIFO30 | 0 | 0 | 0 | 0 | 0 | 0 |
| representative coverage cap30 | 0 | 0 | 0 | 0 | 0 | 0 |
| all completed past trials | 0 | 0 | 0 | 0 | 0 | 0 |

This closes the memory-eviction branch. The remaining full-session-oracle gap
is future-information/noncausality, not a suboptimal bounded-memory policy.
Any next H1 learner must predict or distill the eventual full-session identity
from a short causal prefix; increasing the FIFO capacity or changing its
selection rule cannot help on this dataset.

This result changes the H1 conclusion. Sealed H-C plus causal growing activity
memory is itself a useful new H1 internal-protocol system: it needs no target
labels, gradients, parameter updates, new checkpoint, or carrier change. The
failed source fine-tune shows that retraining for variable cardinality is
unnecessary and slightly harmful; the original sealed H-C model should be
used directly with causal activity-state accumulation.

Immutable result bodies:

```text
M1 result:
tfpd_exploration/results/m1_h1_activity_headroom_v1/m1_fold20120924.json
SHA256 5ea74d3131f0b3bfc4b757ca29748285e48b4978cfdd7b07454f380ea48670d8

H1 result:
tfpd_exploration/results/m1_h1_activity_headroom_v1/h1_fold0_hc.json
SHA256 5ec15848efffd3d0d7d1f6d0cbc077c99dcbdc32787965de3ba36472b50990a9

H1 variable-exposure training receipt:
tfpd_exploration/results/h1_variable_activity_exposure_v1/full.json
SHA256 5cd23406dff6781e0105b60ccc778f349d0e494ce7f3a9a7f791828df381059f

H1 variable-exposure checkpoint:
tfpd_exploration/results/h1_variable_activity_exposure_v1/checkpoint.pt
SHA256 69f642f6aeba78a2c136316b78338dec4d8f31968a6071d860c1dc5935c6e77a

H1 matched score:
tfpd_exploration/results/h1_variable_activity_exposure_v1/score.json
SHA256 1a0f4357cf348acdd90a723d590bec4ac7cc766031f2a24e939613379261af70

H1 five-date frozen activity breadth:
tfpd_exploration/results/h1_date_lodo_activity_headroom_v1.json
SHA256 65c9bb40ad45ab7b74740da88fd8081504b7656e807e76b6eb9db903450adb68

H1 H-S/H-C four-cell raw comparison:
tfpd_exploration/results/h1_date_lodo_activity_system_compare_v1.json
SHA256 0dc4576b92b5df383252d164b29d869ca0888a431756338b4938d0269e29a862

H1 corrected difference-in-differences interpretation:
tfpd_exploration/results/h1_date_lodo_activity_system_compare_v2.json
SHA256 2b51e9a64e66655ac74ccd334b6510ee409c596637918cd111a212e62d6f75c1

H1 causal representative-memory policy result:
tfpd_exploration/results/h1_causal_representative_activity_v1.json
SHA256 cc542f375014144c91e00dab1a47a24a1d6660f80d2b3667f4ac9755fb4abf7f
```

Both bodies and canonical sidecars are regular immutable mode `0444` files.

#### Learned architecture result retained as a negative control

The completed M1 CS-WG experiment is a strict same-fold null/slightly negative
result, not an unscored training run:

| M1 fold-20120924 system | Identical valid windows | Variance-weighted last-bin R2 |
|---|---:|---:|
| CS-WG | 54,849 | 0.5679166316986084 |
| matched ERM | 54,849 | 0.5707439184188843 |
| CS-WG minus matched ERM | exact paired cell | **-0.0028272867202759** |

The paired result is about `-0.495%` relative to ERM. It does not authorize
more CS-WG M1 folds or an H1 port. These absolute values must not be compared
directly with earlier official M1 values from different session/window/scoring
surfaces.

Other architecture or uncertainty-consumption routes provide useful boundary
evidence, not positive paper-main results:

| Route / surface | New route | Comparator | Delta / interpretation |
|---|---:|---:|---|
| TF-SR external governing | 0.2542 | Cell D 0.4179 | **-0.1638**, 1/15 positive |
| TF-SR within governing | 0.5919 | Cell D 0.5697 | **+0.0222**, 4/6 positive; capacity bought within fit, not transfer |
| PMC-D external M30 | 0.0490 | Cell D 0.4179 | **-0.3689**, 0/15 positive |
| PMC-D external M4 | 0.0061 | Cell D 0.1147 | **-0.1086**, 7/15 positive |
| H1 q=4 query-label carrier diagnostic | 0.522652 | support carrier 0.525511 | **-0.002859**; closes only this estimator/consumer |

PIRG remains a near-zero qualitative result, but no canonical exact score
receipt is present in this repository; this ledger deliberately does not
invent or round an unavailable numeric value. Likewise, M1 q8 `0.477283` and
raw-head `0.6702` are retained only as non-comparable diagnostics because they
use different sessions, windows, and label exposure.

### D. Paper claims and remaining evidence boundary

The supported new-design claim is:

> Precision-gated causal dual memory improves short-label cross-subject
> transfer without target gradients or additional target labels, with the
> strongest and statistically cleanest gain at M4; reliable long-prefix
> carriers should remain frozen.

The new M1/H1 development-surface claim is narrower:

> Frozen-weight target activity identity has small, fold-sensitive headroom on
> M1 but broad headroom on H1. Across five H1 dates and eleven recordings,
> sealed H-C plus causal growing activity memory improves every recording by
> an equal-date mean `+0.048937` without target labels, gradients, parameter
> updates, or carrier changes. Source variable-cardinality fine-tuning is not
> required and is negative evidence relative to the sealed H-C system.

The paper should lead with M4 external `+0.102946`, accompany it with M10
external `+0.055546` and the within results, and disclose the M30 V8 failure
`-0.059092`. Before treating the method as a formal benchmark contribution,
the non-formal Precision V2 screen still needs a frozen formal evaluation and
checkpoint-seed replication, together with the final comparison table against
original SPINT, sealed Cell D, direct ridge/fixed-ridge carrier baselines,
CDM-D V8, and Precision V2. For M1/H1, the activity-headroom result must be
expanded across the available frozen folds/dates before it becomes a breadth
claim. That check is complete for the three available official M1 folds and
is negative for causal deployment, so M1 variable-cardinality training is not
authorized. H1 now has a positive five-date, eleven-recording frozen-system
result: sealed H-C plus causal growing activity memory gives `+0.048937`
equal-date R2 and is positive on every recording. The learned variable-
exposure successor remains negative and should not be retried. CS-WG, TF-SR,
posterior-consumer, PIRG, and PMC-D remain negative or
non-supporting routes and should not be presented as positive solutions for
those datasets.

## Implementation status — 2026-08-25

The performance-first ordering is authoritative: run the complete CDM-D and
CS-WG systems before any attribution cells. Exact activity-only,
carrier-only, balanced-mean-only, or penalty-shape ablations are permitted only
after a complete system produces a credible positive governing signal.

CDM-D Stage 0 has passed independent no-data review:

```text
Stage-0 workorder SHA256:
5d4a22bf8d1700b4230f2f9970c9ff98b2e6a31d0a9bc1bd828e6d844ff1c6fc

Stage-0 closure SHA256:
3ab6d3de931e4630cb9c80b07e25e3b38af4c444f3c937d3d28560b596c29590

focused synthetic/CPU tests: 23 passed
```

The accepted core keeps the physically distinct completed-trial views
separate: B3S receives cubic-interpolated/padded `[100,N]` spike counts; the
carrier receives native rewarded-trial binned counts and computes
`mean(counts)/0.020`; velocity validity comes only from W=50 neural-window
availability and trial bounds. The three views bind the same session, trial,
and channel order but are never treated as one array or one mask.

The B3S stack is capped at its 30-trial training contract. Completed-query
activity capacities are M4=26, M10=20, and M30=0. No source data, checkpoint,
CUDA, GPU, target surface, result root, or score has been opened by Stage 0.
The additive source-adapter/safety-audit scaffold is the current implementation
step; GPU execution remains unauthorized until that scaffold is independently
reviewed.

The source gate must honor the sealed theta-validity mask.  Across strict-27,
`sub-C_ses-CO-20131101` has one undefined-direction unit and
`sub-C_ses-CO-20131220` has two.  These physical channel rows remain in all
axis/provenance evidence with complementary-group assignment `-1`, but they are
not valid `[a,c]` cosine observations.  Every authority-valid unit must be
assigned and reassembled exactly once; invalid units must remain explicitly
unassigned and disclosed rather than causing a session-wide failure.

### Live implementation update — 2026-08-25 18:57 HKT

CDM-D Source Execution V2 passed the complete strict-27 source gate on GPU1.
The immutable terminal is:

```text
root:
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v2

implementation closure:
46d73bc54df26f03ad349f0f98547263718e372aa5ca4817bfcf5259fc87103a

terminal SHA256:
a909d5d5210656b17c73272eecefa540182b384db6ea9f5fc70ae2b59d30bdf3

status: PASS_SOURCE_CONSTRUCTIBLE
M30: 27/27 passing
M10: 27/27 passing
M4:  26/27 passing
```

The root has exactly 88 JSON receipts plus 88 canonical sidecars, all regular
non-symlink mode 0444, with no failure or partial artifact. It records 25,344
model forwards, 2,052 completed trials, 194.95 seconds of measured execution,
and zero optimizer steps, backward calls, parameter updates, or target updates.
Within, external, formal, and target surfaces remained unopened. The lone M4
source failure is `sub-C_ses-CO-20131003`; it remains in every later target
denominator and does not weaken the predeclared breadth threshold.

This result proves source constructibility for the frozen V2 state machine,
not target performance. A later root audit found that V2 advances the B3S
activity FIFO only when the pseudo-label carrier update is accepted. That is
not the system specified below: unlabeled activity must advance after every
valid completed query trial, while the carrier remains trust-gated. Therefore
the first matched-score V1 candidate is NO-GO and the V2 terminal is retained
only as immutable historical evidence.

The independent-transition implementation was accepted at the no-data level
under
`WORKORDER_CDM_D_INDEPENDENT_ACTIVITY_SUCCESSOR_V3_20260825.md`, SHA256
`c53511d170069185bae2a7f639466646917f5a0ef84da47fa73f14e102288478`.
It changes no network, checkpoint, normalizer, support rule, threshold, or
target-gradient boundary. It separates the two online state transitions. Its
focused V3 suite passed 20/20; the combined V1/V2/V3 suite passed every
behavioral test, with only the two historical V1/V2 closure-reconstruction
tests correctly rejecting the new `core.py` bytes.

The one authorized V3 M10 smoke then failed closed during `prepare`, before
opening a source body, checkpoint, or CUDA. Its immutable predecessor graph is:

```text
root:
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v3

attempt SHA256:
f77d2f552d52d03e756422f5c1526babbefe9afb1463e3a5ea0b79db73edef3f

launch SHA256:
6b6e170d6d845734f18fcd8eea9d1a95dac2a851e9b7ab0dae4404fbd9c9c47a

failure SHA256:
667808ded590acbb79f143043638e97e64b20f2aeacc7f719d1a1c1858907ccc
```

The cause was an engineering loader seam: both inherited provider
materialization and SWA loading re-enter V1's frozen closure validator, which
must reject the new V3 core bytes. The narrow additive
`WORKORDER_CDM_D_SOURCE_EXECUTION_V4_LOADER_SUCCESSOR_20260825.md`. V4 injects
the current explicit closure into those two runtime-module seams while keeping
V1's default validator unchanged; it does not change the V3 scientific system.
A successor matched score may bind only a fresh accepted V4 gate terminal.
Component ablations remain deferred.

### V4 M10 smoke acceptance — 2026-08-25 21:50 HKT

Root independently accepted the single authorized V4 M10 source-only smoke:

```text
root:
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v4

implementation closure:
7854b667abfdec7cb004eff64e52c9f662e5eed293d7f0faef9d4fc45a094200

terminal SHA256:
305499831e169670e3ff0988a52ae8d3425960daf250e028c7182e5e3a8c73ca

smoke SHA256:
bef1637ea02b54f6f51ea5c0516b3322eeb2d9e29664702e874bd69591a3c02f

status: SMOKE_COMPLETED
```

The root contains exactly five immutable body/sidecar pairs, all regular
non-symlink mode 0444, with no failure or extra leaf. Root rehashed all five
bodies, every canonical sidecar, all 58 live closure members, and the exact V3
failed predecessor graph.

The smoke used `sub-C_ses-CO-20131003`, M10 support positions 0--9, and
completed-query positions 10 and 11. Both carrier updates were rejected by the
unchanged safety gates, while both valid B3S activity transitions committed.
The activity digest changed and the carrier digest remained exact on each
trace row, with the two rows forming one continuous parent/child state chain.
This is the direct physical evidence that V3's scientific repair is active and
that V4's loader seam reaches it.

The run was source-only on GPU1, used the exact sealed Cell-D checkpoint and
normalizers, kept both TF32 flags false, made no backward/optimizer/parameter
or target update, and opened no within/external/formal/target surface. GPU1 was
released after terminal publication. The strict-27 V4 source gate is therefore
the next step that was attempted. Matched scoring remains blocked until a
successor gate has a fresh, complete, independently accepted terminal.

### V4 strict-27 failure and V5 repair — 2026-08-25 22:35 HKT

The one authorized V4 strict-27 gate failed closed after source-authority
publication, on the first M30 row:

```text
root:
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v4

attempt SHA256:
8adff90f9686c6aaa9e1ba7306fee3e417009e0e74b6e09ca82182031cdb8da9

launch SHA256:
45de7f3d0df03f33aef240fb8214559c434108cb0d0f2f9f96047d323d8c03a2

source-authority SHA256:
7264c4c5a57f7edc8fecd1f7144f4603cdb4f92a745fc8cfccf46afbcb5e8324

failure SHA256:
7105726660dbc481d791df7a2e63fa8870c8dd66dc28be02f841dbb3fa05720e

stage: budget_m30
error: V3 finalized row type drift
```

Root and Luna independently verified the exact four-pair 0444 graph, canonical
sidecars, absence of terminal/budget evidence, process exit, and GPU1 release.
The failure recorded eight source model forwards and zero backward, optimizer,
parameter, or target updates. No within/external/formal/target surface opened.

The cause is deterministic and above the scientific mechanism. The executor
correctly returns `FinalizedFourGroupPseudo`. The V1 base backend then performs
the source-only truth audit and intentionally converts it to
`GroupedPseudoAuditRow`. V3 called that base method and incorrectly asserted
that the already-converted return was still `FinalizedFourGroupPseudo`. The
accepted M10 smoke called the executor directly, which is why it correctly
proved the independent activity transition while the full-gate path failed.

The current build/test step is the additive
`WORKORDER_CDM_D_SOURCE_EXECUTION_V5_FINALIZED_ROW_SUCCESSOR_20260825.md`,
SHA256
`503fa0276b3bb32fd31b9da51dca961d1214dfef0a0d7c40cd5e8bec3e87cb9b`.
V5 changes only trace placement: a route-local executor retains the exact raw
event once, the backend invokes the existing V1 audit join once, then consumes
the raw event to record V3 trace evidence and returns the grouped row unchanged.
It adds no forward, truth join, model change, state transition, label, or
threshold. A full-boundary no-data test is mandatory before any new GPU gate.

Root independently accepted the frozen V5 candidate before live execution:

```text
V5 source SHA256:
3f51457a011d818f09bc176e5934339b0ddf9573f528b892270a3f77020f3319

V5 physical SHA256:
22f1b73ab1ab4ea55718a7f8d9c6ca930b21fc7a38dc68f9a70d2e14763fd1b7

V5 CLI SHA256:
d2b688c349c5eddbda1f0464ac368dee91de2b7464783df8287833c2b074aa7a

V5 test SHA256:
937360d2eefc694328ee168e2c567712caf29fd861a5d885a04efc39dd146f09

V5 closure, 63 literal paths:
c51490115ee15509376174855da045e769547a0f0ce9cf1267cd31c7d93e2d0f

V5 identity:
b6a8deca27262997fbb88cf7e08ce5b18f25e94ff24ca5c3234d3cc6d3e9ef1b

isolated V3+V4+V5 tests: 58 passed
```

The final tests directly cover never-captured, duplicate, stale,
identity-mismatched, forged, and twice-consumed raw events. A shared ordered
log covers all 81 synthetic strict-roster budget rows and proves each row is
exactly `raw executor -> capture -> truth join -> one-shot consume`. Root also
recomputed both held V4 graphs, every live closure leaf, the V5 identity and
ephemeral capability, and fresh V5 root before issuing one gate authorization.

### V5 strict-27 source-gate acceptance — 2026-08-26 00:04 HKT

The single authorized V5 strict-27 run completed successfully on GPU1. Root
and the independent watcher accepted the immutable terminal graph:

```text
root:
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v5

implementation closure:
c51490115ee15509376174855da045e769547a0f0ce9cf1267cd31c7d93e2d0f

attempt SHA256:
40edd3fb7beeb7a57c125ae3d7b631c5279cac2af732553adb102ee3f899c3d7

launch SHA256:
a3b66d4603d910fb6a93a240256569026a5b809a27365c031345b8bb7f3e1ad8

source-authority SHA256:
d962f7d6035b6db54048f39ad3d5fa55c37d2c829d53a87a8fe55b04923bc3c5

terminal SHA256:
e2e07244e1021137ba806f8f0156c77474b18608e747ef413e38b5c51a1976ff

status: PASS_SOURCE_CONSTRUCTIBLE
M30: 27/27 passing
M10: 27/27 passing
M4:  26/27 passing
```

The root contains exactly 88 JSON bodies and 88 canonical sidecars, all
regular non-symlink mode 0444, with no failure or extra leaf. Root rehashed the
complete topology, rebuilt the current V5 identity and closure, validated all
81 per-session budget receipts, reconstructed each aggregate, and reconstructed
the terminal payload exactly.

The gate processed 2,052 completed trials and 25,344 endpoint chunks in
202.6645 seconds. Median correct-versus-shuffled cosine was
`0.99337/-0.01660` at M30, `0.98448/-0.03520` at M10, and
`0.96390/0.02463` at M4. The only failed cell was M4 for
`sub-C_ses-CO-20131031`, whose correct cosine `0.927662` was below its
shuffled control `0.944735`; the session remains in the target denominator.

The independent-transition trace is physically active. All 1,242 valid M4/M10
completed-query rows committed the activity transition. Carrier updates were
separate: 324 committed and 918 were rejected, with every rejected carrier
digest unchanged. All 810 M30 audit rows left state, activity, and carrier
exactly unchanged, as required by the zero-capacity M30 contract. The run made
zero target, backward, optimizer, or parameter updates and opened no
within/external/formal/target surface. GPU1 was released after publication;
GPU0 and the independent CPU experiments were not touched.

This is constructibility and safety evidence only. It is not an R2 result and
does not establish that CDM-D improves Cell D. The historical matched-score V1
draft remains NO-GO because it binds the V2 gate and calls the coupled state
transition. A narrow V5-terminal-bound scorer was completed and independently
audited against the real 88-body predecessor, the fixed within-6/external-15
metadata, and a 66-path closure. It reuses the reviewed no-cache target parser
and sealed Cell-D scorer while calling `IndependentActivityCausalDualMemory`
with the exact V5 semantics. The scientific matrix remains the common
post-first-30 query pool against sealed Cell D in the predeclared order M30,
then M10, then M4. No attribution arm is admitted before that complete
performance screen.

That V5 implementation is governed by
`WORKORDER_CDM_D_MATCHED_SCORE_V5_20260826.md`, amended SHA256
`d9939bba929391338113018ec770f14cc5d6716ce36dd8ccc38d3baadba7480f`.
It authorizes one backward-compatible typed profile/hook seam in the unlaunched
V1 scorer plus a thin additive V5 package. It forbids lifecycle/evaluator
copying and global monkeypatching. The performance run is a fixed complete
12-cell matrix; negative M30 performance is reported but does not adaptively
hide the primary M10/M4 measurements.

Root accepted the V5 candidate closure
`8573cb21c00adaf86e00a756e3803fd7e78bed5636e6f5775868cd90908f5946`,
minted the immutable target-free authority pair, and launched exactly once on
idle GPU1. That launch passed closure, V5 predecessor, fixed-target metadata,
authority, device-selection, and fresh-root checks. It then failed before
`attempt.json`, target resolution, checkpoint loading, or CUDA because the
physical entry reserved the V5 score root and the shared lifecycle immediately
reasserted that the same root must be absent. The preserved V5 score root is an
empty mode-0755 directory; GPU1 remained idle and no scientific output exists.

The V6 reservation repair was governed by
`WORKORDER_CDM_D_MATCHED_SCORE_V6_RESERVED_ROOT_SUCCESSOR_20260826.md`, SHA256
`a41b3f9e8a6e3c94938b92010034d668f60f22742ce254497359fe850ba0ea09`.
V6 changes only the reservation/lifecycle seam: prospective freshness before
reservation, followed by exact held-root identity/topology/emptiness validation
before attempt publication. It preserves the V5 authority and empty score root
as historical lineage, uses fresh V6 roots, and leaves the complete 12-cell
science contract unchanged.

Root independently accepted the frozen V6 closure
`20db2a1c473fa1b2986e9b78ecf1d9c61343e6a041e576f5e12bf2cc533b9f88`
after reproducing 33 V1/V5/V6 no-CUDA tests and descriptor-reloading the V5
authority, empty-root incident, 88-body source gate, and fixed within-6 /
external-15 metadata. V6 successfully fixed reservation ordering and durably
published `attempt.json`, but its sole physical run then failed closed in
`prepare`: the inherited V1 scorer requested `RuntimeFlags` from the V5 wrapper
module, while that wrapper exposes the closure-bound V1 module as `v1` and does
not define `RuntimeFlags` at top level. The immutable V6 failure graph is:

```text
authority preflight SHA256: 52ee183b91bb236131f90e7316404ba6b79c7ba0b8445f242296280d015e7b85
authority authorization SHA256: dac888d29ee188f187cf6c40085d12ef6bec2401ecf366b797d4ba9909185971
attempt SHA256: 0c20ad8405768781e8e398c2e8663acddd5939f5249171eba8aa6001fc65c4ee
failure SHA256: 45cf3867933b0c1b2bdd55780ea32e106c734696a26d32364c7163fc3e99f07f
stage: prepare
class: AttributeError
```

The failure occurred after the sealed terminal/SWA/baseline material had been
opened, so the honest receipt records `checkpoint_opened=true`. It occurred
before within/external asset opening, CUDA initialization, or any model/group
forward; all target backward/optimizer/update counters remain zero. GPU1 stayed
idle. V6 must not be retried or repopulated.

The active repair is now governed by
`WORKORDER_CDM_D_MATCHED_SCORE_V7_RUNTIME_FLAGS_SUCCESSOR_20260826.md`, SHA256
`1953fe4238c517289e7e8d3a9eaf73f1c63c9fe25100471d843a04a14cc386ef`.
V7 may add only a typed, closure-bound runtime-flags factory seam and fresh V7
authority/result roots. It must preserve the V5/V6 parser, model, independent
activity state, target contract, metric, and full nonadaptive 12-cell matrix.

Root independently accepted V7 after reproducing 45 V1/V5/V6/V7 no-CUDA
tests and rebuilding its 80-path closure as
`0e5010316cbe164fe6060e58bb42aebb8fbe181f54f72c41e7375c78f961c6f0`.
The live V6 failed graph, V5 history/source gate, fixed within-6/external-15
authority, and fresh V7 roots all passed descriptor-safe review. Root minted
the V7 target-free authority pair (`b88ebb23...` / `320b8611...`) and launched
exactly once on idle GPU1. The run durably published its attempt and input
authority, passed the earlier V5 reservation and V6 runtime-flags failure
points, loaded the sealed model, initialized CUDA, opened the fixed within and
external assets, and completed 4,238 full-system forwards. It then failed
closed at `budget_m30` before any group forward: the V1 scorer looked up
`_variable_prefix_array_digest` on the V5 physical wrapper, while the wrapper
closure-binds the actual helper module as `v1_physical` and does not re-export
that private helper. The immutable V7 graph contains attempt, input authority,
and failure only:

```text
attempt SHA256: 16e320f00cd6ea7ebcc32ca4cc45e80da8595e41816f8d3930e8c97b30c91cb7
input authority SHA256: d5e5ececc27b657473181d39626b94c0c200b8c924caf7f074d136d834372777
failure SHA256: 4eff5f7639026ad8f3abf60477248023fe54872953b2b6854c9d47bfc519f4c8
error SHA256: 7a295bf081bb98796cdf730bb1bcc52cd51c4d685c42d2bbf17f2e532ce0d31a
```

No score or terminal was published; target backward/optimizer/update counts
remain zero, and GPU1 was released. V7 must not be retried.

The active V8 repair is governed by
`WORKORDER_CDM_D_MATCHED_SCORE_V8_PHYSICAL_HELPER_SUCCESSOR_20260826.md`,
SHA256
`100277c6a06365e2e852a6863bb3e44a9c18237938c0f27b0952f3f7dc517c38`.
V8 may add only a typed, closure-bound base-helper-module seam. It must route
the inherited digest, normalized-T4, and variable-prefix-forward calls through
the authenticated V1 physical helper module while retaining the exact V5
one-shot executor and unchanged nonadaptive 12-cell science.

Root independently reproduced 68 V1/V5/V6/V7/V8 no-CUDA tests and corrected a
stale pre-finalization closure report. The exact current 87-path closure is
`62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939`.
The V7 authority/failure graph, transitive V6/V5 history, fixed target
metadata, and fresh V8 roots all passed descriptor-safe live review. Root
minted V8 authority (`688df31b...` / `3cbc7f99...`) and launched exactly once
on idle GPU1. The run reached a clean natural terminal with exact four-pair
topology and no failure. Its accepted score/terminal SHAs are:

```text
attempt: 557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376
input authority: ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07
score: 98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e
terminal: 80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096
verdict: ADVANCE_SHORT_BUDGET
```

All 12 cells completed in the fixed order M30 -> M10 -> M4. Every cell used
the same input authority, eval/no-grad/dropout-off inference, finite
predictions, and unchanged model state. Target backward/optimizer/update and
target-label state-use counts are all zero.

| Budget | Surface | Sealed Cell D | CDM-D | Paired delta | Positive sessions | Bootstrap 95% CI |
|---:|---|---:|---:|---:|---:|---:|
| M30 | within-6 | 0.566518 | 0.558798 | -0.007720 | 1/6 | [-0.021873, 0.005657] |
| M30 | external-15 | 0.428593 | 0.369501 | -0.059092 | 3/15 | [-0.130382, -0.010437] |
| M10 | within-6 | 0.467718 | 0.530380 | +0.062662 | 4/6 | [+0.004171, +0.117576] |
| M10 | external-15 | 0.295456 | 0.340083 | +0.044627 | 12/15 | [-0.068487, +0.127681] |
| M4 | within-6 | 0.308920 | 0.493972 | +0.185053 | 6/6 | [+0.125384, +0.237386] |
| M4 | external-15 | 0.119684 | 0.191126 | +0.071442 | 13/15 | [-0.067941, +0.180296] |

This is a real short-budget win, not a universal win. Within M4 is especially
strong and unambiguous. External breadth is strong at M10/M4, but both
external bootstrap intervals include zero because `sub-M_ses-CO-20140626`
is a severe negative outlier at every budget (M30 -0.4656, M10 -0.6320,
M4 -0.7239). M30 is decisively harmful on external transfer. Because M30 has
zero activity-FIFO capacity, its negative result is descriptive evidence that
online carrier adaptation is the main safety risk; it is not yet a causal
attribution claim. Consistent with the performance-first policy, the next
CDM-D build should directly repair that carrier-transition mechanism rather
than lead with a broad bundle of attribution cells.

A post-terminal audit now makes that carrier-safety diagnosis more specific.
The current gate is uncertainty-blind: `CarrierMemory.propose_pseudo_trial`
reconstructs the candidate and accepts it whenever the fixed B8-compatible
departure ratio is at most `2.0`; it has no input for the support design
precision or posterior covariance of the initial carrier. The immutable V8
receipt shows the counterintuitive budget ordering below:

| Budget | Within accepted carrier updates | External accepted carrier updates | Combined acceptance rate |
|---:|---:|---:|---:|
| M30 | 589 / 1,206 | 1,900 / 3,847 | 49.26% |
| M10 | 609 / 1,206 | 980 / 3,847 | 31.45% |
| M4 | 344 / 1,206 | 1,285 / 3,847 | 32.24% |

For the audited example `sub-M_ses-CO-20140307`, this is exactly 33 accepted
M30 updates versus 9 at M4. That example is not universal, but the aggregate
direction is the same. Across external sessions, carrier acceptance rate is
negatively associated with paired delta at every budget (Pearson correlation
approximately -0.31 at M30, -0.10 at M10, and -0.42 at M4). This does not by
itself prove that accepted updates cause each loss--session difficulty is a
confounder--but it is strong enough to preregister the next performance
mechanism rather than launch an open-ended ablation search.

The short-budget scale also sharpens the story. CDM-D external M4 reaches
`0.191126`, essentially the prior M30-activity/M4-label-limited reference
`0.1902`; CDM-D external M10 reaches `0.340083`, still about `0.0302` below its
corresponding `0.3703` activity reference. These are protocol-level reference
points, not exact causal ceilings, because online query activity and a sealed
M30 calibration stack are not identical. They nevertheless support M4 as the
primary result and M10 as the accompanying, still-unsaturated result.

The exact activity-only diagnostic is already specified by
`WORKORDER_CDM_D_ACTIVITY_ONLY_REFINEMENT_V1_20260826.md`, SHA256
`f2bf743d3f31f3619a0573e34b28b0713839bc9fad5608d84ba1eda1cb8ff336`.
It adds no training and only four new M10/M4 cells. The carrier is frozen at
the honest support initializer while valid completed query trials advance the
activity FIFO. It must preserve the V8 input authority and also pass a
predeclared severe-outlier safety gate. It remains queued as a later causal
diagnostic and may not delay either the new CDM performance successor or the
primary M1 route.

No new M30 activity-only cell is required: M30 has literal activity capacity
zero, so freezing its carrier exactly reduces to the already measured sealed
Cell-D M30 comparator and would recover the observed `+0.059092` external loss
by deployment policy, not by another model run. The next new CDM performance
design is therefore an uncertainty-aware carrier gate. It should use only
support-derived closed-form precision (for example the residual-scale times
`(X^T X + lambda I)^{-1}`) to modulate whether a pseudo-labelled carrier
proposal may commit: high-precision initial carriers receive a stricter gate,
while low-precision carriers may admit more evidence. M30 carrier memory is
disabled by deployment rule. Precision is used at the state-transition
boundary, not fed into the decoder, and no target gradient or additional label
is allowed. This is a new performance mechanism, not an attribution arm.
Coverage-aware FIFO eviction and block refitting remain lower-priority options;
activity-only decomposition and seed replication follow after a complete
successor passes the matched M4/M10 gate.

The first no-data implementation of that precision gate is frozen as review
evidence, not accepted for execution. Its workorder SHA256 is
`ced29701e27da700a44e11d8d11370cfb4f3ab8eda66f42d4eec48f32f1f8e36`;
its eight-path closure is
`2fe6491851da7ffa505811f1e8aaa9c77fbd46496b6a93ad4c22877f980cd1d1`;
12 focused synthetic tests pass. It correctly keeps precision out of the
decoder, makes M30 a literal no-proposal/no-state-change path, and preserves
independent activity when the new typed carrier rejection fires. Root audit
nevertheless found three mathematical launch blockers:

1. The implemented `sigma^2 A^-1 X^T X A^-1` is the sampling/sandwich
   covariance of the ridge estimator, not the Gaussian-ridge posterior
   covariance `sigma^2 A^-1` stated by the mechanism. The successor must name
   and justify one estimand exactly instead of calling both posterior.
2. Applying a per-unit chi-square(2) 95% threshold and then requiring the
   maximum over all valid units to pass is not a 95% session gate. Under the
   nominal independent reference its all-unit pass probability is `0.95^N`
   (`0.0059` already at `N=100`), so the current rule can collapse into an
   accidental activity-only system. A source-only multiplicity correction or
   preregistered aggregate statistic is required.
3. The current statistic compares each proposal with the current carrier.
   Repeated individually admissible small steps can therefore ratchet the
   carrier arbitrarily far outside the frozen support confidence region. The
   repaired rule must bind every candidate to the frozen support-only
   initializer, or enforce an equivalent cumulative bound.

No source, checkpoint, target, CUDA, GPU, or result root was accessed by this
candidate or audit. These findings do not invalidate the V8 result or the
M30-carrier-off deployment rule; they only prevent launching this first
precision-gate formulation.

An additive V2 mathematical successor has now passed independent static
review. Its workorder SHA256 is
`8e25cfd8df7868dbaec2596825211eb8724ccaed98023eaba6731028ba85e879`;
its eight-path closure is
`17d966e90c038e8626b82c60b5645d9fb4ce8aeaef58f3b9126966ceefc5f0d0`.
Root reproduced 25 V1+V2 synthetic tests, py-compile, dry-CLI, and two identical
closure reconstructions. V2 resolves the three V1 blockers exactly:

1. it uses the explicitly named conditional Gaussian-ridge posterior
   `sigma^2 A^-1` rather than the sampling sandwich covariance;
2. it replaces the raw per-unit 95% maximum with the Bonferroni family-wise
   threshold `-2 log(0.05/N_valid)`;
3. it measures every candidate against the immutable support-only initial
   fixed-ridge carrier and rejects wrapper reconstruction after the live
   carrier has moved, preventing cumulative reference reset.

V2 remains no-data and grants no execution capability. The next build is a
thin matched-score composition over the accepted independent-activity/V8
input authority: add only V2 M10/M4 candidate cells, reuse and descriptor-
revalidate the immutable V8 sealed comparators, and report M30 directly as the
already measured sealed deployment rule rather than rerunning a no-op. It may
proceed only if that scorer seam is additive and target-gradient-free.

That four-cell scorer was implemented additively and passed independent static
review: 26 combined no-CUDA tests, py-compile, a Torch-free dry CLI, and two
identical 101-path closure reconstructions at
`e296379bf839c8d647a85c51babc0c1449aa5297bab42eeaeec65ea97c30eab6`.
The live held V8/V5 predecessor and same-input witness also passed before
launch. Its first actual attempt nevertheless failed closed at
`materialize_inputs` before any within/external asset or model forward because
the root launch omitted `SUBC_DATA_ROOT` and `SUBM_DATA_ROOT`. The immutable
attempt and failure bodies are respectively
`59469ac2c250fb22f1df0c94c49d17473f8d1553c1d3ea5a81bf074d6be6c849`
and
`435ccf2c300ec4d6180738c89a51737050b768225fcee28bc165d6ae0179e253`.
The receipt honestly records checkpoint/CUDA initialization during prepare,
zero full/group forwards, no target asset opened, and zero target updates.
This is an execution-environment failure, not a Precision performance result.
V1 must not be retried. A narrow additive successor must bind this graph and
require the exact canonical absolute SUBC/SUBM roots before any authority or
score reservation, publication, capability issuance, or attempt.

That environment-gated V2 successor has now passed root's independent static
review. Its workorder SHA256 is
`3080d27b8c505576c019816d4b32c0c55d74b3a7c9a18f8c8c8dd0f023a29516`;
its explicit 108-path closure was reconstructed twice as
`c6c14b1e1aa9f19639195558604ce6fc2a6517573ca66461ac570b8499e75f08`.
Root reproduced all 29 V1+V2 no-CUDA tests, py-compile, a Torch-free dry CLI,
the held V1 authority/failure graph, and the exact absolute environment gate.
All mutating entry points compare both the supplied mapping and the real
process environment before authority or score reservation. The required
values are:

```text
SUBC_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
SUBM_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
```

The single authorized V2 attempt was then launched on GPU1 with identity
`5d56583032dca0a3e6bdda83fc0f579f288a7569174cc965fe23f59876ce52da`.
It completed once without retry. Root and Luna independently validated the
two authority pairs and four score pairs as regular immutable mode-0444 files,
all canonical body/sidecar hashes, the exact V1 failure and V8 predecessor
bindings, the four-cell M10/M4 execution order, zero target updates, and the
absence of any failure or orphan leaf. Route-native canonical validators then
reconstructed the input, every budget summary, and terminal without loading
Torch. The accepted body SHA256 values are:

```text
official_preflight  6c08320054b24ea9281ab521719462c7a03ca023d8e214730f5b78a723b56536
root_authorization  e6bb2ea84806867ba95622ce6265f281e33df9e87c5f5056acd4ea242d81ce80
attempt             71897d2026b11c7e7ae18b760c189e8eda9c255094d46350f459b84e201e9061
input_authority     5cbd7376f3fed117de02ce532be20ef09cb314972ce02c2e9e05415b851278df
score               4e06ffc544210fbb7cba68c21fd86831c7d0a0407d43ef379a33b9a05bb2bcb1
terminal            dfff5ebb6dd1cbab6e40547d86e6db25715e7a34262054e0406dd144eb8e441c
```

The terminal verdict is `SCREEN_COMPLETE_NO_FORMAL_VERDICT`; M30 was reused
only as the immutable deployment reference and was not rerun. The matched
means are:

| Budget / surface | Sealed Cell D | CDM-D V8 | Precision V2 | Precision minus sealed | Precision minus V8 |
|---|---:|---:|---:|---:|---:|
| M10 within | 0.467718 | 0.530380 | 0.531656 | +0.063938 | +0.001276 |
| M10 external | 0.295456 | 0.340083 | 0.351002 | +0.055546 | +0.010919 |
| M4 within | 0.308920 | 0.493972 | 0.468633 | +0.159713 | -0.025340 |
| M4 external | 0.119684 | 0.191126 | 0.222630 | +0.102946 | +0.031504 |

Against sealed Cell D, the paired external result is 12/15 positive at M10
with bootstrap 95% CI `[-0.02120, 0.11982]`, and 14/15 positive at M4 with CI
`[0.06319, 0.14393]`. Within is 4/6 positive at M10 with CI
`[0.00842, 0.11407]`, and 6/6 positive at M4 with CI
`[0.10188, 0.21211]`. Precision carrier acceptance is deliberately selective:
M10 accepts 76.3% within / 65.4% external, while M4 accepts only 18.2% within /
6.4% external. The mechanism therefore improves the priority external short-
budget surface, especially M4, but does not dominate V8 everywhere: M4 within
falls by 0.02534 relative to V8. Keep Precision V2 as a deployment candidate
for cross-subject M4/M10; do not claim a formal verdict or universal
dominance, and do not launch an immediate gate-ablation chain.

The first CS-WG Stage-0 candidate was superseded before any data or training.
Root CPU materialization proved that M1 has two different time axes:

```text
decoder window W: 100
B3S calibration-trial length: 1024
```

Using the erroneous calibration shape `[10,100,64]` materialized only
14,061,320 parameters. The exact `[10,1024,64]` shape materialized the sealed
15,007,496-parameter M1 graph and emitted `[1,100,16]` without CUDA. The
amended CS-WG Stage-0 work order SHA256 is
`225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d`;
the re-frozen 31-path Stage-0 closure is
`dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51`.
Root independently reproduced all 19 no-data tests, including the real CPU
model gate, so Stage 0 is accepted.

The next bounded build is governed by
`WORKORDER_CS_WG_M1_SOURCE_LIFECYCLE_V1_20260826.md`, SHA256
`ca0c3b754c602ec490e4f5b74c5bf85a93764184e6f9a489a10ec4eed892e043`.
It freezes one performance configuration (`lambda=1.0`, `tau=0.01`) rather
than opening a tuning grid, adds no network parameters, and permits only a
future fold-0 100-step source-only smoke after another independent audit. Full
training, target replay, and H1 remain unauthorized.

Terra completed that first no-data lifecycle candidate with 39/39 combined
Stage-0/lifecycle tests and explicit closure
`2f2078bcd89476b84c42d78abedd6b2430d6114bd6550e1ef5e938352e429bdf`.
Root review did not authorize smoke because the physical reader was still an
injected protocol and because its calibration-ownership check contradicted the
native dataset: one M1 session has one deterministic chronological M10 tensor,
which `FalconDataset` returns for every query window from that session. A
per-row unique-owner requirement would either reject the baseline contract or
manufacture false ownership evidence.

The bounded repair is now governed by
`WORKORDER_CS_WG_M1_SOURCE_READER_REPAIR_20260826.md`, SHA256
`e535af20ba42faf7fee2053d709aeb546a01d592e434f2086f7781e3e24fa129`.
It adds no network or science change. It requires a route-owned all-four-fold
reader built directly from the closure-bound native `prepare_session_data`
and `FalconDataset` primitives, exact selected descriptors without target
directory discovery, session-level sealed calibration digests repeated and
checked on every row, clean coexistence of the experiment package and the
historical top-level `src` package, operational seed-42 RNG setup/restoration,
honest failure progress, and finite resource evidence. No source file or GPU
may be opened until the repaired no-data closure is independently accepted.

Root subsequently accepted the repaired reader/lifecycle at the no-data level.
The exact combined Stage-0/lifecycle/reader suite passed 65/65, py-compile and
both static dry CLIs passed, and the 41-path closure was reproduced twice as
`b5d4fdf5505fd53f160d56160daa642e36e8077171f654d9c5cdff6ee8fb7a28`.
The closure binds the sealed metadata-only M1 manifest
`4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6`.
That manifest fixes the four session paths and body hashes before attempt,
while source byte counts, held descriptors, and body verification are deferred
until `prepare_source`. The repaired reader retains one immutable contiguous
M10 calibration backing per session and still materializes every B32 row
explicitly at the route-owned concatenation boundary. GPU smoke remains
hard-disabled until a successor binds an accepted source-audit authority and
terminal.

The first authorized CPU/source-only audit then failed closed because the
reviewer's internal command imported the route as top-level
`src.cross_session_worst_group_v1`. That occupied the historical top-level
`src` namespace needed by `streaming_calibration_exp`, so the reader correctly
refused to replace it. The failure is engineering launch evidence, not a data
or design result: it occurred before parser completion, NWB body open, model or
checkpoint construction, CUDA, or optimizer work. The immutable V1 graph is:

```text
root:
tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1

attempt SHA256:
5d0cd206644d29b4ac81139d9a7cbe4aaedc404935a1029597ef1a0f3f9a1e75

launch SHA256:
895fb18d342a51a91e97e5a89ef13e53d662bd556fc2b0881ca1e7be5a70a99d

failure SHA256:
f5ec355bd26d4bb13f48117e122a2f2b4c5eb3dac2e18cc9b357dfa10554436b

error class: SourceReaderError
error SHA256: 7730762e166a649a38f201a078398d00337352875f4fa2169150fac19c517e65
source authority: absent
CUDA/model/checkpoint/optimizer: untouched
```

An additive source-audit V2 successor must bind and descriptor-validate that
exact failed graph, use a fresh result root, and enter through the unambiguous
`tfpd_exploration.src.cross_session_worst_group_v1` namespace while leaving
top-level `src` free for the historical streaming package. It may not alter the
scientific design or enable GPU smoke. After another no-data review, V2 gets at
most one CPU/source-only attempt; a later GPU-smoke successor still requires a
fresh accepted V2 source-authority and terminal graph.

That V2 namespace successor passed root's independent no-data review: 74/74
combined tests, a reproducible 45-path closure
`ca77c59103830fcb38e6d51af49b355485b6b916a3b4402e26d4b20e21588fa0`,
and live identity
`e9565bce5d0f2e4e683b599f5e0e0e65a089eae4d0308afd7aa2d58170a6b581`.
It descriptor-validated the exact V1 failed graph before capability issuance,
left top-level `src` unoccupied, and started one low-priority CPU/source-only
audit. The qualified namespace repair worked: V2 entered the historical native
parser and source preparation. It then failed closed after source resolution,
before source-authority publication, model/checkpoint construction, CUDA, or
optimizer work:

```text
root:
tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2

attempt SHA256:
163e73fc5c793b009e7ec86f4af7cc1841bec0bbf5865629c7013a3df468350b

launch SHA256:
9af2bb8c400dad927b0d3dbf711d9d24242c0e4bfb55b8163a96e0a5e4ea4fcc

failure SHA256:
945a7b5f843e14a6cb4008309c9df4ab510b5f60dbda4a33ba54e6dc563889cb

error class: SourcePhysicalError
error SHA256: 52dde6c1a3dbbe88f47a5f32e281da20c1b3aeeb32166a14cb1c7fe36108e543
source authority: absent
CUDA/model/checkpoint/optimizer: untouched
```

At this boundary, the remaining dynamic preparation checks are the physical
common-stratum topology gates. Requiring all three 16-output sessions to expose
identical complete task-stratum sets is stricter than the design's actual need:
training needs a deterministic shared set that can supply each 11/11/10
microbatch, not equality of every rare session-specific stratum. The next
additive V3 source-audit successor therefore predeclares one fallback before
seeing its values: retain only strata with at least two rows in every source
session, require at least ten such common strata, and construct each session's
training pool from that exact identical eligible set. This supports any
rotating 11/11/10 step without row duplication while preserving the original
pooled quantiles, task-stratum definition, labels, model, and optimization
recipe. V3 must receipt-bind original/common/missing/eligible sets and retained
coverage. If fewer than ten eligible strata exist, it stops before model/CUDA
with typed topology evidence. GPU smoke remains disabled until this successor
has an independently accepted terminal and source authority.

That V3 successor has now completed one authorized low-priority CPU/source-only
audit and passed independent root verification. The immutable V3 graph contains
exactly the five expected 0444 body/sidecar pairs and no failure:

```text
root:
tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3

attempt SHA256:
f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287

launch SHA256:
04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c

source-authority SHA256:
0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b

audit SHA256:
ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9

terminal SHA256:
2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230

V3 closure SHA256:
d52168e567188b8ede816f4764cf829ecd920b2540c323fee14569ec7503fa1e

terminal status:
PASS_SOURCE_AUDIT_COMMON_STRATUM_CONSTRUCTIBLE
```

The observed source topology is comfortably constructible rather than barely
passing. The original source pools expose 47, 51, and 50 strata; their union is
56, their raw intersection is 44, and 43 strata have at least two rows in every
source session, versus the preregistered minimum of 10. Deterministic pruning
retains 54,467/54,476, 49,189/49,228, and 54,766/54,783 windows respectively
(all above 99.92%). A physical step-zero B32 episode was constructed as
10/11/11 under the rotating quota, used only eligible common strata, contained
no within-session duplicate sample ID, and preserved the single-concatenated
forward requirement. Target, held-out, minival, EvalAI, and formal surfaces
remained unopened; no model, checkpoint, CUDA, or optimizer work occurred.

Root rehashed all 49 closure members, both immutable failed-predecessor graphs,
all five V3 bodies and canonical sidecars, the fallback digest, and every
terminal provenance edge. This accepts the source authority, not model
performance. The next permitted build is an additive 100-step GPU-smoke
successor that binds this exact V3 graph. It should also remove the audit-only
implementation inefficiency that repeatedly hashes the same 2.5 MiB
session-calibration tensor for every row; that optimization must be
receipt-bound and byte-equivalent and may not change any accepted V3 science
row or digest.

### CS-WG V4 GPU-smoke failure and typed successor — 2026-08-26 09:46 HKT

The single authorized V4 source-only smoke ran the complete physical 100-step
optimizer loop on GPU1, then failed closed before publishing a smoke receipt or
checkpoint. GPU1 was released normally and GPU0 was not used. The immutable
V4 graph contains exactly four 0444 body/sidecar pairs and no extra leaf:

```text
root:
tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4

attempt SHA256:
0d4179d77cd8aa66bdad70d1164682dba38b3d960fab3d524cf5f213e4601b27

launch SHA256:
422dec3d014bf3fa0fed716b8634cc6e8da6076f36f0f303f657a388f7b417ad

source-authority SHA256:
40336ed6ea1f6dd2b6511eee4d22f5cf76c4af29510f8c258819bd396c78b4fe

failure SHA256:
1a8237db7ddf34535bebd83cc773da7efde6b5aa97517a09638b391d6eae91d3

optimizer steps completed: 100
target optimizer/backward/update: 0
```

The durable failure recorded `SourceSmokeV4Error` with error SHA256
`bfeff083be483479e40fb82078a3b8b1e5755b0f04ec0f341cc840af892cf813`.
Enumerating the frozen V4 error domain and hashing each exact exception repr
identifies one unique preimage:

```text
SourceSmokeV4Error('CS-WG V4 inherited physical smoke evidence drift')
```

Therefore this is not evidence of an OOM, CUDA failure, source-reader failure,
or failure to optimize. It is a post-run validation failure inside
`_v4_smoke_payload`, where the real runner result is passed to the inherited V1
aggregate smoke validator. The V4 receipt intentionally hid the nested V1
predicate, so the immutable graph cannot establish which individual field
failed. The leading numerical hypothesis is the strict
`session_objective_derivatives_nonnegative >= 0` check: the centered FP32
log-sum-exp objective has analytically nonnegative softmax derivatives, but a
low-weight entry can acquire a tiny negative value through subtractive
cancellation. This remains a hypothesis until reproduced independently; it is
not grounds for weakening the gate after seeing live data.

V4 must not be retried. The next additive successor must bind both the accepted
V3 graph and this exact failed V4 graph, retain the same model/data/optimizer
and cached common-stratum preparation, and replace only the opaque validation
boundary. It must preregister a synthetic FP32 tolerance against an independent
stable float64 softmax reference, preserve raw min/max/sum/digest evidence,
reject any material derivative error, and record the exact failed predicate in
any future failure receipt. No full M1 training is authorized until that
successor completes one clean 100-step smoke terminal.

### CS-WG V5 prelaunch finding, repair, and launch — 2026-08-26

The first frozen V5 candidate implemented the default-`None` read-only
derivative observer and passed 48 isolated V1+V5 no-data/no-CUDA tests. Root
then rebuilt its 51-path successor closure as
`ce1c0f5b1f0f4cca64c1a9d0efefc8d0e00ee0ca9bc3455d852613fed0e910fb`
and descriptor-read both immutable predecessors before issuing a capability.
The real V3 graph failed the V5 semantic validator at `audit.json`: accepted
V3 audit bytes bind the typed identity by `identity_sha256`, whereas the first
V5 validator incorrectly required a duplicate top-level `identity` mapping on
that one body. Attempt, launch, source authority, and terminal do carry the
full identity. This is a launch-code schema mismatch, not a V3 provenance
failure and not a scientific or GPU result.

The check failed before V5 root reservation, source opening, model/checkpoint
loading, CUDA initialization, or capability issue. The canonical V5 root
remained absent. The repair is restricted to the V5 predecessor semantic
validator and an actual-shaped descriptor-only regression; historical
V1/V2/V3/V4 files and result graphs remain immutable. A V5 GPU smoke may be
scheduled only after root independently reruns the real predecessor validator,
current closure, environment, and freshness checks.

The V5-only repair now requires the canonical digest of the full attempt
identity, exact nested identity equality in launch/source-authority/terminal,
and the same digest in `audit.json.identity_sha256`; it also rejects an
invented audit-side identity. Root independently reproduced 49 V1+V5
no-data/no-CUDA tests, rebuilt the repaired 51-path closure as
`2e6ec1e017a6e560789e0a244a2ea19ebef94f5735db2affa3a9a6c829cbd23c`,
reloaded both real held graphs, and issued an in-process capability for V5
identity `851f7b24e4839e00db32e9b554dfa35432e0543d1b4c72589a28a1a664a81d7b`.

One V5 source-only 100-step smoke was then launched on the otherwise idle
GPU1. It terminalized naturally as a fail-closed source-preparation attempt;
Luna stopped monitoring after the event and GPU1 returned to its idle
23 MiB state. GPU0 was not queried or touched by the watcher. The immutable V5
root contains exactly three 0444 body/sidecar pairs and no extra leaf:

```text
root:
tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5

attempt SHA256:
ae6bf550dfff4179d51dbb62616a7fa97d2bfb33d1e7de5ab8c4917096afb9c4

launch SHA256:
03a33580948f9c95dd8f022ea1947ed9f696f551974fe546c5edb03c94d3756f

failure SHA256:
0cbd5c0551fa654bf7d9b2b45dfe334d7023bcfb142087fefd297ed85199ce11
```

The failure receipt records `SourceAuditV3Error`, error SHA256
`840a7a668324571626ba202fd63a3f9590252701be5b91707f2a0b11d2d3d9db`,
and the unique static error preimage:

```text
SourceAuditV3Error('CS-WG V3 source selection/outer-target ordering drift')
```

The receipt is honest: source resolution began, but no source authority was
published; model construction, CUDA initialization, optimizer steps, and all
target operations remained zero. The exact engineering cause is in
`V5CachedCommonStratumSourceProvider.prepare()`: V5 passed
`identity.inherited_v1_smoke_identity.spec` directly to
`v3.build_common_stratum_prepared_audit`, whose contract accepts only an audit
spec. The accepted V4 provider already supplies the correct narrow seam: take
the audit spec from the accepted V3 identity, build the V3 audit preparation,
then call `rebind_v3_audit_prepared_to_smoke` for the smoke identity. V5 lost
that rebind while adding the derivative observer.

V5 must not be retried or modified in place. The next candidate is an additive
V6 that binds this exact six-leaf V5 failed graph, restores only the accepted
V4 audit-spec-to-smoke rebind, and retains the V5 observer, FP32-vs-FP64
derivative evidence, model, data, optimizer, and 100-step contract unchanged.
It needs a complete no-data/root audit before one separately authorized smoke.

That V6 candidate is governed by
`WORKORDER_CS_WG_M1_SOURCE_SMOKE_V6_AUDIT_SPEC_SUCCESSOR_20260826.md`,
SHA256
`3baf485008655aa5dc9e6a24d2a8e1805b791e1c78e377b1eeb45b3eceea1a3c`.
Its 55-path current successor closure is
`4aa89da216fdf401e6c129a12af28cbb45e9f858d5f4db092fb24b1d77c6947c`.
The closure reader itself was repaired before launch to use held
`O_NOFOLLOW` directory/leaf descriptors with post-read named-identity
revalidation. Root independently reproduced 62 V1+V5+V6 no-CUDA tests,
recomputed the closure twice, descriptor-validated the actual V3 and V5 graphs
as `5b6d16fd...` and `5c01b73b...`, verified the V6 root absent, and attested
the unchanged GPU1 runtime/device profile. The accepted V6 identity is
`cd8c387317e27865f2b1bbc2b6594aa7d258dd10c9eb472493df0029fdda8879`.

The sole V6 source-only smoke then completed successfully on GPU1. The accepted
terminal body SHA256 is
`dab4c9cfaad593c76d14b9d300642da31d7545248d4daf22bf2691c2b8cceb01`.
Root and Luna independently descriptor-checked the exact 16-leaf immutable
graph, all body/sidecar digests, the V6 identity and closure, both checkpoint
manifest links, and the absence of a failure receipt. The run completed 100
optimizer steps in 38.2904 seconds (2.6116 steps/s), covered all 31 trainable
parameters with finite gradients, strictly reloaded the best and last
checkpoints, and kept target/heldout/formal/minival/EvalAI unopened. Peak CUDA
reserved memory was 899,678,208 bytes. GPU1 was released after terminal; GPU0
was untouched. This clean smoke authorizes design and review of an additive
full source-training successor, not direct reuse of the smoke root and not an
automatic long-run launch.

The first full-successor seam audit then found that the executable runner and
lifecycle are hard-coded to the 100-step smoke even though the typed route
spec can describe `run_kind="full"`. Copying that optimizer loop into an
additive wrapper is forbidden because it would silently fork the accepted V6
training semantics. Root therefore authorized only a minimal backward-
compatible shared runner/lifecycle seam. The full route must use the already
frozen source recipe: 20 epochs, Adam at `1e-5`, weight decay `0`, no scheduler,
source-train-loss-only best checkpoint, and a last checkpoint. It explicitly
has **no SWA**: the accepted lifecycle workorder says “no SWA in this route,”
and no closure-bound M1 authority defines an averaging window or algorithm.
The successor must reject any `swa.pt` leaf or SWA claim rather than inventing
one. At its implementation freeze this was a no-data/no-CUDA code candidate,
not yet a full-run authorization; the separate root audit and launch decision
are recorded below.

That successor is now frozen under workorder SHA256
`a71a7402d43f5012028dcd237d42861b647ec327e433f23df9c7a1bb10090179` and
55-path closure
`5399ce39697a5badfe414a0a4679116a6fb10ec70be39a399f81467d63d29b89`.
Terra's 72-test combined smoke/full regression and root's independent rerun
both passed; root also descriptor-loaded the real accepted V6 graph, rebuilt
the closure twice, issued an ephemeral no-write capability, confirmed the
prospective fold root fresh, and observed `torch_loaded=false`. The accepted
full identity is
`c6b23446bca124b26b47c01485de19354be4d47a42a2bc5e59e43ee0b34b56d3`.

The exact source authority gives 4,951 optimizer steps per epoch, hence 99,020
steps across 20 epochs. The V6 smoke throughput projects about 10.5 hours.
One wrapper invocation exited before root reservation because the exact parent
namespace did not yet exist; it created no fold root, attempt, data access, or
CUDA state and is not an experiment attempt. Root then created only that exact
empty namespace and repeated all freshness/preflight checks. The first durable
attempt is now active on GPU1. Its initial immutable receipts are:

```text
attempt  df01f5ab568f549b59cb64839c6b4d3e164b1b5f9bab86516c1892f86d6c1476
launch   b29674183398b73f89cccfa332e9f8f865d4701769aced5f04d3451808cb7780
source authority  fbec26d53f65e4dbb59c6dd0f80043181bba75b81a287ca2c9ae26ef10ca0df5
```

At 2026-08-26 14:48 HKT, root descriptor-checked the newly published source
authority body and canonical sidecar. Both are regular, non-symlink, mode-0444
leaves and the body rehashes to the digest above. It binds the accepted full
identity `c6b23446...`, confirms source-only execution, records all target,
held-out, formal, minival, and EvalAI surfaces unopened, and fixes the exact
20-epoch / 4,951-step / 99,020-update no-SWA plan. GPU1 had initialized and was
performing the first epoch; the Python process and tmux remained live and the
log contained only the already known HDMF namespace warnings. No epoch receipt
had yet published, so no loss trend or performance conclusion is claimed at
this boundary.

Epoch 00 subsequently published at 2026-08-26 15:19:31 HKT. Root verified its
regular mode-0444 body/sidecar pair and canonical rehash as
`b5828850ed02150da66eb6bc63798e6becb4be5ce2fb1b1e571efbe866d41611`.
The receipt exact-binds the full identity, accepted V6 graph, and source
authority; records exactly 4,951 completed optimizer steps; and reports finite
objective, model, and Adam state with every non-source surface still unopened.
The arithmetic-mean complete source objective is `0.11030983995631942`.
Source-authority publication to epoch-00 publication took approximately
32 minutes 9 seconds, or about 2.566 steps/s. If that rate remains stable, the
remaining 19 epochs require approximately 10.2 hours. This first value is a
baseline only: one epoch does not establish a decreasing loss trend.

Epoch 01 published at 2026-08-26 15:51:59 HKT and independently passed the
same pair/mode/sidecar/link checks. Its body SHA256 is
`0c71fad75663869dff98e2a78f853c6b49dd42582edeaf055f23cfa631fdb6fd`.
It records another exact 4,951 steps (9,902 cumulative), finite objective/model/
Adam state, and source loss `0.10085561305887951`. Relative to epoch 00 this is
an absolute change of `-0.00945422689743991`, or `-8.5706%`. The epoch-to-epoch
interval was approximately 32 minutes 27 seconds (about 2.542 steps/s). This is
a healthy early optimization signal, not evidence of held-out improvement;
those surfaces remain unopened and the matched source-only evaluation remains
gated on a clean full terminal.

Epoch 02 published at 2026-08-26 16:24:30 HKT. Its verified body SHA256 is
`198ee4b3e53be6c7f0d5a4253d882cd795de5ca5c7a812d6ca36005c41e312dd`;
the exact cumulative step count is 14,853 and source loss is
`0.09126316086213995`. This is `-0.009592452196739557` (`-9.5111%`) versus
epoch 01 and approximately `-17.27%` versus epoch 00. Three consecutive points
therefore establish a stable early source-optimization decline, while still
not establishing held-out transfer. Model, Adam, and objective remain finite;
all non-source surfaces remain unopened and no failure receipt exists.

Epoch 03 published by 2026-08-26 16:57:57 HKT. Its verified body SHA256 is
`eec8fb183643ee06f16c352a5540c8aa1f93059fc44a7c886e7d5772d7fbab63`;
the exact cumulative step count is 19,804 and source loss is
`0.08356036190857757`. This is another `-8.4402%` versus epoch 02 and
`-24.2494%` versus epoch 00. The fourth consecutive point strengthens the
source-optimization signal, while still making no held-out-transfer claim.
The model, Adam state, and objective remain finite; all target, held-out,
minival, EvalAI, and formal surfaces remain unopened and no failure receipt
exists.

Epoch 04 subsequently published with verified body SHA256
`3f12a0d9171bf5daab167bd15b07311fc26bc45fb9a160de308f7984cca13c55`.
Its source loss is `0.07717238780535446` at 24,755 exact cumulative optimizer
steps, another `-7.6447%` versus epoch 03 and `-30.0403%` versus epoch 00.
All model/Adam/objective finiteness and source-only boundaries remain valid.

Epochs 05--07 subsequently passed the same immutable body/sidecar and semantic
checks. Their verified SHA256 / loss / cumulative-step triples are:

```text
epoch 05  f27b9ad28c6930e1c597007d06b035e6b1e097c7e46f50e677d24febd4fa0611  0.07368252142286408  29,706
epoch 06  42a4ba6dab02b9ca64070c906e43cf7a5dd3c61cdedde6d2e7a86e4d2d1bb1cf  0.07067302775433258  34,657
epoch 07  df8fe0491ccecac6d3f422f828f5c1a53c550d9778f0bf4b3537d0b4f46e0d6c  0.06964815307786291  39,608
```

The epoch-07 loss is `-36.86%` relative to epoch 00. The decline is still
monotone across all eight published epochs, although the latest incremental
gain has narrowed to about `-1.45%` versus epoch 06. This is a healthy source
optimization trajectory with early signs of flattening, not yet a held-out
performance result. All eight receipts report finite objective/model/Adam,
exactly 4,951 steps per epoch, no SWA, and every non-source surface unopened.

Epoch 08 then published as the ninth consecutive valid receipt. Its body SHA
is `74c31a699c15aa174d5230be28f7e8e15111ac25a996772d037f9119322858c2`,
source loss is `0.06649833143459434`, and cumulative updates are 44,559. This
is `-4.52%` versus epoch 07 and `-39.72%` versus epoch 00, so the prior apparent
flattening was not yet a sustained plateau. The same finite/source-only/no-SWA
boundaries remain true.

Epochs 09--13 have since published without a gap or failure:

| Epoch | Source loss | Cumulative optimizer steps |
|---:|---:|---:|
| 09 | 0.06348262487686275 | 49,510 |
| 10 | 0.06314315133784708 | 54,461 |
| 11 | 0.06126227504471243 | 59,412 |
| 12 | 0.05925524554113229 | 64,363 |
| 13 | 0.05736489626036295 | 69,314 |

Epoch-13 body SHA256 is
`ac069bdd216883a9386118596886f89e476a0bffa2134aa158cabe44c9712a69`.
Its body and canonical sidecar are regular mode-0444 leaves and rehash exactly.
The run is now 14/20 epochs complete. Epoch-13 loss is `-47.997%` versus epoch
00 and `-13.735%` versus epoch 08. All 14 epoch losses remain monotonically
decreasing; finite objective/model/Adam and source-only boundaries remain
valid. The latest four epoch intervals are 1,938--1,945 seconds, giving a
current projection of approximately 2026-08-27 01:35 HKT for epoch 19, before
final best/last checkpoint and terminal publication.

The projection was exact: epoch 19 and the complete terminal graph published
at 2026-08-27 01:35 HKT. The Python process and tmux exited normally and both
GPUs returned idle. Root then performed a descriptor-held, checkpoint-tensor-
free audit of the complete result. The topology is exactly 28 approved bodies
plus their canonical sidecars (56 mode-0444 regular leaves), with no failure,
SWA, symlink, or extra leaf. All bodies rehash to their sidecars; the current
55-path implementation closure and accepted V6 predecessor rebuild exactly;
all 20 epoch, training, manifest, checkpoint, and terminal semantic validators
pass. Key immutable bodies are:

```text
training             0250b689de465b8777485883e9646c5fc64bfa9d5ea710b8670afffece5406b1
checkpoint manifest  e190f6c813d5f9403e7d4f505cebbeb4830edb63f1ae2a11d15bdc9907a40550
terminal             efd084573843e05ada6c28c050c28e8bbf01976bec443d864bc3a0a7921afdc5
best checkpoint      2cefa5cbeec5653a4fed76cacfb61113ae47bf4546879cf6cc9f6f427890d9e9
last checkpoint      2cefa5cbeec5653a4fed76cacfb61113ae47bf4546879cf6cc9f6f427890d9e9
checkpoint state     d5d86325e21b5a257a44ee2eff4a9e35ddba9d1d6b0de591e607538bbbb41db7
```

Epoch 19 is both the first strict minimum and the last epoch, so best and last
checkpoint bytes/state are exactly identical. Final source loss is
`0.051446598575743366`, down `53.3617%` from epoch 00; all 20 losses are
strictly monotonically decreasing. The terminal records exactly 99,020 source
optimizer steps, no early stopping, no SWA, and every held-out/target/formal/
minival/EvalAI surface unopened. This is a clean training terminal, not yet a
performance verdict. The next indispensable paired cell is the same-fold
ordinary-ERM full training and matched outer-target evaluation; additional
CS-WG folds alone cannot establish a delta without that reference.

Root then prioritized the direct held-in metric instead of treating source
loss as a performance proxy. A new additive metric-only scorer was frozen with
a 51-path closure and 12/12 no-CUDA focused tests, then executed exactly once
on GPU1 against the immutable best-source-loss checkpoint. Its canonical
result root is:

```text
tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_heldin_score_v1
```

The result terminalized successfully with no failure. All five body/sidecar
pairs are regular mode-0444, nlink-one leaves with exact canonical sidecars.
The governing result is:

| Surface | Checkpoint | Valid final-bin windows | Variance-weighted R2 |
|---|---|---:|---:|
| M1 held-in session `20120924` | CS-WG fold-20120924 best source-loss epoch 19 | 54,849 | `0.5679166316986084` |

The evaluation performed 429 logical B128 forwards in `eval()` and
`no_grad()` mode. Prediction SHA256 is
`781bf769021a67aaedd8839b51b90ee65203e28a8ded366612e51fb8c85f336f`.
The strict model-state digest before and after evaluation is identically
`d5d86325e21b5a257a44ee2eff4a9e35ddba9d1d6b0de591e607538bbbb41db7`;
target backward, optimizer, and update counts are all zero. Immutable receipt
SHAs are:

```text
attempt          5145133c258f6ae5565a3a6d13cebacd46d9f3b3e93cf6e82341ece56736396e
input authority  342c78953a5e7070e3168db7c7644f9d543ee36459243bc9323fe815ce8a9a00
score            d5a08db493ced3c7284fc3d8b295c9e9609326658cf7deb9ef49f9826d6bb605
terminal         0db36a9a5ee314cf4904b80449febe393ff107e510369b0f2b176b9945caa7e5
```

This closes the immediate question of absolute R2, but it does not yet answer
whether CS-WG is better than ordinary training: it is one fold and no matched
ERM checkpoint has been scored on the identical 54,849-window input record.
The next decisive experiment remains the same-fold matched ERM full route,
followed by a paired score on this exact target authority. Only that delta can
justify expanding CS-WG to the remaining M1 folds or H1.

That control has now passed its no-data boundary and root review. Its 61-path
closure is
`7e158610b8ec4b90d8c6ffc4d6e038b536945251848d093816b57436dbd69fae`;
80/80 source/V5/V6/full/ERM no-CUDA regressions pass. It reuses the same M1
graph, seed 42, source sessions, B32 episode stream, 20 epochs, Adam/LR,
checkpoint rule, and no-SWA lifecycle. The sole objective difference is
`system=MATCHED_ERM`, `lambda=0`, `tau=.01`; its autograd receipt requires the
uniform `(1/3,1/3,1/3)` session-loss derivative. The live identity is
`999351c489c32970dc8988d686c1b2a3854e1aedbdf18a5431cc6df4c6a8a399`.

The first orchestration command stopped before artifact reservation because
the new canonical parent directory did not yet exist. It created no attempt,
failure, data access, or CUDA state. Root created only that declared parent,
revalidated the unchanged identity/closure/V6 graph/fresh child root, and
started the sole durable attempt on GPU1. Its canonical root is:

```text
tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/fold_20120924_matched_erm
```

At launch handoff, PID `540993` and tmux
`cswg_m1_matched_erm_fold20120924_20260827` are live; immutable attempt and
launch pairs are present, source preparation is CPU-only, GPU1 has not yet
initialized, and no failure is present. Luna owns read-only event-first
monitoring with 30-minute aggregate reports. GPU0 is outside this route. No
retry is authorized if this durable attempt terminalizes as failure.

The matched-ERM producer subsequently terminalized cleanly. Source authority
and all epochs 00--19 published without a failure. The source-authority body
SHA256 is
`41e4ca3233f9673ccc5e03cacf16239a43631560c6a4d49c92fbab9fe8827310`
and reconstructs against the exact ERM identity, accepted V6 graph, 4,951
steps/epoch, and 99,020 total updates. Source objective loss decreased
from epoch 00 through the final epoch:

```text
epoch 00  0.0985941126347461
epoch 01  0.08674240504566244
epoch 02  0.07875782314788041
epoch 03  0.07170910342727958
epoch 04  0.06609958258497549
epoch 05  0.0628624660666664
epoch 06  0.06002388507958446
epoch 07  0.05914701766830515
epoch 08  0.05619786407891078
epoch 09  0.053669263024396265
epoch 10  0.05336162906781161
epoch 11  0.05155756396010677
epoch 12  0.04994913633193101
epoch 13  0.04814271551995741
epoch 14  0.04791307359114831
epoch 15  0.04760095414793421
epoch 16  0.04636659329278171
epoch 17  0.044324925500606856
epoch 18  0.0443174803421282
epoch 19  0.042958451470428254
```

Every epoch pair rehashes, has mode 0444, reports finite objective/model/Adam,
and binds zero target updates with all forbidden surfaces unopened. The run is
20/20 epochs complete with exactly 99,020 optimizer steps and no failure.
Epoch 19 is both the first strict source-loss minimum and the last epoch; its
body SHA256 is
`8ed117233d3b9269bcf88555d130aad9631c8dffe950d7d4b0137a8c85339360`.
Best and last checkpoint bytes are therefore identical, with body SHA256
`91bd46c0261df141b4fd601eeaa0f105fb637e75aed572d734b94ae0e8b9486e`
and strict state SHA256
`2d1bdb18e99100ea3c798513f353c5686db81b4687f8ec623e14ac32fc535152`.
The training, checkpoint-manifest, and terminal body SHA256 values are,
respectively,
`9dacb5d2bab90499e99d7f027c52186efe7b80230199590b76c89357a27b6e28`,
`10b1177f4f72f1f0b392e404cf54166ed3f6479d2cf8877b3e6e4a4ba33bba2c`,
and `f74fccada57e70c5c7d9186a0b3ade1d52346d32c74b0e5638cb7bf05a7041f4`.
These losses monitor optimization only;
because ERM and CS-WG have different complete objectives, cross-system loss
values are not used as a performance comparison.

The additive matched-ERM held-in scorer passed its no-data/no-CUDA review and
then executed exactly once after the producer terminal. It reuses the accepted
fold-20120924 reader, model builder, strict checkpoint reload, final-bin-only
prediction, and variance-weighted R2 path. Its typed producer binding pins
every immutable ERM attempt/launch/source-authority/training/identity/manifest/
checkpoint/terminal fact. It descriptor-reloads the accepted CS-WG input and
score receipt pairs and exact-compares the underlying 54,849-window target,
calibration, ordered-window, query-identity, eval-mask, target-array, and reader
recipe evidence rather than incorrectly requiring producer-dependent input
authority bodies to hash identically.  Root independently reran the exact
V1-and-successor no-CUDA suite: 25 tests passed.  The 60-path successor closure
reconstructed twice as
`9cd35e257cc541a25d3a47cae4a9f465f39e6832cc2c92147cda9ad3e249fcae`;
the work order SHA256 is
`4b9da84fb4f5a9cf08d0bf1cb019f00ef45fbdb28f233fdcf2c02d25d9f70e2e`.

The matched-ERM score root is
`tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_v1`.
It contains exactly five immutable body/sidecar pairs, no failure, and passed a
fresh semantic revalidation through the frozen scorer without importing
Torch. Its governing result is `0.5707439184188843`, using 429 logical B128
forwards on the same 54,849 final-bin windows. Prediction SHA256 is
`e2b2e2f8976b07f3a1d60f659ea32aae285788c6887b02ee2fd05320ec298842`;
target SHA256 is
`e913d03a972154a4a7ad3eae9963174fb54dbeefd3bc32b542b5cb76bc0ff8aa`;
and model state before/after is exactly
`2d1bdb18e99100ea3c798513f353c5686db81b4687f8ec623e14ac32fc535152`.
The immutable result SHAs are:

```text
attempt          152ae5bd06959021a9594d33dfa98ce3a8a6e28d81e9e6703d732fd77e35be52
launch           bba9fe2083195c459dab3c432bbaa5bcaa029195c6042b9579699199098d6f2b
input authority  efe1a76a672c134d0983a29995a2a6bbd98b955829ab58fb5c07da482ee96596
score            a3eeb0b3efeb33bdcd209ce5113336d4e636f87d5e9c0b3c57cc59064d3eb59f
terminal         b0717e65f43b5c791edafe427a209ffd87ae1dded2c96aef2a4acce599dbada3
```

The decisive paired comparison is therefore:

| System | Same target/input | Variance-weighted last-bin R2 |
|---|---:|---:|
| CS-WG | 54,849 windows | `0.5679166316986084` |
| matched ERM | 54,849 windows | `0.5707439184188843` |
| CS-WG minus matched ERM | exact paired cell | `-0.0028272867202759` |

This is a clean single-fold negative/null mechanism result: CS-WG is about
`0.495%` below matched ERM relative to the ERM score. It is not a universal
formal benchmark rejection, but the first strict paired cell provides no
positive signal to justify more expensive CS-WG folds on M1 or an H1 port.
CS-WG expansion therefore stops; future compute returns to genuinely new
performance mechanisms rather than attribution or replication of this route.

Before the paired result was available, root audited whether the next LOSO
fold could start in parallel. The mathematical Stage-0 factory
can construct outer target `20120926` with sources
`20120924/20120927/20120928`, but the accepted V6 source authority cannot
authorize it: V6 and the current full physical rebind both hard-bind the
historical audit target `20120924` and require the full spec to have the same
outer target and source-session set. Reusing that graph would therefore be a
provenance error, not dynamic scheduling. The minimal safe parallelization
sequence is a fresh fold-20120926 common-stratum/source audit, its own 100-step
source-only smoke using the existing V5 derivative observer, and only then a
disjoint full-training root. The negative paired fold-20120924 result removes
the scientific justification for paying that cost, so this successor chain is
not authorized and no additional CS-WG fold should be launched.

### CDM-D pure-speed engineering successor — 2026-08-26

The operator throughput review found that historical CDM-D scoring is CPU-
dispatch bound: it repeatedly recomputes the same B3S identity and duplicates
every logical forward for receipt auditing. A new additive Stage-0 accelerator
therefore implements only two execution changes while retaining logical B128:

1. compute the B3S identity once at batch size one for each exact activity /
   normalized-carrier / held-view state and reuse it through the existing
   direct-identity decoder path; and
2. repeat the canonical full-path first chunk once per distinct causal state,
   plus one fixed mid-session held-group-0 coordinate, rather than duplicating
   every full and group chunk.

Root independently verified on the real CPU Cell-D graph that B=1 cached
identity and eager batch-expanded identity/predictions are bitwise equal. The
same audit showed that a larger physical batch is numerically equivalent but
not byte-identical: B128 versus B2048 changed prediction bytes, with maximum
absolute drift `2.60770320892334e-07`. Stage 0 therefore kept B128 because its
original contract required an exact prediction digest. The user subsequently
relaxed that engineering constraint. Phase 2 now uses B1024 as the preferred
optimized physical batch, with deterministic B512 then B128 fallback on OOM,
while retaining B128 for the sealed predecessor baseline. O3 group subsampling,
O5 multi-process scheduling, disk caching, and TF32 remain outside this route.

The Stage-0 workorder SHA is
`264468964aa6ec233046e30a2ec5ead20b0e2114ff5e95e94fcae68bf878364f`;
its explicit 109-path closure is
`caca8db6483bcd95ff424c13def36b4a8ebfa211b055528a0b7bea3b65247bb3`.
Root reproduced the closure twice and independently passed the bounded
Precision-V2 / V8 / speed regression (`59 passed`). This is acceleration
machinery, not a launchable scorer: it has no authority/result lifecycle or
execution CLI. The active Phase-2 task is to bind the accepted Precision-V2
graph, persist typed speed evidence, and run one future GPU0 same-input
prediction-SHA/R2/throughput smoke. GPU1 and the active CS-WG run remain outside
that task.

After the user authorized a faster, less exhaustive engineering gate, root ran
a no-data/no-checkpoint GPU0 synthetic test on the real Cell-D graph with TF32
disabled. For B512 / M30 / N64, historical eager-repeat execution averaged
`0.0211109 s` and cached-identity/sampled-repeat execution averaged
`0.00922982 s`, a `2.287x` decode-path speedup; peak allocated CUDA memory was
152,517,120 bytes. The GPU test also supersedes Stage 0's CPU-only bitwise
assumption: batch-one cached identity versus batch-128 eager identity changed
prediction bytes (`max_abs=3.5762786865234375e-07`) and therefore changed the
prediction SHA. The launchable Phase-2 contract must consequently run old B128
and optimized B1024 paths on the same held input, require the old path to
reproduce its sealed predecessor row, require exact transition chronology, and
gate the optimized path on `max_abs_prediction <= 1e-6` and
`abs_delta_R2 <= 1e-7`. It must label the result numerically equivalent rather
than bitwise identical, disclose the selected/fallback batch and memory peaks,
and require only a positive end-to-end speedup for the smoke (`>=1.5x` remains
the adoption target).

Root then exercised the relaxed B1024 choice on GPU0 with the real Cell-D graph,
TF32 disabled, synthetic inputs only, and no data or checkpoint access. Across
8,192 windows, the historical B128 eager-repeat path took `0.328504 s` and the
B1024 cached-identity/sampled-repeat path took `0.079700 s`, a `4.1218x`
speedup. Identity-encoder forwards fell from 128 to 1 and model forwards from
128 to 9. The optimized path used 954,813,952 peak allocated CUDA bytes. Its
maximum absolute prediction drift was `5.960464477539062e-07` and its absolute
synthetic last-bin R2 drift was `5.98374754190445e-08`, so both relaxed numeric
gates passed. This is descriptive synthetic evidence, not the live predecessor-
bound speed result; it is sufficient to make B1024 the first physical batch for
the one-session Phase-2 smoke.

Root independently accepted the frozen Phase-2 candidate after reproducing
all seven owned-file hashes, `34/34` combined no-CUDA tests, and the explicit
123-path closure twice as
`71b97d7d04856718f6b321831d5815c9778fc85a0fe20472f13a2f8ca7e5fd85`.
The live accepted Precision-V2 12-leaf graph descriptor-validated, and the
new authority and score roots were fresh. Root additionally exercised the
actual production `_PhysicalBatchIdentityCacheAccelerator` on GPU0: physical
B1024 retained `max_abs=5.960464477539062e-07`, used about 0.95 GB peak
allocated memory, and recorded 64 logical chunks as 9 model forwards plus one
identity forward.

The one authorized live speed smoke launched on GPU0 at 2026-08-26 HKT. Its
identity is
`a4555723206ddd2c9f53e878238f256f472702c59fb98bab8f8ece9ea7c866b8`.
The immutable authority pair SHAs are
`b53fa34f36616713388db838540c9f7e9f826130aa1bfaa5ed2287693111a7fe`
and
`83f4cfd719002b0ba928977fff0ce8ff7de0786dc1ea4b35327bc5452015e7d1`.
Runtime monitoring binds tmux `precision_cdmd_speed_smoke_v1_gpu0`, Python PID
498876, log `/tmp/precision_cdmd_speed_smoke_v1_gpu0.log`, and result root
`tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1`.
At launch the result root contained only the valid immutable attempt pair;
GPU1 CS-WG PID 486447 remained unchanged. Luna owns read-only event-first
monitoring of both jobs at no more than 30-minute intervals.
The attempt body SHA is
`68419c40f381737c3700b9893cfd7d4fa489c4a887efdaf50ea999a8a98d992b`.
At the first live checks the speed-smoke process remained CPU-active in the
inherited full-input-authority materialization stage, with no failure receipt
and no GPU0 forward yet.

The V1 smoke subsequently terminalized fail-closed at `budget_m4` after input
authority publication and after both evaluation passes. Its immutable body
SHAs are: attempt
`68419c40f381737c3700b9893cfd7d4fa489c4a887efdaf50ea999a8a98d992b`,
input authority
`18b58d85cea9099360f7f0f4123ff99e45a258b3ea9a6928f2cf1240e77e111c`,
and failure
`dd1f6b9bd4ced42ff24a4a4ef1e43eb031987541282f1e8be638cb1b772f37a4`.
The failure is a `SpeedSmokeError` from the combined numerical/speed predicate;
CUDA and the checkpoint were opened, full/group forward counters reached
1,016/3,493, and target optimizer/backward/update remained exactly zero. It
was not a CUDA OOM. V1 did not persist the individual comparison scalars in
its failure receipt, so the exact failed member among max-absolute prediction,
R2 delta, and speedup is not recoverable from the immutable graph. No V1 retry
is permitted. GPU0 was released and the independent GPU1 CS-WG run was not
affected.

An additive V2 diagnostic successor is therefore required before another live
attempt. It must bind the exact V1 failed graph, persist all comparison scalars
on both completion and failure, retain the exact B128 predecessor anchor and
transition chronology, use relaxed informational tolerances of `2e-6` and
`2e-7`, and report speedup without making `>1.0` a terminal condition. The
original V1 tolerances and their pass/fail values remain disclosures.

That V2 diagnostic successor is now frozen and independently reviewed. Its
seven owned files are bound by the exact 130-path closure
`8b975917da3021caa754667fbe474e1eb23506d9efab6aa8956a75aeb6c6dbb1`;
root reproduced the closure twice, py-compile, dry CLI, and the combined V1+V2
no-CUDA suite (`20 passed`). Its accepted identity is
`6289359119567ca3d2b40b89a62dac63d15f1c9437a616fe860f160a36730270`.
It descriptor-binds the exact six-leaf V1 failure graph before every mutating
boundary and records the raw numerical comparison before applying V2 policy,
so either a terminal or a post-comparison failure must disclose the measured
prediction error, R2 delta, selected batch, memory, forward counts, and
speedup. V1's three predicate outcomes remain visible; V2 gates only
`max_abs <= 2e-6` and `abs_delta_R2 <= 2e-7`, while speed is descriptive.

Root published and descriptor-reloaded the V2 target-free authority graph. Its
four leaves are regular mode-0444 files with canonical digest sidecars:

```text
official preflight  d45d8f0950c14485b6e440b7b6c4807a4a86a95b9b525283071499b256562b67
root authorization dba8aa5d60e4e741d11fbc14d58e3f87bb62820266f7f09d47758670dae53e61
V1 failure binding  022dac5f0b3f137de9ec62a86dfcb391c91dba409616be2d1b73b3f62f34a753
```

The sole V2 live attempt then launched on otherwise idle GPU0 under tmux
`precision_cdmd_speed_smoke_v2_gpu0`, Python PID 503938, and log
`/tmp/precision_cdmd_speed_smoke_v2_gpu0.log`. Its immutable attempt pair has
published; CUDA initialized on GPU0 while GPU1 CS-WG PID 486447 remained
unchanged. No score, terminal, or performance conclusion is claimed until the
natural V2 receipt event. There will be no V1 retry and no V2 intervention.

V2 subsequently terminalized fail-closed at `budget_m4`, again without a
retry. Root descriptor-validated the exact six-leaf immutable graph, all
regular mode-0444 body/sidecar pairs, and reconstructed the route validators:

```text
attempt          42a64097ba0be8ac8146e27258ca602462f8e617659d86a0368dc70e51210c5d
input authority  da3069cbffabbddf81f0b85e0aec1d6147bd8aed9b85e7fb4ff25c4fd0a2d83a
failure          86e2af8485e1ffe4410dbd631d4f50065d431102bdd6524bb2b6858e7968ce86
```

Unlike V1, the V2 failure preserves the decisive measurements:

| Measurement | Historical B128 | Optimized B1024 | Result |
|---|---:|---:|---:|
| Governing session R2 | 0.3249187469482422 | 0.3249187469482422 | absolute delta 0 |
| Wall time | 20.0898657 s | 17.5696235 s | 1.14344x speedup |
| Full-system forwards | 722 | 294 | reduced |
| Group forwards | 2,888 | 605 | reduced |
| Prediction bytes | distinct SHA | distinct SHA | max abs 3.8146973e-6 |

Transition-record SHA is exact across both paths and all target optimizer,
backward, and update counts remain zero. The V1 R2 and positive-speed
predicates pass; both V1 and V2 max-absolute predicates fail. The result is
scientifically harmless at the governing metric but does not reach the 1.5x
engineering recommendation, and its pointwise drift exceeds V2's already
relaxed `2e-6` limit. Therefore B1024 is not adopted into the production
scorer. A further tolerance-only V3 is not justified for a 14.3% end-to-end
gain. GPU0 was released and the GPU1 CS-WG process remained unchanged.

## 1. Decision first

The program should remain performance-first. Do not spend the next round on a
long chain of precise ablations before testing a method that can materially
improve the system.

The next two performance bets are:

1. **M2/SUA M4/M10: Causal Dual-Memory Cell D (CDM-D).** Keep sealed Cell D,
   but use completed unlabeled query trials to update both activity identity and
   functional carrier state. Test the complete system first. Separate the two
   memories only after a positive result.
2. **M1 first, H1 later: Cross-Session Worst-Group SPINT (CS-WG).** Keep the
   exact baseline graph and last-bin loss. Change source training to optimize
   balanced, task-stratified worst-session transfer instead of average source
   fit. This directly targets the observed generalization gap without adding a
   recurrent decoder.

Do not launch TQ-FOD under its current specification. TF-SR is its direct
ancestor and already showed that additional temporal capacity improves within
fit while harming external transfer.

The missing honest oracle matrix, B3S-only extension, SFORA, and component
ablations remain useful. They are now **post-success attribution or opportunistic
diagnostics**, not blockers that must consume the next experimental round.

## 2. Facts that the new plan must respect

### 2.1 TF-SR is the relevant failed ancestor

TF-SR already tested:

```text
B3S calibration activity + normalized closed-form T4
    -> causal per-time fused unit tokens
    -> state-conditioned permutation-invariant read-in
    -> GRU recurrence
    -> velocity
```

Its matched result was:

| Surface | TF-SR seed 42 | Sealed Cell D | Delta |
|---|---:|---:|---:|
| External governing | 0.2542 | 0.4179 | -0.1638, 1/15 positive |
| Within governing | 0.5919 | 0.5697 | +0.0222, 4/6 positive |

The result is not explained by T4-only identity, destructive Hadamard fusion,
or a large parameter budget. TF-SR used B3S plus T4, non-destructive fusion,
Cell-D whole-unit dropout, and 1.65M parameters.

The usable mechanism conclusion is:

> More temporal capacity bought in-distribution fit, not transfer.

Any future temporal architecture must explain why it should reverse this exact
pattern. Contrasting only with Large-v1 is insufficient.

### 2.2 M1/H1 do not have demonstrated output-side headroom

The earlier document inferred that failed behavior-latent and carrier routes
implied an output-readout bottleneck. That inference was too strong.

For M1:

- The q8 oracle 0.477283 and raw-head 0.6702 are not the same scoring surface.
- The q8 value is a leakage diagnostic fitted on full sessions, including query
  labels, over four sessions and about 53k query bins per session.
- The raw-head value is a three-fold deployment replay with about 26k bins per
  target fold.
- Their difference is not an output-capacity estimate.
- Source-session decoder query R2 is about 0.80-0.81, while target-fold raw-head
  R2 is about 0.65-0.68. The visible gap is cross-session generalization.

For H1:

- Query-fitted q=4 carrier 0.522652 does not beat support carrier 0.525511.
- The source document states that this only bounds the tested frozen
  PCA/ridge/U/EB estimator and consumer. It does not upper-bound another
  source-learned analytic operator.
- The carrier effect is still real in other comparisons; it is strongly
  concentrated in `rx`.

The correct routing statement is:

> M1/H1 are currently transfer-limited. The evidence does not yet identify a
> new output topology as the missing mechanism.

Do not call them "high-dimensional" in the scientific argument. Their effective
behavior dimensions are low even though the raw output vectors have 16 and 7
coordinates.

### 2.3 Actual M1/H1 architecture facts

| Dataset | Window W | Outputs | Training loss | Live parameters |
|---|---:|---:|---|---:|
| M1 | 100 | 16 | final bin only | 15,007,496 |
| H1 | 700 | 7 | final bin only | 16,855,196 |

`FalconLitModule.model_step()` slices both prediction and target to the final bin
when `decode_last_timestep_only=true`, which is enabled for both datasets. The
deployed wrapper also returns the final bin.

The baseline is already wall-clock causal with a trailing window. Its limitation
is no recurrent state across windows, not non-causal deployment.

A W=100/700 recurrent replacement would either serialize many steps for one
supervised output or change the loss to all-bin supervision. That choice was not
resolved in TQ-FOD.

The earlier 2M-4M TQ-FOD budget would also have reduced capacity by four to eight
times. It was not a matched first cell.

### 2.4 M2/SUA evidence supports target-time information acquisition

Current honest total-calibration results are:

| Budget | Current recipe | Within R2 | External R2 |
|---:|---|---:|---:|
| M4 | Cell D + D-opt-first30 + fixed ridge 0.1 | 0.3089 | 0.1197 |
| M10 | Cell D + chronological support + fixed ridge 0.1 | 0.4677 | 0.2955 |
| M30 | Cell D + chronological support + fixed ridge 0.1 | 0.5665 | 0.4286 |

Important observations:

- Using M4 labels but M30 B3S activity raises external R2 from 0.1197 to
  0.1902, about +0.0705 without additional carrier labels.
- The old C3 full-label oracle reaches 0.4449, although it also uses M30 B3S
  activity and therefore does not isolate honest M4 carrier headroom.
- Four scattered labels and ridge tuning do not close the gap.
- CBM-D and PMC-D show that more source-side budget variation is unsafe.
- Cell D is explicitly trained under U(0,1) whole-unit dropout, making
  complementary-unit prediction feasible without retraining.

External-15 is zero-shot cross-subject transfer from sub-C source training to
sub-M evaluation. It is not a same-subject holdout.

### 2.5 PMC-D is a strong negative training result

PMC-D trained Cell D with posterior carrier samples, then evaluated with
ordinary deterministic OLS point T4. Its matched external results were:

| Budget | PMC-D | Sealed Cell D | Delta | Positive sessions |
|---|---:|---:|---:|---:|
| M30 | 0.0490 | 0.4179 | -0.3689 | 0/15 |
| M4 | 0.0061 | 0.1147 | -0.1086 | 7/15 |

This closes another source-training marginalization bundle. It does not close a
causal target-time information update on sealed Cell D.

## 3. Performance Design A: Causal Dual-Memory Cell D

Short name: **CDM-D**.

This is the primary M2/SUA M4/M10 performance design.

### 3.1 Core idea

Cell D has two calibration inputs:

1. B3S activity identity, derived from neural calibration activity;
2. T4 functional identity, derived from labeled trial direction and firing
   rate.

At deployment, keep the decoder sealed but maintain two causal memories from
completed query trials:

```text
completed unlabeled query trial
            |
            +--> activity memory --> updated B3S activity identity
            |
            +--> complementary-unit predictions
                    --> pseudo trial direction
                    --> cross-fitted T4 sufficient statistics

updated B3S + updated T4 --> sealed Cell D --> next-trial predictions
```

The prediction for trial `j` may use support plus completed trials `< j`. It may
not use future bins from trial `j` or any later trial.

This first experiment is deliberately a **system performance cell**. It changes
both target-time activity memory and carrier memory. If it succeeds, the two
memories are separated in later attribution experiments. Do not claim either
component as the mechanism from the first result alone.

### 3.2 Activity memory

Start from the honest M4 or M10 B3S support. After each completed query trial:

1. apply the exact calibration-trial neural preprocessing;
2. append the trial to a bounded FIFO activity memory;
3. recompute the shared B3S identity from that memory;
4. keep the carrier unchanged in this branch of the update.

No query behavior label is used. Trialized interpolation, padding, activity
segments, and roster order must match the source B3S contract. Arbitrary
continuous windows must not be passed into a calibration-trial encoder.

The deployment contract must explicitly permit past unlabeled query activity.
This is a legality/correctness condition, not an ablation experiment.

### 3.3 Carrier memory

Use K=4 deterministic complementary unit groups. Balance groups using initial
T4 direction and magnitude only; do not use query labels.

For each completed trial and group `k`:

1. hide group `k`;
2. predict the trial trajectory from the other groups;
3. integrate predicted velocity over the declared movement interval;
4. convert the displacement to one pseudo reach direction;
5. compute one scalar firing rate per held group unit over the same interval;
6. update only group `k`'s carrier sufficient statistics;
7. restore the full population for the next prediction.

The unit never generates the pseudo direction used to update its own carrier.

### 3.4 Match the production T4 estimand

Production T4 fits:

```text
rate_i(theta) = b_i + a_i cos(theta) + c_i sin(theta)
```

Its input is one direction per trial and one scalar firing rate per unit per
trial. It averages rates by discrete canonical direction before OLS.

The CDM-D first implementation should therefore:

- snap the completed-trial pseudo direction to the nearest canonical direction;
- maintain per-direction count and per-unit rate sum;
- refit the same closed-form cosine estimator after an accepted update;
- keep the existing fixed ridge rule where that rule is the honest baseline;
- never feed `[T,2]` velocity directly to the carrier updater.

Before target scoring, run one clean-label parity check proving that true trial
directions and the same selected trials reproduce the production carrier within
a frozen tolerance. This is an implementation correctness gate, not a separate
scientific round.

### 3.5 Trust and fallback

Accept an update only if:

- integrated displacement and movement speed pass fixed minimums;
- complementary-group directions agree;
- the pseudo direction is sufficiently close to one canonical direction;
- the design remains well-conditioned;
- the carrier stays inside the existing departure/freeze trust region;
- enough completed accepted trials exist;
- no non-finite value appears.

Reuse the existing B8 thresholds and departure logic where they apply. Do not
tune them on external-15.

If any gate fails, preserve the previous state. The no-update fallback is sealed
Cell D and remains in the denominator.

Cell D's whole-unit dropout may make complementary predictions too similar for
disagreement to rank errors. Record the disagreement distribution. Do not
silently relax the gate if its dynamic range is poor.

### 3.6 First performance matrix

Compare only:

1. sealed Cell D under the honest budget;
2. complete CDM-D.

Score M30, M10, and M4 in that order during one reviewed evaluation lifecycle.
The scientific emphasis remains M10 and M4; M30 establishes whether the
pseudo-label source is usable before the weakest-budget case.

Hold fixed:

- sealed Cell-D SWA;
- last-bin, variance-weighted session R2;
- equal-session weighting;
- within-6 and cross-subject external-15;
- support selection and fixed ridge rules;
- target backward, optimizer, and parameter updates equal to zero.

Proposed performance gate:

- M30 external no worse than -0.01 versus sealed Cell D;
- M10 external delta at least +0.02 and at least 10/15 positive;
- M4 external delta at least +0.05 and at least 10/15 positive;
- no surface may be rescued by dropping failed/no-update sessions.

If the bundle is negative at M30, stop before M10/M4. If it is positive at
M10 or M4, run only these attribution arms afterward:

1. activity memory only;
2. carrier memory only;
3. complete dual memory.

The missing honest M-budget oracle can then size how much of the remaining gap
is estimator-limited. It is not required before the first performance screen.

## 4. Performance Design B: Cross-Session Worst-Group SPINT

Short name: **CS-WG**.

This is the primary new GPU design for M1. H1 follows only if M1 shows a useful
transfer gain.

### 4.1 Why this route

The strongest M1 evidence is a source-to-target generalization gap. TF-SR shows
that adding temporal capacity can improve within fit while making transfer
worse. The next GPU design should therefore change the training pressure toward
cross-session invariance while preserving the validated decoder graph.

### 4.2 Held graph

Keep exactly:

- the current M1 SPINT graph;
- W=100;
- 16 raw behavior outputs;
- final-bin-only raw-output MSE;
- B3S identity path;
- source/target folds;
- checkpoint and governing scorer;
- 15,007,496 live parameters after lazy materialization;
- no teacher, unit table, target fine-tuning, or target backpropagation.

Do not add GRU, SSM, timestamp query, behavior latent, output reconstruction, or
new inference-time module in the first system.

### 4.3 Changed training system

For each source fold, construct balanced source-session episodes. Within each
episode, stratify source query windows by coarse task state derived from source
labels only:

- inactive versus active output;
- source-only quantile of the raw output-vector norm;
- dominant raw output coordinate, defined by `argmax(abs(y))`.

Compute one last-bin loss per source session after equalizing the task-state
mixture. Optimize a smooth worst-group objective:

```text
L_session[s] = equal-task-state mean MSE for source session s

L_CS-WG = mean_s L_session[s]
        + lambda * tau * logsumexp((L_session[s] - mean_s L_session) / tau)
```

The implementation keeps the baseline total B32 window count per optimizer
step. In a three-source outer fold it forms rotating `11/11/10` per-session
microbatches, concatenates them for one unchanged-graph forward, and separates
the outputs again for the three losses. It must not run three B32 forwards or
triple the effective batch/MAC. The later all-four-source final model uses
`8/8/8/8`.

`lambda` and `tau` must be selected once using source-only nested
leave-one-session-out transfer, then frozen across target folds. They may not be
selected on target replay or official results.

This objective makes a source session useful only if the same graph works under
another session's neural distribution after matching behavior composition. It
does not compress the output and does not remove behavior information through a
session adversary.

### 4.4 Why task-state stratification matters

Plain worst-session loss can confuse neural distribution shift with different
behavior mixtures. A session with more inactive bins, a different output-energy
distribution, or a different dominant-output mixture could become the worst
group for the wrong reason.

Task-state stratification uses source labels during source training only. It
does not create a target behavior bottleneck and is absent at inference.

### 4.5 First performance cell

Compare:

```text
exact baseline graph + baseline source training
versus
exact baseline graph + balanced task-stratified CS-WG training
```

This is a training-system comparison, not a claim that one subcomponent is the
mechanism.

Primary M1 reading:

- four-fold held-in LOSO target replay delta at least +0.020;
- at least 3/4 held-in LOSO target sessions positive;
- source-session mean may not improve by hiding a large worst-session collapse;
- exact parameter count and inference latency remain baseline-equivalent;
- no official EvalAI opening for selection.

These four target sessions are the four M1 held-in source recordings, each
held out once for development. They are not the three official held-out M1
sessions. Hyperparameters and checkpoints are selected inside each outer fold
using only its three source sessions. The official held-out surface is opened
only once after a positive development result, using a final all-four-source
model and no further selection.

If negative, stop this design. Do not sweep session penalties, task bins, or
samplers.

If positive, run only two attribution arms:

1. balanced task-stratified mean loss without the worst-group term;
2. complete CS-WG.

### 4.6 H1 continuation

H1 uses the unchanged W=700 baseline graph, so CS-WG does not introduce the
serial-depth problem of TQ-FOD. If M1 is positive, adapt groups as:

- outer group: source date;
- inner balance: recording and source-only task-state strata;
- exact 7-DoF raw output and final-bin loss;
- 16,855,196 baseline parameters;
- preserve the selected H1 label-cost contract;
- report `rx` separately and require the gain not to be `rx`-only.

H1 does not launch from this handoff. It requires a new review after the M1
result.

## 5. What is deferred, not discarded

The following low-cost work remains useful but should not displace the two
performance cells.

### 5.1 Honest M4/M10/M30 oracle matrix

The calibration-gap ledger correctly marks these cells as missing:

- full-session labeled carrier;
- B3S activity limited to M4, M10, or M30;
- sealed Cell D;
- within-6 plus external-15.

These cells replace the old mixed C3 upper bound and explain a positive or
negative CDM-D result. Run them after the performance screen or opportunistically
when they do not delay it.

### 5.2 B3S activity-only attribution

The known M4 difference between B3S=M4 and B3S=M30 is about +0.0705. The
activity-only causal arm is the first CDM-D attribution if the full bundle is
positive.

### 5.3 Repaired SFORA

SFORA is still a valid cheap null test only after two repairs:

1. remove `alpha * I`, because an unconstrained diagonal `gamma` already absorbs
   it;
2. construct DirectRidge support predictions out of fold, leaving out one
   support trial at a time, before fitting the output correction.

Its prior is weak: M1 DirectRidge is 0.3623 versus raw head 0.6702, H1 Ridge is
about 0.2582 versus H-C 0.5255, and the existing M1 50/50 diagnostic mix is
slightly below the raw head. Run it only if it does not delay CS-WG.

### 5.4 Existing B8 gate

B8 code and thresholds already exist. CDM-D should reuse them as an engineering
preflight after the new trial-level pseudo-direction adapter is implemented.
Do not create a parallel pseudo-label gate protocol.

The frozen B8 estimand is carrier-level, not trial-level: refit true,
pseudo-labelled, and deterministically shuffled carriers on the same native
trial-rate rows, then gate the per-unit `[a,c]` cosine distribution. Raw
pseudo-versus-true trial-direction cosine remains a useful diagnostic but may
not inherit B8's thresholds. For the M30 preflight, use source trials after the
sealed first-30 support; replaying the support itself would leak each current
trial's label through its input T4. These post-support M30 rows are an audit
screen only and do not change M30's zero-capacity deployment memory.

The CDM-D extension must also preserve its `K=4` consumer: each held group has
its own pseudo direction, and that label may refit only the units in that
group. Fit/group/slice first, then concatenate per-unit cosine evidence in the
original channel order. Broadcasting one pseudo direction to every unit tests
old B8, not CDM-D. The grouped implementation must reduce exactly to old B8
when all four pseudo columns are identical.  Its population statistic covers
all and only theta-authority-valid units; undefined-direction channel rows stay
in the immutable channel map as `group=-1`/NaN and are receipt-counted, never
silently discarded or counted as failed carrier evidence.

## 6. TQ-FOD disposition

TQ-FOD is a quarantined concept, not the next architecture.

It may return only after all of the following exist:

1. a mechanism and prediction distinct from TF-SR;
2. a surface-matched diagnostic showing headroom for that mechanism;
3. correct W=100/700 cost accounting;
4. an explicit final-bin versus all-bin loss decision;
5. a precise per-time replacement for whole-window `fc_in` and W-dimensional
   B3S identity bias;
6. a parameter-matched system or explicit capacity control;
7. measured MAC, latency, memory, and wall time;
8. M1-first evidence before any H1 launch.

Do not defend it only against Large-v1. TF-SR is the required comparison.

## 7. Execution order

### Track A: M2/SUA, no new training

1. Freeze the CDM-D completed-trial causal contract.
2. Separate unconditional valid-B3S activity transitions from conditional
   carrier transitions in the V3 successor.
3. Preserve the immutable failed V3 smoke and repair only the closure-injection
   seam in the V4 physical-loader successor.
4. Pass the V4 no-data tests and one M10 source-only smoke.
5. Preserve the immutable failed V4 strict-27 gate and repair only its raw-event
   trace boundary in V5.
6. Pass and independently accept the V5 strict-27 gate with unchanged pools and
   thresholds. **Completed.**
7. Bind the accepted V5 88-pair terminal graph into a successor matched scorer.
8. Run the complete CDM-D performance screen M30 -> M10 -> M4.
9. Build the uncertainty-aware carrier-transition successor with M30 carrier
   memory disabled; run its complete matched M10/M4 screen.
10. Only after that new performance result, run activity-only and carrier-only
    attribution and then seed replication.

### Track B: M1, one new GPU design

1. Freeze source-session task-state strata and exact nested source-only selection.
2. Prove baseline graph, parameter count, inference, and last-bin scorer parity.
   **Stage-0 and physical-reader repair complete.**
3. Run one attempt-first CPU/source-only fold-0 audit; stop before CUDA if the
   exact common-stratum topology is absent. **V3 complete and accepted: 43
   eligible common strata, greater than the required 10.**
4. Measure one 100-step source-only throughput/memory smoke. **V4 completed all
   100 steps but failed its evidence boundary; V5 then failed before model/CUDA
   because it lost the accepted audit-spec-to-smoke rebind. Both are immutable
   and must not be retried. The sole additive V6 smoke is now complete and
   independently accepted: 100 steps, 31/31 gradient coverage, strict best/last
   reload, no target access.**
5. Build an additive seed-42 full source-training successor bound to the exact
   accepted V6 terminal; launch it only after no-data tests, current-byte and
   predecessor review, fresh-root review, and a separate explicit capability.
6. If positive, run one balanced-mean control and then replicate.
7. Review H1 only after the M1 sign is known.

The two tracks may run in parallel. An idle GPU is relevant to CS-WG, not a
reason to revive TQ-FOD or another posterior bundle.

## 8. Candidate disposition

| Candidate | Disposition |
|---|---|
| Complete CDM-D | Primary M2/SUA M4/M10 performance system |
| CS-WG on exact M1 graph | Primary new GPU system |
| CS-WG on H1 | Conditional on positive M1 result |
| Honest M-budget oracle | Post-result attribution or opportunistic diagnostic |
| Activity-only B3S extension | Post-positive CDM-D attribution |
| Carrier-only cross-fit | Post-positive CDM-D attribution |
| Repaired SFORA | Cheap optional null test |
| TQ-FOD | Do not launch under current spec |
| Larger behavior bottleneck | Closed |
| Another posterior/PIRG/PMC bundle | Closed |
| More source budget randomization | Closed |
| Same-view self-training | Reject as circular |
| More GCV tuning | Closed |

## 9. Paper-level hypothesis

The earlier binary story was unsupported:

```text
high-dimensional output -> temporal full-output decoder
short-label task         -> carrier estimator
```

The revised performance hypothesis is:

> Cross-session transfer fails because average source training and fixed-prefix
> calibration do not expose the model to the deployment distribution. We attack
> this from both sides: source-side worst-group training for transfer-limited
> M1, and causal target-time acquisition of unlabeled activity and functional
> evidence for short-prefix M2/SUA.

If both systems work, the paper contribution is not one universal decoder. It
is a bottleneck-routed transfer strategy:

- learn against worst source-session shift when target updates are unavailable;
- update closed-form calibration state when causal unlabeled target information
  is available;
- keep target backpropagation at zero in both cases.

## 10. Strongest objections

### Objection 1: CDM-D is a two-factor bundle

Correct. It is a performance system, not a one-factor mechanism test. The first
question is whether target-time causal information can beat sealed Cell D. A
positive result earns two attribution cells; a negative result closes the
bundle without a sweep.

### Objection 2: CDM-D may self-reinforce bad M4 predictions

That is why evaluation proceeds M30, M10, M4 and why each unit group is updated
from complementary units. Completed-trial displacement, agreement, canonical
direction, condition, and departure gates all fail back to sealed Cell D. A
weak M4 decoder cannot authorize its own update by score alone.

### Objection 3: CS-WG is a training recipe, not a new network

Yes. That is a feature. The evidence identifies a generalization gap and shows
that extra temporal topology can worsen it. Holding the exact graph isolates a
new source-side transfer objective while preserving parameter and deployment
cost.

### Objection 4: Worst-group training may chase behavior-mixture differences

Task-state balancing is part of the performance system for exactly this reason.
It equalizes coarse source-label movement composition before computing session
risk. It is selected source-only and disappears at inference.

## 11. Reviewer questions

The independent reviewer should answer:

1. Is CDM-D sufficiently specified as a performance system without claiming
   component attribution?
2. Does the deployment contract permit completed unlabeled query trials in the
   B3S activity memory?
3. Does snapping integrated predicted displacement to canonical direction match
   the production T4 estimand closely enough?
4. Are the existing B8 gates adequate as safety gates for CDM-D, or do they
   depend on the old estimator in a way that prevents reuse?
5. Is M30 -> M10 -> M4 the correct fail-fast order?
6. Does task-stratified worst-session risk directly target the M1 transfer gap
   without introducing a hidden target-selection path?
7. Are source-only nested folds sufficient to freeze `lambda` and `tau`?
8. Should CS-WG remain an exact-graph recipe change, or is one additional
   train-time-only invariant regularizer justified in the first system?
9. Are the post-positive attribution sets minimal enough?
10. Is any remaining fact error large enough to stop design work?

## 12. Evidence and code anchors

- `tfpd_exploration/docs/HANDOFF_TFSR_PIRG_REVIEW_20260823.md`
- `tfpd_exploration/docs/HANDOFF_TASK_FRAME_STATEFUL_READIN_20260819.md`
- `tfpd_exploration/docs/HANDOFF_M4_M10_COMPARISON_AND_OPTIMIZATION_20260824.md`
- `tfpd_exploration/docs/HANDOFF_CALIBRATION_GAP_DECOMPOSITION_20260824.md`
- `tfpd_exploration/src/calibration_gap_v1/ledger.py`
- `tfpd_exploration/results/posterior_marginalized_cell_d_score_v4/score.json`
- `sua_exploration/results/behavior_manifold_v2/e1_oracle_ceiling.json`
- `sua_exploration/results/behavior_manifold_v2/e8_readout.json`
- `SPINT-main/docs/H1_CARRIERID_QUALITY_DIAGNOSTIC_PROGRAM.md`
- `sua_exploration/docs/PSEUDO_LABEL_CARRIER_GATE_PROTOCOL_20260812.md`
- `sua_exploration/mc_maze/d_optimal_calibration_design.py`
- `sua_exploration/mc_maze/carrier_recursive_estimator.py`
- `streaming_calibration_exp/src/models/falcon_module.py`
- `SPINT-main/configs/model/falcon_m1.yaml`
- `SPINT-main/configs/model/falcon_h1_baseline.yaml`
- `SPINT-main/configs/data/falcon_m1.yaml`
- `SPINT-main/configs/data/falcon_h1_baseline.yaml`
