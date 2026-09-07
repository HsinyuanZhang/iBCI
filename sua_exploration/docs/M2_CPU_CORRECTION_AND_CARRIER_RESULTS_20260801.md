# M2 CPU correction and carrier audit: final results

**Completed:** 2026-08-01 HKT  
**Scope:** native FALCON M2 local development data; CPU-only correction and reliability audit  
**Primary deployment endpoint:** unseen-session chronological calibration followed by a fully disjoint future query  
**Not evaluated:** hidden EvalAI/challenge test

This document is the authoritative interpretation of the batch frozen in
[`M2_CPU_CORRECTION_AND_CARRIER_AUDIT_V1.md`](M2_CPU_CORRECTION_AND_CARRIER_AUDIT_V1.md).
It supersedes pending-language about 0a--0f and any use of the contaminated
historical M33 replay as held-out evidence. It does not turn local held-out
development data into a hidden or external test.

## Executive conclusions

1. The historical M33 value commonly quoted as approximately `+0.06979` is
   withdrawn permanently. It scored from trial 0, overlapped the 33-trial
   support, and assigned an apparent endpoint to two sessions that have no
   post-M33 query.
2. A new frozen-checkpoint M33 replay is positive on the **four sessions that
   actually have a future query**: `T4-F0=+0.071980` and
   `T4-TS4=+0.066484`, both 4/4 session clusters positive. With only four
   clusters, the minimum two-sided exact p-value is `.125`; this is a useful
   local development signal, not restoration of the old six-session
   significance claim.
3. The strict six-session M24 source-exact result is also positive in the
   audited cell: `T4-F0=+0.052885`, with 5/6 sessions positive. Common epoch 9
   remains positive at `+0.046628`. Thus the positive held-out sign is not an
   artifact of independent best-checkpoint selection.
4. The internal/held-out sign reversal is real but not uniquely attributable.
   The observed held-in endpoint is one two-trial minival file with 129
   windows, whereas held-out q0 covers full 33--43-trial files and overlaps
   support; a held-in q24 endpoint does not exist. Domain, file composition,
   future horizon, and their interaction cannot be separated by this design.
5. K4's population-level or modulation-weighted `W` is reproducible, but its
   typical per-channel direction is not. At M24 chronological split, median
   flattened-W Pearson is `.6937` and modulation-weighted cosine is `.7812`,
   while median per-channel cosine is only `.1837`. The stable rotation-
   invariant components are `||W||` and `b`.
6. Direction balancing and fixed ridge regularization do not rescue the
   per-channel direction mechanism. The existing strict M24 `K4-T4` gain is
   only `+0.018987`, below the frozen `+0.03` practical margin. The conditional
   two-cell K4/KS4 GPU precision replication is therefore not launched.
7. The `+0.03` threshold is a deployment SESOI, not an empirically powered
   significance threshold. Under the observed six-session dispersion, a
   normal-approximation 80%-power MDE is about `.0955`; even using the smaller
   within-fold seed dispersion gives about `.0444`.

## Comparison and label-information contract

| Contrast | Calibration labels | Permitted interpretation |
|---|---|---|
| T4-F0 | T4 uses one target direction per support trial; F0 uses no target labels | End-to-end value of adding supervised T4 information; not label-matched |
| T4-TS4 | Same target labels and B3S width; TS4 permutes channel rows | Whether correctly attached T4 content matters |
| K4-F0 | K4 uses dense per-bin velocity; F0 does not | Utility including the extra label burden |
| K4-KS4 | Same dense labels and width; KS4 permutes full K4 rows | K4 channel-attachment mechanism |
| K4-T4 | Dense per-bin velocity versus one direction per trial | Descriptive operational comparison only; carrier form and label information are confounded |

Equal support-trial counts do not mean equal label information. No result in
this document licenses describing K4 and T4 as equally supervised.

## 0a: uncertainty and identifiability

The strict clean-SPINT comparison available for uncertainty analysis contains
fold-1 seeds 42/43/44 and six visible local held-out sessions at M24/q24. It is
not a complete fold-by-seed factorial.

- `T4-clean-SPINT` equal-session/equal-seed mean: `+0.036604`.
- Session-cluster bootstrap 95% interval: `[-0.019261,+0.102021]`.
- Descriptive within-fold seed SD: `.038835`.
- Descriptive SD of session-cluster seed averages: `.083517`.
- Six-cluster, two-sided alpha `.05`, 80%-power normal-approximation MDE:
  `.044418` using the smaller seed SD and `.095522` using the observed
  session-cluster SD.

These are conditional sensitivity calculations. They are not a universal M2
noise floor, and they do not identify `sigma_seed`, `sigma_fold`, or their
interaction. Epoch 5--12 fluctuation is autocorrelated within a run and cannot
be substituted for independent run variance. The practical `+0.03` gate may
still be retained as an engineering SESOI, but `n=3 cells` cannot make it a
well-powered inferential threshold.

Authoritative artifact:
`results/m2_uncertainty_identifiability_v1/analysis_v1/aggregate_v2.json`,
SHA-256 `e4da92084c5d0d474ec6cd15b110e857b211172e1098e206c5cd0532bfc29231`.

## 0e: corrected M33 future-query replay

All nine frozen F0/T4/TS4 checkpoints were reloaded for CPU-only test
inference. Support was trials `[0:33]`; scoring began at trial 33; every scored
50-bin history was fully after the boundary. Two sessions with exactly 33
trials were marked `ineligible_zero_query` and received no R2.

### M33-eligible four-session subset

| Arm | Equal-cell mean R2 |
|---|---:|
| F0 | 0.205823 |
| T4 | **0.277803** |
| TS4 | 0.211319 |

| Contrast | Three-cell mean | Positive session clusters | Cluster bootstrap 95% interval | Exact two-sided p |
|---|---:|---:|---:|---:|
| T4-F0 | **+0.071980** | 4/4 | `[+0.044034,+0.090244]` | .125 |
| T4-TS4 | **+0.066484** | 4/4 | `[+0.032760,+0.087133]` | .125 |

The three cell deltas for `T4-F0` are `+0.061517/+0.055583/+0.098840`;
for `T4-TS4` they are `+0.081612/+0.039011/+0.078829`. The positive signal is
consistent across the four eligible local sessions, but the endpoint is a
four-session subset and cannot satisfy a six-session `p<=.05` gate.

Authoritative artifact:
`results/m2_m33_disjoint_replay_correction_v1/aggregate_heldout.json`, SHA-256
`8d56e60825cc3c017b17764146118a0e6f04a33f273f0c6c9191e14de6da70fb`.

## 0f: M24 internal-versus-held-out sign audit

Twelve frozen-checkpoint score cells were explicitly sealed: F0/T4,
best/common epoch 9, held-in q0, held-out q0, and held-out q24. Held-in q24 was
not scored because the minival file has only two trial boundaries and leaves
no future query after M=24.

| Checkpoint policy | Endpoint | F0 | T4 | T4-F0 | Positive sessions |
|---|---|---:|---:|---:|---:|
| best | held-in q0 diagnostic | 0.596991 | 0.446797 | **-0.150194** | 0/1 |
| best | held-out q0 diagnostic | 0.186534 | 0.258502 | **+0.071968** | 6/6 |
| best | held-out q24 future | 0.173901 | 0.226786 | **+0.052885** | 5/6 |
| epoch 9 | held-in q0 diagnostic | 0.594671 | 0.389250 | **-0.205422** | 0/1 |
| epoch 9 | held-out q0 diagnostic | 0.186926 | 0.255187 | **+0.068261** | 6/6 |
| epoch 9 | held-out q24 future | 0.174518 | 0.221146 | **+0.046628** | 5/6 |

Moving held-out scoring from q0 to q24 lowers F0 by `.01263/.01241` and T4 by
`.03172/.03404` for best/epoch 9. The T4 advantage therefore shrinks by about
`.0191--.0216`, but it does not reverse. Moving from best to epoch 9 changes
held-out T4 by only `-.00332` at q0 and `-.00564` at q24; the held-in T4 score
drops by `-.05755`. Independent checkpoint selection is not the cause of the
observed endpoint sign reversal.

The strongest valid statement is deliberately limited: the T4-F0 sign changes
between the observed held-in and held-out endpoints under both checkpoint
policies. It is not valid to call this a pure session-domain effect because
domain, file composition, trial count, and evaluation horizon are not crossed.

Seal list:
`results/m2_m24_domain_query_sign_audit_v1/seal_list_12_cells_v1.json`, SHA-256
`cabb0050f2269e72e5ac6b4be7125da8675220c2ee5ba3567a1c2afdd2c8911a`.
Aggregate:
`results/m2_m24_domain_query_sign_audit_v1/aggregate_12_sealed_cells_v1.json`,
SHA-256 `1dff82eff12a678aeb6cc7351011b7db0ace90237d802d2230f8916f7aef288e`.

## 0b/0c: lag and split factorial for K4

The full grid used seven held-in-calibration sessions, M in
`8,10,12,16,20,24,28,30,32,33`, behavior leads from -100 to +200 ms, and three
split definitions. It did not read held-out/query R2 or a decoder.

### Chronological primary split

| M | Nested global lead | Flattened-W Pearson | Per-channel W cosine | Modulation-weighted cosine |
|---:|---:|---:|---:|---:|
| 20 | +40 ms | .7004 | .1548 | .7915 |
| 24 | +40 ms | .6937 | .1837 | .7812 |
| 33 | +40 ms | .7514 | .3199 | .8198 |

The original +40-ms global lag is supported from M20 onward by train-session
nested selection. It is not supported as a stable per-channel lookup: at M24,
half-to-half leave-one-trial-out lag agreement is only `.125`, median absolute
lag difference is four 20-ms bins, and Spearman correlation is `.079`.

The odd/even-trial split has much lower values than chronological at M24
(`.2734/.1088/.4765` for flattened/per-channel/weighted), but this split is
confounded by scheduled target-direction composition; T4 is rank-deficient in
all 70 odd/even-trial cells. It must not be described as a clean estimate of
temporal drift. Odd/even-block is an optimistic shared-trial/autocorrelation
upper bound (`.7451/.3186/.8073` at M24), not a deployment estimate.

## 0d: T4 component reliability

At M24 chronological split, all seven sessions are defined:

| Component statistic | Median across sessions |
|---|---:|
| `[a,c]` flattened Pearson | .6369 |
| `[a,c]` per-channel cosine | .0902 |
| `[a,c]` modulation-weighted cosine | .7011 |
| modulation magnitude `m` Pearson | .8508 |
| baseline rate `b` Pearson | .9802 |

T4 therefore has the same structural pattern as K4: global/weighted signal and
rotation-invariant magnitude/baseline are more stable than typical
channel-attached direction. At M32/M33 the T4 per-channel cosine rises to
approximately `.476`, so this is a label-budget and design-coverage boundary,
not proof that direction is never recoverable. Odd/even trial is undefined
because the scheduled directions do not provide rank-3 halves; undefined rows
were not zero-filled.

## K4 components, balancing, and naming

M24 chronological component reliability is:

- `||W||` Pearson: `.8780`;
- baseline `b` Pearson: `.9866`;
- LOSO-standardized `[||W||,b]` flattened cosine: `.9201`;
- full `W` per-channel cosine: `.1837`.

Balanced eight-bin selection is defined in only 4/7 sessions after fit-validity
checks. Against 50 fixed-seed equal-N random controls, its median change is
`+.0393` for flattened Pearson, `0` for per-channel cosine, and `-.01635` for
modulation-weighted cosine. Fixed alpha-1 ridge versus all-block OLS changes
per-channel cosine by only `+.00081`. Neither control rescues the directional
identity mechanism.

Consequently, if a future side feature drops `Wx/Wy` and retains
`[||W||,b]`, it must be named **modulation-depth + baseline-rate identity**.
It may not inherit the claim “movement-aligned directional signature.” Current
data establish component reliability, not decoder equivalence between full K4,
`[||W||,b]`, and `b` alone; equivalence would require a frozen margin and new
evaluation.

Carrier artifacts:

- raw factorial SHA-256
  `bf82a57659eda92afe627d3dbe575a806366c043a9c96a23879dd7a331da9098`;
- aggregate SHA-256
  `3d4baf5f72f94155e7af05e40f98d0f4c36ae2b1615d134a0b52a89525c36959`;
- aggregate-semantics addendum SHA-256
  `4ebccd45573e007793b69840538a46eced8eb5601762d807a07a5d7e43486800`.

## GPU continuation decision

The conditional M24 K4/KS4 precision replication is **not launched**. The
existing strict held-out result has `K4-T4=+0.018987` (5/6 positive), below the
frozen `+0.03` margin, while `K4-KS4=+0.043521` is only 4/6 positive. The CPU
factorial identifies no reliable per-channel directional mechanism to rescue.
Two more training cells would not add independent held-out sessions and would
continue development on the same six already viewed local sessions.

This stop does not erase Gate A's positive future-rate encoding result. It
separates that encoding result from a failed/indeterminate decoding mechanism.
A future `[||W||,b]` study would be a new, honestly renamed hypothesis and must
be frozen on train/held-in evidence before using new evaluation data.

Decision receipt:
`results/m2_cpu_correction_and_carrier_audit_v1/gpu_no_launch_decision_v1.json`,
SHA-256 `fb76bd3e48994bb22d820b07f601b85e6084667f783ff2590c16fb2b2cacd277`.

## Verification

- Carrier factorial: 11 focused tests passed; root independently checked the
  authoritative JSON and semantics.
- M33 correction: 15 focused tests passed; root independently reran the
  aggregate from its explicit seal list.
- M24 sign audit: 13 focused tests passed; root independently reran the
  aggregate and obtained the identical SHA-256
  `1dff82eff12a678aeb6cc7351011b7db0ace90237d802d2230f8916f7aef288e`.
- All correction and sign-audit score executions were CPU-only frozen-
  checkpoint forward passes. No backward pass, optimizer update, held-out
  checkpoint selection, or hidden EvalAI evaluation occurred.

## Final claim boundary

The native-MUA evidence now supports the following careful statement:

> On visible local FALCON M2 calibration sessions, frozen-checkpoint T4 shows a
> positive chronological future-query development signal relative to F0, and
> correctly attached T4 content beats its shuffled control on the M33-eligible
> four-session subset. The evidence is not yet a hidden-test or six-session
> multi-cell significance claim.

It does not support the following stronger statements:

- the withdrawn historical M33 number is valid;
- T4 is 6/6 stable at M24;
- the internal/held-out sign reversal has a uniquely identified cause;
- K4 improves T4 through a reliable per-channel directional identity;
- K4 and T4 have matched label information;
- `[||W||,b]` is decoder-equivalent to full K4;
- INT8 should begin despite the FP32 stability gate remaining unresolved.
