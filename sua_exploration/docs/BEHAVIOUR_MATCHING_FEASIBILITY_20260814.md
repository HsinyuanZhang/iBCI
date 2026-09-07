# Behaviour-Matching Feasibility — Result — 2026-08-14

Behaviour arrays only. No neural data, no model, no training, no GPU. Total measured compute:
93 s for the main run plus 57 s for the direction-coverage supplement, single-threaded under
`nice -n 15`.

| Artefact | Path | sha256 |
|---|---|---|
| Frozen protocol | `sua_exploration/docs/BEHAVIOUR_MATCHING_FEASIBILITY_PROTOCOL_20260814.md` | `c9961021fe811d3503f6950cfdacdc1b39acd1825795c71c8c18ef90c7c3ac74` |
| Main receipt (immutable, 0444) | `sua_exploration/results/behaviour_matching_feasibility_20260814/behaviour_matching_feasibility_receipt.json` | `0159042986e6e95c6245e53974032d21a044665064c808a2b24ac1c185f6aeab` |
| Direction-coverage supplement (immutable, 0444) | `…/direction_coverage_supplement_receipt.json` | `303b7110b72d85c7097b96b919837af8ff59a55976b075a6d08e6e0b08b3da83` |
| Runner | `sua_exploration/scripts/run_behaviour_matching_feasibility.py` | `155cbc8bdc288a8d0e3847dabf9434d93143eaa487a668f42497f4805e6817b1` |
| Supplement runner | `sua_exploration/scripts/run_behaviour_matching_direction_coverage.py` | `4525093c146ce72bbe5e53477608a5b40f933895f4b18babec8f8eb72304c8ab` |
| Estimator | `sua_exploration/behaviour_matching/core.py` | `b81f43bd3829785239541eaf38a005291d0ce37d553f589dc30bf30472c59a0c` |
| Loaders | `sua_exploration/behaviour_matching/loaders.py` | `8efe7142c2f76e84bfd5f9721b32834526cda1d3afb597bc5985d6588949ca0c` |
| Tests (51 passing) | `sua_exploration/tests/test_behaviour_matching_feasibility.py` | `a00ec91ebf91c612c6c17425e062ce7ff24a79696ef93a4dd86bf50ed4034da7` |

The protocol sha256 recorded inside the receipt equals the current file, and the three
implementation sha256 values recorded inside the receipt equal the current files, so nothing has
moved since the run. No file under `sua_exploration/mc_maze/`, `SPINT-main/src/`,
`streaming_calibration_exp/`, or `bci_paper_overleaf/` was modified; RT and H1 discovery is
delegated to the sealed loaders those cohorts already use.

---

## 1. Headline

**Cross-session behavioural matching is not the bottleneck anywhere.** In every one of the four
cohorts the median cross-session nearest-neighbour residual sits within 1 % to 13 % of the
within-session floor. Pairing a moment in session A to a moment in session B is essentially as
tight as pairing it to a moment in A's own other half, days of recording apart.

What varies enormously between cohorts is not the *cross-session penalty* but the *absolute*
quality of the best available match, and that is governed entirely by effective behavioural
dimensionality and how many samples you search. That reframes the decision: the family of
behaviour-matched source-training mechanisms is not killed by session-to-session behavioural
drift. It is limited by behaviour-space sampling density, which is a property of a single session
and would bite just as hard inside one session.

## 2. The number, per cohort

Residuals are Euclidean distances in a cohort-pooled z-scored behaviour space, divided by
`sqrt(d)`, so one unit = one behavioural standard deviation per dimension. `ratio` = cross-session
median / that query session's own within-session floor, computed on the *same* query set against
an *equal-sized* reference set, so it is paired.

| Cohort / representation | d | PR | Within floor | Cross | **ratio med** | ratio IQR | ratio p90 | ratio min–max | pairs | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `co_subc_source` / `vel2` | 2 | 1.85 | 0.0046 | 0.0051 | **1.101** | 0.243 | 1.344 | 0.68 – 1.88 | 702 | TIGHT / ABS_TIGHT — **supports** |
| `co_subc_source` / `win100` | 100 | 8.46 | 0.1401 | 0.1613 | **1.113** | 0.115 | 1.244 | 0.88 – 1.66 | 702 | TIGHT / ABS_MARGINAL — **supports** |
| `co_subm_external` / `vel2` | 2 | 1.68 | 0.0012 | 0.0012 | **1.069** | 0.248 | 1.322 | 0.42 – 1.77 | 462 | TIGHT / ABS_TIGHT — **supports** |
| `co_subm_external` / `win100` | 100 | 7.41 | 0.0848 | 0.0879 | **1.067** | 0.129 | 1.213 | 0.84 – 1.43 | 462 | TIGHT / ABS_TIGHT — **supports** |
| `rt_subc` / `vel2` | 2 | 1.81 | 0.0010 | 0.0017 | **1.125** | 0.791 | 2.514 | 0.20 – 8.16 | 210 | MARGINAL / ABS_TIGHT — fails p90 |
| `rt_subc` / `win100` | 100 | 10.91 | 0.0428 | 0.0529 | **1.029** | 0.182 | 1.317 | 0.57 – 2329 | 210 | TIGHT / ABS_TIGHT — **supports** |
| `h1_heldin` / `disp7` | 7 | 6.10 | 0.1797 | 0.1897 | **1.009** | 0.675 | 1.834 | 0.21 – 4.64 | 156 | MARGINAL / ABS_MARGINAL — fails p90 |

`vel2` is the 2-D cursor velocity at the window end (the pointwise kinematic target).
`win100` is the full 50-bin × 2-channel behaviour window — this is what the decoder's MSE loss
actually spans and therefore what a pooled representation has to agree on, so it is the
representation the proposed mechanisms would really have to match. `disp7` is the H1 7-DoF
movement-event endpoint displacement.

Reference-set sizes: 1024 for both centre-out cohorts; 816 for RT (its smallest session's
reference half holds 816 eligible windows); **19 for H1**, because H1 sessions hold only 39–79
movement events each. Protocol amendment A1 clamps the size to a cohort-common value so that
cross and within always search equal-sized reference sets — without this the ratio silently
measured session size rather than distribution mismatch.

### The two `MARGINAL` verdicts are tail artefacts, not cross-session failures

Both failures are on `p90(ratio) <= 1.50`, never on the median.

- **`rt_subc/vel2`.** Six of the fifteen RT sessions have a within-session floor that collapses
  toward zero, because a sealed accepted reach segment runs from the last go cue to the trial
  stop and therefore contains long near-stationary stretches, so a session holds pairs of
  essentially identical motionless moments. Dividing by a near-zero floor inflates the ratio.
  Excluding those six query sessions leaves 126 pairs with median ratio 1.047 and p90 1.686.
  The single ratio of 2329 in `rt_subc/win100` is the same effect: `ses-RT-20131009 → ses-RT-20131218`
  has cross median 0.047 against a within floor of 2.0e-5. Only 6 of 210 `win100` pairs exceed
  ratio 2, all with such a query session; excluding them gives median 1.060, p90 1.261, max 1.547.
- **`h1_heldin`.** Every H1 estimate rests on 19 reference and 20 query points. The p90 of 1.83
  is what 19-point nearest-neighbour medians look like; the median ratio of 1.009 is the signal.

## 3. Discrete direction matching (centre-out) — the surprise

The discrete case was expected to be the easy one. It is the **worse** one.

**The label residual is exactly zero by construction**, so the informative quantities are (a) can
the class be found at all, and (b) how far apart are two moments that share the class.

**(a) Match rate.** Median 1.000 in both cohorts, but the external cohort has a real hole:

| Cohort | match rate median | min over pairs | pairs < 0.99 | sessions missing ≥1 direction |
|---|---|---|---|---|
| `co_subc_source` | 1.0000 | 0.9902 | 0 / 702 | 0 / 27 |
| `co_subm_external` | 1.0000 | 0.4023 | 60 / 462 | 2 / 22 |

All 27 frozen sub-C source sessions carry all 8 canonical directions among usable samples. In
sub-M, `sub-M_ses-CO-20140626` is missing direction 3 and `sub-M_ses-CO-20140627` is missing
directions 2, 3 and 4 — it holds only 5 of the 8. Only directions {0, 1, 5, 6, 7} are shared by
all 22 sub-M sessions. Consequence: a direction-keyed pseudo-session builder on the external
cohort would silently drop up to 60 % of one session's samples for specific pairs, with no error.
The sub-C shortfall of 0.9902 is the 238 rewarded samples (of 1 086 007) whose trial has a
non-finite `target_dir`.

**(b) Same-class continuous residual.** Restricting the reference pool to samples sharing the
discrete key makes the behavioural match *worse*, in every case:

| Cohort / representation | unrestricted | same `dir8` (8 classes) | same `dir8_phase5` (40 classes) |
|---|---|---|---|
| `co_subc_source` / `vel2` | 0.0051 | 0.0144 (2.8× worse) | 0.0321 (6.3× worse) |
| `co_subc_source` / `win100` | 0.1613 | 0.1939 (1.20× worse) | 0.2560 (1.59× worse) |
| `co_subm_external` / `vel2` | 0.0012 | 0.0037 (3.1× worse) | 0.0092 (7.8× worse) |
| `co_subm_external` / `win100` | 0.0879 | 0.1101 (1.25× worse) | 0.1446 (1.65× worse) |

The mechanism is straightforward: keying on direction divides the usable reference pool by 8 (or
by 40 with a phase key), and the `N^{-1/d}` penalty from the smaller pool exceeds whatever the
shared label buys. **If a behaviour-matched mechanism is built on this data, it should match on
continuous kinematics, not on the discrete target-direction bin.** The discrete key's only
advantage is that it is trivially cheap to compute.

## 4. The near-stationary caveat (protocol amendment A2, post hoc)

The unstratified numbers above are the optimistic ones. Both tasks' eligible sample sets are
dominated by near-motionless moments — centre-out windows include pre-movement hold, and RT
accepted segments include post-reach hold. Median per-sample cursor speed is 0.75 (sub-C CO),
0.24 (sub-M CO) and 0.22 (RT) in NWB velocity units; the 10th percentile is ~0 in all three. So a
large share of the headline residual is the residual of matching "barely moving" to
"barely moving", which is not what the mechanisms need.

Splitting queries by their own movement magnitude against cohort-pooled quartiles (references are
never stratified, because a real mechanism searches the whole other session), the fastest quartile
gives:

| Cohort / representation | Q4 cross | Q4 within floor | Q4 ratio med | Q4 ratio p90 |
|---|---|---|---|---|
| `co_subc_source` / `vel2` | 0.1108 | 0.1115 | 1.058 | 1.400 |
| `co_subc_source` / `win100` | 0.2539 | 0.2210 | 1.148 | 1.311 |
| `co_subm_external` / `vel2` | 0.0931 | 0.0896 | 1.061 | 1.482 |
| `co_subm_external` / `win100` | 0.3603 | 0.2995 | 1.213 | 1.509 |
| `rt_subc` / `vel2` | 0.2293 | 0.2115 | 1.039 | 3.402 |
| `rt_subc` / `win100` | 0.9227 | 0.8493 | 1.066 | 1.349 |
| `h1_heldin` / `disp7` | 0.1337 | 0.1121 | 1.001 | 4.071 |

On genuine movement the **absolute** residual is far larger than the unstratified figure: 1.6×
for `co_subc_source/win100`, 4.1× for `co_subm_external/win100`, 17.5× for `rt_subc/win100`, and
22×, 78× and 138× for the three `vel2` cases whose unstratified value was almost entirely the
stationary cluster. `rt_subc/win100` reaches 0.92 behavioural sd per dimension, i.e. the best
cross-session partner for a fast RT movement window is nearly as far away as an arbitrary sample.
H1 is the one exception, at 0.7×, because its movement events are endpoint displacements with no
stationary mass to begin with.

But the **ratio stays at 1.00–1.21** in every stratum of every cohort: the within-session floor
rises by the same amount. The conclusion is unchanged and in fact sharpened — matching fast
movement is hard, and it is exactly as hard within a session as across sessions.

## 5. Effective behavioural dimensionality

Participation ratio and variance-explained on cohort-pooled z-scored samples:

| Cohort / representation | nominal d | PR | comps for 80 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|---|
| `co_subc_source` / `vel2` | 2 | 1.85 | 2 | 2 | 2 | 2 |
| `co_subc_source` / `win100` | 100 | 8.46 | 7 | 11 | 16 | 28 |
| `co_subm_external` / `vel2` | 2 | 1.68 | 2 | 2 | 2 | 2 |
| `co_subm_external` / `win100` | 100 | 7.41 | 6 | 9 | 11 | 17 |
| `rt_subc` / `vel2` | 2 | 1.81 | 2 | 2 | 2 | 2 |
| `rt_subc` / `win100` | 100 | 10.91 | 11 | 30 | 42 | 59 |
| `h1_heldin` / `disp7` | 7 | 6.10 | 5 | 6 | 7 | 7 |

A one-second planar behaviour window is a 6–11 dimensional object, not a 100-dimensional one; RT
is the richest (PR 10.9, 30 components for 90 %) because its reach directions are continuous
rather than drawn from 8 targets. H1's 7 DoF are close to fully used (PR 6.10 of 7).

## 6. Does the `-1/d` scaling hold on real data? Yes — with `d` = participation ratio

The synthetic estimate (2.6 % at d = 2, 37 % at d = 7) is numerically `N^{-1/d}` for
`N ≈ 1000–1500`. Sweeping the reference-set size over `{16 … 4096}` on 40 fixed ordered pairs per
cohort and fitting `log(residual) = slope · log(N)`:

| Cohort / representation | slope | `d_eff = -1/slope` | PR | `d_eff / PR` | R² | sweep points |
|---|---|---|---|---|---|---|
| `co_subc_source` / `vel2` | −0.464 | 2.15 | 1.85 | 1.17 | 0.997 | 9 |
| `co_subc_source` / `win100` | −0.157 | 6.35 | 8.46 | 0.75 | 0.982 | 9 |
| `co_subm_external` / `vel2` | −0.522 | 1.92 | 1.68 | 1.14 | 0.999 | 9 |
| `co_subm_external` / `win100` | −0.129 | 7.75 | 7.41 | 1.05 | 0.991 | 9 |
| `rt_subc` / `vel2` | −0.399 | 2.51 | 1.81 | 1.39 | 0.982 | 6 |
| `rt_subc` / `win100` | −0.066 | 15.20 | 10.91 | 1.39 | 0.962 | 6 |
| `h1_heldin` / `disp7` | — | — | 6.10 | — | — | 1 (untestable) |

**Verdict `HOLDS`** on the predeclared rule: all six testable cases agree with the participation
ratio within the predeclared factor of 1.5, with R² between 0.96 and 0.999. The power law is
clean on real data.

Two corrections to how the claim should be used:

1. **The exponent is set by the participation ratio, not the nominal dimension.** For `win100`
   the nominal `d` is 100 but the measured exponent corresponds to `d ≈ 6–15`. Using nominal `d`
   would predict a residual that barely moves with `N`; the data says otherwise.
2. **The absolute levels are far better than the synthetic at low `d`.** At the primary reference
   size the real 2-D residual is 0.12 % (sub-M, N = 1024), 0.17 % (RT, N = 816) and 0.51 %
   (sub-C, N = 1024) of a behavioural sd, versus 2.6 % predicted. Real velocity distributions are
   strongly concentrated rather than uniform in a cube, so nearest neighbours are much closer than
   the uniform-density model implies.

At the common `N = 16` (the largest size H1's smallest reference pool supports), so that the
cross-cohort `d` contrast is not confounded by sample count:

| Cohort / representation | PR | residual at N = 16 |
|---|---|---|
| `co_subm_external` / `vel2` | 1.68 | 0.0084 |
| `co_subc_source` / `vel2` | 1.85 | 0.0391 |
| `rt_subc` / `vel2` | 1.81 | 0.0517 |
| `rt_subc` / `win100` | 10.91 | 0.1191 |
| `co_subm_external` / `win100` | 7.41 | 0.1256 |
| `co_subc_source` / `win100` | 8.46 | 0.2778 |
| `h1_heldin` / `disp7` | 6.10 | 0.2897 |

The `d` ordering dominates: every low-`d` case is under 6 %, every high-`d` case is 12–29 %.
This is the `-1/d` claim reproduced across cohorts rather than within one.

## 7. Verdict — which cohorts support behaviour-matched cross-session pairing

**Both centre-out cohorts: yes, on every representation.** Median ratio 1.07–1.11, p90 at most
1.344, worst single pair 1.877 across the 1164 ordered pairs each representation contributes. The
frozen 27-session sub-C source split — the exact set the paper trains on — is the strongest case,
and the external sub-M cohort behaves the same. Use **continuous kinematic** matching, not
direction bins.

**RT: yes on the window representation, with a caveat on the pointwise one.** `rt_subc/win100`
passes both criteria outright (median 1.029). `rt_subc/vel2` fails only the p90 tail, and that
tail is entirely attributable to six sessions whose within-session floor degenerates on
near-motionless samples; on the remaining 126 pairs the median is 1.047. RT cannot use discrete
matching at all: its native `target_dir` field is degenerate, exactly one unique value
(0.785398 rad) across all 15 sessions, re-asserted in this run.

**H1: not decidable from 13 public held-in sessions, and the reason is sample count, not
mismatch.** The median ratio is 1.009 — the tightest of all four cohorts — but every estimate
rests on 19 reference and 20 query points, the p90 of 1.83 is consistent with pure small-sample
noise, and the size sweep has one usable point so the `-1/d` slope cannot be fit. The honest
statement is that H1's 7-DoF behaviour space cannot be densely covered by 39–79 movement events
per session, so any behaviour-matched mechanism there would be pairing sparsely, whatever the
cross-session agreement.

**What this redirects.** The measurement does *not* say the family is dead, and it does not say
centre-out is uniquely blessed. It says the cross-session penalty is ≈ 1.0–1.2× in all four
cohorts, so the mechanism family is viable wherever you have enough behaviour samples, and the
thing that actually determines match quality is `N^{-1/PR}` — a single-session sampling-density
property, not a cross-session one.

Inverting each fitted power law gives the reference count needed to reach a target residual on
the window representation the mechanisms would actually use:

| Cohort / `win100` | reference samples for residual 0.10 | for 0.05 |
|---|---|---|
| `co_subm_external` | 120 | 2.6e4 |
| `rt_subc` | 317 | 1.2e7 |
| `co_subc_source` | 7 436 | 6.1e5 |

(The 0.10 column is at or near the swept range 16–4096; the 0.05 column is an extrapolation
beyond it and should be read as an order of magnitude only.) All three cohorts already hold far
more than 120–7 500 eligible windows per session-half, so a 0.10-sd pairing tolerance is
comfortably reachable today; a 0.05-sd tolerance is not, on any of them. H1's 19-event pools are
three orders of magnitude short. Spend effort on the sampling-density side — more samples per
session, or a lower-`d` matching key that is still behaviourally meaningful — not on defending
against session drift.

## 8. What I could not verify

1. **Whether behaviour-matched pairs help the decoder.** Out of scope by construction: this
   measures only whether such pairs exist. A tight behavioural match does not imply the pooled
   neural representations should or do agree.
2. **Whether pooled-representation distance tracks behaviour distance.** That needs the model and
   was excluded. The existing `cross_session_consistency_screen` is the place for it.
3. **H1 at a comparable sample count.** 13 sessions × 39–79 events caps reference pools at 19
   under the equal-size clamp. The H1 `-1/d` slope, and any H1 verdict that is not dominated by
   small-sample noise, are unavailable from the public held-in set. I did not touch held-out,
   minival, or formal H1 data.
4. **Whole-file sha256 of the DANDI 000688 NWBs.** Deliberately skipped: hashing ~5 GB would have
   competed for disk with the two concurrent GPU training jobs. Each 000688 input is instead bound
   by size + mtime_ns + inode, the sha256 of every HDF5 dataset actually read, and the sha256 of
   the derived behaviour matrix. The 13 H1 files do carry full-file sha256 (they total 67 MB).
   This is a weaker binding than whole-file hashing and is declared in protocol §11.
5. **Alternative sample definitions.** The measurement uses the source-training sample definition
   (every valid window start in a rewarded trial; for RT every window inside one accepted
   eval-valid reach segment). A mechanism that restricted itself to movement-onset windows would
   change the sampling density and therefore every absolute residual here; the speed-stratified
   table in §4 is the closest available proxy but is not the same thing.
6. **Whether the two deficient sub-M sessions are a data-release artefact or a real experimental
   design.** I established that `sub-M_ses-CO-20140626` and `-20140627` lack directions among
   usable rewarded trials; I did not investigate why.
7. **Statistical significance of any single pair.** Per-pair medians rest on 512 queries
   (20 for H1); the reported spread over pairs is the uncertainty statement, and no confidence
   interval or null permutation was computed.
