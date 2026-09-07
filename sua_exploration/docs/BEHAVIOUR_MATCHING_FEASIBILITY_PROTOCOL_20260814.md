# Behaviour-Matching Feasibility Protocol — 2026-08-14

Status: **FROZEN before measurement.** Written and committed to disk before any residual,
floor, ratio, participation ratio, or slope was computed. Only data-loading probes
(session counts, array shapes, event counts, unique `target_dir` counts, wall-clock timing)
were run before freezing; none of those touch the quantities this protocol decides on.

Schema: `behaviour_matching_feasibility_v1`.

---

## 1. Question

A family of proposed source-training changes (cross-session consistency loss on pooled
representations; CEBRA-style "pseudo-session" unit-set mixing) all require pairing samples
**across sessions by behavioural similarity**. The family is only viable if a moment in
session A has a genuinely close behavioural neighbour in session B.

This protocol measures, per ordered session pair `(A, B)` within a cohort, the distribution of

> `d(q, NN_B(q))` for `q` drawn from A's behaviour samples,

in behaviour space only, and compares it against the **within-session floor**, the same
quantity computed with the reference set drawn from A itself at a disjoint time.

No neural data enters any measured quantity. No model, no training, no GPU.

## 2. Cohorts (priority order) and how sessions are discovered

Discovery is bound to existing frozen/sealed sources. No raw globs where a sealed index exists.

| Cohort id | Sessions | Discovery route |
|---|---|---|
| `co_subc_source` | 27 | `session_splits.train` of `sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` (the paper's frozen sub-C CO source-training set), resolved under `sua_exploration/data/dandi_000688/sub-C` |
| `co_subm_external` | 22 | `sub-M_ses-CO-*_behavior+ecephys.nwb` under `sua_exploration/data/dandi_000688/sub-M` (external cohort; no frozen source split exists for it — this is the full public CO set for sub-M) |
| `rt_subc` | 15 | `streaming_calibration_exp.src.data.rt_k4_loader.find_rt_sessions`, with **every** path additionally passed through the sealed `sua_exploration.mc_maze.rt_classical_comparators.session_name_from_nwb_path`, which raises on anything not matching `sub-C_ses-RT-*`. Count is asserted equal to `rt_classical_comparators.EXPECTED_FOLDS == 15`. |
| `h1_heldin` | 13 | `sua_exploration.mc_maze.h1_sparse_event_endpoint.index_heldin_calib(SPINT-main/data/000954)` **only**. Never a glob. The sealed index enforces the 13-session public held-in allowlist and rejects held-out / minival / formal / private path tokens. |

`data/000129/sub-Indy` is not touched and is not part of any cohort.

## 3. Behaviour samples

The sample unit is the unit a source-training mechanism would pair.

### 3.1 Centre-out cohorts (`co_subc_source`, `co_subm_external`)

Reproduced from `sua_exploration/mc_maze/multisession_datamodule.py::_load_dandi688_session_uncached`
semantics, reading only the behaviour/trial/spike-time datasets (no spike binning):

- `bin_size_ms = 20`, `window_size = 50`, `trial_result_filter = "R"`.
- Bin grid: `edges = arange(t_min, t_max + 0.02, 0.02)` where `t_min`/`t_max` are the min/max
  over **all** unit spike times (this is the datamodule's grid definition; spike times are read
  only to fix the grid, never binned).
- `cursor_vel` linearly interpolated onto bin centres, `bounds_error=False`, `fill_value=0.0`.
- Rewarded trials with `stop_bin - start_bin >= 50` are retained, chronologically.
- A **sample** is a valid window start `s` (every `s` in `[trial.start, trial.stop - 50]`),
  exactly `_compute_valid_starts`. Calibration trials are **not** excluded: this is a behaviour
  census, not a train/eval split.

Representations of sample `s`:

- `vel2` (d = 2): velocity at the window end, `behavior[s + 49]`. This is the pointwise
  kinematic target and the case the `-1/d` synthetic estimate was quoted for.
- `win100` (d = 100): the full `behavior[s : s+50]` flattened. This is what the decoder's MSE
  loss actually spans and what a pooled representation of the window encodes, so it is the
  representation the proposed mechanisms would actually have to match.
- `dir8` (discrete, 8 classes): the sample's trial `target_dir`, snapped to the nearest of the
  8 canonical multiples of pi/4. Sessions are asserted to expose exactly 8 finite unique values.
- `dir8_phase5` (discrete, <= 40 classes): `dir8` crossed with the quintile of
  `(s - trial.start) / (trial.stop - 50 - trial.start)`. This is the discrete key a real
  implementation would use, because `dir8` alone pools all movement phases.

### 3.2 RT cohort (`rt_subc`)

Loaded with the RT pipeline's own loader `rt_k4_loader.load_rt_session`:

- `covariates` = 20 ms mean-binned `cursor_vel`, in the NWB unit, unscaled.
- A **sample** is a bin index `t` with `eval_mask[t]` true and all of `t-49 .. t` inside the
  same reach segment (`k4_segment_id[t-49:t+1]` constant and `!= -1`). This guarantees the
  50-bin window is a real, contiguous, eval-valid movement window and never spans a
  missing-velocity bin.
- Representations: `vel2` and `win100`, defined as in 3.1.
- **No discrete matching.** The native `target_dir` field is degenerate on RT
  (`rt_classical_comparators.direction_degeneracy_verdict` reports
  `PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE`); the runner re-asserts the degeneracy and records
  it rather than manufacturing a discrete key.

### 3.3 H1 cohort (`h1_heldin`)

- `h1_sparse_event_endpoint.load_event_session` per indexed session.
- A **sample** is one `MovementEvent`; its behaviour is the 7-DoF endpoint `displacement`.
- Representation `disp7` (d = 7). There is no window analogue; H1 is included solely as the
  high-`d` comparison point.

## 4. Reference / query construction (identical for cross- and within-session)

For every session, samples are ordered chronologically and split at the **median index**:

- first half -> REFERENCE pool
- second half -> QUERY pool

Then, with `rng = numpy.random.default_rng(seed)` and
`seed = blake2b(f"{GLOBAL_SEED}|{cohort}|{session}|{role}").digest()[:8]` as a uint64
(`GLOBAL_SEED = 20260814`), draw without replacement:

- `m_query = 512` points from the QUERY pool,
- `m_ref = 1024` points from the REFERENCE pool.

**Amendment A1 (2026-08-14, after the H1 dry run, before any receipt was written).** The sizes
above are additionally clamped to a **cohort-common** value:
`m_ref_cohort = min(1024, min over the cohort's sessions of |REFERENCE pool|)` and
`m_query_cohort = min(512, min over the cohort's sessions of |QUERY pool|)`. Reason: H1 sessions
hold 39-79 movement events, so without the clamp a small session's within-session floor would be
computed against ~19 reference points while its cross-session residual used ~39 from a larger
session, and the ratio would silently measure reference-set size rather than distribution
mismatch. The clamp binds only on H1 (every centre-out and RT reference pool exceeds 1024). No
threshold in section 8 was changed. The dry-run H1 ratio observed before this amendment
(median 0.836) is superseded by the clamped run and is recorded here only for disclosure.

Both the cross-session and the within-session measurement use **the same query set** `Q_A` and
**equal-sized** reference sets, so the ratio below is paired and free of sample-count bias:

- cross `(A, B)`: `Q_A` against `R_B`
- within floor `A`: `Q_A` against `R_A`

The chronological (not random) half split is deliberate: a random within-session split lets a
query bin match its own temporal neighbour 20 ms away, which would drive the floor to near zero
and make every cross-session number look bad for the wrong reason. The mechanisms pair samples
that are days apart, so the floor must be temporally disjoint too. Where a pool is smaller than
the requested size (H1, and any small session), the whole pool is used and the achieved size is
recorded per session; pair statistics are only reported for pairs meeting the size floor in §7.

## 5. Normalization and distance

Per cohort and per representation, a **cohort-pooled** per-dimension mean and standard deviation
is computed over the concatenation of all sessions' samples. Dimensions sharing a normalization
group share one mean/sd: for `win100` the group is the velocity channel, so the 50 lags of a
channel are not individually rescaled. Concretely, for the centre-out and RT cohorts the two
channel statistics are accumulated (count, sum, sum of squares) over the union of all bins
covered by any sample window, pooled across the cohort's sessions, and the same two statistics
normalize both `vel2` and `win100` — these are the same physical quantity in the same unit, so
giving them one shared scale keeps the two representations directly comparable. For H1 each of
the 7 endpoint-displacement dimensions is its own group. Every sample is z-scored with those
statistics.
Distances are Euclidean in the z-scored space, so one distance unit = one behavioural standard
deviation per dimension.

Two reported forms of the residual:

- `nn_l2` — the raw L2 distance in z-space.
- `nn_per_dim = nn_l2 / sqrt(d)` — the **d-comparable** residual, "fraction of a behavioural
  standard deviation". This is the quantity comparable to the synthetic 2.6 % (d = 2) /
  37 % (d = 7) claim, and the quantity the `-1/d` scaling is stated in.

## 6. Statistics reported per ordered pair

For each ordered `(A, B)` and each representation:

- `cross_median`, `cross_iqr`, `cross_p90` of `nn_l2` over the 512 queries; and `cross_per_dim`.
- `within_median_A` (the floor), same query set, reference `R_A`.
- **`ratio = cross_median / within_median_A`** — the headline.
- `centroid_median` — median distance from query to the **mean** of `R_B`, i.e. the residual
  with no matching at all. This bounds how much matching buys.
- Discrete (centre-out only): `discrete_match_rate` (fraction of `Q_A` whose class occurs in
  `R_B`), and the class-restricted continuous residual `cross_median | same class` in `win100`
  and `vel2`, with its own within-session class-restricted floor and ratio.

Cohort summaries report median, IQR, min and max **over pairs**, never a bare grand mean.

**Amendment A2 (2026-08-14, after the RT dry run, before any receipt was written).** A post-hoc
**movement-magnitude stratification** is added as a reported diagnostic. The RT dry run showed
that the eligible sample sets are dominated by near-stationary moments (per-sample cursor speed
has a median near zero in several RT sessions, because a sealed accepted reach segment runs from
the last go cue to the trial stop and therefore includes post-reach hold time; the centre-out
trial windows likewise include pre-movement hold). An unstratified nearest-neighbour residual on
such a set largely measures how easily "barely moving" matches "barely moving", which is not the
question. Therefore, for every representation, queries are additionally split into four strata by
their own movement magnitude (2-D cursor speed at the window end for centre-out and RT, 7-DoF
displacement norm for H1) against **cohort-pooled quartile cut points**; the reference set is
never stratified, because a real mechanism searches the whole other session. Per stratum the
cross residual, the within-session floor and their ratio are reported.

This is explicitly **post hoc**: it changes no threshold and does not replace the section 8
verdict, which stays defined on the predeclared unstratified statistic. Both are reported, and
the report states plainly that the unstratified number is the optimistic one.

## 7. Guards

- A pair is scored only if both achieved sizes satisfy `m_ref >= 16` and `m_query >= 16`.
- `nn_per_dim` and `ratio` must be finite; a non-finite value fails the run.
- The z-scoring sd floor is `1e-8`; a dimension below it fails the run.
- RT sessions must pass the sealed `session_name_from_nwb_path`; H1 sessions must come from
  `index_heldin_calib` and equal the 13-name allowlist.
- Centre-out sessions must expose exactly 8 unique finite `target_dir` values.

## 8. Decision rule (predeclared, before any number was seen)

Per cohort and representation, using the **median over ordered pairs** of `ratio`, and `p90`
denoting the 90th percentile over pairs:

| Verdict | Condition |
|---|---|
| `TIGHT` | `median(ratio) <= 1.25` **and** `p90(ratio) <= 1.50` |
| `MARGINAL` | `median(ratio) <= 2.00` and not `TIGHT` |
| `LOOSE` | `median(ratio) > 2.00` |

A separate **absolute** criterion, because a ratio of 1.0 against a useless floor is still
useless:

| Absolute verdict | Condition on `median(cross_per_dim)` over pairs, at `m_ref = 1024` |
|---|---|
| `ABS_TIGHT` | `<= 0.10` (residual within 10 % of a behavioural sd) |
| `ABS_MARGINAL` | `<= 0.25` |
| `ABS_LOOSE` | `> 0.25` |

A cohort/representation **supports behaviour-matched cross-session pairing** iff it is
`TIGHT` **and** at least `ABS_MARGINAL`. Anything else is reported as not supporting it, with
the failing criterion named.

## 9. Effective behavioural dimensionality

Per cohort and representation, on the cohort-pooled z-scored samples:

- covariance eigenvalues `lambda_i`,
- participation ratio `PR = (sum lambda)^2 / sum(lambda^2)`,
- cumulative variance-explained curve and the component counts reaching 80 / 90 / 95 / 99 %.

## 10. Test of the `-1/d` scaling claim

The synthetic estimate (2.6 % at d = 2, 37 % at d = 7) is numerically `N^{-1/d}` for
`N ~ 1000-1500` reference points, so the claim to test is:

> `nn_per_dim(N) ∝ N^{-1/d}`, i.e. `slope of log(nn_per_dim) vs log(N)` equals `-1/d`.

Predeclared test: for each cohort/representation, sweep
`m_ref ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}` (truncated to what each cohort's
reference pools support), on a fixed deterministic subsample of at most **40 ordered pairs**
per cohort (the first 40 in a seeded permutation of the pair list; the full pair matrix is run
only at the primary `m_ref = 1024`). Fit `log(median nn_per_dim)` on `log(m_ref)` by least
squares; report the slope, `d_eff = -1/slope`, and compare to the nominal `d` and to `PR`.

The claim is judged:

- `HOLDS` if `d_eff` is within a factor of 1.5 of `PR` for every cohort/representation with at
  least 4 usable sweep points;
- `PARTIAL` if that holds for some but not all;
- `FAILS` otherwise.

Additionally, all four cohorts are compared at a **common** `m_ref = 16` (the largest power of
two supported by H1's smallest reference pool), so the cross-cohort `d` contrast is not
confounded by differing sample counts.

## 11. Receipt

One immutable JSON (mode 0444, `O_EXCL`, canonical sorted-key bytes) at
`sua_exploration/results/behaviour_matching_feasibility_20260814/behaviour_matching_feasibility_receipt.json`,
containing: this protocol's path and sha256, all per-pair rows, all cohort summaries, all
verdicts, the dimensionality block, the scaling block, per-session achieved sizes, the seed,
input bindings, and an environment fingerprint.

**Input binding scheme.** Two GPU training jobs are concurrently saturating this host's CPUs and
disks. Hashing every byte of ~5 GB of NWB would compete with their data loaders for I/O for no
extra evidential value, because this measurement reads only a few small datasets per file.
Therefore each input file is bound by:

- `size`, `mtime_ns`, `inode` of the file, **and**
- the sha256 of every HDF5 dataset actually read from it (dtype + shape + bytes), **and**
- the sha256 of the derived behaviour sample matrix that entered the measurement.

H1 additionally carries the full-file sha256 that `load_event_session` computes itself (those
files total 67 MB). This is recorded as a deliberate deviation from whole-file hashing and is
restated as a limitation in the report.

## 12. Out of scope

- Any neural data, model, checkpoint, training, or GPU.
- Any modification of `sua_exploration/mc_maze/**` or `SPINT-main/src/**`.
- Any modification of `bci_paper_overleaf/**`.
- Whether a behaviour-matched pair helps the decoder. This protocol only measures whether such
  a pair exists.

## 13. Compute cap

Every invocation runs under `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `nice -n 15`, `python -s`.
If the full run is projected to exceed 15 minutes it is aborted and reported as such.
