# Work Order: CDM-D Stage 0

Date: 2026-08-25  
Status: authorized for additive CPU/synthetic implementation only  
Scientific role: performance-system scaffold, not an attribution experiment

## 1. Objective

Implement the Stage-0 core of **Causal Dual-Memory Cell D (CDM-D)** described in
`HANDOFF_BOTTLENECK_ROUTED_NEXT_DESIGNS_20260825.md`.

The complete candidate combines:

1. a causal unlabeled activity memory for B3S; and
2. a complementary-unit trial-level carrier memory for production-compatible
   T4 updates.

This work order does not authorize real source data, target data, checkpoint
loading, result-root creation, CUDA, GPU work, scoring, or a launch.

## 2. Ownership and boundaries

Create only additive files under:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py
```

Do not edit shared Cell-D, B3S, T4, datamodule, scorer, result, or prior-route
files. The worker is not alone in the repository: preserve all unrelated work
and do not revert or reformat files outside this ownership.

No canonical result or authority root may be created.

## 3. Frozen Stage-0 semantics

### 3.1 Completed-trial causality

For query trial `j`, the returned state and prediction inputs may depend only on:

- the selected support trials; and
- completed query trials with index `< j`.

The update from trial `j` becomes visible only to trial `j+1`. The API must make
`predict/read state -> observe completed trial -> commit update` explicit.

### 3.2 Activity memory

The activity-memory core accepts only already preprocessed, trialized neural
activity. Stage 0 must not invent raw-NWB preprocessing.

The B3S activity view and carrier-rate view are distinct and must never be
silently reused as one tensor:

- `b3s_trial_activity` is the exact interpolated/padded `[100,N]` spike-count
  trial consumed by the existing B3S path;
- `carrier_trial_counts` is the native binned-spike `[T,N]` rewarded-trial
  view.  The production scalar firing rate is
  `mean(carrier_trial_counts, axis=time) / 0.020` in spikes/s.

The two views must bind the same session, trial identity, and channel order.
The Stage-0 API must accept them separately.  It is forbidden to estimate the
carrier rate by averaging the cubic-interpolated B3S tensor.
Use distinct typed immutable capabilities for the two views, each carrying the
trial ID and channel-order digest; a bare array or mismatched capability must
fail closed.

Requirements:

- bounded FIFO memory;
- preserve the B3S training-shape budget by freezing the maximum total activity
  stack at 30 trials.  Therefore the completed-query FIFO capacity is
  `30 - M`: 26 for M4, 20 for M10, and 0 for M30.  Capacity zero is a valid
  no-query-memory mode and must leave the M30 activity stack exactly unchanged;
- support rows are immutable and precede query rows;
- append only a complete accepted query trial;
- deterministic roster/channel order;
- exact state digest;
- no behavior label in the activity-memory API;
- rejected or invalid trials leave the state bitwise unchanged.

The output is the trial stack that a future reviewed source adapter will pass to
the existing shared B3S encoder. Do not reimplement B3S.

### 3.3 Complementary unit groups

Use exactly `K=4` groups. Build them deterministically from initial raw T4
`[a,c,m,b]` using direction `atan2(c,a)`, magnitude, and stable channel ID as the
final tie break.

Requirements:

- every valid unit appears exactly once;
- no empty group when `N>=4`;
- unit permutation plus the same channel IDs yields the same channel-to-group
  assignment;
- no query label or session/subject table is accepted;
- the held group never contributes to the pseudo trajectory used to update that
  group.

### 3.4 Trial-level pseudo direction

Input is an already completed predicted velocity trajectory `[T,2]` over the
declared rewarded-trial prediction interval and a boolean valid-prediction mask
`[T]`.  This mask must be derived from neural/window availability and trial
bounds, never from target behavior.

Compute:

```text
displacement = sum_t velocity[t] * dt over accepted movement bins
theta_raw    = atan2(displacement_y, displacement_x)
theta_index  = nearest canonical direction with deterministic circular tie break
```

Fail closed on non-finite values, too few movement bins, insufficient integrated
displacement, insufficient mean speed, or excessive distance to the nearest
canonical direction.

All thresholds are immutable constructor fields recorded in the state/receipt
payload. Stage 0 must not choose values from evaluation results.

### 3.5 Production-compatible carrier sufficient statistics

For each group and each canonical direction, maintain:

- accepted trial count;
- per-unit scalar-rate sum;
- optional per-unit squared-rate sum for diagnostics only.

One completed trial contributes one scalar firing rate per unit from the exact
native rewarded-trial count interval, divided by the frozen 0.020-second bin
width.  The velocity prediction may have fewer valid endpoints because W=50;
therefore do not pretend the B3S tensor, carrier-rate tensor, and velocity mask
are one array.  They must instead be bound to the same completed trial ID and
their respective interval/start/stop evidence must be recorded.

After an accepted update, reconstruct per-direction means and call or exactly
match the production cosine carrier estimator:

```text
rate_i(theta) = b_i + a_i cos(theta) + c_i sin(theta)
T4_i          = [a_i, c_i, magnitude_i, b_i]
```

Do not pass per-bin velocity to the carrier fit. Do not treat `[T,N]` activity
as one RLS observation. Do not replace production per-direction means with an
unlabelled sequential estimator.

The Stage-0 carrier bank must expose two explicit, non-interchangeable fit
modes:

1. `ordinary_ols_by_direction`: reproduce
   `fit_carriers_from_selected_trials`, which gives each present direction mean
   one design row regardless of its trial count;
2. `fixed_ridge_by_trial`: reproduce `fit_ridge_t4(...,
   normalized_lambda=0.1)`, in which repeated trials retain their count weight
   and the a/c penalty is `n_trials * 0.1`.

The same direction counts and rate sums are sufficient for both closed-form
systems, but the weighting and penalty are different. The implementation must
not use OLS parity as evidence for the fixed-ridge M4/M10 candidate.

### 3.6 Trust and fallback

An update is accepted only if all configured checks pass:

- trajectory quality;
- agreement among complementary predictions;
- canonical-direction proximity;
- sufficient direction/design rank;
- finite carrier;
- departure ratio within the existing B8-compatible threshold;
- minimum accepted evidence.

Any failure returns a typed reason and leaves both memories unchanged. The
public API must expose the previous ordinary point-T4 and support activity as
the exact no-update fallback.

### 3.7 System wrapper

Provide a parameter-free wrapper around an injected sealed Cell-D-like callable.
It may:

- request complementary predictions with a held-unit mask;
- construct future B3S input from the activity memory;
- construct future T4 input from the carrier memory;
- call the injected decoder for the current state.

It may not:

- own trainable parameters;
- mutate decoder/model state;
- run backward or optimizer logic;
- load checkpoints;
- import a target scorer;
- silently normalize T4 or activity;
- use the current trial update in the current trial prediction.

## 4. Mandatory tests

The focused test must cover at least:

1. exact completed-trial prefix causality for every prefix;
2. future-trial mutation invariance;
3. activity FIFO and immutable support behavior;
   this includes exact M4/M10/M30 capacities 26/20/0 and proof that M30 never
   appends a 31st activity trial;
4. activity rejection leaves state/digest unchanged;
5. group assignment completeness, balance, and permutation equivariance;
6. explicit held-group exclusion from its pseudo trajectory;
7. velocity integration, circular nearest-direction, and tie behavior;
8. non-finite/low-speed/low-displacement/short-mask rejection;
9. scalar per-unit production rate construction from native counts divided by
   0.020, plus rejection of a cubic-interpolated B3S tensor passed as the
   carrier-rate view;
10. per-direction sufficient-statistic accounting;
11. clean-label parity against `fit_carriers_from_selected_trials` on a
    synthetic table with repeated directions;
12. clean-label bitwise or frozen-tolerance parity against
    `fit_ridge_t4(..., normalized_lambda=0.1)` on a synthetic table with
    unequal repeated-direction counts;
13. proof that OLS direction-equal weighting and ridge trial-count weighting are
    not silently collapsed to one estimator;
14. rejection when either production parity is deliberately perturbed;
15. trust-region/departure freeze and exact fallback;
16. no cross-group same-label update;
17. zero new trainable parameters in the wrapper;
18. injected decoder state before/after equality;
19. global Python/NumPy/Torch RNG invariance where local deterministic RNG is
    used;
20. static dry CLI imports no Torch and performs no write or launch.

Tests must run with CUDA hidden and no real NWB/checkpoint/result access.

## 5. Required handoff from the worker

Return:

- every owned file and SHA-256;
- explicit closure path list and aggregate closure SHA;
- exact isolated pytest command and result;
- py_compile and whitespace results;
- parameter-count proof for the wrapper;
- a list of remaining live-only requirements;
- explicit disclosure that no data, checkpoint tensor, CUDA/GPU, result root,
  receipt, score, or launch occurred.

Stop at this no-data review boundary. Do not start source-smoke implementation
or any GPU activity without a follow-up work order.
