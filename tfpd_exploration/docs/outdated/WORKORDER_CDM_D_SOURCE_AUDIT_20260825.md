# Work Order: CDM-D Source Adapter and Safety Audit Scaffold

Date: 2026-08-25  
Status: authorized for additive no-data/no-CUDA implementation only  
Scientific role: source-only feasibility gate for the complete CDM-D performance system

## 1. Objective

Build the route-owned source adapter, physical forward seam, source-audit
contract, and dry lifecycle for Causal Dual-Memory Cell D (CDM-D).  Bind and
reuse the accepted Stage-0 core without changing it.

Accepted Stage-0 authority:

```text
workorder SHA256:
5d4a22bf8d1700b4230f2f9970c9ff98b2e6a31d0a9bc1bd828e6d844ff1c6fc

Stage-0 closure SHA256:
3ab6d3de931e4630cb9c80b07e25e3b38af4c444f3c937d3d28560b596c29590
```

This work order authorizes code and synthetic/CPU tests only.  It does not
authorize opening NWB data, loading checkpoint tensors, creating an authority
or result root, initializing CUDA, using a GPU, scoring within/external/formal
surfaces, or launching a process.

## 2. Ownership

Create only additive files:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_audit.py
```

Do not edit Stage-0 files or shared datamodule, Cell-D, T4, scorer, checkpoint,
result, or prior-route files.  The worker is not alone in the repository;
preserve all unrelated edits.

## 3. Frozen scientific protocol

### 3.1 Source-only scope

The future safety audit may use only the strict 27 sub-C source sessions.
Source rewarded-trial labels may be read for audit comparison only.  They may
not enter a pseudo update.  Within-6, external-15, formal, target, or official
evaluation assets must remain unresolved and unopened.

The scaffold must dependency-inject all data/model access.  Public CLI paths
remain inert and fail closed.

### 3.2 Exact budget protocol

Use the already selected deployment recipes:

| Budget | Labeled support | Carrier estimator | Unlabelled first-30 pool |
|---:|---|---|---:|
| M4 | causal D-optimal 4 selected inside the first 30 rewarded trials | fixed ridge 0.1 | the other 26 |
| M10 | chronological first 10 | fixed ridge 0.1 | trials 10-29 |
| M30 | chronological first 30 | fixed ridge 0.1 | none |

Support labels are sealed before pseudo processing.  For M4, the selected
D-opt trials may be non-contiguous in the first-30 pool.  The remaining 26
trials are processed in chronological order only after the support authority
is fixed.  They are all calibration history before any governed query score.

At the start of governed scoring, the B3S activity stack must contain exactly
the same first-30 trial multiset as the existing M30-activity reference:

- M4: 4 support plus 26 completed unlabelled pool trials;
- M10: 10 support plus 20 completed unlabelled pool trials;
- M30: 30 support and zero query-memory rows.

Because B3S averages over its trial axis, require bitwise or frozen-tolerance
identity parity between this reordered first-30 multiset and the ordinary
chronological M30 activity stack before allowing a live audit.

### 3.3 Three physical trial views

Construct the accepted Stage-0 capabilities from independent source arrays:

1. B3S view: exact datamodule cubic-interpolated/padded `[100,N]` spike counts;
2. carrier view: exact native rewarded-trial binned spike counts `[T,N]`;
3. velocity-validity view: W=50 neural-window endpoints lying inside that
   rewarded trial, derived only from trial bounds and neural availability.

The source adapter must descriptor-bind raw session/trial/channel identity.
It must not authorize a native-count capability merely because a caller labels
an arbitrary array as native.  Recompute the native slice from the held source
record and prove exact bytes, interval bounds, nonnegative integer-valued spike
counts, 20-ms bins, and channel order.  The cubic B3S array must be rebuilt by
the same datamodule helper/semantics and compared to the accepted calibration
view.

### 3.4 Physical velocity direction

The sealed model predicts behavior in the datamodule's standardized coordinate
system.  Before displacement integration, restore physical velocity using the
exact sealed source behavior mean and standard deviation.  Never integrate the
z-scored output directly.

The validity mask may use only W=50 window availability and rewarded-trial
bounds.  It may not use target velocity, speed, movement, success, direction,
or an evaluation mask derived from behavior.  True source direction is joined
only after the pseudo result is final and only for audit metrics.

### 3.5 Real held-group exclusion

For group `k`, physically slice group `k` out of all three unit-aligned model
inputs:

- current neural windows;
- B3S calibration/activity stack;
- normalized T4 side features.

Do not implement exclusion by zeroing values while leaving held tokens in the
attention set.  Preserve the exact remaining channel order.  The source model
must run in eval/no-grad mode; dynamic dropout calls must be zero; repeated
forward must be bitwise equal; parameter/buffer/lazy topology and RNG state
must be unchanged.

The model output for one completed trial is the ordered set of valid last-bin
predictions at its W=50 endpoints.  The Stage-0 wrapper then couples this
trajectory to its target-label-free validity capability.

### 3.6 Source-only constructibility gate

Keep two estimands separate.

The per-trial pseudo-direction diagnostics are descriptive only.  They may
record pseudo-versus-true circular error/cosine, disagreement, movement, rank,
condition, and departure, but they do not instantiate the frozen B8 gate and
must not authorize the live route by themselves.

The governing B8 estimand must reuse the existing carrier-level construction
from `sua_exploration/mc_maze/pseudo_label_carrier_gate.py`: on a fixed set of
native rewarded-trial rate rows, fit one carrier from true direction indices,
one from the already-final pseudo direction indices, and one from the exact
deterministically shuffled pseudo indices.

CDM-D produces `K=4` complementary pseudo directions per completed trial, not
one shared direction: pseudo direction `g` is fitted only to the units in held
group `g`.  The governing gate must preserve that consumer mapping.  For each
group, fit true/pseudo/shuffled carriers on the same native rate rows using
that group's pseudo labels, retain only that group's unit rows, concatenate all
four groups back into the immutable source channel order, and then apply the
frozen thresholds to the complete per-unit `[a,c]` cosine distribution.  Use
the same frozen deterministic B8 trial permutation for every group; do not
select a favorable shuffle per group.  If all four pseudo-label columns are
identical, the grouped implementation must reduce algebraically to the frozen
single-label B8 implementation.

"Complete" here means every unit marked valid by the sealed theta authority,
not every physical channel row.  The strict-27 authority contains three
undefined-direction rows (one in `sub-C_ses-CO-20131101` and two in
`sub-C_ses-CO-20131220`).  Those rows already have complementary-group
assignment `-1`; they must remain in immutable channel-order evidence but must
not be treated as carrier-cosine observations.  Reassembly must prove that
every and only authority-valid row is assigned exactly once, that all invalid
rows remain unassigned/NaN, and must receipt-bind total/valid/invalid unit
counts plus the valid-mask digest.  It is forbidden to silently drop a valid
row, count an undefined row as failed evidence, or require all channel rows to
have a defined tuning direction.

The reported statistics remain median correct carrier cosine, median correct
minus median shuffled carrier cosine, and fraction of units with correct
carrier cosine at least `0.40`.  Do not broadcast one group's label to every
unit, apply these thresholds to raw trial-direction cosine, or replace
median-minus-median by a difference of means.

The audit rows are fixed before source execution:

- M4: the 26 chronological first-30 trials outside the sealed D-opt support;
- M10: chronological trials 10--29 outside the sealed first-10 support;
- M30: the next 30 rewarded source trials after the sealed first-30 support.

The M30 rows are a source-only constructibility screen.  They do not enter the
M30 deployment activity memory or carrier state, whose query-memory capacity
remains exactly zero.  This separate post-support set is necessary because
replaying the M30 support trials would let each current trial's true label
enter the T4 used to generate its own pseudo direction.

For each budget and source session, record without tuning:

- accepted/rejected/fallback trial counts and every typed reason;
- pseudo-versus-true circular error and cosine;
- correct pairing minus deterministic shuffled pairing;
- complementary-group disagreement distribution;
- canonical-distance, displacement, speed, rank, condition, and departure
  distributions;
- state/prefix digests and proof that current-trial labels never enter input;
- B3S first-30 parity evidence;
- full/held model-state and repeated-forward evidence.

Reuse the existing B8 carrier-level thresholds unchanged:

```text
median cosine >= 0.50
correct minus shuffle >= 0.01
fraction with cosine >= 0.40 is at least 0.50
```

Report all M30, M10, and M4 rows, but decide source viability in the fail-fast
order M30 -> M10 -> M4.  Do not choose or alter Stage-0 trajectory thresholds
from target/external results.  A weak or non-discriminative disagreement
distribution is an honest STOP, not permission to relax the gate.

## 4. Required no-data tests

At minimum cover:

1. exact Stage-0 workorder and closure binding;
2. strict source-only roster and explicit rejection of non-source surfaces;
3. exact M4 D-opt / M10 chronological / M30 chronological support identities;
4. support labels sealed before first pseudo update;
5. M4/M10/M30 first-30 activity multiset and B3S parity;
6. independent reconstruction of B3S, native-count, and velocity-validity views;
7. rewrapped/interpolated/fractional arrays rejected as native counts;
8. native rate parity with `mean(raw_counts)/0.020`;
9. standardized-to-physical velocity restoration before direction integration;
10. validity derived only from W=50 endpoints and trial bounds;
11. group slicing removes identical unit rows from neural, B3S, and T4;
12. actual CPU synthetic Cell-D-like set reader accepts variable N and gives
    finite repeated outputs without state/RNG mutation;
13. source true direction cannot be passed to the update constructor;
14. deterministic shuffle and exact B8 threshold logic;
15. exact carrier-level B8 parity against the frozen implementation, including
    proof that raw trial-direction cosine cannot stand in for carrier cosine;
16. group-aware B8 uses each held group's pseudo direction only for that
    group's units, is invariant to a joint unit/group permutation, uses one
    common frozen shuffle order, and exactly reduces to frozen B8 when all four
    pseudo columns are identical;
17. authority-invalid unit rows retain assignment `-1`, remain NaN/excluded
    from the carrier-cosine estimand, and are disclosed by exact mask/count
    evidence; every authority-valid row is reassembled exactly once;
18. a one-direction broadcast that differs from the true four-group labels is
    rejected or gives demonstrably different evidence;
19. M30 uses only post-support source audit trials and never replays its own
    labelled support rows as pseudo inputs;
20. fail-fast M30 -> M10 -> M4 decision semantics;
21. attempt-before-data lifecycle ordering for a future source execution;
22. any failure publishes at most a typed failure in a future injected root;
23. static CLI imports no Torch and performs no data/model/output/GPU action;
24. the physical forward proves Python, NumPy, and Torch/CUDA RNG state is
    unchanged rather than inferring this only from repeated output equality.

## 5. Handoff and stop boundary

Return every owned SHA, explicit closure, exact tests, compile/whitespace proof,
and remaining live-only requirements.  Stop at the no-data boundary.  Do not
run the source audit, load the sealed checkpoint, initialize CUDA, create a
receipt/root, or launch without a separate root authorization after review.
