# Track-B v2: H1-excluded fair CEBRA comparator contract

**Status:** additive source-only live-adapter and authority scaffold. Its
canonical source-only smoke may open explicitly named development **source**
NWB files after the SHA/symbol gate; it cannot open a target support/query
block, import/train CEBRA, load a checkpoint, score a query, write a paper
number, use a GPU, or mint an official execution receipt. It does not supersede
the historical `TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md`.

**Owned implementation:**

- `src/track_b_v2_contract.py`
- `src/track_b_v2_live_contract.py`
- `src/track_b_v2_source_adapter.py`
- `src/track_b_v2_target_query_scaffold.py`
- `src/track_b_v2_metric_pointer_authority.py`
- `src/track_b_v2_rt_source_authority_runner.py`
- `src/track_b_v2_fixed_gpu_engineering.py`
- `scripts/run_track_b_v2_preflight.py`
- `scripts/run_track_b_v2_cpu_controls.py`
- `scripts/run_track_b_v2_source_only_smoke.py`
- `scripts/run_track_b_v2_target_query_scaffold.py`
- `scripts/run_track_b_v2_metric_pointer_dry_plan.py`
- `scripts/run_track_b_v2_root_metric_pointer_audit_attestation_dry_plan.py`
- `scripts/run_track_b_v2_target_adapter_preflight.py`
- `scripts/run_track_b_v2_rt_15fold_source_authority_plan.py`
- `scripts/run_track_b_v2_rt_source_authority.py`
- `scripts/preflight_track_b_v2_fixed_gpu_engineering.py`
- `scripts/run_track_b_v2_fixed_gpu_source_cost.py`
- `tests/test_track_b_v2_contract.py`
- `tests/test_track_b_v2_live_contract.py`
- `tests/test_track_b_v2_source_adapter.py`
- `tests/test_track_b_v2_target_query_scaffold.py`
- `tests/test_track_b_v2_metric_pointer_authority.py`
- `tests/test_track_b_v2_rt_source_authority_runner.py`
- `tests/test_track_b_v2_fixed_gpu_engineering.py`

The v2 files must not import the old comparator. This keeps historical H1/M2
bindings and the six known harness faults out of the new route.

## Frozen source-domain parity and source-only authority

`subject_m` is an external target domain, not a leave-one-sub-M-out source
task. The fair source domain is exactly the 27 strict `sub-C_ses-CO-*` train
sessions in `sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`
(SHA256 `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`).
The V9 C1 source lineage manifest
`.../t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/c1_train_val_33_manifest.json`
(SHA256 `1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb`)
must bind that same roster. Its 6 validation and 6 formal sub-C sessions are
not source CEBRA fit data. The 15 sub-M sessions stay target-only outer folds.

`track_b_v2_source_adapter` derives source file names from this frozen roster
(or from canonical RT source IDs) and has no target argument or arbitrary raw
path. The full subject-M roster is required for a source authority. A smaller
strict27 subset is legal solely with `--loader-smoke`, emits no authority, and
is non-citable/nonselecting.

The source-only bundle keeps three transforms distinct and independently
SHA-bound:

- neural CEBRA input is the parameter-free,
  dimension-agnostic `identity_float32_binned_counts` transform. A per-unit
  source mean/std or source-channel prefix mapping is forbidden because an
  unseen target has a different unit count;
- subject-M dense behavior auxiliary scaling is exactly the strict27
  `fit_behavior_stats` mean/std transform already applied by
  `load_session_with_trials`; its values, array SHAs, state SHA, and final
  normalized-output SHAs are bound. A second z-score/refit is forbidden. RT
  starts from its native loader velocity and has its separately named
  source-pooled 2-coordinate scaler, with neither target support nor target
  query in its fit; and
- source-readout embedding standardization is a separately named explicit
  identity authority. Target-support-only and hybrid embedding scalers remain
  separately receipted future target-stage requirements.

For pMUA, the source smoke executes the canonical SUA loader plus
`electrode_ids_from_units` / `pool_spikes_by_electrode` replay, rejects a
mismatch, and records input-SUA, electrode, channel-order, and output-pMUA
feature SHAs. Merely changing `signal_view` is not a pMUA implementation.

The current development-only, non-citable strict27 SUA authority bundle was
materialized with the frozen full-continuous-source exposure and without
target, CEBRA, GPU, checkpoint, or score access:
`cebra_exploration/results/track_b_v2_source_authority_20260814_strict27_sua_continuous_v2_dev/`.
Its immutable body SHAs are roster
`f812ad5dab601b230d72c91770d6e18864f77cb070d376a3fa260acd45ce4d92`,
coverage `3828e766f21f5a2b8fc5d9ebcf9e73cef9a0cf7152b7fdc25b37b60a1805f5c7`,
neural identity `e860b4a05f3b1de4d6f5f3af0da51bf9da8a7645779eadb4ff64734786db2983`,
canonical behavior authority
`53e64c55da7b839e018a3ae87e3b2979f752b41aaa3c42cda98ace7f0d0358e7`,
source embedding identity
`ebc6f09c3af9516245496c1454e9ca2a4ca059a78588aae1ca22eeb9dbd3e1cd`,
and unexecuted dual-selector plan
`7d51cfd0d2d1359acaada0a126b4f8695762c729fdbd4bd2bc9303d8fea19f3d`.
All six body/sidecar pairs are mode `0444`; this bundle is source lineage, not
a CEBRA model, live score, canonical T4 pointer, or official execution receipt.

The matching strict27 pMUA development-only bundle is
`cebra_exploration/results/track_b_v2_source_authority_20260814_strict27_pmua_continuous_v2_dev/`:
roster `1fada745ef8e171a4da779330e4d5f315273c6143ffb7e38692b047c565c5947`,
coverage `2b070292746e7119717b05fe77ceb730693d90f1498affb9f06bb860ccec9bd5`,
neural identity `47d7609e0a0dc6393d7d2c2d80b419088cff602813d88a8fd9200613522967ac`,
behavior `e3831ae8524352ba348096991131c5f653ccd8625f5bf7ffbedf76997fada8ef`,
embedding identity `87a46af556adf423178aa13a5b9484dcc175c0ed34c8875f7ecdc5d6e0abe51c`,
and unexecuted selector plan
`83c92e1d02b9d19b1f72f98265acd1b6d61166a1221b22546c7c17c12e5ab365`.
All 27 real pMUA replay records report exact equality and ascending electrode
channel order. These are likewise development source lineage only.

### Source-observation exposure disclosure

The current frozen standard CEBRA source authority is fit on each strict-27
session's full continuous raw-bin range. It is therefore **not** source-
observation-exposure matched to the A2/T4 source route, whose fit windows are
restricted to rewarded (`trial_result_filter='R'`) trial intervals. Full
continuous CEBRA bins can include non-rewarded or intertrial observations, so
the source coverage receipt sets
`source_observation_exposure_matched=false` and
`bias_direction=favors_CEBRA_accuracy`. This disclosure is distinct from the
future target M30/M50/M24 label and neural-exposure comparison; source rows
must not substitute those target counts.

A rewarded-trial-only source CEBRA sensitivity is predeclared but **not
implemented**. It may only be added after preserving native trial boundaries
and auditing CEBRA positives across boundaries. Concatenating rewarded trials
and labeling that result “matched” is prohibited. The existing immutable
development source-authority bundles are retained as historical development
lineage; they are not rewritten or relabeled as target-comparison receipts.

## Canonical adapter and scope gate

`track_b_v2_live_contract.canonical_adapter_spec` is the only starting point
for a future live adapter. It accepts exactly these static semantic bindings:

| Scope | Signal view | Canonical loader/query binding | Budget |
|---|---|---|---:|
| `subject_m` | `sua` | `eval_adaptation_dandi688.load_session_with_trials` + `build_calib_trials_for_indices`, with `multisession_datamodule.session_name_from_path` | M50 |
| `subject_m` | `pseudo_mua` | `electrode_ids_from_units` + `pool_spikes_by_electrode`, bound with the same loader/query semantics | M50 |
| `rt` | sorted-SUA only | `rt_k4_loader.load_rt_session`, `find_rt_sessions`, and `rt_classical_comparators.rt_outer_window_layout` | M24/q24 |

The gate validates `dataset`/`view` **before** it inspects a proposed path or
discovery object. It rejects H1, M2, all undeclared datasets, every raw
data-path argument, and arbitrary discovery callbacks. It binds the current
source files by path, symbol, and SHA without importing a loader. For pMUA it
also freezes float32 binned sorted-unit input, exactly-one-electrode-per-unit
assignment, sum aggregation, ascending `np.unique` electrode channels, and
float32 output. A later
sealed loader-semantics receipt must match those bindings exactly before any
source or target discovery can occur.

RT native direction is degenerate and is not a permitted CEBRA auxiliary.
There is no sparse-direction CEBRA arm in v2. All legal subject-M and RT arms
freeze `auxiliary=continuous_velocity_dense_bin_level`.

## Scope and comparison claim

Only these dataset scopes are legal:

| Dataset | Views | Carrier-equivalent target support budget |
|---|---|---:|
| `subject_m` | `sua`, `pseudo_mua` | M50 |
| `rt` | canonical RT view | M24 |

`falcon_h1`, `falcon_m2`, arbitrary paths, and all data-path arguments are
out of scope. The runner's CLI grammar therefore has no H1/M2 choices and no
input-data option.

The accuracy comparison, if authorised later, is the CEBRA
`MultiSessionSolver` family. The **accuracy-table model arm** is the standard
supported `cebra_joint_behavior`; its frozen accuracy readout estimand is
`target_support_only_standard_cebra_accuracy`. The vendored,
deployment-oriented `cebra_frozen_source_adapt` is mandatory
mechanism/sensitivity evidence, not a substitute for standard CEBRA if it
happens to score better. Within each model arm, the source-only consumer route
is mechanism/alignment and the pooled source+support route is sensitivity only.
`cebra_adapt_unaligned` remains a declared diagnostic control and is never a
best-CEBRA result. Neither model-arm nor readout-arm selection is permitted at
runtime or after target outcomes are known.

The vendored `UnifiedSolver` is represented separately. `UnifiedDataset`
concatenates units from all training sessions, fixes model width to that sum,
and its transform requires all training-session streams. A source-only fitted
unified model cannot accept an unseen target alone. v2 emits only
`UNIFIED_UNSEEN_SESSION_UNSERVABLE` with `NO_ACCURACY_EMITTED`; adding the
target to unified training or calling a transductive all-session result a
few-shot deployment result is forbidden.

## Target support/query boundary

A later live adapter must materialize a `SupportQueryAdapterContract` for each
outer fold before it may fit or score. It records the target session, a unique
source pool, the exact carrier-equivalent budget, and the following immutable
rules:

| Block | Neural in CEBRA fit | Labels in CEBRA fit | Scored |
|---|---:|---:|---:|
| target support | yes | yes, dense velocity into CEBRA encoder | no |
| target query | **no** | **no** | **yes, only target score block** |

Only the **neural-input preprocessor** is currently source-fitted: it is fit
on outer source neural inputs and is not refit on target support or query
neural activity. It is not a generic normalizer. Behavior-auxiliary scaling
and readout-embedding standardization require separate, named receipts because
their legal fit scopes differ; neither may be inferred from the neural-input
preprocessor SHA. The historical source-selector design, if inspected as
lineage, may not read either target support or target query; it is not a live
geometry-selection requirement. A live route must additionally bind exact support/query index
SHAs and prove support/query disjointness; a boolean declaration is not
score-level proof.

The future model receipt must bind a separately source-fitted
`behavior_auxiliary_scaler` (outer source dense velocity only; neither target
support nor target query velocity may fit it). Target-support dense velocity is
then allowed into CEBRA encoder fitting only *after that fixed source scaling*.
Readout embedding standardization is distinct again and follows the named
readout route: source embeddings only for mechanism/alignment; target-support
embeddings only for the standard accuracy head; and source+target-support
embeddings only for the declared hybrid sensitivity. Target-query embeddings
never fit any standardizer. Each may explicitly use identity, but only with its
own immutable identity receipt.

The index manifest therefore records all **raw observation coverage** fed to
each block, not merely output endpoints. This matters for RT: causal support
and query windows can have distinct endpoint indices while still exposing
overlapping bins. The verifier rejects any overlap in the common raw-coverage
namespace, as well as a non-chronological M50/M24 support prefix.

### Label-density fairness disclosure

Matched prefix length does **not** mean matched label information. The primary
CEBRA-Behavior auxiliary is dense bin-level continuous velocity; the T4
reference uses sparse trial-level labels. Every support/query manifest,
neural-input-preprocessor receipt, historical-selector lineage where present,
control record, and
provenance plan must bind all of the following rather than collapsing them into
one ambiguous count:

- `auxiliary=continuous_velocity_dense_bin_level`;
- `target_support_label_scalar_count` and
  `target_support_label_unique_rows`;
- T4 `label_event_count`, `label_row_count`, `label_scalar_count`, and exact
  label semantics;
- T4 and CEBRA neural support trial counts, plus CEBRA dense-label support
  trial count;
- `neural_exposure_matched=false` for subject-M M50 and its bias direction;
- `information_matched=false`;
- `bias_direction=favors_CEBRA_accuracy`.

Specifically, subject-M T4 M50 has activity calibration from its first 30
trials and 50 direction-angle annotations (the derived `[cos, sin]` form is
not two independent annotations), while standard CEBRA gets neural activity
and dense two-coordinate velocity over 50 support trials/all eligible bins.
Both `neural_exposure_matched=false` and
`label_information_matched=false` therefore favor CEBRA accuracy. This is a
disclosure, not a reason to shrink the standard CEBRA arm. RT M24 similarly
records 24 trial events separately from its actual eligible endpoint/reach-row
and coordinate-scalar counts; matched M24 does not establish equal information.

Those values are reported, never used to choose an arm, a geometry, or a
target-specific hyperparameter. Both `linear_ridge` and `knn_cosine_k3` remain
mandatory reports.

### No-data target/query successor scaffold

`track_b_v2_target_query_scaffold` is an additive, array-free successor
contract. It accepts a grammar-checked target **identifier** only; it accepts
no target path, discovery callable, execution switch, output directory, CEBRA
import, checkpoint, score function, or GPU option. It rejects H1/M2 before it
can inspect a proposed target-like object. Formal data discovery and formal
receipt minting remain explicitly false.

It binds all six independently sealed source-authority body SHAs and a
root-audited **not-minted** pointer proposal payload SHA plus independent root
audit-attestation SHA. That binding is deliberately not a metric authority:
root must still mint a new immutable pointer body+sidecar before a baseline
metric can be used. The scaffold never reopens target/formal data or legacy
bodies.

For each future outer fold it freezes these unmaterialized exact requirements:

- subject-M target support is rewarded-trial ordinals `0..49`; query trials
  are strictly after ordinal 49. It separately records T4 neural exposure 30,
  T4 one-angle labels 50, and standard CEBRA neural/dense-velocity support 50.
  This is deliberately **query-after-50**, not query-after-30. A future paired
  subject-M score must bind the exact V9 asset/view/seed commit receipt, its
  runtime `base_input_trace` SHA (the 30 activity-identity trials), and its
  runtime `query_behavior_trace`/`predictions_targets.npz` byte lineage (the
  post-50 ordered query), rather than inferring either from a session name;
- RT support is chronological trial ordinals `0..23`; query is only the
  sealed q24 eligible full causal-window layout. Its future receipt must bind
  real endpoint/reach rows and coordinate scalar counts rather than inventing
  them from M24;
- an actual future paired claim requires support/query raw coverage indices,
  exact ordered T4 and CEBRA **prediction-target raw-bin endpoints**, equal
  float32 target byte SHAs, the T4 runtime/NPZ lineage where available, and
  the exact parent two-output R² implementation/reduction authority. For V9,
  prediction targets are causal endpoints exactly
  `ordered_prediction_target_raw_bin_indices = valid_window_start_indices + 49`;
  valid starts are not score rows and cannot substitute for the endpoint
  lineage; and
- every scored embedding's entire neural receptive field must be wholly inside
  the target query raw coverage and disjoint from support. A valid endpoint
  alone is insufficient for causal RT windows.

`QueryReceptiveFieldAndTargetByteProof` is a pure, synthetic-testable
verifier for that final condition. It consumes explicit integer index sets and
SHA labels in memory, never an array or file path. Its emitted authority is
compact: support/query coverage is losslessly expressed as sorted half-open
intervals plus expansion SHA/count/first/last; prediction endpoints are
ordered SHA/count/first/last; and receptive fields are bound by the actual
vendored-alignment authority SHA, ordered RF digest, row count, width bounds,
and zero containment violations. It never serializes one JSON tuple per scored
row. The model must bind its actual per-endpoint temporal alignment; assuming a
symmetric CEBRA window is prohibited. If target bytes, causal endpoints, or
ordered rows differ, the paired claim is rejected rather than silently recast
as a matched comparison.

Query-disjointness does not by itself establish causal parity. The vendored
default `offset10-model` has `Offset(5,5)`, whereas V9/T4 scores a causal
window's behavior at `valid_start + 49`. The compact proof therefore reports
the exact model-alignment authority, a digest of future-bin counts per
prediction endpoint, their maximum, and
`causal_temporal_exposure_matched=false` with
`bias_direction=favors_CEBRA_accuracy`. The standard joint CEBRA arm remains
the accuracy-table model arm, but it may never be described as online or
latency-equivalent. A causal-alignment sensitivity is predeclared and remains
unimplemented: its architecture and alignment must be frozen before scoring;
post-hoc embedding shifts are prohibited, and it cannot replace the standard
headline arm.

### Continuous-prefix support construction

CEBRA's sklearn interface receives one continuous sequence per session. The
standard headline therefore uses a **continuous chronological target prefix**
from the canonical record start through the stop of rewarded trial 50
(subject-M) or chronological trial 24 (RT), including every intervening raw
row. It is `boundary_matched=true`, but is explicitly not trial-bin or neural-
exposure matched to T4 and carries `bias_direction=favors_CEBRA_accuracy`.
It must bind exact prefix start/stop, raw-coverage expansion SHA/count, and
the post-boundary T4 endpoint/RF/padding authority used for query scoring.

Concatenating only rewarded trial segments is forbidden: it would create
convolutional/contrastive positives across artificial trial boundaries. A
rewarded-segment-only sensitivity is only predeclared; it requires reviewed
valid-index/segment handling that prevents cross-boundary positives and may
not be silently substituted for the standard arm.

For subject-M pMUA, each outer-fold index manifest must also carry an exact
feature SHA for every source session and its target after the frozen
`electrode_ids_from_units`/`pool_spikes_by_electrode` transformation. A pMUA
manifest with a missing, extra, or malformed session feature SHA is rejected.
SUA and RT must not supply a pMUA feature map.

### Frozen readout interpretations

Target support dense labels do enter the CEBRA encoder for every legal
MultiSession arm. They have **three different, all predeclared and separately
receipted** readout roles; a successor may not leave `may_fit_on` open and pick
one after seeing query scores.

| Receipt name | Readout fit data | Target-support dense labels in encoder | in readout | Scientific use |
|---|---|---:|---:|---|
| `source_only_consumer_mechanism_alignment` | source fit only | yes | no | Mechanism/alignment decomposition: transfer the source readout to target-query embeddings; mandatory, but not the accuracy-table headline. |
| `target_support_only_standard_cebra_accuracy` | target support only | yes | yes | **Frozen accuracy-table/headline CEBRA arm**. Standard target-support-fitted CEBRA decoding, with its additional target-specific estimator and dense labels declared. |
| `source_plus_target_support_hybrid_sensitivity` | source fit + target support | yes | yes | Pooled-domain sensitivity analysis; explicitly **not** an accuracy upper bound because domain shift can make it lower or higher than either constituent route. |

All three receipts must report linear ridge and cosine kNN. Target query neural data
and labels are prohibited from both encoder and either readout fit. The
per-fold provenance plan records both support-label booleans explicitly.

The runner is forbidden from choosing a headline estimand at runtime. The
frozen accuracy-table/headline is
`target_support_only_standard_cebra_accuracy`, which gives CEBRA its standard
target-support readout while disclosing the extra target estimator and dense
auxiliary labels. The source-only route is mandatory mechanism/alignment
evidence. The hybrid remains sensitivity-only; it cannot replace either route
or be chosen as best after query scores. All three routes must be reported once
a score run is authorised.

## Source-only selection

### 2026-08-14 root decision: use fixed canonical geometry

The exhaustive selector specified below is retained as historical design
provenance, but it is no longer an executable prerequisite for Track-B v2.
One strict source-only engineering run measured the cheapest grid point
(`d=3`, `iterations=250`, seed 42) on the subject-M SUA first pseudo-target
fold.  It used exactly one CEBRA fit, excluded the held post-M query from fit,
emitted no R2 or winner, opened no outer target/formal data, and used no GPU.
Its immutable cost-only receipt is
`cebra_exploration/results/track_b_v2_actual_cpu_source_microbenchmark_sua_firstfold_d3it250_s42_v1/receipt.json`
(body SHA256
`1c9f960641d0418da089006245dbe0141769c8c07761b9004f349e9ac160cff2`).
The measured fit-and-transform wall time was `5563.348 s`; total wall time was
`5729.817 s` (`95.50 min`) and peak RSS was `6,116,484 KiB` (about `5.83
GiB`).  Even assigning that cheapest measured cost to all 324 fits gives a
`500.7 h`/`20.9 d` serial lower bound.  Iteration-linear extrapolation of the
full grid gives about `286.9 CPU days`, before charging higher dimensions.

Accordingly:

- the 27-fold x 12-geometry selector is **NO-GO on measured CPU economics**
  and must not be launched;
- the proposed 12-fit simultaneous/support-only grouped selector is also not
  adopted, because it changes source exposure from 26 full peers plus one M50
  pseudo-target to 27 M50-only streams, has unknown deployment bias, and is
  still estimated at roughly `10.6 CPU days`;
- Track-B v2 instead freezes one externally preregistered, non-selected
  geometry for both subject-M views and RT: embedding dimension `8`,
  `10,000` iterations, normalized linear-ridge lambda `0.01`, and cosine kNN
  `k=3`; and
- neither source nor target scores may change those constants.  Receipts must
  state `source_geometry_selection_performed=false` and
  `source_selector_fit_count=0`.  This is a fixed comparator, not a tuned
  CEBRA claim, and that limitation must accompany every accuracy table.

Before any outer-target execution, an independently reviewed source-only GPU
engineering smoke must establish that the exact vendored CEBRA 0.6.1 CUDA
route is executable and measure cost.  It may use `d=8`, `iterations=250`,
seed 42 solely for engineering extrapolation, but may not emit R2, a selector
candidate, or a scientific winner.  It must bind the same source/query
exclusion and offset10 authority as the CPU receipt, record exact CUDA/device
identity, wall time, peak device memory and host RSS, and keep outer target and
formal access false.  This cost smoke cannot alter the frozen `d=8`, `10,000`
scientific geometry.

### Superseded historical selector contract (not a live blocker)

The remainder of this section documents the superseded exhaustive-selector
contract so that historical scaffolds and receipts can be interpreted; it is
not an instruction to execute that grid, a prerequisite for a target fold, or
an authority to alter the fixed canonical geometry.

`SourceOnlySelectorSpec` freezes all candidates before data are read. It makes
two independent source-only selections rather than allowing a linear-ridge
winner to tune the CEBRA geometry reported for kNN:

Each inner source fold must simulate the deployable `MultiSessionSolver`
topology. The held source session becomes a **pseudo-target**, with exactly
its continuous chronological prefix through the M50/M24 trial boundary
included in the joint fit alongside the peer source sessions; it includes all
intervening rows and never concatenates rewarded segments. Its post-M query
neural activity and dense labels remain wholly out of fit and are the only
inner score block. A plain
leave-one-session-out solver fit is invalid because a solver has no fitted
encoder for the excluded held session. That superseded plan forbids transforming
an unfitted held session and `adapt=True`. Before its historical selector execution it would require
per-pseudo-target support/query index, feature, endpoint, and receptive-field
alignment authorities. These are all source-session pseudo-target blocks, not
outer target data; no outer target access may have influenced a geometry selection.

- output dimensions: `{3, 8, 16}`;
- source iterations: `{250, 1000, 2500, 10000}`;
- standardized **linear-ridge** λ: `{1e-6, 1e-4, 1e-2, 1e-1, 1}`;
- linear selection criterion: source-inner-query linear-ridge pooled R² over
  `(d, iterations, λ)`;
- kNN selection criterion: source-inner-query cosine-kNN pooled R² over
  `(d, iterations)`; λ serialises as `NOT_APPLICABLE` and must not influence
  that selection;
- required reporting: both source-selected `linear_ridge` and independently
  source-selected `knn_cosine_k3`.

One linear `SourceOnlySelectionResult` is required for every `(d, iterations,
λ)` point, and one `KnnSourceOnlySelectionResult` for every `(d, iterations)`
point. Each winner is the highest matching source-inner-query R² with a
canonical-key tie break. Both exact selected geometries must be carried into
their own control and target-query receipts. No target score field exists in
either schema.

This fixes the F15/F16 failure mode: neither iterations nor λ can remain an
unexamined inherited constant, and the non-linear decoding view cannot be
silently omitted.

## Fixed-geometry runtime-validity controls

The scientific comparator geometry is fixed, not selected: `d=8`,
`iterations=10000`, ridge lambda `.01`, and cosine-kNN `k=3`. Linear ridge and
cosine-kNN remain separate mandatory decoder reports at that shared embedding
geometry. A future runtime-validity control must bind those constants, query
exclusion, actual offset10 alignment, and the full named arm/readout
declaration; neither source nor target outcomes may modify them.

The former 3-arm × 8-seed / 24-control-measurement CPU scaffold is retained as
historical no-data control design only. It is **not** a current hard execution
instruction for 24 CPU fits at 10,000 iterations, and it may not be revived by
calling its geometry “selected.” The exact concrete runtime-control scale,
device, iteration count, and seed coverage must be frozen by root only after
the separately reviewed `d=8`, `iterations=250`, seed-42 GPU cost receipt has
established the feasible execution envelope. Until then, no control run is
authorised and no target score is interpretable as having passed a runtime gate.

The unaligned arm remains a fully reported `diagnostic_distribution_only`
design, not an eight-of-eight `<0.20` gate. The separate deranged-support hard
null remains pending a real synthetic smoke: it must deterministically derange
only target-support auxiliary rows, preserve target-support neural rows and
the auxiliary-label multiset, bind a seed-independent permutation SHA, and use
true target-query labels only for scoring. Its threshold and any future control
execution scale are both frozen before real target access, never from target
outcomes.

`run_track_b_v2_cpu_controls.py` remains fail-closed/no-data scaffolding; its
historical `3 arms × 8 seeds` shape does not itself authorise a CPU execution,
GPU execution, or target evaluation.

The future hard null is deliberately **separate**: a joint-MultiSession
synthetic CPU control must deterministically derange only target-support
auxiliary rows. It must bind a fixed nonidentity permutation SHA independently
of CEBRA seed, preserve the neural rows and label multiset exactly, and use
true target-query labels only for scoring. Its threshold is pending a real
synthetic smoke and must be frozen before any real target data are opened. It
does not replace the all-seed unaligned diagnostic distribution.

### Root-frozen first live cells and score semantics (2026-08-15)

The cost receipt is necessary but is not itself a target-execution authority.
After that receipt is reviewed, root must freeze and publish the exact
fixed-geometry synthetic-control scale and deranged-support hard-null threshold;
the resulting immutable control pair must validate before any development
target path is resolved or opened.

The first end-to-end development cells are fixed before target access:

- Subject-M primary engineering cell: view `sua`, outer fold
  `subject_m_sua_external_target_20140307`, target
  `sub-M_ses-CO-20140307`, CEBRA seed `42`.
- Subject-M paired-view engineering successor: view `pseudo_mua`, the same
  target date/session and seed `42`; it follows the SUA engineering cell by
  predeclaration, never because of the SUA target score.
- RT primary engineering cell: `rt_outer_fold_00`, its canonical authority-bound
  held target session, CEBRA seed `42`.

These cells are implementation gates, not session- or seed-population evidence.
An execute interface that accepts another fold/date/seed in their place is not
authorised. The full 15-session x 3-seed matrices remain separate future
terminal requirements and are never launched automatically from a one-cell
score.

Vendored CEBRA pads before `transform`, so the returned full-length embedding
contains edge rows whose receptive field used padding. Every decoder-fit block
must therefore be cropped independently using the fitted model's exact
`Offset(5,5)`: each source session separately, the target-support block
separately, and the target-query block separately. It is forbidden to
concatenate sessions and then crop once, or to let padded edge rows enter the
source-only, target-support-only, or hybrid ridge/kNN fit. Receipts bind the
ordered valid-row and receptive-field digests for every block and prove exact
auxiliary-label row parity.

Every scientific score uses the sealed comparator definition exactly:
`torchmetrics==1.5.1`, `R2Score(multioutput="variance_weighted")`, CPU float32,
one update/compute scope per target session. A NumPy/float64 expression may be
used only as a checked diagnostic; algebraic equivalence is not exact metric
implementation parity. Receipts bind ordered prediction and target float32
bytes, TorchMetrics version, device, dtype, and update scope. Aggregation first
forms one R2 per session, then aggregates sessions and CEBRA seeds; dense bins
are never independent aggregate samples.

### Comparator seed policy (live gap)

CEBRA is stochastic. Engineering smoke uses seed `{42}` only and is
non-citable/nonselecting. Terminal development uses exactly `{42,43,44}`;
one same-seed model bundle must score every model arm, named readout route, and
decoder, with no best-seed choice. Aggregate session then seed, never dense
bins as independent observations. Subject-M compares this CEBRA
session×seed distribution to the sealed three-seed T4 aggregate without a
false one-to-one stochastic pairing claim. RT uses the same CEBRA seed
sensitivity against its fixed reference lineage.

## Immutable authority and receipts

The legacy `load_immutable_reference_authority` interface accepts only a
regular, non-symlink, read-only JSON body with a read-only SHA256 sidecar. The
sidecar may use the standard `digest  basename` form. It remains usable only
for a genuinely single-fold authority. Its body SHA must match, scope must be
subject-M/RT, and all of the following full-precision bindings are mandatory:

- canonical reference terminal-receipt SHA;
- source-session roster SHA;
- target-support index SHA;
- target-query index SHA;
- source-fitted neural-input-preprocessor SHA;
- finite exact reference metrics.

This interface intentionally has no default reference float table. A later
live integration must identify the dataset's canonical sealed receipts and
their immutable sidecars; copying rounded values into a config is not
authority.

Known canonical bodies predate sidecars and must not be rewritten. The
source-only `canonical_metric_pointer_mint_proposal` verifies their exact
bytes/mode only and returns a **not-minted** proposal; root must independently
audit and mint a new pointer if it is ever to become authority:

- subject-M M50 T4 aggregate:
  `sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json`
  (`12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2`,
  mode `0444`, no adjacent sidecar). Its M50 summary is a cross-check only:
  the per-session×seed M50 lineage is
  `sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805/aggregate/endpoint_aggregate_torchmetrics151.json`
  (`8a5ba373169cc28237666917f21fc893003afcc9ca486aa8f4b78d3f65aca7f4`),
  filtered by `arm=shared_t4` and view. Future paired subject-M scores also
  need the sealed runtime body plus `predictions_targets.npz` target-byte
  authority for each session×seed;
- RT absolute T4d metric/query-lineage body:
  `sua_exploration/comparators/receipts/rt_classical_comparators/rt_classical_comparators_receipt.json`
  (`c51cb0ff7dadd3c40ca7861dea92d80f3457c709801ea8ec7ba3e91b9e52b042`,
  mode `0444`, no adjacent sidecar), whose exact absolute metric is
  `/results/arms/t4d_reference/mean`; the per-fold lineage lives in
  `.../rt_terminal_stage2_20260811_canonical/rt_sparse_t4d_b2_forward_reeval_v2_20260811/RT_T4D_VS_B2_D1024_FORWARD_ONLY_15FOLD_FINAL_v1.json`
  (`c36ec0e31ed913ed4e8077f9a4d9d634d53529ce037ad06af1f48d279b16820e`);
- RT stage-2 matrix aggregate:
  `sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/STAGE2_MATRIX_AGGREGATE_v1.json`
  (`bb2806953e979180c408fb55744534be6fa470d4144f210cc50917a9b1006b7d`,
  mode `0444`, no adjacent sidecar), which contains only T4d-minus comparator
  deltas and therefore is a diagnostic companion, **not** absolute T4d metric
  authority.

For these, the metric-authority builder requires a **new independent immutable
root-audited pointer receipt**. The pointer itself is body+sidecar/O_EXCL/0444
and must bind the old body's absolute path, SHA256, `0444` mode, schema field
and value, confirmed lack of adjacent sidecar, and JSON pointers that extract
the exact metrics from the verified old bytes. The builder accepts no metrics
argument or rounded literal. Because either canonical body is a **multi-fold
aggregate**, that authority is explicitly *metric-only*: it cannot claim that
one source roster, neural-input preprocessor, or support/query split represents every fold.

Each later outer fold instead needs a separate immutable, sidecar-verified
source-roster receipt, source-only-neural-input-preprocessor receipt, and full raw-coverage
index manifest. `build_live_fold_reference_binding` validates their exact
scope, fold ID, target session, roster/index/neural-preprocessor SHAs, and the exact
same root-audited aggregate-body pointer, then produces an in-memory
**one-live-fold lineage** object with no reference metric. A future sealed
result must bind both that aggregate pointer and its own per-fold lineage. It
may not use one target support/query SHA or one source roster as aggregate
authority.

`track_b_v2_metric_pointer_authority` now supplies the bounded publisher and
validator for exactly three scopes: `subject_m/sua`,
`subject_m/pseudo_mua`, and `rt`. Its dry-plan command only reads the explicit
sidecarless legacy bodies through the same-FD immutable reader and renders a
non-writing pointer template. Root's separate, explicit publish call is the
only code path that can write the new O_EXCL/0444 body+standard sidecar pair;
the reviewed successor pairs have now been minted and revalidated. They are
metric-only authorities, not CEBRA or target results.

Before that pointer may be minted, root must first mint the corresponding
canonical `root_metric_pointer_audit_attestation` **body+sidecar pair**. A
bare 64-hex “attestation SHA” is not accepted. The immutable attestation is
scope-bound and contains the exact fresh dry-plan/template SHA, canonical
legacy-body/metric/lineage evidence, current route-code and focused-test file
SHAs, and the frozen focused-test/dry-plan commands. Pointer publication and
validation re-read this specific 0444/O_NOFOLLOW attestation pair and require
byte equality to the fresh evidence before accepting it.

Each scope has exactly one current official output pathname under
`cebra_exploration/results/track_b_v2_root_metric_pointer_authority_v2/`:
`subject_m_sua_metric_pointer.json`,
`subject_m_pseudo_mua_metric_pointer.json`, or `rt_metric_pointer.json`; the
attestation has the analogous `*_root_audit_attestation.json` name. Publishing
at a caller-chosen filename, a relative/alternate alias, copy, or symlink is
rejected. The target CPU preflight delegates to the same validator, so it also
rejects any non-canonical pointer pair before touching its opaque target ID.

The prior `...authority_v1/` pairs are preserved as immutable superseded
history. They deliberately fail current validation because their attestation
bound an earlier `track_b_v2_live_contract.py` byte revision. They must never
be copied, aliased, or substituted for v2. Current pointer body SHAs are SUA
`bc965600d6288576715a68e51343b1bd1a5b66ad5456fcb5ab63f5af4acdf45c`,
pMUA `62f74eb29cd165a6f888234a693580ae3f90675fc92240dfc3a880694f943f67`,
and RT `d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6`.

The pointer's only metric authority is the canonical summary/aggregate body.
It separately binds, with `metric_authority=false`, subject-M's filtered
per-session×seed cells and RT's per-fold body plus Stage-2 delta companion.
Before a subject-M pointer is valid, it proves that the filtered
`arm=shared_t4, view` lineage has exactly 45 finite cells = 15 unique sessions
× seeds `{42,43,44}`, no missing/duplicate session×seed pair, and that its
arithmetic mean equals the summary metric extracted from the same verified
bytes. It binds an ordered
`(session_id, seed, r2, query_window_count, asset_id)` digest. Before RT is
valid, it proves folds exactly `0..14`, 15 unique sessions, and exact equality
between the ordered per-fold T4d mean and the canonical aggregate scalar. The
Stage-2 body remains diagnostic-only and can never become the absolute metric.

`build_source_only_target_adapter_preflight` consumes an actual validated
root pointer pair before it even coerces the opaque target session ID. Absent
that pair it fail-closes with no target path/discovery/array access. Even with
a valid temporary/test pair, the preflight remains CPU/source-code-only:
target discovery/open, formal access, CEBRA, score emission, GPU, and execution
receipt minting all remain false. It preserves the continuous M50/M24 prefix,
causal `valid_start + 49` endpoint rule, offset10 noncausal/favors-CEBRA
disclosure, target-support-only headline readout, and no post-hoc
geometry/seed/readout selection.

### RT full outer-fold source authority

RT has no shared two-session source authority.  Before an RT target adapter is
even eligible for integration, `build_rt_15fold_source_authority_plan` derives
the complete **15-fold leave-one-session-out topology** from the same-FD
verified per-fold T4d lineage body.  Each outer fold fixes exactly 14 RT source
session IDs and one opaque held-out target ID; the latter is never supplied to
the source loader, discovered, or opened by the source stage.  The plan binds
M24 support, the q24 eligible-window rule, and the continuous chronological
prefix through trial 24 (including intervening rows, never concatenated trial
segments).  It intentionally omits per-fold R² values and marks the lineage as
`metric_authority=false`, so target outcomes cannot route source construction
or selection.

One development source-authority bundle is required for each such fold; its
source roster receipt carries the opaque held-out ID solely as no-access
lineage. `build_rt_outer_fold_source_authority_bundle` rejects a roster with
anything other than those 14 sources or any materialization of the held-out
session. The argument-free
`run_track_b_v2_rt_15fold_source_authority_plan.py` command renders this
non-writing CPU/source-code plan only. It is not a source-NWB materializer,
CEBRA runner, score command, or official receipt mint.

For the still no-target CPU preflight, RT additionally requires this complete
plan and requires its `outer_fold_id`/opaque held-out ID to agree before any
future discovery may be considered. This is in addition to—not a replacement
for—the validated root metric-pointer pair and the later immutable per-fold
source authority pairs.

### RT isolated source-materialization gate

`track_b_v2_rt_source_authority_runner` is the only planned materializer for
the RT 15-fold development source authority. Before it derives a source path,
it validates the sealed canonical RT metric pointer
`d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6`.
The pointer must therefore still pass its root-attestation validation; an
invalid pointer blocks process launch before any source path is derived.

Each outer fold is run in a **fresh child process**, given only the exact 14
source session IDs from the immutable 15-fold plan. It has no target argument,
never loads all 15 sessions and slices one later, and records every canonical
source-path derivation. The held session must have zero derivations and zero
opens. Every successful fold writes six independent O_EXCL/0444 source
authority body+sidecar pairs plus a fold execution receipt binding: canonical
pointer SHA, 15-fold plan SHA, loader semantics/code/runtime, source NWB SHAs,
isolation proof, and CPU cost. It records zero CEBRA parameters/MAC/state,
zero target/formal access, zero GPU use, and zero score emission.

The public command exposes only the predeclared one-fold smoke
`rt_outer_fold_00`; it has no full-batch, target-data, GPU, CEBRA, or score
option. A full 15-fold loop exists only as a Python root-flag-gated continuation
after independent smoke review. A successful later full loop would produce
ordered root manifest and cost-aggregate immutable pairs binding all 15 fold
receipts; it cannot be inferred from one smoke fold.

That gate has now been exercised exactly as specified. A root-reviewed fold-00
smoke passed, after which the full loop was explicitly authorised and
completed under
`cebra_exploration/results/track_b_v2_rt_15fold_source_authority_20260814_dev/`.
Its ordered root manifest SHA is
`775212fbd800129eb32ca03a68e53089a0ac97746be748491a840f8a694d0e02`
and cost-aggregate SHA is
`66dd4a42567de87d0113f1f3c312208308e150f1643e1f5ee21776033f508b35`.
All 15 folds used separate child processes, derived exactly 14 planned source
paths, and recorded zero held-target derivations or opens. Across all folds,
source materialization took 114.849 s with 693,924 KiB maximum RSS; target,
formal, CEBRA, GPU, and score flags remained false. These receipts establish
source authority only and do not authorise a selector or target evaluation.

Both sealed body+sidecar and pointer-bound legacy body reads use one
`O_NOFOLLOW` file descriptor, `fstat` before/after, hash and parse those same
bytes, then reject a pathname identity change. This closes lstat/read TOCTOU
and open-then-rename path poisoning rather than merely checking a name before
reading it.

`write_immutable_json` uses canonical JSON, `O_EXCL`, `O_NOFOLLOW` where
available, fsync, and mode `0444`. It refuses any existing or symlink path.
`write_immutable_receipt` reserves body and sidecar together and removes its
new body if a sidecar race occurs, so a sidecar collision cannot leave an
orphan receipt body. Unified source-session unit counts are canonically sorted
*with their session IDs*, not sorted independently. The present routes can
write only no-data preflight/Unified serviceability receipts or the explicitly
marked **development source-only** bundles above. None can mint an official
live execution, model, target, score, or reference-pointer receipt.
`refuse_score_execution()` always rejects a score request.

### Canonical development target/query authority (no target data)

`track_b_v2_development_target_authority` closes a narrower receipt-graph gap
without making a target adapter executable.  The earlier no-data scaffold
accepts six source-authority SHA strings so it can support synthetic tests; it
is therefore not by itself the canonical input to a live target successor.
The new authority has no caller-supplied data or source-receipt path.  For each
scope it verifies the canonical root-minted v2 metric-pointer pair and then
reads the fixed source authority pairs through the existing same-FD,
`O_NOFOLLOW` verifier:

- subject-M SUA uses only
  `track_b_v2_source_authority_20260814_strict27_sua_continuous_v2_dev`, and
  pMUA uses the separately materialized
  `...strict27_pmua_continuous_v2_dev` bundle.  Each must prove the exact 27
  sub-C roster and its six-body source SHA graph.  The opaque sub-M target ID
  must also be one of the pointer-bound V9 15-session × three-seed lineage
  sessions; the authority binds its asset ID and identical per-seed query
  window count, but no target array or score.
- RT verifies the completed canonical 15-fold root manifest
  `775212fbd800129eb32ca03a68e53089a0ac97746be748491a840f8a694d0e02`,
  its cost aggregate
  `66dd4a42567de87d0113f1f3c312208308e150f1643e1f5ee21776033f508b35`,
  the selected fold receipt, and exactly that fold's six immutable authority
  pairs.  It rejects a target ID other than the sealed selected fold's opaque
  held-out ID and proves its 14-source roster excludes it.

The resulting in-memory plan freezes the continuous M50/M24 prefix,
post-M-only query, causal T4 endpoint rule `valid_window_start + 49`, and
offset10's actual noncausal exposure (`Offset(5,5)`, disclosure bias favoring
CEBRA).  It binds the current fixed-canonical successor/engineering closure:
shared embedding geometry `d=8`, `iterations=10000`, ridge lambda `0.01`, and
cosine-kNN `k=3`, with source and target geometry selection both false and
zero selector fits.  Linear ridge and cosine kNN remain separate mandatory
readouts but may not select separate geometries.  The old source-bundle
dual-selector member is retained only as
`historical_unexecuted_lineage_not_authorizing_fixed_geometry`; the 324-fit
selector is NO-GO on measured CPU cost and the 12-fit grouped alternative is
not adopted because its exposure differs.  It names all three readout routes,
with `target_support_only_standard_cebra_accuracy` as headline.  It cannot
resolve a target path, discover/open target or formal data, import/train CEBRA,
fit a readout, score, use a GPU, write, or mint an execution receipt.  A later
live successor must consume this plan and separately materialize the exact
support/query coverage, ordered target bytes, and per-endpoint receptive-field
proof; a declaration in this plan is not that proof.

### Development target/query materializer boundary (no target data)

`track_b_v2_development_target_materializer` is the additive successor to the
canonical development authority, not an alternative authority or a caller-SHA
interface.  Its dry-plan path first rebuilds
`track_b_v2_development_target_authority` and binds that freshly rebuilt
authority SHA; it intentionally mints no competing materializer or asset-ledger
SHA.  Only then may it look up the pointer-bound target asset.  It
does not accept a target pathname, a target-discovery callback, an execute
flag, target arrays, a CEBRA object, or a score option.

For subject-M, the dry plan independently reads the existing three immutable
byte authorities through same-FD `O_NOFOLLOW` reads and post-read pathname
identity checks: the A2 official-preflight pair
`8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd`, the
mode-`0444`, sidecarless schema-ledger body
`1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283`, and
the mode-`0444`, sidecarless frozen scope-manifest body
`68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55`.
It joins A2's 15 admissible external sessions, the manifest row, the verified
download row, and score-blind eligibility to bind exactly one canonical local
path, file-size, and SHA for the selected pointer-bound sub-M asset.  A later,
separately authorised loader must stream-hash that file from one
`O_NOFOLLOW` FD before the NWB parser, then repeat it after parsing and require
the same inode/metadata as well as the same SHA and byte count.  This protocol
does not open any NWB while building the ledger authority.

For RT, the sealed metric pointer and completed 15-fold source authority first
bound the held session, reference query lineage, 14-source topology, and an
exact 14-fold consensus local pathname/file SHA for each of the 15 assets, but
did not record byte counts.  Root subsequently supplied the 15 explicit byte
counts without directory discovery and published the canonical immutable
authority
`track_b_v2_rt_local_asset_authority_v1/rt_15session_local_asset_authority.json`
at body SHA
`771ab920531a322fbccc9a745d80cc1cb6b3325bcc469283161e2561f7da9352`.
The materializer now validates that pair, its live implementation closure, the
full parent fold graph, and the selected held-session/fold/path/size/SHA row
before accepting the no-data plan.  Its status is
`CANONICAL_DEVELOPMENT_AUTHORITY_AND_RT_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS`.
It still must not scan a directory or glob a filename, and this no-data ledger
gate does not authorize opening or parsing the target.  A future reviewed
loader must use one `O_NOFOLLOW` descriptor, verify size/SHA, and parse that
continuously held inode or a private snapshot copied from it; pre/post pathname
hashes alone remain insufficient.  The live entrypoint must be
`track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority`;
the materializer deliberately exports no caller-supplied RT ledger mapping or
pathname-reopen verifier.

The array-level verifier is synthetic/in-memory only at this stage.  It
derives one continuous prefix through M50/M24, makes every query row strictly
post-M, derives prediction endpoints exactly as `valid_window_start + 49`,
and requires every actual offset10 receptive field to be wholly inside query
and disjoint from support.  It binds exact ordered `[Q,2]` float32 target
bytes equal to the T4 reference bytes, dense-support row/scalar/unique-row
counts, the relevant runtime/query/metric lineage, and pMUA's actual target
pooling provenance.  Subject-M emits a cross-view authority over the ordered
endpoints, T4 target bytes, and query identity; SUA and pMUA must agree on it
before a paired cross-view result can be reported.  Vendored `Offset(5,5)`
expands an endpoint with the half-open offsets `[-5,+5)`: ten neural bins,
five preceding bins, the endpoint itself, and four bins strictly after it.
The receipt therefore keeps
`causal_temporal_exposure_matched=false` and bias `favors_CEBRA_accuracy`.
The materializer carries the three frozen readout roles—source-only mechanism,
target-support-only accuracy headline, and source+support hybrid sensitivity—
at the fixed d8/10,000/.01/k3 geometry.  It has no legacy selector field and
no route/decoder/geometry/seed/readout outcome choice.

For a later live Subject-M loader, hashing the canonical pathname before and
after a pathname-based NWB parse is only a hardening check, not proof of the
bytes actually consumed: an intermediate replace-and-restore could evade it.
The parser must consume the continuously held `O_NOFOLLOW`-verified inode, or
a fresh private snapshot copied from that descriptor and independently bound
before parsing.  The execution receipt must bind that consumed inode/snapshot;
pre/post pathname equality alone is explicitly insufficient.

## Safe preflight and source-only commands

The first three commands have no NWB/checkpoint argument and make no CEBRA/GPU
call. The final command is intentionally different: it opens only two explicit
strict27 **source** NWBs for a non-citable pMUA loader/replay smoke. It has no
target-session option, query option, checkpoint, CEBRA model, scoring option,
or GPU path.

```bash
python3 cebra_exploration/scripts/run_track_b_v2_preflight.py \
  --dataset subject_m --view sua

python3 cebra_exploration/scripts/run_track_b_v2_preflight.py \
  --dataset rt --unified-unseen-serviceability \
  --source-unit-count source_a=24 --source-unit-count source_b=31 \
  --target-session-id held_target --target-unit-count 37

python3 cebra_exploration/scripts/run_track_b_v2_cpu_controls.py \
  --dataset subject_m --view sua --output-dimension 8 \
  --source-iterations 10000 --normalized-lambda 0.01

/home/xinyuan/miniconda3/envs/spint/bin/python cebra_exploration/scripts/run_track_b_v2_source_only_smoke.py \
  --dataset subject_m --view pseudo_mua --loader-smoke \
  --source-session sub-C_ses-CO-20131003 \
  --source-session sub-C_ses-CO-20131022

/home/xinyuan/miniconda3/envs/spint/bin/python \
  cebra_exploration/scripts/run_track_b_v2_development_target_authority.py \
  --dataset subject_m --view sua --outer-fold-id subject_m_sua_external_target_20140307 \
  --target-session-id sub-M_ses-CO-20140307

python3 cebra_exploration/scripts/run_track_b_v2_development_target_materializer.py \
  --dataset subject_m --view sua --outer-fold-id subject_m_sua_external_target_20140307 \
  --target-session-id sub-M_ses-CO-20140307
```

`--score` is deliberately fail-closed; the source-only command emits no model,
target-query score, checkpoint, or official receipt.

## Live-integration NO-GO list

Do not authorise a real fold until all items below are met in a separate
reviewed change:

1. The successor invokes canonical subject-M SUA/pMUA and RT adapters only
   after the current source SHA/symbol receipt passes, maintains strict27
   sub-C source / sub-M target-domain parity, and records every pMUA feature
   construction SHA. It must not modify the sealed loaders.
2. Root independently audits and mints multi-body metric-only pointers: subject-M
   aggregate cross-check plus per-session×seed M50 lineage and runtime
   targets; RT absolute metric/query lineage plus per-fold T4d lineage. No old
   body, stage-2 delta, or rounded summary is authority by itself.
3. The full source-only bundle is materialized and immutable: exact neural
   identity authority, source behavior auxiliary scaler, source readout
   embedding authority, and the root-fixed canonical contract
   (`d=8`, `iterations=10000`, ridge lambda `.01`, kNN `k=3`) with source and
   target geometry selection false and zero selector fits. The historical
   dual-selector receipt is lineage only and cannot authorise the comparator.
4. Runtime-validity controls at the fixed geometry must be separately frozen
   only after the reviewed `d=8`, `iterations=250`, seed-42 GPU cost receipt
   establishes feasible device/cost bounds. The old 24-fit CPU/10,000-iteration
   scaffold is not a hard launch instruction. Any authorised control must still
   report both decoders and retain unaligned as a diagnostic distribution; the
   deranged-support hard-null threshold is frozen from synthetic smoke before
   target access.
5. Target support/query raw-window coverage, dense/sparse event-row-scalar
   label counts, subject-M 30-vs-50 neural exposure disclosure, no-query-in-fit,
   and query-only scoring are proven from live array provenance, not declared.
6. Every live score proves byte-identical ordered T4 target rows for a paired
   claim, target-query receptive fields wholly inside query/disjoint from
   support, exact parent 2-output R² implementation/reduction, and aggregates
   session then seed. Otherwise it is explicitly an unmatched-query sensitivity.
7. Per-fold model, source embedding, target-support embedding, readout,
   prediction and score receipts are O_EXCL/immutable and include all model
   arms, the three fixed readout routes, both decoders, and exact target-label
   use in encoder/readout. Score execution remains separately authorized.
8. The process rejects H1 and M2 before discovery/import; the new source-only
   route still has no target, checkpoint, CEBRA train, GPU, or scoring path.
9. A future aggregate binds the root-audited metric-only pointer(s) plus each
   fold lineage. The standard `cebra_joint_behavior` / target-support-only
   readout headline, frozen-source sensitivity, and source/hybrid readout
   roles are fixed and cannot be outcome-substituted.
10. Formal CEBRA seeds are exactly `{42,43,44}` with session×seed aggregation;
    engineering seed 42 is non-citable/nonselecting. Subject-M's 3-seed T4
    aggregate and RT's fixed lineage remain explicitly non-identically seeded
    references.

One predeclared subject-M SUA outer fold may then serve only as an end-to-end
engineering gate. It is not a citable comparator result and cannot select an
arm or hyperparameter. A comparison-table result still requires the complete,
frozen subject-M and RT fold sets.

## Additive paired Stage-P live-executor plan (no-data only; 2026-08-15)

`track_b_v2_subject_m_stagep_live_executor.py` is an additive execution-plan
scaffold, not a target loader, CEBRA runner, scorer, or receipt publisher. It
has exactly one possible paired development order: predeclared SUA
`sub-M_ses-CO-20140307`, seed 42, then the already-predeclared paired pMUA
cell for the same record/date/seed. Neither date, seed, target path, output
root, geometry, decoder, checkpoint, nor GPU may be supplied by an operator.
The planned paired completion remains a development pilot and supports no
session or seed population inference.

Each view is required to independently materialize its own strict27 source
representation, held continuous M50 support, 28-session
`cebra_joint_behavior` encoder fit, checkpoint, embedding bundle, and six
route×decoder scores. SUA and pMUA may share only a verified target record and
must exact-cross-check behavior ordering, endpoint ordering, and T4 target
float32 bytes; they may not share an encoder, checkpoint, embedding, readout,
or score. Target-support neural and dense velocity auxiliary rows enter the
serviceable joint encoder; query rows enter neither encoder nor readout fit.
Every fit/readout block uses offset10 `Offset(5,5)` and interior `5:-5`
validity, i.e. RF `[endpoint-5, endpoint+5)` with future bins `+1..+4`.
The standard arm is therefore noncausal relative to the causal T4 endpoint and
that temporal exposure mismatch is disclosed as favoring CEBRA accuracy.

The future child process must set literal `CUDA_VISIBLE_DEVICES=1` before any
Torch or CEBRA import and prove that its logical `cuda:0` resolves to the
physical UUID bound by the exact fixed d8/it250 cost receipt. There is no
caller GPU override. Each cell has literal fresh paths for an immutable start,
target materialization, encoder receipt, checkpoint pair, embedding pair, six
score pairs, terminal, and completion. The required official preflight is
pre-existing immutable input, not a fresh output. A held target asset may be
parsed only through the continuously held verified FD or its private verified
snapshot. That snapshot is never a result artifact and is unlinked after its
target receipt and derived-array hashes are independently verified; an abort
must cleanup before any completion receipt.

Before a real launch, eight separately reviewed producers remain required:

1. canonical root/start/completion O_EXCL publisher;
2. sealed strict27 source materializer with pMUA replay;
3. held-target same-FD private-snapshot parser/materializer;
4. isolated GPU/CVD physical-identity producer;
5. independent 28-session primary joint encoder fit;
6. checkpoint/embedding immutable persistence plus same-byte reload proof;
7. six-route×decoder scorer using TorchMetrics 1.5.1 CPU float32; and
8. capability-bound live validators, terminals, SUA→pMUA parity, and paired completion publisher.

The capability is constructed only by rebuilding the exact current
`build_stagep_live_admission` result for both views. It binds the two official
preflights, one root authorization, one runtime-control pair, the cost pair,
and implementation closure. The public `--execute` path checks fresh planned
outputs and those immutable admissions, then deliberately refuses before
target path resolution, snapshot creation, Torch/CEBRA import, GPU use,
parser call, fit, score, or any write. Synthetic layout tests and existing
synthetic-only receipt validators are explicitly not live authorization.
