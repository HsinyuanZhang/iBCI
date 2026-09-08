# Cross-session calibration ablation protocol

Status: primary training started after numerical preflight; no final new outcome is claimed here.

## Question and primary contrasts

The user clarified that the question is the benefit of **calibrating a new
session**, starting from a separately trained decoder without target-session
calibration and successively adding activity calibration and functional
carriers. The experiment therefore has three separately source-trained arms:

| Arm | Online query spikes and RIFT | Target activity calibration | Functional carrier |
|---|---|---|---|
| `Z_NONE` | enabled | none | none |
| `B_ACTIVITY_ONLY` | enabled | enabled | absent from encoder and direct decoder input |
| `D_JOINT` | enabled | enabled | enabled in encoder conditioning and direct decoder input |

The activity-calibration increment is `B - Z`; the additional carrier increment
is `D - B`; the total increment is `D - Z`. These are conditional, sequential
increments. This design does not separate the two carrier-routing paths or
estimate carrier utility without an activity encoder.

The online decoder and its initialization are shared within each paired fold
and seed. B and D have identical randomly initialized calibration encoders.
All used encoder and decoder parameters are jointly trained on source data.
Z does not retain an unused calibration encoder merely to inflate its parameter
count. Actual parameter counts, including the small additional activity encoder,
must be reported; the initial acceptance threshold is a difference below 5%.
Historical target-selected encoder checkpoints are not initialization sources.

## Dataset and split definitions

Primary seed is 42. The complete primary grid is 33 fits: three arms for each
of four M1 folds, six H1 folds, and one M2 source/external-target split.

| Task | Outer split | Calibration | Source gradients | Source validation | Target query |
|---|---|---|---|---|---|
| M1 | Four-fold leave-one-session-out over 20120924/26/27/28 | First 10 trials | Trials `[10,310)` of the other three sessions | Trials `[310,end)` of source sessions | Trials `[10,210)` of the excluded session |
| M2 | Existing seven source sessions; four external target sessions | First 33 trials | First 80% of post-M33 whole source trials | Final 20% of post-M33 whole source trials | Existing EXT4 trials after M33 |
| H1 | Six-fold leave-one-date-out over 19250101/08/13/15/19/20 | First three available eval-valid native trials | Available trial indices `[3,-2)`, valid endpoints at stride 4 | Final two available trials, valid endpoints at stride 4 | All valid endpoints after the first three available trials |

M1 ranges and H1 ordered-trial indices are zero-based half-open. H1 records
contain 8, 13, or 15 available trials, and some native trial IDs start at 3.
The exact inventory is `results/cross_session_v1/h1_raw_trial_inventory.json`.
The H1
source-session count is determined by the excluded date, not hard-coded: the
13 sessions are distributed over six dates, including a date with three
sessions. For M2, the precise whole-trial cut, integer rounding rule, source
window inventories, and context-reset masks are sealed by preflight before
training: reserve the final `ceil(0.2 * post_M33_trial_count)` whole trials;
retain validation windows whose complete raw context starts at or after the
validation boundary. Their first scored endpoint is boundary + 49. Source
training and validation must have no shared raw context.

M1/M2 use continuous history within each query/validation segment and reset
at its start. H1 resets at each native query trial while retaining all valid
target endpoints, including those with short left-padded histories. Invalid
left padding is masked and cannot be treated as observed zero neural activity.
Target calibration and query trial sets, raw intervals, and permitted input
contexts must be disjoint. Target sessions are absent from gradient training,
source normalization, source basis fitting, and checkpoint selection.

M1 and H1 public held-out-calibration records are insufficient for this test:
their M10/M3 support uses all available trials. H1 minival records are calibration
prefixes. A root raw-array audit also found exact neural-prefix matches for two
M2 minival records; separate files and local trial-ID declarations do not prove
sample independence. These surfaces are not the new selection/evaluation sets.
The M2 audit is recorded in
`results/cross_session_v1/raw_m2_minival_prefix_audit.json`.

## Functional carrier and activity-encoder provenance

M1 uses rSyn3: fit the NMF dictionary and EMG scale on source trials `[0,310)`;
fit source neural encoding carriers on source M10 and normalize using source
carrier rows. Target M10 is transformed with those frozen source quantities.
Source validation and target query behavior do not enter source fitting.

M2 uses MOVE-T4 and a B3S-derived HoldContrast-FiLM activity encoder. B contains
activity-derived contrast but no T4 values in either input path. D opens both
T4 paths. The source-only normalization and actual M33 carrier arrays must be
bound by content hashes; legacy teacher/identity weights are not loaded.

H1 uses the C2-shaped activity encoder and a newly reconstructed H-C plan for
each outer fold: source M3 only, PCA rank 12, ridge penalty 10, source output
SVD basis, uncertainty shrinkage, and source RMS normalization. All six folds
must support fresh reconstruction, including 19250101. Existing five-fold or
all-source authorities are references, not mandatory runtime prerequisites.

The paper must identify these task-specific modules rather than call every
activity encoder literally identical B3S or every carrier directional T4.

## Optimization, selection, and execution

Use the existing task-bound RIFT architecture: M1 R100/D4 full concat; M2
R50/D4/P16; H1 R300/D4/P16, all with recency attention and the local backend.
Train M1/M2 for 24 epochs and H1 for 32, batch 32, AdamW, MSE, weight decay
0.01, gradient clipping 1, EMA 0.9995, one warmup epoch, and cosine learning
rate floor 0.1 of peak. Peaks are 1e-4 for M1/H1 and 3e-4 for M2. Unit dropout
is 0.1, using paired masks and a paired sampler. Decoder CUDA computation may
use bfloat16 while calibration encoding and loss accumulation remain float32.
Task-native target/output scaling must be verified in preflight.

All arms use the earliest maximum equal-source-session validation R2 over all
EMA epochs. Target labels are read for final scoring only. Report the selected
EMA and, as a prespecified sensitivity analysis, the common final EMA epoch.
Save every epoch's EMA parameters, one atomic resumable full checkpoint with
all RNG state, and exact source/data/initialization/sampler/checkpoint digests.
No target score may trigger extra epochs, architecture changes, or repicking.

The root agent executes all numerical operations and owns GPU scheduling.
Implementation workers do static work only. Preflight must actually check
shared initialization, parameter counts, finite forward/backward, active
gradients, arm interventions, raw data/context separation, strict EMA loading,
and resume behavior. A written plan or a self-declared flag is not a pass.
Two RTX 3090 GPUs are available. Preserve old assets and bound new disk use by
EMA-only histories plus one resume state per run. No EvalAI submissions and no
DANDI 000688 work are part of this experiment.

## Reporting and completion evidence

Retain per-target-session predictions, targets or exact target bindings, R2,
checkpoint identity, and selected source-validation epoch. Plot the three arm
scores and paired `B-Z`/`D-B` increments for M1, M2, and H1. Use a common arm
order and identify the number of sessions/folds and configured seed. H1 date
groups and repeated sessions are not independent training-seed replicates.
Do not turn across-session spread into a seed confidence interval or claim
statistical significance from one primary seed.

Completion requires all primary cells, verified artifacts and paired protocol
matching, the standalone figure, and its inclusion with accurate methods and
results in `bci_paper_overleaf/paper_4pp.tex`. Render and inspect the figure and
the resulting paper. The existing official H1 result and older overlap-surface
analyses remain separate from this new public cross-session experiment.
