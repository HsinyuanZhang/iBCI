# Work order: M1 EMG-Syn3 Functional Carrier Memory, one-fold V1

**Date:** 2026-09-02  
**Status:** frozen scientific and implementation contract; live execution is
conditional on Stage-0 acceptance and a separate opaque capability  
**Design authority:**
`tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md`  
**Design SHA-256:**
`766cb7952913fc0ee57f231bc398b1d50b4d768f4d361e77ca185c61ee6a8259`

This work order authorizes implementation and CPU Stage 0. It pre-registers one
GPU one-fold three-arm pilot, but does not by itself authorize opening data,
reserving a canonical result root or launching a GPU process. Those actions
require the exact gates in this document.

## 1. Scientific question and estimands

The primary question is whether an independently injected, source-frozen,
bin-level nonnegative EMG functional carrier adds cross-session M1 information
beyond a topology-matched Zero4 carrier.

Primary estimand:

```text
S-Fix − Z-Fix
```

Secondary predeclared estimands:

```text
S-Acyc − S-Fix
CDM-A − Static within each trained arm
(CDM-A gain in S-Fix) − (CDM-A gain in Z-Fix)
```

`S-Acyc` is descriptive unless the static primary carrier-content gate passes.
No CQC, mechanism-control or full-fold launch follows automatically.

## 2. Exact M1 development scope

The one-fold pilot is fixed to:

```text
task: FALCON M1
outer fold: 0
seed: 42
source sessions: ses-20120926, ses-20120927, ses-20120928
left-out development target: ses-20120924
target support trials: [0,10)
target report query: [10,210)
formal/official/minival access: forbidden
target backpropagation/optimizer/update: 0
```

The provider must resolve exact allow-listed held-in-calib source files without
a broad recursive discovery. Paths containing `minival`, `held-out`, `formal`,
`EvalAI` or `test` reject before data materialization. Source and target file
body digests, dev/inode identities, trial boundaries and query-window digest
must be recorded.

## 3. Source-frozen EMG-Syn3

### 3.1 Qualified signal view

Use movement-window EMG aligned to fixed 20-ms bins. Before NNMF, Stage 0 must
report per source session and channel:

- finite count, minimum, maximum, zero fraction and negative fraction;
- sampling/time-alignment law;
- rectification/envelope and resampling provenance;
- valid bins, valid seconds and contributing trials.

If the stored signal is signed, direct NNMF rejects. A rectification/envelope
operator may be introduced only if it is deterministic and source-frozen in
this work order before target scoring. Target-derived offsets or shifts are
forbidden.

### 3.2 Positive scaling

For each EMG channel, compute source-only RMS over the qualified source bins:

```text
scale_j = max(sqrt(mean_source(emg_j^2)), 1e-8)
x_j = emg_j / scale_j
```

No mean subtraction is allowed. The target uses the exact source scales.

### 3.3 Deterministic source NNMF

Fit the source matrix `X_source [bins,emg_channels]` with rank 3:

```text
X_source ~= Z_source @ H_source
Z_source >= 0
H_source >= 0
```

Freeze the following before any target score:

```text
rank = 3
loss = Frobenius
solver = coordinate descent
initialization = NNDSVDa
random_state = 42
tolerance = 1e-5
maximum_iterations = 1000
L1 regularization = 0
L2 regularization = 0
```

Record the exact library/version and fail closed if the solver does not
converge. Normalize each dictionary row to unit L2 norm and compensate its
activation column. Order components by descending source activation energy;
ties use the lexicographic hash of the normalized dictionary row. Bind source
scales, raw/normalized dictionary, activations, ordering and reconstruction
digests.

Target activations are obtained with the frozen dictionary by a deterministic
nonnegative least-squares transform. The target cannot refit, rotate, reorder
or rescale the source dictionary.

### 3.4 Bin-level per-unit encoding

For each session, support budget and unit, align 20-ms firing rate `r_i(t)` with
the qualified EMG activation `z(t)`. Rows cannot cross trials or the support
boundary. Any neural/EMG lag is fixed from source-only authority; the V1
default is exact recorded-time alignment with lag 0 unless Stage 0 fails that
physical alignment contract before any decoder R² is available.

Fit:

```text
r_i(t) = b_i + W_i^T z(t) + epsilon
carrier_i = [W_i1,W_i2,W_i3,b_i]
```

Use the valid-bin-normalized ridge objective:

```text
(1/n) * ||r - b - ZW||^2 + 1.0 * ||W||^2
```

The intercept is unpenalized. This normalization keeps the meaning of
`lambda=1` fixed when the number of bins changes across M10/M6/M4/M2.

Fit and disclose M10, M6, M4 and diagnostic M2. M2 is not rejected because it
has two trials; it is judged by actual bin-level design and coverage evidence.

### 3.5 Coverage and reliability evidence

For every fold/budget/session, publish:

- valid bins, duration and trial count;
- four-column design rank and condition number;
- normalized-Gram eigenvalues and smallest eigenvalue;
- per-synergy dispersion and trial-stratified occupancy;
- finite carrier count, coefficient norms and carrier digest;
- trial-stratified split-half correlation of `W` and intercept;
- corresponding historical trial-mean PCA-AFC4 reliability evidence.

This table is disclosure, not a decoder-performance gate. Poor split-half
stability does not silently block the predeclared pilot. Nonfinite/undefined
fits, unqualified signal view, leakage or chronology failure do block it.

## 4. Carrier normalization and controls

Fit the four-coordinate carrier normalizer from source-session carriers only.
Target uses the exact frozen normalizer. Build:

```text
Zero4: four exact normalized zeros; no target carrier fit
Syn3: correctly attached normalized EMG-Syn3
RS4: deterministic complete source-unit row shuffle
LS4: deterministic support EMG/neural association derangement before refit
B4: normalized baseline coordinate only, three synergy coordinates zero
PCA3: matched bin-level signed-PCA basis under the same independent injection
```

The work order pre-registers conditional matched-training arms `S-RS4`,
`S-LS4`, `S-B4` and `S-PCA3`. Each uses the exact Stage-1 recipe and may launch
only if the static primary gate passes. Inference-only substitution is
descriptive and never satisfies a mechanism control.

There is no trial-mean Syn3 arm. Therefore no result from this work order may
claim that bin-level fitting is better than trial-mean fitting.

## 5. Independent carrier injection

The model retains the complete B3S activity identity `E^A`. Add one route-owned
carrier projection:

```text
P = Linear(4, model_dim, bias=False)
u_i = fc_in(x_i + E_i^A) + P(carrier_i)
```

Requirements:

1. `P.weight` is exact zero at common initialization.
2. Z-Fix/S-Fix/S-Acyc have byte-identical common active state before the first
   optimizer step.
3. Zero4 produces exact-zero `P(carrier)`.
4. Carrier and neural/activity branches share unit order, correspondence
   permutation and the exact same whole-unit dropout mask.
5. A dropped unit cannot retain an unmasked carrier token.
6. C1 changes only `E^A` input prefix; the M10 carrier digest is constant.
7. CDM-A updates only next-trial activity state; carrier/model state is constant.
8. No MLP, bias, gate, FiLM, carrier attention or target-learned encoder exists.

The receipt separately binds activity identity, raw/normalized carrier,
projected carrier token, common unit mask and prediction digests.

## 6. Stage-1 three-arm training contract

Train exactly:

| arm | carrier | activity prefix |
|---|---|---|
| `Z-Fix` | Zero4 | fixed M10 |
| `S-Fix` | Syn3 M10 | fixed M10 |
| `S-Acyc` | Syn3 M10 | step cycle `(10,5,2)` |

All arms use:

```text
fold = 0
seed = 42
epochs = 12 exactly
selected checkpoint = fixed last epoch_011
loss = task-only MSE
optimizer = Adam
learning rate = constant 1e-4
weight decay = 0
scheduler = none
batch size = 32
num_workers = 0
SWA = none
target validation during training = none
```

Initialize the common decoder from the reviewed source-only M1 checkpoint:

```text
streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/
2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/
checkpoints/best_ckpt/epoch_018.ckpt
sha256 = f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be
```

The implementation must descriptor-verify the exact checkpoint and strict-load
its expected state before CUDA/model training. If current immutable authority
provides a stronger body/state digest pair, the work order requires a reviewed
literal amendment before launch; runtime discovery is forbidden.

The selected checkpoint cannot read the left-out query. Source loss and fixed
epoch are training evidence; target R² is report-only.

## 7. Paired deployment scoring

Every trained checkpoint is scored under:

```text
Static: fixed M10 B3S identity + fixed M10 carrier
CDM-A: causal activity FIFO initialized at M10 + same fixed M10 carrier
```

CDM-A appends one completed query trial only after its score and changes state
starting from the next trial. It performs zero backward/optimizer/model update.
The carrier never changes. Use the governing variance-weighted last-bin R² and
the exact same ordered query windows for every arm/deployment.

Report six cells, per-session prediction/target digests, paired deltas, model
state before/after and target update counters.

## 8. Predeclared decisions

### 8.1 Static carrier-content gate

Open matched mechanism-control training and any CQC design only if:

```text
S-Fix − Z-Fix >= +0.03 R2
```

on the fixed fold-0 report query, with all validity gates passing. This retains
the historical Version-B threshold; it is not relaxed to give the new bundle a
second chance.

### 8.2 S-Acyc interpretation

If the static carrier-content gate fails, S-Acyc is descriptive only. It cannot
support a “carrier makes C1 work” claim, cannot open CQC and cannot rescue the
carrier branch. A positive CDM-A result may retain an activity-memory story
under Outcome C.

### 8.3 Conditional controls

If the static gate passes, train the already-defined `S-RS4`, `S-LS4`, `S-B4`
and `S-PCA3` arms. Syn3 carrier-content evidence requires matched-control
superiority. NNMF-specific evidence additionally requires superiority to
matched PCA3.

### 8.4 CQC

CQC is forbidden unless the static content gate passes. Its later exact matrix
is:

```text
S-Fix:  activity M10,        carrier M10
S-Acyc: activity 10/5/2,     carrier M10
S-Ccyc: activity M10,        carrier 10/6/4
S-Jcyc: activity 10/5/2,     carrier 10/6/4
```

Syn3×CDM-A positivity with static content failure does not open CQC.

### 8.5 Full-fold and official expansion

All-development-fold expansion requires equal-session mean delta `>= +0.01`,
at least three of four fold targets positive and no fold delta below `-0.05`.
Only one source-selected frozen candidate may later request a single official
M1 evaluation. Official data never select any design choice.

## 9. Stage-0 and implementation tests

Before GPU-capability issuance, tests must prove:

1. signed signal rejection and qualified nonnegative signal acceptance;
2. deterministic source dictionary and target-invariant dictionary bytes;
3. finite/nonnegative NNLS transform and frozen ordering;
4. exact support/query chronology and bin alignment;
5. M2 judged by observed coverage, not trial-count arithmetic;
6. Zero4 exact-zero carrier projection and identical initialization;
7. nonzero controlled carrier changes prediction and receives gradient;
8. shared unit permutation/dropout mask across activity and carrier;
9. C1 carrier SHA constancy;
10. CDM-A next-trial-only activity transition and carrier/model immutability;
11. fixed-last selection cannot read target query;
12. public CLI is dry/inert and cannot mint a GPU capability;
13. dry import opens no data/checkpoint/result root and imports no Torch/CUDA;
14. failure lifecycle is mutually exclusive with terminal and records honest
    progress/source/target/CUDA facts.

Stage-0 receipt must contain the complete all-fold/budget reliability table and
an explicit pass/fail decision before any live capability is created.

## 10. Additive ownership

Implementation is confined to new route-owned paths:

```text
tfpd_exploration/src/m1_emg_syn3_fcm_v1/**
tfpd_exploration/scripts/run_m1_emg_syn3_stage0.py
tfpd_exploration/scripts/run_m1_emg_syn3_pilot.py
tfpd_exploration/tests/test_m1_emg_syn3_fcm_v1.py
```

The completed T0/C1 packages, result roots and immutable receipts are read-only
historical evidence. Shared source edits require a separate seam audit and
explicit approval; a route-local additive composition is preferred.

## 11. Resource isolation and live launch gate

No GPU launch occurs while PACD P0 or another owner occupies the selected
physical GPU. The launcher must bind physical UUID, CUDA index, parent/named
root identity, CPU affinity, exact environment and live process ownership. It
must reject any overlap or root collision before attempt publication.

An opaque one-shot capability may be issued only after:

1. independent review of current source closure and this work order;
2. accepted Stage-0 terminal and source authority;
3. exact checkpoint authority and strict-load proof;
4. fresh canonical arm roots;
5. idle selected GPU with no conflicting process;
6. exact source-only data scope and formal-isolation proof.

GPU execution remains serial by arm unless a separately reviewed scheduler
proves disjoint device/process/root ownership. No action may signal, reprioritize
or change affinity of an existing job.

## 12. Stop rules and paper language

Stop the Syn3/CQC branch if the static `+0.03` carrier-content gate fails. Do
not sweep rank, NNMF solver, lag, nonlinear injection, carrier width or target
normalization after reading the result. CDM-A may continue only as an
activity-memory contribution if its own paired development evidence is
positive.

Until all matched controls pass, the only paper-safe statement is:

> We evaluate a corrected M1 carrier-aware system that preserves the complete
> activity-derived identity and independently injects a source-frozen,
> bin-level nonnegative EMG functional carrier. The experiment separately
> measures carrier content, activity-prefix robustness and causal activity
> memory; it does not reinterpret the earlier carrier-absent T0/C1 run as a
> carrier result.
