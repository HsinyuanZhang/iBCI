# Work order: M1 EMG-rSyn3 Functional Carrier Memory, one-fold V1

**Date:** 2026-09-02  
**Status:** frozen scientific and implementation contract for the ReLU
successor; live GPU execution is conditional on this Stage-0 acceptance and a
separate opaque capability  
**Design authority:**
`tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_SUCCESSOR_20260902.md`  
**Parent design (unchanged bytes):**
`tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md`  
**Parent work order (unchanged bytes):**
`tfpd_exploration/docs/WORKORDER_M1_EMG_SYN3_FCM_ONEFOLD_V1_20260902.md`  
**Parent FAIL interpretation:**
`tfpd_exploration/docs/RESULT_M1_EMG_SYN3_STAGE0_SIGNED_REJECT_20260902.md`

This work order authorizes implementation and CPU Stage 0 of **EMG-rSyn3**.
It does not rewrite the sealed EMG-Syn3 FAIL. It does not by itself authorize
opening a GPU process.

## 1. Scientific question and estimands

EMG-Syn3 is not abandoned. The parent Stage-0 FAIL proved only that stored
`preprocessed_emg` cannot enter NNMF without a frozen nonnegative projection.
The primary question remains whether an independently injected, source-frozen,
bin-level nonnegative EMG functional carrier adds cross-session M1 information
beyond a topology-matched Zero4 carrier. EMG-rSyn3 answers that question after
one frozen projection.

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

Identical to the parent:

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

The provider must resolve the same allow-listed held-in-calib source files.
Paths containing `minival`, `held-out`, `formal`, `EvalAI` or `test` reject
before data materialization. Source and target file body digests, dev/inode
identities, trial boundaries and query-window digest must be recorded.

## 3. Parent FAIL is sealed evidence, not a rewrite target

Stage 0 must descriptor-verify the parent root before writing any successor
receipt:

```text
root = tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0
decision.json sha256 = 6cc34cdd434917907d8c90b739c3c02003a511c3ae919277baf6de041702bf39
signal_view.json sha256 = 27e681f86aca6bf7a2d7163cb422ba2c6b1f04c6bb8c5414687335c3453858cf
terminal.json sha256 = f8436c9e2e1f1e62a3a3fb4f78415cba7bef395c3821437ddb2ee67b99a3f7ba
decision = FAIL
failure = stored preprocessed_emg is signed; NNMF rejected without a frozen rectifier
```

Any write, chmod, truncation, or re-reserve of that root is a contract
failure. The successor result root is disjoint.

## 4. Source-frozen EMG-rSyn3

### 4.1 Unique rectifier

Before RMS scaling and before NNMF, apply exactly:

```text
x_rect = np.maximum(x_raw, 0.0)
```

Law:

```text
name = relu_nonnegative_projection
elementwise = true
dtype = float64
threshold = 0.0
learnable_parameters = 0
per_session_threshold = forbidden
per_channel_threshold = forbidden
sweep = forbidden
forbidden_alternatives = abs, offset, envelope_filter, smoother
query_values_read = false
```

The same operator is applied to every source session and to target support.
The stored (possibly signed) view remains in the receipt. The rectified view
must have `negative_fraction = 0` and `minimum >= 0`. Record clipped-count,
clipped-fraction, negative mass and maximum undershoot. If any rectified value
is negative, Stage 0 fails.

This operator is not a target-learned correction and is not a provenance claim
that stored `preprocessed_emg` is a rectified envelope.

### 4.2 Qualified signal view

Use movement-window EMG aligned to fixed 20-ms bins. Stage 0 must report, for
the stored view and the rectified view, per source session and channel-set:

- finite count, minimum, maximum, zero fraction and negative fraction;
- sampling/time-alignment law;
- rectifier law above;
- valid bins, valid seconds and contributing trials.

Direct NNMF on the stored signed view remains rejected. NNMF is entered only
on `x_rect`.

### 4.3 Positive scaling

For each EMG channel, compute source-only RMS over the qualified **rectified**
source bins:

```text
scale_j = max(sqrt(mean_source(emg_rect_j^2)), 1e-8)
x_j = emg_rect_j / scale_j
```

No mean subtraction is allowed. The target uses the exact source scales after
the same rectifier.

### 4.4 Deterministic source NNMF

Fit the source matrix `X_source [bins,emg_channels]` with rank 3 on rectified,
RMS-scaled bins. Freeze the parent solver law:

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
scales, raw/normalized dictionary, activations, ordering, rectifier digest and
reconstruction digests.

Target activations are obtained with the frozen dictionary by a deterministic
nonnegative least-squares transform of **rectified**, source-scaled target
support bins. The target cannot refit, rotate, reorder or rescale the source
dictionary, and cannot choose a different rectifier.

### 4.5 Bin-level per-unit encoding

Identical to the parent: ridge `lambda=1` on `W`, unpenalized intercept, lag 0,
budgets M10/M6/M4/M2. M2 is judged by coverage evidence, not trial-count
arithmetic.

### 4.6 Coverage and reliability evidence

Identical disclosure table to the parent, plus rectifier mass statistics.
Poor split-half stability does not silently block the predeclared pilot.
Nonfinite/undefined fits, unqualified signal view, leakage, chronology failure,
parent-root mutation, or rectifier-law drift do block it.

## 5. Carrier normalization and controls

Fit the four-coordinate carrier normalizer from source-session carriers only.
Target uses the exact frozen normalizer. Build:

```text
Zero4: four exact normalized zeros; no target carrier fit
rSyn3: correctly attached normalized EMG-rSyn3
RS4: deterministic complete source-unit row shuffle
LS4: deterministic support EMG/neural association derangement before refit
B4: normalized baseline coordinate only, three synergy coordinates zero
PCA3: matched bin-level signed-PCA basis on the same rectified bins under
      the same independent injection
```

The work order pre-registers conditional matched-training arms `S-RS4`,
`S-LS4`, `S-B4` and `S-PCA3` under the parent launch rule. There is no
trial-mean rSyn3 arm.

## 6. Independent carrier injection

Unchanged from the parent:

```text
P = Linear(4, model_dim, bias=False)
u_i = fc_in(x_i + E_i^A) + P(carrier_i)
```

Zero init, shared unit mask, C1 carrier-SHA constancy, CDM-A activity-only
updates, and no MLP/bias/gate/FiLM remain mandatory.

## 7. Stage-1 three-arm training contract

Train exactly:

| arm | carrier | activity prefix |
|---|---|---|
| `Z-Fix` | Zero4 | fixed M10 |
| `S-Fix` | rSyn3 M10 | fixed M10 |
| `S-Acyc` | rSyn3 M10 | step cycle `(10,5,2)` |

All arms use the parent recipe: fold 0, seed 42, 12 epochs, fixed-last
`epoch_011`, task-only MSE, Adam `1e-4`, batch 32, no scheduler, no SWA, no
target validation during training. Initialize from the same teacher
checkpoint:

```text
streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/
2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/
checkpoints/best_ckpt/epoch_018.ckpt
sha256 = f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be
```

If GPU 1 is idle after an accepted Stage-0 terminal, the three arms may share
that one physical GPU as concurrent processes with disjoint result roots,
because arm VRAM is small relative to 24 GiB. They may not share a result
root, checkpoint directory, or CUDA context. GPU 0 remains refused. No action
may signal, reprioritize or change affinity of an existing job. Concurrent
launch still requires the opaque capability in Section 11.

## 8. Paired deployment scoring

Unchanged from the parent: Static M10 versus CDM-A with the carrier held
fixed. Six cells, per-session prediction/target digests, paired deltas, model
state before/after and target update counters.

## 9. Predeclared decisions

Identical thresholds to the parent:

- static carrier-content gate: `S-Fix − Z-Fix >= +0.03 R²`;
- failed static gate makes `S-Acyc` descriptive only;
- matched controls and CQC open only if the static gate passes;
- full-fold rule: equal-session mean `>= +0.01`, at least three of four fold
  targets positive, no fold delta below `-0.05`.

Do not relax the gate because a rectifier was added.

## 10. Stage-0 and implementation tests

Before GPU-capability issuance, tests must prove:

1. parent FAIL receipts remain byte-identical and are never written;
2. stored signed EMG still rejects parent EMG-Syn3 NNMF without a rectifier;
3. `relu_nonnegative_projection` is elementwise `float64` `max(x, 0.0)` and
   is not `abs`, offset, envelope or smoother;
4. the rectifier runs before RMS scaling and NNMF, identically on source and
   target support, with zero learnable parameters and no query read;
5. rectified input has `negative_fraction = 0` and is accepted by NNMF;
6. deterministic source dictionary and target-invariant dictionary bytes;
7. finite/nonnegative NNLS transform and frozen ordering;
8. exact support/query chronology and bin alignment;
9. M2 judged by observed coverage, not trial-count arithmetic;
10. Zero4 exact-zero carrier projection and identical initialization;
11. nonzero controlled carrier changes prediction and receives gradient;
12. shared unit permutation/dropout mask across activity and carrier;
13. C1 carrier SHA constancy;
14. CDM-A next-trial-only activity transition and carrier/model immutability;
15. fixed-last selection cannot read target query;
16. public CLI is dry/inert and cannot mint a GPU capability;
17. dry import opens no data/checkpoint/result root and imports no Torch/CUDA;
18. failure lifecycle is mutually exclusive with terminal;
19. successor Stage 0 cannot reserve the parent result root.

Stage-0 receipt must contain stored and rectified signal views, rectifier mass
statistics, the complete all-fold/budget reliability table and an explicit
pass/fail decision before any live capability is created.

## 11. Additive ownership

Implementation is confined to new route-owned paths:

```text
tfpd_exploration/src/m1_emg_rsyn3_fcm_v1/**
tfpd_exploration/scripts/run_m1_emg_rsyn3_stage0.py
tfpd_exploration/scripts/run_m1_emg_rsyn3_pilot.py
tfpd_exploration/tests/test_m1_emg_rsyn3_fcm_v1.py
tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/**
```

Read-only historical evidence:

```text
tfpd_exploration/src/m1_emg_syn3_fcm_v1/**
tfpd_exploration/results/m1_emg_syn3_fcm_v1/**
tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md
tfpd_exploration/docs/WORKORDER_M1_EMG_SYN3_FCM_ONEFOLD_V1_20260902.md
tfpd_exploration/results/m1_t0c1_prefix_v1/**
tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/**
```

Shared source edits require a separate seam audit. Prefer importing parent
helpers over copying bytes, provided parent behavior is not changed.

## 12. Resource isolation and live launch gate

No GPU launch occurs while another owner occupies the selected physical GPU.
The selected device remains GPU 1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`. GPU 0 UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` is refused.

An opaque one-shot capability may be issued only after:

1. independent review of current source closure and this work order;
2. accepted Stage-0 terminal and source authority;
3. exact checkpoint authority and strict-load proof;
4. fresh canonical arm roots disjoint from the parent FAIL root;
5. idle selected GPU with no conflicting process, or a reviewed concurrent
   schedule that keeps arm roots and processes disjoint on GPU 1;
6. exact source-only data scope and formal-isolation proof.

## 13. Stop rules and paper language

Stop the rSyn3/CQC branch if the static `+0.03` carrier-content gate fails.
Do not sweep rank, NNMF solver, lag, rectifier alternatives, nonlinear
injection, carrier width or target normalization after reading the result.

Until all matched controls pass, the only paper-safe statement is:

> We evaluate a corrected M1 carrier-aware system that preserves the complete
> activity-derived identity and independently injects a source-frozen,
> bin-level nonnegative EMG functional carrier. The carrier is EMG-rSyn3:
> stored preprocessed_emg is projected onto the nonnegative cone by the
> parameter-free operator R(x)=max(x,0) before RMS scaling and rank-3 NNMF.
> This operator is not claimed to recover a physiological rectified envelope.
> The parent EMG-Syn3 Stage-0 FAIL remains evidence that the stored series
> cannot enter NNMF unsigned; it is not evidence against synergy content,
> independent injection, or CDM-A.
