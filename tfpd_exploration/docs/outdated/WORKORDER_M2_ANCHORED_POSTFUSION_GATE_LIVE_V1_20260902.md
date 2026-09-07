# Work Order — M2 Anchored Post-Fusion Gate Live V1

Date: 2026-09-02. Status: **conditional single-run live authorization**.

This work order authorizes exactly one canonical APFG V1 execution after all
no-data tests and the independent audit pass, the result root is still fresh,
and the immediate GPU0-only preflight passes.  GPU0 may be used only for the
source-only scalar-alpha fit, selection, and all-source refit; target
materialization and all 13-session scoring must run on CPU.  GPU1 is
permanently out of scope: the route must not query, attach to, signal, change,
or otherwise interfere with GPU1 or its process.

The launch environment is frozen to:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONPATH=/home/xinyuan/Work_host/SPINT
```

Any preflight, source safety-gate, fixed-zero sentinel, closure, authority, or
final revalidation failure terminates this one attempt without target scoring
and without an automatic retry.  A retry or a second scientific execution
requires a successor work order.

## Authority and predecessor

The scientific design is `DESIGN_ANCHORED_POSTFUSION_GATE_V1_20260902.md`,
SHA-256 `2b78e5be4de814d5f0bf00048df781c80d5bed695cad8c550ddd3ea3021d4e4f`.
The immutable successful operator-corrected graph is a required historical
witness, but not a mutable input:

- root: `tfpd_exploration/results/m2_postfusion_operator_corrected_score_v1`;
- bodies attempt / launch / input-authority / score / terminal:
  `0f4e441c77f6c48c4358f0face4048ecf668bf8a6936818df3c279f436e4f57d`,
  `10f4ed88a31f2cf09016c854dbbf82d51e1e4f5e670a05fbbb1f18fb6ccfa9aa`,
  `f0bd485ec578b208c1dddb2d2b8f9c5c90d9677262b7debcfef4df9697d3ec89`,
  `156d7fab27c70bb01a7804bfdbb3434164c71dcab81ce3b1ad8a371eb169175d`,
  `f0f137c14c99da6e843b4887b4f3102610844fc0a3ee73fc0dfae4c547c41b01`;
- closure: `95b8e9e07e39700089e8364f67b15a020d17eb723a56cb18c2222305a1b25225`.

The graph has exactly these five body/sidecar pairs, all regular `0444`,
single-link leaves, no failure or extras. Its terminal is schema
`m2_postfusion_operator_corrected_score_v1_terminal_v1`, has 78 rows, zero
parameter/target updates, `cuda_initialized=false`, and exact PF-MEAN V2
control evidence.

## Route law

1. Before any checkpoint, data, CUDA, or result-root operation, a root-only
   opaque one-shot capability must bind the current source closure, canonical
   fresh root/parent identity, GPU0-only profile, historical corrected graph,
   selected-T4 POOLED authority, and frozen source roster/surface contract.
2. The executor reserves its canonical root and publishes immutable attempt
   before descriptor/checkpoint/data/CUDA access. It must revalidate closure,
   historical witness, canonical root, GPU0 identity, and opaque capability
   before final or failure publication.
3. Prepare PIT/source materialization exactly once. Strict-load selected-T4
   POOLED before installing APFG. Freeze all inherited parameters permanently;
   keep inherited modules eval/no-dropout while scalar alpha retains autograd.
   Adam must contain only scalar float32 alpha.
4. Source selection is lexical 5/2, exactly 12 epochs, earliest equal-session
   validation maximum and the design safety gate. Refit alpha from positive
   zero on all seven source sessions for that fixed epoch count; no target
   observation changes selection or duration.
5. Every pool state binds session identity, endpoint, ordered membership,
   ordered activity SHA256s, and the exact historical selected-support-four
   carrier SHA256. M4/M10/M30 are
   support4 / support4+recent6 / support4+recent26 only. No cross-session
   grouping, clamp, pad, repeat, current, or future trial is legal.
   The activity authority for alpha fit, two-session validation, and all
   thirteen deployment rows is exactly `POOLED/G00m linear`:
   `m2_precision_cdm_v2_screen_v1._native_trial_views` followed by the same
   G00m support/query reconstruction. PIT constructs only the model,
   DataModule, normalizer, and source raw-session authority. Its cubic
   `dataset.calib_trialized_neural_features` must not be mixed into APFG
   activity pools. Every receipt records
   `activity_authority='pooled_g00m_linear'`.  The historical carrier is the
   fixed-ridge fit from `support_rates_hz30[selected_D-opt-k4]`; the first-30
   ridge array remains audit evidence only and must never be substituted as
   the decoder side input.  Preserve the historical float64 carrier fit and
   multiplication by `MODEL_BIN_SECONDS` before the final float32 cast and
   normalizer application.
6. Scoring uses the locked 13-session surfaces and two laws. It emits
   APFG-ZERO and APFG-LEARNED. `APFG-ZERO/FIXED30` must exactly reproduce all
   13 sealed POOLED rows before learned contrasts may be interpreted.
   `APFG-ZERO/UNCAPPED` is a genuine memory-only counterfactual and must not be
   duplicated from, or claimed equal to, the historical bounded POOLED row.
   Receipts report the memory-only, matched-law gate-only, and total
   `APFG-LEARNED/UNCAPPED - sealed POOLED/FIXED30` effects separately. The
   external promotion gate (`mean >= +0.010`, at least `4/6` positive) applies
   only to that total deployed effect. Every paired contrast also reports a
   deterministic 10,000-resample ordinary session-bootstrap 95% interval
   (seed 42), descriptively and without changing the gate.
   GPU0 is restricted to source alpha fit/refit. After a CUDA synchronization,
   both frozen ZERO and LEARNED models must be moved to CPU and every target
   row must be decoded on CPU with the historical POOLED batch/numeric law;
   this is required for the fixed-zero prediction-SHA sentinel. Receipts bind
   `source_training_device='cuda:0'` and `target_scoring_device='cpu'`.
7. Alpha training uses exactly source behavior task-only **last-bin MSE**. It
   calls frozen `student.decode_with_identity(neural, h)`, divides the output
   by the inherited `behavior_scaling_factor=5.0` because
   `predict_scaled_behavior=true`, takes `[:, -1:, :]`, and compares it with
   the same batch target `[:, -1:, :]`. No teacher-y, teacher-identity,
   consistency, full-window, target-side, or auxiliary loss is authorized.
   `teacher_forward_calls=0` is a required receipt fact. Source validation and
   target R2 call the exact historical `variance_weighted_r2` implementation,
   including its per-output float64 reduction order.

## No-data acceptance for this work order

- held-FD corrected-predecessor codec plus sidecar/mode/topology adversaries;
- typed opaque one-shot capability and public inert CLI;
- synthetic strict-load/install/freeze/alpha optimizer evidence;
- no-data source split/refit and exact zero/negative-zero sentinel tests;
- compile, clean no-CUDA import/dry checks, explicit closure candidate.

These acceptance items must pass immediately before capability issuance.
After they pass, this work order authorizes the single execution described at
the top of this document; it does not authorize exploratory reruns or any use
of GPU1.

## 7. Frozen source-coordinate/controller and selection metric

The source training controller is not free to choose a convenient pool law.
It must first construct one canonical source batch list containing only
coordinates for which the M30 causal pool is already constructible after the
first thirty trials.  Each entry is grouped by identical
`(session, query_trial, pool_state)` and has at most 32 members.  This
canonical list is the sole source batch population for all 12 epochs.

For epoch `e` (one-indexed) and canonical batch ordinal `b` (zero-indexed),
the requested pool size is exactly:

```text
(4, 10, 30)[(e - 1 + b) % 3]
```

No coordinate may be skipped, clamped, padded, repeated, or supplemented with
a future trial when that requested size is not available.  Because the list is
M30-eligible before cycling, all three sizes exist for every entry; over 12
epochs each canonical batch receives each size four times.

At every epoch, selection is evaluated by the locked source-only deployment
replay on the lexical validation two sessions: B30/D-opt-k4 support, the exact
selected-support-four historical POOLED carrier, decode-before-commit and
**UNCAPPED equal-session R2**.  This validation population is explicitly the
complete governed post-30 deployment stream beginning at trial 30; unlike the
training batches it is not restricted to M30-ready coordinates.  Training and
validation counts and ordered-coordinate digests must be published separately.
The replay also computes an exact alpha-positive-zero UNCAPPED control.  Select
the earliest maximum UNCAPPED equal-session R2 over epochs 1..12.  The safety
gate compares that same selected UNCAPPED replay with its zero control.  After
selection, compute one descriptive FIXED30 replay at the selected epoch for
both its learned alpha and exact positive zero; this readout cannot alter the
selected epoch, safety gate, or refit duration.
