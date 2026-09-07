# Work order — M2 Anchored Output Fusion V1

Bind and execute the frozen design
`DESIGN_ANCHORED_OUTPUT_FUSION_V1_20260902.md` as one additive source-only
STATIC30 screen.

## Authorized science

- Frozen Selected-T4 POOLED checkpoint/state only.
- Exact official `act30_dopt4` PIT-cubic first-30 activity pool:
  `dataset.calib_trialized_neural_features[session][:30]`, in stored
  chronological order. G00m-linear reconstructed activity is forbidden.
- D-opt-k4 selected-support carrier and frozen normalizer.
- Exactly two frozen identities per session: native and per-trial Post-Fusion
  mean.
- Exactly one global unconstrained scalar beta, fitted as the design's
  equal-session float64 **MSE-surrogate** minimizer; it is not a direct R2
  optimizer.
- Exact fit roster, in order:
  `ses-2020-10-19-Run1`, `ses-2020-10-19-Run2`,
  `ses-2020-10-20-Run1`, `ses-2020-10-20-Run2`,
  `ses-2020-10-27-Run1`.
- Exact validation roster, in order:
  `ses-2020-10-27-Run2`, `ses-2020-10-28-Run1`.
- Behavior scaling factor exactly `5.0`, checked against frozen model/payload
  metadata and receipt-bound.
- Direct native branch at IEEE beta `+0.0`.
- Conditional all-seven closed-form refit only after the complete source gate.

No backbone/gate training, optimizer, epoch, seed sweep, vector beta, clamp,
intercept, hidden/external/held-out/EvalAI target access, pseudo-label,
query-memory update, or trial-boundary state is authorized. Source behavior
targets are required and must be disclosed as `source_target_access=true` from
PIT DataModule preparation onward. The inherited source loader eagerly reads
all seven source behavior arrays, so this work order does not claim
filesystem-level blindness to the two validation sessions. Before beta is
frozen, production code must not index, digest, copy into a paired-output
record, summarize, or pass either validation target array to the fitter. It
must complete the five fit records and freeze beta before the two validation
records are materialized and evaluated once.

## Resource and lifecycle

- Physical GPU0 only, with exact UUID and deterministic environment bound.
- Physical GPU1 is outside scope and must not be queried or touched.
- Attempt must precede checkpoint, source data, and CUDA access.
- One source DataModule/model prepare, one strict frozen Selected-T4 load, no
  optimizer construction, independent `B` native/post production decodes, and
  one first-chunk `2B` numerical diagnostic per session.
- Additive fresh result root; immutable attempt, launch, source authority,
  paired-output authority, fit, validation, optional refit, and exactly one
  terminal-or-failure chain.
- No local external, EvalAI, held-out target, or official submission is opened
  by this work order.
- Source authority must bind PIT cubic interpolation, ordered first-30 activity,
  D-opt support, raw/normalized T4, normalizer, frozen state, and the official
  `act30_dopt4` native identity export receipt plus the immutable
  `ridge_activity30_m4 / within_post30` native prediction rows.
- The CPU official-payload identity and GPU sealed-screen identity are separate
  numerical authorities. For every source session, read the CPU identity from
  the exact SHA-bound official payload and exact-match its stored identity SHA;
  a fresh strict-clone CPU recomputation may match it only numerically, with
  max-abs `<=2e-6`. Independently compute the governing GPU identity and
  exact-match the sealed GPU prediction SHA, then bind a payload-CPU-to-GPU
  bridge with identity max-abs `<=2e-6`, prediction max-abs `<=2e-6`, and R2
  absolute difference `<=2e-7`. Never claim recomputation or cross-device SHA
  equality.
- Before launch, both halves of the first-chunk diagnostic `2B` decode must
  match their independent `B` native/post decodes within prediction max-abs
  `2e-6` and R2 abs `2e-7`; the diagnostic output is not used by the fit.
- The executed `beta=+0.0` sentinel must branch directly to native without
  computing the post identity or a second decoder call, and must match the
  sealed GPU native anchor's full-session prediction SHA, window count, starts,
  and target evidence at the frozen batch law; R2 absolute difference must be
  `<=1e-12`.
- Fit receipts must record finite-input checks, exact 5/2 roster, numerator,
  denominator, denominator scale/conditioning evidence, beta, and the explicit
  MSE-surrogate/not-R2-optimum disclosure.

## Decision

The source gate is exactly:

```text
mean validation delta versus native >= +0.005
2/2 validation sessions positive
worst validation delta >= +0.001
beta finite and row-invariant
beta(+0.0) native bitwise equality everywhere
```

Failure terminates AOF V1. Success permits the deterministic all-seven refit
and a separately reviewed EvalAI packaging work order; it does not itself
authorize hidden/external target scoring or submission. EvalAI compatibility
is only architectural at this stage and remains pending exporter, runtime,
container, latency, and exact native-parity validation.
