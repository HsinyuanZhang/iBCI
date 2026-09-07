# H1 QueryAge16 + cold-prefix formal candidate — conditional protocol

**Date:** 2026-09-06  
**Status:** review-only, prospective, and non-authorizing.  
**Candidate:** fresh `FW-QueryAge16` FLAT/ROUTE pair trained with a deterministic
`p=0.5` cold-history prefix treatment.

This document defines a possible future formal-quality candidate only.  It
does not authorize a GPU process, cache rebuild, model forward, training,
output-root creation, checkpoint promotion, registration, deployment, or
submission.  No hash for a not-yet-written runner, launcher, smoke, test, or
authorization is asserted here.

## 1. Purpose and interpretation boundary

The candidate asks a narrow quality question: whether a **fresh H1
`FW-QueryAge16` decoder under a declared H1 cold-prefix training recipe** can
learn and generalize under the existing H1 family-formal evaluation contract.

It is deliberately a compound intervention:

```text
temporal operator:      QueryAge16 rather than the old CausalPE4 control
training input recipe:  p=0.5 cold-history prefix rather than old p=0 formal
```

Consequently, it cannot establish a causal temporal-operator comparison with
the already-completed no-prefix CausalPE formal run.  A statement that
QueryAge16, as opposed to prefix treatment or their interaction, outperformed
old CausalPE would require a separately preregistered matched CausalPE+p=0.5
run.  This protocol makes no such comparison or promotion rule.

The prefix choice is H1-specific.  It is not inferred from M1 or M2 recipe
choices.  It is included because the existing H1 cold-history diagnostic makes
the cold-start intervention a plausible H1 quality candidate; that diagnostic
does not turn this document into a causal attribution study.

## 2. Cumulative admission gates

No formal run may begin unless **all** of the following are true at launch and
are revalidated at the applicable post-launch/finalization boundary.

1. The current prospective QueryAge source-capacity extension has completed
   exactly 1,040 total updates from its own strict 260-update checkpoint and
   reports both arms with:

   ```text
   pooled source208 R2 >= 0.50
   prediction standard deviation >= 0.50 * target standard deviation
   ```

   The extension receipt, its bound 260 receipt, its checkpoint, their
   current code/input closures, owned-artifact manifests, and all associated
   hashes must pass their own fresh validators.  The 260-stage fact that at
   least one arm passed its source gate is not a substitute for this two-arm
   1,040 gate.

2. A new, disposable, source-only, two-GPU resource smoke for the exact
   QueryAge16+p=0.5 candidate has passed.  It must use fresh initialization,
   the actual p=0.5 prefix law, the actual p=0.1 stateless unit-dropout law,
   the intended sampler, microbatching, optimizer, EMA, and per-arm device
   layout.  It may not score minival or complete data, select an epoch, retain
   a learned checkpoint, or create the formal output root.

3. The smoke's conservative forecast, including formal training, twelve
   selection passes per arm, strict checkpoint/reload work, post-freeze
   selected and epoch-12 complete scoring/export, archive writing, and a
   declared allowance for setup/finalization, fits the root-approved resource
   and wall-time envelope.  Peak allocated memory must be within the
   root-approved bound.  A smoke for old CausalPE, old selected weights, or a
   two-condition cold-history diagnostic cannot certify this candidate.

4. An external root-reviewed authorization binds the fresh absolute output
   path, physical-device allocation, exact protocol bytes, exact executable
   closure, immutable inputs, predecessor evidence, and smoke receipt.  It
   must be checked before any model/cache runtime import that could begin the
   execution path and reconstructed after the run.

Failure or absence of any gate means **no launch**.  Passing a gate permits a
new review only; it does not automatically schedule or launch work.

## 3. Immutable sources, evidence, and fresh initialization

The following are immutable dependencies.  A future implementation must hash
and bind their then-current bytes; this document intentionally supplies no
invented digest values.

- The pre-existing source cache and its authority receipt.  The cache must
  already exist: the candidate must not call a cache builder, alter a cache,
  replace arrays, or rebuild source windows.
- The frozen H1 source sampler, source train rows, source minival rows, bank
  tensors, masks, and the frozen minival/complete evaluation definitions.
- The trusted H1 optimizer parameter grouping, EMA implementation, H1
  frontend/base-family code, QueryAge factory and core, sampler/collation
  helpers, score helpers, prefix helper, and new formal runner/launcher/smoke
  code.
- The accepted 1,040 QueryAge capacity evidence and the new smoke evidence.
  These are **eligibility/provenance inputs only**.

The formal candidate always calls
`h1_queryage_family_v1.model.make_queryage_localbalanced_pair(seed=42)` to
create a new paired initialization.  It must never load, copy, resume, or
otherwise warm-start from the source260/source1040 model, optimizer, EMA,
RNG state, or capacity checkpoint.  The formal receipt must say explicitly
that the capacity checkpoint was not used as model state.

Existing source-capacity, old family-formal, cache, model, helper, result,
checkpoint, and authorization files remain untouched.  The formal runner,
launcher, smoke, tests, and result root must be new additive paths.

## 4. Fixed formal recipe

The candidate retains the established H1 split-arm formal schedule except for
the declared QueryAge temporal member and declared prefix treatment.

| Item | Fixed rule |
| --- | --- |
| Arms | FLAT and ROUTE; both fresh QueryAge16 members |
| Seed | 42 for paired initialization and all declared stateless schedules |
| Epochs | 12 |
| Source schedule | Existing ordered H1 source-train sampler; exactly 731 effective batches and 23,212 source windows per epoch |
| Batch geometry | Ragged final batch retained; effective batch at most 32, microbatch 8; each micro-loss weighted by `micro_count / actual_effective_batch` |
| Target | Existing decoder-raw H1 target: native velocity multiplied by 20 |
| Optimizer | Fresh AdamW over the existing trusted H1 decay grouping, weight decay `.01`, gradient clipping norm `1` with nonfinite rejection |
| Learning rate | `1e-4 * global_step / 731` for steps 1–731 of epoch 1; `1e-4` thereafter |
| EMA | Fresh trusted `DecoderEMA`, decay `.9995`, updated once after every successful optimizer step |
| Dropout | Existing deterministic whole-unit keep mask, p=.10, intersected with the immutable bank unit mask |
| Prefix | Deterministic p=.5 cold-history treatment applied to raw input **before** the p=.10 unit-dropout mask is supplied to the model |
| Prefix transform | Chosen rows retain a deterministic uniform right suffix of 1…699 bins and have only the missing left history zeroed; the current bin and target are unchanged |
| Selection | After every completed epoch, score EMA only on the frozen 2,908 minival endpoints; governing metric is pooled native float64 R2; choose the earliest epoch attaining the maximum |
| Complete reporting | Only after both-arm selection freeze: strictly reload selected EMA and independent epoch-12 EMA, then report each on the frozen 20,325 complete masked bins with native float64 prediction archives |

The established selection metric is specifically **EMA pooled R2 on the frozen
2,908 H1 minival endpoints**, evaluated after each complete epoch.  It is not
source208, not a cold-prefix synthetic evaluation, not a complete-bin score,
and not an official/hidden outcome.  The 20,325-bin report is a post-freeze
complete evaluation, not an epoch-selection surface.

## 5. Cross-arm matching contract

FLAT and ROUTE are split across physical GPUs but are a matched experiment.
Before either worker receives the start barrier, both must independently prove:

- fresh seed-42 paired initialization;
- byte-identical non-routing shared state;
- zero ROUTE gate and actual zero-gate FLAT/ROUTE output parity on the same
  source microbatch;
- finite, nonzero ROUTE gate gradient on an actual prefix-treated microbatch;
- exact equality of every epoch's ordered `(session, starts)` sampler digest;
- exact equality of every epoch's stateless p=.10 keep-mask digest; and
- exact equality of every epoch's p=.5 prefix treatment digest, including the
  source batch identity and the generated per-row retained lengths (or an
  equivalent lossless treatment-mask representation).

For every training batch, the order is fixed:

```text
source windows and native targets
    -> deterministic p=.5 left-zero cold-prefix transform on x only
    -> deterministic p=.10 whole-unit keep mask AND immutable bank mask
    -> four or fewer size-weighted microbatches, forward/backward/update/EMA
```

The prefix transform changes neither targets nor the final/current input bin.
It is independent of model outputs and arm identity.  The sampler, keep, and
prefix identities must be committed in checkpoints/worker receipts and be
verified by the supervisor before selection freeze.

## 6. Checkpoint, scoring, and artifact contract

Each end-of-epoch checkpoint is written atomically to a new path and contains
the RAW model, optimizer, EMA, completed epoch/global-step count, sampler/keep/
prefix identities, shared-init identity, protocol/code/input closure, and CPU,
CUDA, NumPy, and Python RNG state.  The checkpoint is immediately disk-loaded
and strictly checked before it is used as evidence.  There is no automatic
restart and no mid-epoch resume claim.

The supervisor freezes selection only after both arms have all twelve valid
EMA minival records and independently reproduces the prescribed earliest
maximum.  Finalizers must strictly reload the selected and epoch-12
checkpoints, verify their input and identity bindings, reproduce the
corresponding 2,908-endpoint EMA score, write separate native float64 complete
archives, and hash every retained artifact.  RAW state must be verified
unchanged by EMA scoring.

The final result must distinguish:

- source-only capacity eligibility;
- formal minival selection evidence;
- post-freeze complete descriptive reporting; and
- the non-causal-attribution limitation caused by the p=.5 prefix difference
  from old no-prefix CausalPE.

No output constitutes an official, hidden/test, submission, or deployment
claim.

## 7. Resource smoke and device policy

The preferred formal layout is two isolated one-thread processes:

```text
physical GPU 0 -> FLAT worker, visible as cuda:0
physical GPU 1 -> ROUTE worker, visible as cuda:0
OMP_NUM_THREADS=1; MKL_NUM_THREADS=1; PyTorch intra/inter-op threads=1
```

The new smoke must use this same split layout, synchronize the two fresh arms
only after their matching identities are proven, and time the actual p=.5
candidate rather than a proxy.  Its formal-wall estimate must use the slower
arm for concurrent train wall time and separately budget all serialized
selection, finalization, exports, reloads, and final audits.

The fixed smoke is20 source updates per arm, with4 warmup updates discarded and
16 steady updates retained. All twelve epochs' source/keep/prefix identities
are committed before the start barrier. The inference-cost proxy uses the exact
208 source-only endpoints under EMA in microbatches of8, matching the formal
scorer's batch boundary; it does not score minival. The checkpoint estimate
includes actual serialization, deserialization and restoration of RAW model,
optimizer, EMA and RNG state, after which the disposable checkpoint is removed.

The per-arm conservative wall estimate is
`1.5 * steady_update_p95 * (12 * 731)` plus
`1.5 * source208_seconds_per_endpoint * (12 * 2908 + 2 * 20325)` plus
`12 * measured_checkpoint_roundtrip_seconds + 1800` seconds. Use the larger
arm estimate for the concurrent pair. Both this estimate and the formal
supervisor's hard wall must fit21600 seconds; each smoke worker and its paired
supervisor have a900-second hard wall. Peak CUDA allocation must not exceed
22 GiB. These are resource-safety constraints, not latency acceptance criteria
against SPINT, and do not weaken the unified-network priority.

A one-GPU alternative is not implicitly allowed.  If separately proposed, it
must be a new, explicitly authorized sequential design with independent fresh
processes, the same stateless identities, summed arm wall-time forecast, and
no claim of a simultaneous paired start.

## 8. Non-goals and prohibitions

- Do not modify or reuse the source-capacity learned checkpoint as formal
  state.
- Do not alter old formal runners, old cache builders, `VARIANTS`, old model
  helpers, old result roots, old authorizations, old checkpoints, or frozen
  evaluation artifacts.
- Do not rebuild or mutate the cache; do not use hidden/test data, official
  outcomes, or submissions for selection.
- Do not rename this compound candidate as a pure temporal comparison.
- Do not run a smoke, formal worker, finalizer, or GPU process from this
  document alone.

Any implementation or authorization must be reviewed as a subsequent,
separate artifact and must bind its own exact closure and fresh output root.
