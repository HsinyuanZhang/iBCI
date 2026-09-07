# Native-M2 post-33 Phase-B v3: score-free implementation status

Date: 2026-08-04  
Protocol: `M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1`  
Phase: `PHASE_B_V3`  
Decision: **NO-GO for GPU launch**

This phase implements and tests the C1–C3/H1 plumbing requested by the independent review. It does not authorize training, read a new endpoint R², open formal SUA data, call a scorer, or submit to EvalAI. The active C1 v3r2 source map and the Phase-A v2 source map were rehashed after the work and remain byte-identical.

## What is now implemented

### C1 — exact source-only SPINT selection

The new versioned Lightning module is `SPINT-main/src/models/falcon_post33_confirm_v3_module.py`; the historical `falcon_module.py` remains untouched.

- Validation accepts exactly the six outer-train source sessions for the requested fold.
- Every source session must have `total > 2` and a finite per-session R².
- The unique outer session is audited with `total == 0` and excluded from the equal-session mean.
- Missing, extra, non-finite, low-total, or outer-contaminated synthetic inputs fail closed.
- Test accepts exactly the unique outer session. The six absent source sessions are not inserted as empty placeholders and do not enter an average.
- The versioned callback saves every epoch 0–34 and uses the explicit policy `max finite equal-session mean; tie -> earlier epoch`. It does not rely on Lightning `ModelCheckpoint` tie behavior.

The v3 Hydra binding uses the new source metric (`val_source/r2_equal_session_mean`) and the new callback. The v1 score-free datamodule implementation is reused through a new v3 data config because it is already hash-pinned and its live all-fold input/query audit passed; it was not edited.

### C2 — exact paired teacher and decoder closure

The SPINT completion-receipt schema binds:

- protocol, phase, arm, fold, seed, outer session, and the ordered six source sessions;
- all 35 selector records and the explicit selected epoch;
- canonical selected checkpoint path, byte size, and SHA-256;
- canonical resolved-config path, byte size, and SHA-256.

Receipt and status writes are write-once with `os.open(..., O_CREAT | O_EXCL)`. A T4 cell has no `teacher_ckpt_path` config field. Its mandatory `paired_spint_completion_receipt: ???` is resolved twice—during cell preflight and model construction—and must match the same fold and seed. The global `paths.teacher_ckpt_path` and its legacy `epoch_034.ckpt` are not referenced by any v3 config or T4 v3 model path.

The decoder lifecycle helper snapshots all 31 SPINT state tensors at `pretrain`, `posttrain`, `reload`, and `prequery`. Equality includes tensor name/key, shape, dtype, byte count, and SHA-256 of contiguous CPU bytes. It also requires:

- zero decoder parameters with `requires_grad=True`;
- zero decoder/optimizer parameter intersection;
- zero changed decoder tensors.

Focused negative tests mutate each of name, shape, dtype, byte count, and bytes, and separately test `requires_grad` and optimizer intersection.

### C3 — deterministic cell identity and write ownership

Canonical cell paths are a pure function of:

```text
protocol / PHASE_B_V3 / arm / fold / seed
```

They cover Hydra output, checkpoints, selector records, resolved config, completion receipt, result, and write-once `started/completed/failed` status files. The score-free preflight claims `ownership.json` before emitting Hydra overrides. An eight-way same-cell concurrency test produces exactly one owner; different arm/fold/seed cells have disjoint paths.

This is the path/ownership substrate, not the final matrix launcher. The existing streaming training program also writes timestamped secondary artifact directories; the final runner/finalizer must either suppress those or bind/canonicalize them without treating them as protocol results.

### H1 — public/internal T4 datamodule truth

`streaming_calibration_exp/src/data/falcon_post33_confirm_v3_datamodule.py` is a dedicated implementation. It does not call the generic base `setup` and never rewrites public `hparams`.

The public values remain `validation_protocol=loso`, the requested `loso_fold`, `heldin_query_start_trial=33`, and `query_start_trial=0`. A separate immutable internal contract states that source minival selection starts at zero and the outer query starts at 33. The generic split resolver deliberately raises, preventing a generic consumer from silently interpreting the public LOSO contract as the old internal minival delegation.

## Verification completed

- New Phase-B focused suite: 26 tests passed (after the local paired-resolver test was added).
- Combined Phase-A + Phase-B focused suite: expected 52 tests; the final receipt records the exact rerun count.
- All new Python files pass `py_compile`.
- Actual Hydra compose tests bind the v3 SPINT source module/callback and the mandatory paired-receipt T4 module.
- Actual runtime constructor test confirms that T4 public `hparams` remain LOSO and the generic consumer fails closed.
- Active C1 v3r2 source map: 34/34 hashes reverified, zero drift.
- Phase-A v2 source map: 30/30 hashes reverified, zero drift.

## Why the status remains NO-GO

No GPU run should start from this Phase-B artifact. The remaining blockers are:

1. **C4 final anti-tamper verifier:** exact expected result/checkpoint/status sets, canonical-path containment, no symlinks/extras, receipt/result hashes, and exact complete matrix accounting.
2. **Final matrix launcher:** uses the O_EXCL preflight for every cell, owns crash/interrupt handling, writes `failed` exactly once, and never allows a direct Hydra launch to bypass ownership or paired-receipt validation.
3. **Final result/checkpoint finalizers:** SPINT completion-receipt support exists, but T4 result finalization and matrix-level aggregation are not complete.
4. **Cost receipt:** parameter count, MAC count, calibration state, and runtime state must be computed for both paired arms under the exact v3 resolved configs.
5. **Runtime evidence:** the 31/31 lifecycle helper is implemented and tested synthetically, but no authorized production checkpoint has yet produced the four-stage decoder receipt.

Accordingly, Phase-B establishes fail-closed plumbing only. It supplies no accuracy evidence and makes no positive T4-vs-SPINT claim.

