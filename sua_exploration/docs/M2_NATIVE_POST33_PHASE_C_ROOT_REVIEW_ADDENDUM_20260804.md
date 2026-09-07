> **SUPERSEDED — historical review evidence only, not current authority.** The NO-GO requirements below were addressed by later V4 launches, V5 device repair, and the r9 launch chain.
> Current launch state: r9 in [`ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](ACTIVE_EXPERIMENT_CONTROL_BOARD.md).
> Preserved as append-only audit evidence; content unchanged.

# Native-M2 post-33 Phase-C root review addendum

**Date:** 2026-08-04 (Asia/Hong_Kong)  
**Protocol:** `M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1`  
**Scope:** score-free implementation review; no GPU launch, endpoint R2 read, formal SUA access,
scorer call, or EvalAI action.  
**Current decision:** **NO-GO** until the Phase-C requirements below are implemented, tested, sealed,
and independently reviewed.

## 1. Phase-B evidence accepted by root

At the initial Phase-B review, before the later v4 query-only edit and semantic restoration, root
re-ran the combined Phase-A/Phase-B suites and obtained `52 passed`. The immutable Phase-B receipt
verifier then returned `PASS_PHASE_B_V3_SCORE_FREE_NO_GO`; the receipt SHA-256 is
`130af6fa2829122ed77127bf01571e0507520a15bec1a524a1c7df138bcdb39e`.

The original Phase-A and Phase-B verifiers now intentionally report the one-byte `v1` source
drift described in Section 4.1. Their historical PASS must not be presented as a current raw
verifier PASS; Phase-C execution requires the explicit EOF supersession plus zero drift for every
non-exception source.

Code review supports the following narrow conclusions:

- the SPINT-side versioned module requires six finite source-session metrics, excludes the outer
  session, and uses an explicit 35-epoch max-then-earlier selector;
- a T4 cell resolves only the same-fold/same-seed SPINT completion receipt and has no direct or
  default teacher-checkpoint escape path;
- the decoder helper can compare all 31 tensors at `pretrain`, `posttrain`, `reload`, and
  `prequery`, including name, shape, dtype, byte count, and raw-byte SHA-256;
- cell paths are deterministic in protocol/phase/arm/fold/seed and ownership is claimed by
  `O_EXCL`.

These are implementation prerequisites, not accuracy evidence or launch authorization.

## 2. Additional blocker found in root review: T4 metric cardinality

The Phase-B T4 wrapper still inherits the generic metric tables and aggregation in
`streaming_calibration_exp/src/models/streaming_calibration_module.py`.

That implementation creates metrics for all seven M2 held-in sessions, then silently skips any
metric whose `total <= 2` before averaging the remainder. The dedicated v3 datamodule normally
constructs the intended six source minival sessions, but the model does not fail if one source is
missing or empty. A checkpoint could therefore be selected from five sessions without violating
the current model-side code. The generic test path similarly accepts any held-in session and
averages all nonempty entries; it does not require exactly the one outer session.

This is not evidence that an existing result is wrong—no native-M2 post-33 GPU cell has run. It is
a prospective fail-closed gap and must be repaired before launch.

The minimum accepted repair is a new versioned T4 module/callback that:

1. requires exactly the ordered six source sessions for the fold;
2. requires integer `total > 2` and finite R2 for each source;
3. audits outer validation total as exactly zero and rejects any extra/outer validation batch;
4. records exactly one checkpoint/metric record for each epoch `0..11`;
5. selects maximum finite six-session equal-session mean, ties to the earlier epoch;
6. writes selector records/checkpoints only inside the owned canonical cell;
7. accepts exactly the unique outer session at test, with integer `total > 2` and finite R2;
8. has negative tests for missing/empty source, extra/outer source-validation input, nonfinite
   values, duplicate/missing epoch, incorrect tie handling, and non-outer test input.

The repair must be added to a new Phase-C source map; the Phase-B receipt and active C1 v3r2
source map remain immutable.

## 3. Remaining Phase-C launch requirements

In addition to the T4 cardinality repair, a GO candidate must prove all of the following.

### 3.1 Exact artifact and matrix closure

- Exact expected sets for ownership, start, completion/failure, checkpoints, selector records,
  resolved configs, per-cell results, and receipts; no foreign or extra protocol artifact.
- Canonical containment under the deterministic cell root; no symlink, path traversal, or
  timestamped secondary result directory.
- Exactly 14 terminal cells for Stage A before its seven paired deltas are read; exactly 42
  terminal cells for a full aggregate.
- Hash and byte-count links across the program receipt, configs, data manifest, SPINT completion
  receipt, T4 result, lifecycle evidence, and aggregate.

### 3.2 Crash-safe execution

- A launch authorization must be verifiably root-issued through a fixed trust anchor (for
  example, a sealed Ed25519 public key plus detached signature). A caller-written JSON boolean is
  not authorization. Bind program/data/evaluator/cost hashes, stage, arm/fold/seed allowlists,
  expiry, and a unique nonce; claim the nonce with `O_EXCL` and reject reuse or scope expansion.
- One launcher owns every cell through the existing `O_EXCL` claim.
- A shard containing seed 43 or 44 must require the same-root exact-14 Stage-A manifest and its
  hash-bound decision `continue_without_positive_claim`. Stage B must fail before that decision,
  after `seed42_severe_negative_futility_stop`, or when the decision belongs to another root.
- Normal completion writes `completed` exactly once; Python exception, child nonzero exit,
  `SIGINT`, or `SIGTERM` writes `failed` exactly once.
- A failed/partial cell is never retried, overwritten, reclassified, or silently omitted.
- T4 cannot start until its paired same-fold/same-seed SPINT completion receipt passes again at
  both preflight and model construction.

### 3.3 Deployment and endpoint evidence

- Outer counts are zero in source train, normalizer, and checkpoint selection.
- Both arms use chronological neural support `[0,33)`; query windows start at or after trial 33,
  including the complete 50-bin input history.
- T4 uses only eligible directional labels from the same support, records count/balance/rank, and
  has rank 3; centre/rest trials are not assigned an artificial direction.
- Query targets are scorer-only and never enter calibration, normalization, selection, or model
  update.
- Target calibration performs zero optimizer steps, zero backward calls, and zero parameter
  updates.
- Decoder lifecycle evidence is 31/31 bit-exact across all four stages, with no decoder parameter
  in the optimizer.

### 3.4 Cost and claim discipline

- Recompute parameters, source-training MACs, calibration operations/state, persistent state,
  online MACs/state, wall time, and peak memory from the exact resolved configs.
- Separate offline source-training cost from deployment calibration and streaming inference cost.
- Report `T4-SPINT` as supervised-versus-neural-only deployment utility: neural exposure is
  matched, label information is not.
- Stage-A non-futility is never called a positive trend. A positive result requires the complete
  42-cell gate frozen in the draft protocol.
- No formal SUA or new EvalAI action is implied by this local native-M2 development protocol.

### 3.5 Evaluator and delayed score opening

- A production GO cannot depend on an unspecified “future evaluator.” The exact evaluator source,
  executable identity, config, and tests must be in the Phase-C source map before authorization.
- The evaluator must load the selector-chosen checkpoint—not `last`, an epoch default, or an
  independently supplied path—and run the exact-one outer post-33 endpoint.
- It must write, with `O_EXCL`, both an opaque endpoint payload and a commitment that binds the
  payload's canonical path, byte count, SHA-256, cell identity, and schema. A digest without a
  retained payload is not a usable delayed-evaluation design.
- Cell finalization may copy and hash-bind the opaque payload but must not parse or display its R2.
- A separately source-mapped Stage-A opener may read scores only after the exact 14 seed-42 cells
  are sealed. It must compute all seven paired deltas and only the frozen severe-negative rule.
- A full-matrix opener may read scores only after exact 42-cell closure. It must emit the complete
  `seed x outer-session` table and all six frozen effectiveness gates. Neither opener may accept a
  missing/extra cell, alter the matrix, or add an arm after values are opened.
- Synthetic fixtures must prove payload substitution, selected-checkpoint substitution, early
  opening, wrong cardinality, duplicate identity, nonfinite score, and gate-boundary behavior all
  fail closed.

## 4. Additional root findings from the cached-deployment review

The first cached-deployment implementation exposed four further blockers. These were found before
any Phase-C GPU cell or endpoint opening, so they are prospective implementation corrections, not
changes made after seeing accuracy.

### 4.1 Preserve the sealed Phase-A implementation

The first query-only patch modified
`SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py`, which Phase A had sealed at size
`15,860` and SHA-256 `4a395ce2fe402ff94ce6e2c5550a2271d8cb9179367ad2413441b6433b8814b3`.
The semantic edits have been removed into a new `v4` datamodule. The restored source is currently
the exact sealed bytes with only the second EOF newline absent: appending one `\n` in memory
reconstructs the sealed size and SHA. Because the workspace editing path cannot preserve that
otherwise empty final line, Phase C must bind an append-only EOF-canonicalization supersession
that proves this byte relation and unchanged AST/compile semantics. It may not silently rewrite
the old receipt, declare zero drift, or weaken source closure.

All query-only, one-shot calibration, and deployment-cost behavior belongs in new versioned v4
files. The Phase-C program receipt must bind those files and reverify every other Phase-A/Phase-B
source at zero drift.

### 4.2 Separate source fit from target deployment by construction

`fit`/`validate` must approve and open exactly the six source calib plus six source minival files,
with zero outer and zero formal files. `test`/`predict` must open exactly the one outer-calibration
file, with zero source and zero formal files. The training stage must perform zero outer descriptor
fits and zero outer directional-label accesses.

The test loader must emit only `(neural_window, behavior_target, session_name)`. Raw support, T4
side features, and electrode identifiers must not be repeated in every query batch. Calibration is
claimed once, identity is computed once, and all query batches consume only the cached identity.
Ordinary and cached forward paths require exact-shape, three-seed, multiple-batch, zero-tolerance
parity tests plus mutation and shape negatives.

### 4.3 Make the deployment package source-data independent

The original test setup reloaded all source NWBs in order to recompute the source-only T4 feature
normalizer. That does not leak query labels, but it is not a self-contained target-session
deployment and its I/O is absent from the claimed calibration cost.

The accepted repair is a small, owned `deployment_constants.json` written at source-fit completion.
For T4 it contains the exact source-derived four-value mean/std and source-session identity; for
SPINT the normalizer is explicitly null. The evaluator must hash-bind this artifact, open no source
NWB, and use it with the one outer file. The completion receipt and finalizer must reject missing,
substituted, cross-fold, cross-seed, cross-arm, or nonfinite constants.

### 4.4 Report state lifetime and latency honestly

The batch-32 scorer path measures throughput, not single-window online latency. Phase C must bind
the exact batch sizes/window count, report total batched time and throughput, and mark a batched
latency claim forbidden. A separate post-score, no-target, cached `B=1` microbenchmark should use
five warmups and twenty CUDA-synchronized measurements and report median/p95.

State must be split into calibration input/accumulators, fitted descriptor, finalized cached
identity, and required post-finalize online state. For FP32 native M2, the cached identity is
`1 x 96 x 50 = 4,800` values or `19,200` bytes. The correct claim is that raw support is not
required by online decoding; the evaluation process may still retain the outer dataset in host
memory for scoring and must not falsely claim that all raw process memory was physically freed.
Descriptor fitting, integrity scans, one-time identity computation, batched throughput, and `B=1`
latency are separate cost fields.

### 4.5 Hash the large source-data closure once, not in every cell verifier

The stage-separated source audit binds fourteen unique native-M2 NWB files totaling
`14,057,075,549` bytes (`13.092 GiB`). One call to `validate_cost_supplement()` recursively invokes
the source-batch audit and rehashes this entire set. The current cell finalizer invokes that path
directly and again through source-cost validation; the immediate cell verifier repeats both.
Stage, matrix, decision, and opener paths recursively call the same cell verifier. Left unchanged,
the 42-cell lifecycle would perform multiple TiB of redundant source-file reads. It is also a
scope and accounting error: evaluator/finalizer/opener hosts would repeatedly open all fourteen
raw source NWBs outside source fit, contradicting the claimed self-contained target deployment
and omitting that I/O from deployment cost.

This is a launch blocker, not merely a performance note. A source-mapped preflight command must
perform the deep source/input hash audit exactly once and write an immutable verification receipt.
The program receipt, portable manifest, signed authorization, nonce claim, and cost supplement must
bind that receipt. Ordinary cell/stage/matrix/finalizer/opener validation may check the receipt and
supplement metadata/SHA plus the applicable audit-row SHA, but may not open or rehash the NWBs. A separate,
explicit deep-audit command must remain available for a deliberate final integrity audit.

### 4.6 Put authorization at every capability-bearing production entrypoint

The current recommended launcher verifies signed authorization before calling training and
evaluation, but the underlying training wrappers and production evaluator/arm workers can be
called directly after only creating or presenting a local cell-owner token. The Stage-A opener
core likewise accepts a caller-supplied `validated_authorization` mapping instead of validating
the signature and nonce itself. These paths can bypass the intended seed-42 delayed opening and
Stage-A stop boundary even though the normal CLI is well behaved.

Every production callable that can start training, open an outer session, compute an endpoint, or
read an opaque score must therefore verify the fixed project trust anchor, detached signature,
claimed single-use nonce, stage, observed host/GPU, shard, exact cell, program receipt, portable
manifest, and cost closure immediately before the protected action. It may not rely on an
upstream caller having done so. Arm-specific evaluator workers must either repeat the same
capability check or accept a non-forgeable, cell-specific capability created by the checked
entrypoint. Endpoint payloads, commitments, and decisions must bind the exact authorization and
claim metadata. Subprocess negatives must prove that a self-created owner token cannot directly
invoke a trainer, evaluator, worker, or opener.

The production verifier must not expose caller-selectable public-key paths or expected key hashes.
Test-key injection belongs in a separate test-only helper. Likewise, synthetic relaxation must not
be exposed on production finalizer/verifier CLIs or be capable of writing into a production
artifact root.

### 4.7 Require the exact checkpoint path and an exact pre-seal run inventory

The selector currently accepts any absolute `epoch_NNN.ckpt` located somewhere inside the run.
That is weaker than selecting the canonical checkpoint produced by the frozen source trainer.
Selector validation, evaluator loading, cell finalization, and later verification must all require
exact equality to `cell/run/checkpoints/epoch_{epoch:03d}.ckpt`. A checkpoint under Hydra output,
T4 secondary artifacts, or any other run subdirectory must fail even when its basename and epoch
look valid.

The run manifest also cannot discover arbitrary pre-existing regular files and then bless them by
hashing them. Define an arm-specific exact directory/file schema, including the permitted Hydra
and T4 secondary artifacts, and reject missing or extra files before sealing. Tests must inject an
alternate-location checkpoint and an extra file before finalization, not only after the manifest
has already been sealed.

### 4.8 Enforce the Stage-A directory boundary before any score is read

Stage-A finalization and verification currently inspect the fourteen expected seed-42 cells but do
not reject seed-43/44, foreign, or partial cell directories that already exist. They must enumerate
the canonical, non-symlink cell directory set and accept exactly the fourteen seed-42 SPINT/T4
cells. The positive lifecycle test must be ordered as `create exact14 -> seal/open Stage A -> sign
the decision -> create Stage B`; creating all 42 cells before Stage-A opening is invalid. A
pre-existing Stage-B cell must be a negative fixture.

The opener core itself must accept raw authorization/signature/program/portable/shard inputs and
perform the fixed-anchor verification before the first payload read. A forged mapping, fake
signature, or mismatched authorization path must fail under the production API, not only at the
CLI wrapper. The private signing key should remain external, but a checked-in, source-mapped,
deterministic handoff must specify exact decision bytes/metadata and write the detached Stage-A
decision signature with exclusive-create semantics; Stage B may not depend on an ad hoc signing
step.

### 4.9 Bind the paired SPINT teacher to the same signed cell root

The T4 teacher resolver checks receipt identity and checkpoint metadata but currently accepts a
same-fold/same-seed completion receipt from another root. The signed cell root and cell identity
must derive the only permitted paired SPINT completion path, and that local cell must pass strict
completion verification before constructing or loading the T4 teacher. A cross-root receipt with
otherwise matching identity is a required pre-Trainer negative.

### 4.10 Make the EOF exception part of the reachable provenance chain

Independent rechecking confirms the exception is factual: all 29 non-exception Phase-A source-map
entries and all 22 Phase-B entries match; the special v1 file is 15,859 bytes with SHA
`36a7743101e4e4e7aabcce0a4580b2c42299f7f96b1263aeb58821d9e01473b2`, and appending exactly one
`0x0A` byte reconstructs the sealed 15,860-byte SHA
`4a395ce2fe402ff94ce6e2c5550a2271d8cb9179367ad2413441b6433b8814b3` with equal AST and successful
compilation. However, the proof remains unreachable from the current program/portable/auth/runtime
chain.

Phase C therefore needs an exact program-receipt/source-map writer and validator. The EOF proof
must be an explicit hash closure, alongside the Phase-A/Phase-B zero-drift checks, through program
receipt, portable manifest, signed authorization/claim, pipeline/evaluator, and final matrix
verification. Omission, substitution, and wrong-root proof fixtures must fail closed.

## 5. Review outcome

Phase-B remains an immutable historical NO-GO prerequisite receipt. Phase C may extend it in new
versioned files only through the explicit one-byte EOF supersession; it may not claim the original
verifier currently passes unchanged. GPU launch is permitted only after the new receipt proves
every requirement above, root re-runs the complete tests and source-map verifier, and an
independent review finds no remaining critical/high blocker.
