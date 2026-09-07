> **SUPERSEDED — historical review evidence only, not current authority.** The NO-GO blockers below were addressed by later V4 launches, V5 device repair, and the r9 launch chain.
> Current launch state: r9 in [`ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](ACTIVE_EXPERIMENT_CONTROL_BOARD.md).
> Preserved as append-only audit evidence; content unchanged.

# Native-M2 post-33 Phase-C v4 adversarial review

**Date:** 2026-08-04 HKT  
**Scope:** production-reachable training, evaluation, delayed opening, provenance, and cost paths  
**Endpoint state:** no Phase-C GPU cell launched; no opaque endpoint value opened; no formal/EvalAI data accessed  
**Verdict:** **NO-GO until every CRITICAL and HIGH item below is closed and independently retested**

## 1. Evidence baseline

The implementation has useful scientific and deployment foundations: source fit and target
deployment use separate v4 datamodules; source fit opens twelve source files and no outer/formal
file; target test opens one outer session and no source/formal file; the T4 normalizer is serialized;
identity is finalized once and cached; query batches no longer repeat support or side features;
batch-32 throughput and post-score batch-1 latency are distinguished. Focused implementation tests
passed 36/36, and root's combined Phase-A/B/C suite passed 88/88 before this review.

Those tests do not authorize a launch. This review followed every public production entrypoint,
not only the intended launcher, and found reachable bypasses plus a multi-TiB source-data validation
loop. The correct conclusion is that the scientific protocol remains frozen but the execution
capability model and artifact closure require another versioned repair.

The historical EOF exception was independently rechecked. All 29 non-exception Phase-A source-map
entries and all 22 Phase-B entries match. The special v1 file is 15,859 bytes with SHA-256
`36a7743101e4e4e7aabcce0a4580b2c42299f7f96b1263aeb58821d9e01473b2`; appending exactly byte
`0x0A` in memory produces 15,860 bytes and sealed SHA-256
`4a395ce2fe402ff94ce6e2c5550a2271d8cb9179367ad2413441b6433b8814b3`. AST equality and compilation
both pass. The proof is factually valid but is not yet part of the reachable authorization closure.

## 2. CRITICAL findings

### C1. Evaluator and worker entrypoints bypass signed authorization

`sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py:32-95` accepts cell identity and
an owner token, validates local output paths/ownership, then directly launches an arm worker. It
does not verify the signed authorization, claimed nonce, stage, observed host/GPU, shard, or exact
cell scope. The SPINT worker (`SPINT-main/src/evaluate_post33_phase_c_v4.py:28-109`) and T4 worker
(`streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py:30-122`) likewise reach
`Trainer.test`, compute the outer score, and write the opaque payload after only local ownership and
path checks. Because `claim_cell` itself is not an authorization gate, a caller can create a local
cell and bypass the delayed-score boundary.

Required repair:

- the top evaluator must validate a fixed-anchor signed, single-use, exact-cell capability
  immediately before opening the outer session;
- arm workers must repeat that validation or receive a non-forgeable per-cell capability that
  cannot be created by an external direct caller;
- payload and commitment must bind authorization and nonce-claim metadata;
- a subprocess fixture that creates only a local owner and invokes the evaluator or worker must
  fail before datamodule construction, CUDA, or `Trainer.test`.

### C2. Stage-A and full opener cores bypass opening authorization

The Stage-A core in `sua_exploration/mc_maze/m2_native_post33_openers_v4.py:78-124` accepts a
caller-injected `validated_authorization` mapping, checks only a few fields, then reads fourteen
scores. It does not verify the signature, fixed anchor, exact authorization/signature paths, or a
consumed nonce. Existing synthetic tests demonstrate that a fabricated mapping can reach this
path. The full opener (`openers_v4.py:248-288`) verifies matrix structure and the Stage-A decision,
but has no signed opening capability at all before reading all 42 payloads.

Required repair:

- define a distinct `scope=opening`, single-use capability for Stage-A and full opening;
- opener cores, rather than only their CLIs, must accept raw authorization/signature/program/
  portable/shard/claim inputs, verify the fixed trust anchor and exact scope, then atomically consume
  the opening nonce before the first payload read;
- Stage-A decisions and full aggregates must bind the exact authorization and claim metadata;
- forged mapping, fake signature, mismatched path, reused nonce, wrong stage, and missing opening
  capability must fail in production-mode direct-call tests.

### C3. Production trust anchor is caller-selectable

`verify_signed_authorization` exposes caller-overridable `public_key_path` and
`expected_public_key_sha256` parameters (`m2_native_post33_authorization_v4.py:83-95`), and bound
claim revalidation exposes the same override (`:270-304`). A direct Python caller can therefore
substitute a self-generated trust anchor. This compounds C1/C2.

Required repair:

- production verifiers and every production capability-bearing entrypoint must have no
  caller-selectable anchor path or digest;
- generated-key injection must live in a clearly separate test-only helper and may not write into
  a production artifact root;
- public `--allow-synthetic` production finalizer/verifier surfaces must be removed or quarantined
  into an unmistakably test-only fixture path.

## 3. HIGH findings

### H1. Training wrappers are directly runnable without capability checks

Both v4 training wrappers validate local owner identity and invoke the legacy GPU trainer, but take
no signed authorization/signature/shard/nonce/stage inputs. The intended cell pipeline checks them,
yet a direct wrapper invocation is reachable. Each wrapper or the immediately downstream immutable
launch gate must perform the same fixed-anchor, claimed-nonce, host/GPU/shard/cell/program validation
before CUDA or Trainer construction. Add self-created-owner direct-wrapper subprocess negatives.

### H2. Raw source data is repeatedly rehashed throughout deployment/finalization

The source audit binds fourteen unique NWBs totaling `14,057,075,549` bytes (`13.092 GiB`).
`validate_cost_supplement()` calls `validate_source_batch_audit()`, which hashes every unique NWB
(`m2_native_post33_cost_v4.py:81-224`, hashes at `:218-220`). Cell finalization calls the supplement
validator directly and again through source-cost validation; `verify_cell_exact` repeats both.
Launchers, stage/matrix finalizers, decisions, and openers recursively call cell verification.
The full lifecycle would therefore read multiple TiB of source data.

This is both an operational and semantic failure: target deployment/finalization hosts repeatedly
open all raw source NWBs outside source fit, and that I/O is not included in deployment cost.

Required repair:

- a source-mapped preflight command performs the deep NWB audit once and writes an immutable receipt;
- program/portable/auth/claim/cost artifacts bind that receipt;
- ordinary cell/stage/matrix/finalizer/opener validation checks only receipt/supplement metadata and
  SHA plus the applicable audit-row SHA; it must never open an NWB;
- retain a separate explicit deep-audit command for a deliberate final integrity check;
- instrument tests so ordinary runtime validation fails if the NWB opener/hash path is reached.

### H3. EOF supersession and Phase-A/B closure are not reachable from authorization

Only `verify_m2_native_post33_upstream_eof_canonicalization_v4.py` creates/checks the EOF proof.
The current portable manifest contains only program-receipt and Phase-A data-audit closures, and the
authorization code metadata-hashes a supplied program receipt without validating its source-map
semantics. No checked-in Phase-C program-receipt writer/validator currently guarantees that the EOF
proof, Phase-A/B zero-drift evidence, deep-data-audit receipt, evaluator, and runtime code are all
bound.

Required repair:

- deterministic write-once Phase-C program-receipt/source-map writer and exact validator;
- EOF proof and Phase-A/B zero-drift checks explicitly transit through program receipt, portable
  manifest, signed auth/claim, pipeline, trainers/evaluator/openers, and final matrix verification;
- omission, substitution, wrong-root, changed-source, and unlisted-entry negatives.

### H4. Selector can escape the canonical checkpoint directory

`validate_selector_payload` requires an absolute basename such as `epoch_NNN.ckpt`, while finalizer
and verifier require only that it lies somewhere under `run`. A same-named file under Hydra or T4
secondary artifacts can therefore be selected while the expected canonical checkpoint set remains
present.

Required repair: selector, evaluator, finalizer, and verifier must all require exact equality to
`cell/run/checkpoints/epoch_{epoch:03d}.ckpt`. Add alternate-location checkpoint negatives under
Hydra and T4 secondary artifacts.

### H5. Pre-seal run inventory is discovered, not constrained

`build_run_manifest` records arbitrary regular files already present; `verify_run_manifest` checks
core files and checkpoint prefixes but does not enforce an arm-specific exact tree. An extra file
created before sealing is consequently blessed. Define the exact SPINT/T4 run directory/file schema,
including permitted Hydra and secondary artifacts, and reject extras before sealing. Add a
pre-finalization extra-file negative.

### H6. Stage-A does not require an exact seed-42 cell directory set

Stage-A finalization and verification check the fourteen expected seed-42 cells but do not enumerate
`cells/` to reject seed43/44, foreign, partial, or symlink cell directories. The full finalizer does
perform an exact scan. Stage-A must accept exactly the fourteen canonical non-symlink seed-42
SPINT/T4 directories. Correct the lifecycle test to `exact14 -> seal/open Stage A -> sign decision
-> create Stage B`, and add a pre-existing Stage-B cell negative.

### H7. Paired SPINT teacher can come from another root

`resolve_phase_c_paired_spint_teacher()` validates receipt identity, checkpoint metadata, and policy,
but receives no signed cell root and does not derive/strict-verify the same-root SPINT completion
path before T4 model construction. Finalization detects a cross-root receipt only after evaluation.
The constructor/preflight must derive the expected paired path from signed root and cell identity and
run strict same-root completion verification before loading the teacher. Add a cross-root
same-identity negative that fails before Trainer work.

### H8. Signed GPU scope is not checked against the actual CUDA device

Authorization/shard artifacts contain `gpu_id`, but `validate_observed_host` checks only hostname.
The matrix launcher normally sets `CUDA_VISIBLE_DEVICES`; the cell pipeline and direct entrypoints
can inherit another, empty, or multi-device environment without comparing it to signed physical
GPU scope.

Required repair: define a canonical physical-ID mapping and validate a unique visible/active device
against signed `gpu_id` immediately before every Trainer/evaluator launch. Record observed mapping in
runtime evidence. Add valid-authorization wrong/empty/multi `CUDA_VISIBLE_DEVICES` negatives.

### H9. Stage-A decision signing has no checked-in operational handoff

Production validation and Stage-B authorization require a detached signature over the Stage-A
decision, but only tests currently demonstrate signing. Keeping the private key outside the
workspace is correct. Add a source-mapped deterministic handoff/protocol that takes exact decision
bytes and expected metadata and writes an exclusive-create detached signature. Ad hoc manual signing
must not be the only path to Stage B.

## 4. Required test matrix before reconsidering GO

The repaired implementation must retain the existing Phase-A/B/C suite and add production-mode
subprocess tests for:

1. direct trainer, evaluator, arm-worker, Stage-A opener, and full opener invocation without a valid
   exact capability;
2. forged trust anchor, fake/mismatched signature, absent/reused/wrong-scope nonce, wrong stage,
   wrong host/GPU/shard/cell/root;
3. pre-existing seed43/44/foreign/partial/symlink cells before Stage-A opening;
4. alternate checkpoint locations and extra pre-seal run files;
5. cross-root same-identity paired teacher receipt;
6. EOF/deep-audit/program-receipt omission, substitution, drift, and wrong-root artifacts;
7. proof that ordinary runtime/finalizer/opener paths never open or hash raw NWBs;
8. the valid lifecycle in the only permitted order: exact14 seed42 cells, sealed Stage-A opening,
   detached decision signature, then Stage-B cells;
9. a real selected-checkpoint evaluator launch/dry-run fixture that exercises the production
   capability chain without opening a score or formal endpoint.

Run tests with plugin autoload disabled because the environment's unrelated DANDI plugin stack is
incompatible with the installed Click version. Passing narrow unit tests is insufficient: root must
rerun the complete source-map/program verifier, combined Phase-A/B/C suite, subprocess adversarial
matrix, and an independent read-only review against the exact final source closure.

## 5. Launch order after repair

1. Generate and deep-verify the source-data audit once.
2. Generate exact program receipt/source map and portable/cost artifacts.
3. Run all root tests and the production-chain dry-run.
4. Run an independent adversarial review on the unchanged closure.
5. After C1 releases the GPUs, run the two clean synthetic capacity benchmarks and seal cost evidence.
6. Root issues only the seed42 Stage-A authorization for exactly fourteen cells.
7. Open Stage A once with an independent single-use opening capability.
8. If the frozen severe-negative rule stops, do not create Stage-B cells. If it continues, sign the
   decision through the checked handoff and issue a new Stage-B authorization for seeds43/44.
9. Full 42-cell effectiveness is opened once and reported under the preregistered gates. No formal
   SUA or EvalAI action is implied.

Until this sequence is implemented and independently verified, Phase-C remains score-free **NO-GO**.
