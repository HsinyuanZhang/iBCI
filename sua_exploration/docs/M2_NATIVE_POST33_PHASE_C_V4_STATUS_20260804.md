# Native M2 post-33 Phase-C v4 historical pre-r3 status (2026-08-04)

> **Historical record, superseded for launch control.** The `NO-GO` below records the state before
> the r3 program/cost/portable/shard closure and root-signed capabilities existed. The current
> authoritative state is **fresh r3 signed and independently verified GO, waiting for its bound
> local RTX 3090 GPUs**. Use
> [`ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](ACTIVE_EXPERIMENT_CONTROL_BOARD.md) and the immutable r3
> receipts under
> `results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/`; do not use this historical
> page to authorize, deny, or reproduce an r3 launch.

## Outcome

At this historical checkpoint, Phase-C v4 had an implemented score-sealed execution path, delayed
score opening, signed authorization, exact staged sequencing, and CPU-recomputed deployment-cost
receipt, but its decision was still **NO-GO**. No Phase-C GPU training, endpoint evaluation,
formal SUA access, EvalAI action, or new endpoint R² read occurred while building or testing this
phase.

The final root review addendum bound by the forthcoming receipt is:

- `sua_exploration/docs/M2_NATIVE_POST33_PHASE_C_ROOT_REVIEW_ADDENDUM_20260804.md`
- SHA-256 `1074fd0f0ef22b045227cbdcfba3e2ae3ec54a7a23f240651a10de0cc455d2cd`

The immutable Phase-B and active C1 v3r2 receipts remain unchanged.

## Implemented contracts

### Cell and matrix closure

- deterministic `protocol/phase/arm/fold/seed` cell paths;
- `O_CREAT|O_EXCL` ownership, started, failed, completed, payload, commitment, and final outputs;
- exact checkpoint budgets: SPINT epochs `0..34`, T4 epochs `0..11`;
- source-only selector with exact six sessions, finite per-session R², total greater than two,
  outer validation total zero, and deterministic earlier-epoch tie breaking;
- canonical containment, no symlink components, no timestamped secondary result directories,
  exact control/run/sealed artifact sets, byte counts, and SHA-256 closure;
- exact 14-cell Stage-A seal and exact 42-cell/21-pair full-matrix seal;
- failure on missing, extra, substituted, duplicated, incomplete, or cross-cell artifacts.

### Production evaluator and delayed opening

- the versioned evaluator derives the checkpoint only from the explicit selector record;
- arm-specific workers run the exact-one outer post-33 endpoint;
- the evaluator retains a deterministic, canonical, write-once endpoint payload and writes a
  commitment binding payload path, size, SHA-256, and cell identity;
- the cell finalizer copies and hash-binds the payload without parsing or displaying its R²;
- the Stage-A opener reads exactly seven seed-42 pairs only after exact-14 closure and applies only
  `(mean42 <= -0.03) OR (pos42 <= 1)`;
- `continue_without_positive_claim` is the only outcome that permits Stage B; it is not a positive
  trend or accuracy claim;
- the full opener runs only after exact-42 closure and emits the 21-row seed-by-session table plus
  all six frozen gates;
- the two-way bootstrap is frozen to 100,000 independent seed/session resamples, NumPy PCG64 seed
  `20260804`, and the linear 0.025 quantile.

### Root authorization and staged sequencing

- trust anchor:
  `sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem`, SHA-256
  `ff9d1b2b985c9cd8c2697cfb0e1e5353b8c19d1f3ff094d2987aa1537c79a375`;
- detached Ed25519 signature over the exact authorization JSON bytes;
- no private-key search, creation, loading, or storage in Phase-C;
- signed binding of program, portable manifest, shard, evaluator, cost receipt, public key,
  absolute cell root, observed hostname, GPU, stage, folds, seeds, arm order, expiry, and unique
  256-bit nonce;
- nonce is claimed once per host/shard with `O_EXCL`; the cell pipeline independently revalidates
  signature, shard membership, and the exact nonce claim, so it cannot be invoked directly for an
  out-of-scope cell;
- Stage-A and Stage-B finalizers revalidate bound authorization bytes/signatures and require exact,
  non-overlapping shard coverage;
- a Stage-B authorization additionally binds the same-root Stage-A decision and its detached root
  signature;
- Stage-B launch fails before a signed `continue_without_positive_claim`, after a severe-negative
  stop, for a forged/wrong-root decision, or for seed/fold scope expansion.

### T4 runtime evidence

- new T4 metric rejects missing/extra/nonfinite/low-total source or outer sessions;
- result evidence binds support `[0,33)`, query start 33, complete 50-bin history, exact-one outer
  session, finite metric total, eligible directional-label count, centre/rest exclusion, design
  rank three, direction balance, and zero query-target leakage;
- target calibration has zero optimizer steps, backward calls, and updated parameter tensors;
- decoder evidence is 31/31 name/shape/dtype/byte-count/SHA closure across pretrain, posttrain,
  checkpoint reload, and immediately prequery, with zero `requires_grad` and optimizer overlap.

## CPU cost receipt

Receipt:
`sua_exploration/results/m2_native_post33_phase_c_v4_cost_20260804/cost_receipt.json`
(SHA-256 `36d7631ca0747f7d0efcd16f15d99b21c07fa4c297f5e268333c0c95b9f14bc7`).

At the frozen reference shape (96 channels, 33 support trials, 100-bin trials, 50-bin windows):

- common decoder: 4,594,888 stored parameters and 84,021,248 MAC/window;
- SPINT identity calibration: 1,875,935,232 MAC and 235,008 bytes estimated peak streaming live
  calibration state;
- T4 arm: 18,290 trainable source encoder parameters, 21,393,408 encoder MAC/session, 10,692 AC4
  fit MAC, and 64,552 bytes estimated peak streaming calibration live state;
- neural exposure is matched, but label information is not: T4 is supervised and SPINT is
  neural-only.

These are formula/source-bound FP32 estimates, not measured production timing.

## Verification completed

- Phase-C focused tests: 30 passed, zero failed at the latest pre-document run;
- combined Phase-A/Phase-B/Phase-C regression before the last authorization hardening: 81 passed,
  zero failed; it must be rerun after this document and final receipt scripts are complete;
- all new Python sources compile;
- focused negatives cover checkpoint/payload/result/pair substitution, symlinks, extra files,
  timestamp directories, missing cells, Stage-A/full cardinality, nonfinite scores, selector ties,
  SIGINT/SIGTERM fail-once, invalid/expired/tampered signatures, nonce replay, host mismatch,
  fold/seed expansion, forged claims, missing/duplicate shard coverage, early Stage B, severe stop,
  wrong-root decisions, and gate boundaries.

## Remaining blockers before any GPU authorization

1. Rerun the full Phase-A/B/C regression and receipt verifier after the Phase-C source map is
   frozen.
2. Independently review the evaluator worker, opener statistics, signature/claim coverage, and
   multi-host absolute-path assumptions.
3. Install no authorization by default. A human/root-controlled signer must issue the detached
   Stage-A authorization; Phase-C cannot self-authorize.
4. The production evaluator path has not yet been exercised on a real Phase-C cell. The first
   authorized cell must prove actual resolved-config loading, selected-checkpoint restore, outer
   query, and runtime evidence without any path fallback.
5. Source-training MACs by actual batch cardinality, source-training wall time, deployment
   calibration/inference wall time, and peak host/device memory remain production runtime fields.
   They must be collected by the authorized run and may not be replaced with estimates.
6. No positive claim is possible until the exact-42 full opener completes all six gates.

Until these are resolved and an independent reviewer returns no critical/high issue, the correct
status is `NO_GO_INDEPENDENT_REVIEW_AND_PRODUCTION_RUNTIME_EVIDENCE_REQUIRED`.
