# Work Order: TF-SR Seed-43 Phase-E Replication Addendum

Status: implementation and no-data review only. This document does not authorize target access,
CUDA, score publication, seed 44, or any modification of either live training process.

Owner split: Terra owns the additive implementation. Root/Sol owns scientific design and review.
Luna remains read-only on the live seed-42/GPU1 and seed-43/GPU0 processes.

## 1. Purpose

Seed 43 is already running as `TFSR_B3ST4_DDROP_SEED43` under accelerated build v2. The frozen
Phase-E scorer is intentionally hard-bound to the seed-42 Phase-D v2 terminal and cannot consume
the seed-43 checkpoint schema, lineage, or build disclosure. Build a separate additive replication
addendum so the completed seed-43 SWA can later be scored against the exact already-terminal seed-42
input authority, estimator, Cell-D replay, and controls.

This is not permission to score now. Both live training roots remain untouched.

## 2. Scientific status and ordering

Seed 42 remains the pre-registered discovery/gating run. Its frozen Phase-E score must terminalize
first. The seed-43 scorer must descriptor-load and bind that exact seed-42 score terminal before it
can cross the data boundary.

- If seed 42 is `CLEAR_GO`, seed 43 is a confirmatory replication.
- If seed 42 is `HOLD` or `STOP`, seed 43 may still be reported after completion as
  `NON_GOVERNING_POST_AUTHORIZATION_REPLICATION`, but it cannot revise or rescue the seed-42 gate.
- A two-seed summary is descriptive only. It is not a three-seed superiority claim and does not
  authorize seed 44.

## 3. Immutable training evidence

The successor must use the seed-43 route's own frozen validators, never coerce its artifacts into
the seed-42 schema:

- canonical root `tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_train_v1`;
- all 48 immutable epoch receipts;
- checkpoint epochs 44--47;
- exact terminal/SWA/checkpoint launch binding, including seed-43 lineage;
- exact seed-43 implementation closure and throughput-v2 evidence receipt;
- build label `v2_accelerated_jit_scripted_step`;
- launch closure equals final closure;
- source-only true and target/formal/scientific/score false throughout training.

The historical sentence `Only the seed differs` is not an acceptable result label because seed 42
used build v1 and seed 43 uses build v2. Receipts and reports must say `seed43, accelerated build
v2` and preserve the measured FP32 trajectory-divergence disclosure.

## 4. Matched score contract

Reuse the frozen seed-42 Phase-E authorities and science without editing its closure:

- exact six within-development and fifteen external sessions;
- exact no-cache held-FD/private-snapshot input boundary;
- exact normalized M30 T4 and SUA unit-axis proof;
- exact final-bin targets and query masks;
- last-bin, variance-weighted R2; equal session weighting;
- exact sealed seed-42 Phase-E input authority and Cell-D replay evidence as upstream parity
  blockers; do not rerun Cell D in the addendum;
- TF-SR aligned, zero-T4, and cyclic wrong-pair T4 modes;
- no target optimizer, backward, update, checkpoint selection, session selection, or formal data.

The seed-43 SWA must be loaded strictly into the frozen `TFSRDecoder`. For every real scoring batch,
run both the ordinary eager eval forward and accelerated build-v2 eval forward under no-grad and
require bitwise-equal predictions before accepting the eager prediction for the common matched
metric. Record both prediction digests and the equality proof. This isolates the accelerated build
to training rather than allowing a second scoring implementation to enter the comparison.

The addendum must descriptor-load the seed-42 `input_authority.json`, `score.json`, and
`terminal.json`, validate them through the frozen seed-42 scorer, and bind their exact SHAs. Its live
input reconstruction must exact-match the seed-42 input records, target/mask digests, rosters, and
normalizers before a seed-43 forward. The addendum may publish its own target-free preflight/root
authorization and score root, but must not reuse, overwrite, or mutate the seed-42 Phase-E
authority/result roots.

### Required additive seam

Do not extract a generic scorer core, edit the frozen seed-42 scorer, mutate its globals, or copy its
4,676-line lifecycle. Implement the addendum by importing the frozen seed-42 scorer as an immutable
dependency and subclassing `PhysicalMatchedScoreBackend`:

- inherit GPU1 attestation, no-cache input parsing, held-FD lifecycle, typed T4 controls, metric, and
  `_score_system_surface`;
- override only model/SWA loading so it validates the seed-43 terminal with seed-43's own
  `train_43` validators and constructs one frozen `TFSRDecoder` plus the accelerated eval binding;
- wrap that model so every forward compares eager versus accelerated-v2 bitwise before returning
  the eager output, and accumulate a separate exact build-parity manifest;
- implement a small addendum-specific receipt lifecycle containing only seed-43 TF-SR
  aligned/zero/wrong-pair evidence, upstream seed-42 bindings, parity manifest, resources, failure,
  and terminal. Do not call or copy the seed-42 Cell-D scoring path.

If subclassing cannot reuse the frozen input/mode implementation without monkeypatching a module
global or silently accepting a seed-42 training binding, stop again and identify the exact method.

## 5. Result semantics

Report, without changing any seed-42 threshold:

- seed43 minus the exact upstream seed-42 Cell-D replay on within and external surfaces;
- external median and positive-session count;
- aligned/zero/wrong-pair controls;
- pooled A2 comparison as non-gating context;
- exact per-session table;
- seed42/seed43 descriptive mean and build labels, only after both individual score terminals exist.

Do not silently pool build v1 and build v2 as if build were absent. A strict same-build three-seed
claim would require seed 44 under v2 and, if demanded for the paper, a separate seed42-v2 bridge.

## 6. Additive implementation boundary

Preferred owned files:

- `tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/score_43.py`;
- `tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score.py`;
- `tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_score_v1.py`.

Do not edit the live seed-42/seed-43 trainers, model, accelerated forward, frozen seed-42 scorer,
existing tests, result roots, or shared data code. Bind every reused runtime dependency explicitly;
do not use a recursive source glob.

The zero-argument CLI must remain static/no-Torch/no-data/no-CUDA/no-write. Public execution flags
must fail closed without an in-process root capability. Implementation review may use only synthetic
CPU/mocked artifacts. Do not open NWB files, deserialize a real checkpoint, initialize CUDA, create
canonical authority/result roots, or publish a receipt.

## 7. Required no-data tests

At minimum prove:

1. seed-42 training artifacts cannot satisfy the seed-43 validator and vice versa;
2. incomplete, failed, nonterminal, wrong-build, wrong-lineage, wrong-closure, wrong-checkpoint, or
   wrong-SWA seed-43 chains fail before any asset/model/CUDA route;
3. the exact seed-42 Phase-E score terminal is a mandatory upstream binding;
4. no seed-42 result or authority path can be selected as the seed-43 output root;
5. eager/build-v2 real-batch prediction inequality fails before metric accumulation, and the
   parity manifest binds every scored batch rather than only one synthetic probe;
6. fixed assets, rosters, target/mask digests, metric semantics, and the upstream seed-42
   input-authority/Cell-D replay cannot drift;
7. `HOLD`/`STOP` seed-42 status forces the non-governing seed-43 label and cannot be rewritten;
8. the two-seed summary rejects missing individual terminals, duplicate seeds, hidden build labels,
   or any three-seed/superiority claim;
9. failure publication is immutable and cannot coexist with a terminal;
10. dry import and dry CLI touch no Torch, data, checkpoint, CUDA, authority root, or score root;
11. the implementation subclasses the frozen physical backend without modifying/monkeypatching its
    globals, never invokes `score_cell_d`, and rejects any seed-42 training artifact offered as the
    seed-43 SWA chain.

Freeze at a no-data/no-CUDA review boundary and report exact file SHAs, explicit closure, focused
tests, dry-plan output, and remaining live blockers. Root review is required before any later
authority publication or score execution.
