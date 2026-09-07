# Work Order: PACD Matched Score V2 Mixed Lineage

Date: 2026-08-31

Status: authorized for no-data/no-CUDA implementation; live scoring remains
forbidden until valid P0 V3 and P1/P2 V4 producer terminals and exact reviewed
body literals exist.

## 1. Purpose

The PACD training lineage is intentionally mixed:

```text
P0 -> PACD full V3, direct V2-failure predecessor
P1 -> PACD admission/multi-GPU V4, held V3-P0 predecessor
P2 -> PACD admission/multi-GPU V4, held V3-P0 predecessor
```

The existing PACD matched scorer V1 requires P0/P1/P2 all to be V3 producers.
It must not be silently relaxed. This work order creates an additive V2 scorer
that changes producer binding and lifecycle identity only while reusing the
entire reviewed V1 scoring science and atomic publication protocol.

The scorer remains CPU-only. It must not use or query either GPU and therefore
must not interfere with PACD training or the separate CDM replay.

## 2. Frozen identities

```text
CELL   = PACD_MATCHED_SCORE_V2_MIXED_LINEAGE
SCHEMA = pacd_matched_score_v2_mixed_lineage
```

Canonical roots:

```text
RESULT_ROOT_RELATIVE =
  tfpd_exploration/results/paired_anchored_calibration_dropout_score_v2_mixed_lineage

AUTHORITY_ROOT_RELATIVE =
  tfpd_exploration/results/paired_anchored_calibration_dropout_score_v2_mixed_lineage_authority
```

The roots must be fresh, canonical, non-symlink, distinct, parent-inode bound,
and published through the existing atomic-directory protocol. Score V1 roots
must never be reused or modified.

The training protocol authority is:

```text
tfpd_exploration/docs/WORKORDER_PACD_P1_P2_ADMISSION_MULTIGPU_V4_20260831.md
SHA-256 34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3
```

This hash is review authority. The score implementation must validate producer
receipts and exact live literals; it must not infer truth merely from this
Markdown file.

## 3. Allowed implementation scope

New owned files:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/__init__.py
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/plan.py
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/binding.py
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/smoke.py
tfpd_exploration/scripts/run_pacd_matched_score_v2_mixed_lineage.py
tfpd_exploration/tests/test_pacd_matched_score_v2_mixed_lineage.py
```

Two narrow backward-compatible shared edits are authorized:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/binding.py
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/lifecycle.py
```

The shared edits may only:

1. extract strict reusable V3 arm validation primitives;
2. add a frozen score execution profile for plan/root/closure/binding-validator
   selection;
3. thread that immutable profile through the existing single authority,
   capability, attempt, atomic commit and failure lifecycle.

No edit is authorized to:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/score.py
tfpd_exploration/src/calibration_gap_v1/p4_stream_stats.py
tfpd_exploration/src/cal_aug_v1/deployment.py
any model/parser/metric/training file
any existing result/authority root
```

No scorer loop, parser, materializer, model swap, metric, bootstrap, gate, or
atomic publication code may be copied into V2.

## 4. Science contract: exact V1 reuse

V2 must inherit the following without alteration:

| Field | Exact requirement |
|---|---|
| systems | `P0, P1, P2, T0, C1, SD` |
| surfaces | `within, external` |
| budgets | `M30, M10, M4` in that canonical order |
| regimes | `honest_total, activity_isolation` |
| sessions | 6 within + 15 external |
| rows | `6 × 2 × 3 × 2 × 21 = 756` |
| materialization | session outer; each session materialized once |
| same-input law | all six systems share the exact record authority per cell |
| activity isolation | literal `selected_by_budget[30]` activity with budget-M ridge side |
| model use | strict fresh SWA swap, eval/no-dropout/no-grad |
| forward | repeated prediction equality and unchanged state |
| metric | governed last-bin `[49]` variance-weighted R² |
| bootstrap | fixed seed 42, 2,000 session-level replicates |
| summaries/gates | exact V1 recomputation from canonical rows |
| target updates | optimizer/backward/update all zero |
| runtime | CPU only, `CUDA_VISIBLE_DEVICES=""`, CUDA uninitialized |

Historical T0, C1 and sealed Cell-D descriptors and route-specific codecs are
reused exactly. V2 does not create new comparator interpretations.

## 5. Shared `ScoreExecutionProfile`

`lifecycle.py` must define an immutable typed profile with at least:

```text
identity
cell
schema
score_root_relative
authority_root_relative
bound_patterns
expected_row_count
binding_type_identity
producer_validator
```

The default profile remains the current V1 contract:

```text
V1_SCORE_PROFILE
  -> current V1 cell/schema/roots/closure
  -> current PACDProducerBinding
  -> current all-V3 producer validator
```

The V2 wrapper selects one module-private frozen profile:

```text
V2_MIXED_SCORE_PROFILE
  -> V2 cell/schema/roots/closure
  -> MixedPACDProducerBinding
  -> strict P0-V3 plus P1/P2-V4 validator
```

The following shared operations must read the supplied profile and default to
V1 when no profile is provided:

```text
source_closure
prepare_live_authority
_revalidate_capability
execute_atomic
attempt/terminal/failure codec identity
canonical root checks
expected row count
producer witness validation
```

The opaque one-shot capability binds profile identity, binding digest, score
and authority root witnesses, closure, producer witness, comparator witness,
CPU profile and runtime-attestor identity. A V1 capability cannot execute V2;
a V2 capability cannot execute V1 or a different root/binding/profile.

The public CLI cannot accept a profile, binding validator, root override,
runtime callback, producer literal, CPU attestor, or authorization boolean.
Dry mode is inert.

## 6. Mixed producer binding

### 6.1 Binding type

V2 defines an immutable `MixedPACDProducerBinding` with three route-specific
arms:

```text
P0: V3P0ProducerArm
P1: V4AdmissionProducerArm
P2: V4AdmissionProducerArm
```

The binding also includes the exact accepted V2 smoke predecessor, the exact
V2 full-failure predecessor, mode (`synthetic` or `live`) and all future body
SHA literals required by Sections 6.2 and 6.3.

Synthetic bindings are test-only and can never mint a live capability. Live
mode is impossible while any producer literal is absent, guessed, malformed
or not exact-reviewed.

### 6.2 P0: strict historical V3 producer

Canonical root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42
```

Required schema/status:

```text
pacd_matched_full_training_v3_terminal
PACD_FULL_TRAINING_COMPLETE
```

The V3 P0 validator must use held directory FDs and `O_NOFOLLOW`, exact body
and sidecar SHA, regular mode-`0444` leaves, exact 58-body/116-leaf topology,
and no failure leaf. It must validate:

- exact attempt/launch/source-authority/terminal links;
- P0 arm, M30/M30 identity and root;
- seed 42, batch 32, workers 0;
- 48 × 33,925 = 1,628,400 optimizer steps;
- exact 27-session source-only authority and no target access;
- 48 ordered epoch receipts and stable sampler-order evidence;
- zero P0 prediction and identity mismatches;
- zero RNG/prefix/finiteness/decoder-gradient violations;
- typed accepted-zero encoder aggregate law;
- exact CP44--47 order and links;
- final-four SWA strict reload/eval/no-grad/repeat/finite/dropout-zero/state law;
- launch/final closure equality and historical closure literal
  `3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`;
- direct exact V2 full-failure predecessor with failure body SHA
  `c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0`.

After the V4 shared device seam changes current source bytes, this validator
must continue to trust the descriptor-validated historical P0 terminal
closure. It must not demand that the current checkout rebuild to the old V3
closure.

### 6.3 P1/P2: strict V4 admission producers

Canonical roots:

```text
P1 tfpd_exploration/results/paired_anchored_calibration_dropout_full_v4_admission/p1_m4_seed42
P2 tfpd_exploration/results/paired_anchored_calibration_dropout_full_v4_admission/p2_m10_seed42
```

Required schema/status:

```text
pacd_p1_p2_admission_multigpu_v4_terminal
PACD_FULL_TRAINING_COMPLETE
```

Fixed arm mapping:

| Arm | short-M | Physical profile |
|---|---:|---|
| P1 | 4 | GPU0, UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`, PCI `00000000:01:00.0`, CVD `"0"`, logical `cuda:0` |
| P2 | 10 | GPU1, UUID `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`, PCI `00000000:03:00.0`, CVD `"1"`, logical `cuda:0` |

Each V4 validator must use the same held-FD/mode/sidecar/topology discipline as
P0 and validate:

- exact V4 attempt/launch/source authority/48 epochs/CP44--47/SWA/manifest/
  terminal descriptor graph and no failure;
- exact arm/root/short-M/device mapping;
- fixed budget, source roster, sampler, optimizer and update topology;
- source-only/no-target facts;
- RNG replay, equal paired unit mask, prefix immutability, finite model/Adam,
  typed accepted-zero encoder aggregate, positive encoder/decoder coverage;
- final-four SWA proof and launch/final V4 closure equality;
- exact extended device profile and the staged idle→current-PID→final evidence;
- `attempt["predecessor"]` equal to the exact composite schema
  `pacd_p1_p2_admission_multigpu_v4_predecessor_v1`;
- nested P0 witness schema `pacd_v3_p0_admission_witness_v1`;
- P0 witness root, attempt, launch, source authority, 48 epoch SHA descriptors,
  CP44--47, SWA, manifest, terminal and historical V3 closure equal to the
  separately validated P0 producer;
- nested/fresh V2 failure evidence equal to the separately validated lower
  predecessor.

P1/P2 must not be required to have a direct V2-failure predecessor. Their
legal chain is V4→held V3 P0→V2 failure.

P1/P2 must not be required to have P0 full/full prediction or identity
equality. Those counters apply to P0 only. P1/P2 retain the applicable RNG,
paired-mask, prefix, finiteness, gradient coverage, sampler and SWA laws.

The exact V4 field placement is frozen in Section 6.1 of the V4 training work
order. V2 must implement that schema literally and reject aliases.

## 7. Deferred live literals

The no-data implementation may use typed synthetic 64-hex fixtures only in
tests. Live binding remains `None` or otherwise nonconstructible until root
independently supplies every exact body literal.

At minimum, live mode requires for P0, P1 and P2:

- attempt SHA;
- launch SHA;
- source-authority SHA;
- 48 epoch receipt SHAs;
- CP44--47 body SHAs;
- SWA body SHA;
- manifest SHA;
- terminal SHA;
- producer closure SHA;
- exact predecessor/witness identity.

Values are obtained only after natural immutable terminals. They are not read
from mutable logs, guessed from prefixes, inferred from a work order, or
selected according to performance.

## 8. Source closure

V2 uses an explicit no-glob closure containing:

- all V2 package files except tests;
- the V2 CLI and this work order;
- the actual shared V1 lifecycle, score and binding helpers it imports;
- every direct CAL-AUG/P4/model/metric/receipt dependency used at runtime;
- exact historical comparator codec dependencies.

Tests and all result roots are excluded. Producer result files are descriptor
inputs, not source closure members. V2 should freeze V4 identity strings in its
own plan rather than importing V4 implementation merely to learn a schema.

Closure is reconstructed before authority publication, after immutable
attempt, before target materialization, before atomic complete-bundle rename,
and at terminal/failure revalidation. Drift fails closed.

## 9. Lifecycle and atomic publication

V2 reuses the hardened V1 lifecycle:

```text
target-free producer/comparator/closure/CPU preflight
→ private authority stage
→ atomic rename to authorized/
→ opaque one-shot capability
→ immutable attempt before target materialization
→ CPU materialize/score
→ private .stage-* complete bundle
→ descriptor validation
→ atomic rename to complete/
```

Successful root topology remains:

```text
attempt.json
attempt.json.sha256
complete/
  input_authority.json
  input_authority.json.sha256
  score.json
  score.json.sha256
  terminal.json
  terminal.json.sha256
```

Pre-commit failure topology remains exact attempt+failure. A failure after the
atomic `complete/` rename never creates a contradictory failure receipt and
never removes the committed bundle.

Every target-facing boundary revalidates the mixed producer witness,
historical comparators, current V2 closure, CPU profile, authority/root inode
identity and capability binding.

## 10. Required tests

The no-data/no-CUDA test suite must prove:

1. a complete synthetic mixed graph P0=V3 and P1/P2=V4 validates;
2. P0=V4 or P1/P2=V3 rejects;
3. P1/P2 arm/root/short-M/GPU/UUID/PCI/CVD/logical-device swaps reject;
4. P1/P2 direct V2 lineage, absent P0 witness or wrong transitive V2 witness
   rejects;
5. any P0-witness descriptor/SHA/closure/epoch/checkpoint/SWA/manifest drift
   rejects;
6. historical V3 closure validates from P0 receipts after current source moves
   to V4;
7. V4 extra/missing/symlink/mode/sidecar/body/epoch/checkpoint/SWA/device/
   terminal-link drift rejects;
8. P1/P2 do not require P0 equality, while all applicable paired/RNG/prefix/
   finite/typed-zero/SWA/source-only laws remain strict;
9. current real T0/C1/SD descriptor integration regression passes without
   tensor deserialization or target data access;
10. V1 default profile retains V1 schema/roots/closure/all-V3 validator;
11. V1 and V2 capability/profile/root/binding substitutions and reuse reject;
12. V2 CPU profile drift and any CUDA initialization reject;
13. exact 126 InputRecords, 756 rows, canonical order, same-input parity,
    selected-M30 activity isolation, summaries, bootstrap, contrasts and gates
    reuse V1 behavior;
14. success, each pre-commit injection, terminal-sidecar corruption,
    post-commit observer failure and root replacement preserve atomic topology;
15. `python -S --dry-run` imports no Torch and probes no producer/result/data/
    checkpoint/GPU path;
16. V2 explicit closure covers all runtime imports, contains no test/result
    path, and reconstructs twice identically;
17. `py_compile`, scoped `git diff --check`, V1 scorer regression and V2 focused
    suite pass in the exact no-user-site/no-CUDA environment.

## 11. Execution gate

No public CLI live execution is authorized. After all three producer terminals
exist, root must:

1. independently audit producer topologies and exact literals;
2. bind those literals into the immutable V2 live producer binding;
3. rerun V1+V2 regression, closure and dry checks;
4. verify empty CVD, uninitialized CUDA and fresh distinct authority/result
   roots;
5. mint the target-free authority and opaque capability in process;
6. execute the single CPU matched score once;
7. audit the atomic complete bundle and every 756-row-derived result.

There is no retry or overwrite. Failure requires a new successor identity and
fresh root.

## 12. Success condition

The design is successful only if:

- P0 V3 and P1/P2 V4 are validated under their exact distinct lineages;
- scoring science and gates remain exactly V1;
- the CPU-only atomic matched score completes on all 756 rows;
- every effect is computed relative to the matched P0 and the historical C1,
  T0 and sealed Cell-D comparators; and
- no producer, target, GPU or result-root fact is inferred from an unreviewed
  or mutable source.

Until the live literals and score terminal exist, this is a launch-safe scorer
design, not a PACD performance result.
