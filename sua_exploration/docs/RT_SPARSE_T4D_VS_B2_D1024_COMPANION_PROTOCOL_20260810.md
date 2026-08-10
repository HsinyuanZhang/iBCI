# RT T4d versus matched SPINT-structured B2-D1024 companion

Status: **pre-registered companion protocol; no RT Stage-2 arm, launcher, or
primary gate is changed by this document.**  The frozen Stage-2 three-arm
matrix remains the authoritative content experiment:

```text
15 outer LOSO folds x {R-T4d, R-Full/afc4_vel, R-Zero4} x seed 42.
```

Its primary endpoint is `R-T4d - R-Zero4`.  This document introduces a
separate, read-only comparison to the pre-existing Stage-R `zero4` B2-D1024
system.  B2-D1024 is called **matched SPINT-structured B2-D1024**, never
"released-code original SPINT": it is a separately source-trained B2
implementation whose identity width was selected to be SPINT-scale.

## Question and scope

After the Stage-2 matrix has closed, does the sparse endpoint-direction T4d
system exceed the existing SPINT-structured, no-side-carrier B2-D1024 system
on the same RT outer-session protocol?  This is a system-level companion, not
a pure carrier ablation: architecture and source training differ.  It neither
replaces nor relaxes the T4d--Zero4 content gate.

The companion is executable only after all three conditions hold:

1. the Stage-2 matrix is exactly 45/45 closed;
2. `rt_sparse_endpoint_stage2_terminal_verify.py` returns PASS for that exact
   bundle; and
3. every one of the 15 B2-D1024 scores has a **fresh, uniform forward-only
   re-evaluation receipt** bound to the Stage-2 query schedule. Archived-source
   reconstruction is diagnostic evidence only and never substitutes for a new
   score.

There is no B2 training or retraining phase in this protocol. A CPU replay of
the **current** evaluator from a bound old split/config/NWB establishes only
implementation compatibility; it is not retroactive proof of the historical
ordered query set. A matching *count* is not evidence of a matching query set.
Consequently, no historical B2 score is accepted by this companion. Only the
15 newly generated, execution-bound forward-only outer receipts enter the
paired statistics; a missing receipt makes the companion fail closed.

### Legacy audit finding (2026-08-10; not a score comparison)

The initial apparent fold-1 mismatch was resolved by replaying the frozen
`SessionBatchSampler`, rather than comparing scored rows to all eligible rows.
Both historical B2 and Stage-2 T4d report 19,296 scored windows, while the
all-eligible query audit has 19,310. The 14 omitted rows are exactly the
incomplete tail of a 32-window batch. The same mechanism explains Stage-2
fold-0 `24632 -> 24608` (remainder 24) and fold-2 `29895 -> 29888`
(remainder 7). Strong query identity is therefore defined over the actual,
ordered complete batches and not merely their count or the all-eligible audit.

The original remote B2 fold-0 tree contains the selected checkpoint, teacher
metadata, and old sampler/evaluator bytes. A CPU-only remote reconstruction
receipt has been sealed at
`results/rt_sparse_t4d_b2_companion_v1/RT_B2_FOLD0_REMOTE_PROVENANCE_RECONSTRUCTION_v1.json`
(SHA-256 `c87986b295113caee8ae43ba70d52d98eb0260ccea6e8a4fa0b4c7c3d462e84b`).
It proves equality of all three actual complete-batch query digests between
the preserved old B2 source and Stage-2 T4d for fold 0. Its status deliberately
states the remaining limitation: the imported historical receipt lacked a
contemporaneous full source-tree manifest. It is a **diagnostic only**, not a
substitute score and not a special acceptance route for fold 0.

The score comparison therefore uses one uniform policy: after Stage-2 45/45
terminal PASS, restore all 15 sealed B2-D1024 checkpoints and run exactly one
new forward-only outer evaluation per fold on the Stage-2-bound evaluator and
query schedule. Fold 0 runs in its preserved remote absolute-path layout; its
checkpoint, config, selection receipt, split manifest, and teacher metadata
are also SHA-verified into a new local evidence root for preservation. Folds
1--14 run from their local sealed artifacts. No historical B2 outer R2 enters
the final paired comparison.

`rt_sparse_t4d_b2_forward_reeval_terminal.py` is the only permitted plan and
finalizer. It requires the terminal Stage-2 verifier before it emits any
commands, restores a sealed B2 checkpoint, starts no optimizer or Trainer,
and demands unchanged model-state hashes plus exact actual three-digest
equality after every pass. It does not retrain B2 and refuses every existing
output path. Every command pins its working directory to the evaluator tree
because the legacy resolved configs contain `paths.root_dir: .`; the plan
records this cwd and the device environment policy. CPU inference is valid but
expected to be slow for the 5.274B-MAC B2-D1024 session path, so a single-GPU
**forward-only** run is the minimal practical execution once the terminal
Stage-2 bundle exists.

## Frozen pairing and compatibility requirements

For every outer fold, the audit verifies both B2 and Stage-2 evidence for:

- outer target session and ordered inner-train and inner-validation lists;
- seed 42; chronological `M=24`; `q=24`; 50-bin window; max trial length 100;
  session-window budget 4096; and 35 source epochs;
- optimizer type and learning rate, checkpoint monitor and selected-by rule;
- joint source decoder training, no target-session backpropagation, and
  unchanged target-evaluation model state;
- exact teacher checkpoint SHA and exact target NWB SHA/data identity; and
- the three ordered query digests: window starts, target/eval-mask rows, and
  their combined identity.

Intentional differences are recorded rather than rejected: Stage-2 uses B3S
with a four-wide side interface, while the comparator is B2-D1024 / `zero4`.
No claim of released-code equality follows from this matching.

The historical Stage-R B2 artifacts are distributed across imported fold 0,
local folds 1--2, the folds 3--14 supervisor, and a race-recovered fold 10.
The original raced fold-10 cell is forbidden; only the pre-existing recovery
receipt is eligible.  A legacy teacher path without an immutable teacher-SHA
witness is also insufficient: it is an audit failure, not an assumed match.

## Analysis and score-opening discipline

The companion reads the already-written raw R2 values only after the terminal
conditions above.  It reports `T4d - B2-D1024` full 15-fold descriptive
statistics (mean, median, positive/zero/negative sign counts, full ordered
deltas, and leave-largest-absolute-out mean).

Fold 0--2 T4d scores had been opened before this companion protocol.  Fold 3
had a closure before the protocol but has no independently verifiable,
pre-existing unopened-score attestation.  Therefore the prospective
confirmation subset is frozen as **folds 4--14 (n=11)**, unless the auditor
finds a dated, immutable attestation predating this protocol that proves fold 3
was unopened.  In its absence, folds 0--3 remain in full-15 descriptive output
only.  No verbal assertion of non-reading substitutes for that attestation.

For the prospective subset the frozen companion gate is:

```text
mean(T4d - B2) > 0
median(T4d - B2) > 0
strict positive-sign majority.
```

The same full delta list and LLO statistic are always reported.  This gate is
an independent companion finding, not an amendment to the primary Stage-2
T4d--Zero4 gate.  The report must not use “equivalent”, “no cost”, or
“released-code SPINT” without a separately pre-specified study.
