# Work Order: CDM-D Matched Performance Score V1

Date: 2026-08-25  
Status: implementation specification; no launch authority  
Scientific role: first complete performance test of Causal Dual-Memory Cell D

## 1. Objective

Build and test one additive matched evaluation route for the complete CDM-D
system. Compare only:

1. the sealed Cell-D SWA with the honest fixed calibration budget; and
2. the same sealed Cell-D SWA with the complete causal activity-memory plus
   cross-fitted carrier-memory system.

Evaluate M30, M10, and M4 in that fail-fast order on the fixed within-6 and
cross-subject external-15 surfaces. This is the performance experiment, not an
attribution study. Do not add activity-only, carrier-only, zero-carrier,
wrong-pair, oracle, posterior, or architecture controls before this system
result.

This work order authorizes additive implementation and CPU/no-data tests only.
It does not itself authorize opening evaluation NWBs or checkpoint tensors,
initializing CUDA, creating a canonical authority/result root, or launching a
score.

## 2. Accepted predecessor

The scorer must descriptor-reload and validate the exact immutable completed
CDM-D Source Execution V2 gate before reserving any score artifact root:

```text
root:
  tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v2

implementation closure:
  46d73bc54df26f03ad349f0f98547263718e372aa5ca4817bfcf5259fc87103a

attempt.json:
  7861eef4e146e2ba92a700aca405d2527f9a8ae7df514a119b76bb875c138e33
launch.json:
  fd1be4592d078091fcb1b6d547facb76cccb16f0938cfba567373a33164ba6f2
source_authority.json:
  54bcadd924ae6660bb26dd8c2e1ab51ec85e231b91de5ba44b7c0a9160e073de
budget_m30_aggregate.json:
  bc046653f1d394b40c650ad3feed9db127ad074e191216fd0566a1a66a33cefa
budget_m10_aggregate.json:
  b637f5f3778e6a0d51e01eabd38b88aae736aa61c99882cb86adca6115e5aee4
budget_m4_aggregate.json:
  5b6e93c5e711223720240c06b0902cd151f00f94a932adf8ec0c720a14a872d0
terminal.json:
  a909d5d5210656b17c73272eecefa540182b384db6ea9f5fc70ae2b59d30bdf3
```

The predecessor has exactly 88 JSON bodies and 88 canonical sidecars, all
regular non-symlink mode 0444. Its terminal status is
`PASS_SOURCE_CONSTRUCTIBLE`; it binds all 84 budget evidence bodies. The
scorer must validate that exact graph under one held `O_NOFOLLOW` directory FD,
including every terminal-referenced evidence hash, before accepting a durable
score authorization. Prefixes, copied JSON payloads, or a caller-supplied
summary are insufficient.

The source-gate result is an implementation prerequisite, not a target score:

```text
M30: 27 / 27 source sessions pass
M10: 27 / 27 source sessions pass
M4:  26 / 27 source sessions pass
```

Do not exclude the one M4 source failure from any later target denominator.

## 3. Frozen science contract

### 3.1 Systems and model

Both systems use the same sealed Cell-D SWA, exact parameter state, normalizer,
last-bin head, and evaluation inputs. The CDM-D wrapper owns no trainable
parameter and may not mutate the model state. Run in `eval()` and `no_grad()`;
dynamic dropout must make zero calls.

The sealed comparator is not a historical aggregate score. It must be scored
again on the exact same input records used by CDM-D in this lifecycle.

### 3.2 Budgets and initial state

For each session and each budget, reset both systems from the same honest
support:

- M4: exact sealed D-optimal four support trials;
- M10: exact first ten chronological support trials;
- M30: exact first thirty chronological support trials.

Build the initial point carrier only from that budget's support using the
accepted `FIXED_RIDGE_BY_TRIAL`, lambda 0.1, exact-duration support-rate rule.
Raw M30 T4 values may provide sealed channel/valid-mask authority but may not
initialize M4 or M10, select their groups, normalize their carrier, or enter
their state. Group assignment is budget-specific and derived from the honest
initial carrier.

The B3S activity-memory capacity is literal:

```text
M4:  26 completed query trials
M10: 20 completed query trials
M30:  0 completed query trials
```

M30 is therefore a carrier-update safety test with an unchanged 30-row B3S
stack, not an accidental `[-0:]` activity-memory path.

### 3.3 Causal trial order

Score trials in the sealed chronological query order. For query trial `j`:

1. freeze the state formed from support and accepted completed trials `< j`;
2. score every valid 50-bin window in trial `j` with both systems;
3. retain only the last-bin prediction and target for the governing metric;
4. after every prediction in trial `j` is complete, construct four
   complementary-group trajectories from the CDM-D system;
5. apply the unchanged source-gated trust, canonical snapping, conditioning,
   departure, and freeze rules;
6. update only accepted held-group sufficient statistics;
7. append the completed trial's unlabeled neural activity to the bounded B3S
   FIFO where the budget capacity is positive;
8. use the new state only for trial `j+1`.

The target behavior label is metric-only. It may not construct a pseudo
direction, gate an update, choose a group, alter the activity FIFO, refit a
normalizer, select a checkpoint, or update any parameter. No bin or activity
from trial `j` may affect any prediction in trial `j`.

### 3.4 Fixed surfaces and metric

Use the existing fixed evaluation authorities for:

- within-6; and
- cross-subject zero-shot external-15.

Do not resolve any other within, external, formal, H1, M1, or source-training
surface. Bind every session UUID/path/descriptor identity and the exact support
and query trial IDs before model execution. The two systems must consume
bitwise-identical raw neural windows, calibration support, valid-bin mask,
last-bin target, and chronological trial partition for every session/budget.

For each session, compute the existing last-bin variance-weighted two-output
R2 over all valid query windows. Then report the equal-session mean and median.
Also report the paired CDM-D-minus-Cell-D delta, positive-session count, all
per-session values, and a deterministic paired session bootstrap 95% interval
with 10,000 draws and seed 42. A failed/no-update session remains in the
denominator as its realized CDM-D score.

### 3.5 Fail-fast performance decision

Run and publish complete session rows for one budget before advancing:

1. M30;
2. M10 only if the M30 external gate passes;
3. M4 only if the M10 stage was reached.

The predeclared system gates are:

```text
M30 external mean delta >= -0.01
M10 external mean delta >= +0.02 and positive sessions >= 10 / 15
M4  external mean delta >= +0.05 and positive sessions >= 10 / 15
```

M30 failure yields `STOP_M30_SAFETY` and M10/M4 are not opened. M10 failure
does not erase its score and does not tune any threshold; M4 may still run only
if the frozen policy explicitly records M10 as reached rather than using M10
performance to modify CDM-D. The terminal must distinguish `PASS`, `HOLD`, and
`STOP` without allowing within or diagnostics to rescue a failed external
gate.

## 4. Required evidence

For every session/system/budget receipt, bind at least:

- input-authority SHA and exact chronological trial/window digests;
- support IDs, query IDs, valid last-bin count, target digest, and prediction
  digest;
- model SWA SHA, strict-load proof, state-before/state-after digest equality;
- budget-specific initial carrier, group assignment, valid mask, normalizer,
  B3S support, and FIFO capacity;
- accepted/rejected update counts and exact rejection-reason counts;
- group-forward count, full-system forward count, dropout count, and any
  non-finite count;
- proof that target labels, target gradients, optimizer steps, and parameter
  updates are all zero;
- current/peak CUDA allocated and reserved bytes, RSS, wall time, windows/s,
  and trials/s;
- source-gate predecessor SHA and current implementation closure.

Record the pseudo-direction disagreement and trust-gate distributions as
descriptive diagnostics. They cannot change thresholds or rescue the result.

Before the first live target forward, run a source-free/synthetic parity gate
and one authorized source-only physical smoke proving:

- sealed Cell-D output is identical through the new route;
- zero accepted CDM-D updates reduce to sealed Cell D exactly;
- a trial's post-update state cannot change that same trial's predictions;
- model state is unchanged;
- M30 activity memory is a strict no-op;
- M4/M10 never consume M30-only labels or future rows.

## 5. Lifecycle and ownership

Create only additive files under a new package, CLI, test, and this work order.
Suggested ownership:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v1.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v1.py
```

Do not edit Cell D, CDM-D Stage 0/source-audit/source-execution files, shared
datamodules, existing scorers, checkpoints, source-gate results, or unrelated
routes. Other agents and user processes share the worktree; do not revert or
rewrite their changes.

Use separate fresh canonical roots for target-free preflight authority and
matched score. The lifecycle is:

```text
dry plan
-> root-reviewed target-free preflight
-> immutable durable authorization
-> attempt before evaluation asset/model/data access
-> input authority
-> budget/session evidence
-> atomic score + terminal
or immutable failure
```

Every publish/reserve/mint/execute helper must independently require the exact
launch environment and durable authorization; no direct helper may bypass the
gate. Hold parent/root descriptors through publication, use `O_NOFOLLOW`,
`O_EXCL`, fsync, canonical sidecars, and immutable mode 0444 pairs. Rebuild the
implementation closure before authorization, before execution, and before
terminal publication.

The public CLI must be static/no-Torch/no-data/no-write with zero or one flag,
and must fail closed without both explicit execution flags plus an opaque
in-process root capability. A code-complete candidate remains NO-GO until root
independently audits it and separately authorizes smoke or score.

## 6. Worker handoff

Return:

- exact owned-file SHA-256 values and an explicit no-glob implementation
  closure;
- focused CPU/no-CUDA tests, py_compile, whitespace, and static dry-CLI proof;
- adversarial tests for future-trial leakage, target-label state leakage,
  system/input mismatch, stale source gate, wrong budget group map, M30 FIFO
  mutation, model-state mutation, direct reserve bypass, partial publication,
  and threshold/gate forgery;
- a clear list of any remaining physical parser, source-smoke, checkpoint,
  CUDA, device, authority, or launch blockers.

Stop at a frozen no-data/no-CUDA boundary for independent root audit.
