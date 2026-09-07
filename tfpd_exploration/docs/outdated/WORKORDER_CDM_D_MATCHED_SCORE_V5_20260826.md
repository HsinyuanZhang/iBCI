# Work Order: CDM-D V5-Bound Matched Performance Score

Date: 2026-08-26  
Status: implementation and CPU/no-data test authority only  
Scientific role: first complete target-performance test of the accepted independent-activity CDM-D system

## 1. Decision and objective

Build one reviewed matched scorer for the CDM-D system whose source
constructibility was accepted in the V5 strict-27 gate.

The experiment compares exactly two systems:

1. the sealed Cell-D SWA under the honest calibration budget; and
2. the same sealed Cell-D SWA plus the complete causal dual-memory wrapper.

The model graph, weights, normalizers, output head, target windows, targets,
metric, and session weighting must be identical. The wrapper owns no trainable
parameter. It changes only the target-time state available to the same model:

- an unlabeled B3S activity FIFO; and
- a closed-form, trust-gated carrier sufficient-statistics state.

Run the complete fixed matrix in this order:

```text
M30 -> M10 -> M4
```

For every budget, run both systems on within-6 and cross-subject external-15.
This is 12 cells: 3 budgets x 2 surfaces x 2 systems.

Performance at an earlier budget must not decide whether a later budget is
opened. M30, M10, and M4 use different activity-memory capacities, and the
purpose of this experiment is to measure the short-label route completely.
Only an integrity, provenance, non-finite, resource, or execution-boundary
failure may stop the matrix. A negative R2 delta is a valid scientific result,
not an execution failure.

Do not add activity-only, carrier-only, oracle, posterior, wrong-pair,
zero-carrier, architecture, or threshold-tuning cells. Attribution remains
blocked until this complete system has a credible positive result.

This work order authorizes code changes and CPU/no-data/no-CUDA tests only. It
does not authorize target data access, checkpoint tensor loading, CUDA/GPU,
authority/result publication, scoring, or launch.

## 2. Exact accepted predecessor

Before durable score authorization or output reservation, descriptor-reload
the exact accepted V5 source gate under one held `O_NOFOLLOW` directory FD:

```text
root:
  tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v5

implementation closure:
  c51490115ee15509376174855da045e769547a0f0ce9cf1267cd31c7d93e2d0f

attempt.json:
  40edd3fb7beeb7a57c125ae3d7b631c5279cac2af732553adb102ee3f899c3d7

launch.json:
  a3b66d4603d910fb6a93a240256569026a5b809a27365c031345b8bb7f3e1ad8

source_authority.json:
  d962f7d6035b6db54048f39ad3d5fa55c37d2c829d53a87a8fe55b04923bc3c5

budget_m30_aggregate.json:
  862f83a82d68bc87a0c46dfd05ef4bc68211092cf5fef2c006c860885e8d1a0c

budget_m10_aggregate.json:
  78689fb91ab6945dff159fbbd3538cabbb0af4e5b0171774500fc8de1b7c2ed2

budget_m4_aggregate.json:
  e046185729675c54f94ae5295daee891da6b286c1e0f3f69b1af98ffa4764b62

terminal.json:
  e2e07244e1021137ba806f8f0156c77474b18608e747ef413e38b5c51a1976ff

terminal status:
  PASS_SOURCE_CONSTRUCTIBLE

exact topology:
  88 JSON bodies + 88 canonical sidecars = 176 leaves
```

Every leaf must be a regular non-symlink mode-0444 file. Every sidecar must be
the canonical basename sidecar. There must be no failure body and no extra
leaf. Bind the fixed terminal body first, then reload every one of its 84
evidence bodies by the exact terminal-provided digest. Recheck the named
directory identity after all reads.

The semantic graph must also prove:

- source-only execution;
- zero target/backward/optimizer/parameter updates;
- strict source roster cardinality 27;
- exact passing/total breadth M30 27/27, M10 27/27, M4 26/27;
- the immutable aggregate field `breadth_min_passing_sessions` is 14 for
  every budget; the denominator 27 above is the retained source-session
  count, not the breadth threshold;
- the V5 identity and closure above;
- the V5 independent-activity evidence contract; and
- the sole M4 source failure remains present rather than being removed.

The source gate is a constructibility prerequisite, not target-performance
evidence.

## 3. Narrow compatibility seam

The existing `causal_dual_memory_cell_d_score_v1` package is a no-result,
NO-GO draft. Its parser and lifecycle are useful, but it hard-codes the
historical V2 gate and coupled memory transition.

Authorize a narrow backward-compatible refactor in only these existing files:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/plan.py
tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/score.py
tfpd_exploration/src/causal_dual_memory_cell_d_score_v1/physical.py
```

The refactor may add only typed, immutable route-profile and transition/runtime
hooks. The existing public V1 route must keep its current V2 predecessor,
identity, roots, receipt schemas, physical transition, and default behavior.
Existing V1 tests must prove this regression property.

The refactor must not:

- replace a module global at runtime;
- mutate `sys.modules` to substitute a route;
- accept a caller dictionary or environment variable as a profile;
- copy the lifecycle or physical evaluator into the successor;
- weaken descriptor, closure, authority, capability, publication, or freshness
  checks; or
- change any shared model, data module, source gate, checkpoint, result, or
  unrelated route.

Recommended immutable contracts are:

```text
CompletedSourceGateContract
CDMDScoreRouteProfile
TransitionPolicy
SessionEvidenceCodec
```

The exact class names are not authoritative. Their typed responsibilities are.
V1 must expose a literal V1 default profile. V5 must expose a separate literal
profile. No public caller may construct an arbitrary profile.

The hooks must cover only:

- completed source-gate root, fixed hashes, schema, topology, and validator;
- authority and score roots;
- implementation closure reconstruction;
- closure-bound runtime/executor factory;
- initial memory construction;
- completed-trial commit operation; and
- route-specific transition evidence encoding and validation.

## 4. Additive V5 scorer ownership

Create only these additive successor paths, plus the work order itself:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/__init__.py
tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/plan.py
tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/score.py
tfpd_exploration/src/causal_dual_memory_cell_d_score_v5/physical.py
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v5.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v5.py
```

Use fresh route-specific roots:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v5
tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v5
```

The V5 package must be a thin typed composition over the reviewed V1 parser,
held-data, sealed-model, metric, transaction, and lifecycle primitives. It may
not contain a copied scorer or copied lifecycle.

## 5. Frozen target science contract

### 5.1 Common target surface

Use only the existing fixed authorities for:

- within-6; and
- cross-subject zero-shot external-15.

Do not resolve source-training, formal, H1, M1, or any other target surface.
Bind each asset path, UUID, descriptor identity, byte count, and SHA before
model execution.

For every session, define one common chronological query pool as all rewarded
trials at positions `>= 30`. M30, M10, and M4 must use that same ordered query
pool. The first 30 trials may provide support only; a trial in the first 30
that is not selected for M4 or M10 must not become a query trial.

Both systems must consume bitwise-identical neural windows, valid-bin masks,
last-bin targets, chronological trial partition, and model state.

### 5.2 Honest support and initial state

For each session and budget, start from a fresh state:

- M4: exact sealed D-optimal four support trials;
- M10: exact first ten chronological support trials;
- M30: exact first thirty chronological support trials.

Fit the initial point carrier only from that budget's support, using
`FIXED_RIDGE_BY_TRIAL`, normalized lambda 0.1, and the exact-duration support
rate rule. Raw M30 T4 may provide channel order and valid-mask authority only.
It may not initialize M4/M10, choose their groups, normalize their carrier, or
enter their state. Each budget derives its K=4 group map from its own honest
initial carrier over the shared valid rows.

The activity FIFO capacities are literal:

```text
M30: 0
M10: 20
M4:  26
```

### 5.3 Independent completed-trial state transition

The V5 target route must construct
`IndependentActivityCausalDualMemory`, not `CausalDualMemory`.

For each completed query trial `j`:

1. score all windows in trial `j` from state formed strictly before `j`;
2. retain only governing last-bin predictions and targets;
3. after every prediction for `j` is complete, construct the four held-group
   trajectories;
4. call `observe_completed_trial()` once;
5. call `commit_independent()` once; and
6. expose the new state only to trial `j+1`.

A valid completed B3S trial must advance the activity transition independently
of the carrier gate. When the carrier update is rejected, the activity state
still commits, the carrier digest remains exact, and the completed-query count
advances once. Invalid B3S evidence commits neither state.

At M30, activity FIFO capacity zero makes the activity stack/digest an exact
no-op even though the completed-query count advances. Carrier updates remain
subject to the same trust gates and may commit or reject. Do not import the
source gate's offline M30 audit no-write policy into target scoring.

No target label may construct a pseudo direction, gate an update, choose a
group, change activity memory, alter carrier state, fit a normalizer, select a
checkpoint, or update a parameter.

### 5.4 Model and metric

Both systems use the exact sealed Cell-D SWA, normalizers, and last-bin head.
Run `eval()` and `no_grad()` with zero dynamic-dropout calls. Model state before
and after every cell must match exactly.

For each session, compute the governing variance-weighted two-output R2 over
all valid last-bin query windows. Then report:

- every per-session R2;
- equal-session mean and median;
- paired CDM-D minus sealed Cell-D delta;
- positive-session count;
- deterministic paired-session bootstrap 95% interval, 10,000 draws, seed 42;
- complete transition and resource evidence.

No target-session gradient, backward, optimizer step, model update, carrier
normalizer refit, or target-based selection is allowed.

## 6. Performance reporting

Always publish all three budget results when execution integrity remains valid.
Report these predeclared gates independently; do not let one gate hide another:

```text
M30 external safety:
  mean delta >= -0.01

M10 external performance:
  mean delta >= +0.02
  positive sessions >= 10 / 15

M4 external performance:
  mean delta >= +0.05
  positive sessions >= 10 / 15
```

The terminal must report the three gate booleans and a descriptive decision:

- `ADVANCE_SHORT_BUDGET`: M10 or M4 passes its full gate;
- `HOLD_WEAK_SIGNAL`: neither full short-budget gate passes, but at least one
  short-budget mean delta is positive;
- `STOP_NO_PERFORMANCE_SIGNAL`: both M10 and M4 mean deltas are non-positive;
- `FAIL_CLOSED`: integrity or execution failed before a complete matrix.

M30 safety is reported separately and cannot erase a real M4/M10 measurement.
Within results and diagnostics cannot rescue an external gate.

## 7. Required durable evidence

For each `(surface, session, budget, system)` bind at least:

- exact input-authority, support, query, window, target, and validity digests;
- exact sealed SWA, strict load, model state before/after, and prediction digest;
- honest initial carrier/rate domain/parity/group/valid-mask/normalizer evidence;
- activity FIFO capacity and initial/final activity/carrier/state digests;
- activity transition committed count;
- activity FIFO changed count;
- carrier transition committed count;
- activity and carrier rejection reason counts;
- a canonical per-query transition digest chain or equivalently exact
  reconstructible transition records;
- M30 capacity-zero activity invariance;
- group-forward and full-system forward counts;
- zero dropout, non-finite, target update, backward, optimizer, and parameter
  update counts;
- current/peak CUDA allocated/reserved bytes, RSS, wall time, windows/s, and
  trials/s;
- V5 predecessor binding and current implementation closure.

The V5 evidence schema must keep activity and carrier outcomes as separate
typed facts. It may not compress both into the historical V1
`accepted_updates` field.

## 8. Lifecycle and launch boundary

The required lifecycle is:

```text
static dry plan
-> root-reviewed target-free preflight
-> immutable durable authorization
-> attempt publication
-> target path resolution / checkpoint load / CUDA
-> input authority
-> complete fixed 12-cell score
-> atomic score + terminal
or immutable failure
```

Attempt must exist before target path resolution, held target directory access,
checkpoint tensor loading, CUDA initialization, parser construction, or model
forward. The V5 source graph, fixed target metadata, closure, durable authority,
selected device, and fresh score root must be revalidated before reservation,
before execution, after the final target forward, and before terminal
publication.

Use held descriptors, `O_NOFOLLOW`, `O_EXCL`, fsync, canonical sidecars, and
immutable 0444 publication. The public CLI must remain static/no-Torch/no-data/
no-write with zero arguments and fail closed without exact flags plus an opaque
in-process root capability.

GPU selection remains dynamic. A later root authority may choose one complete
accepted device profile only after checking active Z1/Z4/Z6/P4 and unrelated
GPU jobs. The scorer must not signal, pause, reconfigure, or inspect data from
those jobs.

## 9. Required no-data tests

At minimum, prove:

1. exact held V5 88-body/176-leaf graph validation and rejection of any body,
   sidecar, mode, topology, terminal map, schema, closure, roster, breadth, or
   no-target-boundary drift;
2. unchanged V1 default profile, V2 predecessor, public payloads, and coupled
   transition behavior;
3. V5 profile cannot be caller-forged or supplied by environment/mapping;
4. V5 runtime/executor identities are exact and closure-bound without
   monkeypatch or `sys.modules` substitution;
5. M4/M10 carrier rejection still commits valid activity and preserves carrier;
6. M30 capacity-zero activity stays exact while the independent completed
   query count advances;
7. accepted carrier transition advances both typed state branches;
8. invalid B3S evidence advances neither branch;
9. no state from one budget/session enters another;
10. every budget uses the exact common post-first30 query pool;
11. same-trial predictions are unchanged by their later commit;
12. sealed and CDM-D use identical input/target/mask/model authorities;
13. V5 receipt codec rejects collapsed activity/carrier booleans, count drift,
    broken digest chains, M30 mutation, and forged rejection reasons;
14. attempt precedes any target/model/CUDA action;
15. target gradients/backward/optimizer/updates and eval dropout remain zero;
16. publication rollback, stale authority, stale closure, wrong environment,
    root collision, symlink, inode replacement, and post-forward predecessor
    drift fail closed.

Run existing V1 tests as regressions in the same isolated command. Use no user
site, no CUDA, no bytecode, disabled plugin autoload, and single-thread CPU
libraries.

## 10. Worker handoff

Stop at a frozen no-data/no-CUDA boundary. Return:

- exact SHA-256 for every owned/changed file;
- an explicit no-glob implementation closure;
- exact focused and V1-regression test commands/counts;
- py_compile, static dry CLI, whitespace, and fresh-root checks;
- a list of any remaining parser, checkpoint, target, CUDA, device, authority,
  or launch blockers.

Do not mint authority, reserve result roots, open NWB/checkpoint tensors,
initialize CUDA/GPU, or launch scoring until root performs an independent audit
and issues a separate explicit authorization.
