# Work Order: CDM-D Source Execution V1

Date: 2026-08-25  
Status: authorized for additive implementation and no-data/no-CUDA tests only  
Scientific role: physical source-only preflight for the complete CDM-D performance system

## 1. Objective

Build the descriptor-safe execution route that turns the accepted CDM-D
Stage-0 and Source-Audit primitives into:

1. one fixed engineering source smoke; and
2. one strict-27 source constructibility screen in the frozen order
   `M30 -> M10 -> M4`.

This is not an attribution experiment and not a target score.  It validates
that the complete dual-memory consumer can be constructed from real source
trials before the reviewed within/external performance screen.

Accepted authorities:

```text
Stage-0 workorder SHA256:
5d4a22bf8d1700b4230f2f9970c9ff98b2e6a31d0a9bc1bd828e6d844ff1c6fc

Stage-0 closure SHA256:
3ab6d3de931e4630cb9c80b07e25e3b38af4c444f3c937d3d28560b596c29590

Source-Audit workorder SHA256:
8489fcbf1c83d5c174fb10e440ecd95387b46a8dca4466a3ac355d0024bb0260

Source-Audit closure SHA256:
cfc192c45f674ccd4d56f19e3d5a58dc9146cf2fb02b02d4f561e7e0bb2db4dc
```

This work order does **not** authorize opening source NWB files, loading a
checkpoint tensor, creating a canonical root, initializing CUDA, or launching
the smoke/full gate.  Those require root review of the frozen implementation
and a separate explicit launch decision.

## 2. Additive ownership

Create only:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution.py
```

Do not edit the accepted Stage-0 or Source-Audit files, shared Cell-D/model,
datamodule, scorers, prior routes, data, checkpoints, or results.  The worker
is not alone in the repository; preserve unrelated changes.

## 3. Fixed assets and source boundary

Bind these exact local assets through descriptor-safe reads:

| Asset | Relative path | SHA256 |
|---|---|---|
| strict manifest | `sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` | `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9` |
| sealed Cell-D terminal | `tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json` | `b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442` |
| sealed Cell-D SWA | `tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt` | `626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd` |
| theta receipt | `tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority_receipt.json` | `d023dd632c4717443f1f55e924a09be1747fc58d5c30a8fb6fa38f4b7b117184` |
| theta artifact | `tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority.pt` | `cef39dc8220aa253214963a32e5457dede1045e64b158e37fc267a6fb4146319` |

Also bind the existing sealed source normalizers exactly:

```text
T4 normalizer semantic SHA256:
293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0

behavior normalizer semantic SHA256:
f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391

behavior mean:
[-0.001148765324614942, 0.002653369214385748]

behavior std:
[8.63547420501709, 8.086690902709961]
```

Only the 27 manifest `train` sessions may be resolved or opened.  The `val`
and `test` names are inert manifest strings: do not resolve paths, count units,
open NWB files, or construct loaders for them.  Within, external, formal, and
target flags must remain false in every receipt.  No target backward,
optimizer, parameter update, or normalizer refit is reachable.

## 4. Exact physical source route

Reuse, do not reimplement, the accepted route-owned primitives:

- fixed M4/M10/M30 support and audit identities;
- typed B3S/native-count/velocity-validity views;
- K=4 complementary groups and grouped B8;
- behavior destandardization before displacement integration;
- invalid-theta `group=-1` semantics;
- local state, prefix, RNG, dropout, and channel-order evidence.

Use the reviewed strict train-only source adapter pattern from
`cell_d_equal_session_v1.py`.  Prebind only the exact source roster before
shared datamodule setup and prove that no non-train resolver or unit counter is
called.  Do not cache target arrays or route through an evaluation parser.

For each source session:

1. descriptor-bind rewarded-trial chronology and source channel IDs;
2. reconstruct the raw T4 and theta-validity mask in the same channel order;
3. build fixed support/audit identities before any model forward;
4. run each completed trial causally and emit four held-group predictions;
5. join the true source trial direction only after all four pseudo outcomes are
   finalized;
6. run the grouped carrier-level B8 estimand from native rate rows;
7. retain every typed rejection in the fixed pool; never drop a row.

The initial carrier is **budget-specific**.  For each of M4, M10, and M30,
reconstruct it only from that budget's sealed support rates and sealed support
directions with the existing production-parity
`fit_ridge_t4(..., normalized_lambda=0.1)` rule.  M4 uses its four sealed
D-optimal support rows; M10 and M30 use their chronological first 10 and first
30 rows.  The initial support rates must match the historical comparator:
raw spike counts in the exact half-open `[start_time, stop_time)` interval
divided by that row's exact exposure duration.  They must not be silently
replaced by the later online-update approximation
`mean(native_20ms_binned_counts)/0.020`; receipts disclose the two rate
domains separately.  The pre-existing M30 raw/OLS T4 artifact may establish
channel order and the sealed theta-validity topology, but its values may not
initialize an M4 or M10 carrier and may not replace the fixed-ridge carrier at
M30.

Build the deterministic K=4 complementary groups separately for each budget
from the exact budget-specific initial carrier consumed by Cell D, while using
the sealed theta authority only for the valid/invalid mask.  The held-unit
masks used by every model forward, the online carrier sufficient statistics,
and the grouped-B8 evidence must all bind that same per-budget group map.  A
single M30-derived group map shared across budgets is forbidden.

Receipts must record, for every `(session, budget)`, the support-only initial
carrier digest, fixed-ridge parity evidence, valid-mask digest, and group-map
digest.  Changing any M30-only carrier value or any label outside the sealed
M4/M10 support while holding that support fixed must leave the corresponding
M4/M10 initial carrier, group map, first prediction inputs, and initial memory
digests exactly unchanged.

M30 uses positions 30--59 after its sealed first-30 support.  These rows are
the separate Source-Audit safety pool, not deployment updates: they may produce
four finalized pseudo directions and offline grouped-B8 evidence, but they may
change neither the zero-capacity B3S activity memory nor the M30 carrier state.
M10 uses positions 10--29.  M4 uses the chronological first-30 complement of
the sealed D-opt support; those two budgets exercise the complete causal
dual-memory update.  A session lacking a required chronological row is an
explicit failed session with a typed count, not a substituted trial or an
exception that erases the rest of the roster.

The sealed Cell-D model must be loaded strictly and used in eval/no-grad mode.
All four held-group calls physically remove the held units from neural, B3S,
and T4 tensors.  Repeated forward, model state, lazy topology, Python/NumPy/
Torch CPU/Torch CUDA RNG, and dynamic-dropout counters must satisfy the
accepted physical evidence contract.

For M4 and M10, the causal B3S stack is not yet 30 rows on the first query:
its length is exactly `M + accepted_completed_queries_before_j` and grows only
after a committed trial, up to the frozen cap of 30.  The real shared B3S
pooling path must receive that variable-length stack.  Do not zero-pad, repeat
support rows, or preload later query activity to satisfy a synthetic `[30,100,N]`
assertion.  The accepted 30-row physical seam remains a useful full-stack
compatibility check, but it is not the implementation contract for an early
causal prefix.

## 5. Engineering smoke

The smoke is fixed and non-scientific:

```text
session: sub-C_ses-CO-20131003
budget: M30
support: chronological positions 0--29
audit rows exercised: chronological positions 30--31 only
groups: all K=4
batch size: at most 128 endpoints per forward chunk
```

It proves the real parser, three physical trial views, raw-T4/validity join,
strict model load, held-group slicing, behavior restoration, grouped outcome
construction, memory bounds, and receipt lifecycle.  It does not apply or
relax the B8 scientific threshold because two rows are not the frozen M30
screen.

The smoke still descriptor-binds the immutable strict-27 manifest, but it may
resolve and open **only** `sub-C_ses-CO-20131003`.  It must not parse or open
the other 26 source NWB files merely to construct a 27-session authority
table.  The full source gate is the first route allowed to open all 27 source
sessions.  Smoke authority/evidence therefore carries one physical session
row plus the separately bound strict-27 manifest roster.

Prospective root:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v1
```

Expected immutable topology is attempt, launch, source_authority, smoke, and
terminal pairs, or attempt plus one typed failure pair.  Attempt must be
durable before source resolution.  A smoke success authorizes only the full
source gate, not within/external scoring.

## 6. Full strict-27 source gate

Run every strict source session at a budget before deciding that budget.
Apply the unchanged three B8 thresholds per session.  The route-level breadth
rule is predeclared here, before opening source data:

```text
at least 14 of 27 source sessions must pass all three frozen B8 gates
```

This breadth rule does not modify any B8 threshold.  Fixed-pool rejection,
missing required chronology, or no defined valid-unit cosine counts as a
non-passing session and remains in the denominator.

Fail fast by budget:

1. evaluate all 27 M30 sessions; stop if fewer than 14 pass;
2. if M30 passes, evaluate all 27 M10 sessions; stop if fewer than 14 pass;
3. if M10 passes, evaluate all 27 M4 sessions.

Report all per-session evidence and aggregate medians/counts.  Do not tune a
trajectory, trust, B8, or breadth threshold after observing these source
results.

Prospective root:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v1
```

The full root contains attempt, launch, source_authority, one immutable
per-session/per-budget evidence pair as reached, one aggregate pair per reached
budget, and terminal; or attempt plus one typed failure.  The terminal must
distinguish `PASS_SOURCE_CONSTRUCTIBLE` from the scientifically honest
`STOP_SOURCE_B8_CONSTRUCTIBILITY`.  STOP is a successful completed screen, not
a runtime failure.

## 7. GPU selection and resource contract

The route may use either idle local RTX 3090 after root selection.  It must bind
one complete existing compatible profile from
`posterior_marginalized_cell_d_v1.plan.COMPATIBLE_DEVICE_PROFILES`; never infer
physical identity from the CUDA ordinal.  Require exactly one visible device,
logical `cuda:0`, matching UUID/BDF/name, separate nominal MiB and Torch byte
authorities, Torch `2.5.1.post303`, CUDA `11.8`, cuDNN `90300`, and both TF32
flags false.  Restore route-local runtime flags on every exit.

Record endpoint chunks/s, trials/s, wall time, RSS, current and peak CUDA
allocated/reserved bytes, and the selected exact device profile.  OOM is a
typed failure and must not trigger an unreviewed batch-size retry.

## 8. Lifecycle and authorization

The public CLI is standard-library-only at import and dry by default.  A public
flag without an opaque in-process root capability must fail before data,
checkpoint, CUDA, or root reservation.

An authorized path must:

1. rehash the current explicit implementation closure;
2. descriptor-reload all fixed authorities;
3. validate selected device environment and fresh prospective root;
4. publish immutable attempt before resolving source data;
5. revalidate closure/authority after execution;
6. publish transactional terminal evidence or one typed failure.

Every body/sidecar is a regular non-symlink `0444` pair with canonical basename
sidecar text.  Hold the result-root directory FD across publication and reject
aliases, symlinks, preexisting leaves, topology drift, or inode replacement.

## 9. Mandatory no-data/no-CUDA tests

At minimum prove:

1. exact accepted Stage-0 and Source-Audit closure binding;
2. exact fixed-asset SHA/mode/sidecar validation;
3. strict-27 train-only prebinding and rejection of every non-source resolver;
4. exact source normalizer constants and raw-before-normalization flow;
5. theta raw/channel/valid-mask join, including the known 1+2 invalid rows;
6. fixed smoke session/budget/row identities;
7. exact M4/M10/M30 chronology and no substitution for missing rows;
8. attempt-before-source/checkpoint/CUDA/root ordering;
9. held-FD result publication, reloading, and failure rollback;
10. strict Cell-D SWA schema/lazy-topology/state-digest validation using a
    synthetic artifact only;
11. all K=4 held inputs are physically sliced and no group label is broadcast;
12. source labels cannot enter a forward or state before four pseudo outcomes
    are finalized;
13. M4/M10 physical B3S calls use exact variable causal-prefix lengths from M
    through 30; future-row injection, zero padding, support repetition, and a
    hidden fixed-30 substitution are rejected;
14. invalid-theta rows remain bound but excluded from B8 defined counts;
15. per-session gates plus exact `14/27` breadth and M30->M10->M4 fail-fast;
16. STOP is terminal science evidence, while parser/model/CUDA/OOM is failure;
17. exact dynamic GPU0/GPU1 profile selection and wrong-ordinal/UUID/BDF/
    memory/runtime/TF32 rejection using fakes only;
18. no OOM retry and resource schema validation;
19. closure is rechecked both before and after injected execution;
20. dry CLI imports no Torch and writes/opens/launches nothing;
21. public execution flags fail before all physical side effects.
22. exact fixed-ridge-0.1 parity for the support-only initial carrier at every
    budget, plus an adversarial M30-mutation test proving no M30 value/label
    can affect the M4/M10 carrier, K4 map, first prediction inputs, or initial
    dual-memory state.
23. initial support-rate parity with exact-duration historical comparator
    semantics, and explicit non-equivalence coverage against the native-20ms
    online update rate when trial boundaries/exposure are not bin-exact.

## 10. Stop boundary

Return the frozen implementation, all owned SHA256 values, explicit closure,
exact tests/compile/whitespace results, and live-only blockers.  Stop without
opening source data, loading checkpoint tensors, initializing CUDA/GPU,
creating roots/receipts, or launching.  Root will independently audit and then
decide smoke/full execution and Luna monitoring.
