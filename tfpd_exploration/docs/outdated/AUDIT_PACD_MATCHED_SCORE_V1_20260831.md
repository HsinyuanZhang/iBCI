# Independent Audit: PACD Matched Score V1

Date: 2026-08-31

Audit status: **code candidate accepted at the no-data/no-CUDA boundary; live
execution remains NO-GO**.

The live NO-GO is intentional. Exact P0/P1/P2 producer terminal, SWA,
manifest, attempt, launch, source-authority, and final-four checkpoint body
digests do not yet exist. The implementation represents them as deferred
typed literals and cannot mint a live capability while any is absent.

## 1. Authoritative scope

Work order:

`tfpd_exploration/docs/WORKORDER_PACD_MATCHED_SCORE_V1_20260831.md`

Owned implementation:

- `tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/`
- `tfpd_exploration/scripts/run_pacd_matched_score_v1.py`
- `tfpd_exploration/tests/test_pacd_matched_score_v1.py`

Current explicit execution closure:

`e81648d0f4ac867665e57893dad421ce210cccbb80f4c1713c748b93a03bb893`

## 2. Requirement-to-evidence matrix

| Work-order requirement | Current implementation evidence | Audit verdict |
|---|---|---|
| Exact system order `P0,P1,P2,T0,C1,SD` | `plan.SYSTEM_ORDER`; canonical row constructor and exact schema validator | Proven in no-data tests |
| Surfaces within-6 and external-15 | exact route rosters and cardinality validation in `materialize_authority` and lifecycle input validation | Proven synthetically; live roster proof deferred |
| Budget order M30, M10, M4 | `plan.BUDGET_ORDER`; canonical receipt ordering | Proven |
| Honest-total and activity-isolation regimes | `authority_record`; activity-isolation uses literal `selected_by_budget[30]` while keeping budget-M T4 | Proven synthetically |
| One target materialization per session | session-outer `materialize_authority`; system callback receives no target locator or parser | Proven by call-count test |
| Same inputs for all six systems | 126 typed `InputRecord`s; exact record digest; `assert_same_input` over every six-row cell | Proven |
| Exactly 756 rows | 21 sessions × 3 budgets × 2 regimes × 6 systems; exact uniqueness/topology validation | Proven |
| Governed last-bin variance-weighted R2 | `prediction[:,49,:]` and reviewed `matched_scorer.session_r2` | Real CPU metric seam proven; CUDA remains uninitialized |
| Strict static no-dropout forwards | strict SWA swap, eval mode, recorder zero-call proof, no target backward/update | Proven synthetically and at real house metric seam |
| Repeated forward and state immutability | prediction/identity repeat digests and equality; state before/after equality | Proven |
| Fixed summaries and seven contrasts | reducers reconstruct from canonical rows; no callback aggregate is trusted | Proven at all cells |
| Primary and incremental gates | exact work-order thresholds; activity-isolation cannot rescue honest-total failure; incremental requires primary | Boundary/adversarial tests pass |
| Historical T0/C1/SD fixed comparators | exact body/SWA literals; route-specific real terminal codecs; descriptor/sidecar rehash | Clean `python -S` real integration passes without Torch/tensor/data load |
| Deferred P0/P1/P2 producer graph | typed `PACDProducerBinding`; exact V3 48-epoch/four-checkpoint/SWA/source-only graph validator | Fail-closed while live literals are absent |
| Attempt before target/model/CUDA | lifecycle publishes immutable attempt before the materialize-and-score callback | Ordering test passes |
| Separate authority and result roots | two root witnesses, independent parent/root inode bindings | Proven |
| Opaque one-shot capability | secret-only `ScoreCapability`; binding/closure/producer/comparator/authority/profile/root witnesses; consumed once | Construction, drift, reuse adversaries pass |
| Atomic score publication | hidden `.stage-*` contains input/score/terminal; one directory rename publishes `complete/` | Success and injected-failure tests pass |
| No partial canonical score on failure | every pre-rename failure removes only private staging and leaves exact attempt/failure topology | Failure after every staged leaf and corrupt sidecar passes |
| No contradictory failure after commit | post-rename observer failure preserves `complete/` and mints no failure | Proven |
| Root replacement protection | parent and named-root dev/inode/name rebound and rechecked | Adversarial replacement test passes |
| CPU-only isolation | route-owned attestation requires empty CVD, CPU selection, CUDA uninitialized; CPU model/prediction placement asserted | Proven; prevents contention with GPU0/GPU1 jobs |
| Explicit transitive closure | fixed no-glob source paths include scorer plus reviewed CAL-AUG/P4/Cell-D/receipt dependencies; tests/results excluded | Double reconstruction identical |

## 3. Verification completed

Focused scorer command under no-user-site, empty CVD, bytecode off, plugin
autoload off, and one-thread math libraries:

```text
19 passed
```

Combined PACD V1/V2/full-V1/full-V2/full-V3 plus scorer regression under the
same isolation:

```text
77 passed
```

One warning remains: a TorchMetrics deprecation warning for the historical
`num_outputs` argument. It does not alter the metric or result.

Additional checks:

- `py_compile`: pass;
- `python -S ... --dry-run`: pass, Torch not imported;
- real T0/C1/SD descriptor integration: pass;
- source closure reconstructed twice identically;
- `git diff --check`: pass;
- PACD running 48-file producer closure remains unchanged.

## 4. Exact atomic topologies

Success:

```text
result/
|-- attempt.json
|-- attempt.json.sha256
`-- complete/
    |-- input_authority.json
    |-- input_authority.json.sha256
    |-- score.json
    |-- score.json.sha256
    |-- terminal.json
    `-- terminal.json.sha256
```

Failure before commit:

```text
result/
|-- attempt.json
|-- attempt.json.sha256
|-- failure.json
`-- failure.json.sha256
```

Target-free authority uses the same directory-commit law with one published
`authorized/` bundle.

## 5. Remaining live-only requirements

The candidate must not be described as launch-ready until all of the following
are true:

1. PACD P0 reaches a valid 48-epoch terminal with checkpoints 44--47, final-four
   SWA, manifest, strict reload proof, source-only facts, and current closure.
2. The work-order P0 validity gate is independently checked.
3. PACD P1 and P2 each run once under their own immutable roots and reach the
   same complete producer topology.
4. Exact full 64-character producer body digests are inserted into a reviewed
   immutable binding; no prefix or caller path is accepted.
5. The scorer closure is recomputed after that binding edit and its focused and
   combined suites are rerun.
6. Historical comparator and three producer graphs are reloaded through held
   descriptors immediately before authority mint.
7. Both prospective roots are fresh and have stable parent identities.
8. The CPU-only live environment is exact and CUDA has not initialized.
9. A separate score launch decision is made; no retry uses the same root.

Until these conditions hold, the public CLI remains inert and live capability
minting remains impossible. This is a positive safety property, not an
implementation blocker.

