# SUA/MUA functional-carrier closure roadmap

**Updated:** 2026-08-11 HKT
**Status:** mainline contraction; no open-ended architecture search

For scientific context and all current numbers, start with
[`docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`](docs/HANDOFF_MAINLINE_CLOSURE_20260811.md).
This roadmap defines only the remaining execution order and stop conditions.

## Goal

Close a paper-ready, evidence-backed claim that an analytic functional carrier can adapt a frozen streaming neural
decoder to new sessions without target-session backpropagation, and can trade substantially lower target-supervision
density for competitive accuracy on the tested movement-decoding tasks.

Completion does not require every proposed carrier, decoder fusion, or quantization experiment to run. It requires
the fixed RT endpoint experiment, its exact SPINT-structured comparator, terminal verification, current documentation,
and reproducible code/evidence references to be complete.

## P0 — RT sparse endpoint matrix

**Owner:** remote RTX 5070 Ti supervisor; Luna event monitor; root review
**Contract:** 15 folds × `T4d/Full/Zero4`, seed42, M24/q24/W50, 35 epochs

Current state: 13/15 complete paired folds (`39/45` cells); fold12 is fully closed and included, and fold13 T4d is
running. Every fold13 score remains excluded from partial aggregates until all three fold13 arms close.

Frozen primary contrast:

```text
T4d - Zero4
```

Report mean, median, positive-fold count, and mean after dropping the single largest session. The mechanism is
supported only if mean and median are positive and the positive-fold count is a majority.

Frozen secondary contrast:

```text
T4d - dense Full
```

This is an operational supervision-density comparison. Do not call it equivalence from a near-zero mean alone.
Report the full paired distribution and the predeclared `0.03 R²` magnitude/non-inferiority annotation. T4d need not
beat Full for the route to be useful.

Stop or fail closed if any cell lacks fresh-fit evidence, exactly-once outer evaluation, identical within-fold query
identity, no-target-backprop evidence, or model-state immutability.

## P1 — RT independent terminal verification

Run only after `45/45` cell closures exist. The verifier must independently reconstruct the grid and paired
statistics from raw outer R²; it must not trust the supervisor aggregate.

Required checks:

- exact 15×3×seed42 grid;
- manifest/source/config/teacher/NWB binding;
- closure→selection→checkpoint→split→outer receipt chain;
- matched ordered query digests within every fold;
- target optimizer absent, backward false, model state unchanged;
- recomputed mean, median, sign count, drop-largest mean, and frozen magnitude annotations.

No paper or `CURRENT_RESULTS` terminal update occurs before this verifier passes.

## P2 — uniform B2 exact-query companion

After P1 passes, evaluate all 15 sealed `B2-D1024` checkpoints on the exact Stage-2 query windows.

Constraints:

- forward-only;
- no retraining, optimizer, backward, checkpoint reselection, or historical-score reuse;
- same Stage-2 NWB allowlist and ordered query digest as T4d;
- model-state hash identical before and after evaluation;
- report the full 15-fold result and the score-blind prospective folds4--14 subset.

The comparator name is `matched SPINT-structured B2-D1024`. Do not call it a bit-exact reproduction of every
released-code RT implementation.

## P3 — paper and documentation closure

After P0--P2:

1. update `docs/HANDOFF_MAINLINE_CLOSURE_20260811.md` with terminal RT and B2 numbers;
2. replace the chronological bulk in `docs/CURRENT_RESULTS.md` with a concise dataset/claim ledger while preserving
   receipt and SHA pointers;
3. reduce `docs/ACTIVE_EXPERIMENT_CONTROL_BOARD.md` to live/terminal state and remove retired queue narratives;
4. update both full and six-page Overleaf sources with terminal-only claims;
5. verify that README, roadmap, handoff, result ledger, and paper use identical numbers and scope language.

## P4 — workspace cleanup and Git synchronization

Cleanup begins only after active RT processes and terminalizers exit. H1 evidence paths remain protected while its
runner is active.

1. rerun `ps`, GPU, tmux, `lsof`, and inbound-reference checks;
2. move byte-identical periodic checkpoint aliases to a recoverable quarantine with SHA and retained-peer paths;
3. archive only historical documents with no active basename/stem/script/test/receipt/paper reference;
4. keep immutable terminal receipts, best checkpoints, query digests, and aggregate evidence;
5. run focused tests and `git diff --check`;
6. stage exact reviewed paths only, commit, push, and verify the remote SHA.

No blanket `git add .`, no deletion of ignored evidence roots, and no destructive cleanup of active runs.

## Secondary work that does not block closure

### H1 CI64

The five-date CI64 consumer-width matrix was already running when the mainline contracted. It may finish under its
original serial schedule or be handed to another maintainer. Its result can refine the H1 compact-consumer section
but cannot support sparse supervision and cannot reopen carrier/decoder search.

### Quantization

INT8/INT16 work is deferred. A hardware-oriented follow-up may quantize the selected encoder/carrier path after the
FP32 scientific claim is frozen. Decoder quantization is out of scope for this closure.

### Native M2 three-seed gate

The fresh r10 three-seed program belongs to a separate workflow. The existing matched seed42 Stage-A result remains
development evidence. Do not splice historical seed42 cells into a new lineage.

## Closed directions

Do not allocate new GPU work to the following under the current endpoints:

- waveform/SNR/static electrode lookup and same-electrode relation features;
- T4GATE and fixed scalar reliability gates;
- N4 static label-free statistics;
- fixed-K memory and tested closed-form alignment pilots;
- RT L-D gain modulation, deeper FiLM, or carrier-biased attention;
- M1 EMG-AFC4, D4, or Version-B rescue;
- H1 H-PCF8 or native-phase post-hoc repairs;
- new carrier widths, harmonics, lags, or label budgets selected after seeing target results.

A future project may revisit a closed family only with a new dataset, a genuinely different estimand, and a new
source-only pre-registration. It is not remaining work for this paper.

## Final completion checklist

- [ ] RT 45/45 closures
- [ ] RT independent terminal verifier PASS
- [ ] B2 exact-query 15-fold forward-only companion PASS
- [ ] terminal numbers in handoff, result ledger, and Overleaf agree
- [ ] active board contains no stale running status
- [ ] historical documents/duplicate checkpoints archived with manifests
- [ ] focused tests and `git diff --check` pass
- [ ] narrow Git commits pushed and remote SHA verified
