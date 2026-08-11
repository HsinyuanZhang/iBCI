# SUA/MUA functional-carrier post-terminal roadmap

**Updated:** 2026-08-11 HKT
**Status:** RT and H1 accuracy closure complete; only CPU supervision-density audit and maintenance remain

For scientific conclusions and all accepted numbers, use
[`docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`](docs/HANDOFF_MAINLINE_CLOSURE_20260811.md).
This roadmap lists remaining work and stop conditions; it does not authorize a new architecture search.

## Completed accuracy closure

### RT sparse endpoint carrier

- all 45 `T4d/Full/Zero4` cells and all 15 paired folds completed;
- independent terminal verification passed;
- all 15 sealed B2-D1024 checkpoints were evaluated forward-only on the exact Stage-2 queries;
- T4d−Zero4 and T4d−B2 are positive in all 15 folds;
- terminal values are synchronized across the result ledger, handoff, full paper, and six-page paper.

Dense Full is separate context only. The terminal matrix does not establish T4d superiority, equivalence, or
non-inferiority relative to dense Full, and endpoint-derived RT directions do not support a native annotation-cost
claim.

### H1 compact consumer and width control

- the original five-date compact CarrierID matrix and organizer-held system comparison are closed;
- all 25 CI32/CI64 source checkpoints and five one-shot held-date evaluations completed;
- the independent 5×5 verifier passed;
- CI64 did not improve CI32, so H64 escalation is stopped;
- H1 remains dense-covariate compact-consumer evidence, not sparse-label evidence.

### Reproducibility and paper synchronization

- RT terminal workflow is committed and clean-clone tested;
- H1 CI64 runtime/config/test closure is committed and clean-clone tested;
- main result ledger and current handoff are synchronized to GitHub;
- full and six-page paper sources are synchronized to Overleaf.

## P0 — corrected Priority-A2a weighting control

The first A2a CPU receipt is retained only as an invalid implementation diagnostic. Its weighted ridge used
`Z^T W Z + lambda I` without normalizing by total weight, so its nominal `lambda=1` was not equivalent to the sealed
normalized Ridge50 penalty. It also did not fail closed on missing direction labels.

The corrected, independent v2 solver and runners are committed at `097d137`; 23 focused tests establish:

1. `(Z^T W Z / sum(w) + lambda I) beta = Z^T W Y / sum(w)` or an algebraically identical formulation;
2. uniform weights reproduce the sealed normalized unweighted solver within a frozen tolerance;
3. globally rescaling all weights leaves predictions unchanged;
4. missing or non-finite direction labels fail closed;
5. all session/view/budget cells bind support starts, query starts, predictions, targets, and zero overlap.

The full A2a v2 CPU run is in progress under `a2a_v2_corrected_20260811_r2`. Its immutable receipt must still be
audited before A2b is released. A corrected A2a can show whether the dense-versus-direction-only contrast is robust
to row weighting. It still cannot isolate label density because the target contents differ.

## P1 — same-target Priority-A2b density dose response

Run only after the corrected A2a receipt passes the sealed 90-cell reproduction audit. Keep dense normalized velocity, neural features, support
trials, query rows, standardization, lambda, solver, intercept, and equal-trial weighting fixed; vary only the nested
number of labelled windows per trial.

This is the only current experiment that can identify a label-density effect. T4 may appear as a frozen reference
line, but `T4−ridge(K=1)` remains a system comparison because T4 uses a source-pretrained decoder and a direction
scalar. A legacy draft A2b process using the invalid solver was stopped before producing a receipt; it is not a
negative scientific result. The versioned v2 A2b runner is fail-closed on the corrected A2a receipt and has not yet
been launched.

## P2 — maintenance closure

1. audit A2a/A2b receipts before changing any paper sentence;
2. keep `CURRENT_RESULTS.md`, the current handoff, README, roadmap, and both paper versions consistent;
3. archive or ignore only files proven unreferenced by docs, imports, Hydra defaults, scripts, tests, and receipts;
4. retain terminal receipts, aggregates, query digests, and canonical checkpoints;
5. run focused tests and `git diff --check`;
6. stage reviewed paths only, commit, push, and verify remote SHAs.

## Scope-limited work outside this closure

- Native M2 has positive seed-42 development and organizer-held system evidence, but lacks the fresh three-seed r10
  aggregate required for a final matched causal or equal-label claim. That missing experiment does not invalidate the
  narrower evidence already reported and is not a blocker for RT/H1 closure.
- INT8/INT16 and RTL work remain future implementation tasks after the FP32 scientific contract is frozen.

## Closed directions

Do not allocate new GPU work under the opened endpoints to waveform/SNR/electrode lookups, T4GATE, N4 static
label-free statistics, fixed-K memory, tested closed-form alignment, RT L-D, deeper FiLM, carrier-biased attention,
M1 rescue, H1 native-phase rescue, larger H1 consumer widths, or post-hoc label-budget searches.

A closed family may be revisited only under a new dataset or genuinely different estimand with a new source-only
protocol. It is not remaining work for the current paper.

## Completion checklist

- [x] RT 45/45 cells and 15/15 paired folds complete
- [x] RT independent verifier PASS
- [x] B2 exact-query 15-fold companion PASS
- [x] H1 CI64 25/25 cells, five held dates, and verifier PASS
- [x] terminal RT/H1 numbers synchronized to GitHub and Overleaf
- [x] RT terminal checkpoint audit and path-preserving duplicate hardlink cleanup complete
- [ ] corrected A2a receipt audited
- [ ] same-target A2b receipt audited, or causal label-density wording explicitly omitted
- [ ] final reference-aware workspace cleanup and Git audit complete
