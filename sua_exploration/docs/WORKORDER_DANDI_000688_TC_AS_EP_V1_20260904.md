# Work Order — DANDI 000688 TC-AS-EP V1

Status: `AUTHORIZED_BY_USER__STAGE0_THEN_CONDITIONAL_GPU`

Date: 2026-09-04 (Asia/Hong_Kong)

## 1. Objective

Implement and evaluate the reviewed T4-Covered Activity-Selected Early-Pooling
route on DANDI 000688. The practical objective is a source-development
performance improvement for small activity support (`K=10`), with
noninferiority as the minimum acceptable outcome. If the activity route is
nonnegative, a separate calibration-profile FiLM successor may be opened to
test the cross-dataset mechanism already positive on M2 and H1.

## 2. Governing design

The governing design is:

`sua_exploration/docs/DESIGN_DANDI_000688_ACTIVITY_SELECTED_EARLY_POOLING_V1_20260904.md`

SHA-256:

`2278e882eebb9728ba7610617627c3b05f38992b05767bd1d4862d91a38cca2b`

The late clarification freezing robust-scale `epsilon=1e-12` is included in
that digest. The independent review verdict was `GO` for Stage 0; the user then
explicitly authorized experiment ownership and GPU execution without another
permission round.

## 3. Frozen route

- parent protocol: V9;
- candidate activity: first 50 legal rewarded trials;
- activity support: 10 trials;
- T4 carrier: first 50 direction labels;
- source query windows: trials 50 onward;
- topology: native B3S early pooling;
- training arms: `M30-V9-REPRO`, `COV-FIX-E10-EP`,
  `COV-FIX-L10-EP`, `COV-RAND10-EP`;
- epochs: exactly 12;
- seeds: 42, then 43/44 only if seed 42 passes the frozen continuation rule;
- optimizer/loss/decoder training: match the selected ordinary B3S/T4 lineage;
- no target-session optimizer, backward pass, or parameter update.

## 4. Stage 0 authority

Stage 0 may open only the 27 source-training and six source-development NWBs
named by the strict manifest. Formal-test names may be receipt-read but their
paths and files must not be resolved or opened. External sub-M15 is closed.

Stage 0 shall:

1. prove `C=50`, `K=10`, `Q=50` are independent;
2. compute raw half-open spike-time rates and the frozen robust feature;
3. construct fixed, random, Top10, and Anti10 direction-covered supports;
4. apply the exact bias/stability gates from the design;
5. prove stateless sampling entropy and no global RNG drift;
6. bind the `20150313` invalid direction behavior;
7. run historical C30/C50 operator parity only on common Q50 windows;
8. exercise the task-only/lambda-E and validation teacher dependency checks;
9. publish immutable attempt, authorities, and terminal/failure receipts.

No decoder R2 or other performance metric may be computed during Stage 0.
Stage 0 selector attempt count is limited to two; this work order names
`stage0_attempt1` and does not silently reuse it.

## 5. GPU authority

After a Stage-0 `PASS`, GPU0 is authorized for the seed-42 four-arm training
screen. Multi-arm coordinated execution on one GPU is preferred when it
preserves:

- byte-identical within-seed initialization;
- identical resident task batches and paired RNG/dropout law;
- independent parameters, gradients, optimizer states, and buffers;
- separate checkpoints and receipts per arm.

GPU1 should remain available for unrelated work unless GPU0 is unavailable.
Seeds 43/44 are conditional on the frozen seed-42 continuation result. This
authorization does not authorize EvalAI submission or formal/external target
opening.

## 6. Fail-closed rules

Stop without repair or configuration drift if:

- Stage 0 fails twice;
- any formal/external target file is opened;
- selector input includes query activity, dense behavior, predictions, losses,
  gradients, or validation R2;
- a direction-invalid trial enters a covered support;
- C/K/Q, T4 budget, query start, training horizon, or seed law changes;
- an existing result or checkpoint root would be reused or overwritten.

An additive successor with a new work order is required for CP-FiLM,
cardinality cycling, external retrospective scoring, or EvalAI packaging.
