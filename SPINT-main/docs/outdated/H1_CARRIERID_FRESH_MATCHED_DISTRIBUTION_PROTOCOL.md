# H1 CarrierID fresh matched D-S4 / D-Q4 distribution diagnostic

**Status:** completed under-exposed development leakage diagnostic; the original
branch gate is invalidated by a training-exposure confound. The exposure-matched
repair is source-audited and running/queued as two independent arms. This document
does not authorize formal test or EvalAI submission.

## Exposure-matched repair execution status (2026-08-08)

The repaired source-only preflight is immutable receipt
`pilot_artifacts/h1_carrierid_distribution_exposure/H1_CARRIERID_H32_FRESH_D_S4E_D_Q4E_SOURCE_CPU_PREFLIGHT_v1.json`
(SHA-256 `21a9a4fa0dd3cbf1b8f59fa7dbffb85d9448a2fc596df1ca4d153aa9f8842bdf`).
It binds both arms to the same 72 eligible schedules, `115,520` unique scheduled
samples per epoch, batch size 32, `3,610` batches per epoch, 50 epochs, common
batch order, common initialization, and the sealed H-C training-exposure scale.

`D-S4e` began running on the local RTX 3090 immediately after the seed-43 H-S
job completed. `D-Q4e` is queued on the remote RTX 5070 Ti after the seed-43 H-C0 job, using an isolated
staging tree. The remote no-launch receipt is mode `0444`, SHA-256
`e41abd2559ff8301bd5b70a4f4a4d279510b041995c33192bfd2cc0b1c48a061`; root
independently rechecked all 15 source-closure hashes and the associated contract
tests passed 6/6. Neither queue has opened a target, minival, formal, or EvalAI
endpoint. `D-Q4e` remains a deliberate leakage diagnostic and cannot become a
deployable or paper-main arm.

## Completed outcome and exposure audit (2026-08-08)

Both fixed epoch-49 checkpoints and the one-shot fold-0 target evaluation passed
their byte/state/data-boundary checks. `D-S4=-0.1622759219`,
`D-Q4=-0.1534490772`, and `D-Q4-D-S4=+0.0088268448`; the two recording deltas
are `+0.0100305099` and `+0.0053611494`. The evaluator used float64 SSE/TSS,
changed no model state, and opened no minival, formal, or EvalAI endpoint.

These numbers **must not trigger the estimator/consumer branch table below**.
The fresh pair used only 72 batches / 2,304 samples per epoch and 3,600 total
optimizer steps. The sealed H-C reference used 3,610 batches / 115,520 samples
per epoch and 180,500 total steps. Training exposure was therefore reduced by
approximately 50.14x, and both fresh arms collapsed far below sealed
`H-C=0.5255107931`. The pair is internally matched but not capacity/exposure
matched to the result whose headroom it was intended to diagnose.

The next valid repair is exposure-matched D-S4/D-Q4 training: retain the common
paired-eligible carrier schedules, but construct the same 115,520 scheduled
samples per epoch, batch size 32, 3,610 batches per epoch, 50 epochs, and common
batch order as the sealed reference family. No estimator-versus-consumer claim
is permitted until the repaired D-S4 recovers a credible absolute baseline.

## Question and correction to the old gate

The prior frozen H-C test-time replacement experiment used a consumer trained
on ordinary M=4 support carriers, then replaced its target-time carrier with
one fitted from query-local labels. Its null result cannot decide whether the
estimator lacks information or whether the consumer receives an
out-of-training-distribution carrier.

The old reading that treated replacement as a mandatory **estimator gate is
withdrawn**. Replacement is neither a strict upper bound nor an
estimator-quality verdict. This protocol supersedes it with a matched,
from-scratch source comparison.

## Two-arm contract

Both arms use the existing H1 CarrierID network unchanged: `h=32`, carrier
dimension 4, seed 42, Adam `lr=5e-5`, fixed 50 epochs / terminal epoch 49, and
no validation selection. Setup opens only the 11 fold-0 source recordings.

| Item | D-S4 | D-Q4 |
|---|---|---|
| Identity | first four trials `t..t+3` | identical S4 identity |
| Carrier | frozen estimator on `t..t+3` | frozen estimator on `t+4..t+7` |
| Query windows | same 32 strict windows beginning at/after first bin of `t+4` | identical |
| Neural/behaviour, session, batch order, normalizer | shared | shared |

Every one of the eight trials needs at least two legal 100-ms rate blocks. The
first four also need cubic identity interpolation. Both raw carrier stacks are
fitted over exactly this common schedule; the single scalar normalizer is
fitted to their concatenation. D-S4 therefore cannot consume any carrier
schedule, normalizer element, query window, identity, or training sample that
D-Q4 does not also receive.

D-Q4's later-block carrier is explicitly a **source analogue of deliberate
query-label leakage**. It matches a possible query-local target distribution;
it is not a deployable calibration rule and uses no target data at this stage.

## CPU preflight and launcher

`scripts/h1_carrierid_distribution_preflight.py` instantiates both real
DataModules and models, loads exactly 11 source NWBs, and makes one CPU forward
per arm. It immutably records source closure, common schedule/query-window and
batch hashes, first-batch neural/target/identity/session/window/carrier hashes,
shared initialization (also bound to sealed H-C initialization), and source
carrier-distance statistics.

It fail-closes if CUDA is visible/initialized or a target, minival, formal, or
EvalAI route is requested. It creates neither a Trainer nor a checkpoint. The
receipt is write-once mode `0444`. The plan-only launcher has no execution
mode: it prints future commands only after re-hashing source closure and keeps
`launch_authorized=false`.

## Frozen interpretation

Every future target-side comparison must be named
`LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT`. It cannot
enter the paper main result, deployment claim, target selection, or support a
target-time-label method.

| Future paired outcome | Frozen conclusion / next route |
|---|---|
| D-Q4 clearly exceeds D-S4 under the full matched contract | estimator hypothesis only; prepare `H1-EST4-SLODO` and require its independent source-only gate |
| D-Q4 does not improve | consumer hypothesis only; prepare `H1-CI64-SLODO`; `H1-H64-SLODO` is prohibited unless CI64 later passes its source-only mechanism gate |
| mixed across paired recordings/seeds | prepare `H1-DIST-XDATE-REPL`; do not select an estimator or consumer architecture, or tune after inspection |

Low cosine or a large Frobenius difference proves only different inputs, not
better inputs. D-Q4 is not a strict upper bound: its extra label scope can
degrade a matched consumer as well as improve one.

`C7` is retired from this H1 protocol: it was undefined here and already names
unrelated experiments elsewhere in the repository.  The canonical names and
the source-only selection/retraining contract are fixed in
`H1_CARRIERID_NEXT_ROUTE_AUDIT_20260808.md`.  The D-S4e/D-Q4e result remains a
leakage diagnostic, so it may classify a development hypothesis but cannot by
itself select a width, epoch, checkpoint, or final architecture.

## Additive isolation

This diagnostic adds `h1_carrierid_distribution` data/model/config/script/test
files only. It does not modify sealed H-C, RS/LS, terminal evaluators, target
evaluators, or historical caches. New cache/receipt output belongs only under
`pilot_artifacts/h1_carrierid_distribution/`.
