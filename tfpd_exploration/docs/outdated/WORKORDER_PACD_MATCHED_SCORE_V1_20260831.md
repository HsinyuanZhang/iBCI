# Work Order: PACD Matched Score V1

Date: 2026-08-31

Status: authorized for scorer implementation and no-data/no-CUDA review only.
Execution is impossible until all three PACD full-training terminals and SWAs
exist and their exact immutable body digests are inserted through a reviewed
producer binding. This work order does not authorize target access or scoring.

## 1. Question and systems

The scorer answers whether paired anchoring improves low-budget transfer while
preserving M30. It evaluates exactly six static Cell-D-weight systems in this
order:

```text
P0  PACD FullFull seed42
P1  PACD M4 seed42
P2  PACD M10 seed42
T0  historical matched CAL-AUG operator-disabled control
C1  historical CAL-AUG chronological prefix-cycle treatment
SD  sealed ordinary Cell-D producer
```

PACD is a training-only intervention. At scoring, P0/P1/P2 each use one
ordinary deterministic no-dropout Cell-D forward. No paired branch, prefix
cycle, training hook, optimizer, state update, or target adaptation is active.

## 2. Fixed completed comparators

The historical comparator artifacts are immutable and fixed before PACD
producer results exist:

```text
T0 terminal
tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/terminal.json
3071d907df2a91cc409d85997ac3f401a9e1af05ab75d902f332370cb8e461c7

T0 SWA
tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/swa_final4.pt
b4781d71ae408ba306edc9268597b6ce83600138c0acaa08dbdf5a1a409e7e86

C1 terminal
tfpd_exploration/results/cal_aug_v1/c1_prefix_cycle/terminal.json
320e2b9991c75ed1bcb87fab73643ca8a12bf398133d5358a33e32c8fd72d738

C1 SWA
tfpd_exploration/results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt
5cc24676777cb1eacb0ce5fd5174c55dc2efd5ca9ede69d24a85f935f8e89f8d

SD terminal
tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json
b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442

SD SWA
tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt
626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd
```

Their terminals must validate 48 epochs, final-four epochs 44-47, strict fresh
reload evidence, and launch/final closure equality. The scorer rehashes their
body and canonical sidecar through held descriptors before use.

## 3. Deferred PACD producer binding

P0/P1/P2 exact terminal, SWA, manifest, attempt, launch/source-authority, and
checkpoint body digests do not exist yet and must not be invented. The scorer
package therefore accepts one typed immutable `PACDProducerBinding` only at
construction. A dry or synthetic test may use a temporary binding; a live
capability requires exact literals from completed canonical producers.

The live binding must prove for every PACD arm:

- exact canonical arm/root identity;
- accepted V2 smoke predecessor;
- 48 epochs and 1,628,400 optimizer updates;
- exact seed, source roster, batch order, optimizer, LR, and paired-step law;
- four checkpoints from epochs 44-47;
- final-four SWA strict-load and state-digest proof;
- source-only/no-target facts;
- terminal status and launch/final closure equality.

No caller-provided path or partial SHA prefix is accepted. The scorer must
descriptor-read and exact-compare the completed producer graph before any
target input is resolved and again before terminal.

## 4. Surfaces, budgets, and evaluation regimes

The fixed surfaces are:

```text
within   exact six predeclared development sessions
external exact fifteen predeclared sub-M sessions
```

The fixed budgets and score order are `M30`, `M10`, `M4`. Two regimes are
required for every system/surface/session/budget:

### 4.1 Honest total calibration

Both B3S activity and T4 support obey budget M:

```text
B3S activity = selected M calibration trials
T4            = fixed ridge-0.1 fit on the same selected M support
M4 selection  = causal D-opt-first30
M10 selection = chronological first10
M30 selection = chronological first30
```

This is the deployment-relevant primary regime and must reproduce the frozen
CAL-AUG deployment input records for T0/C1 before adding PACD systems.

### 4.2 Activity-isolation regime

The carrier remains budget-limited but calibration activity is held at M30:

```text
B3S activity = chronological first30 calibration trials
T4            = the same fixed budget-M ridge-0.1 carrier as total calibration
```

This separates learned short-prefix activity robustness from carrier loss.
It is diagnostic and cannot replace the honest-total primary gate.

All systems reuse the exact same materialized input authority within a row.
The scorer must not reopen or rematerialize a target session per system.

## 5. Metric and evidence

For each row, use the governed last-bin score:

```text
prediction[:, 49, :]
R2Score(multioutput="variance_weighted")
```

Report exact per-session R2 and prediction digest; target, valid-mask,
calibration-activity, selected-support, raw/normalized T4, input-record, model
state, and forward-state digests; valid-window count; eval/no-dropout/no-grad
proof; repeated-forward equality on fixed sentinel chunks; state-before equals
state-after; target optimizer/backward/update counts zero.

For every system/surface/budget/regime, report equal-session mean and median,
fixed-seed session-bootstrap interval, worst-session R2, and roster order.

Required paired comparisons are:

```text
P1 - P0
P2 - P0
P1 - C1
P2 - C1
P0 - T0
P0 - SD
C1 - T0
```

Each comparison reports every paired session delta, equal-session mean,
median, positive count, bootstrap interval, and worst paired delta.

## 6. Predeclared interpretation gates

P0 is valid only if its M30 honest-total delta versus T0 is at least `-0.01`
on both within and external and its inference/state/input parity gates pass.
If P0 is invalid, P1/P2 performance is uninterpretable.

P1 primary gate:

```text
external honest-total M4 delta versus P0 >= +0.03
positive external sessions >= 10/15
external honest-total M30 delta versus P0 >= -0.01
within honest-total M30 delta versus P0 >= -0.01
worst external M4 paired delta versus P0 >= -0.05
```

P2 uses the identical gate at M10. The incremental pairing claim additionally
requires trained-budget external delta versus C1 at least `+0.01`, at least
9/15 positive sessions, and external M30 delta versus C1 at least `+0.01`.

Activity-isolation results explain mechanism but cannot rescue a failed honest
total-calibration gate.

## 7. Lifecycle and target discipline

The live score route has separate target-free authority and result roots. It
publishes an immutable attempt before target materialization, checkpoint
deserialization, CUDA initialization, or forward. Producer and comparator
artifact bytes may be descriptor-rehashed before attempt, but are not
deserialized.

After attempt, every target session is materialized once into one fixed input
authority. All six systems and both regimes consume that authority. External
records cannot select checkpoint, arm, budget, threshold, retry, or branch
weight. No formal/organizer-held data is opened.

Score and terminal publish transactionally only after every predeclared row is
complete. Any exception publishes an honest failure and no partial canonical
score. A failed attempt is not retried under the same root.

## 8. Implementation and review boundary

The additive scorer should compose the reviewed CAL-AUG deployment
materializer/metric/model-swap primitives through a typed multi-system profile.
It must not copy the parser, mutate historical scorers, monkeypatch model
modules, or use private target labels outside the governed metric.

Focused no-data tests must cover exact row order/cardinality, same-input
cross-system equality, producer/comparator drift, system/SWA swaps, both
regimes, P0/T0/C1 anchor reproduction, metric parity, state immutability,
attempt-before-materialize ordering, atomic lifecycle, and every decision
boundary.

Execution remains **NO-GO** until P0/P1/P2 full producer literals exist, the
successor implementation and closure pass independent root review, target-free
authority is minted, score roots are fresh, and a separate scoring launch is
authorized.

