# Work Order: CDM-D Activity-Only Refinement V1

Date: 2026-08-26  
Status: additive low-cost matched-score specification  
Scientific role: post-positive short-budget performance refinement

## 1. Question

V8 showed useful M4/M10 gains but harmful M30 transfer and one severe external
outlier. Test whether the useful short-budget gain survives when completed
query trials update only B3S activity memory while the functional carrier is
frozen at its honest support-only initializer.

This is one bounded refinement, not a new ablation bundle and not a new
training run.

## 2. Accepted predecessor

Bind the immutable V8 result graph exactly:

```text
V8 implementation closure:
62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939

attempt:
557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376

input authority:
ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07

score:
98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e

terminal:
80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096

verdict:
ADVANCE_SHORT_BUDGET
```

Reject a partial graph, failure coexistence, topology drift, body/sidecar
drift, identity drift, input-authority drift, or current implementation-closure
drift before reserving a successor root.

## 3. Exact system

Keep the V8 sealed Cell-D model, parser, support selection, support-only fixed
ridge initializer, complementary groups, query chronology, input authority,
last-bin variance-weighted R2, equal-session aggregation, and target-free state
contract unchanged.

Change one thing only:

```text
valid completed query trial
    -> update activity FIFO
    -> never propose or commit a carrier update
```

The carrier tensor, sufficient statistics, group assignment, digest, and
completed carrier-update count must remain exactly unchanged for every query
trial. M4 activity capacity remains 26 and M10 remains 20. M30 is omitted:
with activity capacity zero and a frozen carrier it reduces to the sealed
support-only state and adds no new information.

No network, parameter, checkpoint, normalizer, support label, target label,
optimizer, backward pass, or training rule may change.

## 4. Four new cells

Run exactly:

```text
M10 within-6 activity-only
M10 external-15 activity-only
M4  within-6 activity-only
M4  external-15 activity-only
```

The fixed V8 sealed Cell-D and complete CDM-D rows are immutable comparators.
Do not rerun them. Descriptor-load their exact per-session evidence from the
accepted V8 score and verify the new materialized input records against the V8
input authority before every activity-only cell.

Every new cell must use eval mode, dropout disabled, no-grad inference, the
same session order, the same valid query windows, and the same metric code as
V8. All target backward, optimizer, update, and label-to-state counts remain
zero.

## 5. Decision rule

Report paired per-session activity-only minus sealed Cell-D and activity-only
minus complete CDM-D deltas.

The refinement advances only if:

- M10 external mean delta versus sealed is at least `+0.02` with at least
  `10/15` positive sessions;
- M4 external mean delta versus sealed is at least `+0.05` with at least
  `10/15` positive sessions;
- neither within mean is below sealed by more than `0.01`;
- no external session is worse than sealed by more than `0.20` at either
  budget; and
- the severe V8 outlier `sub-M_ses-CO-20140626` improves by at least `+0.30`
  relative to complete CDM-D at M10 and M4, or its delta versus sealed is no
  lower than `-0.20` at both budgets.

The outlier clause is a predeclared safety gate. It may reject an otherwise
positive mean; it may not rescue a failed aggregate gate.

If the activity-only route fails, stop. Do not tune FIFO length, thresholds,
groups, ridge strength, or the outlier rule.

## 6. Additive implementation boundary

Use a new additive package, dry CLI, focused tests, authority root, and score
root. Reuse the reviewed V8 evaluator through typed composition. Do not copy
the scorer, monkeypatch module globals, edit V8/V5/shared files, or mutate any
accepted result.

The no-data candidate must prove:

1. exact V8 predecessor loading and all four immutable SHA links;
2. exact four-cell order and no M30 cell;
3. activity transition on valid completed trials;
4. exact carrier nonmutation on every accepted or rejected trial;
5. M4/M10 FIFO capacity and next-trial-only causality;
6. exact V8 input-record revalidation;
7. unchanged model state and zero target optimization;
8. complete decision-rule and outlier adversaries;
9. transactional attempt/input/score/terminal or failure topology;
10. static dry CLI with no Torch, data, CUDA, write, or launch.

No source, target, NWB, checkpoint tensor, CUDA/GPU, authority, result root,
or launch is authorized by this work order. Root must independently audit the
frozen implementation and schedule it only when it does not delay the CS-WG M1
performance route or interfere with user processes.
