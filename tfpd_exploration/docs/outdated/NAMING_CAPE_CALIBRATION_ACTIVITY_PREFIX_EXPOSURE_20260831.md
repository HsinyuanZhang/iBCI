# Naming Decision: CAPE

Date: 2026-08-31

Status: paper-facing naming decision only. This document is not an execution
authority and is outside every active PACD producer and scorer closure.

## Decision

The method previously called `CAL-AUG C1` is named:

> **CAPE: Calibration-Activity Prefix Exposure**

The historical arm identifier remains `C1`. Existing code, result roots,
receipts, manifests, hashes, work orders and immutable lineage must not be
renamed or rewritten.

At first use in a paper or new design document, write:

> Calibration-Activity Prefix Exposure (CAPE; historical arm C1)

Use `CAPE` thereafter when discussing the method. Use `C1` only when referring
to a literal historical arm, receipt field or artifact identity.

## Exact method denoted by CAPE

CAPE is the completed unpaired source-training intervention in which each
optimizer step receives exactly one B3S calibration-activity prefix from the
predeclared deterministic M30/M10/M4 cycle. The query activity, behavior
target, ordinary M30 T4, model, loss, optimizer, source roster and scoring law
remain matched to T0.

CAPE does not denote:

- paired full/short training;
- random calibration-trial sampling;
- a T4 update or short-prefix T4;
- target adaptation or target gradients;
- a learned gate, posterior feature or confidence estimator;
- activity or carrier memory;
- prediction-consistency training.

## Distinction from PACD

| Method | Views in one optimizer step | Full-view anchor |
|---|---|---|
| CAPE / historical C1 | one of M30, M10 or M4 | no |
| PACD-P1 | paired M30 and M4 views of the same query | yes, every step |
| PACD-P2 | paired M30 and M10 views of the same query | yes, every step |

The intended method progression is:

```text
CAPE
    unpaired exposure to short calibration-activity prefixes
    low-budget signal, but possible M30 damage

PACD
    paired short-prefix exposure with a permanent M30 anchor
    tests whether low-budget robustness can be retained safely
```

## Existing result associated with CAPE

Relative to its matched T0 control, the completed CAPE/C1 result reported:

| External surface | CAPE minus T0 |
|---|---:|
| M4 | `+0.026478`, `10/15` positive sessions |
| M10 | `+0.034960`, `10/15` positive sessions |
| M30 | `-0.022008`, `6/15` positive sessions |

These numbers motivate PACD but do not prove PACD. CAPE establishes that
short-prefix exposure can carry a low-budget signal; its M30 loss motivates
the paired full-view anchor.

## Recommended paper wording

> CAPE exposes the source decoder to deterministic calibration-activity
> prefixes one view at a time. It improved M4 and M10 transfer but degraded
> M30, motivating PACD, which pairs every short view with the same query's
> full M30 anchor.

Do not describe CAPE as paired calibration dropout or use CAPE/PACD
interchangeably.

## Provenance rule

This naming note may be cited by future review documents, but it must not be
retroactively inserted into an already running producer closure. A future
successor may bind this note explicitly only through a newly reviewed closure.

