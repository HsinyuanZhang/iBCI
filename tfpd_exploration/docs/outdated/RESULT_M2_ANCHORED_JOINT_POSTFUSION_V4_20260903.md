# M2 AJPF V4 held-out result: method-positive, practical robustness gate failed

Date: 2026-09-03 HKT  
Execution status: `TERMINAL`  
Cross-device bridge: `PASS`  
Method gate (`J-R1/UNCAPPED - J-NATIVE/UNCAPPED`): `PASS`  
Practical robustness gate (`J-R1/UNCAPPED - sealed POOLED`): `FAIL_WORST_SESSION`  
Overall preregistered `paper_success`: **false**

## Executive result

AJPF produced the first preregistered, held-out post-fusion improvement whose
session-bootstrap confidence interval is entirely positive.  On the six
external sessions, `J-R1/UNCAPPED` improved over the matched jointly trained
`J-NATIVE/UNCAPPED` control by:

```text
equal-session mean = +0.01311785484838159 R2
positive sessions  = 4/6
worst session      = -0.009836212677549971
bootstrap 95% CI   = [+0.0007253619527784657, +0.02506999327285399]
```

All three frozen method-gate requirements passed: mean at least `+0.010`, at
least `4/6` positive sessions, and worst session at least `-0.015`.

Relative to the sealed historical strong `POOLED` comparator, the external
mean was also positive (`+0.0319248704828766`, `5/6` positive), but one session
lost `-0.07827959317406641`.  This violates the preregistered worst-session
floor and therefore prevents an overall paper-success declaration.

The correct interpretation is:

- the anchored jointly trained post-fusion residual contains a real,
  transferable improvement over its matched native control;
- it exceeds the strong historical POOLED comparator on average and in five of
  six external sessions;
- it is not yet a robust replacement for POOLED because of one large external
  regression.

## Cross-device validation

V4 corrected only the V3 validation instrument; it did not retrain, change a
checkpoint, alter a score row, change a comparator, or relax either scientific
gate.  The historical POOLED authority was generated on an RTX 3090 CUDA path,
whereas the AJPF scorer is CPU-only.  V4 therefore required exact same-input
authority plus a preregistered numerical R2 bridge.

Observed bridge:

| Quantity | Result |
|---|---:|
| Record keys | `13/13` |
| Target SHA equality | `13/13` |
| Governed-start SHA equality | `13/13` |
| Window-count equality | `13/13` |
| Prediction SHA equality | `0/13` (reported, not required across devices) |
| Maximum absolute R2 difference | `1.3152579714237334e-7` |
| Frozen tolerance | `2e-7` |
| Bridge decision | `PASS` |

This establishes that the earlier V3 sentinel failure was a cross-device
bitwise artifact rather than a changed session, target, window law, or
scientific result.

## Aggregate R2

| Surface | Arm / law | Equal-session mean R2 |
|---|---|---:|
| external | `J-R1 / UNCAPPED` | `0.33098221581491233` |
| external | `J-NATIVE / UNCAPPED` | `0.31786436096653076` |
| external | `J-MEAN / UNCAPPED` | `0.2856350466055448` |
| external | sealed historical `POOLED` | `0.29905734533203573` |
| within | `J-R1 / UNCAPPED` | `0.6788234295639427` |
| within | `J-NATIVE / UNCAPPED` | `0.6668280607778055` |
| within | `J-MEAN / UNCAPPED` | `0.6525876532693122` |
| within | sealed historical `POOLED` | `0.6540862067123891` |

`FIXED30` and `UNCAPPED` were identical on the external surface.  On the
within surface, `UNCAPPED` was slightly below `FIXED30` for every trained arm:
`J-R1 -0.0016852871371480`, `J-NATIVE -0.0018740599850815`, and
`J-MEAN -0.0020990665506502`.  Thus the positive AJPF result should be
attributed to the jointly trained anchored residual, not to unbounded memory
growth.

## Preregistered contrasts

### External: method contrast

`J-R1/UNCAPPED - J-NATIVE/UNCAPPED`

| Session | Delta R2 |
|---|---:|
| `ses-2020-10-30-Run1` | `+0.016790936313901605` |
| `ses-2020-10-30-Run2` | `+0.026906086112969074` |
| `ses-2020-11-18-Run1` | `-0.008059719742674276` |
| `ses-2020-11-19-Run1` | `-0.009836212677549971` |
| `ses-2020-11-24-Run1` | `+0.026910812013641094` |
| `ses-2020-11-24-Run2` | `+0.025995227070002014` |

Decision: **PASS**.

### External: practical contrast

`J-R1/UNCAPPED - sealed historical POOLED`

| Session | Delta R2 |
|---|---:|
| `ses-2020-10-30-Run1` | `+0.1036940853196402` |
| `ses-2020-10-30-Run2` | `+0.019058823607447817` |
| `ses-2020-11-18-Run1` | `+0.04270330134610001` |
| `ses-2020-11-19-Run1` | `+0.004649002805006464` |
| `ses-2020-11-24-Run1` | `-0.07827959317406641` |
| `ses-2020-11-24-Run2` | `+0.09972360299313154` |

Summary: mean `+0.0319248704828766`, median `+0.030881062476773913`,
`5/6` positive, bootstrap 95% CI
`[-0.018190588920537376, +0.07743783654209711]`.

Decision: **FAIL** because the worst session is below `-0.015`.  The failing
session is scientifically informative rather than a general AJPF collapse:
on `ses-2020-11-24-Run1`, AJPF improves its matched native control from
`0.28952664881610135` to `0.31643746082974245` (`+0.02691`), while historical
POOLED is exceptionally strong at `0.39471705400380885`.

### Within confirmation

The matched method contrast is `+0.011995368786137146`, positive in `7/7`
sessions, with bootstrap 95% CI
`[+0.006918204110951295, +0.017376516148503756]`.  This independently supports
the method effect but does not replace the external practical safety gate.

## Mechanistic interpretation

The result narrows the earlier post-fusion conclusion.  Naive post-fusion mean
pooling remains harmful: external `J-MEAN` is `-0.03222931436098596` below
`J-NATIVE`.  In contrast, a residual post-fusion branch trained jointly while
anchored to the native early-pooling solution gives `+0.01312` external and
`+0.01200` within.  Therefore:

> Late-pooling information is useful when learned as an anchored residual to a
> strong early-pooling solution; replacing the early-pooling solution with a
> late mean is still an out-of-distribution operator change.

This is a stronger and cleaner story than claiming that post-fusion pooling is
universally bad.  The remaining problem is session robustness against an
occasionally exceptional POOLED solution, not absence of transferable
post-fusion signal.

## Immutable result graph

Reviewed design SHA:
`ae7d0a8c3e13c9f6ed921bdad6b5e8309eced1be7e2f7a689762d7eb8313e5e5`

Reviewed workorder SHA:
`6207b374aed58448af1a0b3a4c0db7135b9815be3d3e721b385ce252c63ce47d`

Reviewed execution closure (94 leaves):
`d7cb7b5425be2a0236239edb64d0caf431627d5830766b3c5fe8a649a21cac6c`

| Body | SHA-256 |
|---|---|
| `attempt.json` | `8eefb44e1937febcf267936d898546cd851bef4fa1574b0f2e0b8aa54f877f0f` |
| `predecessor_authority.json` | `89b8b22ab3cf8aade6f73a3e0b0c5c66b6f6d19c14296ab2d73b7f383edf5083` |
| `cross_device_bridge.json` | `1784c8935d7fb9a9bfbb7e1260f1f75cb54f6720d95123f0f3c497a386258931` |
| `input_authority.json` | `984c211b78464dd7f95e8faafc2c43a04626ee20b451bf377de361bc38216178` |
| `score.json` | `3c9efcfb1d0a4c8490073c2a7016d4b6b6832fffdb559a736fe4936f4de4712d` |
| `gates.json` | `2d494eb615f014cc352228da3e554d54d4d04ee838fee01dd7c9c6009ab4a8c5` |
| `terminal.json` | `890409f63cf1e66cfd83d586cb984c3cc429e5e0aefd007c08c3a1b1e4023295` |

No target label selected a checkpoint, model, memory law, threshold, or
comparator, and no target/model update occurred during scoring.
