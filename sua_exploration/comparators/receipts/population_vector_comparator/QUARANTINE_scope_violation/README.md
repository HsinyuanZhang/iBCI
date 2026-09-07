# Quarantined: H1 PV audit receipt — scope violation

**Date:** 2026-08-12
**File:** `population_vector_comparator_h1_audit_receipt.json`

## Why

This receipt was produced by reading **14 `sub-HumanPitt-held-out-calib` NWB files**. The H1 scope
discipline used by every other artifact in this project is enforced by
`h1_sparse_event_endpoint.reject_path_scope`, which fails closed on any path containing `held-out`
and requires `sub-HumanPitt-held-in-calib`. The 13 public held-in-calib recordings are the only
legal H1 inputs for these screens.

`population_vector_comparator.py` lines 281 and 297 glob `*held-out-calib*.nwb` directly, bypassing
the sealed loader.

## Severity

**Not a label leak.** Only `nwb.trials` presence was inspected, and it was reported absent; no
neural or behavioural values were consumed. But the receipt is inconsistent with the H1 evidence
chain and must not be cited or bound by anything downstream.

## Status of the verdict

The verdict `PV_INAPPLICABLE_NO_NATIVE_DISCRETE_DIRECTION_FIELD` is likely correct on its merits —
H1 is a 7-DoF task with no discrete reach-direction target — but it was established on the wrong
files and must be **re-derived on the 13 held-in-calib sessions** before use.

## Required fix

Route H1 through `h1_sparse_event_endpoint.index_heldin_calib` instead of a raw glob, then re-run
the audit and write a fresh receipt.

## Note on M2

The M2 audit receipt is **not** quarantined. For FALCON M2 the `held-out-calib` split is the
released few-shot calibration data that the sealed M2 arms legitimately use, so reading it is
correct there. The blanket prohibition that produced this confusion was over-broad: the scope rule
is per dataset, not global.
