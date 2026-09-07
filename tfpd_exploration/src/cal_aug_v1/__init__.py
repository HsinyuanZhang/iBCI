"""CAL-AUG V1: matched T0/C1 B3S-calibration-prefix robustness training.

Work order: ``docs/WORKORDER_CAL_AUG_V1_20260829.md`` (binding).
Guidance: ``docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md``
sections 2.1, 6, 11, 13.

Single axis: the B3S activity identity sees a deterministic chronological
prefix-length cycle ``M_step = (30, 10, 4)[global_training_forward % 3]`` in arm
C1, while query activity, T4, decoder, loss, optimizer, batch order and the
dynamic-dropout p stream stay byte-identical to matched T0 (which runs through
the SAME successor runner with the operator disabled).

This package is new file surface only: no sealed/frozen file is edited.  The
sealed trainers are importlib-loaded after SHA-256 verification of their pinned
literals (``plan.PINNED_SHA256``), fail-closed BEFORE any data or model access.
"""
