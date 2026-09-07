# Design Addendum: H1 Calibration-Profile FiLM V2

Date: 2026-09-04  
Status: `FROZEN_NUMERICAL_ANCHOR_SUCCESSOR`

V2 is the additive successor to the V1 incident.  It preserves every scientific
choice in `DESIGN_H1_CALIBRATION_PROFILE_FILM_V1_20260904.md`: CP-FiLM,
profile definition and mask, early/late 2x2, source roster, support schedule,
12 epochs, optimizer, target surface, contrasts, and decision gates.

The sole change is the zero-anchor policy:

1. each newly computed EP-ZERO and LP-ZERO prediction must reproduce itself
   byte-for-byte in an immediate same-process repeated forward;
2. C1/LP-R3 model state, support activity, carrier, task profile, query starts,
   target, and window count remain exact authorities;
3. recomputed historical R2 must differ by at most `1e-7`;
4. equality to the historical prediction SHA is recorded per row but is not a
   gate.

This distinguishes a material model/input/metric change from harmless CUDA
floating reduction differences.  V2 binds the exact five-body V1 failure graph
and uses a fresh result root.  V1 is not retried or overwritten.

As an additional no-science-change guard, V2 requires the deterministic
fold-19250108 EP-FILM and LP-FILM state digests to equal the already sealed V1
states before accepting the successor.  The two pooling positions, the shared
FiLM module, and both incremental contrasts therefore remain the same 2x2
experiment; V2 is not a new model arm.
