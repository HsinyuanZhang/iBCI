# Budget-Matched Posterior CAL-AUG Source Audit V2

Date: 2026-08-30

## Purpose

V1 failed honestly before publishing a source authority because physical
support position 27 of `sub-C_ses-CO-20150313` has no finite `target_dir`.
M4 and M10 are unaffected; M30 contains 30 activity trials but only 29 usable
direction labels in that session.

V2 does not guess or reconstruct the missing angle and does not overwrite or
retry the V1 root.  It binds the exact immutable V1 attempt/failure graph and
uses the production ordinary-T4 missing-label convention in an explicit
trial-level posterior law.

## Exact law

For each budget M:

1. The B3S activity prefix is always physical chronological positions `[0,M)`.
2. The carrier observation pool is the ordered subset of those same positions
   whose `target_dir` is finite.
3. No future position, reordered position, target-id reconstruction, direction
   imputation, or borrowed M30 variance is allowed.
4. The receipt binds the physical-prefix IDs, finite-label mask, usable-label
   IDs, rate bytes, theta bytes, usable count, design rank, and residual DOF.
5. The cell stops unless the usable design has rank 3 and positive residual
   DOF at every source session and every M4/M10/M30 budget.

The source prior is fitted from each session's labeled subset inside its first
30 physical trials.  The same source-only prior remains fixed at M30/M10/M4.
The posterior normalizer keeps equal budget weight and source roster/unit
order exactly as V1.

This is a source-authority repair only.  It does not train, initialize CUDA,
open within/external/formal data, or authorize a GPU smoke unless all gates
pass.

