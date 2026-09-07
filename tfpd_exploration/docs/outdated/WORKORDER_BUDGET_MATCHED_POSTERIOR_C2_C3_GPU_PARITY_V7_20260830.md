# C2/C3 GPU parity V7

V7 is the final parity-only successor. It binds the exact V6 post-comparison
lifecycle failure and repeats only the first within-session C2/M4 comparison.

The numerical comparison is published immediately as immutable
`comparison.json` before final revalidation. Every later failure receipt must
link that comparison. Therefore a review-only or final-validator problem can
never erase completed numerical evidence or force the same comparison to run
again.

Final runtime validation checks the environment values, canonical source-root
identities, namespace origins, and GPU profile against the pre-attempt record.
It does not repeat the pre-attempt-only assertion that Torch is absent after
Torch has intentionally been imported.

Execution and review paths remain disjoint. Review-only drift is
`ACCEPTED_NON_NUMERIC_DRIFT`, has no effect on numerical acceptance, and never
requires restart. V7 does not run the full 252-cell matrix and does not select
a relaxed parity threshold.

