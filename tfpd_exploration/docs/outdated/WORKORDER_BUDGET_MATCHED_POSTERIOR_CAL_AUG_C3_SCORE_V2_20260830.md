# Corrected C2/C3 Posterior-Input Score V2

V1 failed before model construction because `src` correctly referred to the
TFPD package while one streaming model-builder import incorrectly expected
`src.models`.  Preserve the exact immutable V1 attempt/failure graph.  Do not
delete it, overwrite it, or call it a numerical result.

V2 imports the streaming builder from its independent `models` namespace and
otherwise keeps the V1 target records, posterior carrier, model checkpoints,
comparators, numerical rules, and predeclared decision thresholds unchanged.
Use a fresh V2 result root.  This is a scoring-only successor: C2 and C3
training must not be restarted.

Execution code, model code, checkpoint/data identities, and numerical
dependencies remain strict.  This work order and its focused test are a
separate review closure.  Their drift is recorded as
`ACCEPTED_NON_NUMERIC_DRIFT` and never invalidates completed numerical output.

