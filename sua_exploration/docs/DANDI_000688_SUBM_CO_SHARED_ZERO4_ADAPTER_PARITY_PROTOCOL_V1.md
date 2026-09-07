# Consumed sub-C shared-zero4 adapter input parity V1

Status: `APPEND_ONLY_CPU_INPUT_PARITY_PACKAGE_NO_EXTERNAL_SCORING_CAPABILITY`

## Purpose and boundary

This package closes one prerequisite for a possible future three-arm external
endpoint: the shared-zero4 adapter input semantics.  It is not an external
score runner and creates no capability to access or score formal sub-M data.

The only data fixture is the already-consumed development session
`sub-C_ses-CO-20151103`.  Its path, size, and SHA-256 are fixed in source and in
the prelaunch receipt.  The runner accepts no NWB path, checkpoint, model,
normalizer, authorization, signature, or external-data-root argument.  It is
CPU-only and performs no model forward or R2 computation.

## Frozen parity matrix

- views: `sua`, `pseudo_mua`;
- activity identity: `first_n30`;
- comparator T4 pool and common query boundary: first 50 rewarded trials;
- label controls: original chronology, deterministic target-direction shuffle,
  and target-direction removal;
- chronology: the existing V5 owner chronology, with only the already-reviewed
  exact-integral `start`/`stop` bridge;
- descriptor: direct `float32 [N,4]` positive-zero bits in the standardized T4
  coordinate, constructed only from the signal-view channel count.

## Required proofs

For both views the execution receipt must show:

1. side input is exactly `float32 [N,4]`, including positive-zero bit patterns;
2. descriptor target-direction reads, T4 trial-rate reads, T4 fit calls, and T4
   normalizer value reads/arithmetic are all zero;
3. activity calibration is rebuilt from exactly indices `[0,30)` while the T4
   comparator pool and evaluation query boundary remain 50;
4. owner `valid_starts` equals an independent reconstruction from rewarded
   trials `[50,end)`;
5. SUA and pseudo-MUA side rows equal their respective signal-view channel
   counts, not the pseudo-MUA source-unit count;
6. label shuffle and label removal change label evidence but leave the complete
   would-be frozen-model input record exact: neural, behavior, activity
   calibration, valid starts, and side features;
7. checkpoint opens, model forwards, R2 computations, GPU use, external sub-M
   access, and external sub-M scoring are all zero.

The label controls are input-parity controls.  They do not produce predictions
or a scientific endpoint result.

## Append-only handoff

A future external three-arm scorer may cite the sealed V1 receipt as zero4
adapter evidence.  It must still be implemented as a new, independently
reviewed, authorization-first package.  This V1 package does not sign, grant,
or imply that later capability and must never be modified into such a runner.

