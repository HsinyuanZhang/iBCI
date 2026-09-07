# Decision: Hold the APFG Static Official Probe Push

Date: 2026-09-03
Status: decision recorded; cell closed without submission; image held

## Decision

The user elected **not to spend an EvalAI submission** on the completed
static-pool APFG candidate (`spint-t4-m2:apfg-static-s42-b3d19967`,
payload `b3d19967b258fe94584ad4f84a7d394f8c2c83f85bdc3e198b59fc8178b05af5`),
after the local validation receipt made the official outcome highly
predictable. Submission quota is limited (two remaining at decision time);
the user allocated zero of it to this probe.

## Evidence basis

The local evaluator-semantics replay (`results/evalai_m2_apfg_static_v1`
terminal binding of `artifacts/local_validation_receipt.json`) measured, on
the official static 30-trial pool:

- native arm vs the officially scored `act30_dopt4` rows: bitwise identical
  externally (max delta 0.0; within 1e-07);
- APFG-ZERO vs NATIVE: exactly equal on all compared fields, 13/13 sessions;
- **APFG-LEARNED minus NATIVE on the static pool: external mean -0.0007693
  (3/6 positive), within +0.0004449 (4/7)**.

The gate effect that exists in growing pools (V2: +0.004612 external,
4/6) vanishes when the pool is frozen at 30 trials. The official test
surface is the same six external sessions, so the expected official score is
approximately `act30_dopt4` (HO 0.2897), possibly marginally lower.

## Consequences

- The cell is terminal-complete with `submission_performed=false`. The image
  and payload remain valid and may be submitted later by a successor note if
  an official confirmation point is ever wanted; no re-export is needed.
- The paper-safe statement stands unchanged: Post-Fusion gating is a bounded
  mechanistic signal, not a promoted contribution; the deployable positive
  result on continual M2 remains uncapped **causal activity memory**.
- The remaining EvalAI quota stays available for higher-information
  candidates.
