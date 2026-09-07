# Workorder: DANDI 000688 CP-FiLM + Post-Pool Co-Adaptation V1

Status: `EXECUTION_AUTHORITY__SOURCE_ONLY__GPU0`

Governing design:
`sua_exploration/docs/DESIGN_DANDI_000688_CP_FILM_POSTPOOL_COADAPT_V1_20260904.md`

## Sequence

1. Verify the frozen design, this workorder, strict manifest, teacher, and
   seed-matched M30/T4@30 checkpoint; require a fresh result root.
2. Publish attempt before source data, model, Torch CUDA, or GPU access.
3. Prepare one 27/6 source DataModule; formal-test sessions remain names only.
4. Materialize the existing M10/M30 profile and Q50 authority without changing
   its definition.
5. Strict-load the seed parent. Freeze pre-pool and decoder. Construct four
   exact post-pool copies and four identical zero-init FiLM heads.
6. Prove bitwise native parity, then train all four arms together for exactly
   12 epochs.
7. Form the governing epoch-9--12 parameter average before validation scoring.
8. Score averaged and epoch-12 states on all six Q50 validation sessions.
9. Publish paired native/control contrasts and the frozen seed-expansion
   decision; preserve immutable failure prefixes.
10. Run seeds 43/44 only if seed42 meets the profile-utility predicate.

No formal-test access, target update, continual memory, or EvalAI action is
authorized.
