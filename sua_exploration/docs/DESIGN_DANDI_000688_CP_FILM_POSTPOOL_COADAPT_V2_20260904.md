# DANDI 000688 CP-FiLM + Post-Pool Co-Adaptation V2

Status: `FROZEN_SOURCE_ONLY_REPAIR__GPU0_AUTHORIZED__FORMAL_TEST_FORBIDDEN`

V2 is the additive implementation repair of the V1 design. It binds
`INCIDENT_DANDI_000688_CP_FILM_POSTPOOL_V1_FROZEN_POSTPOOL_20260904.md` and
does not reinterpret or overwrite the V1 result root.

The scientific question and all data/model constants remain unchanged:

- exact 27 train / 6 validation source split; formal-test names only;
- seed-matched M30/T4@30 B3S parent;
- first-30 activity and T4, Q50 query windows;
- M10/M30 low/high-speed robust-z calibration profiles;
- four paired arms `CP10`, `CP30`, `SHUFFLE10`, `EMPTY`;
- frozen activity `pre_pool`, native branch, and decoder;
- per-arm zero-init 1,224-parameter FiLM head plus an exact copy of the
  11,826-parameter `post_pool` MLP;
- 12 epochs, Adam `3e-4`, batch 32, last-bin MSE;
- governing coordinate-wise epoch-9--12 parameter average;
- epoch-12 state as a non-governing diagnostic;
- seed42 first; seeds43/44 only if profile utility passes;
- no target update, formal-test access, continual memory, or EvalAI action.

V2 adds executable invariants that V1 lacked:

1. after copying, explicitly set every `post_pool` parameter trainable;
2. count only `requires_grad=True` parameters and require exactly 13,050;
3. require trainable names to include all six post-pool weight/bias tensors and
   all four FiLM tensors, identically across arms;
4. require each arm's post-pool digest to equal native before training and to
   differ from native after training;
5. require the frozen native student digest to remain unchanged.

The profile-utility law is unchanged: averaged CP must improve native, and its
paired mean over EMPTY must be at least `+0.002` with at least 4/6 positive
sessions. Solid additionally requires at least `+0.010` over native.

