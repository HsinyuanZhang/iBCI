# M2 P1 Contrast Optimization V1

## Question

Is the sealed P1 hold-vs-reach FiLM gain (+0.0134 on first-30 ridge vs
`ridge_static_m30`) carried by labeled mean-rate contrast, cheap enough to
ship, and strong enough on the official first-33 native-T4 policy to prepare
an EvalAI cached-identity payload?

This cell does not push EvalAI.

## Frozen inputs

- Sealed P1 weights: `tfpd_exploration/results/m2_hold_film_probe_v1/film_states.pt`
- Checkpoint `25d7bc72…`, same decoder freeze as the probe
- Official comparison: submission `578221` held-out 0.30324395, online
  cached `E[N,50]`, first-33 native T4, normalized latency 0.0429

## Stages

1. Score-time masks of sealed P1 on first-30 ridge (interpretability).
   `p1_full` must match 0.30862991970180154 within 1e-6.
2. Retrain mean-rate dims only (`delta`, `log-ratio`) at first-30 ridge.
3. Native first-33 T4: P0, sealed-P1 transfer, mean-rate retrain, shuffles.
4. Cost: FiLM is 1224 params offline. Online extra params/MAC = 0 if identities
   are cached like `578221`.
5. Export `decoder.pkl` only if an M33 candidate beats native M33 T4-only by
   mean > 0.005, ≥4/6 sessions, and shuffle mean ≤ 0. No EvalAI push.

## Non-goals

- No TTA, no `side_dim` expansion, no fake angles, no overwriting `578221` artifacts.
