# M2 Mean-Only FiLM Squeeze V1

## Question

Along freeze-decoder native first-33 T4 + hold/reach FiLM, can a cheap
retrain catalog beat local public held-out **0.31905** enough to justify
one EvalAI cached-identity submit against official `578221` (0.30324)?

## Operator policy

EvalAI-max. Shuffle is disclosure, not a reject. Pick the highest 6-session
equal-session mean among arms that beat P0 M33 (0.29913) by **>0.005** and
**≥4/6**. Tie-break: median, then simpler mask, then default FiLM input.

## Frozen inputs

- Checkpoint `25d7bc72…`, decoder + pre_pool + post_pool frozen, `post_pool` 64+4
- Native first-33 T4; never replace T4; no fake φ
- Train FiLM on 7 held-in `within_post30` only
- Online path: cached `E[N,50]` like `578221` (extra params/MAC = 0)
- Do not overwrite `578221` artifacts or sealed probe/opt v1 roots

## Catalog (≤12)

Replica, longer/higher-lr means, native-M33 4-D retrain, contrast-only FiLM,
single-mean ablations, means+reach-std, one extra seed. See
`tfpd_exploration/src/m2_means_squeeze_v1/plan.py`.

## Non-goals

- No TTA, no joint encoder+decoder, no `side_dim` expansion
- Warm-start cell writes `m2_means_squeeze_v2`. Fresh-init `m2_means_squeeze_v1` already selected sealed 0.319.
