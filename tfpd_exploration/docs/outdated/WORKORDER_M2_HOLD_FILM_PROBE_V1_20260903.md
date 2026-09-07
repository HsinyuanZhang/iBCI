# M2 Hold-vs-Reach FiLM Probe V1

## Question

Does a calib-only labeled hold-vs-reach contrast move the frozen
`25d7bc72…` B3S+T4 identity when the decoder, `pre_pool`, and `post_pool`
stay frozen?  If it does not, the contrast does not speak this slot.
If it does, and shuffle/C2 fail, a later joint encoder+decoder retrain
is warranted.  This cell is not an official champion and does not run TTA.

## Frozen inputs

- Checkpoint SHA-256: `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`
- T4 normalizer SHA-256: `d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e`
- Activity: chronological first-30 neural (same as `ridge_static_m30`)
- T4: ridge λ=0.1 on finite-angle trials inside that first-30.  T4 is never replaced.
- Contrast: per-channel robust-z of (reach−hold mean rate), log-ratio, hold std, reach std.  Hold = non-finite `tgt_loc` angle.  No fake φ, no `[1,0,0]` intercept stuffing.
- FiLM: shared Linear(8,8)+ReLU+Linear(8,128), zero-init, applied to pooled `h` only.  `post_pool` remains 64+4.

## Arms

1. `p0_zero_film_t4`: zero FiLM.  Must match sealed external `ridge_static_m30` 0.29521985196829853 within 1e-6.
2. `p1_trained_film`: train FiLM on 7 held-in `within_post30` windows.  Score 6 held-out official query.
3. `p1_shuffle_score`: P1 weights, contrast rows permuted across channels.
4. `c2_train_on_shuffle`: train and score on the same permutation.

## Gate (external official query only)

- Shuffle or C2 mean delta > 0 → `KILL_*_CAPACITY`.  Stop.  No joint retrain from this cell.
- P1 ≤ P0 → `KILL_NO_SIGNAL`.  These labels do not speak the frozen slot.  No joint retrain from this cell.
- P1−P0 mean > 0.005 and ≥4/6 sessions positive, shuffle and C2 ≤ 0 → `CONTINUE_LABELED_SIGNAL`.  Joint retrain is then the follow-up, because the 0.295 champion was jointly trained.
- Otherwise `WEAK_LABELED_SIGNAL`.  Do not auto-launch joint retrain.

Within-post30 is disclosed and not a gate: P1 is trained there.

## Explicit non-goals

- No `side_dim` 8/16 expansion of the T4 concat.
- No TTA.
- No EvalAI push.
- No mutating B3SCF or sealed AJPF roots.
