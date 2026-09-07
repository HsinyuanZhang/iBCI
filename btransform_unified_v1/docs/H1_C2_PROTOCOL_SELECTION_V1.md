# H1 selection = C2 HO-M3（SPINT 原始方法）

Legal epoch-pick for H1 in this series is **C2 HO-M3**, not all-13 minival and not exam 01-20.

## Protocol

1. Train the 13 held-in sessions only. No interleaved validation, no early stop, no pick during training.
2. Prefix cycle per optimizer step: **M7 / M5 / M4 / M3** (same hash as C2 `prefix_schedule`).
3. After all epochs, score every checkpoint on the **14 public held-out-calib recordings** (S6–S12, earliest-M3 identity).
4. Pick with C2 `select_epoch`: higher `val_ho_m3_grouped/r2_mean` → higher worst-session → lower population std → earlier epoch.

HO-M3 is a visible development/model-selection surface. It is not hidden-test query labels.

Current cell: L=200 P16, 32 epochs, GPU1. Dest under `results/h1_c2protocol_l200_p16/`.
