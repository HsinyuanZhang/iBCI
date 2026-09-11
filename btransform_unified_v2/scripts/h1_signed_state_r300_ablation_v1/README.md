# H1 signed-state R300 information ablations

This sealed source-only family implements the two user-authorized information ablations of the signed-state14 H1 RIFT R300 baseline: `ACTIVITY_ONLY` and `NONE`. It does not restore H1 NNMF, fusion, alternate carrier, or exploratory queues.

`ACTIVITY_ONLY` preserves the C2 activity identity for every C2-CAL-1 source budget and every public held-out M3 tag, but passes literal zero `[176,4]` T to C2 materialization and records the rematerialized E0. `NONE` uses literal zero E0 `[176,700]` and literal zero T `[176,4]` throughout, and never calls C2 materialization. Both retain the original signed-state R300 neural-decoder recipe: 13 source sessions, raw R300 input with validity masks, RIFT D4 width 256 dense recency, P16 proj-add, batch/microbatch 32, 731 updates per epoch, 32 epochs, AdamW, one-epoch warmup, EMA 0.9995, and unit dropout 0.1.

`build_banks.py` seals a fresh 27-tag public bank binding. `train.py` is fresh-only (there is no resume CLI); `--smoke-steps` is solely a bounded identity/gradient-route audit. Formal training requires all 32 epochs and then scans all EMA checkpoints with the H1 C2 grouped-seven visible HO-M3 metric and its deterministic tie break. No script reads hidden official labels or submits externally.
