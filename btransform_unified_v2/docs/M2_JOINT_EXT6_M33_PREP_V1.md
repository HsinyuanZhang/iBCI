# M2 joint ext6 raw-M33 preparation

`scripts/rift_v1/prepare_m2_joint_ext6_m33.py` creates a new, immutable six-session directory for the joint B/D picker. It uses the same public held-out-calib record builder as the sealed ext6 query cache and reads no hidden or EvalAI records.

For every session it writes the first 33 calibration trials as `calib_activity.npy` with shape `[33, 100, 96]`, reconstructs native normalized MOVE-T4 as `T.npy`, and materializes frozen native E0 plus per-trial `u` in `e0_u.pt`. The manifest contains raw-record SHA-256 values, source and normalizer hashes, canonical `0..32` trial IDs, array hashes, and byte-equality checks against query-cache E0/T. The four ext4 sessions must also byte-match their training `calib_activity.npy` and `T.npy`.

The program refuses an existing destination and does not modify either existing cache. It constructs no decoder scores. The coordinator runs it with `PYTHONNOUSERSITE=1`; a CPU-only native encoder materialization should take a few minutes and produces roughly 8 MB of raw activity plus small identity artifacts.
