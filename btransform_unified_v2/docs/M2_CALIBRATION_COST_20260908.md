# M2 trained D42/e17 calibration cost — 2026-09-08

正式四-session validation 位于 `results/diagnostics_v1/m2_joint_calibration_cost_formal_validation_20260908.json`，SHA `f64fa1d49061a705054b120d3a66f4d60565c4445103c8434b3d2567c18b68f1`。四个 ext4 sessions 各有三个有效 rounds；12 个 `warm_static_bank_wall` 的范围为 `0.0018419970–0.0023905280 s`。

每个 receipt 绑定同一 trained M2 joint D42 seed42 EMA e17 checkpoint `39dc996425b028ed9ac6df22ebc284c394eb7d6904c63c2da6c1312bbb80a77f`、同一 source normalizer、run metadata、cost script 和 source/source-NWB SHA。所有 session 都验证 raw-source M33 native T 与 cached T byte-equal，trained encoder materialized E0 byte-equal，且每个 round 的 T/E0 SHA 与其 parity record 一致。

计时仅覆盖：已在内存的 raw M33/support objects → native MOVE-T4 carrier/normalization → trained D42/e17 `native_e0_and_u` materializer → static `TaskBank`。它不包括 raw file load、serialization 或 cache installation。`native_e0_and_u` 会一同 materialize E0 和 per-trial `u`，该范围不是 E0-only 最优化时间。
