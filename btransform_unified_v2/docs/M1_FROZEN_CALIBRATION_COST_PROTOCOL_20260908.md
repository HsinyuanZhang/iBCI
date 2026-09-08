# M1 frozen e3 calibration cost protocol — 2026-09-08

该审计测量当前 M1 `M1-RIFT-R100-D4-P16-RECENCY-V1` 的静态校准 bank 重建成本，不运行 decoder、query score 或 target backward。绑定对象为 EMA epoch 3；score receipt SHA 为 `a10eb2fc56256a6c904e179f96747a79f4ed3a1c8eed9f3dc76d55d5bc6d6eca`，checkpoint SHA 为 `5432a95c8e1b8a3297e696f0f36dd0477a85f3858aadb31136751df6e7b8fd1e`。

校准只使用每个 held-out-calib recording 的 M10 support trials `0..9`。carrier 为 raw support 经 rSyn3 `SourceBasis H` 和 source RMS normalizer；`E0` 为同一 M10 neural activity 经冻结 B3 Sfix e11 provider。该 provider 实际 checkpoint 是 `epoch_011.pt`，SHA `7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a`；rSyn3 H/RMS source NPZ SHA `27016199c39d630b4a0455ddeae36aa90e669bc4e913e765d2b00c03637298ab`。输入/构造不读取 query values。

每轮先分别冷加载 carrier raw M10 support、B3 raw M10 neural input、rSyn3 H/RMS objects 和 B3 Sfix e11 provider。随后才启动唯一可报告的 `warm_static_bank_wall` timer：它从以上对象均已在内存时开始，覆盖 carrier solve、E0 embedding build 和 `TaskBank` static materialization，并在 `TaskBank` 构造完成后立即停止。SHA、serialization、JSON 和所有 cold loads 均在该 boundary 外。cold components 不得相加并称为端到端 wall；serialization 单列。

早期 [m1_frozen_calibration_cost_preflight_20260908.json](/home/xinyuan/Work_host/SPINT/btransform_unified_v2/results/diagnostics_v1/m1_frozen_calibration_cost_preflight_20260908.json) 的 production parity 仍有效，但其 timing scope 无效，见 [TIMING_SCOPE_INVALID.json](/home/xinyuan/Work_host/SPINT/btransform_unified_v2/results/diagnostics_v1/m1_frozen_calibration_cost_preflight_20260908_TIMING_SCOPE_INVALID.json)。不能引用其中的时间数字。

替代预检 [m1_frozen_calibration_cost_preflight_v2_20260908.json](/home/xinyuan/Work_host/SPINT/btransform_unified_v2/results/diagnostics_v1/m1_frozen_calibration_cost_preflight_v2_20260908.json) 在 CPU `12,13`、`nice 10`、float32、两线程运行。它针对三个 sessions 验证 reconstructed carrier、E0 和 unit mask 与 `m1_train._ho_material()` production banks 均 byte-equal；实测 `20121004` 一轮 warm static-bank wall 为 `0.0164164 s`。此单 session 预检不是正式三 session 成本数字。

正式审计仅由协调者启动，且必须避开其他已分配作业：

```bash
cd /home/xinyuan/Work_host/SPINT/btransform_unified_v2
nice -n 10 taskset -c 12,13 /home/xinyuan/miniconda3/envs/spint/bin/python \
  scripts/diagnostics_v1/m1_frozen_calibration_cost.py \
  --sessions 20121004 20121017 20121024 --rounds 3 --cpus 12,13 \
  --output results/diagnostics_v1/m1_frozen_calibration_cost_formal_v2_20260908.json
```

该命令拒绝覆盖既有 receipt，限制 affinity 为 `12,13`，并记录每 session 三个实际 rounds、cold setup、真实 warm static-bank wall、组件计时、serialization、RSS、输入与源码 SHA、target BP `0` 和 selected e3 EMA 引用。它不使用 CPU 8–11、14–15，也不写入 live/train/core 或现有 receipt。

## Formal result (coordinator run)

正式 receipt 为 `results/diagnostics_v1/m1_frozen_calibration_cost_formal_v2_20260908.json`，SHA `c2f9981444b45bd719b66cc27a72a88384ed9a8bd7cd7bcd7b5469607012acbf`。它是唯一可引用的正式数字：三个 sessions、每个三个 rounds 的 `warm_static_bank_wall` 为 `0.0497381–0.0522130 s`。每个 round 的 timer 从已在内存的 M10 raw inputs、H/RMS objects 和 B3 provider 起，到静态 `TaskBank` 完成即止；hash、serialization、JSON 和 cold loads 均在 boundary 外。全部三个 session 的 carrier、E0 和 unit mask 均与 production bank byte-equal。

正式启动明确使用 CPU `12,13`、`nice 10`，且 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2`。`dtype=float32` 描述 bank/interface arrays；production NNLS/ridge 内部数值 dtype 由其原生实现决定，不能据此声称整个 carrier solve 都是 float32。早期单 session v2 preflight 的 `0.0164164 s` 只作预检，不替代正式范围。
