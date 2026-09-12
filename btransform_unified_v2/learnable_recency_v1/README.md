# learnable_recency_v1

Self-contained learnable recency-bias family (CABLE / FoX essence) for the frozen RIFT temporal stack. Does not edit `btransform_unified_v2/src` or existing runners.

Defaults: `per_layer=True`, `--layers 4`, flat heads fixed, task-scaled half-life ladder anchored on H1 (`W0/75`). Tiers: `learned_slope` (24 new scalars), `fixed` (0 new params, same ladder), `fox_gate` (8224), `cable` (25216). Score-time for the per-layer default uses `CpuLearnableRecencyRuntime` / the learnable module; a single-vector export into unchanged `CpuRiftTemporalRuntime` only succeeds when all layers agree.

```text
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest \
  btransform_unified_v2/learnable_recency_v1/tests -q
```

Formal launch matrix, ladders (seconds and bins), `--layers`, and the wave-1 M2 pair: `docs/DESIGN.md`.

M1 / M2 / H1 的严格 source-only static 全训路线使用共享可学习通道表，不读取目标 support。EvalAI 选点与 FULL 相同：完整目标面 epoch 扫描 + earliest-max；末轮 EMA 只作 sidecar。模型、数据边界、配方及 smoke 说明见 [STATIC_SOURCE_ONLY.md](docs/STATIC_SOURCE_ONLY.md)。

当前 FULL learned-slope 的 M1 / M2 / H1 flat 对照使用相同 P16、seed 42 和训练 / 选轮配方，仅将全部 recency slopes 固定为零。准备入口只生成配对清单与未来命令，不启动训练，见 [FULL_FLAT_CONTROL.md](docs/FULL_FLAT_CONTROL.md)。

ACT-only v2 保留现有 B3/B3S/C2 encoder 的 early-pool activity 主干，参考 SPINT 的校准协议，与当前 P16/D4 learned-recency decoder 从头联合训练。M1 / M2 / H1 均提供训练和评分入口；本地评估沿用 FULL，最终结果以 EvalAI 为准。机制与数据协议见 [ACTIVITY_ONLY_V2.md](docs/ACTIVITY_ONLY_V2.md)，CPU smoke 与 M2 调度交接见 [ACTIVITY_ONLY_V2_HANDOFF.md](docs/ACTIVITY_ONLY_V2_HANDOFF.md)。
