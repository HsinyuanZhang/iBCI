# 当前 FULL learned-recency 的 flat 对照

本对照覆盖 M1 / M2 / H1 当前 P16、seed 42 的 FULL learned-slope 主线。准备状态为 `PREPARED_NOT_TRAINED`：只搭建配置、配对清单和后续运行命令，不启动训练，不执行 optimizer update，也不运行训练 smoke。

## 唯一实验改动

将 temporal attention 中的 recency bias 固定为零。当前 learned-slope 每层有 6 个可学习 slope、2 个固定 flat heads；对照的 4 层、每层全部 8 个 head 都使用精确零 slope，不创建 `slope_log` 或其他可学习 recency 参数。

实现复用现有 FULL runner，固定传入：

```text
--tier fixed --half-lives none,none,none,none,none,none,none,none
--ladder default --per-layer --layers 4 --seed 42 --proj-dim 16
```

`fixed` 单独使用仍有固定的 recency bias，必须同时给出八个 `none` 才是本对照。FULL 的 `e0_proj(E0)`、carrier、bank 与 unit mask、因果前端、局部注意力窗口、时间 valid mask、readout 均沿用主线。flat 并不使注意力权重均匀：内容相关的 QK 分数仍然生效。

使用同一个 seed 时，flat 与 learned-slope 的共有参数初始化逐字节相同。实验不修改既有 model、config 或 runner；新增清单明确记录 `FULL_FLAT` 和全零的有效 bias。既有 runner 的部分 metadata 仍可能使用 `fixed` / `recency` 名称，实际算子由八个 `None` half-lives 与零 slope 验证确定。

## 配对与固定预算

| 项目 | M1 | M2 | H1 |
|---|---|---|---|
| 当前 FULL learned-slope | `m1_projadd_learned_slope_default_s42` | `m2_projadd_learned_slope_default_s42` | `h1_learned_slope_default_s42` |
| Context / layer windows | R100 / 25,25,25,24 | R50 / 13,12,12,12 | R300 / 75,75,75,74 |
| 架构 | D4 / P16 / width 256 / 8 heads | D4 / P16 / width 256 / 8 heads | D4 / P16 / width 256 / 8 heads |
| 训练预算 | 24 × 6,665 = 159,960 updates | 24 × 3,165 = 75,960 updates | 32 × 731 = 23,392 updates |
| Batch / EMA | 32 / 0.9995 | 32 / 0.9995 | 32 / 0.9995 |
| Flat 训练目标目录 | `results/m1_projadd_flat_p16_s42` | `results/m2_projadd_flat_p16_s42` | `results/h1_flat_p16_s42` |
| 评分与选轮 | HO3，扫描 24 轮 EMA | EXT6，扫描 24 轮 EMA | HO-M3 grouped-seven，扫描 32 轮 EMA |

source、sampler、目标缩放、bank 与 carrier 使用当前 FULL 的同一数据路径；AdamW 参数组、warmup / cosine schedule、whole-unit dropout 和 EMA 均由原训练入口执行。清单绑定当前 FULL 的元数据、选择记录与相关源码 SHA-256，保存原始的数据和 bank 契约供复核。

选轮沿用对应 FULL 主线的 development 规则：按其 session 聚合分数取最优 EMA，平分时取最早 epoch。因此该对照与刚搭建的 STATIC 最后一轮 EMA 规则分别记录。没有使用 official test 选轮。

历史 M1 / M2 flat 使用不同的 concat identity 路线，不能直接填入本次 P16 / proj_add 对照。历史 H1 signed-state14 flat 与当前主线有相同的 shared 初始化、bank 和预算，可保留作参考；本次仍为它建立独立的准备清单，不把旧结果记作本次新训练。

## 准备入口

从 workspace 根目录执行以下命令，只生成准备产物：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/learnable_recency_v1/scripts/prepare_flat_control.py
```

默认生成目录为 [results/flat_control_setup](../results/flat_control_setup)，包含每个任务的配对清单、总清单和 `run_commands.sh`。命令文件仅保存后续正式训练与评分命令，准备入口不会执行它，也不会创建正式训练目录。

## 无训练验收

验收只构建模型并做 forward / stream 检查，核对共有参数初始化、零 slope、dense / local / stream 一致性及跨窗口与 padding 行为；不执行 backward 或 optimizer step。另行离线解析生成的 CLI 参数，确认它们能被各任务现有入口接受。

最终 **17 项测试通过**：8 项配置与准备清单检查、2 项准备入口检查、7 项模型检查。模型检查使用合成 bank / neural 输入，覆盖 M1 105 bins、M2 55 bins、H1 305 bins，包含 padding 和跨窗口行为；与显式 flat 模型及 dense / local / stream 路径的比较均通过 `atol=rtol=1e-5` 检查。共有参数初始化要求逐字节相同，slope 要求精确 FP32 零。

集成测试进程禁止构造 optimizer 或调用 backward，三项禁止调用的计数均为 0。准备 CLI 另由清空 `PYTHONPATH`、从 workspace 根启动的真实子进程验证，仅生成清单。未来训练和评分的目标目录均未创建。

产物：

- [准备总清单](../results/flat_control_setup/manifest.json) 与 [未来运行命令记录](../results/flat_control_setup/run_commands.sh)。
- [集成验收收据](../results/flat_control_verification/acceptance.json)。
- [测试执行记录](../results/flat_control_verification/test_execution.json) 与 [17 项测试结果](../results/flat_control_verification/tests.xml)。

准备清单与测试收据分别保存在 `flat_control_setup`、`flat_control_verification`。此处没有新的训练 checkpoint 或 flat 性能分数。
