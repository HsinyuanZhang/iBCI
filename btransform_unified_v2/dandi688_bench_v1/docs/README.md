# dandi688_bench_v1 — 688 本地基准实验系列（骨架包）

- 创建：2026-09-09。上位设计：`btransform_unified_v2/docs/DESIGN_688_LOCAL_BENCHMARK_PROTOCOL_V1_20260909.md`（划分/对照臂/验收门 + ADDENDUM-TWO-STAGE 两段协议）。
- 数据/模型合同（冻结，只读引用）：`btransform_unified_v2/docs/EXECUTION_688_RIFT_V1_20260907.md`（prepared_cache、50-bin 窗、B3S E0、carrier `[a_R,c_R,m_R,δ_b]`、12ep e8–e11 平均、Nmax padding）。

## 本包定位

- **688 本地基准系列**（local-benchmark 口径，无 FALCON 官方面）：三主臂 T4 / F0 / TS4 + 预注册验收门，在协议 exam 面的 equal-session mean R² 上判读；另加 **T4-concat 融合消融臂**（carrier 原样 + concat 前端，只报读数、不进验收门）。
- **两段实验协议**（ADDENDUM-TWO-STAGE，用户指令"先窄时间跨度后完整"；runner `--protocol`，默认 `exp1_narrow`）：
  - **EXP-1 narrow（`exp1_narrow`）**：train = 2015-06-29→07-10 的 9 session（含首尾 12 天，对照 M2 的 10 天 7 session）；exam = 0713/0714/0715/0716 四 session（训练结束后 3–6 天，镜像 M2 ext4"隔天考卷"）。回答：同 regime 下 RIFT+对齐 carrier 是否复现 M2 量级。实测 ~5.9k upd/ep（session 长度不均匀：2015-06/07 块平均 ~21k 窗/session，低于 27-session 均值 ~33k，故不等于 28,076×9/27 的朴素估算 ~9.4k）。
  - **EXP-2 full（`exp2_full`）**：原 27 train / 6 val 协议不变（train 跨 2 年，val 距最后 train 约 4 个月），即本包 ADDENDUM 之前的全部行为。
  - **SESSION-SUBSET 纪律（`plan.CROSS_PROTOCOL_DISCLOSURE`，每张 receipt 逐字携带）**：EXP-1 的 4 个 exam session 是 EXP-2 的 27-train 成员——两协议独立测量，不得跨协议比较评分面，不得把 EXP-1 exam 判读当作 EXP-2 内的选点证据（反之亦然）。两段差（EXP-2 val − EXP-1 exam）= 长期漂移代价的干净测量。
- **融合轴与 carrier 臂正交**（用户裁定 2026-09-09："M1/H1 上 add 是必须的（维度问题）；M2/688 上 concat 和 add（joint）都要试"）：t4/f0/ts4/`z0` 全部保持 settled `proj_add` 前端（P: 50→16 加到 local，token_in 20）；`t4_concat` = 真 carrier 走 matched concat 前端（`btransform_unified_v2/src/btransform_unified_v2/concat_model.py`，`[local16 | E0_50 | carrier4]`，token_in 70，init 为 proj_add 的函数保持折叠）。`f0` = activity-only（E0 保留、carrier 置零）；`z0` = query-only（E0 与 carrier 都置零），对齐 M1/M2/H1 的 Z_NONE。二者不进预注册门，只报 exam 读数。M2 先验：concat 略优（0.4501 vs 0.4016；RIFT 线 concat 0.3901 > joint 变体），688 与 M2 同宽 70，此臂直接检验该先验是否迁移。融合臂 t4_concat 在 EXP-1 跑（regime 干净，消融解释力最强）；EXP-2 可选。
- **从 rift_v1 主包独立**：代码、结果、receipt 全部落在 `btransform_unified_v2/dandi688_bench_v1/` 内，不修改 rift_v1 的任何历史 root/契约文件。
- **只读复用 rift_v1 的 prepared cache**：在盘的不可变数据束
  `btransform_unified_v2/results/rift_v1/dandi688_prepared_cache_contract_v2`
  （schema `dandi688_rift_v2_prepared`，27 train + 6 val，`formal_test_used=false`，不含任何 formal-test 数据）。本包只加载并逐数组重验 hash，绝不写入；协议过滤是纯 name 级过滤（cache 行已逐 session 存在，`run_688_bench.filter_rows_by_protocol` + `load_rows` 只加载当前协议的 train+exam session）。
  注：rift_v1 训练脚本的 `--prepared-cache` 默认值写作 `dandi688_prepared_cache_v1`，实际在盘并被历次 preflight 合同绑定的目录是 `dandi688_prepared_cache_contract_v2`；本包以后者为准（见 `plan.PREPARED_CACHE_RELATIVE`）。
- 模型定义不复制：训练/打分阶段的 RIFT 模型直接 import `btransform_unified_v2`（rift_v1 同源），PYTHONPATH 方式复用。

## 纪律（本骨架包的硬约束）

1. **GPU 训练已解禁**：用户授权的 2026-09-09 修订（carrier-iteration wave）将 `plan.GPU_POLICY` 翻转为 `ALLOWED_REVISION_20260909`。`--stage train` / `--stage score` 接受 `--device`（默认 `cuda:0`）；`--stage preflight` 仍为纯 CPU，保留 `cuda.is_initialized()` 守卫。
2. **训练循环**：`--stage train` 镜像冻结的 rift_v1 12-epoch AdamW 循环（e8–e11 平均），写出 `data_contract.json`、`epoch_XXX.pt`、`average_e8_e11.pt`、`train_receipt.json`。
3. **formal-test 隔离**：6 个 formal-test session 名单冻结在 `plan.FORMAL_TEST_SESSIONS`，仅作拒绝名单用；`eval_local.load_session` 与 `evaluate_val_face` 对该名单直接 raise（评分面的期望 session 列表里同样拒绝），任何阶段不加载其 spike/behavior/trial 数据。val/exam 面定案前 test 面封存，过门后一次性解封（`plan.FORMAL_TEST_POLICY`）。
4. **manifest SHA 冻结**：`4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`（`sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`，27/6/6）。
5. **协议纪律**：`--protocol` 只取 `plan.PROTOCOLS` 的键；每个 stage 启动时先跑 `plan.verify_protocol_definitions()`（manifest 交叉校验：exp1 的 13 个 session 全部 ⊂ manifest train split；exp2 面恰为 27/6）；gate_report 三臂算术对两协议同样适用（EXP-1 在 4-session exam 面判读，EXP-2 在 6-session val 面判读）。

## 结构

```
dandi688_bench_v1/
  docs/README.md                 # 本文件
  src/dandi688_bench_v1/
    plan.py                      # 冻结常量：SCHEMA/ARMS/FUSION_MODES/验收门/GPU 纪律/manifest 与 cache 绑定
                                 #   + 两段协议 PROTOCOLS/DEFAULT_PROTOCOL/CROSS_PROTOCOL_DISCLOSURE
    ts4_shuffle.py               # TS4 的 session 内 carrier 行置换（seed=42 派生，逐 session 独立）
    arms.py                      # t4/f0/ts4/t4_concat 的 arm→{carrier,fusion} 映射 + carrier 变换纯函数 + digest 断言
    support_resample.py          # M5/M8/M10 重采样 stub（Phase 2 可选，NotImplementedError）
    eval_local.py                # exam 面 equal-session mean R²（接受协议传入 session 列表）+ test-split 拒绝
    receipts.py                  # seal_json/read_sealed（0444 + sha256 sidecar，v1 同款法度）
  scripts/run_688_bench.py       # --stage {preflight,train,score} --arm {t4,f0,ts4,t4_concat}
                                 #   --protocol {exp1_narrow,exp2_full}（默认 exp1_narrow）
  tests/test_dandi688_bench_v1.py
  results/                       # 本包 preflight/未来训练的 dest 与 sealed receipts
    preflight_t4_v1/             # t4 (proj_add, 27/6 全协议) preflight receipt（ADDENDUM 前已封存，勿覆盖）
    preflight_t4_concat_v1/      # t4_concat (concat 融合, 27/6 全协议) preflight receipt（同上）
    preflight_exp1_narrow_v1/    # t4 (proj_add, exp1_narrow 9+4) preflight receipt
```

## 运行

```bash
cd /home/xinyuan/Work_host/SPINT

# EXP-1 narrow（默认协议）：9-session train + 4-session exam 面
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm t4 --protocol exp1_narrow \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_exp1_narrow_v1 \
  --cpu-threads 2

# EXP-2 full：原 27/6 协议（行为同 ADDENDUM 之前；--protocol 缺省时为 exp1_narrow；
# 注意用新 dest，勿重跑覆盖 ADDENDUM 前的 preflight_t4_v1 历史 seal）
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm t4 --protocol exp2_full \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_exp2_full_v1 \
  --cpu-threads 2

# 融合消融臂（concat 前端，token_in=70；独立 receipt；协议同上可选）
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm t4_concat --protocol exp1_narrow \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_t4_concat_v1 \
  --cpu-threads 2
```

验收门（预注册，详见上位设计 §3；双门对两协议同样适用，各自在自己的 exam 面上算）：T4 − F0 ≥ +0.03 且 T4 − TS4 ≥ +0.03；RIFT+T4 对封存参照 SPINT+T4 dev-6 0.5750 非劣（≥ −0.01）。**t4_concat 不进任何门**（`plan.gate_report` 保持三臂算术不变），只在 exam 面报读数。
