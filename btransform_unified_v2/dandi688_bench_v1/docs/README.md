# dandi688_bench_v1 — 688 本地基准实验系列（骨架包）

> **v2 已完成代码与 CPU smoke：**[DANDI000688 SUA／PMUA v2 设计](../../docs/DESIGN_DANDI688_SUA_PMUA_V2_20260911.md)，2015-only 18/6/6、M33 MOVE-T4 余弦 OLS、M2 learned RIFT；[执行入口](../../dandi688_bench_v2/README.md)。正式训练尚未运行。本 README 保留旧 v1 骨架和历史结果，不作为 v2 主矩阵结果。

- 创建：2026-09-09。上位设计：`btransform_unified_v2/docs/DESIGN_688_LOCAL_BENCHMARK_PROTOCOL_V1_20260909.md`（划分/对照臂/验收门 + ADDENDUM-TWO-STAGE 两段协议 + vstate-688 系列增补 + 组件消融定版）。
- 数据/模型合同（冻结，只读引用）：`btransform_unified_v2/docs/EXECUTION_688_RIFT_V1_20260907.md`（prepared_cache、50-bin 窗、B3S E0、carrier `[a_R,c_R,m_R,δ_b]`、12ep e8–e11 平均、Nmax padding）。

## 本包定位

- **688 本地基准系列**（local-benchmark 口径，无 FALCON 官方面）：三主臂 T4 / F0 / TS4 + 预注册验收门，在协议 exam 面的 equal-session mean R² 上判读；另加 **T4-concat 融合消融臂**（carrier 原样 + concat 前端，只报读数、不进验收门）。
- **组件消融（本包主对比轴；用户 2026-09-09 澄清原文："我不是要对比离散标签和连续标签，而是 carrier + activity 对于跨 session 能力的消融实验，各自有多少效果"；同日嵌套阶梯终版裁定："算 carrier 已经拿到 activity 信息，不用白不用"）**——**嵌套阶梯**贡献分解（信息包含关系：算 carrier 必须先读校准神经数据，activity 信息已到手——"有标签无身份"不构成现实部署场景，故**无 CARRIER_ONLY 阶级**），基于 t4（离散方向标签基线），结构化定义冻结在 `plan.COMPONENT_ABLATION`，判读面 = **exp1_narrow exam 面**（0713/0714/0715/0716 四个跨日期 session——"跨 session 能力"的面；辅报 ts4 内容对照），分解公式见 `plan.component_ablation_report`：
  - 三层阶梯（主臂）：

    ```
    FULL（满配）            = t4            E0 = activity + carrier side 熔炼；direct T（carrier token）在
      ⊃ ACTIVITY_ONLY      = f_labelfree   E0 = 零 side 重熔（encoder 权重与活动内容保留）；T = 0
          ⊃ NONE（地板）   = floor         E0 = 0（零张量）；T = 0 —— 只有 raw spike + 权重
    ```

  - 附属（不进预注册分解）：**equiv_zero** = 参数量对照（同 t4 参数量的网络，E0 通路换成同参数量固定随机投影 seed42 不训练、作用于 pre_pool 活动均值、零 side、无身份无标签、非零；T 形状保留值置零）——区分"缺信息"vs"缺通路/容量"；**z0**（E0 置零 + carrier 保留的 carrier-only 网格）降为附属探索读数；**z_vstate_srcbank** 降为附属。
  - **f_labelfree vs floor 的区别（易混，明确披露）**：f_labelfree **保留 E0 通路**——冻结 B3S encoder 权重与活动内容完整保留，仅行为标签 side 输入置零（E0 = post_pool(cat(pre_pool 活动均值, 零 side))），测"activity 校准（无标签）能带来多少跨 session 能力"；floor **把 E0 整个置零张量** + carrier 置零——模型只有原始 spike + 权重，是"零校准信息"的绝对地板。二者差 = **E0 的纯 activity 成分（不含标签）的贡献**。
  - **预注册分解（三项，全部在 exp1_narrow exam 面判读）**：
    - **activity 独立贡献** = ACTIVITY_ONLY − NONE（f_labelfree − floor）
    - **carrier 标签增量贡献** = FULL − ACTIVITY_ONLY（t4 − f_labelfree）
    - **校准总价值** = FULL − NONE（t4 − floor）
    - 算术律：前两项之和 == 第三项（嵌套阶梯）。附属读数（不进门）：equiv_zero 通路价值 = equiv_zero − floor、信息价值 = t4 − equiv_zero；z0 carrier-only 读数 = t4 − z0；f0 direct-carrier 边际 = t4 − f0。
  - **equiv_zero 实现契约**（用户裁定原文大意："对无 calibration 的臂也要补充一个等效 encoder 参数量的网络来对齐，但这个网络不构成任何身份，而是直接开始新 session 上运行 decoding"）：满配 E0 由冻结 B3S encoder 产生；equiv_zero 用**同参数量的固定随机投影**（对 post_pool 逐层 Linear 镜像、seed=42 冻结、永不训练）作用于 **pre_pool 活动均值**（零 side），每个 session 拿到同一随机投影作用于自身活动——参数量等、无标签、非零；carrier 位置照 t4 的 token 形状保留但值置零。严格断言：equiv_zero 模型的可训练参数量 == t4（同一 model class）；随机投影参数量 == 被替换的 post_pool 通路；投影权重 SHA 记入 receipt；无任何训练后的身份注入。实现：`src/dandi688_bench_v1/equiv_zero.py` + `build_vstate_cache.py --variant equiv_zero`（E0 熔炼落在 cache 字节里，臂级变换只置零 carrier——与 f_labelfree 同法度）。
  - **z0 语义迁移披露**：2026-09-09 澄清把 z0 从"双零（query-only Z_NONE）"重定义为"carrier-only（E0 置零、carrier 保留）"；旧语义（双零）归 **floor** 臂；嵌套阶梯终版又把 z0（与 z_vstate_srcbank）降为附属探索（不进预注册分解）。已存档的 `results/train_exp1_narrow_z0`（旧 z0 定义下训练）测的正是今天的 floor。
- **vstate-688 系列（附属探索——2026-09-09 组件消融澄清后不再作为主对比轴；用户指令原文："立刻开始准备 vstate-688，注意消融——**z 系列 = 完全不在新日期校准**（bank 复用旧日期/source），**f 系列 = 不使用任何标签校准**（label-free）"）**：
  - **`vstate`（主臂）**：M2 vstate4 配方（`PLAN_CARRIER_ITERATION_M2_688_20260909.md` §3.2）适配 688——**M10 同集** trial（与 t4 完全相同的 10 个 rewarded trial，支持集不变，只有标签从方向换速度）的 R700/H300 窗口内 **100ms 块**；率原语 = spike-time searchsorted 半开计数/块时长（与 M2 bin 求和在相同边上逐位等价，见对齐审计）；状态 W=[softplus(v_x/rms_x), softplus(−v_x/rms_x), softplus(v_y/rms_y), softplus(−v_y/rms_y)]（v = cursor_vel 块均值，本包独立实现按 20ms bin 中心插值后取块内 5 bin 均值）；**rms 与列归一只用协议 train sessions 拟合**（exp1_narrow = 9 个，exp2_full = 27 个——cache 与协议绑定）；Poisson 标准化（Δ=0.1s）+ n0=10 块收缩；读出 a=R₊ₓ−R₋ₓ, c=R₊y−R₋y, m=hypot, **b=meanₖR（主变体第 4 列对齐 M2，用户裁定 2026-09-09 终版：最大对应优先；逐部件审计见 `btransform_unified_v2/docs/CARRIER_M2_688_ALIGNMENT_MATRIX_20260909.md`）**。**E0 同变重算**：post_pool(cat(pre_pool 活动均值, vstate carrier 归一))——与 t4 臂的 E0 构造同式。判读门：**vstate − t4 ≥ +0.03**（`plan.vstate_gate_report`）。
  - **`vstate` 族的 cache 变体**（对齐矩阵两行的修正产物，定义已落 `build_vstate_cache.py --variant`，未建盘）：`vstate_full` = 支持集扩到全部 30 个 activity-support trial（reliability_audit 名字空间，对齐 M2"全部校准 trial"语义）；`vstate_b_hold` = 第 4 列换回 δ_b：b=meanₖR−R_hold（hold=H300 块的同类条件响应，R700 affine；与主变体共享 M10 rms，a/c/m 逐位相同，消融只隔离第 4 列；切换条件 = 主变体 exp1 读数显著差于 t4 时启用，`vstate.vstate_carrier_from_blocks(..., b_mode="hold_diff")`）。
  - **`z_vstate_srcbank`（z 系列消融，"完全不在新日期校准"）**：模型/训练同 vstate，但 exam session 的 bank（E0+carrier）在数据加载后被**冻结的最近日期 train session bank** 替换——exp1_narrow 全部 4 个 exam→2015-07-10、exp2_full 全部 6 个 val→2015-07-16（"带最近的旧 bank 部署"的诚实退化；映射表与泄漏断言 exam 日期 > source 日期冻结在 `plan.Z_SRCBANK_MAPS`，替换逐行复制 min(exam 行数, source 实行数)，超出部分置零）。读数：z − vstate = 校准缺失代价。注意与 legacy `z0`（"E0 置零"的 query-only 臂）无关，z 系列用 `z_<...>` 前缀避免混淆。
  - **`f_labelfree`（f 系列消融，"不使用任何标签校准"）**：E0 与 carrier 都不用任何行为标签——E0 = post_pool(cat(pre_pool 活动均值, **零 side**))（ACTIVITY-ONLY identity，此前诊断缺失的干净消融；legacy f0 保留的是标签熔炼的冻结 E0，测不出该量），carrier 全零。读数：vstate − f_labelfree = **标签信息总增量**。
  - 实现落点：cache 构建脚本 `scripts/build_vstate_cache.py`（--variant {vstate,vstate_full,vstate_b_hold,f_labelfree,equiv_zero,all}，--protocol 绑定 rms/normalizer 的 train 集；rms 域 = 各支持面的 R700 块，M10 族内共享；equiv_zero 协议无关）；纯数学在 `src/dandi688_bench_v1/vstate.py`（与 `btransform_unified_v2.carrier_profile_v3` 共享估计器模板）与 `src/dandi688_bench_v1/equiv_zero.py`（固定随机投影）；runner 对新臂做 variant-cache 族绑定校验（`--prepared-cache` 的 variant 必须在 `plan.ARM_REQUIRED_CACHE_VARIANT[arm]` 的允许集内；vstate 族 cache 还必须与 `--protocol` 匹配）。
- **两段实验协议**（ADDENDUM-TWO-STAGE，用户指令"先窄时间跨度后完整"；runner `--protocol`，默认 `exp1_narrow`）：
  - **EXP-1 narrow（`exp1_narrow`）**：train = 2015-06-29→07-10 的 9 session（含首尾 12 天，对照 M2 的 10 天 7 session）；exam = 0713/0714/0715/0716 四 session（训练结束后 3–6 天，镜像 M2 ext4"隔天考卷"）。回答：同 regime 下 RIFT+对齐 carrier 是否复现 M2 量级。实测 ~5.9k upd/ep（session 长度不均匀：2015-06/07 块平均 ~21k 窗/session，低于 27-session 均值 ~33k，故不等于 28,076×9/27 的朴素估算 ~9.4k）。
  - **EXP-2 full（`exp2_full`）**：原 27 train / 6 val 协议不变（train 跨 2 年，val 距最后 train 约 4 个月），即本包 ADDENDUM 之前的全部行为。
  - **SESSION-SUBSET 纪律（`plan.CROSS_PROTOCOL_DISCLOSURE`，每张 receipt 逐字携带）**：EXP-1 的 4 个 exam session 是 EXP-2 的 27-train 成员——两协议独立测量，不得跨协议比较评分面，不得把 EXP-1 exam 判读当作 EXP-2 内的选点证据（反之亦然）。两段差（EXP-2 val − EXP-1 exam）= 长期漂移代价的干净测量。
- **融合轴与 carrier 臂正交**（用户裁定 2026-09-09："M1/H1 上 add 是必须的（维度问题）；M2/688 上 concat 和 add（joint）都要试"）：t4/f0/ts4/`z0` 全部保持 settled `proj_add` 前端（P: 50→16 加到 local，token_in 20）；`t4_concat` = 真 carrier 走 matched concat 前端（`btransform_unified_v2/src/btransform_unified_v2/concat_model.py`，`[local16 | E0_50 | carrier4]`，token_in 70，init 为 proj_add 的函数保持折叠）。`f0` = activity-only（E0 保留、carrier 置零）；`z0` = carrier-only（E0 置零、carrier 保留；2026-09-09 前的旧 z0 语义"双零 Z_NONE"已由 **floor** 臂继承；嵌套阶梯定版后 z0 为附属探索）；`floor` = 双零绝对地板；`equiv_zero` = 同参数量随机投影容量对照（附属）。floor/equiv_zero 读数进 `plan.component_ablation_report`（z0 附属读数同）。M2 先验：concat 略优（0.4501 vs 0.4016；RIFT 线 concat 0.3901 > joint 变体），688 与 M2 同宽 70，此臂直接检验该先验是否迁移。融合臂 t4_concat 在 EXP-1 跑（regime 干净，消融解释力最强）；EXP-2 可选。vstate/z/f 系列臂 proj_add；另有 **`vstate_concat`**（vstate carrier × concat 前端，对齐矩阵"融合"行的补臂；只报读数、不进任何门，carrier 变换与 vstate 逐字节相同）。
- **披露（vstate 消费 dense cursor_vel 校准标签）**：vstate 的状态标签来自 M10 支持 trial 的稠密 cursor_vel——与 688 生产路径的 sparse-label 纪律（`materialize_sparse_event_t4` 不读 dense behavior）不是同一个问题：carrier 校准面按 FALCON 口径允许消费支持 trial 的行为标签（M2 vstate4 有同款披露）。旧 688 dense-null（dense-speed CP-FiLM REAL<EMPTY）**不构成反证**，三轴辨析：(1) **符号轴**——null 用无符号速度分位数 profile，vstate 用有符号方向状态；(2) **位置轴**——null 把 dense profile 叠加在完整 T4 之上（加性修饰），vstate 替换 T4 坐上 carrier 席位（M10 支持集不变）；(3) **通路轴**——null 走 CP-FiLM side 调制，vstate 走 carrier token 通路。M2 侧的 dense-profile null 同理（无符号 profile 叠加 T4，非有符号状态替换）。
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
                                 #   + COMPONENT_ABLATION（组件消融 2×2+对照矩阵 + 分解公式，主对比轴）
                                 #   + component_ablation_report（组件分解 receipt 算术）
                                 #   + VSTATE_* 配方常数（n0=10、块长 100ms、状态定义、支持=M10 同集）
                                 #   + Z_SRCBANK_MAPS（z 系列冻结最近日期映射表 + 泄漏断言）
                                 #   + ARM_REQUIRED_CACHE_VARIANT（新臂 ↔ variant cache 绑定）
                                 #   + vstate_gate_report（vstate−t4 门 + z/f 消融读数）
    ts4_shuffle.py               # TS4 的 session 内 carrier 行置换（seed=42 派生，逐 session 独立）
    arms.py                      # 11 臂的 arm→{carrier,fusion,e0} 映射 + carrier 变换纯函数 + digest 断言
                                 #   （z0=carrier-only、floor=双零、equiv_zero=随机投影容量对照）
    vstate.py                    # vstate-688 纯数学：块化率原语/cursor_vel 块均值/rms/估计器读出/E0 remelt
    equiv_zero.py                # equiv_zero 容量对照：post_pool 同参数量固定随机投影（seed42）+ E0 熔炼 + receipt
    support_resample.py          # M5/M8/M10 重采样 stub（Phase 2 可选，NotImplementedError）
    eval_local.py                # exam 面 equal-session mean R²（接受协议传入 session 列表）+ test-split 拒绝
    receipts.py                  # seal_json/read_sealed（0444 + sha256 sidecar，v1 同款法度）
  scripts/run_688_bench.py       # --stage {preflight,train,score} --arm {t4,f0,ts4,t4_concat,z0,floor,
                                 #   equiv_zero,vstate,vstate_concat,z_vstate_srcbank,f_labelfree}
                                 #   --protocol {exp1_narrow,exp2_full}（默认 exp1_narrow）
                                 #   z 臂在加载后替换 exam bank（记录映射与 SHA）；新臂做 variant 绑定校验
  scripts/build_vstate_cache.py  # 变体 cache 构建（CPU）：vstate 族 + f_labelfree + equiv_zero
  tests/test_dandi688_bench_v1.py
  tests/test_vstate_alignment.py # M2 对齐审计测试（并行线产物）
  results/                       # 本包 preflight/训练 dest、变体 cache 与 sealed receipts
    preflight_t4_v1/             # t4 (proj_add, 27/6 全协议) preflight receipt（ADDENDUM 前已封存，勿覆盖）
    preflight_t4_concat_v1/      # t4_concat (concat 融合, 27/6 全协议) preflight receipt（同上）
    preflight_exp1_narrow_v1/    # t4 (proj_add, exp1_narrow 9+4) preflight receipt
    cache_vstate_exp1_narrow/    # vstate 变体 cache（rms/normalizer = exp1 的 9 个 train session）
    cache_f_labelfree/           # f_labelfree 变体 cache（零 carrier + 零 side E0；协议无关）
    cache_equiv_zero/            # equiv_zero 变体 cache（随机投影 E0 + 零 carrier；协议无关；构建后出现）
    train_exp1_narrow_z0/        # 旧 z0 定义（双零）下的训练 dest —— 语义 = 今日的 floor（见 z0 迁移披露）
    preflight_vstate_v1/         # vstate 臂 exp1_narrow preflight receipt
    preflight_z_vstate_srcbank_v1/  # z 系列臂 exp1_narrow preflight receipt（含 bank 替换映射记录）
    preflight_f_labelfree_v1/    # f 系列臂 exp1_narrow preflight receipt
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

# vstate-688 系列：先构建变体 cache（CPU，一次性；--variant all 产出全部四个：
# vstate / vstate_full / vstate_b_hold / f_labelfree；vstate 族 cache 名带协议后缀）
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/build_vstate_cache.py \
  --variant all --protocol exp1_narrow

# 新臂的 preflight（vstate/vstate_concat/z 共用 vstate 族 cache；f_labelfree 用自己的 cache）
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm vstate --protocol exp1_narrow \
  --prepared-cache btransform_unified_v2/dandi688_bench_v1/results/cache_vstate_exp1_narrow \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_vstate_v1 \
  --cpu-threads 2
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm z_vstate_srcbank --protocol exp1_narrow \
  --prepared-cache btransform_unified_v2/dandi688_bench_v1/results/cache_vstate_exp1_narrow \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_z_vstate_srcbank_v1 \
  --cpu-threads 2
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm f_labelfree --protocol exp1_narrow \
  --prepared-cache btransform_unified_v2/dandi688_bench_v1/results/cache_f_labelfree \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_f_labelfree_v1 \
  --cpu-threads 2

# 组件消融（主对比轴）：floor 跑冻结 cache（双零变换即可）；equiv_zero 先建自己的
# 变体 cache（固定随机投影 E0，seed42，协议无关），再 preflight
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm floor --protocol exp1_narrow \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_floor_v1 \
  --cpu-threads 2
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/build_vstate_cache.py \
  --variant equiv_zero
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/dandi688_bench_v1/scripts/run_688_bench.py \
  --stage preflight --arm equiv_zero --protocol exp1_narrow \
  --prepared-cache btransform_unified_v2/dandi688_bench_v1/results/cache_equiv_zero \
  --dest btransform_unified_v2/dandi688_bench_v1/results/preflight_equiv_zero_v1 \
  --cpu-threads 2
```

验收门（预注册，详见上位设计 §3；双门对两协议同样适用，各自在自己的 exam 面上算）：T4 − F0 ≥ +0.03 且 T4 − TS4 ≥ +0.03；RIFT+T4 对封存参照 SPINT+T4 dev-6 0.5750 非劣（≥ −0.01）。**t4_concat 不进任何门**（`plan.gate_report` 保持三臂算术不变），只在 exam 面报读数。**组件消融**（嵌套阶梯 FULL=t4 ⊃ ACTIVITY_ONLY=f_labelfree ⊃ NONE=floor）在 exp1_narrow exam 面用 `plan.component_ablation_report` 分解为三项预注册读数（activity 独立贡献 / carrier 标签增量贡献 / 校准总价值，前两项之和 = 总价值），不设 pass/fail；equiv_zero 参数量对照与 z0 carrier-only 为附属读数；ts4 同面内容对照为辅报。vstate-688 系列门（附属探索）：**vstate − t4 ≥ +0.03**（`plan.vstate_gate_report`）；z − vstate（校准缺失代价）与 f_labelfree − vstate（标签信息总增量）为消融读数，不设 pass/fail。

## 2026-09-10 增补：消融条件轴（方案 B / C1 / D1 / D2，用户指令 2026-09-10）

在不动既有臂语义与结果的前提下新增一条**条件轴**：同一嵌套阶梯（t4 / f_labelfree，B 另含 floor）在不同数据/几何条件下的重测，读数与基线（exp1_narrow：t4 0.8729 / f_labelfree 0.8257 / floor 0.3247，封存 score receipts）并排对比，判读"哪个条件下 carrier 边际价值被放大"。全部只报读数、不进任何预注册门；`plan.ablation_condition_comparison` 生成对比报告（基线读数冻结在 `plan.EXP1_NARROW_BASELINE_R2`）。

- **B `exp1_poverty`（极端数据贫穷）**：train = 仅 20150629/0630/0701 三个最早 session（~65k 窗，exp1_narrow 的 ~1/3），exam 面不变（0713–0716）。纯 train-face 子集过滤，无需新 cache（`plan.PROTOCOLS["exp1_poverty"]` + `--protocol exp1_poverty`）；跑 t4 + f_labelfree + floor 三臂（嵌套阶梯贫穷重测）。
- **C1 `t4_dir16`（16 方向 carrier）**：方向设计矩阵从 8 方向 one-hot 升到 16 方向（22.5° bin），其余配方不动（同 R700/H300、同 Poisson/n0、同闭式一次谐波读出、同 M10 candidate 支持面）。cache：`build_carrier_cache.py --variant t4_dir16`（读同 33 行重算 carrier + E0 重熔）；臂 = `--arm t4 --prepared-cache cache_t4_dir16`。**退化审计（构建时留证）**：全数据集 target_dir 精确落在 8 正则角（68 session / 20089 trial 零偏差），16-bin 全落偶数 bin——实测 dir16 与 u1_m10（同族 8-dir）的 a/c/m 列**逐位相同**（半缩放被 train 列归一化精确抵消），仅第 4 列 b 移动（b16 = M8/2 − H，max 归一化差 0.083）；vs 冻结 t4 cache 的全列差异（~4）属于 u1 估计器家族与 rift_v1 冻结估计器的家族差（u1_m10 也一样）。干净对比器 = t4_dir16 vs **u1_m10 = 0.8826**（同族、逐位同 a/c/m）。
- **D1 `--model-override shortwin`（W=10）**：模型只看每个冻结 50-bin 窗的**最后 10 bin**（200ms 历史），查询目标与 W=50 完全逐位相同——干净隔离"原始历史预算"。RiftDecoder 原生支持 context_bins，无需子类；跑 t4 + f_labelfree 两臂。
- **D2 `--model-override shallow`（时间深度 1）**：temporal 4 层 → 1 层（同一 50-bin 感受野，windows=(46,)）；rift_v1 主包不改，bench 包内子类 `model_variants.RiftShallowDecoder`（同 fork_rng/init 纪律）；跑 t4 + f_labelfree 两臂。
- **合同绑定**：override 写入 bench contract（`model_override` / `input_window_bins` / `model_config.context_bins` / shallow 加子类源 hash）与两张 receipt；train/score 必须传同一 `--model-override`（fail-closed）。2026-09-10 之前的历史 run 无该字段 = "none"，新 runner 对其保持可计分（`_comparable_contract` 归一化）。
- **结果目录**：`results/train_exp1_poverty_{t4,f_labelfree,floor}`、`train_exp1_narrow_t4_dir16`、`train_exp1_narrow_{t4,f_labelfree}_shortwin`、`train_exp1_narrow_{t4,f_labelfree}_shallow`；preflight 目录同名 `preflight_*_v1`；GPU0 串行队列（外来 vstate 队列清空后接续，多臂不共卡）。
- 新模块：`src/dandi688_bench_v1/dir16.py`（16 方向映射 + 一次谐波读出 + 退化律文档）、`src/dandi688_bench_v1/model_variants.py`（浅核子类）。
