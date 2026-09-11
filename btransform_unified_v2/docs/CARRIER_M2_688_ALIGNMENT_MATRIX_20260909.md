# M2 vstate4 ↔ 688 vstate 逐部件对应矩阵与审计（2026-09-09）

状态：**已执行**（对齐审计 + 必要修正已落代码，全部测试绿）。用户指令："688 系列要做到与 M2 **最大程度对应**（carrier 构建方面）"。操作化定义：688 vstate 与 M2 vstate4 **逐部件对齐**，只允许因数据集物理属性不同的必要差异。M2 参考实现：`btransform_unified_v2/scripts/carrier_v3_m2/build_m2_variant_cache.py`（`_blocks` + `_vstate_raw`，2026-09-09 在盘 cache `results/carrier_v3_m2/run_vstate4`）；688 实现：`btransform_unified_v2/dandi688_bench_v1/`（`src/dandi688_bench_v1/vstate.py` + `scripts/build_vstate_cache.py`）。688 侧由并行 vstate-688 agent 于同日先行落地（R700/H300 相位窗块 + δ_b 读出），本审计按矩阵在其成果上做最小修正，未推翻其结构。

## 0. 结论先行

前三列读出（a、c、m）与状态构建、标准化、收缩、归一化的公式在两数据集间**逐字相同**（同一 `carrier_profile_v3` 函数路径，合成输入下逐位一致断言通过）。第 4 列主变体按最终裁定对齐 M2（b = meanₖR）；δ_b 降为 `vstate_b_hold` 消融变体（可一键切换，a/c/m 与主变体逐位相同）。率原语（M2 bin 求和 vs 688 searchsorted 半开计数）在相同边上**逐位等价**（实测 183,658 次块×单元比较最大绝对差 0），无需改实现。矩阵三处 DEVIATION-TO-FIX（支持集 vstate_full、第 4 列、vstate_concat 臂）已全部修正落地。

## 1. 对应矩阵（审计主体）

状态标记：SAME = 公式/语义逐字同；ALLOWED-DIFF = 数据集物理属性差异（必要且已披露）；DEVIATION-TO-FIX→FIXED = 偏离矩阵已修正。

| # | 部件 | M2 vstate4 | 688 vstate（修正后） | 状态 | 核验点与证据 |
|---|---|---|---|---|---|
| 1 | 支持集 | M33 全部校准 trial | 主变体 = **M10 同 10 个 trial**（`_phase_trials(range(10), "candidate")`，与现行 t4/u1_m10 完全同集）；**vstate_full** 变体 = 全部 30 个 activity-support trial（`range(30), "reliability_audit"`） | DEVIATION-TO-FIX→FIXED | 主变体保持 M10（矩阵规定：对齐"现行 t4 支持集"语义）；"全部支持"语义由 vstate_full 承担（变体定义已落 `_variant_spec`，cache 未建，符合"只补构建参数"）。实测 exp1 9 个 train session 全部 10/10 合法（无排除），M30 全部 30/30 合法。预算差 33↔30 为 trial 表结构差异（688 首 30 名单即全部 activity support） |
| 2 | 块化 | trial 内不重叠 100ms 块（5×20ms bin，尾部余数丢弃） | 相同的 100ms 不重叠块；块所在窗 = 688 冻结的 R700（7 块）/H300（3 块）相位窗（`vstate.block_edges`，linspace 钉死首尾边） | SAME（块几何）/ ALLOWED-DIFF（覆盖窗） | 块长 0.1s、不重叠、块内均值语义逐项同（`plan.VSTATE_BLOCK_SECONDS == 0.1 == 5×0.02`）。窗差异：M2 块铺满 trial bin 跨度，688 按用户指令铺在冻结相位窗内（R700/H300 是 688 独有的物理相位，事件锚定；把 688 块铺满全 trial 会拖入冻结分析窗之外的段）。实测 10 trial×7 块 = 70 R700 块、3 块 = 30 H300 块/session（builder 断言） |
| 3 | 率原语 | 块计数（bin 求和，`chunk[sl].sum(axis=0)/0.1`） | spike-time searchsorted 半开计数/块时长（`vstate.block_rate_matrix`，688 现行原语族） | SAME（数学等价，实测逐位一致） | 相同边上两原语恒等：bin 边上的半开计数求和 = searchsorted 差。实测（见 §2 证据 E1/E2）：exp1 9 session M10 支持 183,658 次块×单元比较 **max abs diff = 0**。锚定差异是**窗**属性不是原语属性（见 D3），故按矩阵"一致则不改实现"保持 688 原语 |
| 4 | 速度来源 | calib_covariates（与 calib_neural 同 bin 轴）的块内均值 | cursor_vel（2 维）按 multisession_datamodule 法插值到 20ms bin 中心（线性、界外 0.0），块内 5 bin 均值 | SAME（语义）/ ALLOWED-DIFF（数据源） | 块内速度均值语义同；插值点 = 块内 5×20ms 等距中心，与 M2 的 5 bin 协变量均值同构（线性速度下解析等值，测试断言）。数据源差异（cursor_vel vs covariates）为数据集物理属性 |
| 5 | 状态 | softplus(±v/rms) 4 状态（2 轴×2），rms=sqrt(mean(square)) | 相同（cursor_vel 同为 2 维） | SAME（公式）/ ALLOWED-DIFF（rms 拟合域） | 同一 `v3.signed_state_weights`。rms 域：M2 = 7 held-in 校准块；688 = **协议 train sessions**（exp1=9/exp2=27），且按 D2 定为**各支持面自己的 R700 块总体**（= 进入估计器的块总体，M2 同款规则）。实测（exp1 9-train、M10）：R700 块 rms = [9.5250, 10.0165]（builder 冻结域）；对照口径 R700+H300 = [7.9725, 8.3828]、全 trial 网格块 = [5.0969, 5.4107]。688 `fit_velocity_rms` 对退化输入 fail-closed（M2 为 1e-6 地板），实数据 rms ~5–10 远离零，见 D6 |
| 6 | 标准化 | 块计数 Poisson（Δ=0.1s）：`(r−r̄)/σ`，`σ=sqrt(max(r̄Δ,1))/Δ` | 相同（`v3.poisson_standardize(rates, 0.1)`） | SAME | 公式逐字同（同一函数）；Δ=块时长=0.1s 两侧一致（测试钉死常量） |
| 7 | 收缩 | n0=10 块（`conditional_response` 分母 `Σw+n0`） | 相同（`plan.VSTATE_N0_BLOCKS == 10.0`） | SAME | 逐字同；合成输入逐位一致测试覆盖 |
| 8 | 读出 | a=R₊ₓ−R₋ₓ, c=R₊y−R₋y, m=hypot, **b=meanₖR**（M2 无 hold 相位） | **主变体 b=meanₖR（与 M2 逐字同公式，最终裁定）**；δ_b（b=meanₖR−meanₖR_hold，hold=H300 块同类条件响应）降为 **vstate_b_hold** 消融变体（`b_mode="hold_diff"` 可一键切换） | DEVIATION-TO-FIX→FIXED（原实现 δ_b 为唯一读出） | 裁定时间线见 D1。a/c/m 在两模式间**逐位相同**（同 rms/同输入，测试断言），消融只隔离第 4 列；δ_b 的既有证据（Stage-1 SE-T4−PHASE-R +0.0001，δ_b 列可靠性 0.934）记录在案，不作为反对理由。切换条件：主变体 exp1 读数显著差于 t4 时启用 vstate_b_hold |
| 9 | 归一化 | 7 held-in 列 mean/std | 协议 train sessions 列 mean/std（vstate 族 cache 与协议绑定，`estimator.protocol` 断言） | SAME（规则）/ ALLOWED-DIFF（名单） | 规则同为 train 面列 mean/std（std 地板 1e-6 两侧同 `v3.fit_column_normalizer`）；名单差 = 数据集结构（M2 7 held-in / 688 9 或 27） |
| 10 | E0 同变 | `native_e0_and_u(encoder, calib_activity, empty_contrast_side(T_new))` | `post_pool(cat(pre_pool(calib_trials).mean(1), carrier_norm))`（`vstate.remelt_e0`，与 t4/u1 构建器同式） | SAME（式）/ ALLOWED-DIFF（encoder） | 同为"活动均值 ⊕ 新 T side 进 post_pool"的同变式；encoder 差异（M2 champion/p0 FiLM vs 688 冻结 B3S student）为数据集管线属性。f_labelfree 的零 side 消融把 E0 的标签依赖单独隔离 |
| 11 | 融合/模型 | RIFT R50 D4 **concat**（主线） | bench 现行 proj_add P16 + **新增 `vstate_concat` 臂**（vstate carrier × concat 前端，token_in 70，carrier 变换与 vstate 逐字节相同） | DEVIATION-TO-FIX→FIXED | 臂定义已落 `plan.ARMS`/`arms.ARM_SPECS`（identity carrier + concat fusion）；只报读数不进任何门（`gate_report` 三臂算术与 `vstate_gate_report` 均不含它，测试断言）。M2 concat 先验：ext4 SPINT concat 0.4501 vs proj_add 0.4016，688 同宽 70 直接检验迁移 |

不破坏声明：t4/f0/ts4/t4_concat/z0 语义逐字节未动；并行 agent 的 z_vstate_srcbank（bank 替换）与 f_labelfree（零 side）语义未动（`test_existing_arm_semantics_untouched` 断言）。

## 2. 数值证据（2026-09-09 实测，纯 CPU 只读）

**E1 率原语奇偶（真实 session，逐位）**。session `sub-C_ses-CO-20150629`：unit 0 全网格 22,038 spike，`np.histogram` bin 计数与 searchsorted 半开计数逐 bin 相等；trial 0 的 40 个网格对齐块上 bin 求和 vs searchsorted **逐块差 0**。扩展到 exp1 9 个 train session 的 M10 支持全部块×单元：**183,658 次比较，max abs diff = 0**（逐 session 检查数 18,179/17,556/19,796/22,828/18,170/16,422/23,616/24,156/22,935）。

**E2 锚定的数值后果（为何必须显式定窗）**。同一 trial：网格对齐块（M2 式 5-bin 组）总计数 83 spike；trial 起点绝对时间锚定块（start+0.1i）总计数 84（前 8 块 [3,5,2,4,2,1,3,3] vs [4,4,2,4,3,1,3,3]）。即原语等价的前提是边对齐；688 的 R700/H300 窗是事件锚定（物理相位），这是窗语义差异而非原语差异（D3）。

**E3 块统计与合法性（M10/M30 支持）**。exp1 9 个 train session：M10 全部 10/10 合法（`_phase_trials` 无排除），R700 70 块 + H300 30 块/session；M30 全部 30/30 合法（对照块计数：全 trial 网格块 1,144–1,249/session）。

**E4 rms（exp1 9-train、M10）**。R700 块域（builder 冻结）：**[9.5250, 10.0165]**；R700+H300 拼接域（agent 原方案）：[7.9725, 8.3828]；全 trial 网格块域（审计采集口径）：[5.0969, 5.4107]。对照 M2 vstate4 cache 在盘 rms = [0.00950, 0.01012]（速度单位/尺度不同，无对应关系）。

**E5 逐位一致（合成输入，测试钉死）**。`tests/test_vstate_alignment.py`：相同 (rates, vels, rms) 下，688 主变体核心与 M2 `_vstate_raw` 的逐字转录（M2 builder 因重依赖不可直接 import）在 float64 估计器核心层**逐位相等**（`np.array_equal`），float32 打包为同一数值的相同 cast；b_hold 模式与主变体的差恰为 −meanₖR_hold。9/9 通过。

## 3. 决策记录

- **D1 第 4 列（读出）**。原始矩阵：已知偏离（688 δ_b），建议主变体对齐 M2。执行中出现两次裁定反转：先"δ_b 保留为主变体（数据集特色）"，后撤销回"**主变体 meanₖR（最大对应优先），δ_b 降 b_hold 消融**"。最终落地：`vstate.vstate_carrier_from_blocks(..., b_mode)` 默认 `"mean_k"`；`vstate_b_hold` 为命名 cache 变体；切换条件 = 主变体 exp1 读数显著差于 t4。表述："M2/688 前三列逐字同公式；第 4 列主变体对齐 M2（meanₖR）；δ_b 为 b_hold 消融变体"。
- **D2 rms 域**。M2 的 rms 拟合于**进入估计器的同一块总体**。688 对应实现：各支持面的 **R700 块**（主变体块总体）；vstate 与 vstate_b_hold **共享 M10 rms**（保证消融只差第 4 列）；vstate_full 用自己的 M30 块域。agent 原方案（R700+H300 拼接）会使主变体 rms 依赖一个不进入其估计器的块总体，已改。
- **D3 窗锚定**。M2 块铺 trial bin 跨度（trial 对齐网格）；688 按用户指令铺在 R700/H300 冻结相位窗（事件锚定，`descriptors.is_legal_phase_trial` 的精确半开约定）。两者都是"数据集的物理运动/静止窗"；E2 证明锚定选择有数值后果，故显式记录而非默认对齐。
- **D4 变体消费方式**。vstate_full / vstate_b_hold 是 **cache 构建参数**（非新臂）：`vstate`/`vstate_concat` 臂通过 `--prepared-cache cache_vstate_full_<protocol>` 等消费，`plan.ARM_REQUIRED_CACHE_VARIANT` 的值改为允许变体族（frozenset），`run_688_bench.assert_cache_variant_for_arm` 做成员校验 + 协议绑定。z_vstate_srcbank 仅绑定主 `vstate` 变体（z 系列消融定义在主臂上）。
- **D5 vstate_concat 不进门**。对齐矩阵"融合"行只要求臂定义存在（M2 主线 concat 的最大对应读数点）；`gate_report` 三臂算术、`vstate_gate_report` 的 vstate−t4 门均不含它。
- **D6 rms 退化守卫**。M2 对非正 rms 取 1e-6 地板；688 `fit_velocity_rms` fail-closed（raise）。实数据 rms ~5–10 不触发；差异记录在案，不改（fail-closed 与本包法度一致）。

## 4. 实现映射

| 对象 | 位置 |
|---|---|
| 688 纯数学（块边/率原语/速度块均值/rms/估计器/读出/E0 remelt） | `dandi688_bench_v1/src/dandi688_bench_v1/vstate.py`（`b_mode` 参数，默认 `mean_k`） |
| 冻结合同常量（块/n0/相位窗/状态列/sub-bin/支持面/b 模式/变体族） | `dandi688_bench_v1/src/dandi688_bench_v1/plan.py`（`VSTATE_*`、`VSTATE_B_MODES/MAIN`、`VSTATE_VARIANTS`、`ARM_REQUIRED_CACHE_VARIANT`） |
| 臂注册（vstate_concat = identity×concat） | `plan.ARMS` + `arms.ARM_SPECS` |
| cache 变体构建参数（vstate/vstate_full/vstate_b_hold/f_labelfree） | `dandi688_bench_v1/scripts/build_vstate_cache.py`（`_variant_spec`/`B_FORMULAS`/`_variant_dest`） |
| 变体族 + 协议绑定校验 | `dandi688_bench_v1/scripts/run_688_bench.py::assert_cache_variant_for_arm` |
| 对应性测试（逐位一致/原语奇偶/速度语义/变体与臂声明） | `dandi688_bench_v1/tests/test_vstate_alignment.py`（9 项） |
| M2 参考实现（不改动） | `btransform_unified_v2/scripts/carrier_v3_m2/build_m2_variant_cache.py`；在盘 `results/carrier_v3_m2/run_vstate4/cache/carrier_normalizer.json` |

## 5. 验证状态

- `dandi688_bench_v1/tests/`：**53 passed, 1 skipped**（54 collected；skip = 变体 cache 未建盘的真实 cache 消融测试，符合"不必现在建 cache"）。
- 修正只落在 `dandi688_bench_v1/` 与本文件；父包 `btransform_unified_v2/tests/` 126 passed（`test_h1_training_contract.py` 1 例收集失败为预先存在的 tfpd 环境缺模块 `src.models.components.streaming_encoders`，与本任务无关）。
- cache 尚未构建（矩阵交付为定义 + 参数）；构建命令见 `dandi688_bench_v1/docs/README.md`（`build_vstate_cache.py --variant all --protocol exp1_narrow`）。

## 6. 与并行 vstate-688 agent 成果的关系

并行 agent 同日落地的骨架（R700/H300 相位窗块、searchsorted 原语、cursor_vel bin 中心插值、M10 同集、协议绑定 cache、z/f 消融臂、E0 remelt、测试骨架）**全部保留**。本审计在其上的修正（均为矩阵/裁定驱动）：(1) 第 4 列主变体 δ_b→meanₖR，δ_b 降 `b_mode="hold_diff"`；(2) builder 增 `vstate_full`/`vstate_b_hold` 变体定义与按支持面的块读取复用；(3) rms 域 R700+H300→R700（D2）；(4) 新增 `vstate_concat` 臂与变体族绑定；(5) 相应测试断言更新 + 对应性测试文件 + README/bench 文档同步。其 `z_vstate_srcbank`/`f_labelfree` 语义逐字未动。
