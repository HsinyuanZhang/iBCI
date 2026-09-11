# DESIGN — 688 本地基准实验协议 V1（无 FALCON 官方面）

> **ADDENDUM-TWO-STAGE（2026-09-09 用户指令：先窄时间跨度后完整）**：实验拆两段——
> **EXP-1 窄窗（M2 同级跨度）**：train = 2015-06-29→07-10 的 9 个 session（12 天，对照 M2 的 10 天 7 session）；**exam = 0713/0714/0715/0716 四个 session**（训练结束后 3–6 天，镜像 M2 的 ext4 "隔天考卷"结构）。块选 2015-06/07 而非 2015-03 的原因：只有它自带"紧随其后的考卷 session"。数据量 ≈ 全协议的 1/3（~9.4k upd/ep，仍为 M2 的 ~3×）。回答：**同 regime 下 RIFT+对齐 carrier 是否复现 M2 量级**（T4 双门 + 0.5–0.6 带锚）。
> **EXP-2 完整**：原 27/6 协议不动（train 跨 2 年，val = 2015-11 的 6 session，距最后 train 约 4 个月）。
> **两段差 = 长期漂移代价的干净测量**（EXP-2 val − EXP-1 exam，同架构同配方仅时间跨度不同）。披露：EXP-1 的 exam session ⊂ EXP-2 的 train 集——两协议独立、不得跨协议比面。融合臂 t4_concat 在 EXP-1 跑（regime 干净，消融解释力最强）；EXP-2 可选。

- 创建：2026-09-09，GLM5.3（规划/文档）。上位：btransform_unified_v2 的 [688 RIFT 执行契约](EXECUTION_688_RIFT_V1_20260907.md)（已冻结数据/模型合同，本文件补"实验设计与对照体系"层）。
- 背景约束（用户）：**688 无 FALCON 官方数据**——数据划分、对照实验、验收门全部本地定义；任何结论标注 local-benchmark 口径。

## 0. 之前的分析盘点（本次找到的三层）

| 层 | 文档 | 关键结论 |
|---|---|---|
| **carrier 对齐** | `sua_exploration/docs/DESIGN_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md`（+STAGE0_MASK_AUTHORITY） | 688 carrier = `[a_R, c_R, m_R, δ_b]`，前三列 = **R700 运动窗 T4 列**（与 M2 的 `[a,c,m]` 同估计器族），第四列 δ_b = R700 baseline − H300 hold baseline；R700 从候选时长中选出后冻结（披露为选择证据，禁止再试第三时长）；列序冻结 `[a_R,c_R,m_R,δ_b]`；split-half 信度 0.87–0.94 |
| **方法论先例** | `sua_exploration/docs/PSEUDO_MUA_T4_BRIDGE_48H.md`（COMPLETE） | 27/6/6 严格划分下 F0/T4/TS4 三臂 + 3 种子协议；T4 增益在 pooling 后保留（+0.3177 vs F0）；"formal test 一律不加载"的隔离纪律；score window = e5–12 平均 |
| **数字锚** | FABLE TKD 688 预检（已封存）+ 本次核验 | 封存参照：SPINT+T4 **0.5750** / B0 0.2364 / TS4 0.2845（dev-6 面）→ **T4 增量 +0.339 是 688 上最大的单成分效应** |

## 1. carrier 的 688↔M2 对齐表（实验文档必附）

| 维度 | M2 (MOVE-T4) | 688 (sparse-event directional) | 对齐状态 |
|---|---|---|---|
| 估计器 | 对 `[1,cosθ,sinθ]` 最小二乘 | 同（trig LSQ） | **同族** |
| 前三列 | `[a, c, m=√(a²+c²)]` | `[a_R, c_R, m_R]` | **逐列对应** |
| 第四列 | 绝对 baseline rate | **δ_b = b_R700 − b_H300** | 语义分叉：688 用运动−保持差分隔离运动相关基线漂移（H300 是 M2 没有的相位） |
| 率原语 | 未插值 bin 计数 / 长度 | searchsorted 半开窗计数 / 时长 | 同族不同原语（合同级披露） |
| 支持预算 | M33（全部校准 trial） | **M10**（carrier）/ M30（E0 identity） | 分预算 |
| 归一化 | 7 个 held-in session z-score | **仅 train sessions**（val/test 禁参与） | 688 纪律更严 |
| 置换对照 | —（后续 audit 加过） | TS4 = session 内 channel↔row 置换 | 688 原生带对照 |

## 2. 数据划分（唯一合法切分，SHA 冻结）

`subc_co_27_6_strict_train_val_manifest.json`（SHA `4607e979…`）：**27 train / 6 val / 6 formal-test**。
- **方法选择/开发面 = 6 个 val session**（承担其他任务里 ext6 的角色——"可见官方面"的本地等价物）；
- **formal test 6 session 本地存在但封存**：任何 arm 在 val 面定案前不得加载其 spike/behavior/trial（pseudo-MUA 先例的隔离纪律）；只在全部预注册判读完成后**一次性**解封计分，此后该划分作废（防重复消费）。

## 3. 对照臂体系（本地定义的"官方等价物"）

主臂（RIFT 架构，per 执行合同：50-bin 窗、B3S E0 [N,50]、M2-对齐 carrier、Nmax padding+mask、12ep seed42、e8–e11 权重平均不选点）：

| 臂 | carrier | 回答 |
|---|---|---|
| **T4** | 真值 [a_R,c_R,m_R,δ_b] | 主效应 |
| **F0** | 无 carrier | T4 的总增量 |
| **TS4** | session 内置换 | 增量是否依赖正确 channel↔carrier 对应（内容 vs 容量） |
| **T4-concat**（融合消融，2026-09-09 增补） | 真值原样（与 T4 同字节） | 融合方式消融：concat 前端下主效应是否变化（不进验收门，只报读数） |
| E0-zero | T4 + E0 置零 | identity 贡献分解（可选，M1 RIFT 的 25-condition 方法移植；已被组件消融的 z0/floor 定式化，见下） |
| support-resample | M5/M8/M10 各 n 次重采样 | carrier 支持稳定性（可选，同 M1 协议） |

**组件消融（2026-09-09 用户澄清定版 + 同日嵌套阶梯终版裁定，本协议的主对比轴；澄清原文："我不是要对比离散标签和连续标签，而是 carrier + activity 对于跨 session 能力的消融实验，各自有多少效果"；终版裁定："算 carrier 已经拿到 activity 信息，不用白不用"）**——**嵌套阶梯**贡献分解（信息包含：算 carrier 必须先读校准神经数据，activity 已到手——"有标签无身份"非现实部署场景，**无 CARRIER_ONLY 阶级**），基于 t4（离散方向标签基线），结构化定义冻结在 `dandi688_bench_v1/plan.COMPONENT_ABLATION`，全部在 **exp1_narrow exam 面**（0713/0714/0715/0716 四个跨日期 session——"跨 session 能力"的面；辅报 ts4 内容对照）判读（`plan.component_ablation_report`）：

```
FULL（满配）          = t4            E0 = activity + carrier side 熔炼；direct T 在
  ⊃ ACTIVITY_ONLY    = f_labelfree   E0 = 零 side 重熔（encoder 权重与活动内容保留）；T = 0
      ⊃ NONE（地板） = floor         E0 = 0（零张量）；T = 0 —— raw spike + 权重
附属：equiv_zero = 参数量对照（同参数量固定随机投影 seed42 不训练，作用于
      pre_pool 活动均值，零 side，无身份无标签、非零；T 形状保留值置零）；
      z0 = carrier-only 网格（E0 置零 + carrier 保留）降为附属探索；z_vstate_srcbank 降为附属
```

- **预注册分解（三项，exp1_narrow exam 面）**：**activity 独立贡献** = ACTIVITY_ONLY − NONE（f_labelfree − floor）；**carrier 标签增量贡献** = FULL − ACTIVITY_ONLY（t4 − f_labelfree）；**校准总价值** = FULL − NONE（t4 − floor）。算术律：前两项之和 = 总价值（嵌套阶梯）。附属读数（不进门）：equiv_zero 通路价值 = equiv_zero − floor、信息价值 = t4 − equiv_zero；z0 carrier-only 读数 = t4 − z0；f0 direct-carrier 边际 = t4 − f0。
- **f_labelfree vs floor**（易混披露）：f_labelfree 保留 E0 通路（冻结 encoder 权重 + 活动内容，仅标签 side 置零）——"activity 校准（无标签）能带来多少跨 session 能力"；floor 把 E0 整个置零张量 + carrier 置零——"零校准信息"的绝对地板。二者差 = E0 的纯 activity 成分（不含标签）的贡献。
- **z0 语义迁移**：2026-09-09 澄清把 z0 从旧"双零 Z_NONE"重定义为 carrier-only；旧语义归 floor；已存档 `train_exp1_narrow_z0`（旧定义训练）测的是今天的 floor；嵌套阶梯终版把 z0 与 z_vstate_srcbank 降为附属探索（不进预注册分解）。
- **equiv_zero 实现契约**：同 model class 下严格断言可训练参数量 == t4；随机投影参数量 == 被替换的 post_pool 通路；投影权重 SHA 记入 receipt；无任何训练后身份注入。实现：`src/dandi688_bench_v1/equiv_zero.py` + `build_vstate_cache.py --variant equiv_zero`（E0 熔炼落在 cache 字节）。

**vstate-688 系列（2026-09-09 用户指令增补；组件消融澄清后降级为**附属探索**，不再作为主对比轴。指令原文："立刻开始准备 vstate-688，注意消融——z 系列 = 完全不在新日期校准（bank 复用旧日期/source），f 系列 = 不使用任何标签校准（label-free）"）**：

| 臂 | carrier | 回答 |
|---|---|---|
| **vstate**（主臂） | M2 vstate4 配方适配 688：M10 同集 trial 的 R700/H300 窗口内 100ms 块，searchsorted 半开计数/块时长；状态 W=[softplus(v_x/rms_x), softplus(−v_x/rms_x), softplus(v_y/rms_y), softplus(−v_y/rms_y)]（v=cursor_vel 块均值，rms 拟合于各支持面自己的 R700 块总体、只用协议 train sessions；列归一 train-only）；Poisson 标准化（Δ=0.1s）+ n0=10 块收缩；读出 [a,c,m,b=meanₖR]（第 4 列按"最大对应"最终裁定与 M2 逐字同公式；δ_b=b−meanₖR_hold 降为 `vstate_b_hold` 消融变体，a/c/m 与主变体逐位相同，逐部件矩阵见 `CARRIER_M2_688_ALIGNMENT_MATRIX_20260909.md`） | 有符号速度状态在支持集不变（只有标签从方向换速度）时能否 ≥ t4+0.03；E0 同变重算 post_pool(cat(pre_pool 均值, vstate carrier)) |
| **z_vstate_srcbank**（z 系列消融） | 模型/训练同 vstate；exam session 的 bank（E0+carrier）在加载后被冻结的最近日期 train session bank 替换（exp1 全部 4 个 exam→2015-07-10；exp2 全部 6 个 val→2015-07-16；映射表与泄漏断言冻结在 plan.Z_SRCBANK_MAPS） | "完全不在新日期校准、只带旧 bank"的部署退化 = z − vstate |
| **f_labelfree**（f 系列消融） | carrier 全零；E0 = post_pool(cat(pre_pool 活动均值, 零 side))——ACTIVITY-ONLY identity（此前诊断缺失的干净消融；legacy f0 保留的是标签熔炼的冻结 E0，测不出该量） | "纯活动校准、无任何标签信息"的地板；vstate − f_labelfree = 标签信息总增量 |

**披露（vstate 消费 dense cursor_vel 校准标签，与 688 旧 sparse 纪律的对照）**：vstate 的状态标签来自 M10 支持 trial 的稠密 cursor_vel。这与 688 生产路径的 sparse-label 纪律（`materialize_sparse_event_t4` 明确不读 dense behavior）不是同一个问题：carrier 校准面按 FALCON 口径允许消费支持 trial 的行为标签（M2 侧 `PLAN_CARRIER_ITERATION_M2_688` §3.2 对 vstate4 有同款披露）。旧 688 dense-null（dense-speed CP-FiLM REAL<EMPTY）不构成对 vstate 的反证，三轴辨析：(1) 符号轴——null 用无符号速度分位数 profile，vstate 用有符号方向状态；(2) 位置轴——null 把 dense profile **叠加**在完整 T4 之上（加性修饰），vstate 是**替换** T4 坐上 carrier 席位（M10 支持集不变）；(3) 通路轴——null 走 CP-FiLM side 调制通路，vstate 走 carrier token 通路。M2 侧的 dense-profile null 同理（无符号 profile 叠加 T4，非有符号状态替换）。

**融合轴（与 carrier 臂正交，用户裁定 2026-09-09 原文："M1/H1 上 add 是必须的（维度问题）；M2/688 上 concat 和 add（joint）都要试"）**：T4/F0/TS4 全部保持 settled `proj_add` 前端（P: 50→16 加到 local，token_in=20）；T4-concat = 真 carrier 走 matched concat 前端（v2 `concat_model.py`：`[local16 | E0_50 | carrier4]`，token_in=70，init 为 proj_add 的函数保持折叠，几何/种子/数据合同全同）。动因：M2 的先验是 concat 略优（0.4501 vs 0.4016；RIFT 线 concat 0.3901 > joint 变体），688 与 M2 同宽 70，此臂直接检验该先验是否迁移。M1/H1 不开 concat 臂——E0 原始宽度（700/100）无法不经投影加到 16 个 local 通道上，add（proj_add）在该线是维度上的必须选择。

**M2 侧备忘**（不改 M2 代码）：M2 的 concat vs add 并测在 RIFT 主线已有部分证据（concat 0.3901 vs D42 joint 0.3478），完整双臂补测列入 RIFT M2 待办。

**验收门（预注册，val 面 equal-session mean R²）**：
1. T4 − F0 ≥ **+0.03** 且 T4 − TS4 ≥ +0.03（增量成立的双门——比照 pseudo-MUA 先例里两组比较全正的组级判据）；
2. RIFT+T4 对封存参照 **SPINT+T4 0.5750**（dev-6 同域）非劣（≥ −0.01）为"架构可迁移"判据；超过则为新冠军；
3. 过门后 formal test 一次性解封：报 T4/F0/TS4 三臂终局。
4. vstate-688 系列门（2026-09-09 增补，exam 面同口径；附属探索）：**vstate − t4 ≥ +0.03** 为增益门；z_vstate_srcbank − vstate（校准缺失代价）与 f_labelfree − vstate（标签信息总增量的负值）为消融读数，不设 pass/fail（`plan.vstate_gate_report`）。
5. 组件消融读数（2026-09-09 澄清定版 + 嵌套阶梯终版，主对比轴）：阶梯 FULL=t4 ⊃ ACTIVITY_ONLY=f_labelfree ⊃ NONE=floor 在 exp1_narrow exam 面经 `plan.component_ablation_report` 分解为三项预注册读数（activity 独立贡献 / carrier 标签增量贡献 / 校准总价值；前两项之和 = 总价值），不设 pass/fail；equiv_zero 参数量对照与 z0 carrier-only 为附属读数；ts4 同面内容对照为辅报。

> **ADDENDUM-CONDITION-AXIS（2026-09-10 用户指令：方案 B/C1/D1/D2 都要）**：新增**消融条件轴**——同一嵌套阶梯在不同数据/几何条件下的重测，全部只报读数、不进预注册门；对比报告 = `plan.ablation_condition_comparison`（基线冻结 exp1_narrow 读数 t4 0.8729 / f_labelfree 0.8257 / floor 0.3247）。判读问题：**哪个条件下 carrier 边际价值被放大**。
> - **B `exp1_poverty`**：train 只用最早 3 session（0629/0630/0701，~65k 窗），exam 面不变；纯协议过滤（`PROTOCOLS["exp1_poverty"]`），跑 t4/f_labelfree/floor。
> - **C1 `t4_dir16`**：方向设计 8→16 bin（22.5°），其余配方不动；`build_carrier_cache.py --variant t4_dir16` + `--arm t4 --prepared-cache`。**退化披露**：全数据集 target_dir 精确正则（68 session / 20089 trial 零偏差），16-bin 全落偶数 bin；实测 a/c/m 列与 u1_m10 逐位相同（半缩放被列归一化抵消），仅第 4 列 b 移动（≤0.083 归一化单位）；干净对比器 = u1_m10（0.8826，同族逐位同 a/c/m），vs 冻结 t4 的差异属估计器家族差。
> - **D1 `shortwin`**：`--model-override shortwin`，W 50→10（模型只看每窗最后 10 bin，查询目标逐位不变）；t4 + f_labelfree。
> - **D2 `shallow`**：`--model-override shallow`，temporal 4→1 层（同 50-bin 感受野，bench 包内子类 RiftShallowDecoder，不改 rift_v1 主包）；t4 + f_labelfree。override 进合同/receipt，train/score 同参 fail-closed。

## 4. 与外部队友线的边界

`btransform_unified_v1/handoff/dandi688_*`（v1/v2）已交外部队友（EVALAI_RECOMMENDATIONS 标 EXTERNAL_PENDING_INTERFACE）。本协议的本地基准与队友线独立：**不共用选点、不共用 test 消费记录**；若将来对表，双方 split/scoring 口径必须显式并列（688 RIFT 契约开头同款警示）。

## 5. 执行顺序（待用户批准后派单）

P0 数据/合同复核（prepared_cache_v2 已在盘；t4 与 t4_concat 的 CPU preflight 均已 PASSED 存 receipt）→ P1 四臂训练（T4/F0/TS4 主臂 + T4-concat 融合消融臂，GPU 一张，各 ~1h）→ P2 val 面判读（门 1/2；t4_concat 只报读数不进门）→（过门）P3 formal test 一次性解封三主臂 → P4 可选消融（E0-zero / support-resample）。receipt 全 seal；不改 688 RIFT 契约的任何冻结项。
