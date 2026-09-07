# HANDOFF: 电极空间先验（spatial-functional prior）

**日期：2026-07-31**
**给：下一位接手的 AI / 研究者**
**状态：CLOSED（2026-07-31 17:09 HKT）。Stage 0 免训练诊断已执行完毕并**判负**：`passing_low_budgets=[]`、`stage1_candidate=false`。§四 的 go/no-go 关口已触发 NO-GO，§九 的 Stage 0 停止条件已满足 → **Stage 1（`T4NBR`/`T4NBR_SHUF`/`T4CONCAT`）不实现、不启动，不占用 GPU**。本文档此后仅作为诊断记录与方法学参考，不再是待执行任务。判决详情见 §零。**

---

## 一句话

此前"electrode 信息无效"的结论，是在**没有几何、且功能已被 T4 充分测量、用减法/门控机制、在饱和 R² 上按 +0.03 增量门**这四重条件下得到的——它证明的是「静态电极身份加不进已饱和的功能测量」，**不是**「功能相似 unit 空间聚集」这个先验错了。要真正检验该先验，必须：(A) 让"邻近"可表达，(B) 把先验放到功能测量不可靠的低标签 regime，(C) 改用方差/收敛类判据。

> **后续（2026-07-31）：(A)(B)(C) 三项已全部落实并执行，先验被判负。** 因此本文档的诊断（旧结论测错了对象）依然成立，但对先验本身的结论已从"未被检验"变为"已被公平地检验并否定"。见 §零。

---

## 零、Stage 0 判决：先验在本数据上不成立（2026-07-31 17:09 HKT）

**权威 artifact**：`results/electrode_spatial_prior_stage0_v1/audit.json`（132 KB）。
**脚本**：`scripts/audit_electrode_spatial_prior.py` + `tests/test_electrode_spatial_prior_audit.py`（commit `2db8aa3`）。
**叙述**：`docs/CURRENT_RESULTS.md` §「2026-07-31 17:09 HKT：真 CMP 空间邻域先验 Stage 0 判负」。

### 0.1 两个门都失败

**(a) 空间相干门 —— 失败。** 相邻电极（Chebyshev `d=1`，排除自身电极）的 `(a,c)` 余弦相似度，减去**坐标置换零模型**（每 session 64 次抽样）后：

| 量 | 值 |
|---|---:|
| mean `d1_minus_permuted` | **+0.032323** |
| 为正的 session | **14 / 27** |
| bootstrap 95% CI | **`[−0.008420, +0.075942]`**（跨零） |
| `passes_spatial_coherence_gate` | **false** |

**(b) 收缩目标预测力 —— 失败，且方向明确。** train-only nested，future-rate MSE 比值（<1 为更好）：

| 对比 | M=10 | M=15 | M=20 |
|---|---:|---:|---:|
| **邻域均值 / 零**（本先验 vs 现状） | **1.14537**（0/27 改善） | **1.11938**（0/27） | **1.10413**（0/27） |
| 邻域 / **置换邻接** | 0.96143（19/27） | 0.96522（23/27） | 0.96783（23/27） |
| 零 / 不收缩（现有 `T4W3`） | 0.92830（26/27） | 0.95371（27/27） | 0.97248（23/27） |

字段为 `geometric_session_mse_ratio` / `sessions_improved`。`neighbor_vs_zero` 的
`passes_material_train_only_gate` 在三个预算上全为 `false`。

→ `passing_low_budgets = []`，`stage1_candidate = false`。

### 0.2 结论与边界

1. **在 Utah 阵列 400 µm 间距上，M1 方向调谐的空间局部性 ≈ 0。** 向空间邻域均值收缩在三个低标签预算上都**明确劣于**现有的向零收缩（0/27 session 改善），先验预测的"借邻居统计强度"反而注入了偏差。
2. **这是一次公平的反驳，不是又一次测量假象。** §一 列出的四重缺陷这次全部去掉了：有**真实且已双射校验的几何**（§2.1）、在**低标签 regime**（M=10/15/20）、用**收缩而非减均值/门控**机制、判据是**代理 MSE + 置换零模型**而非饱和 `+0.03` 增量门。因此可以写成阴性结论。
3. **仍存在一点微弱的真结构**：真邻接比随机置换邻接好 3–4%（0.961/0.965/0.968），所以并非"完全没有空间信息"——只是量级远小于噪声，且**收缩到零仍是更好的目标**。
4. **机制性诊断（为何如此）**：置换零模型下 `d=1` 相似度常与真实值相当（如 session `20150309`：实际 0.4064 vs 置换 0.4075），且各 session 的距离曲线异质（`20131022` 随距离衰减 d1=0.285→d2=−0.097，而 `20131003` 随距离**上升** d1=0.229→d7=0.356）。说明该阵列上占主导的是**全局**调谐相干性，而非空间局部相干；坐标置换零模型正确地把纯空间分量隔离出来，而该分量 ≈ 0。
5. **隔离纪律成立**：只打开 27 个 train NWB，`validation_session_nwb_opened=false`、`formal_test_session_nwb_opened=false`，未触碰 formal receipt。

### 0.3 对下游的处置

- **不实现** §4.3 的 `target` 参数改动、`neighbor_index` buffer、`T4NBR`/`T4NBR_SHUF`/`T4CONCAT` 三臂（§六 第 2–7 项全部作废）。
- 低标签 T4 主线继续沿**现有 shrink-to-zero（Wiener λ=3）** 推进，其 train-only proxy 仍是 27/27 改善。
- §五（判据）与 §3.2（位置只经邻接图进入、坐标永不 concat）作为**方法学结论保留**——它们是通用的，独立于本次先验判负。

---

## 一、背景与核心诊断（为什么旧结论测错了对象）

用户的先验是**空间-功能平滑性**：功能相似的 unit 极大概率落在**同一或邻近** electrode。这是一个明确的神经生理先验（皮层功能柱结构）。当前工程从未真正测过它，原因有四：

### 1.1 数据里没有几何，"邻近"不可表达
- NWB 电极表只有 `bank`(A/B/C) + `pin`(1–32) + `label='elecM1bankApin*'`，**没有 x/y/z 坐标**（`docs/UNIT_SIDE_FEATURE_ABLATION.md:61-82` 已记录）。
- 实测确认（2026-07-31，本轮）：sub-C 单块 M1 阵列 96 电极，bank A/B/C 各 32 pin；**所有 session 的 bank/pin/label 完全一致**（跨 session 几何可复用）；sub-J 也是同样结构。
- `bank+pin` 是 **Blackrock 连接器接线序号，不是空间坐标**。Utah 阵列物理上是 10×10、400 µm 间距的规则网格，但 `(bank,pin) → (row,col)` 的映射由该阵列的 `.cmp` 决定。**该 `.cmp` 此后已获得并验证，见 §2.1**（诊断当时仓库内没有）。
- 结论：代码里 electrode 只是**类别整数 id**，`neighbor / distance / adjacency / 坐标` 在整个 pipeline 中不存在。**先验里最关键的"nearby"分量，一次都没被编码，更别说被检验。**

### 1.2 已跑的三个机制都不是"邻域功能相似"的形态

| 机制 | 变体/token | 形态 | 判据 | 结果文件 |
|---|---|---|---|---|
| 学习式 per-electrode embedding | `B3S`/`t4e`,F3 | 全局共享静态 8 维表 | **从未跑**（`results/electrode_ablation_f3/` 不存在） | — |
| reliability gate | `B3SEG`/`t4gate` | per-electrode 标量乘子 | **ineffective** `T4GATE−T4=−0.0108±0.0049` | `results/t4_gate_screen/aggregate.json` |
| same-electrode relation | `B3SER`/`t4rel` | 仅**同电极**等价分组，减组均值 | **ineffective** `REL−T4=−0.00144±0.00088` | `results/sua_electrode_relation_full_v1_scheduler/multiseed_strict_aggregate.json` |

三者没有一个能表达"邻近电极功能相似"：embedding 学的是"某电极本身特殊"、gate 只调可靠性幅度、relation 只有"same"没有"nearby"且**减均值恰好抹掉了先验预测的共享信号**。

### 1.3 冗余混淆（根因）
- T4（每 session 直接重测的余弦调谐，`_fit_cosine_tuning` @ `mc_maze/unit_side_features.py:589-616`）本身极强：`T4−B0≈+0.34`，`T4−TS4≈+0.29`。
- 电极身份对功能的价值，本质是"**用位置预测功能**"的先验；但功能已被 T4 直接测出来。**在功能被高质量测量时，位置先验的边际信息量≈0，这是必然的，不是意外。**
- 所有电极实验都在**已饱和的 T4 之上做加法**（`identity+anchor`、`identity*gate`，且全部 zero-init）。teacher 冻结、T4 饱和 → 加法项拿不到梯度信号 → 判 ineffective 几乎是设计决定的。
- **先验只在功能测量不可靠时才有用**（校准 trial 极少、低标签、unit 掉线、跨天漂移）。旧实验恰恰在功能测量充分的常规 regime 里测，把先验的用武之地排除掉了。

### 1.4 反证：数据本身支持先验
- 多 unit 同电极占比高：6 个 val session 为 **38–61%**（`docs/CURRENT_RESULTS.md:1497`）。
- pseudo-MUA（把同电极多 unit 求和成一个通道）后 **T4 增益依然保留**（`T4−F0≈+0.32`，`§K.1`）。这正面说明**同电极单元功能相干**——与先验一致。但 relation 实验用"减组均值"去测，把这个正信号读成了残差噪声。

---

## 二、CMP 排查结论（用户第 1 问）

### 2.1 CMP 已找到并验证 ✓（2026-07-31）
- **来源**：Miller/Limblab 公开仓库 `github.com/limblab/deprecated_limblab_analysis`，路径 `lib/Map Plotting/Maps/*.cmp` + 索引 `lib/Map Plotting/Mapfile_repo.m`。
- **锁定 sub-C 的阵列**：`Mapfile_repo.m` 明确 `Chewie.M1 = '1025-0394'`。DANDI 000688 的 **sub-C 即 Chewie**，故其 M1 阵列图为 `1025-0394.cmp`（文件头 "Cerebus mapping for array 7J / SN 0394"）。
- **已下载到本地**：`data/array_maps/Chewie_M1_1025-0394.cmp`（96 条数据行）。
- **已交叉验证**：解析出的 96 个 `(bank,pin)` 与本地 NWB 电极表的 `(bank,pin)` 集合**完全双射匹配**（无缺、无多）；坐标铺满 10×10 网格；8-邻域平均每电极 6.88 个邻居（角 4 / 边 6 / 内 8）。→ **这就是真几何，可直接用。**
- **CMP 文件格式**（`c r b e l`）：`c`=0基列(左→右)，`r`=0基行(下→上)，`b`=bank(A/B/C/D)，`e`=bank 内 1基 pin(1–32)，`l`=Central 标签。NWB 的 `label='elecM1bankApin1'` 提供 `(bank,pin)`，据此查 cmp 得 `(col,row)`。解析/绘图参考 `ArrayMap.m`：`elec_num=(hex2dec(bank)-10)*32+pin`。

### 2.2 空间信息是否应跨数据集一致 —— 要分两层，这点很关键
1. **几何/拓扑（谁和谁相邻）：可跨数据集一致，可复用。**
   - Utah 阵列物理网格对所有阵列都是 10×10 / 400 µm，通用。
   - 但**每块阵列的 `(bank,pin)→(col,row)` 接线是阵列专属的**（cmp 按序列号一一对应，`Mapfile_repo.m` 里 Chewie/Mihili/MrT 各有不同 SN）。sub-C/J/M 的 bank/pin 命名方案相同，但**不能假设它们共用同一张 cmp**——每个 subject 要用自己阵列的那张：Chewie=`1025-0394`、MrT.M1=`6250-0896`、Mihili.M1=`6250-000989`（均在该仓库 Maps 目录，sub-J/Jaco 若需另找）。
2. **功能映射（某电极编码什么）：不可跨植入一致，不可复用。**
   - 每次植入采样不同的皮层组织/神经元。"electrode 37"在两块阵列里含义无关（`docs/ELECTRODE_ANCHOR_DESIGNS.md:182-200` 已指出）。
   - **推论（对设计至关重要）：几何应作为「关系型 / 正则化先验」使用（每阵列一张 cmp 决定邻接图），绝不能作为「身份查找表」使用（不可迁移）。** 这也解释了为何静态 per-electrode embedding 注定不 work。

### 2.3 复现命令（已完成，供审计）
```bash
# 下载 Chewie(sub-C) M1 array map
mkdir -p data/array_maps
curl -s "https://raw.githubusercontent.com/limblab/deprecated_limblab_analysis/master/lib/Map%20Plotting/Maps/1025-0394.cmp" \
  -o data/array_maps/Chewie_M1_1025-0394.cmp
# 索引文件（monkey→SN）：lib/Map Plotting/Mapfile_repo.m
# 解析器参考：lib/Map Plotting/ArrayMap.m  (格式 'c r b e l')
```
其它 subject：MrT.M1=`6250-0896.cmp`、Mihili.M1=`6250-000989.cmp`（同目录）。

### 2.4 用真几何做的第一步验证（防止假阴性，强烈建议先跑）
map 已确认与 NWB 双射，但**先验本身**仍需一个便宜、无需训练的 sanity check：
- **邻居功能相干性检验**：把每个 unit 的 T4 向量映到其电极坐标，计算「8-邻域电极对的 T4 相关」vs「随机电极对」，跨多个 train session 聚合。若用户先验为真,前者应显著更高。这既校验几何正确性,又**直接量化先验强度**,决定方向 A/B 是否值得投入。
- 若某 subject 找不到 cmp，退回 §3.2 的 data-driven 功能 kNN 邻接。

---

## 三、最新实验结果（2026-07-31）与 concat 分析 —— 引用前必读

### 3.1 新结果把我们正好推到了先验该起作用的格子
三条新证据方向高度一致（`docs/CURRENT_RESULTS.md`）：

1. **主线已收缩到低标签 T4**（`CURRENT_RESULTS.md:8`）。`decoupled v2` 被 kill，现在就在低校准量 regime 上做——正是本 handoff 预测"先验唯一能赢"的地方。
2. **T4@15 − T4@50 = −0.0588，6/6 validation session 全降**（`:23-30`，`results/sua_t4_shrinkage_m15_v1/t4_m15_s42.json`）。把校准从 50 trial 砍到 15，T4 掉约 0.06 且未过 −0.03 非劣门。→ **低标签下存在一块真实、可回收的 deficit**，这就是先验要补的目标。
3. **Wiener 收缩在 train-only proxy 上 27/27 session 改善**（`:207-229`，M15 MSE ratio 0.954）。**但它收缩的目标是零**：见 `uncertainty_wiener_shrink_t4` @ `mc_maze/unit_side_features.py:689-740`，`shrunk[:, :2] *= factors`，`factor = signal/(signal+λ·uncertainty)`，**没有任何电极/邻居/空间项**，等价于"低置信就假设该 unit 无调谐"——**会丢信息**。

**关键机会**：现有 `T4W3`（shrink-to-zero）是"向邻域均值收缩"在 `neighbor_mean=0` 时的退化特例。把收缩目标从零换成**空间邻域均值**即为严格推广，而四臂框架、Wiener 因子、聚合器、判据都现成（`scripts/audit_t4_confidence_shrinkage.py`、`scripts/aggregate_sua_t4_shrinkage.py`）。核心科学问题因此变得极干净：**"向邻域收缩" 是否打得过 "向零收缩"？**

> **答案（2026-07-31，§零）：不能。** 邻域/零 的 future-rate MSE 比值在 M=10/15/20 分别为 1.145/1.119/1.104，0/27 session 改善。该问题的提法是对的，答案是负的。

### 3.2 为什么 electrode 位置**不应该直接 concat**（这是设计原则，务必遵守）
1. **concat = 身份表形态，已被证伪**。把 `(col,row)` 坐标或学习式 per-electrode embedding 拼进特征，模型只能记住"电极 k 有个固定偏移"——**阵列专属、不可迁移、无平滑归纳偏置**。这正是 `t4e`(从未跑)、gate、anchor、relation 全部 ineffective/未跑的形态（§1.2）。
2. **加参数 = 加方差，方向相反**。低标签 regime 的诉求是**降方差**；concat 引入可学习参数只会加方差，与目标背道而驰。
3. **坐标绝对值本身没有信息**。有信息的是**相对位置（谁与谁相邻）**。concat 两个坐标数并不告诉模型"邻居功能相似"——该先验必须通过**邻接结构**注入。
4. **正确用法**：位置只经由**邻接图**进入（决定"谁向谁借统计强度"），作为**关系型先验 / 平滑正则 / 收缩目标**；坐标值**永不**作为模型输入。这与 §2.2 自洽——几何(邻接)可迁移、身份(坐标查表)不可迁移。
5. concat 仍应作为**一个故意的证伪对照臂**跑（见 §4.3 `T4CONCAT`），用实验结果实锤"concat 形态不如邻域收缩"，而不是只在文档里断言。

---

## 四、实验设计：Stage 0（免训练诊断）+ Stage 1（低标签 A/B）

> **执行状态（2026-07-31）：Stage 0 已执行 → NO-GO（§零）。Stage 1 依设计不启动。** 以下保留原始设计文本，作为该判决所依据的预注册记录，勿据此开工。

总体假设：per-session 校准 trial 越少，per-unit T4 拟合噪声越大；此时"向**空间邻域**均值收缩"比"向零收缩"更能降方差、且不丢信息，故先验价值随校准量下降而上升。

### 4.1 Stage 0 —— 免训练判定诊断（便宜、决定性，必须先跑）
cmp 已验证（§2.1），可直接用真几何回答"先验成不成立 + 邻域收缩能否赢"，**无需任何训练**。严格只用 train sessions，test 封存。
- **邻域相干性 + 距离衰减曲线**：把每个 train session 的 per-unit T4（复用 `_fit_cosine_tuning` @ `unit_side_features.py:589-616`）映到电极坐标（`Chewie_M1_1025-0394.cmp`），计算 T4 向量相关随 Chebyshev 距离的衰减：d=1(8-邻域) vs d=2,3… vs 随机对。**先验成立 ⟺ 单调衰减且 d=1 显著高于随机**；据此选邻域半径。
- **收缩目标预测力（直接预判 A/B 胜负）**：train session 内 nested LOSO，比较用**邻域均值** vs **零**（现有 Wiener 目标）预测留出 trial `[M:50]` tuning 的 future-rate MSE。谁更低就预示 Stage 1 谁赢——无需上 GPU。
- **可交付脚本**：新建只读诊断 `scripts/audit_electrode_spatial_prior.py`，输出衰减曲线 + 两种收缩目标的 MSE 对比 JSON。产物 SHA 绑定，`formal opened=false`。
- **停止条件**：若 d=1 相关与随机无实质差别、且邻域均值不优于零，则先验在本数据上不成立，**不进 Stage 1**（此时才是对先验的真正否定）。

### 4.2 Stage 1 —— 低标签 regime 的决定性 A/B（Stage 0 为正才启动）
在已确认的 `M_T4 ∈ {10,15}`（外加 `{20,30,50}` 作为交互斜率的高端锚点），沿用 `configs/subc_co_27_6_strict_train_val_manifest.json` 的 27/6/6、seeds 42/43/44、`M_activity=30`、共同 eval start=50、12 epochs、与 T4@50 reference 比较。**test 6 session 封存（只读行数）**。臂：

| 臂 | 收缩目标 | 作用 |
|---|---|---|
| `T4`（ordinary） | 无 | 基线（已知 @15 掉 0.059） |
| `T4W3`（现有） | **零** | 现成 shrink-to-zero baseline |
| **`T4NBR`（新）** | **空间邻域均值** | 用户先验 |
| `T4NBR_SHUF` | 置换邻接后的"邻域"均值 | 隔离"空间结构" vs "收缩本身"（**必须有**） |
| `T4CONCAT` | —（把 (col,row) 拼进 side-feature） | **证伪对照**，预期不优于 `T4NBR`（§3.2） |

- 每臂都要有对应的 aligned/shuffled 内容门（`TS4`/`TS4NBR`），沿用现有四臂协议。

### 4.3 机制形态与挂载点
收缩式，天然满足退化下界：
```
t_i_shrunk = (1 - w_i) · mean_{j∈N(i)} t_j  +  w_i · t_i        # w_i = 现有 Wiener 因子
```
- `w_i` 直接复用 `uncertainty_wiener_shrink_t4` 的 per-unit 因子（低 SNR → 更靠邻域）。
- **`neighbor_mean = 0` 时精确退化为现有 `T4W3`**；单例电极/无邻居 → `N(i)=∅` → 退化为 `T4`。保证"不劣于基线"的下界。
- **最小改动实现**：给 `uncertainty_wiener_shrink_t4` 增一个 `target` 参数（默认 `0`，即现状），`shrunk[:, :2] = w*t + (1-w)*target[:, :2]`；`target` 由邻接图对 `a,c`(可含 `m`) 做邻域平均得到。
- **邻接图来源**：从 `data/array_maps/Chewie_M1_1025-0394.cmp` 解析 `(bank,pin)→(col,row)`，构 8-邻域（或 Stage 0 选定半径）邻接，注册为**每阵列一份静态 buffer**，不进可学习参数（§2.2：几何可迁移、身份不可）。`electrode_ids` 已在 batch tuple `(neural,behavior,calib,session_name,side,electrode_ids)`（`train_variant_dandi688.py:407-445`）；`neighbor_index` 走同一通道。
- 挂载在 **ψ/finalize_identity 阶段**，不碰 φ；`SameElectrodeRelationEarlyPoolEncoder`（`streaming_encoders.py:760-848`）的 `_segmented_mean` 可参考，但**改为收缩而非减均值**。

### 4.4 无 CMP 时的回退（仅其它无 map 的 subject 需要）
sub-C 已有真几何。若换到找不到 cmp 的 subject：同电极组用 `pool_trial_rates_by_electrode` @ `unit_side_features.py:396-418`；或用 **train session** 电极间 T4 相关构 kNN 功能邻接（纯 train 拟合防泄漏），作为 proxy 邻域先跑通机制。

---

## 五、方向 C（用户第 4 点）：改判据 —— 先验/正则类效应不能只看饱和 +0.03 门

现协议（`docs/MEASUREMENT_PROTOCOL_V4.md`）的 `+0.03` 硬门 + 配对 `σ_delta` 是为**独立增量特征**设计的。空间先验是**去噪/正则**类作用，在高信噪比（大校准量）下天然趋于 0，用同一把尺子必然判死。需**补充**（不是取代）以下判据：

1. **regime 交互斜率**（主判据）：拟合 `delta(n_calib)` 曲线，检验其**随 n_calib 递减的斜率是否显著为负**、且在小 `n_calib` 处 `mean − 2σ_delta_paired > 0`。这直接检验"先验在低数据处有用、在高数据处无用"的假设，而非要求全 regime 都过 +0.03。
2. **方差 / 稳定性判据**：在低 `n_calib` 格子，比较有/无先验的**跨 seed 方差 σ_seed** 与 **窗口内 σ_run**。先验作为收缩器应**降低方差**——报告 `σ` 的比值与其不确定度，而不仅是均值 delta。
3. **收敛速度**：固定 12 epoch 预算内，比较达到目标 R² 的 epoch 数 / 早期 epoch 的 R²（收缩先验应更快稳定）。
4. 沿用 V4 的硬性纪律：独占 run 目录（M1）、固定 epoch 预算关 early-stopping（M2）、epoch 5–12 平均（M3）、**配对** `σ_delta_paired = stdev(逐seed delta)/√n_seeds`（`MEASUREMENT_PROTOCOL_V4.md:160-173`）、四态结论 `effective/effective_heterogeneous/ineffective/indeterminate`。
5. **诚实边界**：`σ_seed` 主导（≈0.0385），要在 2σ 分辨 +0.03 约需 13 seed（`MEASUREMENT_PROTOCOL_V4.md:82-113`）。因此本方向**优先用交互斜率与方差比**作为证据，单点 +0.03 门作为辅助；若只能跑 3 seed，不得把"未过 +0.03"写成"先验被否定"。

---

## 六、实现清单（落地顺序）

> **执行状态（2026-07-31）：第 1 项已完成（`scripts/audit_electrode_spatial_prior.py`，commit `2db8aa3`），其结论为 NO-GO；因此第 2–7 项全部作废，不实现。** 以下保留原文以记录当时的预注册范围。

1. **Stage 0 免训练诊断（先做）**：新建只读 `scripts/audit_electrode_spatial_prior.py`：解析 `data/array_maps/Chewie_M1_1025-0394.cmp` → `(bank,pin)→(col,row)` → 邻接；用 `_fit_cosine_tuning` 出每 train session 的 per-unit T4；输出 (a) T4 相关随 Chebyshev 距离衰减曲线，(b) 邻域均值 vs 零 的 future-rate MSE 对比。严格只用 train，`formal opened=false`，产物 SHA 绑定。**这是 go/no-go 关口。**
2. **邻接图 buffer**：从上述 cmp 生成 `neighbor_index`（每阵列一份静态表），供 Stage 1 复用；不进可学习参数。
3. **收缩机制最小改动**：给 `uncertainty_wiener_shrink_t4` @ `unit_side_features.py:689-740` 增 `target` 参数（默认 `0`＝现状 `T4W3`），`shrunk[:,:2]=w*t+(1-w)*target[:,:2]`；`target` 由 `neighbor_index` 对 `a,c`(可含 `m`) 做邻域平均。`neighbor_mean=0` 精确退化为 `T4W3`。
4. **新增臂 + 对照**（复用现有 shrinkage 四臂 harness `scripts/audit_t4_confidence_shrinkage.py` / `scripts/aggregate_sua_t4_shrinkage.py`）：`T4NBR`（向邻域收缩）、`T4NBR_SHUF`（置换邻接）、`T4CONCAT`（坐标 concat，证伪对照），各配 aligned/shuffled 内容门。加对应 token（`unit_side_features.py` validity gate）+ `train_variant_dandi688.py` 的 `--variant` 分支。
5. **聚合器**：在 `scripts/aggregate_sua_t4_shrinkage.py` 增加 `T4NBR−T4W3` 增量、regime 交互斜率（delta 随 M 递减）与 σ 比值字段。
6. **校准量**：现有 shrinkage runner 已支持 `M_T4=15`（`results/sua_t4_shrinkage_m15_v1`）；扩到 `M∈{10,15,20,30,50}` 以画交互斜率。
7. 先跑 sub-C 的 `M × arm × 3 seed` 网格；确认机制与判据后再考虑 pseudo-MUA / 其它 subject（各用自己的 cmp，§2.2）。

---

## 七、关键文件索引

- **Stage 0 判负产物（本方向的最终结论）**：`results/electrode_spatial_prior_stage0_v1/audit.json`；脚本 `scripts/audit_electrode_spatial_prior.py`、测试 `tests/test_electrode_spatial_prior_audit.py`（commit `2db8aa3`）；阵列图 `data/array_maps/Chewie_M1_1025-0394.cmp`（SHA-256 `d5fb7d3e…891d`）
- 电极 id 提取 / 池化：`mc_maze/multisession_datamodule.py:216-247`
- T4 拟合 / T4 side-feature：`mc_maze/unit_side_features.py:589-616`（`_fit_cosine_tuning`）、`:396-418`（`pool_trial_rates_by_electrode`）、`:1299-1326`（`load_session_electrode_ids` / `compute_electrode_vocab_size`）、`:94-249`（token 注册与 gate）
- 现有电极机制（改造起点）：`streaming_calibration_exp/src/models/components/streaming_encoders.py:364-457`(embed) / `:594-665`(gate) / `:678-751`(anchor) / `:760-848`(relation)
- 训练 / batch 布线：`sua_exploration/scripts/train_variant_dandi688.py:407-445, 617-708`
- 评价协议实现：`scripts/eval_epoch_window_dandi688.py`、`scripts/select_gradient_free_protocol_dandi688.py`
- 判据：`docs/MEASUREMENT_PROTOCOL_V4.md`（四态、配对 σ、+0.03 门）
- 旧电极阴性结果：`results/t4_gate_screen/aggregate.json`、`results/sua_electrode_relation_full_v1_scheduler/multiseed_strict_aggregate.json`；叙述见 `docs/CURRENT_RESULTS.md §K.3/K.5`、`docs/ELECTRODE_ANCHOR_DESIGNS.md`
- 数据事实（无坐标 / bank+pin）：`docs/UNIT_SIDE_FEATURE_ABLATION.md:61-91`

---

## 八、数据隔离（不变，务必遵守）

- 只用 train + validation session；**6 个 test session 的 spike/behavior/trial 一律不加载**，只允许读 test NWB 的 unit-table 行数以固定 `N<100` regime。
- 不创建 / 不修改 / 不删除 formal-test receipt。
- 本 handoff 下的一切结果均为 validation development evidence，不构成 formal held-out 结论。

---

## 九、风险与停止条件

- **假阴性风险**：错误 `.cmp` 会毁掉真几何实验。本轮已验证 `Chewie_M1_1025-0394.cmp` 与 NWB 双射（§2.1）；换 subject 时对新 map 必须重跑该校验。
- **Stage 0 关口 —— 已触发 NO-GO（2026-07-31）**：`d=1` 相干性减坐标置换零模型后 CI 跨零，且邻域均值预测力在 M=10/15/20 上均劣于零（0/27 改善）。两条判负条件同时满足 → 先验在本数据不成立，**不进 Stage 1**。这构成对先验的**真正否定**（而非"不确定"），因为四重旧缺陷已全部去除（§0.2）。
- **Stage 1 停止条件 —— 已失效（moot）**：Stage 1 未启动，故其停止条件无需评估。原文如下，仅供记录：若在最低 `M∈{10,15}` 下，`T4NBR−T4W3` 相对置换对照 `T4NBR_SHUF` 仍 `mean+2σ ≤ 0` **且** 方差无下降 **且** 交互斜率不显著，则空间邻域先验不优于现有 shrink-to-zero，停止该路线。
- **预算风险 —— 已规避**：Stage 0 用免训练代理在 0 GPU-hour 内给出判决，未消耗 seed 预算。这验证了"先做便宜的 go/no-go 诊断再上 GPU"这一流程本身是有效的，建议后续先验类方向沿用。
- **重开条件（若将来要复活该方向）**：需满足至少一条 —— (a) 换到电极间距更小或有明确功能柱证据的阵列/皮层区；(b) 用比 `(a,c)` 余弦更贴切的功能相似度定义并重跑 §零 的相干门；(c) 有直接证据表明本次的 per-electrode 平均或半径选择引入了偏差。仅"感觉先验应该成立"不构成重开理由。

