# E3/E4 章程：调谐特征与 encoder 架构变体

**状态：章程已冻结，门槛待填（2026-07-26 建立）**
**估计量、不确定度与三态判定遵循 [`MEASUREMENT_PROTOCOL_V4.md`](MEASUREMENT_PROTOCOL_V4.md)**

## 0. 门槛为什么留空

本项目**两次**把门槛设在噪声底之下：

| 轮次 | 门槛 | 实测噪声底 | 后果 |
|---|---|---|---|
| `attention_arch_screen_v3` | `+0.005` | `σ_epoch = 0.0388` | 结论撤回 |
| `side_feature_ablation_v2` | `+0.03` | `2σ_delta = 0.048–0.077` | 全部 `indeterminate` |

第二次的直接原因是我在协议里写了 `σ_variant = σ_run/√3`，**漏掉了 `σ_seed`
这个主导项**。

**因此本章程的数值门槛留空，必须在 E1/E2 产出以下两个量之后才填写：**

1. **E2 给出的收敛 epoch 预算** —— 决定训练配置；
2. **E1 给出的 SWA 后 `σ_seed`** —— 决定可分辨的最小效应。

填写规则：门槛取 `max(2σ_delta_实测, 部署相关性下限)`。在两者填入之前**不得
启动 E3/E4 的正式 screen**。这条规则本身是预注册的，不可事后放宽。

## 1. E3：方向调谐特征

### 1.1 主张

`E_i` 被加到 `Z` 上、喂给 cross-attention 决定每个 unit 的解码权重，因此它
需要编码**功能属性（tuning）**。中心外向任务的每个 trial 都有已知
`target_dir`（8 个方向，间隔 45°；实测前 30 个 rewarded trial 覆盖全部 8 个），
而**当前 encoder 把它整个丢弃**——`_build_calib_trials` 只产出 binned spike
counts。

这是目前唯一大块未被利用的功能信号，且**每 session 现算，天然不漂移**——
正是波形特征栽跟头的地方。

### 1.2 特征组

全部只使用 calibration pool 内的 trial（`pool_size=50`，与既有侧信息同一边界，
复用 `calibration_pool_end_time`）：

| 组 | 内容 | 维度 |
|---|---|---:|
| `t4` | 余弦调谐拟合 `rate(θ)=b+m·cos(θ−φ)` → `[m·cosφ, m·sinφ, m, b]` | 4 |
| `t8` | 逐方向平均发放率向量（固定的跨 session 一致方向顺序） | 8 |
| `ts4` / `ts8` | 同维度 unit 轴置换对照 | 4 / 8 |

复用既有的 `side_features` 通道（`B3S` 在 ψ 输入端 concat），不新建管线。

### 1.3 ⚠️ 这会改变方法的主张

当前卖点是「identity **只从 spike 统计**得出」。加入调谐后变成「identity 来自
一个**有监督的标定块**」。

这仍然是 gradient-free、部署现实的——标定本来就是让受试者按提示做已知方向的
运动——但**假设更强了**。任何引用 E3 结果的文档必须显式声明这一点，不得含糊
带过。

另需诚实记录：余弦调谐拟合本质上就是经典 BCI 的 population vector / OLE 在算
的东西。区别在于我们不用它直接解码，而是把它作为 per-unit 描述子交给
transformer。**若 E3 有效，必须做一个「直接用调谐做线性解码」的对照**，否则
无法排除"收益其实来自经典方法而非架构"。

### 1.4 归一化

`m`、`b`、`t8` 的 8 个分量是率量纲 → 用 **train session 统计量**做 z-score
（沿用 §6.1 契约）。`m·cosφ`/`m·sinφ` 与 `m` 同尺度，同样用 train 统计量，
但不得分别归一化 `cosφ`/`sinφ` 以致破坏该向量的几何含义。

退化 unit（pool 内无 spike、调制为零、出现的方向少于 2 个）用固定填充值并
**计数记入 metadata**，禁止产生 NaN/inf。

## 2. E4：encoder 架构变体

两者都在 session-rate 路径，**计算**成本近乎免费（见
[`ASIC_DEPLOYMENT_CHARTER.md`](ASIC_DEPLOYMENT_CHARTER.md) §3），但参数量影响
weight SRAM，**且 state 占用不一定免费**——见下表实测。

### 2.0 实测成本（`N=64, T=100, M=30`，2026-07-26 独立复核）

| 变体 | 参数 | MAC/session | support state | 备注 |
|---|---:|---:|---:|---|
| B3 | 18,034 | 13,017,088 | **16,384 B** | 基线；流式累加，只需一份 buffer |
| `B3T` | **12,402** | **4,507,648** | 16,384 B | 参数 −31%，MAC **−65%** |
| `B3A` | 18,099 | 13,139,968 | **491,520 B** | 参数与 MAC 几乎持平，但 state **×30** |

**`B3A` 的关键代价不是算力而是 SRAM。** 它必须同时持有全部 `M` 个 trial 的
per-unit 特征才能在 trial 轴上做 attention，因此**破坏了 B3 的流式累加性质**
（B3 只需维护一个 running sum）。`16 KB → 480 KB` 对片上 SRAM 是实质差别。

这一点在写本章程时未预料到，须计入判读：**即使 `B3A` 在 R² 上占优，也要
与这 30 倍 state 代价一并权衡**；若增益不显著，`B3T`（更省且不破坏流式性质）
是更符合项目定位的方向。

`B3T` 的 `temporal_basis` 已核实为**非持久 buffer**（不在 `state_dict` 中、
不可学习），可学习参数仅 `basis_proj` + `post_pool`。

### 2.1 `B3T` —— 时间基 φ

把 φ 的 `Linear(100→64)`（6,400 参数）换成**固定**的 raised-cosine 时间基
投影 + 小的学习线性层：`[T=100] --固定基--> [K=12] --Linear--> [64] --ReLU`
（768 参数）。基函数注册为 buffer，不是可学习参数。

理由：原始 100→64 投影容易过拟合 session 特有的时序细节；平滑基更参数高效，
跨 session 迁移可能更好。

### 2.2 `B3A` —— trial 轴 attention

把对 M 个标定 trial 的 mean-pool 换成**在 trial 轴上的 attention**（逐 unit）。

> **这与 B15 是不同的假设，绝不可混为一谈。** B15 在 **neuron 轴**上做
> attention（跨神经元关系），其证据已于 2026-07-25 撤回。`B3A` 在 **trial 轴**
> 上做 attention，问的是"哪些标定 trial 对这个 unit 更有信息量/更可靠"。
> 后续任何分析都必须保持这个区分。

实现约束：现有流式 API 在 `push_trial` 中增量累加 `sum_feat`，与 trial 轴
attention 不兼容，需在 state 中保留逐 trial 特征。**不得修改抽象方法
`reset_stream`/`push_trial`/`finalize_identity` 的签名**——约 25 个同级
encoder 依赖它们。

## 3. 数据隔离（与既有 screen 相同）

- 只用 train + validation sessions；
- **6 个 test session 的 spike/behavior/trial 一律不加载**，只允许读 test NWB
  的 unit-table 行数以固定 `N < 100` regime；
- 不创建、不修改、不删除 formal-test receipt；
- 泄漏纪律：特征只能用 calibration pool 边界之前的数据，且必须有**行为测试**
  （改 `pool_size` 必须改变特征值）。只测断言函数本身的写法**不被接受**——
  该反模式已在此前审核中被否决过一次。

## 4. 运行矩阵（门槛填好后执行）

seeds 由 E1 的 `σ_seed` 结论决定；若 SWA 未能显著降低 `σ_seed`，则必须相应
增加 seed 数，而不是降低门槛。

| screen | 组 |
|---|---|
| E3 | `F0`(B3) / `t4` / `t8` / `ts4` / `ts8` |
| E4 | `B3` / `B3T` / `B3A` |

E4 的对照就是 B3 本身（同 seed、同预算），不需要置换对照——架构变体没有
"内容 vs 宽度"的混淆问题。

## 4b. 电极身份（原 `F3`）的设计修订（2026-07-27）

### 4b.1 前提变了：基线改为 T4

E3 的结果（[`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §J.5）使这个问题的前提失效。
电极假设原本的动机是"提供跨 session 稳定锚点"，但**从标定 trial 现算的调谐
已经供给了电极先验所能供给的大部分，而且更新鲜**。

因此主对照从 `电极 vs B3` 改为：

> **`T4 + 电极` vs `T4`** —— 在已有实测调谐的前提下，电极身份还能补充什么？

拿电极去打 B3 基线测的是一个已不再关心的问题。

### 4b.2 原 `F3` 实现的缺陷

已实现的 `f3`/`fs3` 把 **F2 波形标量**与电极 embedding 捆绑
（`base_feature_group("f3") -> "f2"`，`post_pool_side_dim = 6 + 8 = 14`）。
这忠实执行了本章程 §6 的原始定义——但那条定义写于 **F2 被测试之前**。
F2 现已是确定的负面项，捆绑使 `f3 − F0` 混淆电极与波形。
（`f3 − fs3` 的配对设计本身是对的，因为 FS3 只置换电极。）

**结论**：新增**无波形**的电极变体，原 `f3`/`fs3` 保留不删。

### 4b.3 三种映射设计

全部要求**初始化时与纯 T4 逐位等价**，且各配**同维度置换对照**
（电极 id 在 session 内打乱，seed 记录在案）。

| 设计 | 形式 | 机制 | 参数 |
|---|---|---|---:|
| **D 可靠性门控** | `E_i ← E_i · (1+tanh(g[e_i]))`，`g` init 0 | 编码"该记录位点可信度"（慢性噪声、阻抗漂移、坏道），**与调谐正交** | 96 |
| **C 锚定先验** | `E_i ← ψ(·) + α·M[e_i]`，`α` init 0 | 直击原假设；可对仅由 30 trial 估计的噪声调谐起正则化 | 96×W + 1 |
| **A embedding** | `nn.Embedding(96,8)` concat 到 ψ 输入 | 最简形式，**最可能被 T4 吸收**；作为 C 的退化对照 | 768 |

**优先级 D → C → A。** D 的机制与调谐正交，最不可能被吸收；A 单独作为
headline 价值最低。

已排除的两个方向：

- **同电极 unit 间聚合**：实测仅 1.3–1.8 unit/电极，多数电极只有 1 个 unit，
  对它们该聚合是恒等映射，覆盖面太小。
- **真拓扑编码**：数据中**无电极坐标**，Utah 阵列 pin 编号非空间单调。需取得
  该实验室的 Blackrock `.cmp` array map 才能还原 10×10 的 (row, col)。
  **这是数据获取问题，不是设计问题**，但它是唯一能带来质变的一步。

### 4b.4 必须写明的代价

A/C/D 都使模型**变成阵列专属**——换植入体或受试者需重训该表。对按患者标定的
芯片场景可接受，但**部分削弱"不依赖固定索引查表"这一卖点**。

必须讲清的区分：**电极索引跨 session 稳定（实测 Jaccard 0.37–0.53 相邻、
0.43 跨两年），而 sorted-unit 索引不稳定。** 不得含糊带过，否则会被正确地质疑。

## 5. 本章程不能声称的内容

- E3 有效不等于"identity 只从 spike 得出"这一原主张仍然成立（§1.3）；
- E3 有效不等于收益来自架构——需 §1.3 的线性解码对照；
- `B3A` 的结果不能外推到 B15 的 neuron 轴 attention，反之亦然；
- validation 筛选结果不是 formal held-out test；
- 单 subject（sub-C）结果不构成 replication。
