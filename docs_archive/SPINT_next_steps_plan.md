# SPINT→片上 M2：历史下一步计划（含 BrainDistill 精读）

> **文档状态：MUA/ASIC 历史支线。** 当前项目主线已切换为 [`sua_exploration/README.md`](../sua_exploration/README.md) 所述的 SUA/MUA 共享编码器探索。本文保留作为 decoder 压缩和片上部署背景，不代表当前执行优先级。

> 定位：精读 BrainDistill(2601.17625) 后，明确"哪些借鉴、哪些必须发散"，并沿两条轴——(A) 调整 few-shot 对齐网络、(B) 尚未有人试过的时间卷积——给出可执行的下一步。
> 关联：`SPINT_deployment_cost_analysis.md`（开销/ASIC 视角 + ridge few-shot 验证）。

---

## 1. BrainDistill 精读

**是什么**：EPFL Shoaran(芯片)/Courtine·Bloch(临床) 组，2026-01。面向"可植入 SoC"的运动解码蒸馏 pipeline = IND(小解码器) + TSKD(任务特定知识蒸馏) + 整数量化。

**IND 架构**：CWT(连续小波变换) tokenization → 线性投影+位置编码 → **2 层线性注意力**(ReLU 核，非 softmax，为量化) → 平均池化 → 1 层线性解码。极小：**30K 参数**；片上变体 320 维输入(5 小波×64 通道 或 10 小波×32 通道)、2 attn 层、32 维 embedding。

**量化/功耗**：W8A8 整数-only，PACT 式可学习 clipping 的 QAT。**FP32 22.84 mW → 量化 5.66 mW（>3×）**，<3% 掉点，落在 15–40 mW 植入热安全预算内。

**TSKD**：两步——先用监督 projector 把 teacher embedding 压进"任务子空间"(TSR 指标量化信息保留)，再对齐 student。**需要标签**。

**FALCON 结果（关键）**：只用 **FALCON-M1**（spikes，teacher=NDT2，target=16 通道 EMG，指标 R²%）：

| | R² |
|---|---|
| Teacher (NDT2) | 0.82 |
| IND (不蒸馏) | 0.376 |
| IND + TSKD (最好) | **0.435** |

→ 在 spike 运动任务上，**蒸馏 student 离 teacher 差一大截（0.44 vs 0.82）**；而在 ECoG 分类任务上 TSKD 几乎追平 teacher。说明它这套"CWT+线性注意力 student + 监督蒸馏"对 **spike 回归并不 work 得很好**。

## 2. 借鉴 vs 必须发散

**借鉴（拿来即用的硬件事实）**：
- 植入目标量级实锤：**~30K 参数 / W8A8 / 5.66 mW** 是 tapeout 可达、热安全的 student 规模——直接当我们 ASIC toy 的目标。
- **线性注意力(ReLU 核) 替 softmax** = 量化友好。若我们保留任何注意力，用这个而非 softmax。
- QAT 配方：PACT 可学习 clipping + STE，W8A8 整数-only。
- ReLU/平均池化等量化友好算子；蒸馏(大 teacher→小 student)范式成立。

**必须发散（BrainDistill 不解决我们的核心问题）**：
1. **它根本不管置换不变/变通道**——注意力在**时间轴**上，通道是固定特征维。跨 session 鲁棒性外包给 teacher(NDT2)+只重标定分类器 fϕ。对 M2 跨天换单元，这条不 transfer。**SPINT 的核心贡献它没碰。**
2. **它的蒸馏是监督的（TSKD 要标签）**；few-shot=有标签重标定。我们 M2 部署目标是**测试天无标签**。
3. **spike 上 student 掉点严重（0.44 vs 0.82）**——它的花哨 student 在 spike 回归上并不比简单基线明显强。这是漏洞，也是我们的机会。
4. **它明确弃用卷积、选 CWT**（Appendix A.7），理由是 ECoG 频带可解释性。但 **CWT 是一组固定的时频卷积核**；对 spike(点过程/binned count)，"频带可解释"这套动机弱，**学出来的时间卷积很可能更好**——正是我们要试的"尚未有人试过的时间卷积"。

**一句话**：BrainDistill = 固定通道 + 监督蒸馏 + CWT+线性注意力 student；SPINT = 变通道 + 无监督对齐 + 神经元轴注意力。**两者正交，没人把"SPINT 式无监督跨天对齐"和"可植入量化 student"缝在一起——这就是我们的空位。**

---

## 3. линchpin 实验：先搞清 0.11→0.26 的增益住在哪

在投入 A 或 B 之前，必须先做这个**归因消融**，它决定后续资源往哪投：

> SPINT(0.26) 相对 CORAL 线性对齐(0.11) 多出来的 0.15，到底来自
> (i) **非线性身份 E**（IDEncoder 比 CORAL 的每通道仿射强），还是
> (ii) **cross-attention 解码器**（比线性 ridge 强）？

做法（都在已训好的 M2 checkpoint 上，无需重训）：
- **E→线性**：把 SPINT 的 E 换成 CORAL 式每通道仿射，保留 cross-attn，测掉多少。
- **decoder→线性**：保留 SPINT 的非线性 E，把 cross-attn 换成"E 注入后 + 固定线性读出"，测掉多少。
- 四象限：{E: 线性/非线性} × {decoder: 线性/attn}，四个角分别对应 CORAL(0.11)、纯 E、纯 attn、完整 SPINT(0.26)。

**决策**：
- 若增益主要在 (i) E → 投 **轴 A**：片上放廉价固定 decoder，把预算花在"更好但仍便宜的对齐网络"。
- 若增益主要在 (ii) decoder → 投 **轴 B**：对齐用 CORAL 就够，把预算花在"比 ridge 强的时间卷积 decoder"。

---

## 4. 轴 A —— 调整 few-shot 对齐网络

目标：找到"保住大部分 0.11→0.26 增益"的**最小/最结构化**对齐网络（IDEncoder 可片外，但越小越好、也可能部分上片）。

- **A1 IDEncoder 瘦身消融**：num_id_layers 3→{1,2}、H 512→{64,128}、去掉 MLP2，看 E 质量/R² 曲线。定位"每 mW 对齐收益"。
- **A2 统计量混合身份**：CORAL(mean/std) 是 E 的线性下界(0.11)。试"每通道统计量 + 一个小 MLP 残差"，用极少参数逼近非线性 E。
- **A3 few-shot 可调的对齐**：让 calib 不只产 E，还产一个**每 session 的小 decoder 修正**(hypernetwork 思路)，把"对齐"和"轻量适配"合一——仍无梯度、闭式/前向。
- **A4 无监督蒸馏对齐**：SPINT 当 teacher 出伪标签，标定 student 的对齐参数(接 `15_ridge_fewshot_curve.py` 已有框架)，验证无监督天花板能否从 0.11 抬向 0.26。

## 5. 轴 B —— 时间卷积（尚未有人在 M2 上试）

现状：FALCON ridge(n_hist=7)=flat 768 路时间 FIR，天花板 ~0.15；SPINT 无时间卷积；BrainDistill 用 CWT(固定核)不学。**"学出来的时间卷积 student" 是空白。**

- **B1 depthwise 时间卷积 student**：每通道共享/平滑的可学习 FIR(TCN) + 线性混合。比 768 自由权重更不过拟合，全静态、无 softmax/attn、ReLU、天生量化友好——**比 BrainDistill 的线性注意力 student 更 ASIC-friendly，且更适配 spike**。目标：M2 held-out 抬过 0.15。
- **B2 CWT vs 学习卷积 head-to-head（on spikes）**：直接复现 BrainDistill 弃用卷积的对比，但在 **M2 spike** 上做——验证"频带可解释"动机在 spike 上不成立、学习卷积更优。这本身是可发表的点。
- **B3 时间卷积 + 输入侧对齐**：B1 的卷积 student 前面接 CORAL/SPINT 的每通道对齐，做成"跨天 + 时间卷积"的完整片上链路。
- **B4 QAT + 功耗核算**：对 B1/B3 做 W8A8 QAT，按 BrainDistill 的方法估 mW，对齐 5.66 mW 参照，给出 tapeout 规格。

---

## 6. 目标空位与优先级路线

**没人做过的组合（= 论文 + 芯片）**：
> 置换鲁棒的**无监督**跨天对齐(SPINT-E / CORAL) → 喂一个**量化的 depthwise 时间卷积 student**(非 attention) → SPINT 无监督蒸馏标定 → mW 级 tapeout。

**优先级**：
1. **§3 归因消融**（1–2 天，无需重训）——决定 A/B 侧重。**先做这个。**
2. **B1 时间卷积 student** + **A4 无监督蒸馏**（并行，复用 `15_` 框架）——最快拿到"能否超 0.15 / 逼近 0.26"的判据。
3. **B2 CWT-vs-conv on spikes**——差异化、可发表。
4. **A1/A2 对齐瘦身 + B4 QAT/功耗**——收敛到 tapeout 规格(参照 30K 参数 / 5.66 mW)。

**验收指标**：M2 held-out R²(对齐 SPINT 0.26 与 ridge 0.11–0.15) · student 参数量 · W8A8 mW(≤15–40 热预算) · 是否全静态无 softmax。

---

## 7. 关键对照数字

| | SPINT | BrainDistill IND | FALCON ridge | 我们的目标 student |
|---|---|---|---|---|
| 参数 | 4.6M | 30K(量化) | ~1.5K(768→2) | 目标 <30K |
| 算子 | softmax cross-attn | 线性注意力(ReLU) | 纯线性 | depthwise 时间卷积+ReLU |
| 注意力轴 | 神经元(置换不变) | 时间 | 无 | 无 |
| 跨天对齐 | 非线性 E(无监督) | teacher+重标定分类器(监督) | 每通道仿射(无/有监督) | 无监督 E/CORAL |
| 功耗 | — | 5.66 mW | — | 对齐 ≤5.66 mW |
| M2 | 0.26 | 未测 M2(只 M1=0.435) | 0.11–0.15 | 目标 >0.15→0.26 |
