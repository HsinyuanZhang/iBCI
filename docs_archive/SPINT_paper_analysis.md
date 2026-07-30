# SPINT 论文深度分析与复现指南

> **本文件定位**：把 SPINT 论文（NeurIPS 2025）的分析写成一份**自包含、可执行**的说明，供另一个 AI 工具/agent 直接用于复现与分析。内容 = 论文方法拆解 + 精确超参与结果基准 + 论文↔开源代码逐项对照 + 复现执行计划。
> **研究背景**：作为 FALCON-M2 片上微调研究的扩展与上限 baseline。M2(DANDI 000953) 已下载在 `FALCON/falcon-challenge-main/data/000953/`；本机有 GPU。
> **配套文件**：`SPINT_reproduction_analysis.md`（代码结构与命令速查）、`spint_mechanism_smoketest.py`（NumPy 机制验证脚本，已跑通）。

---

## 0. 元信息 / TL;DR

- **标题**：SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding
- **作者/单位**：Trung Le, Hao Fang, Jingyuan Li, Tung Nguyen, Lu Mi, Amy Orsborn, Uygar Sümbül, Eli Shlizerman（UW / UCLA / Tsinghua / Allen Institute）
- **会议**：NeurIPS 2025；代码 `https://github.com/shlizee/SPINT`
- **一句话**：把跨 session 神经解码重述为"对一组**无序、可变长度**的神经元 token 做解码"。用一层 cross-attention + 一个从校准数据现算的"神经元身份嵌入"，实现**免梯度、无标签、少样本**的跨 session 适配，在 FALCON M1/M2/H1 上超过所有 zero-shot 与 few-shot 无监督 baseline，部分任务甚至超过有监督 oracle。
- **新范式命名**：GF-FSU = *Gradient-Free Few-Shot Unsupervised*（免梯度少样本无监督）——测试期既不用标签、也不做任何参数更新。

**为什么值得复现**：它给出了 M2 上"不碰权重就能适配"的强上限（held-out R² = 0.26，等于 Wiener Filter oracle），且训练极省资源（单卡 A40、<2GB 显存、M2 约 5 小时）。这对你的轻量片上路线既是天花板参照，也是可蒸馏的思路来源。

---

## 1. 问题定义与动机

**任务**：单 session s 内，神经单元 i 在 t 时刻的 binned spike 记为 $X_{i,t}$，全体单元活动 $X_{:,t}\in\mathbb{R}^{N_s}$。给定过去窗口 $X_{:,t-W+1:t}\in\mathbb{R}^{N_s\times W}$，因果地预测**当前最后一帧**的行为输出 $Y_t\in\mathbb{R}^{B}$（M2 中 B=2，手指 x/y 速度）。模型在 $k$ 个 held-in session 上用梯度下降训练，在 $k'$ 个 held-out session 上评测，**不许梯度更新、不许标签**。每个 held-out session 只给一小段校准期 $X_{i,[C]}$（M 条变长 trial）。

**病因（recording nonstationarity）**：电极位移、阻抗变化、神经可塑性 → 跨 session 记录到的单元**数量、身份、顺序、tuning 都在变**（论文 Fig 1）。传统 decoder 把输入当成固定索引的定长向量，一旦单元集合变化，学到的权重就对不上，跨天性能崩溃。

**已有方法的共同缺陷**：都对神经群体持"固定视角"（固定身份与顺序）。对齐类方法（CCA、linear stabilizer、GAN/RNN/diffusion 对齐）需要显式对齐步骤，且往往需要测试期标签或参数更新，带来部署开销。

**SPINT 的主张**：理想的通用 iBCI 解码器应当**设计上就对神经群体的排列不变**，并能无缝处理可变大小、无序的单元集合，且只用极少数据。

---

## 2. 核心方法（机制原理）

### 2.1 总体框架（Fig 2）

四步流水线：
1. **Neural ID Encoder**：从该 session 的 M 条无标签校准 trial，为每个单元 i 现算一个"身份向量" $E_i\in\mathbb{R}^W$。
2. **身份注入**：把 $E_i$ 加到该单元的活动窗口上，得到"身份感知的活动" $Z = X + E$。
3. **Cross-attention 解码**：一组可学习的行为查询 $Q$（B 个）对 N 个神经元 token（key/value）做 cross-attention，选择性聚合信息。
4. **读出**：投影到最后一帧的行为协变量 $Y$。

关键设计取向：**把"单个神经元的时间窗"当作一个 token**（时间上下文 tokenization），而非把"某时刻的全体活动"当 token。这样单元数量 = token 数量，天然支持可变群体。

### 2.2 神经元身份编码 IDEncoder（论文 eq 1 / A1 / A4；代码 `fc_id_in`+`fc_id_out`）

设 $X_i^C\in\mathbb{R}^{M\times T}$ 为单元 i 的 M 条校准 trial（各插值到定长 T）。身份：

$$E_i = \text{IDEncoder}(X_i^C) = \psi\big(\text{pool}(\phi(X_i^C))\big) = \text{MLP}_2\Big(\tfrac{1}{M}\sum_{j=1}^{M}\text{MLP}_1(X_i^{C_j})\Big) \quad (1)/(A4)$$

- $\phi=\text{MLP}_1$：3 层全连接，$T\to H$，**逐 trial** 作用。
- pool：沿 M 条 trial **均值池化** → 对 M 条 trial 的顺序不变（DeepSets 思想）。
- $\psi=\text{MLP}_2$：3 层全连接，$H\to W$。
- $\phi,\psi$ **跨所有单元与所有 session 共享**。

**这是 few-shot 的载体**：身份不是查表得到的固定 embedding，而是**从校准数据现算的函数**。换新 session → 喂新 calib → 身份自动重算，**不需标签、不更新任何权重**。

### 2.3 置换不变的 cross-attention 解码（论文 eq 2,3 / A5–A8；代码 `MultiLayerCrossAttention`）

身份注入：

$$Z = X + E \quad (2)$$

其中 $Z_i, X_i, E_i$ 是 $Z,X,E$ 矩阵的行（每行 = 一个单元）。$E$ 相当于一种**上下文相关的位置编码**：它随单元顺序等变（换序则 $E$ 同步换序），故对整体保持不变——不同于 Transformer 里固定的位置编码。

解码（完整前向，eq A5–A8）：

$$Z_{in} = \text{MLP}_{in}(Z) \quad (A5)$$
$$\tilde Z = Q + \text{CrossAttn}(Q, \text{LayerNorm}(Z_{in}), \text{LayerNorm}(Z_{in})) \quad (A6)$$
$$Z_{out} = \tilde Z + \text{MLP}_{attn}(\text{LayerNorm}(\tilde Z)) \quad (A7)$$
$$Y = \text{MLP}_{out}(Z_{out}) \quad (A8)$$

其中 $\text{CrossAttn}(Q,Z,Z)=\text{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$，$K=ZW_K,\ V=ZW_V\in\mathbb{R}^{N_s\times W}$，query $Q\in\mathbb{R}^{B\times W}$ 是**随机初始化、训练学习**的行为查询矩阵（B 个协变量各一行）。多任务时可有多组 Q（对应代码里 `learnable_rep` = `rep` 参数，shape `1×C×W`）。

因为 attention 对 key/value 是加权求和、且输出维只由 query 数 B 决定，所以**输出对 N 个神经元的顺序不变、对 N 的大小不敏感**——这正对上了"可变、无序群体"的诉求。

### 2.4 置换不变性证明（Prop 1 / 附录 A.2）

命题：$\text{CrossAttn}(Q,Z,Z)=\text{CrossAttn}(Q,P_RZ,P_RZ)$，$P_R$ 为行置换矩阵。

证明骨架：因 $E_i$ 逐单元独立计算，置换单元 → $E'=P_RE$，故 $Z'=X'+E'=P_R X+P_R E=P_R Z$（等变）。代入 cross-attention，$P_R$ 在 softmax 内的列置换 $P_C=P_R^\top$ 与 $P_R P_C=I$ 相消，softmax 逐行归一化不受列置换前后影响，最终得到 $\text{CrossAttn}(Q,Z,Z)$。MLP、LayerNorm、残差均逐行作用，不破坏不变性。

> **我方独立验证**：`spint_mechanism_smoketest.py` 用纯 NumPy 忠实复刻前向，固定权重仅换神经元轴，实测换序输出 $\max|\Delta|=0$、通道数 96→66 仍输出合法 $W\times C$。数值上坐实了 Prop 1。

### 2.5 dynamic channel dropout（论文 §3.4 / 代码 `dynamic_dropout`）

训练时每个 iteration 从 $U(0,1)$ 采一个丢弃率 p，随机丢弃 p 比例的神经元 token。不同于固定/保守区间丢弃，**变丢弃率**让模型见过任意规模的群体，从而对"单元数量"也鲁棒。消融（Fig 4B / Table A3）证明它是有效正则：M1 上 DD[0,1] 达 0.64 vs 无 dropout 0.51。

### 2.6 gradient-free few-shot 适配闭环（代码 `spint_decoder.py` / `SpintDecoder`）

训练期：用**全部 held-in session** 的有标签数据端到端训练所有参数（IDEncoder + cross-attention），损失 = 最后一帧预测与真值的 MSE。训练时也从 held-in 人工采样 calib 来喂 IDEncoder（`random_calibration`）。

测试期（新 session）：
1. 取该 session 的少量**无标签** calib trial → 过 IDEncoder 现算 $E_i$。
2. 复用已训练权重，`torch.no_grad()` 逐帧解码，无 optimizer、无 backward。
3. 适配一个全新 session = 换一份 calib 特征。

→ 消除了传统方法的显式对齐 / 测试期微调开销，且适用于"测试期天然无标签"的真实 iBCI 场景。

### 2.7 few-shot 深挖（研究重点）

**（a）数据效率**：Fig 3A —— M2 校准 trial 数 ∈{1,7,13,19,25,33}，越多越好且饱和；**M1 甚至 1 条 calib trial 就接近用全部 calib 的最佳**。这直接量化"给多少无标签数据能恢复多少 R²"。

**（b）身份到底编码了什么**：附录 A.3 —— attention 分数与单元的**放电标准差**相关性高于均值（M2：与 std 相关 0.87，与 mean 0.76）。说明 SPINT 学会关注"高方差、行为相关"的活跃单元。这是理解"免梯度身份"为何 work 的关键证据。

**（c）为何免梯度也够**：身份是**输入侧**的加性调制（$Z=X+E$），把"这个单元是谁"直接编码进表征；cross-attention 再按身份选择性聚合。换 session 时只有输入分布变，模型的"如何聚合"逻辑不变——所以不需要动权重。

**（d）与有监督/对齐路线的边界**：SPINT = ZS 与 FSU 的交集（用少量 calib 但不更新参数）。它不需要 CycleGAN/NoMAD 那种"训练一个对齐网络"的测试期开销。

---

## 3. 架构与超参完整规格（论文 ↔ 代码逐项对照）

**Table A9（论文给出的训练超参）与开源 config 对照**：

| 超参 | M1 | **M2** | H1 | 对应代码 config 键 | 备注 |
|---|---|---|---|---|---|
| Batch size | 32 | 32 | 32 | `data.batch_size` | M2 config = 32 ✓ |
| Window size W | 100 | **50** | 700 | `data.window_size` | M2 = 50 ✓ (=1s @20ms) |
| Max trial length T | 1024 | **100** | 1024 | `data.max_trial_length` | M2 = 100 ✓ |
| IDEncoder 层数 (MLP1,MLP2) | 3,3 | **3,3** | 3,3 | `model.net.num_id_layers=3` | ✓ |
| Cross-attn 层数 | 1 | **1** | 1 | `model.net.num_layers=1` | ✓ |
| Hidden dim H | 1024 | **512** | 1024 | `model.net.model_dim` | M2 = 512 ✓ |
| Behavior scaling factor | 1 | **0.2** | 0.05 | `model.behavior_scaling_factor` | 代码存的是 **1/scale**：M2=5.0 (=1/0.2)，"预测 5×真值" |
| Learning rate | 1e-5 | **5e-5** | 1e-5 | `model.optimizer.lr` | M2 = 5e-5 ✓，Adam |
| # heads | — | **64** | — | `model.net.num_heads` | Table A9 未列；A5 消融最优在 32–64；M2 代码=64 |
| dynamic dropout | DD[0,1] | **DD[0,1]** | DD[0,1] | `model.net.dynamic_dropout=true, low=0, high=1` | ✓ |
| trial 插值 | cubic→1024 | **cubic→100** | cubic→1024 | `data.interpolate_trials=true, kind=cubic` | scipy `interp1d` |
| calib trial 数 | (全部) | **33** | (活动段) | `data.calibration_n_trials=33` | H1 用 `use_calib_active_segments` |

**IDEncoder（代码实现细节）**：`fc_id_in`（LazyLinear `T→H` + (num_id_layers-1) 个 [ReLU, Linear H→H]）→ 沿 M 均值 → `fc_id_out`（(num_id_layers-1) 个 [Linear H→H, ReLU] + Linear H→W）。read-in `fc_in` = Linear(W→H)+ReLU+Linear(H→H)，对神经 token 与行为查询共享。

**训练细节**：50 epochs，选 **epoch 50** 的 checkpoint 评测（非 early-stopping best）；scheduler = null（M2）。损失只在最后一帧计（`decode_last_timestep_only=true`），且对 scaled behavior 计损失（`predict_scaled_behavior=true`）。

---

## 4. 数据与预处理（附录 A.4.1–A.4.2）

- **binning**：20ms（阈值过零 spike count），FALCON 标准。
- **因果连续解码**：预测窗口最后一帧的行为（模拟在线 open-loop）；每个 session 开头用 (W−1) 个 0 前置补齐；丢弃"最后一帧落在非评测段（inter-trial）"的窗口。
- **神经窗口用原始 spike count**（M2 config `smooth_calibration=false`）。
- **calib trial 插值**：变长 trial 用 `scipy.interpolate.interp1d` cubic 插值到定长 T（M2 T=100；M1/H1 T=1024）。只对 calib trial 插值，神经窗口不插值。
- **行为缩放**：M2 训练时把预测缩放（乘 0.2，即让网络预测 5× 真值），H1 乘 0.05（预测 20×），M1 = 1。MSE 与 R² 在缩放后的输出与原始真值间计算。
- **数据分割（M2 000953 实际目录）**：`held-in-calib`（训练 + 训练期 calib 源）、`held-in-minival`（val_heldin）、`held-out-calib`（val_heldout，模拟未来天少样本适配）。**公版无 held-out test 标签**（在 EvalAI 服务端）。

---

## 5. 实验结果（作为复现目标基准）

### 5.1 跨 session 主结果（Table 1，held-out 平均 R² ± std）

| 方法 | 类别 | M1 | **M2** | H1 |
|---|---|---|---|---|
| Wiener Filter | Oracle | 0.53±0.04 | 0.26±0.03 | 0.21±0.04 |
| RNN | Oracle | 0.75±0.05 | 0.56±0.04 | 0.44±0.13 |
| NDT2 Multi | Oracle | 0.78±0.03 | 0.58±0.04 | 0.63±0.08 |
| NDT2 Multi | FSS(有监督) | 0.59±0.07 | 0.43±0.08 | 0.52±0.04 |
| WF | ZS | 0.34±0.06 | 0.06±0.04 | 0.16±0.03 |
| RNN | ZS | −0.60±0.45 | −0.07±0.23 | 0.09±0.18 |
| CycleGAN+WF | FSU | 0.43±0.04 | 0.22±0.06 | 0.12±0.06 |
| NoMAD+WF | FSU | 0.49±0.03 | 0.20±0.10 | 0.13±0.10 |
| **SPINT** | **GF-FSU** | **0.66±0.07** | **0.26±0.13** | **0.29±0.15** |

要点：SPINT 超过所有 ZS 与 FSU；M1 上还超过有监督 FSS(NDT2, 0.59)；**M2 上 0.26 = Wiener Filter oracle**、显著高于 FSU（0.20–0.22）。M2 方差偏大（±0.13）。

### 5.2 推理延迟（Table 2，越低越好，<1 = 可实时）

SPINT：M1 **0.13**、M2 **0.13**、H1 **0.14**。与 transformer baseline 相当或更优，且省掉了显式对齐步骤。（对照：NoMAD+WF 因测试期训练对齐网络，延迟 ~0.91–1.03。）

### 5.3 within-session 性能（Table A1，held-in 平均 R²）

SPINT：M1 0.77±0.02、M2 0.59±0.01、H1 0.47±0.06 —— 同样超过 ZS/FSU baseline。

### 5.4 Scaling 分析（Fig 3）

- **calib trial 数**（A）：越多越好、饱和；M1 单条 trial 即接近最佳。
- **训练天数**（B）：越多越好（M1→4 天、M2→4 天、H1→6 天单调提升）→ 有作为大规模预训练模型的潜力。
- **群体规模 %**（C）：对随机子采样鲁棒；M1 只保留 20% 单元仍达 0.52。

### 5.5 消融（Fig 4 + Table A3–A8，多为 M1 sweep）

| 消融 | 结论（R²） |
|---|---|
| 位置编码 (Fig4A) | **context-dependent ID ≫ absolute PE ≈ no PE**（后两者在 M1/M2 崩到负） |
| dynamic dropout (Fig4B / A3) | DD[0,1] 0.64 > 固定 0.4–0.6 (~0.63) > 无 dropout 0.51 |
| dropout 区间 (A4) | 区间越宽越好：[0,1] 0.64 |
| # heads (A5) | 32 最优 0.65，64 = 0.64 |
| # self-attn 层 (A6) | **0 层最优** 0.64（即只用 cross-attn） |
| # cross-attn 层 (A7) | 2 层 0.65，1 层 0.64（为延迟选 1） |
| window size (A8) | 50 最优 0.65（M1） |

### 5.6 attention 与放电统计（Table A2）

attention 分数 vs 放电 std 的 Pearson 相关：M1 0.45 / M2 **0.87** / H1 0.57；vs mean：0.33 / 0.76 / 0.51。→ SPINT 关注高方差活跃单元。

---

## 6. 复现执行计划（可直接交给 AI 工具执行）

### 6.1 环境
```bash
cd /home/xinyuan/Work_host/SPINT/SPINT-main
bash setup.sh            # mamba env create -f environment.yaml && pip install -e .
mamba activate spint
# 若 import falcon_challenge 失败：
pip install -e /home/xinyuan/Work_host/FALCON/falcon-challenge-main
```
Python 3.10 + PyTorch(CUDA 11.8) + lightning + hydra + omegaconf。

### 6.2 数据落位（关键：config 找 `<repo>/data/000953/`）
```bash
ln -s /home/xinyuan/Work_host/FALCON/falcon-challenge-main/data/000953 \
      /home/xinyuan/Work_host/SPINT/SPINT-main/data/000953
```
（M1 000941 / H1 000954 本机未下载，需扩展时再从 DANDI 下载到 `data/000941`、`data/000954`。）

### 6.3 训练（M2）
```bash
python src/train.py data=falcon_m2 model=falcon_m2 trainer=gpu seed=42
# 单卡 A40，<2GB 显存，M2 约 5 小时 / 50 epoch；评测取 epoch 50
```

### 6.4 打包 decoder + 本地评测
```bash
python third_party/falcon_challenge/spint_decoder.py \
  --run_dir logs/train/runs/<run_id> --checkpoint epoch_050.ckpt   # → local_data/spint_m2.pkl

python third_party/falcon_challenge/spint_sample.py --evaluation local \
  --model-path local_data/spint_m2.pkl --split m2 --phase minival --batch-size 7
# batch-size: M1=4, M2=7, H1=8
```
最终 test 分需提交 EvalAI（`--phase few-shot-test-2319`，evalai CLI 装在独立 py3.6 环境）。

### 6.5 复现验收标准（对齐论文）
- **主目标（M2, held-out）**：mean R² ≈ **0.26**（±0.13 方差较大，多 seed 取均值）。minival 上应显著为正且优于 WF ZS(0.06)。
- within-session（held-in minival）参照 **0.59**。
- 延迟应 <1（论文 0.13）。
- 若跑 M1：held-out ≈ 0.66；H1 ≈ 0.29。
- **健壮性自检**：用 `spint_mechanism_smoketest.py` 思路对训练好的模型做通道置换/子采样，R² 应基本不变。

### 6.6 建议复现/扩展的实验清单（M2）
1. **E1 数据效率曲线**：扫 `data.calibration_n_trials ∈ {1,7,13,19,25,33}`，画 R²-vs-calib（复现 Fig 3A）。产出"达 90% 饱和所需最少无标签 trial"。
2. **E2 群体规模鲁棒**：评测时随机保留 {100,80,60,40,20}% 通道（复现 Fig 3C / Prop 1）。
3. **E3 位置编码消融**：`use_learnable_id` 关掉换固定/无 PE（复现 Fig 4A，预期崩塌）。
4. **E4 dynamic dropout 消融**：`dynamic_dropout=false` 重训（复现 Fig 4B）。
5. **E5 结构 sweep**：heads {4,8,16,32,64}、cross-attn 层 {1,2,3}、window {50,100,200}（复现 A5/A7/A8）。
6. **E6 attention-firing 相关**：抽 attention 分数与 calib 放电 std/mean 算 Pearson（复现 A2，M2 目标 ~0.87）。
7. **E7 与 m2-research 对接**：同口径对比 SPINT vs 你的 Ridge/LSTM/LoRA/RLS，把 SPINT 作 R² 天花板并入 `adapt_benchmark_summary.csv`。

固定 seed，跑 3 seed 取均值±std，结果落 `m2-research/outputs/results/`。

---

## 7. 关键洞察、局限与风险

**洞察**：SPINT 的"免梯度"来自把适配负担从"改权重"转移到"改输入（现算身份 + 注入）"。这与你 findings 里的 Level-0（每通道 α/β 增益零偏）思路同源——SPINT 相当于它的深度、非线性、数据驱动版。可考虑把 IDEncoder 蒸馏成轻量形式上芯片。

**论文自陈局限**：
- IDEncoder 与解码端**端到端联合训练**，身份识别绑定了行为标签 → 训练仍需标签（测试期才免）。作者建议未来用自监督（对比/预测学习）解耦。
- 结果均为 **in silico**（离线 open-loop），未做 in vivo 闭环；闭环下感觉反馈会引入神经调制，硬件延迟也可能变化。
- 单被试、单任务；跨被试/跨任务/非运动行为为未来工作。

**复现风险**：
- M2 方差大（±0.13），单 seed 可能偏离，务必多 seed。
- 公版无 test 标签，最终数字要走 EvalAI。
- `evalai` 与 `spint` 依赖冲突，需独立环境。
- 代码 `behavior_scaling_factor` 存的是 1/scale（M2=5.0 对应论文 0.2），勿混淆。
- H1 走 `use_calib_active_segments`（按活动段而非 trial 切 calib），与 M1/M2 不同。

---

## 8. 论文 ↔ 代码差异 & 验证要点

| 项 | 论文 | 代码 | 是否一致 |
|---|---|---|---|
| IDEncoder 结构 | MLP2(mean_M(MLP1)) | fc_id_in→mean(dim=M)→fc_id_out | ✓ |
| cross-attn 层数 | 1 | `num_layers=1` | ✓ |
| self-attn | 0（A6 最优） | 无 self-attn，仅 cross | ✓ |
| M2 H / W / T / lr | 512 / 50 / 100 / 5e-5 | 完全一致 | ✓ |
| behavior scale (M2) | 0.2 | 5.0 (=1/0.2) | ✓（口径反） |
| checkpoint 选取 | epoch 50 | periodic_ckpt/epoch_050 | ✓ |
| trial 特征 | raw（插值后） | `trial_feature_type='raw'`（另支持 fft，论文未主用） | 代码多一个 fft 选项 |

**验证建议**：复现后先跑机制自检（置换/子采样不变性），再核对 minival R² 量级，最后提交 EvalAI 对齐 Table 1。

---

## 9. 参考文献（精选，含可点击链接）

- SPINT 论文（NeurIPS 2025）— 本地：`SPINT/NeurIPS-2025-spint-...-Paper-Conference.pdf`；代码 https://github.com/shlizee/SPINT
- FALCON Benchmark [26]: Karpowicz et al., *Few-shot algorithms for consistent neural decoding (FALCON)*, NeurIPS 2024. https://doi.org/10.1101/2024.09.15.613126 ；挑战页 https://eval.ai/web/challenges/challenge-page/2319/evaluation
- DeepSets [56]: Zaheer et al., NeurIPS 2017（置换不变池化的理论基础）https://arxiv.org/abs/1703.06114
- Set Transformer [58]: Lee et al., ICML 2019 https://arxiv.org/abs/1810.00825
- Attention Is All You Need [59]: Vaswani et al., NeurIPS 2017 https://arxiv.org/abs/1706.03762
- NDT2 [21]: Ye et al., *Neural Data Transformer 2*, NeurIPS 2023（主要 transformer 对照/oracle）
- POYO / POYO+ [22][47]: Azabou et al.（token 化对照）
- CycleGAN 对齐 [29]: Ma et al., eLife 2023；NoMAD [25]: Karpowicz et al., bioRxiv 2022（FSU 对照 baseline）
- M2 数据 [69]: Nason et al., *Real-time linear prediction of two finger groups*, Neuron 2021（DANDI 000953 来源）
- Adam [76]: Kingma & Ba, 2014 https://arxiv.org/abs/1412.6980

*完整 76 条引用见论文 References（PDF 第 10–15 页）。*
