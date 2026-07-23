# B16-like Encoder Optimization Brainstorm

**状态：设计策略 + 首个内部配对 cell 证据；跨 fold 稳定性仍在验证**  
**更新：2026-07-22**

## 1. 目标

优化 B16 类跨 trial 高阶统计 encoder，使它同时满足：

1. 平均跨 session task R² 稳定高于 B3；
2. 不用少数 session 的大幅下降换取均值的小幅提升；
3. 对 calibration trial 子集、seed 和 LOSO fold 不敏感；
4. 保留 B3 的 per-neuron、streaming、轻量和硬件友好属性；
5. 方差无效时能够退化为 B3，而不是重新学习一套不受约束的 identity mapping。

当前 B16 pilot 是有价值的正向信号，但还不满足上述标准：held-out mean R² 从 B3 的 `0.2363` 提升到 `0.2475`，但只在 4/6 sessions 提升，worst paired delta 为 `-0.0538`，peak live state 为 87,552 B。

## 2. 已观察到的失败模式

### 2.1 方差分支不是“小修正”

当前结构直接拼接：

\[
\mu=\frac{1}{M}\sum_m h_m,\qquad
v=\frac{1}{M}\sum_m h_m^2-\mu^2,\qquad
E=\operatorname{MLP}([\mu,v]).
\]

训练后第一层中，variance half 与 mean half 的权重 Frobenius norm 比为 `0.954`。在六个 held-out sessions 上，variance 对第一层激活的 RMS 贡献约为 mean 的 35–46%。因此当前 B16 实际上让一个较噪的二阶统计与均值分支拥有近似同等话语权。

### 2.2 没有 B3 安全回退

- B16 的 `2D -> D` post-MLP 全部随机初始化；
- 没有复制 B3 mean path；
- 没有 zero-init variance columns；
- 没有 residual、gate 或 branch penalty。

所以即使 variance 对某个 session 没有价值，模型也不能自动回到已知有效的 B3 函数。

### 2.3 支持集敏感度与 session 稳定性相关

在有足够 trial 的四个 held-out sessions 上，用八个不同连续 33-trial support 计算 identity variation：B16/B3 比值为 `0.94、0.71、0.90、1.07`。唯一超过 1 的 `2020-11-19-Run1` 也是 B16 明显退化的 session。这不是充分证明，但支持加入 support-consistency 约束。

### 2.4 训练与选择协议放大方差

- 训练使用随机连续 33-trial block；validation/test 固定使用前 33 trials；
- 六个训练 session 的窗口数相差约 1.64 倍，梯度贡献不平衡；
- sampler 在构造时用固定 seed 42 打乱，此后每个 epoch 顺序相同；
- checkpoint 只由一个 LOSO validation session 选择；
- teacher identity 是 mean-pooled representation，和显式 variance 分支存在目标冲突。

### 2.5 研究过程的 held-out 泄漏风险

六个 external held-out sessions 已经用于观察 B16 pilot。后续不能反复根据它们选择 normalization、gate 或 loss，否则即使训练没有读取这些数据，研究流程本身也会过拟合。结构选择必须关闭 external held-out evaluation，先基于多个 held-in LOSO folds 完成。

## 3. 广泛候选空间

### 3.1 数值与统计估计

| 想法 | 预期收益 | 代价/风险 | 优先级 |
|---|---|---|---|
| Welford 在线均值/方差 | 避免 `E[x²]-E[x]²` cancellation，利于 INT8/定点 | FP32 精度增益可能很小 | P1，作为实现卫生项 |
| Bessel correction | 修正有限 M 偏差 | M=33 时只有约 3% 尺度变化 | P2 |
| Shrinkage variance | 将 noisy per-neuron variance 向训练集 feature prior 收缩 | 需要选 shrink 系数 | **P0** |
| `log1p(var/scale)` | 压缩长尾与跨 session 尺度 | 需要 LUT/近似；动态范围需校准 | P1 |
| Fano-like `var/(mean+eps)` | 消除发放尺度影响 | 低均值区域不稳定 | P1，仅带 shrink/clip |
| CV² `var/(mean²+eps)` | 更强尺度不变性 | 低均值爆炸风险更高 | P2 |
| Winsorized/Huber variance | 降低异常 trial 影响 | 需要 clip 阈值或第二遍处理 | P1 |
| Mean absolute deviation | 比 variance 更 robust | 精确计算通常需已知 mean 后第二遍扫描 | P2 |
| Median-of-means | 将 trials 分组取均值，再对组做稳健聚合 | group 数和近似 median 需固定 | P1 |
| Learned trial-confidence weighting | 学习每个 trial 的可靠性，再计算 weighted mean/variance | 增加 `M*N` 标量打分，可能学到数据集偏差 | P1 |
| Raw spike-count reliability | 每个 neuron 只统计 per-trial count 的 mean/var/Fano | 表达力低但稳健、可解释、状态极小 | **P0/P1 高信息 probe** |
| Skewness/kurtosis | 可能捕获非高斯 trial reliability | 33 trials 下估计噪声大、状态和溢出风险高 | **暂缓** |
| Quantiles/IQR | 对 outlier robust | 需要 trial buffer、排序或近似直方图 | **暂缓** |
| Full latent covariance | 捕获 feature 共变 | `O(D²)` state/compute，极易过拟合 | **拒绝当前阶段** |

### 3.2 融合与安全回退

| 想法 | 核心形式 | 预期收益 | 优先级 |
|---|---|---|---|
| B3-preserving zero-init | copy B3 post-MLP；新增 variance columns 初始化为 0 | 初始函数严格等于 B3，variance 只能逐步贡献 | **P0，首选** |
| Output residual | `E=E_B3(mu)+alpha*R(v)`，`R` 末层 zero-init | 最明确的安全回退与可解释 branch delta | P0，但参数更多 |
| Global scalar gate | `E=E_B3+sigmoid(a)R(v)` | 简单限制整体贡献 | P1；太粗，无法解释 session 差异 |
| Per-feature gate | `g_d*var_d`，gate 负偏置初始化 | 抑制不可靠 latent dimensions | **P0/P1** |
| Per-neuron reliability gate | 用 mean、var、trial count 预测 gate | 可按 unit 调整 variance 信任度 | P1；小 MLP 可能过拟合 |
| Separate low-rank variance projection | mean 保持 B3 D64，variance 用独立 Dv 投影 | 避免同一 `W_pre` 同时操纵 mean/variance，易控状态 | P1 |
| Session/global gate | 用 population statistics 调节 variance | 可能保护不可靠 session | P2；破坏纯 per-neuron 独立性 |
| Branch contribution penalty | 约束 `||E(mu,v)-E(mu,0)||` | 直接限制 variance 伤害 | P1；权重过大可能抹掉增益 |
| Stochastic variance dropout | 训练时随机关闭 variance branch | 强迫 mean path 单独可用 | **P1，低实现成本** |

### 3.3 Branch normalization

| 想法 | 硬件含义 | 判读 |
|---|---|---|
| Frozen per-feature scale | 训练 held-in 数据估计 `s_d`，部署时乘常数 `1/s_d` | **首选**；无动态 sqrt，可折叠进第一层权重 |
| LayerNorm/RMSNorm(var) | 需要动态 reduction、sqrt/divide | 研究 probe 可用，首版硬件不优先 |
| Clip + fixed affine | comparator + 常数乘法 | 硬件友好，可控制 outlier |
| BatchNorm running stats | 部署为固定 affine | 可行，但训练跨 session running stats 可能偏置 |

### 3.4 训练目标与优化

| 想法 | 作用 | 风险/控制 |
|---|---|---|
| B3 checkpoint warm-start | 保留已知 mean solution | 每个 seed 必须有配对 B3 checkpoint |
| 两阶段训练 | 先冻结 mean path 只训练 variance 接口，再小 LR 联合微调 | 防止 mean path 被快速破坏；**P0** |
| L2-SP / weight anchoring | 约束 mean-path 权重偏离 B3 | 需要单独报告 anchor strength |
| Lower LR for mean path | variance 学新信息，mean 慢变化 | 比全模型同 LR 更稳；**P0** |
| `lambda_E` sweep `{0,0.03,0.1}` | 检查 mean-only teacher identity 是否压制有效 variance | 必须固定架构后再做，避免组合爆炸 |
| Identity-loss decay | 早期保持 B3，后期让 task 决定 variance | 可能比固定 lambda 更符合两阶段训练 |
| Support consistency | 两个 calibration subsets 的 E 或 prediction 保持一致 | **P0/P1**，直接针对已观察敏感度 |
| Prediction consistency | 同 query、不同 support 的行为输出一致 | 与最终任务更对齐，优先于纯 identity consistency |
| Variance-branch norm penalty | 控制第一层 variance/mean 权重或贡献比 | 可设 soft target 10–30%，不应硬压到 0 |
| EMA/SWA checkpoints | 降低 epoch-1 偶然最优问题 | 成本低，需严格只用 held-in validation 选择 |

### 3.5 数据采样与跨 session 鲁棒性

| 想法 | 预期作用 | 优先级 |
|---|---|---|
| Session-balanced batch sampler | 每个 session 每 epoch 等量更新 | **P0，架构无关稳定性 control** |
| 每 epoch 重新 shuffle | 恢复 seed/epoch 数据顺序多样性 | **P0，基础设施修正** |
| Mixture support policy | 连续 block + uniform random subset 混合 | P1，减少对局部时间段的依赖 |
| Two-support training | 同一 query 同时取两个 support blocks | 支持 consistency loss；P0/P1 |
| Variable M curriculum | M 从 8/16/24/33 变化 | 提升 progressive calibration 鲁棒性；P1 |
| Trial dropout/bootstrap | 对二阶统计做 bagging 式扰动 | P1；必须避免重复 trial 造成方差偏差 |
| Calibration chronology feature | 显式区分快漂移与稳定 variance | P2；增加机制复杂度 |
| GroupDRO/CVaR across sessions | 优先优化高损失 session | P1；先用 balanced sampler 做便宜控制 |

### 3.6 容量和硬件方向

| 想法 | 状态成本 | 建议 |
|---|---:|---|
| Full-D64 variance | 额外 24,576 B FP32 support state | 当前准确率研究基线 |
| Low-rank Dv=16 variance | 额外 6,144 B | P1，先确认 full-D 机制后压缩 |
| Low-rank Dv=8 | 额外 3,072 B | 更接近 64 KiB peak gate |
| Dv=4 | 额外 1,536 B；FP32 peak 约 64,512 B | 可满足旧 <64 KiB gate，但表达力风险高 |
| INT16 second-moment state | 额外状态减半 | 需先验证 overflow/scale；放到 QAT 阶段 |
| Shared/grouped variance | 每 4–8 latent features 共用一个 variance statistic | 低状态，可能比随意低秩更可解释 |

## 4. 推荐主方案：B16-ZS

第一优先不是加入更多 moments，而是把当前 variance 变成受控增量。

### 4.1 结构

\[
h_m=\operatorname{ReLU}(W_{pre}x_m+b_{pre})
\]

\[
\mu=\operatorname{mean}_m(h_m),\qquad
v=\operatorname{WelfordVar}_m(h_m)
\]

\[
\hat v=\operatorname{clip}\left(\rho\odot\frac{v}{s_v+\epsilon},0,c\right)
\]

\[
E=\operatorname{MLP}_{B3\text{-extended}}([\mu,\hat v]).
\]

这里的关键不是公式表面，而是初始化和约束：

- `pre_pool` 和 mean-path 从配对 B3 checkpoint 复制；
- post-MLP 第一层的 mean columns 从 B3 复制；
- 第一层扩展到 `2D -> D`，后续 B3 layers 原样复制；
- variance columns 初始化为 0，因此 step 0 与 B3 bitwise/functionally 等价；
- `s_v` 是仅由训练 held-in sessions 估计的 frozen per-feature scale，可折叠进 variance columns；
- `rho` 是 shrink/gate，初始化为小值并正则化；
- mean path 使用较低 LR，variance columns 使用正常 LR；
- 训练初期冻结 mean path，随后小 LR 联合微调。

这一路线比直接加入 LayerNorm、attention 或三四阶矩更优先，因为它首先修复当前最明确的缺陷：没有安全回退。

### 4.2 推荐训练方式

1. 从同 seed、同 fold 的 B3 checkpoint 初始化；
2. epochs 0–2：冻结 B3 参数，只训练 variance columns/gate；
3. epochs 3+：mean path 用 `0.1x` LR，variance path 用 `1x` LR；
4. variance branch dropout 设为 0.25–0.5；
5. 第一轮保留现有 loss，避免同时改变结构和目标；
6. 结构通过后再做 `lambda_E` 和 consistency ablation。

## 5. 最小可证伪实验序列

### Round A：安全回退是否解决稳定性

不读取 external held-out sessions。使用新 LOSO folds 和配对 seeds。

| Run | 变化 | 回答的问题 |
|---|---|---|
| A0 | B3 paired baseline | seed/fold 基线 |
| A1 | 原始 B16 | 复现未经约束 variance 的行为 |
| A2 | B16-Z：B3 warm-start + zero-init variance columns | 安全回退本身是否有效 |
| A3 | B16-ZS：A2 + frozen variance scale + shrink/clip | 尺度控制是否进一步稳定 |
| A4 | B16-R1：B3 warm-start + zero-init raw spike-count reliability scalars | 高维 latent variance 是否必要 |

建议先用 folds 1/2、seeds 42/43 形成四个 fold-seed cells；只把 A2/A3/A4 中通过门槛者扩展到更多 folds。

Round A 门槛：

- paired mean validation delta `>= 0`；
- worst fold/seed delta `>= -0.02`；
- 至少 3/4 fold-seed cells 不低于 B3；
- variance first-layer contribution不超过 mean 的约 30%，除非多 fold 证据支持更高值；
- support-subset identity variation不超过 paired B3 的 `1.05x`；
- 不允许 external held-out 参与选择。

### Round B：训练稳定性

只对 Round A winner：

| Run | 变化 |
|---|---|
| B0 | winner 原训练目标 |
| B1 | session-balanced sampler + per-epoch reseed |
| B2 | B1 + two-support prediction consistency |
| B3 | B2 + `lambda_E=0.03` |

不要一开始同时上 GroupDRO、variable M、robust variance 和新 loss；否则无法知道稳定性来自哪里。

### Round C：硬件压缩

只有 full-D B16-ZS 在多 fold/seed 稳定后，才比较 `Dv={16,8,4}`、grouped variance 和 INT16 accumulator。否则压缩的是一个尚未成立的机制。

### Final：外部确认

- 冻结结构、loss、seed aggregation 和 checkpoint rule；
- 只运行一次六-session M2 external evaluation；
- 由于这些 sessions 已在 pilot 中被观察，结果应写成 repeated benchmark confirmation，而不是 pristine final test；
- 真正外部确认应使用 M1/H1 或新的多 session SUA 数据。

## 6. 决策指标

不能只看 mean R²。每轮至少记录：

- paired mean/median R² delta；
- worst-session/fold/seed delta；
- 正向 cell 比例；
- paired delta SD 与 seed SD；
- 不同 calibration subsets 的 identity/prediction variation；
- variance/mean branch activation contribution；
- normalized identity MSE 与 task R² 的冲突程度；
- 参数、MAC、support state、peak state；
- 最佳 epoch 分布，检查是否总在 epoch 0–2 后迅速退化。

推荐通过门：多 fold/seed mean delta 至少 `+0.005`，worst delta `>= -0.03`，正向 cell 比例至少 70%，support sensitivity 不高于 B3，且增益不是由单一 session 驱动。

## 7. 当前优先级结论

1. **先做 B16-Z，再做 B16-ZS。** 安全回退比添加第三/第四矩更重要。
2. **保留 B16-R1 作为高信息、低成本 probe。** 若少量 raw reliability scalars 已能获得相同增益，就不应保留 D64 二阶矩状态。
3. **同时修 session-balanced sampler 和 per-epoch shuffle，但作为独立 control。**
4. **结构稳定后再加 two-support prediction consistency。**
5. **再做 `lambda_E` sweep。** 当前 teacher identity 不含显式 variance，loss alignment 值得验证。
6. **最后做低秩 variance 和定点 accumulator。**
7. **暂缓 skewness、kurtosis、quantiles、full covariance 和 B15+B16 大融合。** 它们增加统计噪声或硬件复杂度，却没有先解决 B16 已知的不稳定根因。

## 8. 内部配对实验更新

开发选择已关闭 external held-out，仅使用新的 held-in LOSO cell。fold 1 / seed 42 的新训 B3 test R² 为 `0.77845985`。

| 方案 | 内部 LOSO R² | paired delta | 结论 |
|---|---:|---:|---|
| B16-Z，raw D64 latent variance，LR 1e-4 | 0.77688444 | -0.00157541 | 单调退化，淘汰 |
| B16-R1，raw spike-rate variance，LR 1e-5 | 0.77841365 | -0.00004619 | 几乎回到 B3，但无增益 |
| B16-R1F，log-Fano，LR 1e-5 | 0.77858412 | +0.00012428 | 正向但不具实际意义 |
| B16-R1F，移除 identity distillation | 0.77838612 | -0.00007373 | identity loss 是稳定器 |
| B16-ZF，`log1p(var / abs(mean))`，LR 3e-5 | **0.78335494** | **+0.00489509** | epoch 27 最佳、曲线平滑；主候选 |
| B16-G，直接乘性 reliability gate | 0.77742195 | -0.00103790 | 单调退化，淘汰 |
| B16-ZF，解冻 mean/variance 同 LR fusion | 0.77423275 | -0.00422710 | 破坏 B3 anchor，淘汰 |
| B16-ZF，mean 使用 0.03x LR fusion | 0.77683377 | -0.00162607 | 仍退化，淘汰 |

因此当前优先级已从抽象的 B16-ZS 收敛到一个更具体的 B16-ZF：冻结完整 B3 路径，只训练 zero-init 的 D64 latent log-normalized variance residual。它在首个 cell 的增益只比预注册 `+0.005` 门槛低 `0.00010491`，不能四舍五入成通过，且单一 cell 不足以证明稳定。

对 epoch-27 residual 矩阵的 SVD 显示：rank 4/8/16 分别解释约 `52.5% / 76.8% / 92.8%` 的矩阵能量。若 fold 2 暴露 full-D 过拟合，下一个有证据支持的结构应是 exact-B3 fallback 的 rank-8 residual，而不是继续解冻 mean path。rank-8 参数量为 `2*64*8=1024`，相对 full `64*64=4096` 降低 75%；若同时把 variance statistic 投影到 Dv=8，才能进一步减少 support state。

基础设施方面，sampler 的固定 seed 42 已改为显式 `sampler_seed=${seed}`；seed 42 顺序保持向后兼容，seed 43 开始会真正改变 batch 顺序。标准 B16-ZF 三格结果没有混入每 epoch reshuffle 或 session balancing。三格 screen 完成后，二者已作为独立的 B16-ZF-SB control 启用：不改变每 epoch 总 batch 数，短 session 确定性循环补齐，默认配置仍保持旧行为。

新增的第三个 paired cell（fold 1 / seed 43）中，B3 为 `0.77309322`，B16-ZF 为 `0.77441889`，delta `+0.00132567`。前三格 delta 为 `+0.00489509、+0.00158924、+0.00132567`：3/3 正向，worst 为正，mean 约 `+0.00260333`。因此 B3-preserving + log-normalization 已显著改善“方向稳定性”，但效果量仍只有 `+0.005` 目标的一半左右。

新的机制判断是：full-D B16-ZF 的 variance 坐标来自每个 B3 seed 单独学习的 `pre_pool` latent basis，该 basis 会随 seed 旋转；即使 residual 不伤害 B3，也可能限制可重复的效应量。下一高信息候选改用固定语义的 8 个 trial 时间分箱，在每个 bin 上计算跨 trial log-Fano，再用 zero-init `8->64` residual 映射接入同一 B3 anchor。它保留时间结构、比 B16-R1F 的单个全 trial 标量更丰富，同时只有 512 个新增权重，明显低于 B16-ZF 的 4096 个权重。

该固定语义假设随后被配对实验部分否证。B16-R8F 在 fold 1 / seed 43 达到 `0.77571487`（delta `+0.00262165`），但 fold 2 / seed 42 为 `0.69826943`（delta `-0.00043815`）；加入八个 temporal mean 特征的 B16-R8MF 在 fold 2 仅为 `0.69876456`（delta `+0.00005698`）。因此固定时间分箱可以改变弱 cell 的效应量，却没有比 B16-ZF 更好的跨 fold 证据。

当前实验转向 B16-ZF-SB。fold 1 / seed 43 的六个训练 session 原始完整 batch 数为 `622/488/485/382/459/555`；平衡模式保持总计 2,991 batches，每个 session 约 498-499 次暴露，并以 `seed+epoch` 重洗牌。完整 20-epoch 结果在 epoch 19 达到/test `0.77668756`，相对配对 B3 `+0.00359434`，相对标准 B16-ZF 同 cell 净提升 `+0.00226867`。这已证明训练分布控制不只是加快早期收敛，但单 cell 仍低于 `+0.005` 门槛；fold 2 / seed 42 的原样复制正在进行。

fold 2 / seed 42 的原样复制在 epoch 5 达到/test `0.69990575`，paired delta `+0.00119817`，比该 cell 的标准 B16-ZF 低 `0.00039107`，且逐 epoch R² 在约 `0.69893-0.69991` 间明显振荡。两格 combined mean delta 为 `+0.00239626`、2/2 正向，但离 `+0.005` 仍远。因此下一步先固定 epoch 顺序、只保留 full session balance 做单因素归因；同时已准备 0.5 强度 tempered balance，若 full balance 本身也不能转移，再减少对原始 session 分布的修正强度。

full balance + fixed order 的 fold-2 best/test 为 `0.69986868`（paired `+0.00116110`），说明固定顺序降低了轨迹噪声，却没有提高上限。对其 top-3 epochs 1/4/9 只平均 `var_linear.weight` 的 checkpoint soup 为 `0.69980550`，也未改善。由此 full balance 被降级；当前只保留最后一个 sampler-strength probe：0.5 tempered balance + fixed order。若它仍不超过标准 B16-ZF，则停止在 sampler 上继续调参，转向 loss/support consistency。

0.5 tempered + fixed order 最终为 `0.69981730`（paired `+0.00110972`），仍比标准 B16-ZF 低 `0.00047952`。因此 sampler 路线按预设停止。下一机制实验 B16-ZF-PC 回到原始 empirical/fixed sampler，在同 session batch 内把独立抽到的 calibration blocks 循环错配给相同 query，加入行为 prediction consistency；它直接约束 support 变化对输出的影响，同时保持 B3 路径冻结、external held-out 关闭。

B16-ZF-PC 在 fold 2 / seed 42 的 best/test 为 `0.70031464`（paired `+0.00160706`），只比标准 B16-ZF 高 `0.00001782`；consistency MSE 确实从约 `1.444e-6` 降到 `1.422e-6`，但没有转化为实际 R² 增益。因此不继续扫 consistency 权重。

B16-ZF-E03 保留 identity stabilizer、但把 `lambda_E` 从 `0.1` 降到 `0.03`。fold 2 / seed 42 在 epoch 1 达到 `0.70032972`（paired B3 `+0.00162214`），只比标准 B16-ZF 高 `0.00003290`，随后连续回落并在 epoch 6 early-stop。因此 objective conflict 不是当前主要上限，不扩展 E03。下一项 B16-ZFO 把 normalized log-Fano 从共享 nonlinear tail 前移到独立的 `D->W` identity-output residual，并把每维修正平滑限制为冻结 B3 identity RMS 的 25%；该结构保持 exact-B3 fallback，同时直接限制 reliability branch 对 decoder 输入的扰动。

B16-ZFO 在 fold 2 / seed 42 从首轮负增益缓慢恢复，到 epoch 6/test 为 `0.69900823`（paired B3 `+0.00030065`），但仍比标准 B16-ZF 低 `0.00128859`，因此按预设截止并否决。结论是 bounded direct-output correction 安全但过弱，可靠性信号仍应保留在 B16-ZF 的 latent fusion 位置。下一 probe B16-ZFS 不改变融合拓扑，而把 featurewise latent log-Fano 以 25% 强度向每个神经元的 cross-feature mean 收缩，检验高维 support noise 是否是剩余瓶颈。

B16-ZFS 在 fold 2 / seed 42 的 epoch-1 峰值为 `0.70012337`，比标准 B16-ZF 低 `0.00017345`，随后到 epoch 4 连续回落；固定 shrinkage 降低有效 feature-specific reliability 多于其去除的噪声，因此否决。预算内最后一个 probe 是 B16-ZFD：仅在训练期对 normalized-variance 输入施加 25% dropout，推理图仍为标准 B16-ZF，用最小改动检验 post-peak overfit 是否可被正则化。

B16-ZFD 在 fold 2 / seed 42 的最佳值仅为 `0.69872916`（paired B3 `+0.00002158`，相对标准 B16-ZF `-0.00156766`），epoch 3 又降到 `0.69832194`，因此否决。四个最终判别 probe 的峰值分别为：E03 `0.70032972`、ZFO `0.69900823`、ZFS `0.70012337`、ZFD `0.69872916`；标准 B16-ZF 为 `0.70029682`。没有候选达到预设的 `standard + 0.0005` 复制门槛。

本阶段最终结论不是“B16-ZF 无效”，而是“稳定但效应不足”：标准 B16-ZF 在三个内部配对 cell 中 3/3 为正，mean delta `+0.00260333`、worst delta `+0.00132567`，但没有达到 `+0.005` meaningful-effect gate。继续围绕同一 log-Fano residual 做 sampler、loss weight、shrinkage、bound 或 dropout 微调的边际信息价值已经很低；在提出新的信息源或更强验证设计前，不追加 fold/seed，也不再次触碰 external held-out。
