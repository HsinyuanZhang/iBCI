# Carrier Adaptive v3：源端重复面板方法

本文说明 `carrier_adaptive_v3` 的统计 carrier 构造、源端诊断和与冻结 v2 训练器的衔接。它只描述方法和可复现实行规则；不报告任何实际 R²、性能数值或目标端结论。

## 统一动机：由任务标签到神经元响应 profile

carrier 不是行为标签的复制，而是以可用的行为/任务标签组织每个神经元的响应 profile，再构造成可随校准部署的稳定低维坐标。结构明确、低维的运动任务可直接拟合调谐坐标；对 M1/H1 的相关多维输出和有限校准，则还需估计稳定信息与重复面板变异，再作自适应压缩和收缩。这里的“简单”或“复杂”只描述建模处境，不由标签维数推出方法优劣；实际 profile geometry 才是分析对象。

| 场景 | 标签与实际 carrier profile | 构造原则 |
| --- | --- | --- |
| M2 | native finger velocity 为 2D；`MOVE-T4=[a,c,sqrt(a²+c²),b]` 由 trial target angle 的 cosine tuning 得到，并作 source z-score（[plan](../../tfpd_exploration/src/m2_dual_track_v1/plan.py#L61)、[T4 features](../../streaming_calibration_exp/src/data/falcon_t4_features.py#L82)） | 低维运动调谐坐标，不是直接复制 2D 标签。 |
| DANDI688-like | behavior 为 2D（两个坐标的具体语义未在此确证）；`profile_m10=[a_R700,c_R700,m_R700,b_R700-b_H300]` 由前十个 rewarded trials 的 target direction/事件窗口拟合（[descriptors](../../sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/descriptors.py#L164)、[production](../../sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/production.py#L425)） | source normalizer 与 reliability mask；不消费 dense velocity 标量，也不是复制 2D 标签。 |
| M1 / H1 | M1 为 EMG16；H1 为 7D velocity 派生的 14 个 signed features | 以下用共同的重复面板统计估计稳定信息与变异。 |

PCA-Wiener 偏重主导总方差及其轴间幅度关系；SNR-Wiener 偏重经 noise 归一化后的可靠方向。两者遵守同一 source-only 统计原则、却是不同估计器，须经 inner source 诊断与既定确认流程选择，不能由任务名称、标签维数或全局 trace 比单独决定。

## 输入、原始坐标与面板

每个 source session 生成一个数组 `X_s`，形状为 `[P_s, U_s, K]`：`P_s` 是 panel 数、`U_s` 是原有神经 unit 数、`K` 是 raw feature 数。所有协方差估计均在 source 数据上进行，并令 session 和神经 unit 等权，因而不会因某个 session 的 panel 数或 unit 数更多而获得额外权重。

在每个 panel 内，对时间 bin `t` 和神经 unit `u`，rate 的共同标准化及第 `k` 个 raw feature 为：

\[
z_{tu}=\frac{r_{tu}-\operatorname{mean}_{\rm panel}(r_u)}
{\sqrt{\max(\operatorname{mean}_{\rm panel}(r_u)\,dt,1)}/dt},
\qquad
R_{uk}=\frac{\sum_t w_{tk}z_{tu}}{\sum_t w_{tk}+10}.
\]

也就是说，rate 在**各 panel 内**居中，且 `dt` 是 M1 的 `0.02` 秒、H1 的 `0.1` 秒。M1 的 raw feature 为 16 个 EMG 维度，权重是 `w_tk=max(EMG_tk,0)/source_RMS_k`。M1 panel 是 `[0,10), [10,20), …, [300,310)` 的十 trial 组，首个 panel 是部署校准面板。

H1 的 raw feature 为 14 个 signed-velocity 维度，即 7 个速度维度的正、负两个 softplus 权重。令 `x_tk=velocity_tk/source_RMS_k`，两组权重依次是 `softplus(x_tk)` 与 `softplus(-x_tk)`；它们代入上式的 `w_tk`。H1 panel 是 source training trial values（去除最后两个 native validation trials）中连续的三 trial 组；首组是部署校准面板，无法组成完整三 trial 组的尾部 source trials 被丢弃。

`source_RMS` 的拟合范围先于 panel 构造确定。M1 使用所有声明 source session 的 `[0,310)` trial 范围内 rectified EMG。H1 使用所有声明 source session 的 source training trials，即排除最后两个 validation trials，但**包括**后来因不足三 trial 一组而不能进入 panel 的 training 余数；之后才在 panel 构造时丢弃这些余数。相较 v2 H1，本轮 `behavior_rms` 的拟合范围也已扩大。因此本轮 decoder 的任何差异属于完整 adaptive carrier 的变化，不能单独归因于 SNR projection。

实现：[panels.py](../scripts/carrier_adaptive_v3/panels.py)。bundle 将 `behavior_rms`、面板、首面板、trial groups、来源 SHA-256 写入 NPZ 与 JSON receipt，并对加载的数组逐项哈希核验。

## 重复面板协方差

对 session `s`、unit `u`，定义 panel 均值和重复面板样本协方差：

\[
m_{su}=\frac{1}{P_s}\sum_p X_{spu},\qquad
N_{su}=\frac{1}{P_s-1}\sum_p(X_{spu}-m_{su})(X_{spu}-m_{su})^\top.
\]

代码先在 unit 内求 `N_{su}`，再按 unit、session 等权平均：

\[
N_s=\frac{1}{U_s}\sum_uN_{su},\quad
N=\frac{1}{|S|}\sum_sN_s,\quad
\mu=\frac{1}{|S|}\sum_s\frac{1}{U_s}\sum_um_{su}.
\]

以全局均值为中心的面板均值协方差为：

\[
C_{\rm mean}=\frac{1}{|S|}\sum_s\frac{1}{U_s}
\sum_u(m_{su}-\mu)(m_{su}-\mu)^\top.
\]

重复测量的有限 panel 校正为 `avg_s(N_s/P_s)`，并定义 signal covariance：

\[
C_{\rm signal}=\operatorname{PSD}\!\left(C_{\rm mean}-
\frac{1}{|S|}\sum_s\frac{N_s}{P_s}\right).
\]

其中 `PSD` 对称化后将负特征值截为零。实现：[statistics.py](../scripts/carrier_adaptive_v3/statistics.py)。

## Noise shrinkage 与四维 Wiener carrier

在候选集合 `rho ∈ {0, 0.25, 0.5, 0.75, 1}` 中，noise 向球形目标收缩：

\[
N_\rho=(1-\rho)N+\rho\frac{\operatorname{tr}(N)}{K}I.
\]

对每个 leave-one-source-session-out split，以训练 sessions 的 `N_ρ` 正则化并在 held session 上最小化 Gaussian 风险：

\[
\log\det(N_{\rho,\rm reg})+
\operatorname{tr}(N_{\rho,\rm reg}^{-1}N_{\rm held}).
\]

特征值下限为 `max(1e-6 * trace(N) / K, 1e-12)`。只有一个 source session 时没有 LOO split，按规则使用 `rho=1`，所有 LOO score 记为 `null`。选出的单一 `rho` 用于该次拟合的 regularized noise `Nreg`。

两种方法都固定输出四列；若 `K<4`，其余列为零，并固定每个 loading 的符号。两法分别为：

| 方法 | 方向与 shrinkage | 输出 transform |
| --- | --- | --- |
| `pca_wiener` | 取 `Csignal + N` 最大四个特征向量 `B`；`α_i=(b_i^T Csignal b_i)/(b_i^T(Csignal+N)b_i)`，截在 `[0,1]` | `B diag(α) / sqrt(mean(top4 total variance))` |
| `snr_wiener` | 对 `Nreg^{-1/2} Csignal Nreg^{-1/2}` 取最大四个特征对 `(λ_i,v_i)`；`α_i=λ_i/(1+λ_i)` | `Nreg^{-1/2} V diag(λ/(1+λ)^{3/2})` |

raw 向量 `x` 的四维 carrier 是 `(x-μ) @ transform`，保存为 `float32`。诊断用的 raw-space Wiener reconstruction 不用 transform 的伪逆，避免抵消 shrinkage：

\[
\hat{x}=\mu+(x-\mu)R.
\]

PCA 的 `R=B diag(α)B^T`；SNR 的 row-vector 形式为
`R=Nreg^{-1/2} V diag(α) V^T Nreg^{1/2}`。projection receipt 保存 `mean`、transform、reconstruction、三种协方差、`rho`、LOO scores、eigenvalues、reliability、effective rank、zero axes 和数组哈希。

## 源端诊断与适用范围

[diagnose.py](../scripts/carrier_adaptive_v3/diagnose.py) 只接受精确的 inner source roster：M1 为 `ses-20120924`、`ses-20120926`；H1 为截至 1925-01-15 的九个 session。主诊断按 source date 留出：M1 将每个 session 视作一个日期，H1 从 `ses-YYYYMMDD` 前缀取日期。每个 held date 上，用其余日期拟合两个 projection，再用 held panel `p` 重建同一 unit 的其余 panels 均值。

每个 raw feature 的误差按训练 fold 的
`max(diag(Csignal+N), max(mean(diag(Csignal+N))*1e-6, 1e-12))`
归一化，然后 feature、unit、panel、session、date 逐层等权平均。报告 raw panel、constant training mean、PCA 与 SNR 的误差和相对比值，方法选择规则是最小的 equal-date 归一化 reconstruction MSE；绝对差不超过 `1e-12` 时选 PCA。

另有 full-inner-fit 的首 panel 到其余 panels 均值诊断，以及 first-panel 和 rest-mean 的四维投影重复性。后者先在 unit 维度上逐轴中心化，才计算 Pearson 与 row cosine，避免全局均值造成的常数相关伪象。该脚本不做 target I/O 或 target probes，也不持久化 plan。

该诊断是**共享 source RMS 的 held-date projection diagnostic**，不是 decoder proof，也不是完全独立的 end-to-end CV。尤其，`behavior_rms` 在建 bundle 时用所有 source training dates 拟合；full-inner 首 panel 诊断还使用了包含该 session 的 full fit。

## 目标端、训练桥与审计边界

目标端只将已冻结的 source projection 应用于 support raw：M1 只使用 target 的 M10（前十个 trials），H1 只使用 target 的前三个 support trials（M3）。目标 query labels 不进入 carrier 拟合或选择。实现：[panels.py](../scripts/carrier_adaptive_v3/panels.py)、[prepare_pack.py](../scripts/carrier_adaptive_v3/prepare_pack.py)。source pack、projection receipt、数组、source gate 均通过 SHA-256 绑定；target surface 在读取 target support 前校验这些 source 绑定。

[training_bridge.py](../scripts/carrier_adaptive_v3/training_bridge.py) 只生成 source-only 的不可覆盖命令计划，实际训练由冻结 v2 trainer 的新子进程执行。可复用的固定 scheduler horizon 是 M1 24 epochs、H1 32 epochs；改变 max epochs、scheduler、optimizer、carrier/model injection 或 early-stop 规则时，必须按 [training bridge README](../scripts/carrier_adaptive_v3/training_bridge_README.md) 复制为新的 v3 实现，而不能复用 v2 轨迹。

source curve 的 checkpoint 选择只比较 source validation：按 `equal_date_mean`，缺失时按 `equal_session_mean`，取最大值；并以较早 epoch 打破平局。bridge receipt 会记录 source-only selection 与固定 horizon。用于与既有 source authority 配对的 source 数据复用流程见 [reuse_source_data.py](../scripts/carrier_adaptive_v3/reuse_source_data.py)：它保持 source authority，并仅替换 carrier 数组，同时核验其余数组未变。

### 本轮方法级确认与锁定边界

以下是本轮已冻结的方法确认规则，不是上述静态脚本自动推导出的性能结论。首先以 source 诊断选择 primary adaptive 方法。随后在**较早的留出日期**进行 decoder confirmation：若 source-selected 方法的 inner R² 未超过对应 v2 profile，则测试预先声明的 PCA 备选。仅在实际测试过的 adaptive 方法之间，以较高的 inner 分数确定方法；一旦锁定，outer 阶段只用 outer source 数据重新拟合该方法。最后日期不再用于选择方法或 checkpoint/epoch。

“target query labels 不进入选择”指单次 projection/训练拟合、checkpoint 选择，以及最终 outer 日期的决策都不使用该日期的 query labels；它不排除上述较早 inner 留出日期被用于 outer 开发阶段的预先声明方法确认。这里不报告任何该确认步骤的实际分数或比较结果。

## 统计限制

- 这些 panel 是不同时段、不同 trial group 的摘要，并不等于在相同刺激条件下的独立重复测量；`N` 因而是经验重复面板变异，不是“真实生物噪声”估计。
- `Csignal` 的有限 panel 校正、unit/session 等权和 PSD 截断是经验近似；PSD 截断会引入非负方向偏差，相关 unit、panel 结构、漂移和覆盖差异也会影响它。
- 若 rank-4 signal reliability 很低，Wiener reconstruction 可以接近 training mean；因此必须同时看 raw 与 constant baseline，不能把低 reconstruction MSE 单独解释为有效 signal retention。
- source-date 诊断只约束 projection 的 source 稳定性，且共享 source RMS；它不证明 target 泛化、decoder 性能或真实端到端独立验证。
