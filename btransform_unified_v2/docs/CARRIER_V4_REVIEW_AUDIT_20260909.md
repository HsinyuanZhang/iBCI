# Carrier v4 审阅审计：P0 优先的数学与实现边界（2026-09-09）

## 结论与 P0 决策

先完成 **P0 exact M1** 的字节级复现与全 pipeline 核验，再比较任何 v4 carrier 或 fusion。不能以 cosine-1、局部 frontend 等价、或单个 token fold 替代 byte-exact baseline：只有相同模型、bank、normalizer、source roster、sampler、checkpoint 选择和完整 scoring pipeline 才能复用 checkpoint 或解释 paired delta。原 rSyn 的 source3 dictionary、alignment reader 和 source normalizer 是冻结合同；新 core 的 source4 fitting 不能自动替换它们。

P0 应同时把“无增益”与“非劣”分开：无增益是未超过零差值；非劣须在执行前声明 endpoint、paired comparator 和容忍度，例如 `new - baseline >= -0.01`。后者不能由运行后的正负波动追认。concat、P16、P32 都应在同一 P0 合同下审计；P32 的 function-preserving initialization 只证明初始函数相同，不证明训练后非劣。

## 核验过的估计器

### rSyn 的 ridge 与截距

rSyn 的 source NNMF 在 source EMG 上拟合，部署时用冻结 dictionary 作 NNLS；每个 unit 的 carrier 是三个 ridge coefficient 加一个 intercept。代码构造

\[
D=[\mathbf1,Z],\quad G=D^TD/n,\quad
\hat\beta=(G+\operatorname{diag}(0,1,1,1))^{-1}D^Tr/n.
\]

见 [syn3.py](../../tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py#L73)、[syn3.py](../../tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py#L143) 与 [syn3.py](../../tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py#L164)。所以截距 penalty 为零；把等式乘以 `n` 后，斜率对应的是未归一化 normal equation 的 `nI` penalty，**不是** `I`。这也是 v4 `latent_state3_ridge_intercept` 若声称与 rSyn 同 law 时必须保持的尺度。三个 slope 加一个 intercept 恰为四维；若候选有 `q` 个可学习 state column 再加 intercept，约束是 `q+1<=4`，即 `q<=3`。这项维数约束不适用于 raw behavior 的 16 或 14 列。

### H1 signed-state 条件均值不等于 joint ridge

H1 目前的 signed-state14 先计算 `W=[softplus(v/RMS),softplus(-v/RMS)]`，再按列独立计算

\[
R_{uk}=\frac{\sum_t W_{tk}z_{tu}}{\sum_tW_{tk}+n_0},\quad n_0=10.
\]

实际代码见 [h1_profiles.py](../scripts/carrier_profile_v2/h1_profiles.py#L143) 和 [h1_profiles.py](../scripts/carrier_profile_v2/h1_profiles.py#L177)。该式正是每个 `k` 的 weighted-expanded-state objective

\[
\min_R\ \sum_{t,k}W_{tk}(z_{tu}-R_{uk})^2+n_0\sum_kR_{uk}^2
\]

的解。它与 joint ridge

\[
\min_\beta\ \sum_t(z_{tu}-\sum_kW_{tk}\beta_k)^2+\lambda\lVert\beta\rVert_2^2
\]

不同：后者 normal equation 有完整的 `W^TW` 交叉项。只有权重行 one-hot 且状态列互不重叠时，二者才可退化为逐列问题；softplus 的正负状态列在每一行都为正且彼此重叠，故不满足该条件。将这两种估计器都称作“ridge conditional mean”会掩盖实质差异。

### M2 / 688 不是 review 所称的 one-hot ridge

M2 的实际 T4 是对合法 trial angle 组成 `X=[1,cos(theta),sin(theta)]`，检查 rank 3 后以 `np.linalg.lstsq(X,rates)` 拟合，输出 `[a,c,sqrt(a²+c²),b]`；没有 ridge、one-hot design 或 pseudocount。见 [falcon_t4_features.py](../../streaming_calibration_exp/src/data/falcon_t4_features.py#L76)。688 对 first-M10 的**每方向均值**构造相同 cosine design，再作无正则 least squares；其输出取 R700 的 `a,c,m` 和 `b_R700-b_H300`。见 [unit_side_features.py](../../sua_exploration/mc_maze/unit_side_features.py#L728) 与 [descriptors.py](../../sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/descriptors.py#L133)。

因此，M2/688 可被抽象为“状态条件响应后再投影”，但不能宣称与 H1 softplus conditional mean 或 one-hot ridge 精确相同。若未来统一实现，回归目标应是 `n0=0`、不标准化时对已有 M2/688 输出逐值复现；不应以口头的 one-hot 等价取代该核验。

## 审阅意见的限制

1. **row-weighted centering 不是自动的无判别改进。** 按 row 或 unit 去均值可移除 session-wide / unit-wide common amplitude；若该幅度本身携带任务相关 recruitment 或 gain，它也是有效信号。必须作为独立消融，与不中心化版本共用 source/target、normalizer 和 endpoint；不能预设其去除的都是 nuisance。
2. **split-half 指标必须定义量纲与边界。** 若报告 Pearson `r`，它是无量纲相关而不是 R² 或 noise variance；如需 Fisher 聚合，应先用 `r_c=clip(r,-1+eps,1-eps)` 后计算 `atanh(r_c)`，并记录 `eps`、聚合层级和任何零方差处理。若使用 Spearman–Brown，`2r_c/(1+r_c)` 仅是在同质 split 假设下的相关性外推，`r` 接近 `-1` 时不应硬称“reliability”。不要把新的 split-half normalization 命名为 Wiener：v3 已有 source repeated-panel covariance、LOO noise shrinkage 与 Wiener reliability shrinkage，见 [statistics.py](../scripts/carrier_adaptive_v3/statistics.py#L84) 和 [statistics.py](../scripts/carrier_adaptive_v3/statistics.py#L131)。
3. **不能从 source reconstruction 或 initialization 等价推导 decoder 增益。** v3 的 reconstruction MSE 与 decoder 排序并不等价，已有结果也明确作此限制，见 [CARRIER_ADAPTIVE_V3_RESULTS_20260909.md](CARRIER_ADAPTIVE_V3_RESULTS_20260909.md#L108)。

## 可执行的 v4 调整建议（最多三项）

1. **P0 exact M1 gate：** 先逐 artifact 验证 source3 NMF dictionary、NNLS activation、unit ridge、normalizer、bank、frontend 和 final prediction；通过后才允许把 P0 baseline 作为 fusion 或 carrier 的 comparator。
2. **单一主 carrier 消融：** 若推进 `latent_state3_ridge_intercept/per_column`，将其明确为“NNMF/NNLS state basis + raw-Hz unit ridge + intercept”，与 H1 conditional mean、v3 PCA/SNR-Wiener 分开命名。M1/H1 各自 adapter 只能提供合法的 `fit_behavior` 与 M10/M3 support，core 不得扩展 source 范围。
3. **预注册 fusion 表：** 对 concat、P16、P32 固定 seed、训练与 selected rule；先验收 concat/P16 byte identity、P32 folded initial function identity，再报告固定 endpoint 与 selected endpoint。若目标是非劣，预先采用 `-0.01` paired margin；若目标是增益，使用严格大于零的独立判据。

本审计只基于仓库实现和上述代数推导，不引用未核验的论文或声称任何新方法有效。
