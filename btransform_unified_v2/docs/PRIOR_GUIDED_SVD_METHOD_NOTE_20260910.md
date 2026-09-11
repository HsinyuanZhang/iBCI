# 行为先验条件响应与 SVD 压缩：方法说明

## 核心回答

当前方法最准确的标签是：**行为条件定义 response matrix，再由 SVD 压缩**。SVD 本身不是随机分解；当前实现调用 `np.linalg.svd`，对给定输入矩阵给出确定的全 SVD 与相应的 rank approximation。随机化 SVD 算法当然存在，但不是这里的实现。奇异向量的整体正负号不唯一，重复奇异值时对应子空间内的基也不唯一；这不表示投影是随机的。实现以明确的 sign canonicalization 固定可部署表示。

EMG 和 H1 运行速度不是同一概念。共同的是表示顺序：先以任务定义的行为条件构造每个 unit 的 conditional-response row，再在 source rows 上冻结一个低维、带符号的坐标系。SVD 优化本身没有加入约束或权重；行为知识在它之前进入条件映射与 response estimator。

## 当前三条线路

| 线路 | 进入分解前的对象 | 行为条件与结构假设 | 压缩或后续模型 |
| --- | --- | --- | --- |
| M1 current | pooled unit conditional-response rows (A_{u,1:16}) | 16 路 rectified、source-RMS-scaled EMG 条件 | 对未中心化 rows 做 SVD4；随后冻结符号并按配置标准化 |
| H1 signed-state | pooled unit conditional-response rows (A_{u,1:14}) | 7 路 velocity 各拆为正/负 signed-softplus 条件 | 对未中心化 rows 做 SVD4；随后冻结符号并按配置标准化 |
| 历史 rSyn3 | EMG time-by-16 matrix | 非负 EMG 进入 NNMF | NNMF3 activations/dictionary；随后另行作 unitwise ridge with intercept |

M1 的 SVD 作用于已经估计出的 unit response，不作用于 raw EMG，也不作用于 raw neural time series。对每个当前 support，先在 neural 时间轴上计算 unit mean \(\mu_u\)，以 \(z_{tu}=(r_{tu}-\mu_u)/(\sqrt{\max(0.02\mu_u,1)}/0.02)\) 标准化 rate；再以 \(w_{tm}=\max(EMG_{tm},0)/RMS_m\) 得到 \(A_{um}=\sum_t w_{tm}z_{tu}/(\sum_t w_{tm}+10)\)。因此 response 本身已扣除该 unit 当前 support 的时间均值并按 Poisson 尺度标准化。随后 SVD 直接作用于 pooled unit×16 \(A\) matrix，不再进行跨-unit 的列中心化；投影后的 source normalization 又是单独一步。若对 pooled \(A\) 另作列中心化，会定义不同、更接近 PCA 的对象。M1 和 H1 的 primary definitions 分别在 [m1_muscle_profile.py](../scripts/carrier_profile_v2/m1_muscle_profile.py) 与 [h1_profiles.py](../scripts/carrier_profile_v2/h1_profiles.py)。H1 的 signed-state 方法维持现有主线，本说明不启动任何 H1 新实验或替代方法。

## NNMF 不是严格的“三维加均值 baseline”

历史 NNMF 将非负 EMG matrix 分解为三个非负 activation 与非负 dictionary。之后的 unit model 是独立的、无符号约束的 joint ridge：

\[
r_u \approx b_u + \beta_{u1}Z_1+\beta_{u2}Z_2+\beta_{u3}Z_3.
\]

\(\beta\) 和 \(b\) 都不由 NNMF 的非负约束决定，且

\[
b_u=\bar r_u-\beta_u^\top\bar Z.
\]

因此 \(b_u\) 通常不等于 unit mean rate。把该线路口语化为“三个非负维度加 baseline”可以帮助记忆，但不能把它与 rank-three SVD 加第四坐标视为同一模型。

## 当前“三个 response 坐标加一个 level 坐标”候选

当前 M1 candidate 保留原 carrier 的前三个 SVD 坐标逐字节不变，只以 source-scale-matched arithmetic unit mean rate 替换第四坐标。前三维是上述按当前 support mean 去中心、按 Poisson 尺度标准化的 conditional-response matrix 的前三个 SVD 坐标；SVD 本身不再对 pooled unit rows 作跨-unit 列中心化。第四维补入显式原始 mean-rate level。它不是 ridge intercept，也不等价于 NNMF；mean activity 仍进入前三维的 Poisson scale，因此不能声称前三维与 level 严格独立。

candidate 的配对 seed42 已完成完整 24-epoch 公开 HO3 scan：candidate 在 e4 选中为 `0.5876280069351196`，original42 在 e3 选中为 `0.5690750181674957`，各自选中 checkpoint 的差为 `+0.0185529887676239`；candidate fixed3 为 `0.5861498316129049`，固定 e24 差为 `-0.007281084855397468`，并在 e22–e24 低于 original42。三个 completed original seed 的 selected mean 为 `0.5625720255904727`，sample SD（ddof=1）为 `0.00945691268292847`；fixed3 mean 为 `0.5550560355186462`，sample SD 为 `0.012685350092674524`。candidate 的两个 seed42 值高于三个 original selected 值，但 candidate 只有一个 seed，因而不提供 replicated candidate variance、一般性、因果或 official claim。曲线和逐 epoch 摘要见 [summary_all4_v1](../results/m1_muscle_r100_multiseed_v1/summary_all4_v1/README.md) 与 [curves.png](../results/m1_muscle_r100_multiseed_v1/summary_all4_v1/curves.png)。这些结果不证明 SVD 普遍更好、NNMF 等价、后期训练稳定或因果机制。

## 哪些部分是先验、正则化或约定

- **当前实现的条件与约定**：task-specific condition mapping、非负 EMG 权重、每个当前 support 的 rate mean centering、Poisson scaling、occupancy pseudo-count、source-only fit、冻结 basis、sign canonicalization，以及由具体 carrier 配置决定的投影标准化。
- **可选的先验化方向**：noise-floor 或其他 scaling、以及显式 level 分配都可以是用于稳定性或尺度的选择；不能仅因概念上可用，就写成每条当前线路均已采用的正则。
- **当前 candidate 的结构选择**：三个标准化 conditional-response 坐标加一个显式 mean-level 坐标；不声称两者严格独立。

因此 “prior-guided SVD” 不应暗示 SVD objective 自身已经 constrained 或 weighted，也不表示 SVD 普遍优于 NNMF、或 NNMF 普遍不适用。

## 有效但非当前实验的替代问题

保留 M1 全 16 维或 H1 全 14 维 conditional responses 在概念上成立，但会改变 carrier interface 与 decoder width，并非当前固定接口。更接近旧 NNMF 的受控比较会是：以 behavior PCA/SVD3 构造 basis，再接同形式的 ridge-with-intercept。仓库中 [syn3.py](../../tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py) 已有 `fit_source_pca`、`project_basis` 与 `fit_unit_ridge` 的 signed-PCA route；这里只说明其概念关联，不报告结果，也不启动运行。
