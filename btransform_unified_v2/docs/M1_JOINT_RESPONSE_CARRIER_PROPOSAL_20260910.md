# M1 联合响应 carrier：讨论提案（2026-09-10）

**状态：`PROPOSED_NOT_IMPLEMENTED`。** 本文只记录下一步 M1 carrier 的候选设计和讨论边界；应先向用户说明候选设计，当前未实现、未训练、未评分或排队。

## 问题与建议

当前 `meanrate4` 是现有的最小 paired base：在既有回执的各自 complete-24、visible-HO3 earliest-max 比较中，candidate-s42 为 0.587628，FULL-s42 为 0.569075，ACTIVITY_ONLY-s42 为 0.561827；candidate 的选择 epoch 是 4，其余两者为 3。historical rSyn 为 0.6544709205627441，比 candidate 高 0.06684291362762451；candidate 相对 ACTIVITY_ONLY 为 +0.025800953308741215。该信息来自已完成的 receipt-only 审计 [`root_m1_carrier_context_comparison_v1.json`](../results/m1_muscle_r100_multiseed_v1/root_m1_carrier_context_comparison_v1.json)，不构成新的官方分数、NMF/SVD 因果归因或机制结论。

建议讨论一个后继候选：保留 `meanrate4` 的全部接口和第四列，只把前三列从“逐 muscle 的边际 cross-moment”改为“现有三条 muscle-SVD feature 上的 joint ridge coefficient”。它不读取新数据、不改 M10/reader/source/decoder、仍为 `[64,4]`，但会利用相同 M10 behavior rows 中当前最终 carrier 未显式保留的三维二阶统计量；这只是待检验候选，不是性能或统一成功的结论。

**跨主线硬边界。** H1 当前主线 `signed_state14` 也是边际 conditional-response law：`weights.T @ z / (occupancy + 10)`，没有 (G^{-1})；源码见 [`h1_profiles.py:177`](../scripts/carrier_profile_v2/h1_profiles.py#L177)–[`189`](../scripts/carrier_profile_v2/h1_profiles.py#L189)。只有未选主线 `velocity7` 才是 joint ridge，见同文件 [`163`](../scripts/carrier_profile_v2/h1_profiles.py#L163)–[`174`](../scripts/carrier_profile_v2/h1_profiles.py#L174)；候选名称也由 [`21`](../scripts/carrier_profile_v2/h1_profiles.py#L21)–[`25`](../scripts/carrier_profile_v2/h1_profiles.py#L25) 固定。本 M1 候选借鉴历史 rSyn3 的联合估计，只统一“calibration-support → per-unit behavior-response summary”这一抽象，**不与 H1 主线使用同一个估计公式**；若目标要求公式一致，就不能声称本候选满足该目标。

主比较将是未来实际进行的两个方法各自在**完整 24 epoch** visible-HO3 曲线中按 earliest maximum 选择的 paired s42 比较。固定 epoch 3 只能作为补充性的同 epoch 描述，不能作为成功门槛。

## 当前最小基线与拟议估计

令 session $s$ 的 M10 rows 为 $n_s$，$W_s\in\mathbb R^{n_s\times16}$ 为现有 rectified/source-RMS EMG weights，$Z_s\in\mathbb R^{n_s\times U}$ 为现有按 unit 的 rate centering 和 Poisson $q$ 缩放后的 rate，$V\in\mathbb R^{3\times16}$ 为当前 parent carrier 的冻结 SVD 前三条 basis。保持：source roster、reader、M10、$W$、$Z$、$V$、当前第四列最终 `float32` 值都不变。

现有前三维的 raw law 是：

\[
D_s=\operatorname{diag}(\sum_t W_{s,tm}+10),\quad
R_s=Z_s^\mathsf TW_sD_s^{-1},\quad
P_s=R_sV^\mathsf T.
\]

定义派生 feature：

\[
F_s=n_sW_sD_s^{-1}V^\mathsf T,\qquad
F_{c,s}=F_s-\mathbf1\bar F_s^\mathsf T.
\]

由于 $Z_s$ 已按 session/unit 中心化，$P_s=Z_s^\mathsf TF_{c,s}/n_s$ 精确成立。只从 source4 M10 rows 固定一个 common input RMS：

\[
g^2=\frac{\sum_{s\in\mathrm{source4}}\lVert F_{c,s}\rVert_F^2}
{3\sum_{s\in\mathrm{source4}}n_s},\quad
F_s^*=F_{c,s}/g,
\]

\[
G_s={F_s^*}^\mathsf TF_s^*/n_s,\quad
H_s={F_s^*}^\mathsf TZ_s/n_s,\quad
B_s=(G_s+I_3)^{-1}H_s.
\]

输出归一化的讨论版唯一公式如下。令 $C_s=B_s^\mathsf T$，合并 source4 中 parent 的同一批全部 eligible units，令总数为 $M$：

\[
\mu_B=\frac1M\sum_{s,u}C_{s,u:},\quad
\rho_B^2=\frac{\sum_{s,u}\lVert C_{s,u:}-\mu_B\rVert_2^2}{3M},\quad
\rho_{\rm parent}^2=\frac{\sum_{s,u}\lVert T^{\rm parent}_{s,u,0:3}\rVert_2^2}{3M},\quad
\alpha=\rho_{\rm parent}/\rho_B.
\]

前三列为 `float32(alpha * (C_s - mu_B))`；第四列为 parent `meanrate4` 最终 `float32` 的逐字节 copy，不重新估计。$g$、$\rho_B$、$\rho_{\rm parent}$ 均在 source-only `float64` 中计算一次并冻结，任一非有限或为零即失败；前三列只按预先定义的浮点容差验证 aggregate RMS 匹配，不声称 byte identity。这里 $\lambda=1$ 是以 source average feature variance 为单位的固定 baseline，而非最优参数，也不搜索、不按 HO3 选择。输入 $g$ 会改变有效 ridge shrinkage，输出 $\alpha$ 会影响优化数值尺度，不能把它们说成天然无影响的后处理。

## 能说与不能说的内容

- $V$ 固定，但 $D_s^{-1}V^\mathsf T$ 随 session 的 M10 coverage 改变；因此三个输出坐标固定，原始 16→3 有效方向并不跨 session 固定。
- 该设计保留现有 coverage normalization，并不消除 coverage 依赖；它把已有的 session-local normalization 与 $G_s$ 的 joint covariance 一起用于 coefficient。
- 它不增加任何原始 calibration 数据、reader 输出、trial 或 carrier 维度；但确实使用同一 M10 data 的额外二阶统计 $G_s$。不能笼统称作“不增加 behavior information”。
- 若所有 $G_s$ 都相同，新旧前三维仅相差一个全局可逆线性 remap；若 $G_s$ 不同，remap 是 session-specific。两者都只是代数情形，尚未测量。
- 这是标准的固定派生 feature ridge，不是 reduced-rank regression：rank reduction 已由冻结的 $V\in\mathbb R^{3\times16}$ 在 ridge 前完成，ridge 本身未学习 response-side low-rank map。
- 它不是 full-16 EMG ridge；在 $P=RV^\mathsf T$ 投影中丢失的神经响应方向，不能仅靠该 3D ridge 恢复。

## 可讨论的原始方向

1. 当前 marginal muscle-SVD4；2. 已有前三列不动的 `meanrate4`；3. 本文的 3D joint-response ridge；4. frozen-​source-​denominator 的 3D ridge；5. full-16 ridge 后 SVD3；6. NMF3 ridge + intercept 的 rSyn-style 复现；7. signed PCA3 ridge；8. residualized mean-rate 第四列；9. intercept 替换第四列；10. ridge intercept 与 mean-rate 双路比较；11. source-global Gram；12. session-local Gram；13. shrinkage-to-source-Gram；14. trial-block covariance estimator；15. diagonal-only control；16. orthogonal-rotation control。

其中 4–16 只是讨论地图库，不应并行扩展为多臂搜索。

## Shortlist

| 候选 | 变化 | 讨论判断 |
| --- | --- | --- |
| A. `meanrate4` | 当前基线 | 已有回执上下文；保留为 paired base。 |
| B. 3D joint-response ridge + bytecopied mean-rate4 | 仅把前三列 cross-moment 改为固定 feature ridge | **当前讨论候选**：其表现尚未知道。 |
| C. frozen-source-denominator ridge | 冻结 $D_{\rm source}$ | 方向更跨 session 固定，但不再保持当前 $P_s=Z^TF_{c,s}/n_s$ 的最小重参数化。 |
| D. full-16 ridge→SVD3+intercept | 改为完整 16D joint regression | 更接近独立 muscle coefficient，但变量、正则化与方差风险同时增加。 |

## 当前讨论候选：B

候选 B 使用和 `meanrate4` 完全相同的 M10 rows、source4、SVD basis、reader、decoder 输入宽度及第四列，只替换前三列的统计估计式。它提出一个可讨论的问题：在同一三维 feature 和同一 M10 support 下，边际 cross-moment 是否值得改成联合 ridge coefficient；答案只能由未来实际进行的 paired 比较给出。

## 尚未执行的最小验证

1. **代数与 byte-copy 审计**：验证 $P_s=Z_s^TF_{c,s}/n_s$、矩阵 shape/finite、
   source-only normalizer 合同，以及第四列最终 `float32` byte identity；不读取 query、不训练。
2. **source-only $G_s$ 诊断**：按预先定义的方式报告 $g$、$G_s$ 的 eigenvalues/trace 和 shrinkage operator；这些结果不能用于搜索 $\lambda$、basis 或 scale。
3. **候选方向确定并完成前两项验证后**：一次 paired s42、complete-24、same-query、each-own-selected earliest-max 对比 B 与 A；不以 fixed e3 作为门槛，不扩为参数或多候选搜索。

## 两周 pilot（待讨论，不排队）

若讨论后决定推进，先固定本文公式并完成两项 contract/source 诊断，再进行一个 paired s42 complete-24；之后复盘，是否复种子另行讨论。本文不把这些步骤写成已排队工作。

## 风险与反对意见

最强反对意见是：候选 B 不增加 behavior 的原始输入，且只在既有 3D $V$ space 内作 covariance-dependent shrinkage；有限 M10 support 下的 $G_s$ 噪声可能让 session-specific transform 恶化稳定性。即便 $G_s$ 存在非对角项，也没有静态或既有回执证据表明其条件性、EMG 相关性或该 ridge 变换会提高性能；这些都必须留给未来实际进行的 paired 比较判断。

## 保留边际响应公式的替代方向（同为讨论态）

**状态：`SOURCE_GEOMETRY_DIAGNOSTIC_COMPLETED_CARRIER_NOT_IMPLEMENTED`。** 此方向不替换本文 joint-response 候选。已完成一次 source-only、in-memory centered-SVD 几何诊断；未保存新 basis 或 raw X，未进行新 carrier build/pack、目标或 HO query、decoder 或训练。

令 $X\in\mathbb R^{N\times16}$ 为四个 source session 的 pooled raw muscle-response unit rows，
$\mu=X^\mathsf T\mathbf1/N\in\mathbb R^{16}$ 为列向量，以及 $X_c=X-\mathbf1\mu^\mathsf T$。当前前三列的 basis 是

\[
V_u=\operatorname{Top3Eig}(X^\mathsf TX),\qquad
X^\mathsf TX=X_c^\mathsf TX_c+N\mu\mu^\mathsf T.
\]

尽管当前 basis 来自 uncentered SVD，随后 source projection centering 的输出实际为

\[
XV_u^\mathsf T-\mathbf1(\mu^\mathsf TV_u^\mathsf T)=X_cV_u^\mathsf T,
\qquad V_u\in\mathbb R^{3\times16}.
\]

该替代只把 basis-selection 改成 $V_c=\operatorname{Top3Eig}(X_c^\mathsf TX_c)$，输出随之为 $X_cV_c^\mathsf T$；$R=Z^\mathsf TW\operatorname{diag}(\sum W+10)^{-1}$ 的 marginal response law、rate centering、Poisson scale、source roster、reader、M10 和第四列 parent `meanrate4` 最终 `float32` byte copy 都保持不变。它仍处在与 H1 `signed_state14` 相同的边际 weight/pseudocount response 框架，而不是引入 $G^{-1}$。

这不代表全 pipeline 公式一致：H1 当前 source fit 也采用 uncentered compression（见 [`h1_profiles.py:219`](../scripts/carrier_profile_v2/h1_profiles.py#L219)–[`251`](../scripts/carrier_profile_v2/h1_profiles.py#L251)），只在 M1 改为 centered SVD 会使 source compression rule 不同。当前 M1 uncentered source fit、投影和 global-RMS normalizer 的代码路径见 [`carrier.py:123`](../scripts/m1_muscle_r100_v1/carrier.py#L123)–[`171`](../scripts/m1_muscle_r100_v1/carrier.py#L171)；现有比较背景仍只来自 receipt-only [`root_m1_carrier_context_comparison_v1.json`](../results/m1_muscle_r100_multiseed_v1/root_m1_carrier_context_comparison_v1.json)。

若 $V_u$ 与 $V_c$ 仅在同一 3D subspace 内作正交旋转，且使用相同 aggregate RMS，替代可能没有实质新信息；若两者改变了所选 subspace，centered SVD 可能更集中于 source row variation，也可能丢掉有用的 target 相对 source 均值偏移方向。第四列 mean-rate 不等价于这个 response-mean direction；source 方差也不是 query 预测价值。

不能说 rank-1 的 $N\mu\mu^\mathsf T$ 只影响一个轴，或说 centered SVD 一定腾出一个有用维度：rank-1 update 约束谱结构，却可旋转多个 top-3 eigenvectors。已有只读 [`diagnosis.json`](../results/carrier_v4/diagnosis_m1_source_v1/diagnosis.json) 中 `candidate_svd4.raw4_source4_energy_fraction[0]=0.9029601259547435` 是由 [`diagnose_m1.py:121`](../scripts/carrier_v4/diagnose_m1.py#L121) 的 `rawstack.std(0)` 平方得到的**中心化投影坐标方差**比例，而不是 source 均值能量占比；因此它不支持“均值浪费轴”。该旧投影坐标比例与下文 raw16 诊断的分母不同，不能混用。

**已冻结投影坐标的补充证据。** ROOT 当时只读取冻结 fit 和既有 source receipt，并做标量算术；结果封存在 [`root_existing_projection_mean_energy_v1.json`](../results/m1_muscle_r100_multiseed_v1/root_existing_projection_mean_energy_v1.json)（SHA-256 `6a4e6b9228a847bfa7083e4ccbc6a5aa4cb4adcb62e73305d39fa674ee51e011`，状态为 `COMPLETED_EXISTING_FROZEN_FIT_SCALAR_ARITHMETIC`）。它绑定现有 `fit.npz` 的 SHA `2fea6e917259294fdb8f040f007940550269f601c7cfe8dba6ab14a5420c5434`、[`fit.json`](../results/m1_muscle_r100_v1/carrier_official4/fit.json) 和上述 [`diagnosis.json`](../results/carrier_v4/diagnosis_m1_source_v1/diagnosis.json)：source4 roster 相同，重新由既有四轴 std 计算的 global RMS 与 frozen scale 在 $10^{-12}$ 容差内一致；该旧证据本身没有 new carrier、query projection、decoder 或训练。

对已冻结的四个 projected coordinates，它按

\[
\frac{\mathrm{mean}^2}{\mathrm{std}^2+\mathrm{mean}^2}
\]
计算每轴均值项占未中心化二阶矩的比例，得到 `[0.5468376239904473, 0.03905831797575799, 0.04979960448008927, 0.015040304073789233]`，四轴总体为 `0.5223235903068574`。第一轴的 mean 为 `0.6833532991340003`、std 为 `0.622075395387558`：均值项占该轴二阶矩的 54.684%，但去均值后的该轴仍占旧四轴中心方差的 90.296%。这些数值说明第一轴并非纯均值，不能据此推出 centered SVD 必然释放有用维度或改善性能；它的范围只限既有 4D projected coordinates，不是 full raw16 的 $\mu$ 能量。下文已补充 source-only raw16 geometry，但仍不能据此推断性能会提高或不会提高。

**source-only raw16 geometry addendum（已完成）。** ROOT 以 CPU 2、`CUDA_VISIBLE_DEVICES` 为空完成 [`root_centered_svd_source_geometry_v1.json`](../results/m1_muscle_r100_multiseed_v1/root_centered_svd_source_geometry_v1.json)（SHA-256 `49c72aeb9669900a44c406dc69d74977ee45864b6e2ba455357066eb6db90684`；helper SHA-256 `f12fed66fe9a7274ee362e086ef29916bfa39c6b3a50d117e87373935beff1c2`；`2026-09-10T04:12:10.770528+00:00`）。该诊断只读取四个 source session：trial `[0,310)` 的 source EMG movement rows 仅用于重建 source RMS/normalizer；trial `[0,10)` 的对齐 M10 EMG/rate movement rows 才用于 raw16 response，要求 EMG/rate trial IDs 精确相等并保留每 session 64 个 native rate columns。它在内存中重建 `raw X`（`256×16`）并计算 centered basis；没有保存 raw X 或新 basis，也没有读取 target/held-out paths、query labels 或 official test，未加载 model，未建立 carrier/pack，未运行 decoder 或训练。

ROOT 已读取完整收据，并核验 frozen scalar/spectral identities 与 hashes：原 source RMS、global normalizer 均重建通过，原 uncentered V4 projector gap 为 `0`。raw16 的全 16D $\mu$ mean-energy fraction 为 `0.5166039387513265`；这与上述旧四个投影坐标的 `0.5223235903068574` 分母不同，不能比较为同一比例。frozen first3 与 centered top3 的 principal angles 为 `[0.009042149947039722, 0.5627914672786672, 11.291465160941714]` degrees，projector gap 为 `0.27725140744894916`，所以 centered SVD 确实改变了所选子空间。可是 frozen first3 捕获的 centered variance 是 `0.9648916906074126`，centered top3 是 `0.9654944842469828`，仅提升 `0.00060279363957016496`（`0.060279363957016496` 个百分点）。这削弱“能释放大量有效维度”的动机；它不是性能结果，既不能推出会提高，也不能推出不会提高。该诊断不表示与 H1 whole pipeline 统一，也不把任一 centered-SVD carrier 方向写成已实现或已排队。

| 讨论候选 | 改变与未改变 |
| --- | --- |
| joint-response ridge | 改变 response estimator；在同一 M10 rows 上使用额外的 session-local $G_s$。 |
| centered-SVD basis | 保持 marginal response estimator；只改变 source compression basis-selection，不使用额外 target $G_s$。 |

用户所问“统一程度”仍待讨论：joint-response 分支改变 response estimator；centered-SVD 分支保留边际 response law、但改变 source compression rule；两者均不能据此声称与 H1 主线有完全相同的 whole pipeline。本文不为其中任一方向排队。
