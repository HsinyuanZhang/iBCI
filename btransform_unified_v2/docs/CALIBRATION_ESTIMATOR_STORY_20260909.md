# Calibration estimator：当前主线、共同结构与“无偏移”边界

## 决定与证据范围

当前保留两条已部署的 source-frozen 条件响应估计器：M1 的 `muscle_response16_svd4/global_rms` 与 H1 的 `signed_state14` 后接 source SVD4/per-column scale。H1 不再推进 `latent_state3_ridge_intercept/per_column` 的 NMF3 新方法；封存的 signed-state 仍是 H1 的方法。M1 的 rSyn3 joint ridge（含未惩罚截距）只保留作历史诊断和对照，不能被改名为当前 H1 主方法。

这个结论只说明估计器、实现和已有结果的边界。它不授权新的训练、评分、提交或资源调度，也不从公开 calibration 分数推断 private test 的 carrier 相对效应。

M1 muscle-response R100 的 e3 submission `582205` 已完成：official Held Out R² mean `0.6259910151098528`、Held In R² mean `0.7926300410539316`、Normalized Latency `0.14040520785877766`。官方结果见 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)，其选择在官方返回前已由完整公开 HO3 scan 固定。[M1 官方记录](M1_MUSCLE_OFFICIAL_RETRAIN_20260909.md) 同时确认 rSyn3 joint-D 的 `582150` 和同配置的 `582133` 均为 cancelled、没有 completed official score。因此，muscle 相对 rSyn3 的约 `-0.085` 仅是公开 HO3 的比较，不能写成 official 退化或 official 改善。

## 共同的估计器骨架

对一个 session 的 support bin `t`、行为状态 `k`、神经 unit `u`，两条主线均可写成下面的顺序：

1. 从行为建立非负权重 `w[t,k]`，并且该权重使用 source-frozen 的行为尺度。
2. 在当前合法 support 上，对每个 unit 的 rate 去均值，并以带一计数下限的 Poisson noise scale 缩放，得到 `z[t,u]`。
3. 计算每个 unit×state 的带 pseudo-count 的加权条件响应：

   \[
   A_{u k}=\frac{\sum_t w_{t k}z_{t u}}{\sum_t w_{t k}+10}.
   \]

4. 在 source unit rows 池化 `A`，做未中心化 SVD，保留四个确定符号的分量；目标 session 只能投影到该 frozen SVD4。
5. 以 source projected-row mean 作中心化，再使用该任务自己的 frozen scale 归一化，得到供 decoder 使用的四维 carrier。

这里的“条件响应”是描述统计：它并不识别互相相关的行为状态各自的因果效应，也不保证跨 session 的 decoder 增益。

## M1：16 路 rectified muscle state 与 global RMS

M1 令 `e[t,m]` 为对齐 EMG，source muscle RMS 为 `s[m]`，权重为

\[
w_{tm}=\max(e_{tm},0)/s_m.
\]

对于 20-ms bin，令 `r[t,u]` 为 rate，`\bar r_u` 为当前 support 的 rate mean，`\bar c_u=.02\bar r_u`。实现使用

\[
q_u=\sqrt{\max(\bar c_u,1)}/.02,\qquad
z_{tu}=(r_{tu}-\bar r_u)/q_u,
\]

再按共同骨架得到 `A[u,m]`。公式和 `+10` pseudo-bin 在 [`m1_muscle_profile.py`](../scripts/carrier_profile_v2/m1_muscle_profile.py) 的 `_raw16` 中逐项实现。M1 R100 建构器以四个 official held-in session 的 rectified EMG `[0,310)` 拟合 `s`，以每个 session 的 M10 拟合 raw16，然后对 pooled raw16 做未中心化 SVD4。[`carrier.py`](../scripts/m1_muscle_r100_v1/carrier.py) 的 `fit_source` 与 `project_m10` 是实际部署入口。

投影后，M1 使用 source projected rows 的四维 mean `mu`，并将四个 projected-column standard deviation 先各自下限截断、再合成为一个共同标量

\[
g=\sqrt{\operatorname{mean}_j(\max(\operatorname{sd}(P_{:j}),10^{-6})^2)},
\qquad T=(P-\mu)/g.
\]

所以 M1 的 `global_rms` **不是**逐列 normalizer；四维都除同一个 `g`。实现位置为 [`carrier.py`](../scripts/m1_muscle_r100_v1/carrier.py) 的 source SVD/normalizer 段和 `project_m10`。

## H1：14 路 signed-softplus state 与 per-column scale

H1 先以 source velocity RMS `s[j]` 缩放 `x[t,j]=v[t,j]/s[j]`，再定义每个原 velocity axis 的两条非负权重：

\[
w^+_{tj}=\operatorname{softplus}(x_{tj}),\qquad
w^-_{tj}=\operatorname{softplus}(-x_{tj}).
\]

因此共有 14 个 state。正确的恒等式是 `softplus(x)-softplus(-x)=x` 以及 `softplus(x)+softplus(-x)=2 log(2 cosh(x/2)) >= 2 log 2`；单独一项不具有 `>= log 2` 的下界。它不推出 NMF 中有可分离的常量成分；当前 H1 主线也不使用 NMF。

H1 使用 100-ms bin：`q[u]=sqrt(max(.1*mean_rate[u],1))/.1`，`z=(rate-mean_rate)/q`，并按共同公式计算 `[unit,14]` raw profile。随后用 13 个 official source 的 pooled raw rows 做未中心化 SVD4，投影后按 **每一列自己的** source RMS scale 归一化：

\[
T_{u j}=(P_{u j}-\mu_j)/g_j,
\qquad g_j=\sqrt{\operatorname{mean}(P_{:j}-\mu_j)^2}.
\]

实现见 [`h1_profiles.py`](../scripts/carrier_profile_v2/h1_profiles.py) 的 `_unit_standardize`、`_raw_signed_state14`、`fit_source_plan` 和 `deploy_profile`。这与 M1 的单一 `global_rms` 不同，不能统称为相同 normalizer。H1 的封存 bank 构建器也明确固定 `CANDIDATE = "signed_state14"`，见 [`build_banks.py`](../scripts/h1_signed_state_r300_v1/build_banks.py)。

## rSyn3 joint ridge 的保留定位

rSyn3 采用非负 EMG/NNLS activation `Z`，对每个 unit 解

\[
\min_{a,b}\;n^{-1}\lVert r-a\mathbf1-Zb\rVert^2+\lVert b\rVert^2,
\]

其中截距 `a` 不惩罚；carrier 是三个 slope 与一个 intercept。这一实现见 [`syn3.py`](../../tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py) 的 `fit_unit_ridge`。消去截距后，slope 收缩取决于 centered activation covariance，而不是只取决于写作 `lambda=1`；它与 signed-state 的逐 state 条件均值不是同一估计器。保留它是为解释历史 M1 对照和诊断，不把它升级为 H1 的统一主线。

## “无偏移不造成损害”：当前没有的保证与可成立的充分条件

在以上两个实际部署的 profile 路径中，未发现一个明确的“冻结 baseline 加 residual”构造，也未发现强制 `g(0)=0` 的 identity fallback：两者都是从当前 support 重新估计 rate mean/noise scale 与 conditional response，投影并归一化后直接替换 carrier。M1 的 `_raw16` 和 H1 的 `_raw_signed_state14` 均如此；它们没有对“估计偏移为零”分支返回旧 carrier。因此，不能声称现有实现已经给出“无偏移时数学上不损害”的保证。

一种足够但尚未实现的形式是固定基线预测 `f0`，并构造

\[
f(x,c)=f_0(x)+g(x,c),\qquad g(x,0)=0
\]

为逐输入的严格恒等式；再由一个明确、审计过的 rule 在“使用基线”时令 `c=0`。在同一输入、权重、随机状态和评估路径下，才有 `f(x,0)=f0(x)`，故该分支的预测和相应 deterministic metric 完全复现基线。这是实现该结构后的充分条件，既不是当前 profile 的功能，也不保证非零 `c` 时的泛化不劣。

最后，`estimated shift = 0` 只是估计器输出或门控规则的事件，不等于真实的分布偏移为零。有限 M10/M3 support、噪声、状态覆盖不足和估计偏差都可能使二者不同；即使有上述 identity fallback，也只能保证该规则选择基线分支时不改变预测，不能把它解释为“已经证明真实无偏移”或“所有无偏移任务都无损”。
