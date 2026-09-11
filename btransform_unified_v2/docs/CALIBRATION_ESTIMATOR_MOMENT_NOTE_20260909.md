# Calibration conditional-response 的矩估计对象

本文只澄清现有 conditional-response 公式的代数对象和一个窄的总体一致性表述；不改变估计器、训练、评分或“无偏移无损”边界。

令一个 session 的合法 support 有 `n` 个 bin。对固定 unit `u` 和非负行为状态权重 `w[t,k]`，简写为 `w_t`、`r_t`、`q>0`，并令

\[
\bar w=\frac1n\sum_t w_t,\qquad
\bar r=\frac1n\sum_t r_t,
\]

其中现有实现的条件响应是

\[
A=\frac{\sum_t w_t(r_t-\bar r)}{q(\sum_t w_t+10)}.
\]

定义使用分母 `n` 的经验 covariance

\[
\operatorname{Cov}_n(w,r)=\frac1n\sum_t(w_t-\bar w)(r_t-\bar r).
\]

因为 \(\sum_t(r_t-\bar r)=0\)，有

\[
\begin{aligned}
n\operatorname{Cov}_n(w,r)
&=\sum_t w_t(r_t-\bar r)-\bar w\sum_t(r_t-\bar r)\\
&=\sum_t w_t(r_t-\bar r).
\end{aligned}
\]

所以 covariance 写法严格成立：

\[
A=\frac{\operatorname{Cov}_n(w,r)}{q(\bar w+10/n)}.
\]

若 \(S=\sum_t w_t>0\)，再令 \(\bar r_w=\sum_t w_t r_t/S\)，则

\[
A=\frac{S}{S+10}\frac{\bar r_w-\bar r}{q}.
\]

这第二种写法也严格成立。它显示 pseudo-count 以 \(S/(S+10)\) 收缩 weighted-mean deviation，但并不把该 ratio 变成 unbiased estimator，更不能把它称为因果效应。权重可与其他行为变量相关，`r` 也可含未观测状态、记录条件和有限样本噪声。

## 边界情形

`w` 非负时，若 \(S=0\)，所有 `w_t=0`，原公式的分子为零，故 \(A=0\)。covariance 写法同样给零；但 \(\bar r_w\) 未定义，因此不能使用 weighted-mean ratio 写法。

若 `w_t=c` 为常量，则 \(\sum_t w_t(r_t-\bar r)=c\sum_t(r_t-\bar r)=0\)，等价地 \(\operatorname{Cov}_n(w,r)=0\)，所以 \(A=0\)。这说明该量估计的是相对 state-weight variation 的 response association；恒定权重不产生条件调制信号。

## 窄的 no-shift 总体一致性表述

以下是一个可解释估计对象不变的充分条件，而不是现有实现的无损保证。设两个 session 中同一可比较 unit 的 bin 序列满足同一联合分布 \((W,R)\)，`E|W R|`、`E|W|`、`E|R|` 有限，`p=E[W]>0`；并假设适用 LLN（例如独立同分布，或满足相应遍历/弱依赖条件）。令 bin 宽度为 \(\Delta\)，当前实现的 rate noise scale 为

\[
q(\mu)=\frac{\sqrt{\max(\Delta\mu,1)}}{\Delta},\qquad \mu=E[R].
\]

该函数连续且严格为正。由 LLN、连续映射和 Slutsky 定理，两个 session 各自的 `A` 都收敛到同一总体矩

\[
\theta=\frac{\operatorname{Cov}(W,R)}{q(E[R])E[W]}
=\frac{E[WR]/E[W]-E[R]}{q(E[R])}.
\]

若 source SVD basis 和 normalizer 已冻结，且两个 session 的 raw conditional-response rows 收敛到相同向量，则其后固定的投影与归一化也收敛到相同 population carrier。一般 \(\theta\neq0\)：no-shift 意味着估计对象相同，并不意味着 carrier 为零或应被关闭。有限 support 时，`+10` 对应的 \(10/n\) 项、随机分母和 rate/weight 噪声使 \(E[A]\) 一般不等于 \(\theta\)；因此这项总体一致性不推出有限样本的 carrier equality、decoder non-inferiority 或数学上的零损害保证。

## 可直接用于论文的精炼段落（约 170 字）

我们将每个 unit 的 calibration carrier 解释为行为权重与去均值、noise-scaled rate 的正则化经验 covariance，而非因果效应。对非负状态权重，pseudo-count 将 weighted-mean rate 相对同一 support 的 sample mean 的偏离按状态占据量收缩。若可比较 unit 在两个 session 具有相同的行为—rate 联合分布，且有限矩与 LLN 条件成立，则两者的 conditional-response estimate 收敛到同一 population moment；该 moment 一般不为零。此陈述只说明 no-shift 下估计对象保持不变。有限 calibration 的分母随机性、pseudo-count 和噪声仍存在，故不能推出有限样本 carrier 相等、decoder 不劣或无损保证。
