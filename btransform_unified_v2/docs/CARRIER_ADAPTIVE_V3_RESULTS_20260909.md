# Carrier Adaptive v3：结果记录（2026-09-09）

本文记录本轮已完成的 decoder fit、评分与审计结果。最终机器可读汇总见 [comparison.json](../results/carrier_adaptive_v3/report/comparison.json) 和 [comparison.csv](../results/carrier_adaptive_v3/report/comparison.csv)；它保留 B、legacy D、v2 profile 与 new 的四臂比较，不以已见 outer 结果重选方法或 epoch。

统一的研究动机是：M2 / DANDI688-like 的结构化低维运动任务可以直接估计低维调谐 carrier；M1/H1 的多维相关标签在有限校准下则从 source response profile 的信号、重复面板变异和有效维度设计 adaptive carrier。M2/688 虽为 2D label，其 carrier 是 4D cosine-tuning 而非复制标签；M1 使用 EMG16，H1 将 7D velocity 展为 14 个 signed features。此动机不意味着 adaptive 必然优于既有方法，完整任务与方法边界见[方法说明](CARRIER_ADAPTIVE_V3_METHOD_20260909.md)。

相关方法、审计和锁定记录：

- [方法说明](CARRIER_ADAPTIVE_V3_METHOD_20260909.md)
- [H1 inner audit](../results/carrier_adaptive_v3/runs/inner/h1/final_audit.json)
- [H1 outer audit](../results/carrier_adaptive_v3/runs/outer/h1/final_audit.json)
- [M1 SNR inner audit](../results/carrier_adaptive_v3/runs/inner/m1/final_audit.json)
- [M1 PCA outer audit](../results/carrier_adaptive_v3/runs/outer/m1/final_audit.json)
- [H1 outer method lock](../results/carrier_adaptive_v3/decisions/h1_outer_method_lock.json)
- [M1 PCA fallback lock](../results/carrier_adaptive_v3/decisions/m1_inner_pca_fallback_lock.json)

本轮使用固定总训练轮数：M1 24 epochs、H1 32 epochs；所有运行均为单一 `seed=42`。checkpoint 只按 source-only validation 选择：M1 取 `equal_session_mean` 的最早最大 EMA，H1 取 `equal_date_mean` 的最早最大 EMA。以下同时报告两种审计定义的 R²：

- **Legacy R²**：为保持历史 scorer 可比性，对展平输出、以全局均值为参照计算的 R²。
- **Channel R²**：先对每个 channel 居中，再按 channel 方差加权汇总的 R²。

本轮方法改动是将 M1 的 EMG16 和 H1 的 signed-velocity14 source raw 面板用重复-panel 经验 signal/noise covariance 拟合为固定四维 Wiener carrier；SNR-Wiener 使用 shrinkage regularized noise，PCA-Wiener 是预先声明的备选。H1 的 source RMS 拟合范围也相对 v2 改变，故这里比较的是完整 adaptive carrier，而不是孤立投影算子的消融。完整公式、source support 边界和统计限制见[方法说明](CARRIER_ADAPTIVE_V3_METHOD_20260909.md)。

inner 阶段是本轮预先声明的开发/方法确认边界。outer 不使用 outer target 评分选择方法或 epoch；方法在 inner 后锁定，checkpoint 仍只用 outer source validation 选择。当前这些日期是此前轮次已评估过的公开开发日期，不是从未查看过的盲测证据。目标使用协议保留 M1 前十个 trials / H1 前三个 trials 的 calibration support；query labels 不用于 carrier 拟合或 optimizer 更新。文中数值均来自已完成的 audit，使用单一 `seed=42`。

## H1：inner 与 outer 已完成

### Inner（SNR-Wiener）

固定 epoch 32 与 source-selected checkpoint 在两臂中相同，均为 epoch 32。两条 H1 session 都提高：

| 指标 | 上一轮 v2 state carrier | adaptive SNR | 差值（new − v2 state） |
| --- | ---: | ---: | ---: |
| Legacy R²，fixed / selected | 0.21491071514686472 | 0.2843708956040898 | +0.06946018045722507 |
| Channel R²，fixed / selected | 0.2148900455984185 | 0.2843517087007232 | +0.0694616631023047 |

### Outer（已锁定 SNR-Wiener）

固定 epoch 32 的平均值小幅提高；但与各自 source-selected checkpoint 比较时，adaptive SNR 没有进一步提高：

| 指标 | v2 fixed epoch 32 | adaptive SNR fixed epoch 32 | 差值 |
| --- | ---: | ---: | ---: |
| Legacy R² | 0.563644878694612 | 0.5724801922415144 | +0.008835313546902368 |
| Channel R² | 0.5636293019330516 | 0.5724635623864801 | +0.0088342604534285 |

| 指标 | v2 source-selected | adaptive source-selected | 差值 |
| --- | ---: | ---: | ---: |
| Legacy R² | 0.5739632756029782（epoch 26） | 0.5724801922415144（epoch 32） | −0.001483083361463855 |
| Channel R² | 0.5739482555981759（epoch 26） | 0.5724635623864801（epoch 32） | −0.0014846932116957534 |

outer 新方法的 per-session Legacy R² 为：

| Session | adaptive SNR Legacy R² | 相对 v2 fixed | 相对 v2 selected |
| --- | ---: | ---: | ---: |
| `ses-19250120T115044` | 0.6465953463363299 | +0.05113839961144917 | +0.04748530607876744 |
| `ses-19250120T115537` | 0.49836503814669886 | −0.033467772517644434 | −0.05045147280169526 |

outer 的 B 记录为：Legacy B = 0.2962096887976447，new − B = +0.27627050344386966；Channel B = 0.29618850698339017，new − B = +0.27627505540308994。

因此，目前只能确认 H1 fixed epoch 32 的平均值小幅提高，而 source-selected 比较没有进一步提高；不能据此宣称 adaptive 方法在 H1 上稳定、整体优于 v2。并且本轮 H1 的 source RMS 拟合范围也相对 v2 改变，任何差异都属于完整 adaptive carrier 的结果，不能归因于 SNR projection 单一因素。

## M1：PCA inner 与 outer 已完成

预先锁定的 policy 先审阅 SNR inner。SNR 未超过 v2 profile 后，触发 [PCA-Wiener inner fallback lock](../results/carrier_adaptive_v3/decisions/m1_inner_pca_fallback_lock.json)。PCA inner 已完成，审计见 [PCA inner audit](../results/carrier_adaptive_v3/pca_runs/inner/m1/final_audit.json)。source-selected epoch 分别为 v2 M1 inner 7、SNR inner 7、PCA inner 8。PCA 的 selected Legacy R² 为 0.7996049471796511，高于 SNR 的 0.7892624552019654，因此按预先规则锁定 PCA 进入 outer，锁定记录见 [M1 outer method lock](../results/carrier_adaptive_v3/decisions/m1_outer_method_lock.json)。PCA 相对 v2 selected 的增益很小，不能将此分支视为强证据。

为保留完整的候选比较，未被选中的 SNR-Wiener inner 审计如下：

| 指标 | v2 fixed | adaptive SNR fixed | 差值 | v2 source-selected | adaptive SNR source-selected | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy R² | 0.7748695956921516 | 0.7645440244039877 | −0.010325571288163915 | 0.7990786202146065 | 0.7892624552019654 | −0.009816165012641154 |
| Channel R² | 0.7015580508595971 | 0.6878700568691716 | −0.013687993990425529 | 0.7336505107273884 | 0.7206378062530043 | −0.013012704474384074 |

被锁定的 PCA-Wiener inner 审计如下：

| 指标 | v2 fixed | adaptive PCA fixed | 差值 | v2 source-selected | adaptive PCA source-selected | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy R² | 0.7748695956921516 | 0.7832904328357043 | +0.008420837143552684 | 0.7990786202146065 | 0.7996049471796511 | +0.0005263269650446212 |
| Channel R² | 0.7015580508595971 | 0.712721052401935 | +0.011163001542337847 | 0.7336505107273884 | 0.7343482309922986 | +0.0006977202649102177 |

### Outer（已锁定 PCA-Wiener）

outer source projection 已以三个 source dates 重新拟合，且使用 inner 已锁定的 PCA-Wiener；M1 outer 审计见 [M1 PCA outer audit](../results/carrier_adaptive_v3/runs/outer/m1/final_audit.json)。fixed horizon 的平均值提高，但 source-selected 比较下降：

| 指标 | v2 fixed | adaptive PCA fixed | 差值 | v2 source-selected | adaptive PCA source-selected | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy R² | 0.6478321035131733 | 0.6563771181352396 | +0.00854501462206625 | 0.6837372226077683（epoch 6） | 0.6392328051480063（epoch 5） | −0.04450441745976197 |
| Channel R² | 0.5402363627174389 | 0.5513920842992935 | +0.011155721581854627 | 0.5871113570501845 | 0.5290097724067135 | −0.05810158464347093 |

M1 outer 的 B 记录为：Legacy B fixed = 0.6561786870346022，new − B = +0.00019843110063733072；Legacy B selected = 0.6733517771642251，new − B = −0.03411897201621883。Channel B fixed = 0.5511330277370995，new − B = +0.0002590565621940488；Channel B selected = 0.5735529088794222，new − B = −0.04454313647270869。v2/new 的 M1 outer source-validation maxima 分别为 0.8483105679169546 / 0.8485796464665647：source 的微小增加没有转化为 target 的 source-selected 改善。

## 最终受限结论

在 fixed horizon 的比较中，M1 与 H1 都有小幅平均提高；但预设的 source-selected 比较中，M1 明显下降，H1 没有增益。因此，本轮结果不支持以 adaptive v3 整体替换 v2。统计上的方法动机仍然成立，但尚不是性能已经得到证实的主张。特别地，不能因为已见 outer fixed 值而将 fixed epoch 24 改作预设 selected comparator，也不能回改 inner method lock。

## 共同统计原理、回顾性机制证据与限度

M1 与 H1 共享同一 source repeated-panel 的经验 signal/noise construction、四维 Wiener reliability shrinkage，以及预先声明的 inner decoder confirmation 和 outer 锁定流程。两数据集不需要使用完全相同的单一估计器：M1 PCA-Wiener 按总方差选方向，H1 SNR-Wiener 在 noise whitening 后按信噪比选方向；两者的差异也包括 noise whitening 与 common/per-axis scaling，并非只替换方向。

在方法锁定后，对原先冻结的四个 inner projection JSON/NPZ 作了只读回顾性检查，记录见 [source mechanism retrospective diagnostic](../results/carrier_adaptive_v3/diagnostics/source_mechanism_retrospective.json)。该检查状态为 `PASSED`，无新拟合、无 target I/O；每个数据集的两种方法在 `mean`、`S`、`N`、`Nreg` 上均为逐数组相等。它因此说明共同 source 几何的条件性机制，不参与已锁定的方法选择，也不被表述为事先预测。

| 数据集 | `tr(S)/tr(N)` | signal participation rank `(tr(S))²/tr(S²)` | `cond(Nreg)` |
| --- | ---: | ---: | ---: |
| M1 | 5.254965744396633 | 1.1520364258419944 | 22.9507 |
| H1 | 0.7446765859237902 | 4.613528465865482 | 10.0375 |

这里的 participation rank 描述当前 EMG16 / signed-velocity14 raw carrier 特征下**估计 signal 的谱集中度**，不能当作真实神经系统的固有维数。M1 的估计 signal 高度集中：PCA 实际变换后的 signal variance 为 `[2.6933359334, 0.0823935138, 0.0010067722, 0.0007568721]`，SNR 为 `[0.8464810178, 0.4557806174, 0.1274738621, 0.0480887012]`。这与“保留主导变化及其幅度关系可能更适合 M1”的条件性解释相符，但不证明 decoder 因果。H1 的估计 signal 更分散，且 noise 相对 signal 更占主导，因此 noise-normalized 方向与收缩有多个可靠方向的动机；这不是“H1 noise 更各向异性”的说法，实际 `cond(Nreg)` 为 M1 22.9507、高于 H1 10.0375。

source held-date reconstruction MSE 在两数据集上均由 SNR 更低：M1 PCA 0.2695754416、SNR 0.2565670466；H1 PCA 0.5779563832、SNR 0.5539014216。这一重复性诊断不能预测 M1 的 decoder 排序，因此不构成对 PCA outer lock 的替代证据。H1 本轮没有运行 PCA decoder，不能声称 decoder 上已实测 SNR 战胜 PCA。trace 比或上述几何描述也不能推出必然胜者，本文不提出未验证的阈值规则，亦不将尚未实施的连续 noise-normalization 写成当前方法。

实际 decoder 的只读架构核查显示，M1/H1 carrier 进入 identity side path 与 direct token concat 前仅发生 dtype 转换，没有 carrier-specific 的 row L2、LayerNorm 或 clamp：M1 路径见 [cross_session_m1_model.py](../src/btransform_unified_v2/cross_session_m1_model.py#L52) 与 [concat_model.py](../src/btransform_unified_v2/concat_model.py#L45)，H1 direct token 路径见 [model.py](../src/btransform_unified_v2/model.py#L139)。LayerNorm 位于学习的 affine 与非线性混合之后，因而不会系统性消除输入四轴的相对幅度差异；尺度差异在架构上可以被 decoder 利用，但这不证明性能差异由尺度造成，且 M1/H1 的 identity encoder 也不同。

## 独立官方 H1 signed-state 回执

[官方回执 582196](../../tfpd_exploration/submissions/evalai_h1_rift_r300_signed_state_e16_v1/artifacts/OFFICIAL_582196.json) 的状态为 `finished`，方法名为 `H1 RIFT R300 signed-state e16 cached`。这是一条 official13/fullpool signed-state 工作流，不是本文 chronological adaptive SNR32 实验；其 `local_ho_m3_pick` epoch 16（mean 0.46765143362182887）也属于该独立流程。因此，下表只记录 signed-state 方案整体的官方比较，不与上文 adaptive 表合并，不能把改善隔离归因于标签编码或 Wiener，也不能作为 adaptive v3 增量的证明；checkpoint 与数据协议均不同。

| 官方回执 | HO Mean | HO Std | 相对旧 HO Mean |
| --- | ---: | ---: | ---: |
| prior 582073 | 0.4027782688014744 | 0.14526658466582196 | — |
| 582196 | 0.45665779063606854 | 0.11878803990661022 | +0.05387952183459416 |

透明起见，HI 从 0.6681165838896308 降至 0.6497448658416408，差值为 −0.018371718047989982。表中的 `HO Std` 保留官方回执字段原名，不将其解释为多 seed 标准误或置信区间。
