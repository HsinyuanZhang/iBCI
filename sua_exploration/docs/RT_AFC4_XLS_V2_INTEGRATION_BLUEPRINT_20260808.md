# RT `afc4_xls_v2` 强 label-pairing null：待集成蓝图（不启动）

## 结论与边界

`afc4_xls_v2` 不是新的性能方法，而是 RT `afc4_vel` 的更强 **label–neural pairing null**。它的目的，是回答 R-LS 当前的 within-reach cyclic label shuffle 可能过弱时，R-C 的增益是否仍依赖正确的连续速度标签与对应神经 block 的配对。

已冻结的 CPU support audit 通过了 null 的可用性门：

- receipt: `RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json`
- receipt SHA-256: `ad2468ca04c3ed2542c7c37b0f1d27c17d4e517943fe64a7fa34e6cf9636c899`
- 每个 RT session 的 block permutation SHA 已由新的 support-only primitive 重算并逐个复现；不是“同一统计量接近”。
- pooled reach-direction cosine mean/median: `-0.01875 / -0.02349`；source-LOSO shared 2×2 linear transfer mean R² `-0.00148`，orthogonal transfer mean R² `-0.25040`。

这些数字只说明该 **null 不像固定旋转/全局反向那样可由 source-trained consumer 反演**；它们不预测 `R-C − XLSv2` 的符号，更不是 GPU 结果。

## 已隔离的 primitive

实现：`streaming_calibration_exp/src/data/afc4_xls_v2.py`。

```text
permutation = deterministic_random_cross_reach_derangement(
    support_reach_group_ids,
    support_velocity_blocks,
    session_name=session,
    seed=42,
)
```

该 generator 的可见输入严格只有 event-qualified **support** block group IDs、对应的 support velocity `(vx, vy)`、session 名及 seed。它：

- 保留 velocity label 的精确 global multiset；
- 保留 neural block 布局与每 reach 的 neural block 数；
- 要求真 permutation、每个 label 均改变、无 same-reach donor；
- 用 session-namespaced deterministic RNG，并排除本 session reach-mean cosine 接近 `+1/-1` 的候选；
- 没有 rates、AFC4 `W/b`、decoder、模型分数或 query-label 参数，也不拟合 OLS。

未来 adapter 仅能做 `fit_afc4(rates, support_velocity[permutation])`，输出同宽 `[W_x, W_y, ||W||, b]`。它不可将 rates 或任何 query labels 送入 generator。

## 绝对禁止的“修复”

不允许在 source 训练或 target calibration 时拟合/应用共同 `2×2` inverse、rotation、Procrustes、ridge map 或任何 `W_null @ A → W_aligned` 的补偿。v2 receipt 的 shared-map audit 之所以存在，正是为了排除该类可 source-learn 的几何关系；在 XLSv2 上加入其 inverse 会令 null 失效。每个 xls descriptor 必须由该 session 的 M24 support 重新拟合，且 split receipt 绑定上述 audit file SHA 与 **该 session 的 permutation SHA**。SHA 不相等时 fail closed。

## 延后到现有 RT 收口后的接线清单

在当前 R-RS/R-LS controls 及 priority-1 MB4 都已有终端收据、且 MB4 15/15 aggregate 已写出之前，不修改下列任何 active 文件：

- `src/data/falcon_k4_features.py`；
- RT/nested DataModule、loader、runner；
- 任何现有 Hydra config；
- `rt_controls_continuation.py`、`rt_mb4_matched_continuation.py` 或它们 import 的文件。

这些是届时需要**以新 version、新 root**加入的接线点，而不是现在的 patch：

| 接线点 | 后续新对象 | 必须固定的 contract |
| --- | --- | --- |
| arm spec | `afc4_xls_v2` | canonical arm，`side_dim=4`，坐标 `[W_x,W_y,||W||,b]`；不可把 descriptor 改成宽度不同的 side path。 |
| support feature adapter | 新 `afc4_xls_v2` adapter | 从 M24 `[0,24)` 的 event-qualified blocks 得 group/velocity；调用 primitive；用 `velocity[perm]` 重拟合 OLS；逐 session SHA parity。 |
| normalizer | new arm-aware source-only normalizer contract | 仅 inner-train source sessions 拟合，`feature_group=afc4_xls_v2`，显式排除 outer target；不得将 target support/query 用于 normalizer。 |
| split manifest | `rt_clean_nested_loso_xls_v2_split_v1`（新 schema） | M24 chronological support、q24/window50、target/inner session binding、audit file SHA、session permutation SHA、`target_calibration_optimizer_steps=0`、query labels only for scoring。 |
| fit receipt | `rt_clean_nested_loso_xls_v2_selection_v1` | source-only fresh training，inner validation checkpoint only，target session 未在 fit/open/query-label/normalizer/checkpoint 选择中出现。 |
| outer evaluator | `rt_clean_nested_loso_xls_v2_outer_eval_v1` | 对已选 checkpoint 做一次 target eval；无 BP、无 optimizer、model eval、state SHA before/after 相同；query label 仅 scoring。 |
| Hydra | 新 `rt_joint_afc4_xls_v2_m24_loso.yaml` 和新 `rt_b3s_afc4_xls_v2_m24_loso.yaml` | 从 matched R-C/B3S 复制所有 architecture/epoch/metric 设置，只将 side group 改为 xls；不覆盖现有 yaml。 |
| supervisor + finalizer | 新 `rt_xls_v2_matched_continuation_v1.py` | 新 raw root、fresh imports、15 格 exact matrix、immutable copy-import、最终只写一次 aggregate。 |

## 冻结的 GPU 协议（仅在前置完成后）

前置：R-RS/R-LS control queues 的全部相关 terminal 已完成；`RT_MB4_MATCHED_FULL15_AGGREGATE_v1` 已以 15/15 immutable imports 写出；本蓝图对应的 manifest 与 patch 必须另行 review。此前不启动 XLSv2 GPU。

| 项目 | 冻结值 |
| --- | --- |
| arm / reference | newly source-trained `afc4_xls_v2` / sealed R-C `afc4_vel` |
| data | RT sub-C，development clean nested outer-LOSO，same 15 folds |
| seed / support | seed 42，chronological M24 `[0,24)` |
| architecture | 与 R-C 同一 B3S family、same `side_dim=4`、same batch/window/model dimensions；不引入新 FiLM、memory、attention 或 decoder branch |
| checkpoint | 每 fold fresh source training；仅 inner-validation `val_heldin/r2_mean` 选 checkpoint；没有 warm start 或 R-C checkpoint reuse |
| target pass | q24，window 50 bins，one-shot；target no BP/no optimizer/no state change；target query labels only scoring |
| all-fold rule | 全部 15 folds 必跑；不得因中间 `R-C−XLSv2` 正负、均值或 p-value 停止/增臂 |
| statistic | paired `R-C − XLSv2` rows，mean、median、全部 signed rows、exact two-sided sign test、fixed-seed paired fold bootstrap |

## 成本与解释

该 null 在 online deployment 不应改变 R-C 的模型参数量、encoder/decoder MAC、token/state 大小或 side width：它仍是 M24 后一次性的 4-D 静态 support descriptor。它会增加离线实验的 fresh source-training 成本，但不增加运行时 side path 宽度。所有关于参数/MAC/state 相同的数字必须来自 future fit/eval receipt 的实际 accounting；在此之前只能写 “intended matched width”。

若 `R-C − XLSv2 > 0` 在 15-fold paired endpoint 上成立，解释为：R-C 的证据同时依赖 continuous velocity label 与正确 neural-block pairing，且该结论不由 v2 可共用的 2×2 transform 解释。若差异不成立，必须写作“在更强的 cross-reach label null 下，RT pairing-specific claim 不成立/不稳健”，不能用旧 weak LS 或 MB4 结果替代。
