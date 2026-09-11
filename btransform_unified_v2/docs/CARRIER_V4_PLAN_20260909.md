# Carrier v4：保留 M1 基线的共享估计器、信息与融合对照

## 目标与当前证据

本文件记录的原 v4 方案固定 M1 的封存 source3 rSyn3 carrier，并实现 M1/H1 共用的 `latent_state3_ridge_intercept/per_column` core。该历史方案曾包括 concat、proj-add P16/P32 和四信息臂。

本文件保留事前实验目标、已完成 evidence 与停止边界，不再承诺完成全部十二个 cell。用户已停止 H1 新方法和资源投入；H1 保留既有 signed-state 主线。ROOT 已 SIGTERM H1 v4 scoring process 并停止混合队列，记录见 [user_stop_h1_v4_20260909.json](../results/carrier_v4/user_stop_h1_v4_20260909.json)。原混合队列不会恢复，其中未开始的 M1 v4 concat/P16/P32 cells 也不再排队。后续信息消融、local/pack/official 验收和 H1 signed-state 的有限例外均以 [最终信息消融合同](FINAL_ABLATION_OFFICIAL_PLAN_20260909.md) 为准；它不恢复 H1 NMF/fusion path。独立 M1 多 seed 与 fourth-column carrier 工作仍见 [M1 muscle multi-seed carrier plan](M1_MUSCLE_MULTISEED_CARRIER_PLAN_20260909.md)。

## 固定的 v4 主方法

共同实现为 [common_estimator.py](../scripts/carrier_v4/common_estimator.py)，方法名 `latent_state3_ridge_intercept/per_column`。

1. 使用 source 行得到的行为 RMS。M1 对 16 路 EMG 取 ReLU 后缩放；H1 对 7 路 velocity 缩放后，展开为正、负 softplus 的 14 个非负状态。只有这一输入映射及数据采样合同属于任务 adapter。
2. 对 pooled source 状态矩阵拟合 rank-3 NMF。两任务共用 `nndsvda/cd/Frobenius/seed=42/tol=1e-5/max_iter=1000`，不使用 alpha/L1 正则。字典逐行 L2 归一化，按 source activation 能量降序及行 SHA 排序。
3. 每个 session 在其允许的 calibration support 上，用冻结字典做 NNLS 得到三个 activation。以原始 Hz rate 为响应，解带截距 ridge：`design=[1,z1,z2,z3]`，`gram/n + diag(0,1,1,1)`。不减去单元均值，也不对 rate 做 Poisson 标准化，保留可学习的单元 baseline 信息。
4. carrier 为 `[beta1,beta2,beta3,intercept]`，统一用 source pooled 单元行的逐列 mean/std 归一化；`std<=1e-6` 的列采用尺度 1。目标端不重拟合字典或 normalizer。

**P0 后修订：原 v4 M1 line 导入封存 rSyn3 的 source3（26/27/28）字典、RMS 和 normalizer，使用旧 reader 的时间对齐。** 四个 source 仍全部参与 decoder 训练，但 24 与三个 HO session 的 carrier 均由同一个 source3 fit 部署；单元回归仅使用各自 M10。不重新拟合 source4 字典，也不在该 v4 line 同时改变 spike 时间约定。共享 core 实际执行 NNLS、逐 unit GEMV/solve 和 normalization，adapter 不能以旧 T 缓存冒充 core 的输出。这不限制已单列、不会复活 v4 queue 的 muscle fourth-column candidate。

历史 H1 v4 dictionary 使用十三个 official source 的 training trial 行，排除各 source 最后两个 validation trials；normalizer 使用各 source first-M3 的单元回归。H1 的 27 个公开 tag 保留既有 official M3 activity，并重算 T/E0；训练 C2 的 M7/M5/M4/M3 schedule、M3/M4 carrier prefix 分配均保持既有规则。这些是 stopped path 的审计事实，不再启动各信息臂或调参。

原 H1 signed-state14 的逐列条件均值是另一个估计器，不等于带交叉项的 joint ridge；不能只改公式名称宣称已经统一。H1 v4 新路径已停止，不再以 decoder 实验追求或判定其有效性；signed-state 保持 H1 主线。

P0 已运行 [p0_exact_m1_receipt.json](../results/carrier_v4/p0_exact_m1_v2/p0_exact_m1_receipt.json)：统一模板与当前旧生产实现的 raw 输出在全部 7 个 tag 上完全一致；经旧 normalizer 后的 float32 T 与原模型输入也全部逐字节一致。封存 source3 的历史 float64 raw 与当前重算最多差 `2.132e-13`，使用 `atol=1e-10,rtol=0` 核验并单独记录，不声称该历史中间量 byte-exact。共享 core 的 [exact pack receipt](../results/carrier_v4/m1_exact_pack_v1/carrier_pack.json) 已对四个 source 和三个 HO tag 的最终 float32 T 逐一记录 `baseline_byte_equal: true`。

## 固定实验矩阵

每个任务共六个实验单元：reference fusion 下四个信息臂，加 FULL 下另外两个融合设置，共十二个单元。满足完整合同的已有训练可复用，但须有初始化、输入、训练 recipe 和评分证据。

| Task | 信息臂 | E0 融合 | 投影宽度 | 固定训练 |
| --- | --- | --- | --- | --- |
| M1 | FULL / ACTIVITY_ONLY / CARRIER_ONLY / NONE | full E0 concat | 无投影瓶颈 | R100/D4，24 epochs，159960 updates |
| M1 | FULL | local16 + P(E0) | 16 | 同上 |
| M1 | FULL | 两组 local16 + P(E0) | 32 | 同上 |
| H1 | FULL / ACTIVITY_ONLY / CARRIER_ONLY / NONE | local16 + P(E0) | 16 | R300/D4，32 epochs，23392 updates |
| H1 | FULL | full E0 concat | 无投影瓶颈 | 同上 |
| H1 | FULL | 两组 local16 + P(E0) | 32 | 同上 |

信息臂定义为：FULL `(E0=f(activity,T), direct T=T)`；ACTIVITY_ONLY `(E0=f(activity,0), direct T=0)`；CARRIER_ONLY `(E0=0, direct T=T)`；NONE `(E0=0, direct T=0)`。M1 在 FULL/ACTIVITY_ONLY 保留 live B3S 的训练，在 CARRIER_ONLY/NONE 不调用 encoder；不为了消融而冻结所有臂的 encoder。H1 ACTIVITY_ONLY 必须重新 materialize E0，不能保留 FULL E0 中已经混入的 T。神经 query 和训练 sampler 保持不变，NONE 仍是正常的 neural decoder。

局部卷积通道保持 16；temporal width=256、D4、其余结构和优化参数保持各自正式协议。P 是 E0 的投影宽度，不是 carrier 维度或 temporal width。将报告各组实际可训练参数量；concat 与 proj-add 参数量本来不同，不声称二者等参数量。

初始化以共同 P16 reference 为依据。concat 使用已有函数保持的第一层展开。P32 保留 P16 投影/权重，新投影块独立初始化，而对应新增 token 权重置零；因此初始函数匹配，新增块仍能通过训练生效。共同非融合参数须逐项一致，并检查前向、streaming 与新增块的梯度。M1 保留 live B3S，H1 保留既有 C2 E0 构造。GPU1 [fusion audit v2](../results/carrier_v4/fusion_audit_v2/fusion_audit_receipt.json) 已通过：两任务 concat/P16/P32 的 matched forward 均在 `2.385e-7` 内，P16/P32 的 effective fold 相同，P32 新增 token 输入列首步有梯度、新 projection rows 在一步更新后有梯度；并记录 P32 owner 的 `token_in=36`、`proj_out_dim=32`、`proj_groups=2`。M1 FULL/ACTIVITY_ONLY 的实际 M10 bank 两步 smoke 已分别通过 [FULL](../results/carrier_v4/m1_full_concat_smoke_v1/smoke_receipt.json) 和 [ACTIVITY_ONLY](../results/carrier_v4/m1_activity_only_concat_p16_smoke_v1/smoke_receipt.json)，包含 static/live、legacy D/B、streaming 与 post-update 检查；四臂与旧 D/B 完整 forward/参数核验见 [arm invariant v2](../results/carrier_v4/m1_arm_invariant_smoke_v2.json)。

执行状态由 [experiment manifest](../results/carrier_v4/experiment_manifest_v2.json) 保留为历史合同。12 个主 cells 中，M1 FULL concat 复用已完成的旧 D 24-epoch training/replay，M1 ACTIVITY_ONLY concat 复用已完成的旧 B 24-epoch training/replay；两者的可复用性由 [legacy concat reuse audit](../results/carrier_v4/m1_legacy_concat_reuse_audit_v1.json) 绑定。其余 cells 不再因为该 manifest 被执行：H1 新方法已停止，混合队列也不会恢复未开始的 M1 v4 fusion cells。

比较 reference 由 manifest 中每个 cell 的显式关系决定，而不是笼统按 fusion 字符串推断：M1 FULL P16 相对 FULL concat，M1 FULL P32 相对 FULL P16；H1 FULL concat 与 FULL P32 都相对 FULL P16。H1 old signed FULL 仅是新 H1 FULL P16 的 external baseline，不能作为新 carrier 四信息臂的 FULL reference。

## 评分、诊断和交付

M1 完整扫描 24 个 EMA，主指标为 HO3 每 session 的 channel-centered variance-weighted R² 后等权平均，最早最大值选 epoch；同时报告 legacy flattened 曲线。复用已核验的 baseline replay，并逐项验证 targets、窗口起点与预测 artifact 哈希。

H1 的 32-EMA earliest-maximum 原为历史评分合同，现已停止且没有正式 selected epoch。训练虽完成 32 epochs，但 SIGTERM 时只完成 score 28/32；日志中的 best observed e17 `0.360018` 与 fixed e16 `0.352676` 都不是完整扫描的 selected result。停止由用户决定，不把队列中的 `FAILED/-15` 误写成方法失败。隐藏 test 或 EvalAI 结果均不参与该未完成 v4 路径的选择。

在新的训练结果产生前声明非劣容忍度为 R² `0.01`：比较量为 `candidate-reference`，下界达到 `-0.01` 才支持相应描述。信息比较使用 manifest 指定的同 carrier-family FULL reference；融合/宽度比较使用上述显式 P reference；同时报告同固定 epoch（M1 e3、H1 e16）和各自完整扫描 selected epoch。以 session 配对差异和区间为主，M1 n=3 的区间只作有限的敏感性证据；H1 public M3 的结果仍是 development evidence。只跑 seed42，不能把“未观察到增益”写成总体无效，也不能把“不显著”直接写成非劣。

另做一项限定的 M1 frozen-e3 输入干预：原 rSyn3、归一化截距列置零、三个 slope 列置零、全部 T 置零。每次同时修改 B3S side 与 direct T，原始分支必须复现既有 e3 预测。该诊断衡量固定模型对输入分量的依赖；它不是重新训练的消融，也不能给各分量分配唯一的因果贡献。

交付保留：可复用的 unified v4 code/source fit、已完成 M1 信息对照和 stopped H1 path 的可审计记录。它不再包括承诺交付 M1/H1 全部四臂或融合/宽度比较。任何未来 M1 carrier 增益仍必须由相同评分面的实测结果支持。
