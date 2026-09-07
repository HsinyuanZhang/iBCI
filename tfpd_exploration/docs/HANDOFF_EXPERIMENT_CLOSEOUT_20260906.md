# 当前实验交接总览

更新时间：2026-09-06 15:16 HKT。本文只描述本执行链状态；旧逐时进展仅作追溯。

## 当前决定

**H1 训练已停止，本地评估已完成；因同面差距仍很大，保持这条训练线暂停。** 不自动恢复训练、不追加轮次、不修改网络、不提交官方评测。用户13:09追加到24轮的预算已被15:00的停止指令取代，本次不能记为完成24轮。三任务质量总目标尚未完成。

H1 监督进程与两个训练进程均已退出，所有文件保留。FLAT完整提交至21轮，ROUTE至22轮；本地评估固定使用共同完成的21轮 EMA，不按完整评估分数另选轮次，ROUTE22亦保留。独立评估于15:14成功结束，总耗时418.51秒，两个评估进程亦已退出。14项CPU测试通过；两臂2,908点选择分数精确复现，20,325点完整预测与Original及原12轮档案的目标／session／endpoint逐点一致，原始训练状态和RNG前后不变。原12轮完整实验不变。

## 三任务质量：当前主线

下表中的分数是各自指定评分面的 pooled R²，不能跨行、跨数据面相减。

| 任务／当前网络配方 | 已冻结选择 | 同面 Original | FLAT／ROUTE | 目前能支持的结论 |
| --- | --- | ---: | ---: | --- |
| M1 QueryAge16；固定24轮 | 两臂 EMA e6 | .809289 | .811652／.812196 | 31,252个 source-minival 点上的小幅描述性提升；一个 session 退化，未证明正式非劣 |
| M2 QueryAge16 + prefix；固定24轮 | FLAT EMA e2；ROUTE EMA e20 | .229198 | .143064／.296396 | 2,069个 ext4 开发点上 ROUTE pooled 提升，但一个 session 严重退化；FLAT pooled 下降，未证明整体非劣 |
| H1 QueryAge16 + prefix；续训已停止 | 共同终点21轮 EMA | 完整面 .960784；C2 .888499 | **.452043／.443402** | 相对C2为−.436456／−.445097，两臂13/13 session均低于C2；暂停继续训练，不据此推断held-out排名 |

H1第21轮的**另一个评分面**是2,908点选择集：FLAT `.490809`、ROUTE `.483968`，不能与完整面混用。完整面的equal-session R²为`.440952/.432322`；最差session分别为`ses-19250120T115044`（`.310980`）、`ses-19250115T111328`（`.181193`）。原12轮完整面pooled `.278085/.284643`，续到21轮分别提高`.173958/.158759`，但同面差距仍很大。原12轮selected与epoch12是同一权重的两个标签，不是独立重复。

**本地诊断与held-out目标分开。** C2官方581920的HO均值为`.375989`，原版paper-LR 578474约`.2615`、released-LR 578473约`.2099`。因此C2虽本地完整面`.888499`低于Original`.960784`，已有官方成绩反而更高。不能要求新模型必须本地达到`.96`才可能匹配原版HO，也不能由本地`.88`保证C2的HO成绩；C2使用可见HO开发面选轮的历史须披露。

关键风险：M1 的增量仅约 `.0024/.0029`；M2 ROUTE 的 `ses-2020-11-19-Run1` R² 为 `−.238162`。这些开发／选择面有历史暴露，不能称未触碰测试集或正式统计非劣。

## Latency 与网络统一性

- **M1：** 已完成实际所选权重的完整流式等价验证及同镜像本地测速，P95 相对 Original 约12.9–16.9倍加速；不是官方服务 latency。
- **M2：** 已完成实际所选 QueryAge 权重的 cached/uncached 完整流式验证和六次本地测速。T1 平均比 Original 慢，T2 平均及P95更快；不能合并成一个跨配置“加速倍数”。
- **H1 最新 QueryAge：** 尚无这次所选权重的完整 public-runtime 证明或测速。已有旧 H1 结果不能替代；这两项未启动，冻结期间不自动执行。
- 三任务属于共同 QueryAge16 网络设计族，但不是相同权重／所有设置完全相同；H1有明确的 unscaled-dot/local-balanced 空间 preset，必须披露。

## 请优先审核

1. **H1 的数据／计算对齐与基线差距。** 局部索引、缩放、EMA 和缓存形状自查未发现简单错配，但全部原始 NWB→cache 的独立分箱核对未执行。
2. **基线与历史分数身份。** M2历史581919镜像默认 payload 路径与声明的所选 payload 不一致；远端有无命令覆盖未知，分数归属待查。
3. **质量结论的范围。** M1小幅增益、M2严重 session 退化、H1大差距不能用 runtime PASS 或局部速度提升掩盖。

完整任务、验收要求与证据定位见[第三方审核清单](THIRD_PARTY_AUDIT_CHECKLIST_20260906.md)。清单是审核建议，不是自动执行队列。

## 原始证据入口

- [M1同面完整质量表](../results/family_runtime_v1/m1_p1_frozen_same31252_quality_v1.json)
- [M2冻结所选权重ext4结果](../results/m2/queryage_family_v1/queryage_prefix_pair24_selected_ext4_v1/receipt.json)
- [M2所选权重六次latency总表](RESULTS_M2_QUERYAGE_SELECTED_LATENCY_20260906.md)
- [H1已完成12轮父实验收据](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/receipt.json)、[父实验选择冻结](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/selection_freeze.json)
- [H1累计24轮续训边界与恢复协议](PROTOCOL_H1_QUERYAGE_CONTINUE_24_V1_20260906.md)
- [H1续训执行目录](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_continue24_v1/)、[共同放行记录](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_continue24_v1/barrier/START)
- [主动停止记录](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_continue24_user_stop_20260906.json)、[共同21轮本地评估目录](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/user_stop_epoch21_local_eval_v1/)
- [第21轮本地评估与基线比较收据](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/user_stop_epoch21_local_eval_v1/receipt.json)：SHA `27e53118a92d9065ba4d8dae5a351e88238ba43c2b874f2369d5c88064137d06`；原生预测NPZ及plain-EMA导出在同目录`exports/`。
- [C2官方held-out结果](../submissions/evalai_h1_c2_ho_epoch15_v1/artifacts/official_result.json)、[原版官方历史](../h1_series_20260830/docs/HANDOFF_H1_SUCCESSOR_AGENT_20260903.md)
- [网络族定义与真实例外](AUDIT_QUERYAGE_THREE_TASK_NETWORK_FAMILY_20260906.md)
- [历史逐时记录](PROGRESS_CRST_FAMILY_20260906.md)：包含旧状态和旧实验，不作为当前状态表。

所有旧实验与失败记录保留。H1 CausalPE、cold-prefix/dense诊断、C2、历史e8、teacher/Sfix属于历史对照，不与本页当前主线混为同一模型。本次授权的停止与本地评估已经完成；本执行链没有自动排队的新训练、消融、原始NWB重建、public-runtime证明、测速或官方提交。
