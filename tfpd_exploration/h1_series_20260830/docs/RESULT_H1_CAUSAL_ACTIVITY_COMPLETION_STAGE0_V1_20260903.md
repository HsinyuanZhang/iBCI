# Result: H1-CAC Stage 0 Frozen-Weight Screen V1

日期：2026-09-03  
状态：`COMPLETE_H1_ACTIVITY_CONTENT_BUT_BOUNDARY_FREE_DETECTOR_GATE_FAILED`  
设计：`DESIGN_H1_CAUSAL_ACTIVITY_COMPLETION_V1_20260903.md`  
工单：`WORKORDER_H1_C1_M3_TO_M7_ACTIVITY_TTA_V1_20260903.md`

## 0. 结论

H1 在 **M7 以内**仍有稳定的 activity-memory 正头寸，但本轮预注册的主部署律 `D-EMED7` 没追回足够比例：

- 真 trial `M4→M7`：相对 static M4 **`+0.031999 R²`，5/5 dates，11/11 recordings 为正**；Gate 1 通过。
- 固定块 `C-FIX7`：**`+0.012209`，4/5 dates，8/11 recordings 为正**。
- running-median 能量门 `D-EMED7`：**`+0.009024`，3/5 dates，8/11 recordings 为正**。
- D 对真 trial 头寸的 recovery 为 **28.2%**，低于预注册 50%；Gate 2 失败。

正确读法不是“activity 无效”，而是：

> 完整、已结束的真实 trial activity 对 H1 identity 有稳定价值；简单 fixed chunk 已产生小而正的无标签结果，但 M2 的 running-median 能量门迁到 H1 后没有改善 fixed chunk，也未达到预注册的部署恢复门。

本结果不能写成 C1 成功，因为 Stage 0 使用的是既有 H-C date-LODO checkpoint、M4 carrier 和 M4 seed。它回答的是 detector feasibility 与 M7 范围内的内容头寸。

## 1. 权威

结果根：

`tfpd_exploration/h1_series_20260830/results/h1_causal_activity_completion_v1/stage0/`

| artifact | SHA-256 |
|---|---|
| `attempt.json` | `55218ad54e148fa7a361efdb0bee94e58db69627ad60aef4b1c5833e330a4ec3` |
| `input_authority.json` | `681bb119dd9bd054cad58ea56a22649a48aa929acfdda0e770b1e1fb3d27cdb3` |
| `score.json` | `4c9011ad1368e994e28179aa086a98c863e1f7551ceeda7ef5798c51060ed6b6` |
| `terminal.json` | `6587cb3c931ef76daaa17cc67ffa34199ad1d2846f8595534a0648d4ddbb1f15` |

全部 body 与 sidecar 为 0444。GPU0 单卡冻结推理；GPU1 未使用。四臂累计 decoder 推理约 74.7 秒。无 optimizer、backward、target model update、held-out/minival/EvalAI access。

## 2. 主表（等 recording，再等日期）

| outer date | A Static M4 | B True-trial M7 | C Fixed-chunk M7 | D Energy-median M7 | B−A | C−A | D−A |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19250108 | 0.545514 | 0.577483 | 0.570330 | 0.569379 | +0.031969 | +0.024815 | +0.023865 |
| 19250113 | 0.339962 | 0.356667 | 0.346840 | 0.334511 | +0.016705 | +0.006878 | −0.005451 |
| 19250115 | 0.521320 | 0.541993 | 0.526149 | 0.524081 | +0.020673 | +0.004829 | +0.002761 |
| 19250119 | 0.317487 | 0.342381 | 0.310863 | 0.317000 | +0.024894 | −0.006624 | −0.000487 |
| 19250120 | 0.349828 | 0.415581 | 0.380975 | 0.374262 | +0.065754 | +0.031147 | +0.024434 |
| **equal-date mean** | — | — | — | — | **+0.031999** | **+0.012209** | **+0.009024** |

相对 B 的 recovery：

- C：`0.012209 / 0.031999 = 38.2%`
- D：`0.009024 / 0.031999 = 28.2%`

## 3. 逐 recording 广度

| arm delta vs A | positive | total | mean delta | worst recording |
|---|---:|---:|---:|---:|
| B True-trial M7 | **11** | 11 | +0.031996 | **+0.014494** |
| C Fixed-chunk M7 | 8 | 11 | +0.013355 | −0.031724 |
| D Energy-median M7 | 8 | 11 | +0.010373 | −0.031724 |

B 的最差 recording 仍为正，说明 M4→M7 内容结论很稳。C/D 的均值不是由所有 recording 一致提高构成；三个 recording 为负，必须保留披露。

## 4. 两道门

### Gate 1：内容

要求：B−A 至少 +0.010，至少 4/5 dates 为正。

观察：`+0.031999`，5/5。**PASS**。

这还说明原 cap30 `+0.048937` 的大约 65% 已在 M7 前出现；不需要先把 C1 推到训练外的 M30。

### Gate 2：主部署律

要求：D 回收 B 的至少 50%，至少 3/5 dates 非负，worst date ≥−0.010。

观察：28.2%，3/5，worst −0.005451。只有 recovery 条件失败，因此整体 **FAIL**。

不得事后把 50% 改小来宣布过门。

## 5. 能量门为什么没有帮助

D 相对 C 的 equal-date 差为约 `−0.003185`。这不是单纯“少提交 member”造成：

- 多数 recording 最终仍到 M7；
- 但能量门经常跳过早期块，改为选择更晚的块；
- 19250113 的一个 recording 中 C/D 都选前三块，另一个 recording 的 D 需要六个候选才凑到三块；该日期 D 从 C 的 +0.006878 变成 −0.005451；
- 19250119 上 D 反而比 C 少坏，说明能量门不是全局错误，而是“mean-rate 高”不等于“更适合作为 H1 identity member”。

因此 M2 Ce-NAT 的 running-median rate gate **不能无条件迁移到 H1**。H1 的 identity member 是完整轨迹的时序形状，不只是活动量；按 mean rate 选择可能丢掉轨迹覆盖。

## 6. C-FIX7 的位置

C-FIX7 是新的正数：等日期 +0.012209、4/5 dates。但本单预注册它为描述臂，不能在同一结果里事后替换 D 并宣布部署门通过；而且它只收回 B 的 38.2%，逐 recording worst −0.0317。

合法的后续有两种：

1. 在新的、明确标记的 successor 工单中把 C 作为 source-OOF 选出的候选，再让远程 Held Out 做外部确认；或
2. 先在 C1-specific 五折 M3→M7 中同时保留 C/D，不用本结果修改其超参，并把 C 的选择历史完全披露。

不合法的是：在本根中换主 arm、改 L、扫 phase offset 或看 target 后调能量定义。

## 7. 对 C1 Stage 1 的含义

Stage 1 仍值得做，理由变得更具体：

- C1 训练明确见过 M4/M5/M7；
- B 表明 M7 内就有 +0.032 的稳定内容头寸；
- C 表明完全不使用边界也能得到小幅正收益；
- 但 D 的失败说明不能直接照搬 M2 gate。

Stage 1 必须回答：C1 的 prefix-cycle consumer 是否能把 M3→M7 的真 trial 内容和 fixed-chunk 内容转成更大的增益。它仍需五个 date-LODO C1 checkpoint；all-source checkpoint 只能用于最终 package/接口锚，不能替代 LODO。

## 8. 已恢复的官方 all-source C1

用户提供的固定 Git commit 文件已经 CPU 严格验证并保存为只读 imported artifact：

`tfpd_exploration/h1_series_20260830/artifacts/h1_c1_all_source_epoch49_v1/epoch_049.ckpt`

- checkpoint SHA：`0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`
- model-state SHA：`bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85`
- strict load：0 missing / 0 unexpected
- source commit：`84c7aaecb656be812d046317fec07783fa6701a4`

这消除了最终 all-source package 的重训需求，但没有消除 Stage 1 五折 checkpoint 的需求。

## 9. 下一步

1. 请求队友发布五个 C1 LODO checkpoint bodies（SHA 已写入 work order）。
2. 在收到后先复现原 C1 M4/M5/M7 sealed rows，确认 artifact/operator 一致。
3. 新建 Stage 1 work order：A=M3，B=true trial M3→M7，C=fixed chunk，D=原 running-median；保持同 carrier/query/target。
4. Stage 1 不重训。若真 trial内容门失败，停止；若 fixed chunk 仍小正，则由新的工单决定是否值得一次官方 external confirmation。
5. PACD 只在 C1 真 trial内容过门、且问题明确是 M3 extrapolation/训练支持时考虑；它不能修复错误的 activity selector。

