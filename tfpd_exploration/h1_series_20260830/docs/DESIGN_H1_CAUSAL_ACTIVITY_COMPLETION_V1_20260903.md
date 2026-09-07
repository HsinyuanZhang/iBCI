# H1 Causal Activity Completion（H1-CAC）设计 V1

日期：2026-09-03  
状态：`READY_FOR_ADDITIVE_IMPLEMENTATION`  
范围：FALCON H1 / DANDI 000954 / 冻结权重、无标签、因果 activity memory  
不在范围内：新 carrier、target 反传、trial-boundary 恢复、EvalAI 提交

## 0. 一句话

H1 现有最可信但尚未部署的余量不是 carrier，而是 **校准后的新神经活动**：五日期 LODO 中，冻结 H-C 的 causal growing activity 相对 static M4 提升 `+0.048937 R²`，5/5 日期、11/11 recording 为正。

本设计把它改造成 H1 官方接口可执行的状态机：C1 仍以 M3 identity 和 M3 H-C carrier 启动；评测流中只看神经活动，用固定长度 chunk 构造最多四个新 activity member，使 identity 从 M3 因果补到 M7，然后冻结。方法名为 **H1-CAC**。

## 1. 已知事实与纠正

### 1.1 已知正结果

- 官方 C1 prefix-cycle M3：Held Out `0.284139`，相对 T0 `+0.043029`，相对 paper-LR SPINT 约 `+0.0226`。
- 冻结 H-C 五日期 LODO：static M4 `0.414822`，causal growing cap30 `0.463759`，等日期增益 `+0.048937`。
- carrier×activity 交互为 `-0.0064`；activity 的增益不依赖继续改善 carrier。
- C1 训练见过 M7/M5/M4，但官方部署 identity 固定在 M3。M3→M7 与它的训练支持相邻，不需要把 cardinality 推到 30。

### 1.2 `1024` 不是流上的 chunk 长度

`H1CarrierIdSpint` 的 identity member 长度 1024 来自：取一个真实 trial 的 eval-valid 神经 bins，再做三次插值到 `[1024,176]`。它不是“连续 1024 个原始 20 ms bins”。

对 13 个 held-in-calib session 的只读几何检查得到：

| 量 | min | median | max |
|---|---:|---:|---:|
| 完整 TrialNum span（bin） | 653 | 783 | 980 |
| eval-valid bin / trial | 633 | 759 | 964 |

因此，直接切 1024-bin 流块通常会跨越 trial，并系统性长于训练 member。H1-CAC 的正确做法是：

1. 在 source-only 数据上冻结原始 chunk 长度 `L`；
2. 部署时按 `L` 个连续神经 bins 形成候选；
3. 再把候选按位置三次插值到 `[1024,176]`；
4. 用不变的 C1 `carrier_pre_pool → mean → carrier_post_pool` 算 identity。

### 1.3 冻结的块长

`L` 定义为该折 source dates 全部真实 trial span 的中位数，取最近的 32 的倍数；若等距取较小值。外日期不可参与选择。

| outer date | source trial 数 | source median span | `L` |
|---|---:|---:|---:|
| 19250108 | 139 | 782.0 | 768 |
| 19250113 | 140 | 784.5 | 800 |
| 19250115 | 142 | 783.0 | 768 |
| 19250119 | 140 | 782.5 | 768 |
| 19250120 | 140 | 783.0 | 768 |
| final all-source | 170 | 783.0 | 768 |

这些数只决定时间尺度，不使用 velocity、R² 或外日期结果。

## 2. 核心假设

H1-CAC 检验两个不同问题，不能混为一个门：

1. **内容头寸**：在 C1/H-C consumer 上，M3/M4 之后完成的新 activity member 是否仍提高解码？
2. **无边界构造**：不使用 TrialNum、eval mask、velocity 或 `on_done`，固定长度 chunk 是否能追回真 trial 上界的一部分？

真 trial 臂失败，说明 activity 头寸没有转到目标 consumer；此时不要怪 detector。真 trial 臂成功但 chunk 臂失败，说明内容存在、状态构造还不对；此时不要重训网络。

## 3. 冻结状态机

### 3.1 共同合同

- 权重始终 `eval()`、`no_grad()`；target optimizer/backward/model update 全为 0。
- carrier 永远是 reset 时由最早 M3（Stage 1）或 M4（Stage 0）拟合并冻结的 H-C；query activity 不重拟合 carrier。
- 初始 activity members 永远保留，按 chronological order 放在池前部。
- 总 cardinality 到 M7 后永久冻结；无 FIFO、无 cap30、无代表点替换。
- 每个预测 bin 严格 `decode → append current neural bin → maybe commit`。当前 bin 完成的 chunk 只能影响下一 bin。
- `on_done` 保持 no-op。状态只在 decoder `reset(dataset_tags)` 时清空。
- 每个 batch slot 独立保存 buffer、候选历史、activity pool 与 cached identity。
- identity 只在 commit 后重算；同一状态的多个 query windows 共享缓存。

### 3.2 四臂

| arm | activity law | 角色 | 官方合法性 |
|---|---|---|---|
| `A-STATIC` | 初始 M 不变 | 主对照 | 是 |
| `B-TRIAL7` | 真 trial 完成后提交，M→M7 | 内容上界 | 否，只限本地诊断 |
| `C-FIX7` | 每个完整 `L`-bin 块提交，M→M7 | 无门固定块诊断 | 是 |
| `D-EMED7` | 完整块通过能量门才提交，M→M7 | 主部署候选 | 是 |

`C-FIX7` 只解释能量门的作用，不参与 target-driven 选型；预注册主候选固定为 `D-EMED7`。

### 3.3 D-EMED7 能量门

沿用 M2 Ce-NAT 中已经实现并审计过的纯神经纪律，但不复制它的 100-bin 超参：

- 候选能量 `e = mean(chunk)`，对候选的所有 bin×channel 取 float64 均值；输入是非负 spike/count activity。
- 第一候选必过。
- 此后仅当 `e >= median(previous_candidate_energies)` 时提交。
- 无论接收或拒绝，每个完整候选能量都进入历史。
- median 当前允许 `np.median` 重算；池最多只需接受 3/4 个 member，计算量可忽略。不得写成 O(1) median。
- 门不读取行为、预测、误差、TrialNum、eval mask 或 session 的未来 activity。

该门是 activity-level test-time adaptation，不是“恢复 trial 边界”。若 chunk 比真 trial 更好，这是一个独立的、更密时间采样效应；不得把它写成“等价恢复”。

### 3.4 本地流的 phase origin

本地 LODO 文件把 calibration 与 query 放在同一 NWB。为了模拟官方的“calibration 文件 reset 后进入独立 eval 流”：

- Stage 0（M4）phase origin = 第 5 个真实 trial 的首个原始 bin；
- Stage 1（M3）phase origin = 第 4 个真实 trial 的首个原始 bin；
- origin 只定义模拟 query 流从哪里开始，不供 detector 推断边界；
- 正式 decoder 的 origin 固定为 `reset()` 后收到的第一个 eval neural bin。

## 4. 两级实验

### Stage 0：H-C detector feasibility

目的：先回答“无边界 chunk 能否在 H1 上追回已知 activity headroom”。

- 模型：已封印的五日期 H-C date-LODO checkpoints。
- carrier：各 outer date 原来的 frozen M4 H-C，不变。
- activity：A=M4；B/C/D 最多 M7。
- query：原五日期 strict post-M4 surface；等 recording，再等日期。
- GPU：只做冻结推理，可单卡顺序跑五折。

Stage 0 不证明 C1+M3 成功，也不能直接提交 EvalAI；但它能独立判定 detector 是否值得带到 C1。

### Stage 1：C1-specific M3→M7

目的：回答最终方法问题。

- 模型：五日期 C1 prefix-cycle LODO checkpoints，必须逐 SHA 绑定；若大文件不可得，只能按原封印训练协议重建，不能拿 H-C 或 all-source C1 顶替。
- carrier：每个 outer session 的 frozen M3 H-C；四臂完全相同。
- activity：A=M3；B/C/D 最多 M7。
- query：共同 strict post-M3 surface；同一窗口、target 与 carrier。
- 指标：每 recording variance-weighted last-bin R²，先等 recording，再等 outer date。

Stage 1 是唯一能把结论写成“C1+H1-CAC”的本地证据。

### Stage 2：all-source package / EvalAI

只有 Stage 1 两道门都通过，才把 `D-EMED7` 加到官方 C1 package：

- all-source `L=768`；
- reset seed = held-out-calib M3；
- carrier = 原 C1 M3 H-C；
- eval 流因果补到 M7；
- `IsTestTimeAdaptive=true`，明确是无标签 activity-state TTA；
- EvalAI 提交必须另行获得用户授权。

## 5. 预注册读数与门

每折必须输出 A/B/C/D 的逐 recording R²、逐日期等权 R²、prediction/target/window SHA、commits、候选数、接收序列、commit bins、cardinality trace、identity digest trace、模型状态 before/after。

### Gate 1：内容仍存在

在五日期等权面：

- `mean(B-TRIAL7 − A-STATIC) >= +0.010`；且
- 至少 `4/5` outer dates 为正。

失败：`STOP_H1_ACTIVITY_COMPLETION_CONTENT_DID_NOT_TRANSFER`。不训练 PACD，不调能量门。

### Gate 2：部署律追回内容

令 `H = mean(B-A)`，`R = mean(D-A)`：

- `H > 0`；
- `R / H >= 0.50`；
- 至少 `3/5` outer dates 的 `D-A >= 0`；
- worst date `D-A >= -0.010`。

失败：登记 detector null，保留 B 的内容结论；不得用 C 的 target 数字改 D 的门或 `L`。

若 `D > B`，仍可通过，但必须单独写成“subtrial chunk sampling”，不能声称 trial-equivalent。

### Gate 3：是否值得一次官方提交

Stage 1 通过只授权打包/本地接口哨兵，不自动授权 EvalAI。若之后得到官方数字：

- 相对 C1 submission `581748` 的 Held Out gain `< +0.020`：停止 H1 精度探索；
- `>= +0.020`：才可作为精度贡献讨论；仍需结合官方方差谨慎表述。

## 6. 来自 M2 / DANDI / M1 的配方筛选

本轮使用 brainstorming 的 failure-analysis、composition/decomposition 和 simplicity 框架。候选账本如下：

| 候选 | 来源 | H1 判断 |
|---|---|---|
| 因果 activity memory | H1/M2 | **现在做**；H1 自身有 +0.049 强头寸 |
| 能量门 fixed chunk | M2 Ce-NAT | **现在做**；只迁移状态纪律，不复制 chunk100 |
| prefix/calibration dropout | H1 C1 | **保留现有 C1** |
| paired anchored calibration dropout | DANDI PACD | **条件后继**；仅当 Stage 1 内容过门但部署恢复不足或 M3 extrapolation 明显 |
| activity-only dual memory / CDM | DANDI/M1/M2 | 暂不做；H1 到 M7 即停，尚无慢/快双时标证据 |
| dense auxiliary temporal loss | DANDI | 已有 H1 masked-dense-aux source null（3/5），不先开 |
| 更强 dropout / shorter horizon | M1/M2 | 仅作后继配方；不能与本轮冻结权重机制绑在一起 |
| D-opt / top-K support | M2/M1 | 不做；H1 一 trial 含密速度轨迹，carrier 已饱和 |
| 新 PCA/NNMF/rSyn3 carrier | M1/B1 | 不做；模态和任务定义不匹配 |
| J-R1 / post-fusion | M2 | 不做；晚池化迁移已负且训练/部署易失配 |
| output affine/matrix fusion | M2 | 不做；收益小且不解决 H1 activity 头寸 |
| output EMA | H1 | 已关闭（约 +0.0028） |
| 全量 M30/cap30 | H1 headroom | 暂不做；C1 只见到 M7，先测训练支持内 completion |

## 7. 若 H1-CAC 成功，论文故事是什么

最强、最窄的故事是：

> Few-shot calibration 不应被理解为 reset 时一次性生成的静态 identity。C1 让 consumer 对短校准前缀具有鲁棒性；H1-CAC 则在不更新权重、不读取标签、也没有 trial-boundary signal 的情况下，继续用流中的神经活动补全该 identity。二者分别解决“少量初始校准”和“持续暴露利用”，是正交机制。

它不证明新 carrier，不证明 trial recovery，也不证明任意增长记忆。范围只到 M7。

## 8. 实现优先级与资源

1. 先实现纯状态机、重采样、因果 commit 与单测。
2. 先跑 Stage 0，预计是分钟级冻结 GPU 推理，不训练。
3. 同时恢复五折 C1 checkpoint bytes；没有字节就停止在 Stage 0，不伪造 Stage 1。
4. Stage 1 同样只冻结打分；若必须重建五折 C1，另开训练收据并披露它是复现 checkpoint，不是新方法训练。
5. PACD、CDM、dense aux 都排在本结果之后，不并入第一格。

## 9. 当前 artifact blocker

队友分支保留了 C1 的收据、逐折 checkpoint SHA、评估数字和源码，但 Git 克隆内没有大 checkpoint bytes；本机也没有提交用的 `c1.pt`：

- all-source C1 checkpoint SHA：`0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`
- all-source package SHA：`bfd02e51d2c0309a74b5e835668105f5102b55fe41458d519fce8895f1db5411`
- all-source model-state SHA：`bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85`

Stage 0 不受影响。Stage 1 必须先恢复五个 C1 LODO checkpoint，或用完全相同的 source-only 协议重新生成并明确标记为 reproduction。

## 10. 参考权威

- `HANDOFF_H1_SUCCESSOR_AGENT_20260903.md`
- `WORKORDER_H1_DATE_LODO_ACTIVITY_HEADROOM_V1_20260828.md`
- `h1_date_lodo_activity_headroom_v1.json`
- 队友分支 `exp/h1-cal-aug-all-source-m3-deployment-v1` @ `5dd9bb4a7377a5431b7dbac4f1378e529130eb1a`
- `SPINT-main/src/models/components/h1_carrierid_spint.py`
- `tfpd_exploration/docs/DEPLOYMENT_M2_CENAT_CHUNK100E_TTA_V1_20260903.md`

