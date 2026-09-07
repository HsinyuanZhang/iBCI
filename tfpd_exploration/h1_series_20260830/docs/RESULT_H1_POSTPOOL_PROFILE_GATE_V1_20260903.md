# Result: H1 Post-Pool Profile Gate V1

日期：2026-09-03  
状态：`COMPLETE_STOP_H1_POSTPOOL_PROFILE_FAMILY`

## 1. 结论

在五个 source-grouped date-LODO C1 checkpoint 上，700 参数的共享时间 profile
能够从 MLP 后 pooling residual 中提取正的平均增益，但不能稳定迁移到新日期。
五折平均增益为 `+0.007046 R²`，超过预注册的 `+0.005` 效应门；然而只有
`2/5` 日期非负，worst 为 `-0.005267`，没有达到 `4/5` 的稳定性门。因此整体
裁决为 FAIL，不训练 all-source profile，也不制作或提交 EvalAI package。

这不是“MLP 后 pooling 完全没有信息”的结果。正确信息是：晚池化 residual
具有可学习内容，但当前低容量 source-learned profile 的收益主要集中在一个日期，
不能作为跨日期 few-shot calibration 方法部署。

## 2. 五折结果

| outer date | source gain | outer-date delta | optimizer steps | max abs tanh(profile) |
|---|---:|---:|---:|---:|
| 19250108 | +0.003596 | +0.007622 | 768 | 0.6502 |
| 19250113 | +0.014578 | **+0.037656** | 912 | 0.8592 |
| 19250115 | +0.004914 | -0.003875 | 912 | 0.8481 |
| 19250119 | +0.001506 | -0.000905 | 888 | 0.6753 |
| 19250120 | +0.003544 | -0.005267 | 912 | 0.8122 |

所有折的 profile gradient 均非零；每折 C1 基模型的 state before/after 完全相同；
outer date 没有 optimizer、backward 或模型更新。均值主要由 19250113 的
`+0.037656` 拉动，因此不能用正均值掩盖 `3/5` 日期为负的事实。

## 3. Exactly-M3 官方边界

H1 的全部 14 个 held-out calibration 文件都只有恰好三条合法 calibration
trial。因而：

- M3 是官方 calibration 资源上限，不是人为选择的 budget；
- M3 activity identity 与 M3 H-C carrier 已耗尽 reset 时可用的校准 activity
  和标签；
- 评测流中已经过去的 neural bin 可以在合法的无标签 TTA 合同下形成因果记忆，
  但不能称为第 4 条或第 7 条 calibration trial；
- 任何使用 M4/M5/M7 的训练方法都必须说明它是在 source 训练中扩大 support，
  或是在部署时从无标签 query activity 完成 identity，而不是额外读取 target
  calibration label。

现有 C1 的训练 prefix cycle 为 M4/M5/M7，官方部署却从 M3 冷启动。后续若继续
H1，首要问题应是训练分布与官方 M3 合同的匹配，而不是把记忆容量继续增大。

## 4. 三层晚池化结论

截至本格，冻结 C1 上的三层后继已经闭合：

1. 直接 post-pool identity：primary mean `-0.001379`，FAIL；
2. source-fit signed scalar output residual：mean `-0.002385`，FAIL；
3. 700 参数时间 profile：mean `+0.007046`，但仅 `2/5` 日期非负，FAIL。

因此不再在冻结 C1 上追加 per-unit gate、matrix、rank 或 profile 宽度搜索。该结论
只关闭“在既有 early-pool C1 解之上做小型晚池化补丁”的路线；它不证明从匹配的
M3 训练分布中联合训练 late-pool 网络必然失败。

## 5. 权威收据

- result root：`tfpd_exploration/h1_series_20260830/results/h1_postpool_profile_gate_v1/`
- attempt SHA256：`83ede9315689a40c66ac76d609ea1f805bfc641ccf99bb427df6f4eec0851cf3`
- score SHA256：`d70524f4b0df907ad8017e64bede429a1801738e17aa6bd11c038285a51a541e`
- terminal SHA256：`905d0a46bda86a69a31ea290b73a90db674e958c981cfef154101fddec68e3b4`
- formal held-out opened：false
- EvalAI opened/submitted：false
- target optimizer/backward/model updates：全部 0
- GPU：只使用 GPU0；GPU1 未触碰

