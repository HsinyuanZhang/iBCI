# Work Order: H1 M3 Cross-Record Joint Post-Pool V1

日期：2026-09-03  
状态：`AUTHORIZED_SOURCE_ONLY_GPU0_DATE_LODO`

实现并执行 `DESIGN_H1_M3_CROSS_RECORD_JOINT_POSTPOOL_V1_20260903.md` 的五折三臂
screen。仅使用物理 GPU0；不得查询、占用、signal 或修改 GPU1 上的任何作业。

## 固定合同

- predecessor：五个 immutable C1 epoch-49 date-LODO checkpoints；
- negative-result predecessor terminal：
  `h1_postpool_profile_gate_v1/terminal.json`, SHA256
  `905d0a46bda86a69a31ea290b73a90db674e958c981cfef154101fddec68e3b4`；
- source：fold source roster 的 held-in-calib + held-in-minival；
- target validation：outer date 的 held-in-calib + held-in-minival，只在训练完成后打开；
- formal held-out/EvalAI：本阶段禁止；
- arms：`FROZEN-C1`, `N3-XR12`, `J3-XR12`；
- M3 support/carrier、W700、stride4、12 epoch、batch32、seed42、Adam `5e-5`、
  weight decay0、last-bin MSE `/20`；
- J3 只有一个新增 scalar alpha，IEEE +0 初始化；不得添加 profile/matrix/rank；
- 两个训练臂必须 batch/RNG paired；第一步前 prediction bitwise equality；
- fixed epoch 11，无 checkpoint/epoch/sweep selection；
- outer target optimizer/backward/update 全部为 0。

## 生命周期

新 root：
`tfpd_exploration/h1_series_20260830/results/h1_m3_cross_record_joint_postpool_v1/`

先发布 immutable `attempt.json`，然后逐折发布训练/评分 receipt，最后发布
`score.json` 与 `terminal.json`；异常只发布 `failure.json`，不得覆盖或重试该 root。

五折 gate 按设计文档字面执行。FAIL 时停止；PASS 时另立 all-source/package work
order，不能在本 root 内继续训练或访问 EvalAI。

