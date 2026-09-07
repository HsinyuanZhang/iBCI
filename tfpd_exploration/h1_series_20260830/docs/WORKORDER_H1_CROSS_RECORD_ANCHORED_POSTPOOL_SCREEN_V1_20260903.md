# Work Order: H1 Cross-Record Anchored Post-Pool Screen V1

日期：2026-09-03  
状态：`AUTHORIZED_SOURCE_ONLY_GPU0_SCREEN`  
设计：`DESIGN_H1_CROSS_RECORD_ANCHORED_POSTPOOL_MEMORY_V1_20260903.md`

## 1. 目标

执行五折 H1 C1 冻结权重、calibration-recording→minival-recording 屏。只比较 A-STATIC、RN(g) 与 RP(g)，`g={0,.05,.10,.20,.50,1}`；禁止打开 held-out、EvalAI 或私有标签。

## 2. 权威输入

- 五个 C1 date-LODO epoch-49 checkpoint 及 source plan：沿用 `h1_causal_activity_completion_v1.stage1.C1_AUTHORITIES` 的完整 SHA 绑定；
- H1 public held-in-calib 13 recordings；
- H1 public held-in-minival 对应 13 recordings；
- window=700、activity member length=1024、chunk length=768、M3 protected、最多 M7；
- GPU0 UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`；GPU1 不得查询、占用、发信号或修改进程。

## 3. 执行顺序

每折：

1. 写 attempt；
2. 校验 checkpoint/plan/source closure；
3. 只打开 source-date calib/minival，计算候选 source score 并冻结 `(family,g)`；
4. 写 selection receipt；
5. 才打开 outer-date calib/minival，计算一次 selected arm 与全候选描述曲线；
6. 写 fold receipt；
7. 五折完成后写 score/terminal。

如实现暂时不能做到 target 文件延迟打开，则不得称严格执行本工单；需要先修 loader。

## 4. 必须验证

- H1 evaluator continual 合同：不使用 `on_done`；
- query chunk 只读 raw neural，origin=0，first commit exclusive=768；
- eval mask/velocity/TrialNum/trial_change 不进入 state construction；
- g=+0 direct branch 与 static prediction SHA 相同；
- 所有模型参数 before/after SHA 相同；optimizer/backward/model update 全为 0；
- source selection 与 outer scoring session/date 集严格不交；
- RP/RN 使用相同 members、carrier、neural windows、targets；
- 全流 primary 与 post-commit descriptive 分开；
- GPU0 only；GPU1 untouched。

## 5. 停止/继续

按设计 §6 原样执行，不在结果后修改门。冻结权重 OOF 过门才允许打包；RP 非负但未过门才允许另开一个 12-epoch matched-training work order；其余情况停止。

本工单不授权 Docker push 或 EvalAI POST。目标达到 package-ready 后，由根代理依据用户持续目标和最终本地哨兵执行独立提交门。

