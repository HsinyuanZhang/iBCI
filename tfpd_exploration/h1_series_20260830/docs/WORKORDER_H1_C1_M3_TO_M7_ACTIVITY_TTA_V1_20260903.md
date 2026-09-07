# Work Order: H1-CAC M3/M4→M7 Frozen-Weight Screen V1

日期：2026-09-03  
状态：`AUTHORIZED_STAGE0_IMPLEMENT_AND_SCORE_ONLY`  
设计：`DESIGN_H1_CAUSAL_ACTIVITY_COMPLETION_V1_20260903.md`

## 1. 任务

实现 H1 Causal Activity Completion（H1-CAC），先在五日期 H-C LODO checkpoints 上完成 Stage 0 四臂冻结权重打分。Stage 1 的 C1 M3→M7 代码可以实现，但在五个 C1 checkpoint bytes 未恢复且逐 SHA 校验前不得打分。不得训练、不得打开正式 held-out、不得提交 EvalAI。

## 2. 只允许新增

建议新增：

- `tfpd_exploration/h1_series_20260830/src/h1_causal_activity_completion_v1/`
- `tfpd_exploration/h1_series_20260830/scripts/run_h1_causal_activity_completion_v1.py`
- `tfpd_exploration/tests/test_h1_causal_activity_completion_v1.py`
- `tfpd_exploration/h1_series_20260830/results/h1_causal_activity_completion_v1/stage0/`

不得修改或覆盖：

- 五日期 H-C checkpoint/config/terminal/source manifest；
- 队友 C1/T0 结果根；
- 现有 `h1_date_lodo_activity_headroom_v1.json`；
- FALCON evaluator、sealed H1 source files、现有 submission。

## 3. Stage 0 authority

必须复用 `h1_date_lodo_activity_headroom_v1.plan.AUTHORITIES` 的五折：

`19250108, 19250113, 19250115, 19250119, 19250120`

每折在打开 checkpoint/target 前重验 terminal、checkpoint、config、source manifest SHA。只读 `SPINT-main/data/000954/sub-HumanPitt-held-in-calib`。以下计数必须保持 0：

- held-out-calib opened
- held-in-minival opened
- EvalAI/test opened
- target optimizer steps
- target backward steps
- target model updates

## 4. 固定常数

- seed support：Stage 0 M4；Stage 1 M3。
- maximum total members：M7。
- H1 identity member output shape：`[1024,176]`。
- decoder window：W700。
- batch size：32。
- carrier：每 session 原 frozen H-C，四臂逐字节一致。
- chunk raw length by outer date：
  - `19250108:768`
  - `19250113:800`
  - `19250115:768`
  - `19250119:768`
  - `19250120:768`
- final package raw length：768（本单不打包）。

块长不得根据 outer-date R²、prediction 或 target 修改。

## 5. 四臂精确合同

顺序固定：`A-STATIC, B-TRIAL7, C-FIX7, D-EMED7`。

### A-STATIC

永远使用最初 support members。

### B-TRIAL7

仅限本地上界。一个真实 post-support trial 完成后，将该 trial 的 eval-valid activity 按现有 cubic law 插值到 `[1024,176]`；先 decode trial 的最后 bin，后 commit。池到 M7 后冻结。

### C-FIX7

从 query stream origin 开始，使用不重叠、无间隙、固定 `L` 的原始神经块。完整块按位置 cubic 插值到 `[1024,176]` 后必提交；到 M7 后冻结。

### D-EMED7

候选块与 C 相同。候选 float64 mean rate 为能量。第一候选接受；之后若且仅若当前能量不低于此前所有候选能量的中位数才接受。每个候选无论是否接受都更新历史。到 M7 后冻结。

四臂必须满足：

- 当前 endpoint 只能看 completion bin 严格小于 endpoint 的新 member；
- initial support 永不删除；
- 不读取 velocity/target 做状态更新；
- 不读取 TrialNum/eval mask 构造 C/D；
- C/D 的唯一 phase origin 是模拟 eval stream 的起点；
- model state before SHA = after SHA。

## 6. 必须先过的测试

CPU synthetic：

1. raw `L` → cubic `[1024,176]` shape/finite/deterministic；
2. decode-before-commit 的 off-by-one；
3. M4→M7 后不再接受；
4. M3→M7 后不再接受；
5. D 第一块必收、后续 running-median 判定、拒绝块仍入历史；
6. batch slots 状态隔离；
7. `on_done` 不改变 pool/buffer；
8. A 与现有 static replay prediction/R² 数值校验；
9. B 与现有 causal selection 在 M≤7 区间一致；
10. C/D 不接收 TrialNum、eval mask、velocity、target 参数。

在 GPU 正式 Stage 0 前做一个 session smoke，并输出：A parity、四臂同 target/window/carrier、state immutable、CUDA device/UUID。

## 7. Stage 0 产物

结果根一旦创建不得复用。成功根至少包含：

- `attempt.json` + sidecar
- `input_authority.json` + sidecar
- `score.json` + sidecar
- `terminal.json` + sidecar

失败根：保留已发布不可变前缀，再写 `failure.json` + sidecar；不得删除或覆盖前缀。

`score.json` 必须含：

- 五折、每 recording、四臂 R²；
- prediction/target/window/carrier SHA；
- chunk length 与 phase origin；
- candidate/accept/reject/commit counts；
- candidate energy history SHA、accepted-bit SHA、commit-bin SHA；
- cardinality trace 与 identity-state SHA；
- per-date 与 equal-date `B-A, C-A, D-A`；
- recovery `mean(D-A)/mean(B-A)`；
- Gate 1 / Gate 2 独立判定；
- model state before/after；
- formal held-out/minival/EvalAI access 均为 0。

## 8. 裁决

Gate 1：`mean(B-A) >= +0.010` 且 `positive dates >=4/5`。

Gate 2：`mean(B-A)>0`，`mean(D-A)/mean(B-A)>=0.50`，`D-A>=0` 至少 3/5，worst `D-A>=-0.010`。

- Gate 1 fail：整条 activity completion 停止。
- Gate 1 pass / Gate 2 fail：内容存在、无边界 detector 失败；不训练网络救 detector。
- 两门 pass：允许进入 Stage 1 artifact recovery/reproduction；不自动授权 EvalAI。

`C-FIX7` 永远只描述，不参与把 D 换掉。看见 target 后不得换主 arm。

## 9. Stage 1 硬阻塞

五个 C1 date-LODO checkpoint body 当前不在本机克隆中。训练收据记录的 SHA 为：

- 19250108 C1 `904216a596fb71abc72ce2fcdba9ae207eeb258be5fa32fb2aaf533ab42dd047`
- 19250113 C1 `2a114c5f2edddef9df19167e1d756ab80c1cfa04ae20b46343cb7a848f252d35`
- 19250115 C1 `0a15fbed098a6488496f8ec7caa5e9ed2ed04c165db80b9ee817c456698af932`
- 19250119 C1 `276a789edae2419704f4649465a477511b7571185441267c1de1073374fecc42`
- 19250120 C1 `5d7632c604ad92e52a6ae2bfbc77a875ed7b7a12b5ebc44bbfbb35aa1691d186`

只有两条合法路径：

1. 恢复与上述 SHA 完全一致的 checkpoint；或
2. 新建独立 reproduction work order，按原 source-only、seed42、50 epoch、M7/M5/M4 prefix-cycle 协议重新训练，并明确新 checkpoint 不是旧字节。

禁止使用普通 H-C date-LODO checkpoint、all-source H-C、DANDI CAL-AUG checkpoint 或 all-source C1 代替五折 C1。

## 10. GPU 与运行纪律

- Stage 0 只占一个明确指定的空闲 GPU；不得查询、signal、改 affinity 或占用另一张卡上的进程。
- 首个 session smoke 预计总时超过 2 小时则停止并登记，不悄悄降 query 或换指标。
- 无 checkpoint 训练，无 optimizer。
- 所有选择在 target 打分前冻结。
- 本单不执行 Docker build、network push 或 EvalAI API。

