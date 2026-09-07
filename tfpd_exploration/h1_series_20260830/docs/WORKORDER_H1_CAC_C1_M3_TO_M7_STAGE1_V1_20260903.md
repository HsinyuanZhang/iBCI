# Work Order: H1-CAC C1/M3→M7 Frozen-Weight Stage 1 V1

日期：2026-09-03  
状态：`AUTHORIZED_BY_USER_AFTER_EXACT_FIVE_CHECKPOINT_UPLOAD`  
父设计：`DESIGN_H1_CAUSAL_ACTIVITY_COMPLETION_V1_20260903.md`  
父结果：`RESULT_H1_CAUSAL_ACTIVITY_COMPLETION_STAGE0_V1_20260903.md`

## 1. 目的

在五个 source-grouped date-LODO C1 checkpoints 上，直接测量 C1 consumer 从静态 M3 activity 到因果 M7 activity 的可用内容及部署律。只做冻结权重推理；不重训 C1，不打开 H1 held-out-calib、minival 或 EvalAI。

## 2. 固定上传权威

远端分支：`exp/h1-cal-aug-prefix-cycle-v1`  
固定 commit：`5b21de415afc35a5f4ad63dd2e8a459d925dbbf7`

五个 C1 checkpoint SHA：

- `19250108`: `904216a596fb71abc72ce2fcdba9ae207eeb258be5fa32fb2aaf533ab42dd047`
- `19250113`: `2a114c5f2edddef9df19167e1d756ab80c1cfa04ae20b46343cb7a848f252d35`
- `19250115`: `0a15fbed098a6488496f8ec7caa5e9ed2ed04c165db80b9ee817c456698af932`
- `19250119`: `276a789edae2419704f4649465a477511b7571185441267c1de1073374fecc42`
- `19250120`: `5d7632c604ad92e52a6ae2bfbc77a875ed7b7a12b5ebc44bbfbb35aa1691d186`

必须同时验证每折 C1 terminal/config/checkpoint metadata、strict model state，以及原 C1 训练使用的 source-selected H-C plan 和 normalizer。禁止复用旧 Stage 0 的 `q=16, λ=100` carrier plan：C1 的五折计划分别为 `(q,λ)=(8,10),(8,10),(16,10),(8,10),(12,10)`。

远端只上传了 `plan.npz` sidecar 而没有 body。允许用完全相同的 source-only 原始数据、固定 source partition、远端代码和已冻结 `(q,λ)` 确定性重建；放行条件是：

1. 五个数组的逐数组 SHA 与 `plan.json` 完全相等；
2. `s_src` 与 `normalizer.json` 完全相等；
3. 重建 `plan.npz` 整体字节 SHA 与远端 sidecar 完全相等；
4. source preparation 的 target recordings/bytes 都是 0。

## 3. 评分面

- 数据：只读 `SPINT-main/data/000954/sub-HumanPitt-held-in-calib`。
- outer dates：`19250108, 19250113, 19250115, 19250119, 19250120`。
- 模型：对应 outer date 的 C1 epoch-49 checkpoint。
- identity seed：每个 target recording 的前三个 chronological trials，M3。
- carrier：同三个 trials 在该 outer fold 的精确 source-selected H-C plan 下拟合并用其 source RMS normalizer 冻结归一化。
- query：从第 4 个 trial 的第一个 eval-valid bin 起，所有 W700 且 endpoint eval-valid 的窗口。
- 指标：每 recording variance-weighted last-bin R²；先等 recording，再等 outer date。
- GPU：只使用物理 GPU0，通过 `CUDA_VISIBLE_DEVICES=0` 映射为逻辑 `cuda:0`；GPU1 不查询、不 signal、不改 affinity。

## 4. 四臂

顺序和 Stage 0 完全相同：

1. `A-STATIC`: M3 固定；
2. `B-TRIAL7`: 真实 trial 完成后提交，M3→M7，本地内容上限，不可官方部署；
3. `C-FIX7`: Stage 0 已登记的 outer-date fixed `L`，完整无重叠 chunk 必收，M3→M7；
4. `D-EMED7`: 同一 chunk 候选，第一块必收，之后 mean-rate energy 不低于此前候选中位数才收，M3→M7。

所有臂必须 decode-before-commit、保护 initial M3、到 M7 冻结、carrier 完全相同、target 不进入状态、模型 state before/after 完全相等。

## 5. 决策纪律

原预注册 primary 仍是 `D-EMED7`，不得因 Stage 0 看见 C 较好而在本结果中换成 C。原两门原样应用：

- 内容门：`mean(B-A) >= +0.010` 且至少 `4/5` 日期为正；
- primary 部署门：`mean(D-A)/mean(B-A) >= 0.50`，至少 `3/5` 日期非负，worst `D-A >= -0.010`。

同时允许报告一个明确标为 **post-Stage0 selection-informed、non-governing** 的 C 读数。它使用同一阈值只回答 fixed chunk 是否值得另开 successor confirmation；不得替换 D 的正式结论，不得把同一批 target 数字再称为独立确认。

## 6. 输出与停止边界

独立新根：

`tfpd_exploration/h1_series_20260830/results/h1_causal_activity_completion_v1/stage1_c1_m3/`

成功拓扑至少为 attempt、input_authority、score、terminal 四个 body+sidecar；失败保留不可变前缀并写 failure。不得覆盖 Stage 0。

本单不授权：

- 再训练、PACD、改 chunk length、改 energy statistic；
- all-source package；
- Docker；
- EvalAI 提交。

