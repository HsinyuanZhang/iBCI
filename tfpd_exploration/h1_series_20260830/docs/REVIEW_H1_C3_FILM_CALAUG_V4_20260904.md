# C3 实验设计审核文档（V4：CAL-AUG C2 配方 + EP-FiLM 联合训练）

Date: 2026-09-04（设计冻结版，v1.1 同日修订：双面选择 + profile 粒度修正）
Status: **WAITING_FOR_DESIGN_REVIEW —— 实验未启动。** 已完成的只有：代码、
保真度测试、工件下载与 SHA 验证、attempt 收据。训练/选择/打包/提交均未发生。
预注册工作单：`WORKORDER_H1_C3_FILM_CALAUG_V4_20260904.md`（同目录）。
代码工作区：`/home/xinyuan/Work_host/ibci_c3_film`（iBCI 分支
`exp/h1-cal-aug-m3-aware-dual-selection-v2` 的克隆 + C3 增量，提交 `2740c35`）。

---

## 一、背景与动机（为什么做这个实验）

1. **我方现状（SPINT/tfpd 线）**：V3 提交 581866（冻结 C1 + MAT7 readout +
   EP-FiLM）官方 HO R² = 0.2675，低于其冻结基座 C1（0.2841，队友 581748）。
   分解显示 readout 校准单独 −0.043（581792），FiLM+EP 在链内收回 +0.027，
   但整个"冻结基座+加模块"路线被压制。
2. **队友现状（cal-aug 线，独立 EvalAI 队伍）**：C2 = 把 M3 鲁棒性训练进
   decoder 本体（identity 前缀循环 M7/M5/M4/M3），官方 HO：E49 无选择
   0.3050、HI-E45 0.3240、**HO-E15 0.3760**（HO 面 = held-out-calibration
   录音上的 development/model-selection 面）。
3. **用户决定**：把队友的整套训练配方迁移过来，加入我方 V2 本地验证过的
   EP-FiLM 设计，联合训练后按队友的标准选择方法选 epoch，提交拿分。

## 二、C3 一句话定义

**C3 = 队友 C2 配方原样（不改任何超参/数据绑定） + 我方 V2 验证过的
648 参数 EP-FiLM 以零初始化并入联合训练 + 双面选择（HI/HO），
部署候选 = HO 面选出的 epoch，打包为无 readout 的缓存 identity 提交。**

与近邻的区分：

| 臂 | decoder 训练 | FiLM | 选择 | 官方 HO |
|---|---|---|---|---|
| C2（队友） | ✓ cal-aug 配方 | 无 | HI 或 HO 双面 | 0.3050 / 0.3240 / 0.3760 |
| V3（我方 581866） | 冻结 C1 | ✓（单独训） | 无 | 0.2675 |
| **C3（本实验）** | ✓ 同一 cal-aug 配方 | ✓ 零初始化并入 | 同 C2 双面 | 待产生 |
| C1（队友 581748） | ✓（无 M3-aware 循环） | 无 | 无 | 0.2841 |

## 三、配方迁移清单（对 C2 的"原样"承诺与绑定）

以下全部 SHA 绑定，来自队友密封的
`h1_cal_aug_m3_aware_dual_selection_v2_contract.py`（Git 分支
`exp/h1-cal-aug-m3-aware-dual-selection-v2`，克隆后原样 import）：

| 绑定项 | 值 |
|---|---|
| V1 初始态 SHA | `bc6dc8a0543c760811f770206c7ee22ae35eaf970c6dad0ec259a84172e4d04b` |
| V1 dropout 概率流 digest | `c1dd24d682878f477050cb4e5886dd1f34aff3424b58bdcb158c12c11ba1d247` |
| V1 源权威 / plan / normalizer / carrier cache / schedule / batch order / M7 schedule / source tensor | 见契约常量（`V1_*_SHA256`），全部校验 |
| 前缀循环 | `(M7,M5,M4,M3)` 确定性平衡循环，4133 batch/epoch |
| 训练 | seed 42，batch 32，Adam 5e-5（单一组，见 §4 优化器），wd 0，FP32，50 epochs，206650 步，dynamic dropout，last-bin MSE after `/20` |
| checkpoint | 每 epoch 保存、SHA 记录、0444 |
| 中途行为 | 无验证、无选择、无早停、无 warm start |

**明确的非目标**：不追求与队友 C2 运行位级一致（跨 GPU 不可能）；绑定的是
配方与 V1 权威，不是他们的 checkpoint。C3 是新臂，完整性收据是自指的。

## 四、FiLM 集成（"不错位"的证据链）

### 4.1 设计来源 = V2 本地验证过的原版

V2 的 LODO 2×2 结论（本地密封收据）：EP-FiLM vs EP-ZERO **+0.0239，4/5
日期正，分类 `FILM_EARLY_REPLICATION_ONLY`**。C3 的 FiLM 模块、条件输入、
参数预算全部 verbatim 移植自 `h1_calibration_profile_film_v1/core.py` +
`plan.py`（即该验证所用的实现）。

### 4.2 执行过的证明（非口头断言）

`tests/test_h1_c3_film_fidelity.py`，**11 项全部 PASS**（提交 `2740c35`）：

1. 15 个常量逐一相等（UNITS=176、TRIAL_LENGTH=1024、CONTEXT_DIM=8、
   FILM_RANK=8、FILM_PARAMETERS=648、PROFILE_MASK=(1,1,0,0)、分位数等）；
2. `build_film` 同种子初始化逐张量相等；末层零初始化；参数计数 648；
3. C3 算子 `film_identity_prefix` 与 V1 `film_identity(late=False)` 在 M=3
   下 `torch.equal` 位级相等；
4. 零初始化 FiLM identity 与 `H1CarrierIdSpint.carrierid_identity_projection`
   （C2 模型原生 identity 路径）在 **M=3 和 M=7 下都 `torch.equal`**——
   这保证训练起点与 C2 架构严格重合；
5. robust_z / profile_from_arrays / profile_from_support 三个构建器位级相等。

### 4.3 修正过的一处错位（审核重点）

初版把训练期 profile 写成"每 session 固定最早 3 trial"。**V2 的设计是按
support block 计算 profile**（该 block 的 3 个 trial，与该 block 的 M3
carrier 同源）。已改为按调度块逐 block 计算；部署 payload 的 block 即
session 的公开最早-M3（start=0），训练/部署共用同一 profile 构造。
审核人可对照 V2 的 `_profile_session_row`（per-block profile）与本工作单。

### 4.4 零初始化锚与 dropout digest 绑定

- **锚**：训练第 0 步前，在首个调度 batch 上断言
  `torch.equal(model(直接前向), decode_with_identity(model, film_identity_prefix(零初始化)))`，
  在 eval() 下运行（dropout 透传、无随机性），随后**恢复 python RNG 状态**。
- **digest 为什么仍然成立**：dropout 概率流是 python `random`（结构化的
  调用序列，与激活值无关）；FiLM 构造只消耗 torch RNG（与该流独立）；
  锚消耗的 draw 被状态恢复撤销。故 C3 的概率 digest 应逐字节等于 V1 的
  `c1dd24d6…`，训练后由 integrity 阶段硬校验——**若不等，训练判失败**。
- **优化器决定**：单一 Adam 5e-5 覆盖 substrate+FiLM（配方的忠实迁移）。
  V3 的 FiLM-only lr 3e-4 不适用于联合训练——这是明确的制度切换，
  已写进预注册 §3。

## 五、选择协议（按用户确认：held-out 选 epoch 是 FALCON 团队标准做法）

- 复用队友密封的 `select_epoch`（tie-break：higher mean → higher
  worst-session → lower std → earlier epoch）与 `_surface_metrics` 聚合。
- **双面**：HI-M3（held-in minival，13 文件）与 HO-M3（held-out-calibration
  14 录音，development/model-selection 面）。50 epoch + C1-e49 基线全部双面
  评估（约 51×2 次推理，GPU 上约 40–60 分钟）。
- **部署候选 = HO 面选出的 epoch**——与产生 C2-HO-E15 (0.3760) 的实践完全
  相同；提交 metadata `IsHeldOutZeroShot=false`、`IsTestTimeAdaptive=false`。
- held-out 录音**只**用于该选择面（用户确认的官方认可标准实践），不读任何
  hidden-test query label，无任何在线适应。

## 六、打包与部署设计（未实施，设计已定）

- payload：选中 epoch 的 checkpoint（substrate+FiLM 融合）+ 27 个 session 的
  缓存 (activity[3,1024,176], carrier[176,4], profile[176,4])。
- decoder：`h1_c3_spint_decoder.py` = 我方 V3 EP-FILM decoder 的去 readout
  移植（reset 时算一次调制 identity，predict 零拟合零更新）。
- **无 MAT7 readout**（官方证据：readout 单独 −0.043）。
- 容器 CPU smoke + host/container parity（紧阈值 1e-4）照 V3 模式。

## 七、权威链与工件（全部 SHA 已验证）

| 工件 | 来源 commit | SHA-256 | 本地验证 |
|---|---|---|---|
| plan.npz | `8f08c1f0`（队友推送） | `73eb2d05…cec44` | 与本地 sidecar MATCH，已落位 0444 |
| carrier_cache.npz | 同上 | `d9e17664…6b64c` | MATCH，落位 |
| schedule.npz | 同上 | `0cc24ef8…a96c` | MATCH，落位 |
| c2_epoch_015.ckpt | `8ea860c3`（队友推送） | `ce46267e…0215` | MATCH（=密封 C2-HO-E15） |
| c2_epoch_045.ckpt | 同上 | `83d56e8c…352f` | MATCH（=密封 C2-HI-E45） |
| C1 epoch-49（基线） | 本机原有 | `0f406a8e…2cd06` | 长期密封 |

落位：npz 在 predecessor root
`/tmp/ibci-h1/deployment-v1/.../h1_cal_aug_all_source_m3_deployment_v1/source_authority/`；
ckpt 在克隆 `tfpd_exploration/h1_series_20260830/artifacts/c2_references/`。

## 八、当前状态与已知问题（诚实清单）

1. **已发生**：attempt 收据（无数据访问）、代码+测试提交、五工件下载验 SHA、
   保真度测试 ALL PASS。
2. **source 阶段失败（唯一一次运行尝试，fail-closed 正确拦截）**：
   predecessor root 的密封 JSON（authority.json 等）在 9/3 同步时丢失 0444
   权限（664），队友代码的 `verify_sidecar` 拒绝。**修复 = 对该目录密封文件
   `chmod 444`（不改内容，SHA 不变）**，待审核通过后随重跑执行。
3. **GPU 偏离**：工作单原定 GPU0；启动时 GPU0 被另一 agent 的 dandi688
   seed 任务占用（pid 1429114/1430600），改为空闲 GPU1，原因已记录进
   attempt 收据与工作单 §6。
4. **尚未发生**：parity 门（见 §9）、训练、integrity、双面选择、打包、
   smoke、提交。提交动作无论如何都由用户本人在自己终端执行（≤1 次授权）。

## 九、训练前的第三道门（设计已定，未运行）：C2 参照 parity

用我们刚下载的 E45/E15 checkpoint 经**我们的** payload 构建 + 他们的 decoder
+ 我们的评估链重算，与队友密封的逐-recording R² 对比：

- E45 @ HI 面 vs `selection/c2_hi.json`（13 recordings，密封均值 0.9675）；
- E15 @ HO 面 vs `selection/c2_ho.json`（14 recordings，密封均值 0.4056）。

判据：逐-recording |ΔR²| ≤ 1e-3（跨 GPU FP32 推理量级）。**不过则停止，
不得进入训练**——这同时验证 materialization、payload、decoder、评估器四层。

## 十、预注册解读框架（出分前锁定）

| 对照 | 官方 HO | 含义 |
|---|---:|---|
| **C2-HO-E15 (581814)** | **0.3760** | **同类锚（同选择实践、无 FiLM）：C3 有意义当且仅当超过它** |
| C2-HI-E45 (581813) | 0.3240 | 诚实选择参照 |
| C2-E49 (581812) | 0.3050 | 无选择参照 |
| C1 (581748) | 0.2841 | 冻结基座参照 |
| V3 EP-FILM (581866) | 0.2675 | 我方前作 |

- 允许的主张：*"在 C2 cal-aug 配方与相同选择协议下，V2 验证过的
  calibration-profile FiLM 带来/未带来 X 的官方 HO 变化"*。
- 不允许的主张：FiLM 改进 SOTA（若未超过 0.3760）；任何 per-trial TTA 效果
  主张（本设计无在线适应）；把本地 HO 面数字当作 held-out 泛化估计。
- 诚实预期：本地 HO 面 → 官方有偏移（E15：本地 0.4056 → 官方 0.3760），
  本地增益不保证迁移；本地任何数字都不得用于事后调整。

## 十一、给审核人的检查清单

1. **FiLM 等价性**：跑 `tests/test_h1_c3_film_fidelity.py`（CPU，~1 分钟），
   确认 11 项 PASS；抽查 §4.3 的 profile 粒度修正是否与 V2 设计一致。
2. **零初始化/digest 论证**：§4.4 的 RNG 流分析是否成立；integrity 阶段的
   digest 硬校验是否足以作为失败判据。
3. **选择协议**：§5 的双面选择与队友密封实践是否完全同构
   （对照其 `run_offline_validation`）；部署候选=HO 是否为团队共识。
4. **绑定完整性**：§3 表中的 V1 SHA 绑定是否有遗漏/放宽（尤其前缀循环改为
   本地重算而非读密封 npz——`prefix_schedule()` 是确定性 SHA-256 推导，
   值得复核）。
5. **合规边界**：§5 的 held-out 使用范围声明是否需要向主办方再确认
   （用户已确认为官方标准做法；此条仅为复核提示）。
6. **打包面**：§6 无 readout 的决定（−0.043 官方证据）是否认可。
7. **解读框架**：§10 的锚点与禁则是否认可。

## 十二、全部路径速查

- 预注册工作单：`SPINT/tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_C3_FILM_CALAUG_V4_20260904.md`
- 本文档：`SPINT/tfpd_exploration/h1_series_20260830/docs/REVIEW_H1_C3_FILM_CALAUG_V4_20260904.md`
- 代码：`/home/xinyuan/Work_host/ibci_c3_film`（提交 `2740c35`）：
  - `SPINT-main/src/h1_c3_film_calaug_v4_{film,contract,exec}.py`
  - `SPINT-main/third_party/falcon_challenge/h1_c3_spint_decoder.py`
  - `run_h1_c3_v4.py`（stages：attempt→source→train→integrity→evaluate→terminal）
  - `tests/test_h1_c3_film_fidelity.py`
- C3 结果根（待生成）：`ibci_c3_film/tfpd_exploration/h1_series_20260830/results/h1_c3_film_calaug_v4/`
- V3 收据（581866）：`SPINT/tfpd_exploration/h1_series_20260830/docs/RESULT_H1_EPFILM_EVALAI_SUBMISSION_V1_20260904.md`
- 队友密封对照：分支 `exp/h1-cal-aug-m3-aware-dual-selection-v2-evalai-a1`
  的 `official_results.json` / `selection/c2_{hi,ho}.json`（已存
  `/tmp/c3_downloads/sealed_c2_{ho,hi}.json`）
