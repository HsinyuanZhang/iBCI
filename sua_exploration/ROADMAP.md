# SUA/MUA Shared Encoder Roadmap

**状态：当前执行计划**  
**更新：2026-07-30**

## 总目标

在统一、可复现的评估口径下回答三个递进问题：

1. B3 是否是可靠的跨信号类型架构基线？
2. ~~B15/B16 是否稳定优于 B3，且改进是否为 SUA 特有？~~ **已回答（否）**，见 P2。
3. 最终应选择按信号类型重训、共享 backbone，还是联合/context-conditioned 模型？

部署侧的取舍标准见 [`docs/ASIC_DEPLOYMENT_CHARTER.md`](docs/ASIC_DEPLOYMENT_CHARTER.md)：
本项目的贡献点是芯片架构的可重构能力，性能不是核心诉求。因此
"改动落在哪个速率域"与 R² 同等重要。

## 当前实验计划（2026-07-30 更新）

### 当前主线优先级

1. **M2 FP32：T4 vs 本地 full SPINT**——同一个 chronological first-33
   calibration prefix；T4 可使用其中 target labels；all-held-in T4 三种子。
2. **SUA FP32：T4 vs 原始 SPINT B0——已完成并 strict-positive。**
   B0/T4/TS4 × 三种子，activity、label side-pool 与 evaluation pool 均为
   first-30；`T4−B0=+0.33856`、`T4−TS4=+0.29045`，全部预设 gate 通过，
   formal test 仍封存。
3. **额外实验 1：confidence-conditioned FiLM**——固定 activity budget 和
   common evaluation start，测试 calibration confidence 能否将 labeled T4
   budget 从 50 降到 10/15/20，同时保持普通 `T4@50` 的性能。
4. **额外实验 2：T4-conditioned decoupled cross-attention**——在 T4 表征
   冻结后公平比较 coupled decoder 与 `K(E,T4),V(x)`；只有第 3 项通过时才把
   confidence 加入 key。
5. **INT8**——只量化前述 FP32 实验选出的最终架构，依次做 encoder PTQ、
   必要时 QAT；decoder 量化不在本地重复。
6. **Formal held-out**——最终架构和量化状态全部冻结、G1 scope 治理解决后
   才执行一次，模型选择期间保持 test sessions 封存。

协议与验收规则：
[`docs/FP32_T4_MAINLINE_PROTOCOL.md`](docs/FP32_T4_MAINLINE_PROTOCOL.md)。

`side_feature_ablation_v2` 的结论是 `indeterminate`，但它交付的**方法学
产出比科学产出更重要**：`σ_seed = 0.0385` 是主导方差源，M3 完全没触及它，
门槛 `+0.03` 再次落在噪声底之下（[`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §I）。

E1/E2 已完成：延长训练不是杠杆，SWA 也不降低跨 seed 方差。真正改变局面的
是 E3 的功能侧信息 T4/T8；pseudo-MUA bridge 随后证明该收益不依赖 sorted-unit
view。当前执行重点从“修测量”转为“拆解并优化 T4，同时保持跨 view 与部署约束”。

| # | 实验 | 目的 | 状态 |
|---|---|---|---|
| E1 | SWA/EMA 权重平均 | ✅ 完成，**次要**。通用训练技巧，非本项目科学贡献。`+0.022`，不降 `σ_seed`。见 §J.2 |
| **E2** | 收敛性长 run | ✅ **完成**。**不存在跨 seed 共享的收敛点**；延长训练不改善平均表现且放大 seed 分歧（1.7×）。epoch 预算不是杠杆，沿用 `E=12`。见 §J.1 |
| **E3** | 方向调谐特征 | ✅ **完成，项目首个 `effective`**。T4 `+0.2528` / T8 `+0.2567`，6/6 session、3/3 seed 全正，置换对照落在基线上，`σ_seed` 减半；两个经典线性对照也已完成。见 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §J.5 |
| **E4** | 时间基 φ（`B3T`）/ trial 轴 attention（`B3A`） | ✅ **完成**。`B3T` 六 seed 全正 `+0.0418 ± 0.0090 SE`、参数 −31%/MAC −65%，但 session 4/6 正，判 `effective_heterogeneous`；`B3A` 零信号 + 30× state → **放弃**。见 §J.3 |
| **P-MUA** | SUA→pseudo-MUA T4 bridge | ✅ **完成**。9/9 run；T4−F0 `+0.3177`、T4−TS4 `+0.3657`，均 `effective`。SUA residual advantage 仅 `+0.0406`；view interaction 仍 `indeterminate`。见 §K.1–K.2 |
| **T4GATE** | T4 + 静态 electrode reliability gate | ⛔ **`ineffective`**。T4GATE−T4 `−0.0108 ± 0.0049 SE`；停止该方向。见 §K.3 |
| **N-MUA** | 原生 FALCON M1/M2 的 F0/T4/TS4 | ✅ **internal 18/18 + local held-out-calib test-only replay 18/18 完成**。held-out replay 中 M2 `T4−F0=+0.06979`、`T4−TS4=+0.06201`，均 3/3 cells 正；M1 两项略负。使用 chronological first-10/33、校准无反向传播；不是隐藏 EvalAI test。见 [`docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md`](docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md) |
| **SUA-REL0** | 同电极 source-separation Stage-0 | ⛔ **`ineffective`，停止**。完整 4 arms × 3 seeds：`REL−T4=−0.00144`（0/3 seed 正），`REL−REL-NG=+0.00650` 也排除 `+0.03`；不补 seed、不进入 relative amplitude。见 [`docs/SUA_AUXILIARY_STAGE0.md`](docs/SUA_AUXILIARY_STAGE0.md) |
| **T4-CFILM** | 拟合置信度 × pooled activity FiLM | 🧪 **代码与 CPU 合约完成；SUA 主线 strict gate 已通过，`T4@50` 三种子锚点正在排队运行。** `M_activity=30`、`M_T4=10/15/20/30/50`、共同从 trial 50 评估；五臂含 T4 continuation、C-shuffle、NoFiLM 参数匹配与 TS4。seed-42 只作 triage，三 seed 严格门通过后才能称 `effective`。 |
| **T4-B3T** | 流式 temporal basis × T4 | 🧪 **实现、成本审计、CPU tests 与 fail-closed runner/aggregator 已完成，等待 T4-CFILM triage 后接续。** 实测参数 `18,290→12,658`（−30.79%）、session MAC `13.03M→4.52M`（−65.29%）、support state 不变；将以 fresh T4/B3T+T4/B3T+TS4 判断 `−0.03` 非劣与 T4-content gate。 |
| **T4-NEXT** | T4 分量归因与网络融合优化 | 🧭 **可做，但需保留任务异质性**。优先 `[a,c] / m / b` 分解、T4×B3T 与低标签 learning curve；硬件流简化最后。不得复活已失败的静态 electrode gate 或同电极 relation。见 [`docs/T4_OPTIMIZATION_DIRECTIONS.md`](docs/T4_OPTIMIZATION_DIRECTIONS.md) |

### E1/E2/E4 最终结果

**完整数据、表格与判读见 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §J。**
本节此前记录的两版中期分析（基于 1 个和 2 个 seed）**均已被后续 seed 推翻**，
已删除以免与最终结论矛盾；被推翻的过程作为方法学教训保留在 §J.4。

摘要：

- **J.0 确定性验证通过** —— E4 的 B3 臂与 `side_feature_ablation_v2` 的 `f0`
  在三个 seed 上相差 `1.7e-7`–`3.4e-5`，两个独立 screen 结果一致。
- **J.1 E2**：不存在跨 seed 共享的收敛点；延长训练不改善平均表现且放大
  seed 分歧 1.7×。**epoch 预算不是杠杆**，沿用 `E=12, burn_in=4`。
- **J.2 E1**：SWA last20 增益 `+0.022`（3 seed 全正，约 2 SE），但
  **不降低 `σ_seed`**。SWA 只压轨迹内抖动，`σ_seed` 是轨迹间差异。
- **J.3 E4**：`B3T` 六 seed 全正 `+0.0418 ± 0.0090 SE`、参数 −31%、
  MAC −65%，但 session 异质；`B3A` 零信号 + 30× support state → **放弃**。
- **J.4 教训**：本会话三次小样本读数被增加一个 seed 推翻。在
  `σ_seed ≈ 0.023–0.047` 下，**n<3 的方向性判断不应写入文档**。
- **K.1/K.2 bridge**：T4 在 pseudo-MUA 上同样 `effective`；T4 后
  SUA−pseudo-MUA 只有 `+0.0406 ± 0.0153 SE`，不能表述为“很大提升”。
- **K.3 T4GATE**：静态 electrode-index scalar gate 已被两项配对共同判为
  `ineffective`，不再投入。
- **K.4 native FALCON MUA**：internal 18/18 训练完成；冻结 checkpoint 的
  local held-out-calib 18/18 replay 也完成。后者 M2 相对 F0/TS4 都明显且
  3/3 cells 一致，M1 两项略负，不能写成跨任务稳定有效或隐藏 benchmark 结果。
- **SUA relation Stage-0**：完整 12-arm artifact 复核后为 `ineffective`；
  REL 在 3/3 seed 上低于 T4，停止 relation 与 relative-amplitude 路线。
  SNR/波形稳定性继续只保留为 read-only negative diagnostics。

### E3/E4/bridge 代码与产物状态（2026-07-29）

- 编码器：`t4`/`t8`/`ts4`/`ts8` 特征组、`B3T`/`B3A` 变体，均已实现并测试；
- runner：`run_e3_tuning_ablation.sh`、`run_e4_encoder_variants.sh`——
  **`--max_epochs` 与 `--seeds` 均为必填、无默认值**，缺失即拒绝运行并指向
  [`docs/E3_E4_ENCODER_PROGRAM.md`](docs/E3_E4_ENCODER_PROGRAM.md) §0；
- 估计量：`eval_epoch_window_generic_dandi688.py` 接受显式 `--total_epochs`/
  `--burn_in`，窗口为 `burn_in+1..total_epochs`；在 `(12,4)` 时退化为冻结的
  `5..12`，与既有 screen 向后兼容；
- 聚合器：三态判定 + 实测 `σ_delta`（复用已修正的 `sigma_delta_standard_error`）；
  E4 聚合器额外把 `cost_profile()` 实算的 `support_state_bytes` / `mac_per_session`
  写进输出，使 R² 不会脱离部署代价被单独引用（`B3A` 的 state 是 30 倍）。
- pseudo-MUA：signal-view-aware loader、独立 cache namespace、专用 runner 与严格
  aggregator 已完成，`results/pseudomua_t4_bridge_v1/summary.json` 是权威聚合；
- T4GATE：9 个 artifact 已补聚合到 `results/t4_gate_screen/aggregate.json`，
  结论 `ineffective`；B3SEA/B3S+t4e 虽已实现，只保留为未运行候选。

E1 与 E2 可由**同一批长 run 同时回答**：跑 40 epoch、每 epoch 存 checkpoint，
则收敛曲线与事后 SWA（沿轨迹平均权重）都从这一批产出。

### E3 的一个重要前提

加入调谐特征会**改变方法的主张**：当前卖点是"identity 只从 spike 统计
得出"，加了调谐就变成"identity 来自一个**有监督的**标定块"。这仍然是
gradient-free、部署现实的（标定本来就是让受试者做已知方向的运动），但
假设更强，必须在结果文档中显式声明，不得含糊带过。

### 判据（沿用章程 §2 的功能-vs-解剖标准）

`E_i` 被加到 `Z` 上、喂给 cross-attention 决定每个 unit 的解码权重，因此
它需要编码**功能属性（tuning）**而非**解剖属性**。侧信息消融正是栽在这一点：
波形是 sorter 用来定义 unit 的东西，既是解剖的、又会跨日漂移。

## 历史优先级（2026-07-25）

按依赖顺序，前两项互为前置：

| # | 事项 | 类型 | 状态 |
|---|---|---|---|
| 0 | 决定 formal-test receipt 悬空问题如何处理 | **需用户决策** | 阻塞 |
| 1 | 修复 `learned_prior` 无标定公平对照 | 代码 | 见 [`docs/HANDOFF_SIDE_FEATURES.md`](docs/HANDOFF_SIDE_FEATURES.md) Task A |
| 2 | per-unit 侧信息消融（SNR/波形/electrode） | 代码 | 章程已固定，等 Task A |
| 3 | fixed-slot router 的 top-K / random / activity 对照 | 代码 | 章程已固定，未排期 |

结构扩展（B17+、更多 attention 变体）继续暂缓。

## P0：公平 SUA 基线

### 当前状态

- 训练脚本已支持 `--seed`，并保存 teacher SHA-256、训练配置和最佳 checkpoint 信息到 `run_metadata.json`。
- 比较脚本会拒绝 teacher 或核心超参数不一致的 checkpoint，并默认保存带 artifact SHA-256 的 JSON。
- B3/B15/B16 已全部使用 seed 42、相同 teacher、数据、loss、decoder、40-epoch 上限、early stopping 和 checkpoint 规则重训。
- 完整 internal validation 为 B15 `0.90977`、B16 `0.90848`、B3 `0.90781`；相对 B3 分别为 `+0.00196` 和 `+0.00068`。
- 严格比较保存在 `results/p0_s42_full_comparison.json`；机制复验保存在 `results/p0_s42_mechanism_diagnostics.json`。

### 工作

- [x] 使用当前 teacher：
  `sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt`
- [x] 通过 `train_variant_mc_maze.py` 严格重训 B3、B15、B16 seed 42。
- [x] 固定相同 teacher、datamodule、训练规则、loss、decoder 和 evaluation batches。
- [x] 保存带 checkpoint 路径、SHA-256、参数和协议的比较 JSON。
- [x] 分开记录 checkpoint validation R²、完整 loader R² 和 200-batch 机制消融 R²。

### 验收

- [x] B3-v2 checkpoint 和 JSON 结果存在。
- [x] 结果表中不存在跨 teacher 的直接排序。
- [x] JSON 能追溯 teacher/student checkpoint、数据路径、batch 上限和评估日期。
- [x] 三种 student 的 seed、teacher SHA、核心训练配置和 checkpoint 选择规则一致。

### 决策

- P0 严格 seed-42 证据为 B15 相对 B3 `+0.00196`、B16 相对 B3 `+0.00068`。
- 两个增益都很小，不通过稳定性决策门；B15 进入 capacity/LayerNorm 对照，三者都需补额外 seeds。

## P1：多 seed 稳定性

### 工作

- [x] 为训练脚本增加显式 `--seed`，并在 trainer/model/datamodule 初始化前固定随机性。
- 对 B3、B15、B16 运行至少 3 个完全相同的 seeds。
- 报告 task R²、identity normalized MSE、cosine、Pearson 的 mean/std 和逐 seed 值。

### 验收

- 每个变体至少 3 个有效 checkpoint。
- 同一 seed 下三种变体共享 teacher、split 和评估脚本版本。
- 结论基于重复方向，而不是单个最佳 checkpoint。

### 决策门

- B15 若仅单 seed 领先，则保留为探索结果，不进入部署主线。
- B16 若以较低硬件代价稳定接近或超过 B15，则优先作为部署候选。
- 若 B3 与 B15/B16 差异不稳定，则停止结构扩展，采用按信号类型重训 B3。

## M：测量修复（2026-07-25 晚新增，**阻塞其余一切实验**）

`attention_arch_screen_v3` 的逐 epoch 曲线诊断（[`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §H）
显示当前测量无法分辨被测效应。在修好之前，跑任何新实验都只会产出噪声。

- [x] **M1 唯一 run 目录**（2026-07-25 完成）。根因是 `configs/hydra/default.yaml`
      的 `run.dir` 只用秒级时间戳，同秒启动的两个进程解析到同一目录。现改为
      `${now:...-%f}_rid-${run_id}_f${loso_fold}_s${seed}`，并在 `src/train.py`
      启动时调用 `assert_run_dir_is_fresh()` 硬断言目录内无既有
      checkpoint/tfevents，否则 abort。`paths.output_dir` 即 hydra run dir，
      故守卫正对冲突点。10 个测试。
- [x] **M2 固定 epoch 预算**（2026-07-25 完成）。新增 `no_early_stopping` 开关
      （Hydra 侧与 `train_variant_dandi688.py --no_early_stopping`），
      从 callback 列表中过滤掉 `EarlyStopping`；选择记入 `run_metadata.json`。
- [x] **M3 确定性 checkpoint 规则**（2026-07-25 完成）。
      `--checkpoint_every_epoch` 每 epoch 存一份到 `epoch_ckpts/`；
      新脚本 `scripts/eval_epoch_window_dandi688.py` 硬编码 `E=12`、
      窗口 `epoch 5–12`、`first/n=30/pool=50`，复用
      `select_gradient_free_protocol_dandi688.py` 的协议指标实现（未重写），
      输出逐 epoch 逐 session R²、8-epoch 平均的 `variant_score` 及 SHA-256。
      注意 Lightning 的 `{epoch}` 是 0-indexed，协议 epoch 5–12 对应文件
      `epoch_004..epoch_011`；该转换集中在 `lightning_epoch_index()` 一处。
      21 个测试。
- [x] **M4 重设可达门槛** — 已写入
      [`docs/MEASUREMENT_PROTOCOL_V4.md`](docs/MEASUREMENT_PROTOCOL_V4.md)：
      估计量为 `E=12` epoch、取 epoch 5–12 的协议指标平均；先验
      `σ_delta ≈ 0.016`，故门槛设为 `+0.03`（与部署相关性给出的量级一致）。
      新增**三态判定** `effective / ineffective / indeterminate`——V3 把
      "不确定"记成了 `gate: false`，这正是我误读为"已否定"的直接原因。
- [x] **M5 用 V4 协议跑首个 screen**（2026-07-26 完成）。首个 screen 为
      `side_feature_ablation_v2`（15 runs）。判定 `F1/F2 = indeterminate`；
      实测 `σ_seed = 0.0385` 为主导方差源。结果见
      [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §I。
- [x] **M7 修聚合器的 `sigma_delta`**（2026-07-27 完成）：旧实现的
      `sqrt(σ_A²+σ_B²)` 不仅需要 `/√n`，还忽略同 seed 配对。当前共享实现以
      `stdev(per_seed_mean_delta)/sqrt(n)` 为主要估计量，并保留修正后的 unpaired
      quadrature 与隐含 seed correlation 作为诊断；E3/E4/T4GATE/pseudo-MUA
      aggregator 均已使用配对口径。

  **2026-07-25 顺序调整**：首个 V4 screen 改为**侧信息消融**
  （`side_feature_ablation_v2`，15 runs：`F0/F1/F2/FS1/FS2` × seeds `42/43/44`），
  而不是 attention 重跑。

  理由：两个设计共用同一个 B3 基线、同一套 V4 估计量、同样的总算力，
  因此先后顺序只决定**哪个问题先得到答案**。attention 是已被降级的问题
  （其阴性结论已撤回，状态为"未知且不紧迫"）；侧信息才是当前的活问题。
  首个 screen 同样会实测 `σ_delta` 并产出 B3 基线，两者都可被后续 attention
  臂复用。

- [ ] **M6 attention 臂（已排队，非当前优先）**：`B15P/B15D/B15` × seeds
      `42/43/44` = 9 runs，复用 M5 的 B3 臂作基线。v4 的 runner/aggregator
      代码已写好待用。MUA 臂再往后。

## P2：MUA 机制对照 — ~~已完成，阴性~~ **结论撤回，需重跑**

> **2026-07-25 晚撤回。** 此前记录的"attention 未通过参数匹配对照"结论
> 依赖不可靠的测量（§H）。attention 是否有效**当前未知**。下方保留原始
> 数字作为记录，但不构成机制结论。重跑见上方 M1–M5。

历史 `attention_arch_screen_v3` 在 SUA 与 MUA 两域同时运行 B3/B15P/B15D/B15
参数匹配对照，当时输出为“五个 `gate: false`”。这只是旧三态缺失 API 下的原始
记录，不等于五个机制均被判 `ineffective`。章程见
[`docs/ATTENTION_ARCHITECTURE_SCREEN.md`](docs/ATTENTION_ARCHITECTURE_SCREEN.md)，
数字见 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §E。

关键 paired delta：

- SUA：`B15 − B15P = +0.0064`，6 个 session 中仅 **2** 个为正（门槛需 ≥4）；
- MUA：`B15 − B3 = −0.0142`，**0/3** 为正。

~~旧判读：按 P2 判读表，这落在"两者均不稳定 / B15 ≈ B15P"一行，因此跨 neuron
self-attention 不是本项目的有效机制；B15 相对 B3 的增益约 84% 由 B15P
复现。~~ **该判读已撤回。** 当前只能说这些 v3 数字低于可分辨噪声底，attention
状态是未知。

### 当前决策

- attention 重测保持低优先级，但理由是 T4 主线效应更大、更可测，**不是**
  attention 已被否定。
- `advance_to_paired_pilot = false` 只属于撤回的 attention-v3 分支，不能再作为
  全项目 gate。后续 T4 驱动的 pseudo-MUA bridge 使用独立冻结问题与章程，已于
  2026-07-28 完成；真实 M1 external replication 仍未执行。
- v3 的原始数字只可作为测量失败案例，不可写成阴性机制结论。
- B16 的高阶统计路线未被本次筛选覆盖，仍停留在单 fold/seed 候选状态；
  在 P1 补齐前不作为部署候选。B16 优化思路见
  [`docs/B16_OPTIMIZATION_BRAINSTORM.md`](docs/B16_OPTIMIZATION_BRAINSTORM.md)。

### 历史记录

- B16 的 FALCON M2 fold 0 / seed 42 对照：held-out `0.248 ± 0.137`，B3 为
  `0.236 ± 0.102`，mean delta `+0.0112`，4/6 session 提升。单 fold/seed，
  未通过稳定性决策门。

## P3：SUA 泛化协议

### 当前问题

- MC_Maze 只有单 session 内部 trial validation。
- `val_dataloader()` 目前重复返回同一个 loader，`val_heldout` 名称具有误导性。
- NLB test 文件没有可直接使用的行为标签。
- 网络审计已确定主数据集：DANDI 000688 的 `sub-C`、`sub-M`、`sub-J` 共 99 个 manually sorted SUA sessions；`sub-T` 的 12 个 threshold-crossing sessions 必须排除。证据和备选集见 [`docs/MULTISESSION_SUA_DATASETS.md`](docs/MULTISESSION_SUA_DATASETS.md)。

### 可行性验证（2026-07-23 完成）

- sub-J 3 个 CO sessions 已下载（84 MB），NWB schema 已验证。详见 [`docs/DANDI_000688_FEASIBILITY.md`](docs/DANDI_000688_FEASIBILITY.md)。
- 数据为 event-level spike_times + per-spike waveforms + cursor_pos/vel/acc，非预分箱。
- Trial 过滤策略确定：只保留 `result == 'R'`（rewarded），所有 R trials 均 >2.3 s，无需额外 duration 阈值。sub-J 每 session 约 195–203 个可用 trials。
- Unit 数跨 session 可变（sub-J: 18–38），架构已支持。
- POYO 论文确认同一数据源的可解码性（single-session R²≈0.935 on CO），作为上界参考。

### 工作

- [x] 下载 sub-J，验证 NWB schema、trial 结构和行为数据。
- [x] 确定 trial 过滤策略（result='R' only）。
- [x] 编写 `MultiSessionDataModule`：multi-file registry、session-level chronological split、per-session variable-N calibration、cursor_vel 插值。
- [x] 在 sub-J 上完成 smoke test（forward pass + 梯度回传）。
- [x] 下载 sub-C（9.7 GB，53 CO sessions）。
- [x] 编写 `single_session_datamodule.py` + Step 0 单 session 上界（R²=0.694）。
- [x] Step 1 端到端跨 session（task_only，freeze_decoder=False）。
- [x] Step 2 冻结的 gradient-free streaming calibration 评估脚本；历史 few-trial finetune 仅保留为 diagnostic oracle 对照。
- [x] **发现关键混淆**：chronological split 把 unit 数 regime 跳变（~60 → ~250）切成了 train/test 边界。详见 [`docs/P3_CROSS_SESSION_ANALYSIS.md`](docs/P3_CROSS_SESSION_ANALYSIS.md) critical finding。
- [ ] **修正 split**：在旧 regime（39 sessions，38–91 units）内做 27/6/6 chronological split；Step 1 仅 train/validation 并锁定 checkpoint，Step 2 才运行唯一正式 held-out test。
- [ ] B3/B15/B16 为预先指定的 encoder-capacity 训练比较；所有结构选择和超参数调整只使用 validation sessions。
- [ ] 正式 test 前仅在 validation sessions 上 sweep 并锁定 gradient-free calibration protocol；唯一正式 test 只消费该 lock，test 后不改协议或重跑，失败只记录结论并转向新数据/独立 replication。
- [ ] checkpoint 锁定后只在冻结的 gradient-free 协议下评估一次 held-out test；test 只用于记录/解释，绝不驱动变体选择、架构搜索、重跑或权重更新。
- [ ] 将 `sub-M` 保留为 external-subject evaluation。
- DANDI 000121 作为独立来源的第二阶段 replication；原始 Zenodo MC_RTT 作为 MATLAB ingestion 备选。
- 明确 unit、trial、session 三种 holdout 的差异，并预先固定主指标。

### 验收

- 至少有一个不与训练共享相同 session/trial 集合的 SUA 泛化结果。
- 文档能够明确回答 holdout 的对象是什么。

## P4：共享训练策略

仅在 P0–P3 证明结构收益后开展：

- MUA/SUA 共享 backbone、独立小 head；
- signal-type 或 dataset context embedding；
- MUA pretrain → SUA fine-tune；
- SUA+MUA joint training；
- 与“分别重训两个小 encoder”的成本和准确率对照。

零样本权重共享不再作为默认目标；它已经在当前 B3 设置下失败。

## P5：部署接口（与 P3 并行）

固定 token 接口是 [`docs/ASIC_DEPLOYMENT_CHARTER.md`](docs/ASIC_DEPLOYMENT_CHARTER.md)
第 5 节识别出的真问题：实时代价由 per-token read-in 支配，而 token 数就是 `N`。

- [x] `fixed_slot_router_pilot_v1`：接口、缓存等价、压缩 gate 通过；
      精度 gate 未通过（`K=32` 0.1820 vs 阈值 0.3299）。
- [ ] 同 `K` 的 cached top-K scorer，配 random-K / activity-K control；
      规则已在 [`docs/FIXED_SLOT_ROUTER_PILOT.md`](docs/FIXED_SLOT_ROUTER_PILOT.md)
      "固定槽位与 token pruning 的取舍"一节预先写定。
- [ ] 若 pooling 与 pruning 都无法接近变长基线，则记录为 rate–distortion
      结论，并把固定 shape 的代价直接写进硬件章程，而不是继续调 router。

## 暂缓工作

- 更大型的 self-supervised foundation model；
- 在公平基线前继续添加 B17+ 结构；
- **继续投入无新机制假设的 attention 变体**（旧阴性结论已撤回，但 T4 效应远大于
  当前 attention 可分辨效应，应先推进 T4 主线）；
- T4 上的静态 electrode anchor/embed（gate 已 `ineffective`；除非分量/置信度
  诊断提出新机制理由，否则不运行已实现的 C/A）；
- 真实 threshold-crossing MUA external replication（pseudo-MUA bridge 已完成，
  外部 scope 与数据协议需另行冻结）；
- 仅依据 internal validation 进行 ASIC 定型；
- 用 MUA LOSO 数字与 SUA internal-validation 数字直接排名。

## 治理事项（需处理，非实验）

### G1：formal-test receipt 悬空

`results/p3_formal_test_816cdd8bf9f26abd1a3e6251e5fbf8537eb6c6cb4de1e8f3312980ddbf478379_receipt.json`
状态为 `"started"`（2026-07-24T22:52），protocol lock 指向 B16，**没有任何
对应的结果文件**。按预注册规则该 scope 已被占用且不可重跑，等于
`sub-C/CO/27-6-6` 唯一一次正式 held-out test 名额已消耗但未留下结论。

可选处理（需用户决策，不可由实施者自行选择）：

| 方案 | 含义 | 代价 |
|---|---|---|
| A 作废并记录 | 标注 receipt 为 aborted，永久放弃该 scope 的 formal test | 该 split 只剩 development evidence |
| B 换 scope | 以 `sub-M` 或 `sub-J` 建立新的 formal-test scope | 需新 split 与新 protocol lock |
| C 一次性豁免 | 明文记录豁免理由后重跑一次 | 削弱预注册制度的可信度 |

在 G1 解决前，任何实验都只能产出 validation development evidence。

### G2：挂起 run 的诊断

`b16_dandi688_co_learnedprior_s42` 占用 GPU 约 18h42m 且无产出，
原因未查（详见 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §F.3）。
在重跑 learned-prior 基线前必须先定位，否则会重复浪费。

## 推荐的下一步

1. 冻结并执行 T4 分量消融、T4×B3T 与低标签 learning curve；分别报告
   SUA、M1、M2，不把 M2 的强结果外推到 M1。
2. 停止同电极 relation、relative amplitude、静态 SNR/waveform gate 与
   electrode lookup；现有阴性 artifact 足以关闭这些路线。
3. 若 calibration confidence 在低标签预算下通过 gate，再做 zero-init low-rank
   FiLM；随后才测试 `K(E,T4),V(x)` 的 decoupled cross-attention。后者是
   decoder-changing 实验，必须与 coupled decoder 公平训练/蒸馏，并报告缓存 key
   后的实际在线 MAC，不能写成冻结 decoder 的 encoder 改进。
4. T4 硬件流简化保持最低优先级，只在分量和低标签结果明确后推进。
5. 处理 G1、G2；随后再考虑 fixed-slot router 的 top-K/random/activity 对照。
