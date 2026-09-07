# WORK ORDER — B1-SFCJ V1 Build (CPU-only, pre-GPU)

**日期：** 2026-09-03
**Planner：** fable（第三轮审核人）
**Builder：** grok
**Governing design：** `tfpd_exploration/docs/DESIGN_B1_SPECTRAL_FUNCTIONAL_CARRIER_JR1_V1_20260903.md`（第三轮审核后，含本 work order §1 的 6 处修订）
**授权范围：** 设计文档修订、Stage 0A、Stage 0B、全部代码模块、CPU tests、CPU-only smoke。
**硬停止：** 不得启动任何 GPU 训练（Stage 1/2/4）、不得 docker build、不得 EvalAI 提交、不得读取或推断 hidden query label。Stage 1 之前交回第三方复核。

---

## 0. 环境与边界

- Python：`/home/xinyuan/miniconda3/envs/spint/bin/python`（torch 2.12 cu130、pynwb 4.0、sklearn 1.5.2、`falcon_challenge==1.0.2`）。
- 所有 torch 代码必须能在 `CUDA_VISIBLE_DEVICES=""` 下运行；tests 与 smoke 一律 CPU。GPU0 有其他 job 在跑，不要碰任何 GPU。
- 数据只读：`SPINT-main/data/001046/**/*.nwb`（9 个文件）。禁止写入 `SPINT-main/`。
- Owned paths（新建，additive）：
  - `tfpd_exploration/src/b1_sfcj_v1/`（包）
  - `tfpd_exploration/tests/test_b1_sfcj_*.py`
  - `tfpd_exploration/results/b1_sfcj_v1/`（receipts；JSON，sort_keys，含输入 SHA）
  - 设计文档本身（仅 §1 列出的修订）。
- 不修改 `SPINT-main/src`、`third_party/falcon_challenge`、任何 M1/M2/H1 路线文件。

---

## 1. 设计文档必须先落地的修订（第三轮审核结论）

按原文精确章节修改，不改其他内容：

1. **§6.5.1 TPL 模板族。** 每个 TPL 同时预注册三种 neural-free 聚合：raw 逐坐标算术均值、log 空间均值（几何均值）、逐坐标中位数。所有 TPL 门（Stage 1 gate、§10.5、Outcome A/E）使用三者中 **MSE 最低者**（`TPL-*-BEST`），三者数值全部报告。依据（只读复算，official metric，700 帧）：TPL-M3 raw-mean/median = 0.000390/0.000350、0.000553/0.000384、0.000646/0.000511；TPL-SRC raw-mean/median = 0.001155/0.000778、0.000653/0.000521、0.000621/0.000383。算术均值是弱化 15–40% 的稻草人。
2. **§6.5 新增 `A0-RT`（A0 + M3 residual template）。** 在 standardized-log 空间对 M3 calibration 拟合逐坐标残差模板 `c(t,f) = median_m[y_m(t,f) − p_m(t,f)]`（同时报 mean 版本），预测 `p(t)+c(t)`，无 λ、无反传；A0 对 calib trial 的预测使用 §6.5.2 同一 self-contained M3 pool 约定，并按 §6.5.2 同一部署律应用到 query stream。加入 §8.3 Tier 2、§10.5 强基线清单（与 A0-OR158、DR-158-ML 同等地位）、Outcome A/E 的 `min[...]`。
3. **§10.3 选择顺序逻辑补丁。** 维度规则只能在**已通过 §10.2 双律内容门的 `(topology,q)` 之间**选择；若某 topology 只有一个 q 过门，该 q 直接冻结，维度差仅作描述性报告；若两个都未过，该 topology 退出产品候选。
4. **§9 Stage 4 / §10.7 官方基线部署律。** A0-NATIVE 作为 trained arm 同样按 §10.6 独立选择 `L_A0`；official package 中 A0 使用 `L_A0`、方法使用 `L_method`（best-vs-best），并在 §10.7 明示 `official_gain` 是 best-vs-best，而本地内容门是 same-L；两者都报告。
5. **§9 Stage 1。** 明示 H* 选择与 Stage 1 gate 使用 governing `GROWING` 律的 in-range coordinates，同时报告 FIXED3。
6. **§6.5.3/§6.5.4 与 §9 Stage 0B 步 3。** DR-158-ML/SL、DR-PC8 的 neural 输入明确为与 SFC 相同的 5 ms box 平滑 1 ms rate view `r_i`；SFC lag grid 与 DR lag grid 统一为 `{0,10,20,30,40,50,60} ms`。

修订后在文档头部“审核状态”追加“第三轮 6 项修订已落地（work order V1）”。

---

## 2. Stage 0A — Data & evaluator contract（CPU）

产出 `results/b1_sfcj_v1/stage0a_inventory.json` + `stage0a_receipt.md`，逐文件记录设计 §9 Stage 0A 列出的全部字段。必须包含并断言：

- 9 文件 path/size/SHA-256；split/session/trial 数 = 11/29/8、2/2/2、3/3/3。
- tx `[27000·n, 85]`、dtype、值集 `{0,1}`；timestamps 步长 `1/30000`，每 trial 恰 27000 采样，`sample i → bin i//30` 位置律：每个 1 ms bin 恰含 30 个采样（用 timestamps 证明），spike 总数守恒。
- spectrogram `[158,880]`/trial，`spectrogram_times[0]=0.01024`，步长 1 ms；mask 恰为 `[90,790)`，158 频点共享；raw 最小值恰为 `1.0`（登记为已含正偏移；正式变换 = `log(raw)`）。
- evaluator `trial_change` 索引与 timestamp 推导的 trial 末样本一致（`26999+27000k`）。
- calib 与 minival 各自 timestamp 从 0 重启，登记“跨文件 chronology 不可恢复”，固定约定 `calib 4..N → minival 1..2`。
- `hash_dataset` 本地返回 undotted `20210626`，`DATASET_HELDINOUT_MAP['b1']` 为 dotted；登记安装版 `evaluate()` 已知 mismatch。
- installed `falcon_challenge/{config,dataloaders,evaluator,interface}.py` SHA 与设计 §18 一致。
- 规则快照：保存 FALCON README（raw.githubusercontent main）全文 + SHA + 抓取时间；EvalAI overview 页若不可抓取，登记不可达而非伪造。

门：任一断言失败 → 停止并报告，不在 loader 内修复。

---

## 3. Stage 0B — SFC constructibility + 闭式基线（CPU）

三折 date-LODO（§8.1）。对每 fold 产出 `results/b1_sfcj_v1/stage0b/fold{k}/…` receipts，并汇总 `stage0b_summary.json`。

1. **Acoustic basis（§6.2）：** 仅 train dates、valid frames：`log(raw)` → per-frequency mean/std → 确定性 full-SVD PCA8；sign 固定；`B_3 = B_8[:3]`。保存 mean/std/basis/sign SHA。
2. **Rate view：** 30 kHz → 1 ms counts（位置律）→ 5 ms centered box → `r_i(t)`。frame k ↔ bin `floor(10.24 + k)`（显式对齐表写入 receipt）。
3. **SFC lag（§6.1 修订版）：** grid `{0,…,60}`；每个 training date 用 first M3 拟合 date-specific SFC9（ridge λ=1，intercept 不惩罚），在同 date calib 4..N 上算 encoding score；两 training date 等权聚合选 fold lag；validation date 不参与。
4. **SFC4 / SFC9：** 在 `z_3` / `z_8` 上**独立**拟合 target-date（validation date）first M3；coefficient normalizer 只由 training dates 的 M3 系数拟合；pad 到 9 维；Zero9 literal。报告 rank（4/9）、condition number（≤1e4）、coefficient norm、split-half（first M3 vs next M3，仅 training dates）、per-channel cosine、held-calibration encoding R²、label-derangement control。
5. **TPL-SRC / TPL-M3（§6.5.1 修订版）：** 三种聚合各一版，raw 空间，在 validation date 全部 10/28/7 query 上按 official metric 评分；成员 SHA、聚合顺序、prediction SHA 入 receipt。
6. **DR-158-ML / DR-158-SL / DR-PC8（§6.5.3–6.5.5）：** 595 维（7 lag × 85）`r` 输入；输出 standardized-log 158 维（DR-PC8：z8 → 逆 PCA）；λ（SL 同时 lag）由 target-date M3 内三折 LOO、完整 inverse-to-raw official metric 选择，grid `{1e-4..1e4}`，tie → 更大 λ、更小 lag；随后全 M3 refit 并冻结。在 10/28/7 query 上评分，报告 in-range `10/8/7` 与完整流两套 equal-date 数字。
7. **Official metric 独立复现：** 自实现 `b1_official_metric(pred[T,158], tgt[T,158], mask)` 与 `FalconEvaluator.compute_metrics_spectrogram_distance` 对同一合成/真实输入逐位一致；常数预测 fail-closed。
8. **Direct159：** 不在 0B 执行（已移至 Stage 3）。

门：设计 §9 Stage 0B 冻结门全部字段写入 receipt 并给出 pass/fail；不通过不阻塞代码模块交付，但阻塞 Stage 1。

---

## 4. 代码模块（`tfpd_exploration/src/b1_sfcj_v1/`）

| 模块 | 内容 |
|---|---|
| `data.py` | NWB 读取、位置律 binning、rate view、valid mask、LODO manifest（fold/train/val/query 顺序、K_train_max=28/10/28、in-range mask 10/8/7）、`calib→minival` 约定 |
| `metric.py` | official metric 复现、raw/log 空间转换、fail-closed 检查 |
| `acoustic_basis.py` | log/standardize/PCA8/sign/SHA |
| `sfc.py` | SFC4/SFC9/Direct159 闭式拟合、normalizer、Zero9/pad、lag 选择 |
| `baselines.py` | TPL 三聚合、A0-OR158、A0-RT、DR-158-ML/SL、DR-PC8、M3-LOO λ authority |
| `model.py` | B1-SPINT substrate：`[B,900,85]`→unit tokens、identity `pre_pool/post_pool`（common 9-D carrier 接口，carrier columns 零初始化）、J-R1 融合（`alpha` IEEE +0，`tanh`）、158 frequency queries、1 层 cross-attn（D=512, heads=64）、`Linear(D,880)` head、standardized-log 输出 + 逆变换；六臂由同一类 + 配置构成，共享 native substate |
| `memory.py` | growing memory：训练 whole-stack O(K) 重算（带梯度）；冻结部署 running-sum O(1)；`FIXED3`；pool digest；decode-before-commit 状态机；reset |
| `schedule.py` | 预生成只读 schedule（trial order、pool manifest、dropout mask、seed 42/43） |
| `decoder.py` | `BCIDecoder` 子类：`predict([1,85,27000]) → [158,880]`、dotted/undotted tag resolver、六份 M3 payload 加载、reset 归零 |
| `gates.py` | §10 全部估计量：in-range/OOD 分账、双律内容门、维度规则（修订 §10.3）、`L_arm` 规则、tie rule、Tier-2 实用门 |

训练循环（`train.py`）只需实现并通过 CPU dry-run（tiny D、2 steps），**不得**在 GPU 上运行。

---

## 5. CPU tests（`tfpd_exploration/tests/test_b1_sfcj_*.py`）

逐条实现设计 §12.2 清单（含第三轮新增：TPL 三聚合可复现、A0-RT 无 λ、§10.3 修订逻辑、A0 `L_A0` 独立选择）。额外：

- `predict` 3-D 返回必须被 decoder 自身拒绝；
- evaluator 端到端 shape 组装：用 `FalconEvaluator.predict_files` 的 B1 分支逻辑（可 monkeypatch `load_nwb` 用真实 minival 文件）验证 `[158,880]` 组装后 `prd.shape == msk.shape`；
- first-step 六臂 CPU float64 位级相等；carrier columns/alpha 首批梯度可达；
- 训练 O(K) whole-stack 与冻结 O(1) running-sum 在 float64 位级一致；
- fold-1 in-range 8 / OOD 20 掩码正确。

全部 tests 在 `CUDA_VISIBLE_DEVICES=""` 下通过，运行时间 < 10 min。

---

## 6. 交付与停止

交付物：

1. 修订后的设计文档（§1 六项）；
2. `results/b1_sfcj_v1/stage0a_*`、`stage0b/**`、`stage0b_summary.json`；
3. 代码包 + tests，`pytest tfpd_exploration/tests/test_b1_sfcj_*.py` 通过记录；
4. `results/b1_sfcj_v1/BUILD_REPORT.md`：Stage 0A/0B 门 pass/fail、TPL/DR 基线三日数字、已知偏差、未决问题。

**到此停止。** 不进入 Stage 1（A0 GPU horizon）。Stage 1 启动需要第三方复核本 BUILD_REPORT 后另开 work order。
