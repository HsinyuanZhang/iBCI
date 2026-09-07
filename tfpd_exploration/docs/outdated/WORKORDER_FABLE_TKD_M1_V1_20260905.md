# WORKORDER_FABLE_TKD_M1_V1 — TKD M1 线（M2 降级承接）

Date: 2026-09-05
Type: 预注册工单
Parent: `WORKORDER_FABLE_TKD_M2_V1`（含 ADDENDUM-1..6）；`RESULT_FABLE_TKD_M2_V1_20260905.md`（降级依据）
Experiment id: `fable_tkd_m1_v1`
Result root: `tfpd_exploration/results/fable_tkd_m1_v1/`
Status: FROZEN_FOR_IMPLEMENTATION

---

## 0. 主线与定位（不变）

低成本（数量级 MAC 优势）+ 零梯度校准（session 相关量全闭式）。M1 的 decoded 输出 = 16 维 EMG，identity = rSyn3 载体 [64,4]（3 个 NMF synergy 权重 + 截距，`m1_emg_syn3_fcm_v1/syn3.py`）。

**M2 教训预装为 Day-1 默认**（不再作为变量）：GRU 时间模型（SSM 已证伪）；蒸馏 λ_E=1.0（教师保真主导）；值域 = 池化常数 σ、μ 恢复截距（D12 律）；目标空间/尺度律严格镜像 M1 champion 训练惯例（实现时读 `m1_b3_allsource` resolved config 确认缩放与损失面）；固定末 epoch（fold-local 惯例 `FIXED_LAST_EPOCH_INDEX`，无选择律问题）；输出尺度校准 init（D13b 律）。

## 1. 面

- **主面（pilot）**：fold-local LOSO fold 0（target ses-20120924；源 = 其余 3 个 held-in session）。评分 `variance_weighted_last_bin_r2`、`static_m10_repeated_calibration`、26517 窗——逐字镜像 `m1_emg_rsyn3_fold_local_v1` 的面。REF：Z-Fix **0.6374** / S-Fix 0.6203 / S-Acyc 0.6218（`results/m1_emg_rsyn3_fold_local_v1/pilot_r3/score_static.json` governing_r2）。
- identity：rSyn3 载体，fold-local bank（digest 锚 `FOLD0_M10_RAW_CARRIER_DIGEST`）；ρ = 每 unit ridge 拟合 R²（闭式，同 M2 律）。
- **Surface B（later-day 3 session）**：只在 §4 双 kill-gate 全过后开。

## 2. 架构实例化

TKD（复用 `fable_tkd_m2_v1/model.py` 的类，参数化）：N=64、W=100、B=16（readout Linear(128→16)）、d_v=d_k=64、L=8、GRU(128)。key = Φ_k(t)（rSyn3 4 列，源律归一化）；ε 内容项保留；SHUF = 载体行置换（seed 42，训练+部署同置换）；POOL = t 广播均值。

**A1-M1 锚（生成式 NMF 读入）**：rSyn3 的结构本身给出闭式生成解码器——`ŝ(t) = (WᵀW+λI)⁻¹Wᵀ(r(t)−b)`（W=[64,3] synergy 权重，λ 用 carrier 的 RIDGE_LAMBDA 律），`ŷ(t) = D@ŝ(t)`（D = 冻结 NMF 字典 [16,3]）。这是"identity 即读入权重"论题的最纯形式（读入 = W 伪逆）。断言：TKD 的 **read-in 阶段**（attention+merge，identity 时间模型 γ=0 直通）在 init 时与该生成式读入的复合映射相关 ≥ 0.99（批量窗上；构造法自定，仿射不变；[0.90,0.99) 披露带，<0.90 阻断）。训练模型仍为 GRU（随机 init）——锚证明的是 read-in 起点，B-M1 臂（冻结 read-in 只训 GRU+readout）据此定义。

## 3. 臂

| 臂 | 定义 | 说明 |
|---|---|---|
| REF | Z-Fix 0.6374（+S-Fix/S-Acyc context） | 不重跑 |
| **A′** | TKD-GRU + 蒸馏 λ_E=1.0 + 任务损失 | 主臂 |
| SHUF | A′，载体行置换 | identity 内容 |
| POOL | A′，t 广播均值 | 无 identity 下界 |
| B | A′ read-in 冻结于生成式锚 init，只训 GRU+readout | 生成式锚之上的学习增量 |

蒸馏教师 = M1 champion 管线（b3s_rsyn3_freeze 家族；教师输出缓存 digest 钉死，只覆盖源 session 训练窗）。调度：镜像 fold-local 惯例（12 epochs、lr/optimizer 读源 config；不引入 M2 的 3e-4——M1 用自己的已验证惯例）。

## 4. Kill-gate（预注册）

pilot = A′ seed42 单跑 + SHUF seed42 单跑（fold-local 面）：
- **K1（水平门）**：A′ LOSO R² ≥ **0.5374**（Z-Fix − 0.10）。
- **K2（可测性门）**：A′ − SHUF ≥ **+0.02**（identity 内容对小 decoder 键控有可测信号）。
- K1∧K2 → 全格（A′/SHUF/POOL/B × seeds 42/43/44）+ Surface B。
- K1 过、K2 不过 → TKD 作为 M1 小 decoder 成立但 identity 键控不可测：停，向用户报告（成本叙事可独立成文）。
- K1 不过 → 小 decoder 线在两个快数据集上关闭：停，向用户报告并给出 688 建议（唯一已知 T4 大增量面）。

## 5. 纪律

同 M2 工单（GPU1 only + UUID + foreign-pid preflight；receipt 原子写 0444+sidecar；attempt 先行；fail-closed；密封物只读；无 EvalAI 提交）。资源：pilot ~2×1h GPU1；全格 ~6–9h。

## 6. 证据路径

- M1 face/惯例：`tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/`（plan.py 常量、carrier_bank.py、pilot_r3 receipts）、`tfpd_exploration/src/m1_emg_syn3_fcm_v1/{plan,syn3,data}.py`
- M1 champion：`streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3s_rsyn3_freeze.yaml`、teacher ckpt（sha c81a2bbd…，plan 钉死）
- M2 复用资产：`tfpd_exploration/src/fable_tkd_m2_v1/`（模型类、receipt 律、teacher-cache 管线模式）

## ADDENDUM-1（2026-09-05，A1-M1 锚结构性失败后的裁定）

1. **M1-D5 追认**：数值容差律（1e-9 vs 密封数值；漂移量化 3e-16 rel；双 digest 记录）在库末位漂移下为正确处置。记录在案。
2. **廉价 CPU 诊断先行**（在任何 fork 承诺前）：闭式生成解码器本体在 fold-local 面上评测——逐 session ŷ_gen(t) = diag(scale)·D·(W_sᵀW_s/n+λI)⁻¹W_sᵀ(r(t)−b_s)（ρ=1 与 ρ 加权两版），variance_weighted_last_bin_r2 于 (a) fold-0 target 全 26517 窗（LOSO 读数）与 (b) 源 session（within 读数）。receipt `stage0/gen_decoder.json`。
3. **预注册 fork 规则**：target 生成式 R² ≥ 0.35 → 实施 fork (a) 并跑双臂 pilot（K1/K2 不变）；< 0.35 → M1 以负结果结案（共享仿射读入的锚结构性不可达 + 生成式基底过弱），报告并停止。
4. **Fork (a) 规格 —— TKD-M1b "生成式读入头"**（仅当规则通过）：y(t) = M_s·v_t + y_res(t)；M_s = diag(scale)·D·(W_sᵀW_s/n+λI)⁻¹·diag(ρ/σ 池化律) ∈ R^{16×64} 为校准期由载体闭式计算的逐 session 读入矩阵（零梯度，~1k MAC/bin 增量）；v_t = 现行值律输出；y_res = readout(GRU(merge(attention(v, keys))))，**readout 零初始化**使 init ≡ 生成式解码器（锚按构造逐位成立——A1-M1 锚据此重立规格，planner 附注记录）。损失 = MSE(y, target) + λ_E·MSE(y, teacher) 照建。SHUF：载体行置换 → W_s 行置换 → M_s 列一致置换（identity 控制保持）。POOL：广播均值载体 → 池化 M_s。B 臂：只训残差路径的 GRU+readout（attention 冻结于 init）。锚断言：model(init) 与 ŷ_gen 复合逐位相等（或 ≤1e-6）；随后双臂 pilot A′+SHUF seed 42，fold-local 面，K1 ≥ 0.5374 / K2 ≥ +0.02 照预注册。grid 仍仅凭 planner 指令。
