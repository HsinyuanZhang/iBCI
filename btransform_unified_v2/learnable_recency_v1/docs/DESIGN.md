# 可学习 recency bias（CABLE / FoX 本质）

独立子项目。不修改 `btransform_unified_v2/src`、既有 runner、或任何 `results/` 正式产物。

## 统一公式

对每个 head \(h\)、query \(t\)、key \(s\)：

\[
\mathrm{bias}_h(t,s)=-\,g_h(t)\,(C_h(t)-C_h(s)),\quad
C_h(t)=\sum_{l\le t}a_h(l),\quad a_h(l)\ge 0
\]

\(C(t)-C(s)=\sum_{l=s+1}^{t}a(l)\)。窗口内用反向 cumsum，缓存长度 \(\le 75\)，无无界前缀和。

固定 recency 是特例：\(a_h\equiv s0_h=\ln 2\cdot 0.02/\mathrm{half\_life}_h\)（两只 flat head \(s0=0\)），\(g\equiv 1\)，于是 \(\mathrm{bias}=-s0\cdot\mathrm{age}\)。

**增量输入**：当前层 pre-LN residual `h`（进入 `norm1` 之前）。不使用 norm 后的向量。

## 四档

| 档 | \(a_h(l)\) | \(g_h(t)\) | 初始化 | 默认新参（D4, 8H, width 256, per_layer） |
|---|---|---|---|---|
| `learned_slope` | \(s0_h\exp(\theta_{\ell h})\)，`slope_log` 形状 `[L,H]=[4,8]`，flat 列被 mask | \(1\) | \(\theta=0\Rightarrow a\equiv s0\) | **24**（4 层 × 6 只活跃 head） |
| `fox_gate` | \(-\log\sigma(w\cdot x+b)=\mathrm{softplus}(-(w\cdot x+b))\) | \(1\) | \(w=0\)，\(b=\mathrm{logit}(e^{-s0})\) | **8224** |
| `cable` | \(\mathrm{softplus}(\mathrm{MLP}(x_l))\)，隐层 16 | query gate；`--cable-nw` 时 \(g\equiv 1\) | 权重 0，bias 对齐 \(s0\)/1 | **25216** |
| `fixed` | 常量任务阶梯写入普通 `RiftTemporal.recency_slopes` | \(1\) | 无新参 | **0** |

`learn_flat_heads` 默认关。`per_layer` 默认 **True**（`--per-layer` / `--no-per-layer`）。`fixed` 是对照：只换阶梯、不学 slope，用来把「阶梯变了」和「斜率被学」分开。H1 阶梯与 `DEFAULT_HALF_LIVES` 相同，不需要 `fixed` 对照。

## 任务阶梯（锚在 H1）

`half_life_task = DEFAULT_HALF_LIVES × (W0 / 75)`，`W0` 是 `RiftTemporalConfig.for_context(context_bins, layers)` 的第一层窗口。flat head 保持 `None`。精确分数，不四舍五入。函数：`scaled_half_lives(context_bins, layers, bin_seconds)`。`--ladder default` 回到未缩放的 `DEFAULT_HALF_LIVES`；`--half-lives` 逗号列表可覆盖（允许 `none`）。

默认 D4：

| 任务 | context | W0 | 因子 | 半衰期（秒） | 半衰期（bin，÷0.02） | windows |
|---|---|---|---|---|---|---|
| H1 | 300 | 75 | 1 | 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, None, None | 4, 8, 16, 32, 64, 128, None, None | 75,75,75,74 |
| M1 | 100 | 25 | 1/3 | 0.08/3, 0.16/3, 0.32/3, 0.64/3, 1.28/3, 2.56/3, None, None | 4/3, 8/3, 16/3, 32/3, 64/3, 128/3, None, None | 25,25,25,24 |
| M2 | 50 | 13 | 13/75 | 0.08×13/75, …, 2.56×13/75, None, None | 4×13/75, …, 128×13/75, None, None | 13,12,12,12 |

H1 阶梯不变，既有 H1 固定 recency 正式跑仍是合法对照。

## `--layers`（默认 4）

传入 `RiftTemporalConfig.for_context(..., layers=)`。KV 窗口重分配，例如 H1 D3 → `(100,99,99)`，M2 D3 → `(16,16,16)`。缩放阶梯用所选深度的 W0。配对初始化对照是 **同样 layers** 的 seed-42 固定 recency。正式命令 **不传** `--layers`。

包装器在 decoder `__init__` 之后按深度/阶梯重建 `RiftTemporal`（同构则偷走已按 `seed+0x52494654` 初始化的 `blocks`，只换 slope buffer）。

## 导出与打分路径

`export_constant_slopes()`：`per_layer` 时必须给出 `[L,H]` 表。单向量写入未改动的 `CpuRiftTemporalRuntime` **仅当所有层一致**（`export_shared_slope_vector()`）。默认 `per_layer=True` 训练后层间会分开，**score-time 走 `CpuLearnableRecencyRuntime` / 可学习模块本身**，不走单向量导出。

## Init-parity

1. `from_initialized` 偷走已初始化的 `blocks`，不调用 Linear 默认 init。
2. 新参常量注册，不消耗 torch RNG。
3. 共享 `named_parameters` 与同样 layers 的 seed-42 固定 recency 逐字节相等。`fixed` / 缩放阶梯下 **只有** `temporal.recency_slopes` buffer 与库存 DEFAULT 阶梯不同。
4. 前向与 **同一阶梯** 的固定 recency `allclose(atol=1e-6)`。

## 数据集默认表

| 任务 | layers | per_layer | learn_flat_heads | 阶梯 | 默认档 | 新参 WD |
|---|---|---|---|---|---|---|
| M2 | 4 | True | False | 缩放（13/75） | `--tier` | 0 |
| M1 | 4 | True | False | 缩放（1/3） | `--tier` | 0 |
| H1 | 4 | True | False | 未变（1） | `--tier` | 0 |

配方：seed 42，AdamW，warmup-cosine，EMA 0.9995，unit dropout 0.1。新参单独 group，`wd=0`。

## 正式跑决策规则（单 seed）

相对固定 recency 参考（M2 e9 = 0.3900576650553505 / worst 0.24492；H1 e16 = 0.46765143362182887；M1 e3 = 0.5690750181674957；H1 flat e15 = 0.4685904900076509 作对照）：

- **非劣**：equal-session 均值不低于参考 −0.01，**且** worst session 不低于参考 worst −0.02。
- **改进**：均值 ≥ 参考 + 0.02。
- 单 seed：不得把一次涨跌写成方法结论。
- M2 / M1 必须同时看 `learned_slope` 与 `fixed`：若 `fixed` 已涨、`learned_slope` 不再涨，则是阶梯效应不是学习效应。

## Wave 1 建议

先跑 **M2 `learned_slope` + `fixed`**（约 29 min / arm）。确认阶梯对照与学习档都活着，再铺 fox/cable 和 M1/H1。

## GPU 启动命令

Python：`env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python`。

结果目录：`btransform_unified_v2/learnable_recency_v1/results/<task>_<tier>_s42/`。目录非空则拒绝覆盖。正式命令不传 `--layers`。

```bash
PY='env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python'
LR='btransform_unified_v2/learnable_recency_v1'

# Wave 1 — M2 learned_slope + fixed control (~29 min / arm)
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_train.py --tier learned_slope --device cuda:0 \
  --dest $LR/results/m2_learned_slope_s42
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_train.py --tier fixed --device cuda:0 \
  --dest $LR/results/m2_fixed_s42

# M2 fox / cable
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_train.py --tier fox_gate --device cuda:0 \
  --dest $LR/results/m2_fox_gate_s42
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_train.py --tier cable --device cuda:0 \
  --dest $LR/results/m2_cable_s42

# H1（无 fixed：阶梯未变）
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/h1_learnable_train.py --tier learned_slope --device cuda:0 \
  --dest $LR/results/h1_learned_slope_s42
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/h1_learnable_train.py --tier fox_gate --device cuda:0 \
  --dest $LR/results/h1_fox_gate_s42
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/h1_learnable_train.py --tier cable --device cuda:0 \
  --dest $LR/results/h1_cable_s42

# M1 FULL + fixed control
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --tier learned_slope --device cuda:0 \
  --dest $LR/results/m1_learned_slope_s42
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --tier fixed --device cuda:0 \
  --dest $LR/results/m1_fixed_s42
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --tier fox_gate --device cuda:0 \
  --dest $LR/results/m1_fox_gate_s42
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --tier cable --device cuda:0 \
  --dest $LR/results/m1_cable_s42
```

## 评分 / 选择命令

```bash
# M2 EXT6
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_score.py \
  --run-dir $LR/results/m2_learned_slope_s42 \
  --dest $LR/results/selection_m2_learned_slope_ext6_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_score.py \
  --run-dir $LR/results/m2_fixed_s42 \
  --dest $LR/results/selection_m2_fixed_ext6_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_score.py \
  --run-dir $LR/results/m2_fox_gate_s42 \
  --dest $LR/results/selection_m2_fox_gate_ext6_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/m2_learnable_score.py \
  --run-dir $LR/results/m2_cable_s42 \
  --dest $LR/results/selection_m2_cable_ext6_s42 --device cuda:0

# H1 HO-M3
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/h1_learnable_train.py --stage score --tier learned_slope \
  --dest $LR/results/h1_learned_slope_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/h1_learnable_train.py --stage score --tier fox_gate \
  --dest $LR/results/h1_fox_gate_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=0 $PY $LR/scripts/h1_learnable_train.py --stage score --tier cable \
  --dest $LR/results/h1_cable_s42 --device cuda:0

# M1 HO3
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --stage score --tier learned_slope \
  --dest $LR/results/m1_learned_slope_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --stage score --tier fixed \
  --dest $LR/results/m1_fixed_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --stage score --tier fox_gate \
  --dest $LR/results/m1_fox_gate_s42 --device cuda:0
CUDA_VISIBLE_DEVICES=1 $PY $LR/scripts/m1_full_learnable_train.py --stage score --tier cable \
  --dest $LR/results/m1_cable_s42 --device cuda:0
```

Smoke（仅 CPU，`--max-updates-smoke 1`）写在 `$LR/results/smoke/*_v2/`。

## ADDENDUM 2026-09-10 · proj_add P16 统一入口（用户指令）

用户于 2026-09-10 决定：M2 首个实验（concat `learned_slope`）完成后，三个数据集统一使用 `proj_add`、`P=16`（与当前 H1 signed-state 结构相同），不再新增 concat 臂。H1 本就是 `RiftDecoder(task="h1", proj_dim=16)`，无需改接口（仅修复 stolen-temporal 设备 bug：新参数继承 `recency_slopes.device`）。

- M2：`scripts/m2_projadd_learnable_train.py` / `m2_projadd_learnable_score.py`（子代理交付，父级审核）。配对参照 `results/rift_v1/m2_r50_recency_s42_formal_v1`（与 concat 参照同 manifest/缓存）。其 `score_receipt` 的 e11=0.34081 是 **ext4** 选择；EXT6 面参照由 `m2_projadd_reference_ext6_score.py` 对参照 checkpoint 重评产生。
- M1：`scripts/m1_projadd_learnable_train.py`（子代理交付；父级把优化器分组改回参照的名字免 decay 分组 + 新参 wd=0 组）。配对参照 `results/rift_v1/m1_r100_recency_s42_formal_v3`（HO3 e3=0.70469，equal_session_mean earliest-max）。
- H1：参照 `0.46765`（阶梯未变，无 fixed 臂）。
- 正式臂：M2/M1 各 `learned_slope` + `fixed`（fixed = 缩放阶梯、零新参，用于分解阶梯效应 vs 学习效应）；H1 仅 `learned_slope`。验收门槛沿用上表（各任务对各自 proj_add 参照）。

## ADDENDUM 2026-09-10（二）· 用户裁决：统一默认半衰期 + proj_add 优先

M2 concat `learned_slope`（缩放阶梯）EXT6 pick e7 = 0.3206（参照 0.39006，非劣失败；缩放后最尖头半衰期 0.693 bin 过强，e7 后曲线持续下滑）。用户裁决：**三个数据集一律 `--ladder default`（半衰期 4/8/16/32/64/128 bins 跨数据集统一，即各参照原版阶梯）**；先全部走 proj_add，仅 M2 补一条 concat 臂。

默认阶梯下 `learned_slope` 初始斜率与参照完全一致，`fixed` 臂退化为参照复刻——故不再跑任何 `fixed` 臂，参照数字直接充当 fixed 基线。缩放阶梯的已完成/半途产物仅作失败证据保留（`m2_learned_slope_s42` 完整、`m2_fixed_s42` 15/24 epochs 被停）。

新臂清单：`m2_projadd_learned_slope_default_s42`、`m2_concat_learned_slope_default_s42`、`m1_projadd_learned_slope_default_s42`、`h1_learned_slope_s42`（本就是 default 阶梯，未中断）。判定：M2 proj_add vs 参照 EXT6 重评；M2 concat vs 0.39006；M1 vs 0.70469（HO3）；H1 vs 0.46765。

## ADDENDUM 2026-09-10（三）· 消融与第二 seed 的统一口径

用户要求消融口径统一。核对结论（2026-09-10 15:4x UTC）：

1. **网络/配方**：ACTIVITY_ONLY、NORM_ONLY、FULL、s43 全部 = proj_add P16 + `learned_slope`（24 新参）+ default 阶梯 + 论文配方；M2 侧 `ensure_frozen_recipe` 拒绝漂移，H1 侧镜像模板。
2. **唯一变量**：身份注入。消融臂共享参数与 FULL 臂同 seed42 stock decoder byte-equal（配对断言在各自 smoke/收据中）——FULL−ACT/NORM 之差纯为身份贡献。
3. **评分**：M2 一律冻结 EXT6 epoch-pick（earliest-max equal_session_mean，EMA）；H1 一律 HO-M3 grouped-seven `select_epoch`。FULL、参照重评、消融、s43 同评分器同规则。
4. **NORM 公式**：`z=(rate_sess−μ_src)/max(σ_src, eps)`，`eps=1e-6` 双任务一致（M2 原为 1e-8，2026-09-10 对齐；重跑测试 11/11 与 CPU smoke）。μ/σ 仅由 held-in 会话计算；评分从 run_meta 读回并以 sha256 封印，HO 数据不参与。rate=当日 label-free 校准支持集池化率：M2 用 M33 固定 33-trial 支持（与 B3S E0 同一支持文件）；H1 用逐 bank (session,start,budget) 支持集（与其校准机器同构）。
5. **ACT 语义**：E0=活动身份原样、carrier T=0。M2 银行后置零（新建数组）；H1 冻结 C2 材料化器零 side-carrier 且 fail-closed 断言输出为零。
6. **对照系**：消融的内对照=同网络 seed42 FULL（M2 0.36338 / H1 0.4624）；固定 recency 参照（0.36399 / 0.46765）为外锚。旧 fixed-recency ACT 官方数字（582217/582218）配方不同，不与本次消融直接相减。
7. **seed 臂**：`--seed 43` 仅改 run 种子域（模型/采样/dropout/记录），参照绑定与数据契约仍锁 seed42 参照 run；M2 评分脚本已改为从 run_meta 读 seed（2026-09-10 补）。
8. **GPU 排程**：GPU0 = M1(train+score) → M2 ACT → M2 NORM（各含 EXT6 评分）→ M2 s43 → H1 s43；GPU1 = H1 ACT → H1 NORM（各含内嵌 HO-M3 评分）。消融链以 `DRIVER_GPU0_M1_DONE` 为门（避免与 M1 评分重叠）。

## ADDENDUM 2026-09-10（四）· ACT 标签泄漏审计（用户质询成立）

用户要求确认 ACTIVITY_ONLY 无任何标签。逐行审计结论：

- **H1 干净**：E0 = 按 TrialNum（试验编号、时间顺序去重）选取的校准 trial 纯 spike 计数的时间轴 cubic 重采样（`h1_m4_eb_pilot.interpolate_trial_identity` → `eval_trial_neural`：`neural[eval_mask & trial_num==k]`）；行为数据（velocity/targets）为 record 独立字段，身份路径不读。ACT 臂材料化器 side 输入传字面零矩阵并 fail-closed 断言。
- **M2 原activity_only 被污染**：缓存 E0 由 `_compute_identity_for_dest` 计算，编码器 side 输入 = `champion.empty_contrast_side(move_t4)` = **[MOVE-T4, 0]**——MOVE-T4 由校准期目标角度（行为标签）构造，泄漏进 E0。该臂（EXT6 0.35948）重新定性为 **"carrier token 通道消融（E0 side 含标签派生 MOVE-T4）"**，不得作为 label-free ACT 下结论。
- **修正**：新增 `activity_only_empty_side`（编码器 side=zeros(N,8) 全零，从 calib_activity.npy 现算 E0；训练/评分同变换、同冻结编码器、sha 封印）。M2 嵌套阶梯以该臂为 ACT。NORM 不受影响（E0 整体替换为发放率 z，不走编码器）。
- FULL 允许校准标签（side 含 MOVE-T4 + carrier token），语义不变。

## ADDENDUM 2026-09-11 · M1 载体代际纠正（用户质询成立）

用户指出最终 M1 载体应为 muscle_response16_svd4/global_rms（与 H1 统一的 SVD 载体线，official **582205 = 0.6260**，载体包 `m1_muscle_r100_v1/carrier_official4/carrier_pack.npz`）。核查确认：现有 M1 proj_add 全家（参照 `m1_r100_recency_s42_formal_v3`=0.70469 与本轮 P16 0.7075 / P32 0.6949 / P8 半途）绑的是**旧代 rSyn3-refit-v1**（09-05，EMG-NMF；"rSyn3"是机制名、数据同为肌肉来源，但属旧代、与主线包逐位不同）。

处置：
- 旧代 M1 proj_add 结果保留为 rSyn3 线记录，不进论文主线。
- 新线 `m1_projadd_muscle_train.py`：proj_add（P16/P8/P32）+ muscle_response16_svd4 载体 + learned_slope@default；先跑 fixed 与 learned P16 两臂建立主线参照，ACT 消融随其后（agent 已改道）。E0（b3_identity 校准神经）不变。
- M1 P8（rSyn3）已在途终止并清理。
