# H1 继任交接：精度线现状与下一步（2026-09-03）

**给接手的 AI。** 本文是 2026-09-03 一次分析会话的完整交接，不是新实验结果。读完应能独立推进，不必回原对话。推进前先把本文和引用收据对一遍；数字以收据为准。

| 项 | 值 |
|---|---|
| 会话日期 | 2026-09-03 |
| 工作区 | `/home/xinyuan/Work_host/SPINT`（git `main`，工作树很脏，**禁止** `git add .`） |
| 分析者 | Cursor Grok 4.6（本会话只读分析 + 克隆队友仓库；**未**改 SPINT-main、**未**训 GPU、**未**再提交 EvalAI） |
| 队友仓库 | `https://github.com/Michael-XVII/iBCI` |
| 队友部署分支 | `exp/h1-cal-aug-all-source-m3-deployment-v1` @ `5dd9bb4a7377a5431b7dbac4f1378e529130eb1a` |
| 相关训练分支 | `exp/h1-cal-aug-all-source-heldout-v1` |
| 本地克隆（可能已删） | `/tmp/ibci-h1/deployment-v1` |
| 解释器 | `~/miniconda3/envs/spint/bin/python`（Torch 2.5.1 / CUDA 11.8） |
| 挑战 | EvalAI 2319，phase `few-shot-test-2319` / ID 4599 |

**一句话结论：** 官方 H1 精度没有被任何变体显著超过 paper-LR SPINT。C1 相对 T0 有官方增益，相对 SPINT 只有约 +0.023 HO，且 HO 标准差 ~0.13。Carrier 方向（含 top-K / D-opt / 再抠 PCA）对 H1 已关闭。剩下可测的精度杠杆是 **冻结 C1、冻结 M3 H-C，在评测流上把 identity 从 M3 合法补到 M7**。Held-out 校准文件恰好 3 个 trial，`on_done` 为空操作，不能从 calib 再拿活动。

---

## 0. 给继任者的任务边界

### 现在授权你做的

1. 把本文 §8 写成 **CPU-only work order**（解码器状态机、源侧能量门阈值、五日期冻结打分、闸门）。
2. 在用户明确同意后，跑 **冻结权重** 本地屏（真 trial 上界 vs 能量门可部署律）。
3. 两道闸都过之后，再单独请示 GPU 训练或 EvalAI。默认 **先不训、不提交**。

### 未过闸门或用户未授权时禁止

- 改 carrier / φ / D-opt / top-K / 新 PCA 秩 / query-oracle 重训 consumer 当主线
- 把 M2 `chunk100e` 超参原样拷到 H1
- 用 held-in-minival ~0.97 当官方或选模型
- 对 H1 做 trial-boundary 复原（M2 已失败；H1 没有 `on_done`）
- 可变基数微调的同一 recipe（已 `STOP_VARIABLE_ACTIVITY_SUCCESSOR`）
- `git add .`、改封印文件、EvalAI push、开 held-out 标签做选择

### 合同硬约束

- H1 官方是 **continual**：评测流不按 trial 切，`on_done` 被规定为空。
- 全部 14 个 `held-out-calib` NWB **恰好 3 个合法 trial**。M3 用尽校准标签和校准活动。泄漏无关的 HO 分数只存在于远程 `eval/` 流。
- 本地无 `eval/`。公开数据在 `SPINT-main/data/000954/`：`held-in-calib`、`held-in-minival`、`held-out-calib`。
- 官方指标：`sklearn r2_score(..., multioutput='variance_weighted')`，同日集合拼接（S0…S12），**不**丢 W=700 warmup。
- 开发面常用 last-bin、等 recording、再等日期。两种指标不要混报。
- 封印 H1 文件（约 2026-08-12 前的 `SPINT-main/src|configs|scripts`）只许 import/subclass。

---

## 1. 官方分数（唯一治理 HO）

| 系统 | 提交 | HO R² | HI R² | 相对 paper-LR SPINT HO |
|---|---|---:|---:|---|
| **C1** prefix-cycle M3 | `581748` | **0.2841 ± 0.1348** | 0.4587 ± 0.0401 | **+0.023** |
| CarrierID 历史 H-C | `578689` | 0.2749 ± 0.1272 | **0.4731** | +0.013 |
| **paper-LR SPINT** | `578474` | **0.2615 ± 0.1487** | 0.4704 | — |
| T0 固定 M7→M3 | `581747` | 0.2411 ± 0.1157 | 0.3557 | −0.020 |
| released-LR SPINT | `578473` | 0.2099 ± 0.1142 | 0.4390 | 弱 LR 基线 |

- C1−T0 官方 HO **+0.043**（相对 T0 +18%），HI **+0.103**。裁决：`COMPLETE_H1_M3_EVALAI_OFFICIAL_C1_IMPROVES_T0`。这只证明 prefix-cycle 相对固定 M7 对照有用，**不是**超过 SPINT。
- C1 HI **低于** paper-LR HI（0.459 vs 0.470）。
- 官方无 per-session 分数，不能做配对显著性。HO 标准差 0.13–0.15。
- 论文安全说法：密 H-C + ~58k identity，延迟约快 12%（0.114 vs 0.129）。不要写 H1 精度主导。

收据：

- 队友官方：克隆内 `tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_evalai_submission_v1/{official_results.json,EXPERIMENT_RECORD.md}`
- 历史 CarrierID：`sua_exploration/docs/CURRENT_RESULTS.md` §H1；`HANDOFF_H1_MASTER_ANALYSIS_AND_PAPER_LINE_20260812.md`

---

## 2. 队友 GitHub 系统（读分支文档 + 解码器后的事实）

克隆命令（GitHub MCP 当时不可用，用的 `git clone`）：

```bash
git clone --branch exp/h1-cal-aug-all-source-m3-deployment-v1 \
  https://github.com/Michael-XVII/iBCI.git /tmp/ibci-h1/deployment-v1
# HEAD 5dd9bb4  message: exp(h1): seal all-source M3 deployment and EvalAI result
```

### 2.1 两个臂

| | T0 | C1 |
|---|---|---|
| 网络 | `H1CarrierIdSpint`，h=32，W=700，~1.09e7 参数 | 同结构 |
| 训练 identity | 固定 **M7** | 确定性 prefix-cycle **M7/M5/M4**（同一预定 M7 块内） |
| 训练载体 | 该块最早 **M4** 的 H-C | 同 |
| 训练不含 | **M3 从不进训练循环** | 同 |
| 部署校准 | 最早 3 trial：identity `[3,1024,176]` + `fit_deployment_carrier(..., 3)` → `[176,4]` | 同 |
| 源 | 13 个 held-in-calib；解析 **q=12，λ=10** | 同 |
| 优化 | seed 42，50 epoch，只留 epoch 49；Adam 5e-5，wd=0，FP32；last-bin MSE，输出 `/20`；batch 32 | 同 |

M3 是 **部署预算**，靠 prefix-cycle 外推（内部诊断 `STRONG_M3_PREFIX_EXTRAPOLATION`），不是新训练格子。

### 2.2 部署解码器

文件：`SPINT-main/third_party/falcon_challenge/h1_carrier_id_spint_decoder.py`（队友分支；本工作区未必有同文件）。

- `reset`：按 FALCON `hash_dataset` 装入该 session 的冻结 M3 identity+carrier。
- `predict`：因果 W=700 缓冲，last-bin，`/20`。
- **`on_done` 是 no-op**。注释写明：H1 continual，trial 和 eval-mask 间隙都不重置校准或历史。
- 校准之后 **identity 永不增长**。这是和本地活动头寸实验的关键差距。

### 2.3 为何必须 M3

工作令原文：全部 14 个 held-out-calib **恰好三个合法 trial**。用 M4 载体会吃掉第 4 个 trial，本地 HO-calib **没有**独立 query 面。因此：

- 不能在 HO-calib 三 trial 上算泄漏无关 R²（那是拟合面）。
- 不能从 calib 再扩 activity 池（和 M2 的 first-30 unlabeled calib 不同）。
- 官方 HO 只能来自远程 eval 流。

### 2.4 本地 vs EvalAI 不是 Docker 坏了

| 面 | T0 | C1 | 它实际是什么 |
|---|---:|---:|---|
| 本地 held-in-minival | 0.953 | **0.968** | 训练日期；每文件 **2 trial** ~32s；**丢掉前 699 bin（W=700 warmup）** |
| EvalAI Held In | 0.356 | **0.459** | 远程 `eval/` 长流；**全部 eval_mask bin**；6 个 session 池化 R² |
| 本地 HO-calib 在 3 个拟合 trial 上打分 | ~0.32 | ~0.36 | 新日期但 **标签泄漏** |
| Date-LODO M3 转移 | ~0.29 | ~0.34 | 更好的本地类比 |
| **EvalAI Held Out** | 0.241 | **0.284** | 7 个新日期，官方 |

Minival NWB：均值 T≈1609，2 trial；warmup 后可评约 894 bin（约 eval-valid 的 54%）。Calib 文件约 10k bin、8–15 trial（held-in）；held-out-calib 合法 trial=3。Minival 数组曾 pad 到 1831（batch_size=13 `pad_sequence`）。

**禁止**用 0.97 minival 谈精度。短流还会让「第一块能量门」主导分数（M2 chunk100e minival 曾到 −0.032，机制相同）。

包装 sanity 记录：克隆内 `.../h1_cal_aug_all_source_m3_deployment_v1_package_a1/EXPERIMENT_RECORD.md`（C1 minival 0.968，终端 SHA `4137495462…`）。

---

## 3. 「有没有变体显著超过 SPINT？」

**官方 HO 精度：没有。**

| 对照 | 含义 |
|---|---|
| C1 0.284 vs SPINT 0.261 | +0.023，淹没在 HO std 里；C1 的对照臂是 T0 不是 SPINT |
| 开发 H-C−H-S **+0.056**（4/5 日期） | 相对 **activity-only SPINT 拓扑**，不是 EvalAI SPINT |
| H-SE5 | fold-0 正，第二日期复制失败，不能当稀疏主线 |
| CI64 | vs CI32 **−0.020**，2/5 日期；加宽停 |
| output EMA α=0.7 | 等日期 **+0.0028**，3/5；`COMPLETE_NO_TRANSFER` |

论文线（已封印）：密多阶段 H-C + 紧凑 consumer + 延迟，外加稀疏事件的失败边界。见 `HANDOFF_H1_MASTER_ANALYSIS_AND_PAPER_LINE_20260812.md`。

---

## 4. 已关闭、不要再开的路线

| 路线 | 结果 | 处置 |
|---|---|---|
| 稀疏 H-SE5 / context-event | 跨日期失败 | 边界，不救种子 |
| CI64 / 加宽 consumer | −0.020 | 停；H64 禁止 |
| Output EMA | +0.003，3/5 | 无转移 |
| Query-oracle 换局部 H-C | **−0.00286**（载体明显被挪动，R² 不动） | 冻结 consumer 不吃更好局部载体；**不是**估计器数学饱和 |
| 可变基数 identity 微调 | growing 全面变差 | `STOP_VARIABLE_ACTIVITY_SUCCESSOR` |
| 代表点 / FIFO cap30 vs 先长后冻 | 0 增益 | `STOP_CAUSAL_REPRESENTATIVE_ACTIVITY_CAP30` |
| H1←M2 共享轴 / M=2 T4 映射 | 无头寸 | 关 |
| D-opt / top-K 选 trial（H1） | H1 无原生 θ；C1 用最早 3 个 chronological trial | 不要抄 M2 |
| 通道 top-K / T4GATE | 相对 T4 为负（他任务） | 关 |
| B1 SFC 类身份匹配 | H1 未作为精度线；M1/B1 已 NO-GO | 关 |
| 无标签神经协方差 top-K 再对齐 | H1 oracle 上界关 | 关 |
| 把 M1 calibration-aware MLP 硬投影到 C1 | M1 上对满 SPINT **−0.022** | 禁止 |

Query-oracle 细节（泄漏诊断，不可进主结果）：

- support 0.525511 vs query-M4 0.522652，Δ **−0.002859**
- 相对 Frobenius 0.83 / 1.05，预测相关 0.994
- 收据：`SPINT-main/docs/H1_CARRIERID_QUALITY_DIAGNOSTIC_PROGRAM.md`
- 若有人要「重训 matched query-carrier consumer」，那仍是泄漏诊断，不能当官方精度实验。

---

## 5. 仍开放的头寸：因果活动记忆

冻结 H-C、五日期 LODO（19250108/13/15/19/20），载体始终是前 4 个 labelled trial。等 recording last-bin R²，再等日期均值。

权威：

- `tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_DATE_LODO_ACTIVITY_HEADROOM_V1_20260828.md`
- `tfpd_exploration/h1_series_20260830/results/h1_date_lodo_activity_headroom_v1.json`（SHA `65c9bb40ad45ab7b74740da88fd8081504b7656e807e76b6eb9db903450adb68`）
- 交互解释 V2：`.../h1_date_lodo_activity_system_compare_v2.json`

### 5.1 Growing vs static（过闸）

| 日期 | Static M4 | Causal growing cap30 | Δ |
|---|---:|---:|---:|
| 19250108 | 0.545514 | 0.585001 | +0.039487 |
| 19250113 | 0.339962 | 0.375285 | +0.035324 |
| 19250115 | 0.521320 | 0.548556 | +0.027236 |
| 19250119 | 0.317487 | 0.363780 | +0.046293 |
| 19250120 | 0.349828 | 0.446173 | +0.096346 |
| **等日期** | 0.414822 | 0.463759 | **+0.048937** |

5/5 日期、11/11 recording 为正。Bootstrap 95% `[+0.033, +0.074]`。裁决：`PASS_H1_ACTIVITY_HEADROOM_BREADTH_FOR_NEW_STATE_DESIGN`。

Growing 臂标记为 `CAUSAL_CARDINALITY_OOD`（训时 identity 基数不是 30）。

### 5.2 Rolling FIFO M4（基数匹配，但不够当主臂）

从同一 JSON 抽出的 equal-recording rolling：

| 日期 | Rolling M4 | vs static |
|---|---:|---:|
| 19250108 | 0.578829 | +0.0333 |
| 19250113 | 0.354775 | +0.0148 |
| 19250115 | 0.518955 | **−0.0024** |
| 19250119 | 0.359273 | +0.0418 |
| 19250120 | 0.432503 | +0.0827 |
| **等日期约** | | **+0.034** |

收回 growing 的大部分，但 **19250115 为负**。不要把 rolling 当 5/5 主臂。

### 5.3 载体 × 活动

V2 正确交互：`(H-C growing − H-C static) − (H-S growing − H-S static)` = **−0.0064**。裁决：`NO_MATERIAL_CARRIER_BY_ACTIVITY_INTERACTION`。H-S 上活动增益甚至略大。终点仍是 H-C+growing 最高（`H-C_CAUSAL_GROWING_RETAINS_HIGHER_FINAL_LEVEL`）。**增益来自 identity，不是更好的 H-C。**

### 5.4 可变基数训练已失败

`WORKORDER_H1_VARIABLE_ACTIVITY_EXPOSURE_V1_20260828.md`：从封印 H-C fold0 微调，50/50 重放 M4 vs M5–全部。因果 growing 相对封印 checkpoint **变差**（fold0 growing −0.005）。`STOP_VARIABLE_ACTIVITY_SUCCESSOR`。不要加 epoch/种子救同一配方。

### 5.5 对 C1 的含义

C1 **训练已经见过 M4/M5/M7**，部署却把 identity **冻在 M3**。本地 +0.049 是在 **H-C date-LODO、M4 静态 vs 长到 ~14–30** 上测的，不是 C1、不是 M3、不是无 trial 边界。因此下一步必须 **在 C1+M3 上重新量头寸**，不能直接把 +0.049 写成官方预期。

乐观换算（仅供停线用，不是预测）：若约一半转到官方，C1 0.284 → **~0.30–0.31**。若官方相对 C1 **< +0.02**，停 H1 精度线。

---

## 6. H1 vs M2：载体拟合不该一样

| | **M2** | **H1** |
|---|---|---|
| 行为 | 2D 圆心角 θ | 7 维开环速度，有效秩 ~4（`U ∈ R^{7×4}`，行为 PR≈3.14） |
| 观测 | trial 均值发放 | trial 内 **逐 bin 密速度** |
| φ | `[cos θ, sin θ, 1]` 每通道 T4 | `[1, z_PCA]` → 7 维速度 → `U` 到 4 维 |
| 短预算 support | first-30 有限角里 **D-opt k=4** | **时间上最早 3/4 trial** |
| 正则 | λ=0.1，只罚 a,c | λ=10（C1）或 100（历史 all-source q=16）；截距不罚；EB→μ |
| M4 可辨识性 | chronological 4 点常病态 | 每 trial 是长轨迹，chronological M4 已满 |
| M4 缺口 | ~78% 载体估计 / 22% 活动 | query-oracle 近零；活动头寸大 |

M2 一条校准 trial ≈ 一个方向样本。H1 一条 trial ≈ 几百个 7 维速度点。所以 M2 要 D-opt+强收缩；H1 不要抄。

M2 官方锚（不要拿来选 H1）：T4 M30 HO **0.303**；D-opt M4 + act30 **0.290**。决策记录：M4 才开 D-opt，M10 起关掉。M1 TOP-4 D-opt freeze HO 0.640，仍低于 original SPINT 0.649。

H1 拟合代码：`SPINT-main/src/data/h1_m4_eb_pilot.py` 的 `fit_deployment_carrier`（允许 3 或 4 个 unique trial，禁止 padding）。M2：`tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py` 的 `_ridge_side` + `fit_ridge_t4`。

---

## 7. PCA：H1 与 M1 不是同一方法，不要互抄

H1 H-C 有 **两层源冻 SVD**：

1. **神经 `pcs`（q=12/16）**：发放率标准化后 SVD。目标 rates → z，再 **神经→行为** ridge：`速度 ~ 1+z`。
2. **行为 `U`（7→4）**：各源 `raw_rows` 再 SVD，EB 收缩。

M1 上叫 PCA 的至少三套：

| M1 | 做什么 | 结果 |
|---|---|---|
| EMG-AFC4 | 16 维 EMG→3 PC，然后 **行为→神经**（编码，像 T4） | 紧凑 decoder Full−Zero4 **−0.0065**，停 |
| 源冻 PCA / predictive PCA | EMG 流形当 M10 DirectRidge 瓶颈 | **+0.0001 / +0.003** |
| Calibration-aware MLP q=8 | 训练目标里展开部署 ridge | vs DirectRidge **+0.065～0.072**；硬投影到满 SPINT **−0.022** |

结论：无监督 PCA 在 M1 是零对照。M1 有用的是「部署算子写进源目标」，且 **只对闭式适配器**。C1 已是满 SPINT consumer，正是 M1 禁止硬投影的那一类。H1 下一步不是新 PCA。rSyn3/NNMF 是非负肌电，不能套有符号 H1 速度。

权威：`sua_exploration/docs/HANDOFF_M1_CALIBRATION_AWARE_BEHAVIOR_BOTTLENECK_20260824.md`，`M1_EMG_AFC4_FEASIBILITY_AND_MINIMAL_BLUEPRINT.md`。

---

## 8. 推荐下一步（请按此写 work order，先 CPU）

锚点：**已提交 C1 权重 + 冻结 M3 H-C**。只改 identity。不要先长到 30（C1 只在 M4/M5/M7 上训过；可变基数微调已失败）。

### 8.1 三臂（同一 C1、同一 M3 载体、同一 query 面）

| 臂 | 律 | 官方合法？ |
|---|---|---|
| A | 静态 M3（C1 现状） | 是 |
| B | 用 **真 trial 边界** 把 identity 从 M3 提交到 M4→M5→**M7 后冻结** | **否**（本地上界） |
| C | **能量/定长 chunk** 提交，同样长到 M7 后冻 | 是 |

Chunk 不要抄 M2 `chunk100e`（那是 100 bin、W=50）。H1 identity 是 `[M,1024,176]`，窗口 W=700。提交块应是 **1024-bin 神经段**（或与训练 identity 相同的插值/重采样）。能量门阈值只在 **源会话** 上冻结，目标会话不许再选。保护校准 prefix-3，类似 M2 保护 prefix-4 而非 D-opt 行。

Decode-before-commit：先用当前 identity 解码当前 bin，再把候选块送进门；门过才重算 identity，供 **下一 bin** 使用。

### 8.2 闸门（预声明，失败即停）

在 **五日期 held-in 外日期** 上打（有 trial 4+，才能算 B）。不要用 minival 决定生杀。

1. **B−A ≥ +0.01 且 ≥4/5 日期为正**。否则头寸换到 C1+M3 后消失 → 停 H1 活动线，不要训。
2. **C 收回 B 的 ≥50%**（等日期）。否则是检测器问题，不要训网络、不要提交。
3. 两道都过：本地长流烟测 + 包装。Minival 只验证接口/`on_done` 仍为 0。短 minival 上 C 可以很差，这是律的真行为（M2 已见）。
4. 若用户授权 EvalAI：相对 C1 `581748` 的 HO 若 **< +0.02**，停 H1 精度线，不要再开训练救。

### 8.3 明确不要并行

- 为「见过 M>7」重训 identity：只在闸门 1 过且冻结 M7 仍明显低于 growing-30 上界之后，另开 work order。
- Rolling FIFO 当唯一部署律：一日期为负。
- 无标签恢复真实 trial。
- 改 carrier、换 φ、加宽、EMA、D-opt。

### 8.4 参考实现（机制类比，不是超参）

M2 可部署能量门（合同不同，只抄纪律）：

- `tfpd_exploration/docs/DEPLOYMENT_M2_CENAT_CHUNK100E_TTA_V1_20260903.md`
- 纪律：49-bin 零预历史对齐训练；保护 chronological seed 而非 D-opt 行；decode-before-commit；`on_done` no-op；权重冻结。

H1 没有 trial `on_done`，所以 C 必须是 **纯神经统计量**（能量/长度），在源上定死。

---

## 9. 权威路径速查

### 队友 C1/T0（克隆或 GitHub）

- 部署工作令：`tfpd_exploration/h1_series_20260830/H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_WORK_ORDER.md`
- 执行修正：`.../docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_EXECUTION.md`
- M3 为何不能在 HO-calib 打分：`H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1_WORK_ORDER.md`（「Why H1 held-out-calib cannot provide a clean M3/M4 local R²」）
- 解码器：`SPINT-main/third_party/falcon_challenge/h1_carrier_id_spint_decoder.py`
- 官方结果：`.../results/h1_cal_aug_all_source_m3_evalai_submission_v1/`

### 本工作区 H1 头寸

- 活动广度：`tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_DATE_LODO_ACTIVITY_HEADROOM_V1_20260828.md`
- 交互 V2：`.../WORKORDER_H1_DATE_LODO_ACTIVITY_SYSTEM_COMPARE_V2_20260828.md`
- 可变基数失败：`.../WORKORDER_H1_VARIABLE_ACTIVITY_EXPOSURE_V1_20260828.md`
- 代表点失败：`.../WORKORDER_H1_CAUSAL_REPRESENTATIVE_ACTIVITY_V1_20260828.md`
- H-C 拟合：`SPINT-main/src/data/h1_m4_eb_pilot.py`
- Query-oracle：`SPINT-main/docs/H1_CARRIERID_QUALITY_DIAGNOSTIC_PROGRAM.md`
- 论文线：`sua_exploration/docs/HANDOFF_H1_MASTER_ANALYSIS_AND_PAPER_LINE_20260812.md`
- 总表：`sua_exploration/docs/CURRENT_RESULTS.md`
- 缺口分解（M2 78/22；H1 oracle）：`tfpd_exploration/docs/HANDOFF_CALIBRATION_GAP_DECOMPOSITION_20260824.md`

### 数据与评测

- NWB：`SPINT-main/data/000954/`
- FALCON evaluator：env 内 `falcon_challenge/evaluator.py`

---

## 10. 本会话还涉及、但不要当成 H1 下一步的

- **M1 TOP-4**（rSyn3 上 D-opt k=4）：freeze HO 0.640 / acyc 0.624 vs original 0.649。少标签协议，不是 H1。
- **M2 AJPF-C / chunk100e**：活动 TTA，carrier 仍冻 D-opt4。H1 只借「continual + 能量门 + 冻载体」纪律。
- **B1 Spectral Functional Carrier**：天花板分析后 NO-GO。不要重开。

---

## 11. 建议的执行顺序（继任者清单）

1. 确认本机仍能读五日期 H-C checkpoint 缓存与 C1 包（队友克隆或用户指定的 checkpoint 根）。没有 C1 权重就先找用户，不要用 H-C date-LODO checkpoint 冒充 C1。
2. 写 CPU work order：臂 A/B/C、1024-bin 块、源侧门、五日期面、§8.2 闸门、禁止项。
3. 合成/CPU 测试：decode-before-commit、保护 M3、M7 后不再提交、`on_done` 仍空、不碰载体。
4. 用户签字后再跑冻结 GPU 打分（无 backward）。
5. 打印闸门收据。失败停。成功再单独请示包装/EvalAI。

若用户只想要文档、暂不实验：把 work order 放在 `tfpd_exploration/h1_series_20260830/docs/`，文件名建议 `WORKORDER_H1_C1_M3_TO_M7_ACTIVITY_TTA_V1_YYYYMMDD.md`，并标明 `CPU_REVIEW_ONLY` 直到另一次授权。
