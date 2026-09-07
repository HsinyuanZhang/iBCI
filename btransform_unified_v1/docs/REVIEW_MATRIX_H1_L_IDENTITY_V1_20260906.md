# REVIEW — MATRIX_H1_L_IDENTITY_V1 进度 + 审核

- 对象：`docs/MATRIX_H1_L_IDENTITY_V1_20260906.md` + `scripts/h1_matrix_mf250_proj_add.py` + `results/h1_matrix/`
- 日期：2026-09-06 17:01 本地。审核方：进度/合同审核（不执行训练）。
- 结论：**骨架与 LODO 面清单已就绪；训练未启动。按现在的脚本，即使启动也会踩两处合同违反（CAL-1 被静默降级；SEL-2 仍在同源 2,908 上选 epoch）。先改再训。**

---

## 0. 进度（此刻）

| 项 | 状态 | 证据 |
|---|---|---|
| 五臂 L=100 | 4/5 终局，(e) 未落盘 | a 0.080 / b PROXY 0.178 / c 0.009 / d 0.020；进程已停，GPU 空 |
| 骨架 pytest | READY | `M_F250_20260906T084902Z/wait_skeleton.log` 16:54 `54 passed` |
| LODO 面清单 | READY | `face_inventory.json` 17:01：holdout `1925-01-20` 两 session；train 18,935 / exam 2,952 / sel 2,908 |
| M-F700 仲裁 | PASSED，未开训 | `M_F700_arbitration/gpu0_arbitration.json` 16:58；仓库里 **没有** M-F700 训练脚本 |
| M-F250 训练 | **未启动** | 无 `BTRANSFORM_MF250_TRAIN` 进程；无 heartbeat / metrics / ckpt |
| GPU | 空闲 | GPU0 453 MiB / GPU1 23 MiB，无 compute app |

矩阵 8 cell（A250/A350/B250/B350/C250/F250/D250/E250）全部未开训。执行修订里的 F700 也没有可跑入口。

---

## 1. 阻断（启动前必须改或显式废合同）

### A. CAL-1「全局启用」被脚本单方面改成 CAL-2/M3

工单 §0：`CAL-1 {7,5,4,3} → 部署 M3`，违反即废。  
脚本自己写：「coordinator ruling，降为 CAL-2 固定 M3」，因为 `build_h1_bank` 对 7/5/4 抛 `NotImplementedError`。训练全程吃 budget=3 的冻结 bank，只把 CAL-1 日程写进 `run_meta`。

这不是实现细节：H1 唯一官方证实的增益成分（+0.043）整轴被拿掉。F250/F700 若带着 CAL-2 当「同注入 L 轴对照」，后面 A/B cell 再上真 CAL-1，L 轴和喂法轴都会乱。

要二选一，写进工单修订，不能藏在脚本 docstring：

1. **先接线 CAL-1**（从 NWB 重 trialize 7/5/4），矩阵再开；或  
2. **整矩阵显式改成 CAL-2/M3**（与五臂同口径），§0 合同改掉，不再声称 CAL-1。

现状 = 口头 CAL-1、手上 CAL-2。

### B. SEL-2 仍在 13 session 的 2,908 上 pick epoch —— 刚被证伪的选择器

工单 §0 / M1 LOSO：方法选优考卷是 **留出日期 LODO**。  
脚本：`_sel2_pick` 吃的是 `sel2908_ema`（全部 13 个 session 的 query-grid，含 11 个训练日）。LODO exam（01-20，2,952）只辅报。

这正是 M1「31,252 同源面打平、整 session 留出掉 0.14」的同一错误。2,908 可以留作与 formal12 对齐的辅报，**不能当 earliest-max 的选 epoch 面**。应预注册：pick 只看 holdout 日期的 exam EMA（pooled 或 session-mean，写死一条）。

### C. M-F700 写进了执行修订，但不是可运行 cell

- `MATRIX_L = (250, 350)`，`cell_geometry` 默认拒 L=700。  
- `MATRIX_CELLS` 仍是 8 个，没有 M-F700。  
- 仓库只有 `h1_matrix_mf250_proj_add.py`。  
- §2.5 原文：「全窗 L=700 不在矩阵内，胜出后再跑」；§2 修订框：「F700 提前、与 F250 双卡并行」。两段互相打架。

在 F700 脚本 + 几何豁免 + 工单单一说法落地之前，**不要占 GPU0 空转仲裁**。F250 可以单独开（改完 A/B 之后），不要死等一个还不存在的 F700。

---

## 2. 重要（不阻断启动，但会废解读）

1. **(b) 必须是真 C2 pre-pool 36-d。** `adapters.build_h1_bank` 已实现 `joined36 = cat(pre_pool32, H-C4)`，并拒绝五臂 PROXY。五臂 b=0.178 是 PROXY，**不能**当成 M-A 的先验。A250 才是第一次量真 36-d。  
2. **五臂 (e) 不必再等。** 排序已定；c < d 已足够把 M-C250 近乎划除（与修订一致）。  
3. **字母容易反：** M-A = 用法 (b) joined；M-B = 用法 (a) concat。写 receipt / 口头对齐时用「joined / concat」，少用 A/B。  
4. **TRN-1 731 upd/ep 对不上 LODO。** 18,935 / 32 ≈ 592。脚本有记录，不要在 receipt 里写成 formal12 同口径。  
5. **视界判读阈值（§2.1）在 250 vs 350 上只有 100 bin。** 降幅 0.01–0.02 会落在「≤0.01 取最短」和「>0.02/100bin 记拐点」之间的空洞。预先写死：0.01 < Δ ≤ 0.02 时怎么办。  
6. **F250 的 `gpu0_touched: False` 与修订「双卡并行」冲突。** 现在 GPU 都空，F250 钉 GPU1 没问题；不要再写「必须等 GPU0 五臂 pid 1803132」——那个进程已经不在。

---

## 3. 已做对的

- LODO 留出最后一天 1925-01-20、两 session，满足 §3「≥2」。  
- 目标坐标冻结：L 只改回看多长（`query_starts+699` / `eval_mask`），P0-4。  
- `proj_add` 数学定义清楚：P: 700→16 bias-free，加到 local16，token_in=20；P=0 退化为 no-E0。  
- 真 joined36 与 PROXY 在 adapter / 测试里被拆开。  
- 无 EvalAI、不写历史结果根。  
- M-C250 条件化 + 五臂 c=0.009 的降级是一致的。

---

## 4. 建议启动序（改完 A/B 之后）

1. 只开 **M-F250**（GPU1 或任一张空卡），SEL-2 改成 holdout exam。  
2. 2×2 主矩阵：A250 / B250 先于 350（同一 L 上先分喂法）。A 必须走真 joined36。  
3. F700 要么补成正式 cell 再与 F250 成对，要么从修订里删掉，回到 §2.5。  
4. C250 默认不跑。D/E 在 250 喂法胜出后再跑对照。
