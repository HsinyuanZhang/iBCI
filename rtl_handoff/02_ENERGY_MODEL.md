# 02 — 整体能效计算方法

这台机器不是 EDA 服务器，因此这里给的是 **pre-PDK 的解析（bottom-up）能效方法**：可用纯 Python 计算、可解释、可逐项审计。它**不是**综合/布局后功耗，任何 mW 数字在拿到真实 SRAM macro + PDK 系数前都不得当作可信芯片功耗上报。

工具：`tools/energy_model.py`（可跑，系数为占位）。

---

## 1. 能量分解

对每个执行阶段（session 或 frame）：

```text
E_total = E_mac + E_sram + E_leak

E_mac  = mac_count      × e_mac_pj          # 逻辑 MAC 能量
E_sram = reads × e_rd_pj + writes × e_wr_pj # 存储访问能量（按逻辑字）
E_leak = leak_mw        × active_ms × 1e6   # 漏电 (mW·ms = 1e6 pJ)
```

- `e_mac_pj`：单次 INT8×INT8 MAC 能量（来自 PDK/文献）。
- `e_rd_pj / e_wr_pj`：单次 SRAM 读/写访问能量（来自 SRAM compiler 数据表，按 macro 位宽/字宽）。
- `leak_mw`：阵列 + SRAM 上电时的总漏电。
- `active_ms = ideal_cycles / (freq_MHz × 1e3)`，`ideal_cycles = mac / PE`。

---

## 2. 必须 session / frame 分开

不能把 session adaptation 的 MAC 摊到 frame 上掩盖峰值，也不能用 session average 弱化 frame 主项。

| 指标 | Session-rate（每 session 一次） | Frame-rate（每 20 ms 一帧） |
|---|---|---|
| 报告量 | latency/session、energy/session、cycles vs M/N、encoder 利用率、peak state | latency/frame、energy/frame、50 Hz 平均功耗、worst-shape、deadline margin |
| 主导成本 | coefficient SRAM、`SUM_feat`、可编程性 | `H×H` projection + FFN 权重访存、LN/softmax |

**Duty-cycle 加权平均功耗**（frame 路径）：

```text
avg_power = E_frame(J) × frame_hz          # 默认 frame_hz = 50
```

session 能量按「每 session 发生一次」单独摊，不进 frame 平均。

---

## 3. 输入数据从哪来

| 量 | 来源 |
|---|---|
| `mac_count` | 解析公式（见下）或 cycle model 输出 |
| `reads/writes` | cycle model 的 per-block 访问计数（`hardware_pe_sram/07 §7`） |
| `ideal_cycles` | `mac / PE`（100 MHz / 64 或 128 PE） |
| `e_mac_pj / e_rd_pj / e_wr_pj / leak_mw` | **PDK / SRAM compiler 数据表**（替换占位系数） |

B3 encoder MAC：`M·N·(T·D) + N·(2·D² + D·W)` → M2 约 **21.4M/session**。  
Decoder（cached-query baseline）约 **82.87M/frame**；静态-query 编译候选约 **63.90M/frame**。

---

## 4. 100 MHz 理想时间参考（利用率=100% 上界）

| PE | baseline 82.87M | compiled 63.90M |
|---:|---:|---:|
| 32 | 25.90 ms | 19.97 ms |
| 64 | 12.95 ms | 9.98 ms |
| 128 | 6.47 ms | 4.99 ms |

结论：32 PE 连理想情况都难覆盖 20 ms；**64 PE 是最小合理起点**。真实时间要用 cycle model 把利用率、bank conflict、NLU 周期、tail 都算进去。

---

## 5. 精度分级（`07_VERIFICATION_PPA §9`）

| 阶段 | 允许上报 | 不得上报 |
|---|---|---|
| Pre-PDK / analytic（本方法） | MAC、bytes、read/write count、理论 cycles、利用率、带宽、live state | 可信 mW/mm² |
| SRAM compiler 后 | + macro 面积、读写能量、leakage、频率、bank/port/ECC overhead | — |
| RTL 综合后 | + 逻辑面积、关键路径、PE/NLU/controller 拆分、重构互连 overhead | — |
| Post-layout | + 线负载、真实时钟、activity 动态功耗 | — |

---

## 6. 架构公平对照（论文需要，能效方法一致套用）

同一 process / 频率 / SRAM 容量 / 位宽 / 模型 / workload / activity / 时序约束下比较：

```text
A. unified reconfigurable 64-PE
B. fixed 8×8-only
C. fixed 1×64-only
D. two dedicated engines
E. optional 128-PE unified
```

报告：logic+SRAM 面积、重构 overhead、energy/session、energy/frame、leakage、利用率、latency。**只比 TOPS/W 不足以支撑可重构主张。**

---

## 7. 使用示例

```bash
cd tools
python3 energy_model.py --preset b3_m2                 # session 路径
python3 energy_model.py --preset decoder_m2_baseline   # frame 路径 + 50Hz 平均功耗
python3 energy_model.py --preset decoder_m2_compiled --coeffs my_pdk.json
```

先用你 PDK/SRAM 数据表写一个 `my_pdk.json` 覆盖占位系数，再解读输出。
