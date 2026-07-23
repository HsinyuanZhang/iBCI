# SPINT-M2 片上部署：Few-shot 对齐开销 & 小型 decoder 可行性分析

> 定位：回答两个部署问题——(1) 仅"few-shot 对齐"部分的计算/存储开销；(2) 能否用 few-shot 调一个更小的 decoder。
> 依据：`src/models/components/spint.py`、`configs/*/falcon_m2.yaml`、`third_party/falcon_challenge/spint_decoder.py`。所有数字由 `scratchpad/cost.py` 按 M2 超参精确核算。
> M2 超参：W=50, H(model_dim)=512, T(calib)=100, C=2, dim_feedforward=2048, num_heads=64, num_id_layers=3, cross-attn 层=1；N=96（Utah array），calib M=33；解码 50 Hz（20ms bin）。

---

## 0. 一句话结论

- **对齐（IDEncoder）不是部署瓶颈**：一次性 3.75 GFLOP（随 M 线性，M=1 时仅 0.22 GFLOP）、1.13M 参数、结果只需缓存 18.8 KB。可以整体丢到片外/宿主机，在标定时算一次。
- **真正的循环开销是 cross-attention decoder**：8.34 GFLOP/s @50Hz，参数 3.47M，其中 FFN(dim_ff=2048) 独占全模型 46%。cross-attn 的变长 N + softmax 才是片上麻烦的根源。
- **参考实现有个坑**：`spint_decoder.py` 每帧都把 calib 传进 `forward()` 重算 IDEncoder → 比"缓存 E"贵 **23.5×**（196 GFLOP/s）。片上第一步就是把 E 缓存成每 session 算一次。
- **能用 few-shot 换小 decoder，而且 M2 尤其划算**：session 内 N 是固定的，置换不变/变长只为跨天迁移服务，部署时并不需要。可用冻结的 SPINT 当 teacher，对本 session 无标签数据出伪标签，闭式拟合一个 **固定 N 的线性/小 MLP student**。M2 上 Wiener oracle=0.26=SPINT held-out=0.26，**线性 decoder 本就在 SPINT 天花板上**，student 每帧仅 9600 MAC（比 SPINT 便宜 ~8700×）。

---

## 0.5 ASIC 流片视角（实验室 toy tapeout）— **主视角**

> 上面的 GFLOP/s 是 CPU/GPU 视角，**对流片不适用**。ASIC 该看：片上 SRAM 面积（存权重）、硬件敌对算子（softmax/exp/除法/rsqrt）、定点位宽、每帧周期数、是否静态 shape。约束不是吞吐（20ms/帧对硅片是天文级宽松），而是**面积与功耗**。

### 对齐（IDEncoder）：不上片

标定期算一次的东西，放片外/宿主机，结果通过 SPI/I2C 灌进片上。硅片上：
- IDEncoder 网络（1.13M 参数）→ **不上片，0 面积、0 敌对算子**。
- 身份向量 E（N×W=96×50 int8）→ 一段 **4.7 KB SRAM** + 一个加法器做 Z=X+E。

对齐在硅片上退化成"一段可下载偏置 + 加法器"，**不是流片要操心的部分**。

### 全 SPINT decoder 上片 = 一颗真加速器，不是 toy

cross-attn 的代价不在 MAC 数，在于逼你造敌对硬件块：

| 上片需要的硬件块 | ASIC 代价 |
|---|---|
| softmax over N（变长）×64 head | max 规约树 + **exp LUT** + N 次**倒数/除法** + 变长控制 |
| LayerNorm ×3 | mean/var + **rsqrt 单元** + 逐元素除 |
| FFN 512→2048→512 权重 | **2.1 MB int8 SRAM**（全模型 46%），只服务 C=2 个 query，面积利用率极差 |
| K/V 投影 512×512 | 每帧 50 MMAC，单 MAC@100MHz 要 0.83s → 得铺几百 PE 才够实时 |
| 变长 N | 变形 matmul + 动态循环控制，与固定尺寸加速器八字不合 |

合计 ~4.6 MB 权重 SRAM + softmax/LN 引擎——实验室 toy 流片吞不下。

### Toy tapeout = SPINT 蒸馏出的固定-N 线性 student

结构 = `X_{96×50} · A → 2`，本质是一个 **Wiener 滤波器**（每通道 FIR + 线性合并）。M2 的 oracle 恰是 Wiener=0.26=SPINT held-out=0.26 → 它是"**用 SPINT 伪标签标定系数的 Wiener 滤波器**"：跨天对齐由片外 SPINT 负责，片上只剩众所周知可流片的线性核。

| 指标 | Toy 线性 student | 全 SPINT decoder |
|---|---|---|
| 每帧 MAC | 9,600 | 83.5 M |
| 权重 SRAM(int8) | **9.6 KB** | ~4.6 MB |
| E 存储 | **0（吸收进 A）** | 4.7 KB + 需 IDEncoder |
| 敌对算子 | **无**（纯 MAC+累加） | softmax/exp/rsqrt/除法 |
| shape | 全静态、定尺 | 变长 N |
| 达实时最小 PE | **1 个 MAC**（96µs≪20ms，快 200×） | 数百 PE + 专用引擎 |
| 每天下载 | A，9.6 KB | 需重灌 E |

**E 不用单独存**：student 直接从原始 X 拟合到 SPINT 输出，A 已吸收 E 与 attention。每天片外重算 A、下载 9.6KB，片上纯静态 int8 线性——一个 MAC + 累加器 + 一块 ~10KB SRAM 即可。

保留非线性的退路：flatten 4800 → 16/32 隐层(ReLU) → 2，约 75–150 KB 权重，ReLU 只是 mux，仍无 softmax/LN、仍静态。

**分工**：片外（宿主/标定期）= IDEncoder 对齐 + SPINT teacher 出伪标签 + ridge 闭解 A；片上（流片）= 一个静态线性/小 MLP 核。这才是能进实验室 tapeout 的 demo 形态。

---

## 1. 参数与存储分解

| 模块 | 参数 | 说明 |
|---|---:|---|
| fc_in（readin，神经元+query 共享） | 288,768 | 每帧 |
| **fc_id_in（IDEncoder MLP1）** | 577,024 | 仅对齐时 |
| **fc_id_out（IDEncoder MLP2）** | 550,962 | 仅对齐时 |
| rep（可学习行为 query） | 100 | 常量 |
| fc_out（readout） | 25,650 | 每帧 |
| transformer（1 层 cross-attn） | 3,152,384 | 每帧；含 MHA 1.05M + **FFN 2.10M** + LN |
| **合计** | **4,594,888** | 18.38 MB fp32 / 4.59 MB int8 |

拆成两块看部署：
- **对齐网络 IDEncoder** = 1,127,986 参数（4.51 MB fp32）→ 只在标定时用一次，**可放片外**。
- **循环 decoder** = 3,466,902 参数（13.87 MB fp32）→ 每帧都要，其中 **FFN(512→2048→512) 占 2.10M（全模型 46%）**，是最该砍的块。

**每 session 需要落盘的状态**：缓存的身份向量 E ∈ ℝ^{N×W}=96×50 = **18.8 KB fp32**（比存原始 calib 特征 M×T×N=1238 KB 小 66×）。这就是"跨一天适配"要携带的全部增量。

---

## 2. Few-shot 对齐部分的开销（问题①）

对齐 = 拿新 session 的 M 条无标签 calib，过 IDEncoder 现算 E_i，**只做一次**。

| 项 | MAC | 备注 |
|---|---:|---|
| fc_id_in（M·N=3168 token，T→H→H→H） | 1.823 GMAC | 主导，**随 M 线性** |
| fc_id_out（N=96 token，H→H→H→W） | 0.053 GMAC | |
| **对齐合计（M=33）** | **1.876 GMAC ≈ 3.75 GFLOP** | 一次性 |
| 对齐合计（M=1，few-shot 下限） | ≈ 0.22 GFLOP | Fig 3A 显示 M1 单条 calib 已近饱和 |

结论：**对齐开销可忽略且不复发**。3.75 GFLOP 是"一天一次"级别，随便一颗 MCU/宿主 CPU 几十毫秒就算完；M2 若像 M1 那样用极少 calib，还能再降一个量级。存储上对齐网络 4.5MB（且可片外），缓存产物 19KB。**所以 few-shot 对齐不是片上部署的成本问题**——它恰恰是 SPINT 免梯度范式最便宜的部分。

**必修的实现优化**：当前 `spint_decoder.py:predict()` 每帧把 calib 传进 `forward()`，等于每 20ms 重算一遍上面 3.75 GFLOP：

| 方案 | 每帧 | @50Hz |
|---|---:|---:|
| 现状（每帧重算 IDEncoder） | 3.92 GFLOP | 196 GFLOP/s |
| 缓存 E（每 session 算一次） | 0.167 GFLOP | 8.34 GFLOP/s |

**23.5× 免费加速**，纯工程改动、不改数值结果。片上部署第一步。

---

## 3. 循环 decoder 的开销 & 为什么 cross-attn 片上麻烦

每帧（缓存 E 后）：

| 项 | MMAC | 备注 |
|---|---:|---|
| fc_in（N 个神经元 token） | 27.62 | 随 N 线性 |
| cross-attn | 51.58 | **K/V 投影 over N 神经元 = 50.3**，每帧重算（src 变） |
| ffn（C=2 个 query） | 4.19 | |
| fc_out | 0.05 | |
| **每帧合计** | **83.45 MMAC ≈ 0.167 GFLOP** | @50Hz = **8.34 GFLOP/s** |

片上麻烦点，不在算力（8 GFLOP/s 不大），而在 **结构**：
1. **变长 N**：token 数=神经元数，跨 session 变（96 只是 M2 当前值）→ 变形 matmul + 变长 softmax，对固定尺寸的加速器/展开流水线不友好。
2. **softmax over 神经元轴** + **64 head**（head_dim=8）：softmax 的 exp/规约在定点/近似硬件上代价高、精度敏感。
3. **FFN 2048 宽**：占 46% 参数与近半算力，但对 C=2 的 query 其实是杀鸡用牛刀。

**但关键观察**：置换不变 + 变长 N 只服务于"**跨 session 免重训迁移**"。**一旦部署到某个具体 session，N 是固定的、E 是固定的、神经元顺序也固定**。所以 session 内推理根本不需要 cross-attn 的那套通用机制——它是"编译期"能力，不是"运行期"必需。这直接引出问题②。

---

## 4. 用 few-shot 调一个更小的 decoder（问题②）

三条路线，按推荐度排序。

### 路线 A（**推荐，M2 尤其合适**）：SPINT 当 teacher，逐 session 蒸馏出固定-N 小 student

思路：SPINT 测试期虽无标签，但它**会输出运动学预测**，可当伪标签。
1. 片外/标定期：冻结 SPINT，在本 session 的 calib（及可用的开跑数据）上跑一遍 → 得到 (神经窗口 X_{N×W}, SPINT 预测 Ŷ_{C}) 配对。
2. **闭式拟合**一个固定 N 的线性 student：Ŷ = X·A，ridge 闭解，无梯度、无 backward（契合 SPINT 免梯度精神）。想要非线性就用一层小 MLP，几十步即可。
3. 片上只跑 student：N×W→C 线性 = **9,600 MAC/帧**，比 SPINT 每帧便宜 **~8700×**，且是规整定尺 matmul，无 softmax、无变长。

为什么 M2 特别值：**Table 1 里 M2 的 Wiener Filter oracle = 0.26 = SPINT held-out = 0.26**。也就是说 M2 上一个线性读出已经顶到 SPINT 的天花板——蒸出来的线性 student **有望无损匹配** SPINT，而片上成本近乎为零。跨天迁移仍由 SPINT（片外、每天一次）负责：新的一天 → SPINT 重算 E、重出伪标签 → 重拟合一个新的 per-session A。等于 **"SPINT 做跨天对齐 + 廉价线性做片上实时"** 的分工。

> 待验证（诚实标注）：student 质量上限=teacher 质量；SPINT 在部署 session 上的 within-session R²=0.59（held-in），作为 teacher 应该够好，但"伪标签蒸馏能否在 M2 held-out 复现 0.26"必须实测。这是一个明确、便宜的实验，不是既成结论。

### 路线 B：直接瘦身 cross-attn（保留非线性/跨天鲁棒，但仍是 attn）

- model_dim 512→128、dim_feedforward 2048→256（FFN 占 46% 参数，收益最大）、heads 64→32（消融 A5：32 head 最优）。
- 预计参数从 4.6M 砍到 <0.5M，每帧算力同比例下降。
- 缺点：softmax/变长 N 的硬件麻烦仍在；需要重训 + 重跑 Fig3/Table1 验掉点多少。适合"想保留 SPINT 全部跨 session 能力、只是嫌大"的场景。

### 路线 C：few-shot 直接调 student 参数（无 teacher）

- 用 calib 的少量数据 + 少量梯度步微调一个小 decoder。但 SPINT 场景测试期**无标签**，纯 few-shot 调参缺监督信号——除非引入自监督目标（论文局限里也提到 IDEncoder 未来可自监督解耦）。比路线 A 复杂、收益不明，**不推荐**作为首选。

---

## 5. 建议的部署形态（综合）

```
┌── 片外 / 宿主机，每天/每 session 一次（标定期） ──────────────┐
│  M 条无标签 calib                                              │
│    → IDEncoder(1.13M, 3.75 GFLOP) → E (N×W, 18.8 KB)  ────┐   │
│    → 冻结 SPINT 出伪标签 → ridge 闭解 → per-session A ─────┼─┐ │
└──────────────────────────────────────────────────────────┘ │ │
                                                              ▼ ▼
┌── 片上，每帧 20ms（50Hz） ─────────────────────────────────────┐
│  方案①(保稳): 缓存 E 的瘦身 cross-attn  ~低GFLOP/s，仍有 softmax │
│  方案②(极简, 荐 M2): student  X_{N×W}·A → C   仅 9.6 KMAC/帧     │
└───────────────────────────────────────────────────────────────┘
```

**落地顺序（低风险 → 高收益）**：
1. 先做 §2 的缓存-E 修复（23.5× 免费，纯工程，先在 minival 验数值不变）。
2. 复现 M2 held-out 0.26 基线（多 seed，方差 ±0.13）。
3. 跑路线 A 的蒸馏实验：SPINT 伪标签 → per-session ridge student → 在 M2 held-out 比 R²。若 ≈0.26 即拿下"近零成本片上 decoder"。
4. 若路线 A 掉点，退路线 B 瘦身 attn；把两者与 FALCON 侧 Ridge/LSTM/LoRA/RLS 同口径并表。

---

## 6.5 已验证：few-shot 在 ridge 上也成立（FALCON m2-research 实跑）

用 `FALCON/m2-research/scripts/15_ridge_fewshot_curve.py`（冻结 ridge W0，N_HIST=7，held-out calib 上做输入侧对齐，6 sessions 均值）实测：

| adapt arm | 类型 | 4s | 8s | 16s | 32s | 41s(full) |
|---|---|---:|---:|---:|---:|---:|
| coral_diag | **无监督** | 0.105 | 0.114 | 0.114 | 0.114 | 0.114 |
| fmllr_diag | 有监督 | 0.102 | 0.118 | 0.132 | 0.144 | 0.147 |
| fmllr_lowrank_r4 | 有监督 | — | 0.122 | 0.135 | 0.146 | 0.150 |
| retrain_ridge | 有监督重训 | — | — | 0.051 | 0.081 | 0.097 |

参照：static(不适配)=0.049 · SPINT=0.26 · Wiener/held-in LOO oracle=0.268。

结论：
1. **few-shot 有效且下限极低**：无监督 CORAL 仅 4s calib→0.105、8s 饱和 0.114（对应 SPINT Fig 3A 的"单 trial 近饱和"）。
2. **别重训 decoder**：retrain_ridge 数据饥渴、上限仅 0.097 < 冻结+对齐——硬证 SPINT 主张，片上不需可训练权重。
3. **有监督多花 calib 换 ~0.03**（0.11→0.15），仍够不到 0.26。
4. **结构同构**：冻结 ridge + 每通道仿射(CORAL/fMLLR) ≡ SPINT(冻结 cross-attn + 加性身份 E)；**CORAL 的每通道 (scale,shift) = SPINT 身份 E 的线性版**。SPINT 用非线性 IDEncoder 把天花板从 0.11 抬到 0.26。

**0.11→0.26 缺口怎么填（两条，均 ASIC-friendly）**：
- **ridge→时间轴卷积**：n_hist=7 的 flat 768-ridge 换成 per-channel depthwise 时间卷积(共享平滑 taps)+线性混合，减过拟合、抬 R²；depthwise 时间卷积在硅上极便宜。
- **SPINT 蒸馏**：用 SPINT(0.26) 当无监督 teacher 出伪标签标定 ridge/conv 系数，把无监督天花板从 0.11 拉向 0.26，且片上仍只跑线性/卷积核。桥接两个项目。

产物：`m2-research/outputs/results/ridge_fewshot_curve.csv`、`outputs/figures/15_ridge_fewshot_curve.png`。

---

## 7. 关键数字速查

| 量 | 值 |
|---|---|
| 全模型参数 / 大小 | 4.59M / 18.4 MB fp32 / 4.6 MB int8 |
| 对齐网络 IDEncoder | 1.13M 参数，3.75 GFLOP/次（M=33），0.22 GFLOP（M=1） |
| 缓存身份 E | 96×50 = 18.8 KB/session |
| 循环 decoder | 3.47M 参数，0.167 GFLOP/帧，8.34 GFLOP/s@50Hz |
| FFN 占比 | 全模型参数的 46% |
| 参考实现每帧重算 IDEncoder 的浪费 | 23.5×（196 GFLOP/s） |
| 蒸馏线性 student | 9,600 MAC/帧，≈8700× 便宜于 SPINT 帧 |
| M2 天花板参照 | Wiener oracle = SPINT held-out = 0.26（线性即达顶） |
