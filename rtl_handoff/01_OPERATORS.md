# 01 — 要实现的算子与整数合同

范围：**第一版 = B3 EarlyPool encoder（W8A8）**。decoder 算子在文末列为 `DEFERRED`，仅供预留接口，勿现在写死 RTL。

> 完整部署目标是 SPINT Dim256/TopK64 A8W8 网络，其算子清单（W8A16 ID 分支、A8W8 的 fc_in/Q/K/V/O/FFN、P_FP 的 LN/softmax/fc_out）与参数化尺寸见 `05_SPINT_DIM256_SPEC.md`。本文的 B3 是它最简、已冻结的子情形。

整数运算语义以 `golden/b3_quant_engine.py` 为准；本文是它的可读规格。

---

## 1. 算子清单（B3，全部 `FROZEN`）

| # | 算子 | 输入→输出 | 关键数值格式 |
|---|---|---|---|
| OP1 | INT8×INT8 仿射内积 | `a[in] · W[out,in] + bias` | acc **INT32**，bias 在 INT32 acc 域 |
| OP2 | Requant + 可选 ReLU | INT32 acc → INT8 | `acc(i32) × mult(i32)` → **INT64** → round → 算术右移 → clamp |
| OP3 | DeepSets 累加 | `SUM_feat[N,D] += feat[m]` | **INT32** 累加 |
| OP4 | Reciprocal mean | `SUM×recip >>> shift` | 中间 **INT64**，输出再量化 INT8 |

只有这四个算子 + 逐层布线即可拼出完整 B3。**没有除法器、没有 FP32 datapath、没有 per-PE 的 exp/rsqrt。**

---

## 2. 逐层布局（PyTorch `y = x @ Wᵀ + b`，W 为 `[out,in]`）

| 层 | 权重 | 输入 | 输出 | 激活 |
|---|---|---|---|---|
| pre_pool | `[D,T]`=`[64,100]` | `[T]` | `[D]` | ReLU |
| post0 | `[D,D]`=`[64,64]` | `[D]` | `[D]` | ReLU |
| post1 | `[D,D]`=`[64,64]` | `[D]` | `[D]` | ReLU |
| post2 | `[W,D]`=`[50,64]` | `[D]` | `[W]` | 无 |

数据流（每 session 一次）：

```text
for m in M, for n in N:  feat[m,n] = ReLU(pre_pool(x[m,:,n]))   # neuron 独立、权重共享
SUM_feat[n]  = Σ_m feat[m,n]        # INT32
mean[n]      = SUM_feat[n] * recip >>> recip_shift  → 再量化 INT8
E[n]         = post2(ReLU(post1(ReLU(post0(mean[n])))))         # INT8, [N,50]
```

**不变量**：ReLU 在 mean 之前，不能先 mean 再 pre_pool；trial 顺序置换不改 `E`；同套权重对所有 neuron 共享；`N` 只改循环边界/mask，不改权重布局。

---

## 3. 数值合同（逐 edge）

| Edge | 格式 |
|---|---|
| calibration input | signed INT8 `[M,T,N]` |
| weight | signed INT8，per-output-channel symmetric scale |
| bias | INT32（accumulator 域） |
| dot accumulator | INT32（当前工作点无 overflow，`K≤100`） |
| activation（每 edge） | shared per-tensor INT8 scale |
| `SUM_feat` | INT32 |
| reciprocal product | INT64 |
| mean 输出 | INT8 |
| requant multiplier | per-output-channel INT32，默认 shift=31 |
| requant product | INT64（`acc_i32 × mult_i32`） |
| E | signed INT8 + 标量 `E_scale` |

**Rounding（OP2/OP4 统一）**：

```text
rounded = (product + (1 << (shift-1))) >>> shift     # 算术右移，与软件一致
if relu: rounded = max(rounded, 0)
q = clip(rounded, qmin, qmax)                         # INT8: [-128,127] 或 [0,127]
```

Reciprocal mean：`recip = round(2^recip_shift / M)`，默认 `recip_shift=20`；固定 `M` 时可把 `1/M` 融进 post0 requant，多 `M` 用小 reciprocal LUT，**不要通用除法器**。

---

## 4. PE 阵列映射（做 B3 够用）

- **Mode A `8T×8O`**：8 token × 8 output 并行，每 cycle 读 8 activation + 8 weight。B3 默认（weight 被 8 token 复用）。
- **Mode B `1T×64O`**：单 token 广播、64 weight/cycle。用于 `64→64` bring-up 和小 token 路径。
- 后处理（requant）**共享** 8/16 lane，不必与 64 PE 一一对应。
- 双 accumulator bank：A 算 / B requant+writeback，交换。

---

## 5. RTL 实现顺序（建议里程碑）

1. `1×8` tiny dot-product（OP1 最小体）
2. 扩为 `1×64`
3. INT32 accumulator overflow assertion
4. 8/16-lane requant（OP2：INT64 product + round + shift + clamp + ReLU）
5. **B3 tiny golden 对齐**（`--profile tiny`，逐 stage exact）
6. `8×8` Mode A broadcast mapping
7. banked SRAM address generator
8. **B3 full-shape**（`T=100,D=64,W=50,N=96,M=33`）
9. reduction tree split（OP3/OP4：SUM + reciprocal mean）
10. （DEFERRED）decoder affine microkernel
11. （DEFERRED）NLU（LN/softmax）
12. （DEFERRED）full compiled decoder

里程碑 1–9 完成即达成「B3 tiny/full bit-exact RTL」这一系统验收门的第一项。

---

## 6. DEFERRED：decoder 算子（勿现在写死）

在线主路径：`X+E → read-in MLP → LN1 → 1-layer cross-attention → FFN → 末步读出`。这些算子的位宽、`X+E` 边界（shared INT8 / INT16 add / 分别 requant 三候选）、LN（sumsq 位宽、rsqrt LUT）、softmax（logit/exp LUT/denominator/reciprocal）都还是 `OPEN`，必须先由软件侧 task-R² 实验决定。RTL 侧现在只做参数化接口预留，见 `hardware_pe_sram/05,06`。
