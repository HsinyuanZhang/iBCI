# SPINT vs NDT2 vs UniBCI: 架构深度对比与融合分析

> **文档状态：实验前架构假设与文献背景。** 后续实验已否定“B3 权重可直接 MUA→SUA 零样本迁移”和“简单 normalization 可弥合差异”两项乐观假设。当前结论见 [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md)，执行计划见 [`../ROADMAP.md`](../ROADMAP.md)。

## 1. 架构概览

### SPINT (本工作基础)

```
Calibration trials [B, M, T, N]
        │
        ▼
┌─────────────────────────┐
│   ID Encoder (teacher)  │
│  fc_id_in: T → H → H   │  ← per-neuron, per-trial
│  mean over M trials     │  ← cross-trial aggregation
│  fc_id_out: H → H → W  │  ← per-neuron
└─────────────────────────┘
        │
        ▼
  Identity E [B, N, W]
        │
        ▼
  neural + E [B, N, W]
        │
        ▼
┌─────────────────────────┐
│   Online Decoder        │
│  fc_in: W → H → H      │  ← per-neuron
│  Transformer cross-attn │  ← over N neurons
│  fc_out: H → W          │  ← per-covariate
└─────────────────────────┘
        │
        ▼
  Behavior [B, W, C]
```

**B3 Student Encoder** (18K params):
```
pre_pool:  Linear(T → D) → ReLU     ← per-neuron, per-trial
cross-trial: mean over M
post_pool: Linear(D → D) → ReLU → Linear(D → D) → ReLU → Linear(D → W)
```

**关键特性**:
- ID encoder 是 **per-neuron 独立** 的（不假设 neuron 间关系）
- Identity 是 **加性** 的：`src = neural + E`
- 没有 context embedding（每个 session 独立）
- 没有预训练（teacher-student distillation）

### NDT2 (NeurIPS 2023)

```
Binned spikes [B, T, N]
        │
        ▼
┌─────────────────────────┐
│   Patching              │
│  Group neurons → patches│  ← 空间分组
│  Temporal patching      │  ← 时间分组
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Context Embedding     │
│  session/subject/task   │  ← learned context vectors
│  + position embedding   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Transformer Encoder   │
│  Spatiotemporal attention│
│  (masked autoencoding)  │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Asymmetric Decoder    │
│  Only decode masked bins│
└─────────────────────────┘
        │
        ▼
  Reconstructed spikes / Behavior
```

**关键特性**:
- **Masked autoencoding** 预训练（自监督）
- **Context embedding** 区分不同 session/subject/task
- **Patching**：将 neurons 分组处理（假设空间局部性）
- 明确比较了 **sorted vs unsorted** data
- Multi-context pretraining 跨 session/subject/task 有效

### UniBCI (arXiv 2026)

```
Raw spikes [T_raw, C_raw]
        │
        ▼
┌─────────────────────────┐
│   Spike Train Norm      │
│  Bin → group by area    │  ← 统一时间分辨率
│  Pad/truncate to C_norm │  ← 统一通道数
└─────────────────────────┘
        │
        ▼
  X_norm [T_norm, A, C_norm]
        │
        ▼
┌─────────────────────────┐
│   CST Tokenization      │
│  Channel embed: C→d     │
│  Metadata embed (NLP)   │  ← species/dataset/region/task/session
│  Spatiotemporal pos     │
│  token = X ⊕ meta ⊕ pos│
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   IAA Blocks (×L)       │
│  ILA: temporal attention│  ← within interval
│  ASWA: spatial attention│  ← across areas
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Masked Reconstruction │
│  Predict masked tokens  │
└─────────────────────────┘
```

**关键特性**:
- **直接在同一个模型中混合 SUA + MUA**（Table 1）
- **Context-conditioned tokenization**：metadata 编码为 context
- **Area grouping**：将 channels 按脑区分组
- **Spike train normalization**：统一不同数据集的格式
- Foundation model 思路（大规模预训练）

## 2. 核心差异对比

| 维度 | SPINT | NDT2 | UniBCI |
|------|-------|------|--------|
| **目标** | Few-shot calibration | Multi-context pretraining | Foundation model |
| **ID 机制** | 显式 identity encoder | Context embedding | Context-conditioned token |
| **Neuron 处理** | Per-neuron 独立 | Patching（分组） | Area grouping（按脑区） |
| **预训练** | 无（teacher-student） | Masked autoencoding | Masked reconstruction |
| **信号类型** | MUA only | Sorted + unsorted | SUA + MUA mixed |
| **Context** | 无 | Session/subject/task | Species/dataset/region/task/session |
| **参数量** | ~18K (B3) | ~3M | ~3M+ |
| **硬件友好** | ✓ (设计目标) | ✗ | ✗ |

## 3. 关键洞察

### 3.1 SPINT 的 Per-Neuron 独立性是优势

SPINT 的 ID encoder 对每个 neuron **完全独立** 操作：
- `fc_id_in`: 每个 neuron 的 calibration trial 独立编码
- `mean over M`: 每个 neuron 独立平均
- `fc_id_out`: 每个 neuron 独立投影

**这意味着**：
- 不假设 neuron 间的空间关系
- 不依赖 neuron 数量 N
- 不依赖信号类型（SUA/MUA）

**对比 NDT2/UniBCI**：
- NDT2 的 patching 假设相邻 neurons 有相关性
- UniBCI 的 area grouping 假设同一脑区的 neurons 有共性
- 这些假设在 SUA 和 MUA 混合时可能不成立（SUA 的 "neuron" 是 sorted cluster，MUA 的 "neuron" 是 electrode）

**结论**：SPINT 的 per-neuron 独立性天然适合 SUA/MUA 共享。

### 3.2 Identity 的本质：时间动态模式

SPINT 的 ID encoder 学到的是什么？

```
Input:  calibration trial 的 spike count 轨迹 [T]
Output: identity embedding [W]
```

它学的是 **"这个 neuron 在 calibration 期间的活动模式 → 它是谁"**。

对于 SUA：
- 活动模式 = 单个神经元的发放模式（稀疏，低发放率）
- Identity = "这是哪个 sorted cluster"

对于 MUA：
- 活动模式 = 电极上所有神经元的叠加活动（密集，高发放率）
- Identity = "这是哪个 electrode"

**关键问题**：这两种 "活动模式 → identity" 的映射是否相似？

**假设**：是相似的。因为：
1. 都是 spike count 轨迹（只是量级不同）
2. 都包含 "这个 neuron/cluster 对任务的响应特征"
3. 时间动态结构（task-modulated firing pattern）是共通的

### 3.3 NDT2 的 Context Embedding 可以借鉴

NDT2 用 learned context embedding 区分不同数据来源：
```python
context_embed = nn.Embedding(num_contexts, d_model)
token = patch_embed + context_embed[context_id] + position_embed
```

**对 SPINT 的启发**：
- 给 B3 encoder 加一个 `signal_type` context：`{SUA, MUA}`
- 或者更细粒度：`{dataset_id, signal_type}`
- 让 encoder 知道 "我现在处理的是 SUA 还是 MUA"

**但注意**：SPINT 的 B3 是 per-neuron 独立的，context embedding 需要广播到每个 neuron。

### 3.4 UniBCI 的 Spike Train Normalization 是必要的

UniBCI  explicitly 做了 spike train normalization：
```
X_norm[t, a] = [X[t, c1] || X[t, c2] || ... || X[t, c|C_a|]]
```
然后 pad/truncate 到固定维度。

**对 SUA/MUA 的意义**：
- SUA 发放率：~1-20 Hz（稀疏）
- MUA 发放率：~10-50 Hz（密集）
- 如果不 normalize，encoder 会偏向 MUA（信号更强）

**建议**：在 B3 的 pre_pool 之前加 per-neuron normalization：
```python
# Option 1: Z-score per neuron
x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

# Option 2: Max normalization
x = x / (x.max(dim=-1, keepdim=True)[0] + 1e-6)

# Option 3: Log transform (for spike counts)
x = torch.log1p(x)
```

### 3.5 Teacher-Student vs Self-Supervised

| | SPINT | NDT2/UniBCI |
|--|-------|-------------|
| 训练方式 | Teacher-student distillation | Self-supervised (masked reconstruction) |
| 需要 teacher? | 是 | 否 |
| 需要 labeled data? | 是（teacher 的 identity） | 否 |
| 适用场景 | 有 full SPINT teacher 时 | 无 teacher，只有 raw spikes |

**对 SUA 的意义**：
- 如果我们没有 SUA 的 full SPINT teacher，就不能直接用 teacher-student
- 但可以用 self-supervised pretraining（NDT2/UniBCI 方式）先学通用表示
- 然后再 fine-tune 到 identity estimation

**或者**：先在 MUA (FALCON) 上训 teacher-student，然后 zero-shot/fine-tune 到 SUA。

## 4. 融合方案

### 方案 A: 最小改动（推荐先试）

在 B3 基础上加两个改动：

1. **Input normalization**（来自 UniBCI）:
```python
class NormalizedEarlyPoolEncoder(EarlyPoolEncoder):
    def __init__(self, ..., norm_mode='zscore'):
        ...
        self.norm_mode = norm_mode
    
    def _normalize(self, x):
        # x: [B, N, T]
        if self.norm_mode == 'zscore':
            return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        elif self.norm_mode == 'log':
            return torch.log1p(x)
        return x
    
    def push_trial(self, state, trial, ...):
        trial = self._normalize(trial)
        return super().push_trial(state, trial, ...)
```

2. **Signal type context**（来自 NDT2）:
```python
class ContextEarlyPoolEncoder(EarlyPoolEncoder):
    def __init__(self, ..., num_contexts=2):
        ...
        self.context_embed = nn.Embedding(num_contexts, hidden_dim)
    
    def finalize_identity(self, state, context_id=None):
        feat = state["sum_feat"] / state["trial_count"]  # [B, N, D]
        if context_id is not None:
            ctx = self.context_embed(context_id)  # [D]
            feat = feat + ctx.unsqueeze(0).unsqueeze(0)  # broadcast
        return self.post_pool(feat)
```

### 方案 B: 中等改动

在方案 A 基础上，加 **multi-dataset pretraining**：

1. 在 FALCON M2 (MUA) + MC_Maze (SUA) 上联合训练 B3
2. 用 dataset_id 作为 context
3. 验证：联合训练的 encoder 是否在两个数据集上都有效

### 方案 C: 大改动（如果 A/B 有效再考虑）

借鉴 NDT2 的 masked autoencoding：
1. 在大量 unlabeled spike data 上预训练 B3（self-supervised）
2. 然后 fine-tune 到 identity estimation
3. 这需要重写训练流程，工作量大

## 5. 实验优先级

| 优先级 | 实验 | 目的 | 工作量 |
|--------|------|------|--------|
| P0 | B3 on MC_Maze (no modification) | 验证 B3 能否直接处理 SUA | 小 |
| P1 | B3 + normalization on MC_Maze | 验证 normalization 是否必要 | 小 |
| P2 | Zero-shot: FALCON M2 B3 → MC_Maze | 验证跨信号类型迁移 | 中 |
| P3 | B3 + context on MUA+SUA joint | 验证 context embedding 是否有效 | 中 |
| P4 | Self-supervised pretraining | 验证无 teacher 场景 | 大 |

## 6. 预期结果与风险

### 乐观预期
- B3 的 per-neuron 独立性使其天然适合 SUA/MUA 共享
- Normalization 解决发放率差异后，zero-shot 迁移衰减 < 10%
- 联合训练进一步提升两个数据集的性能

### 风险
- SUA 太稀疏（很多 bin 为 0），B3 的 Linear(T→D) 可能学不到有用特征
- SUA 的 N 通常 > MUA（182 vs 96），identity space 维度不同
- Sorting 的不稳定性使 SUA 的 "identity" 本身就不一致

### 缓解措施
- 如果 SUA 太稀疏：增加 T（用更长的 trial），或用 firing rate 代替 spike count
- 如果 N 不匹配：B3 不依赖 N，应该没问题
- 如果 sorting 不稳定：这正好是 SPINT 要解决的问题（identity uncertainty）

## 7. 与现有工作的定位

| 工作 | 贡献 | 我们的区别 |
|------|------|-----------|
| SPINT | Few-shot identity encoder for MUA | 我们扩展到 SUA |
| NDT2 | Multi-context pretraining | 我们做轻量 encoder，不做 foundation model |
| UniBCI | Unified SUA+MUA foundation model | 我们验证 18K 参数的小模型也能迁移 |
| FALCON | MUA benchmark | 我们加入 SUA benchmark (MC_Maze) |

**我们的独特故事**：
> "UniBCI 证明了大模型可以统一 SUA/MUA。我们证明，**18K 参数的轻量 identity encoder 也能做到跨信号类型迁移**——这对资源受限的硬件部署（ASIC/嵌入式）至关重要。"
