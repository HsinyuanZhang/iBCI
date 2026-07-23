# SUA-MUA Shared Encoder Exploration — 初始计划

> **文档状态：历史假设与初始实验设计。** Phase 0 复选框已不代表实际进度。当前入口见 [`README.md`](README.md)，当前执行计划见 [`ROADMAP.md`](ROADMAP.md)，当前口径化结果见 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md)。本文件保留用于追溯最初假设。

## 核心假设

SUA (sorted single-unit activity) 和 MUA (multi-unit threshold crossing) 可以共享 encoder 层，因为：

1. **输入格式相同**：都是 `[B, M, T, N]` 的 binned spike counts
2. **本质问题相同**：都是 neuron identity 的不确定性
   - MUA：ID 不精确（电极混合多个神经元）
   - SUA：ID 不稳定（sorting 跨 session 漂移）
3. **B3 encoder 架构无关**：per-neuron 独立操作，不依赖 N 或信号类型

## 文献支持（已读论文关键发现）

### NDT2 (Ye et al., NeurIPS 2023)
- **明确比较了 sorted vs unsorted data**（Figure 3A vs 3B）
- Multi-context pretraining 跨 session/subject/task 有效
- 使用 **context embedding** 区分不同数据来源
- 使用 **stitching/padding** 处理不同 session 间 neuron 维度不一致
- **启示**：如果跨 session/subject/task 预训练有效，跨 signal type (SUA/MUA) 也应该可行

### UniBCI (Hong et al., arXiv 2026)
- **直接在同一个模型中混合 SUA 和 MUA 数据集**（Table 1）
  - Pretraining: M1-CO1 (sorted), M1-CO2 (MUA), Pac-Man (sorted), LICK (MUA)
  - Downstream: MC-Maze (sorted), Perich (sorted), PPC-FINGER (sorted)
- 使用 **context-conditioned tokenization**：将 species/dataset/subject/region/task/session 编码为 context
- **Spike train normalization**：统一不同时间分辨率和通道数
- **直接证明 SUA 和 MUA 可以联合训练**
- **MC-Maze 是他们的 downstream 数据集之一**（与我们下载的数据集相同）

### 对我们工作的意义
1. UniBCI 已经证明 SUA+MUA 联合训练可行 → 我们的方向有文献支持
2. NDT2 的 context embedding 思路可以借鉴 → 给 B3 加 signal_type context
3. 但 UniBCI 是大型 foundation model，我们是轻量 B3 encoder → 需要验证小模型是否也能迁移
4. **我们的独特贡献**：不是做 foundation model，而是验证轻量 identity encoder 的跨信号类型迁移性

## 数据集

### 已下载 / 下载中

| 数据集 | DANDI | 大小 | Units | 脑区 | 任务 | 状态 |
|--------|-------|------|-------|------|------|------|
| MC_Maze | 000128 | 694 MB | ~182 sorted | M1 + PMd | 延迟到达（有障碍） | 下载中 |
| MC_RTT | 000129 | 51 MB | ~100+ sorted | M1 | 自由到达（随机目标） | 下载中 |

### 已有（MUA baseline）

| 数据集 | DANDI | Units | 脑区 | 任务 |
|--------|-------|-------|------|------|
| FALCON M2 | 000953 | 96 MUA | M1 | 手指运动 |
| FALCON M1 | 000941 | 64 MUA | M1 | 伸手抓握 |
| FALCON H1 | 000954 | 176 MUA | M1 (human) | 7DoF 到达抓握 |

### 候选（后续）

| 数据集 | 来源 | Units | 备注 |
|--------|------|-------|------|
| MC_Maze_Large | DANDI 000138 (149 MB) | 更多 units | MC_Maze 扩展版 |
| Perich 2018 | CRCNS pmd-1 | 3,139 sorted | 最大 SUA 数据集 |
| M1-CO1 | UniBCI paper | 232 sorted | 8方向 center-out |

## 实验计划

### Phase 0: 数据探索（当前）
- [ ] 检查 MC_Maze NWB 格式：units 表、spike times、行为数据
- [ ] 确认 sorted SUA：是否有 waveform_mean、electrode 映射
- [ ] 统计：N units、T trial length、M trials、行为维度
- [ ] 与 FALCON M2 对比数据格式差异

### Phase 1: MC_Maze Teacher 训练
- [ ] 写 MC_Maze datamodule（适配 SPINT 接口）
- [ ] 训练 SPINT teacher on MC_Maze（sorted SUA）
- [ ] 验证 teacher heldout R²

### Phase 2: B3 Student on SUA
- [ ] 训练 B3 student on MC_Maze（同 FALCON 流程）
- [ ] 评估 heldout R²
- [ ] 与 FALCON M2 的 B3 结果对比

### Phase 3: 零样本迁移（核心实验）
- [ ] FALCON M2 训练的 B3 → 直接在 MC_Maze 上提取 identity → 评估
- [ ] MC_Maze 训练的 B3 → 直接在 FALCON M2 上提取 identity → 评估
- [ ] 分析：迁移是否有效？衰减多少？

### Phase 4: 联合训练 / 微调
- [ ] MUA + SUA 联合训练 B3
- [ ] MUA pretrain → SUA fine-tune
- [ ] SUA pretrain → MUA fine-tune

### Phase 5: 分析
- [ ] 可视化：SUA vs MUA 的 identity embedding 分布
- [ ] 分析：encoder 学到了什么（发放率？时间动态？）
- [ ] 消融：T、D、normalization 对迁移的影响

## 关键科学问题

1. B3 encoder 学到的是 "电极/cluster 特异性" 还是 "神经元活动模式的通用特征"？
2. SUA 的稀疏性（低发放率）是否需要不同的 normalization？
3. N 的变化（SUA 通常 N > MUA）是否影响 identity embedding 质量？
4. 跨信号类型迁移的衰减有多大？是否在可接受范围内？

## 文件结构

```
sua_exploration/
├── PLAN.md              ← 本文件
├── PAPERS.md            ← 需要阅读的论文清单
├── data/                ← 数据集
│   ├── 000128/          ← MC_Maze
│   └── 000129/          ← MC_RTT
├── configs/             ← Hydra configs
├── scripts/             ← 实验脚本
├── docs/                ← 分析文档
└── papers/              ← 论文 PDF（用户放入）
```

## 时间线

- Day 1: 数据下载 + 格式探索
- Day 2-3: MC_Maze datamodule + teacher 训练
- Day 4-5: B3 student + 零样本迁移实验
- Day 6-7: 联合训练 + 分析 + 文档
