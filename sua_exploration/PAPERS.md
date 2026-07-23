# SUA-MUA Shared Encoder: 文献索引

> 本文用于文献导航，不代表当前实验优先级。项目状态和下一步分别见 [`README.md`](README.md) 与 [`ROADMAP.md`](ROADMAP.md)。

## 优先级 P0（必读，直接相关）

### 1. SPINT（本工作基础）
- **标题**: SPINT: Spatial Permutation-Invariant Neural Transformer for Few-Shot BCI Decoding
- **来源**: NeurIPS 2025 / arXiv:2507.08402
- **为什么读**: 我们的基础架构，理解 identity encoder 的设计
- **PDF**: 已在 workspace 根目录

### 2. NDT2（最直接对比）
- **标题**: Neural Data Transformer 2: Multi-context Pretraining for BCI Decoding
- **作者**: Ye et al.
- **来源**: NeurIPS 2023
- **链接**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10541112/
- **为什么读**: Multi-context pretraining 思路与 SUA-MUA 共享 encoder 高度相关；已在 FALCON 上验证
- **重点关注**: 如何处理不同 context（session）的 neural data，pretraining 策略

### 3. FALCON Benchmark
- **标题**: Few-shot Algorithms for Consistent Neural Decoding
- **作者**: Ye et al.
- **来源**: NeurIPS 2024 Datasets & Benchmarks
- **链接**: https://proceedings.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Conference.pdf
- **为什么读**: FALCON 数据集的官方论文，理解 MUA threshold crossing 的设计选择
- **重点关注**: 为什么选择 MUA 而非 SUA，few-shot 评测协议

### 4. Neural Latents Benchmark (NLB)
- **标题**: Evaluating latent variable models of neural population activity
- **作者**: Pei et al.
- **来源**: arXiv:2109.04463
- **链接**: https://arxiv.org/abs/2109.04463
- **为什么读**: MC_Maze/MC_RTT 数据集的原始论文，理解 sorted SUA 数据格式
- **重点关注**: 数据预处理、trial 结构、behavioral outputs

## 优先级 P1（重要，方法论参考）

### 5. LFADS
- **标题**: Inferring single-trial neural population dynamics using sequential auto-encoders
- **作者**: Pandarinath et al.
- **来源**: Nature Methods 2018
- **链接**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/
- **为什么读**: 使用 sorted SUA 的经典方法，理解 SUA 数据的统计特性
- **重点关注**: 如何处理 SUA 的稀疏性和噪声

### 6. CORP（已放入 workspace）
- **标题**: Plug-and-Play Stability for iBCI: A One-Year Demonstration
- **作者**: Fan et al.
- **来源**: NeurIPS 2023
- **PDF**: 已在 workspace 根目录
- **为什么读**: Continual learning + pseudo-label 思路，SUA 场景下更必要
- **重点关注**: Replay buffer、data augmentation、在线适应

### 7. Generalist Intracortical Motor Decoder
- **标题**: A Generalist Intracortical Motor Decoder
- **来源**: NeurIPS 2025
- **链接**: https://openreview.net/forum?id=ONOe6cAE9I
- **为什么读**: 跨被试通用解码器，与 SUA-MUA 共享 encoder 思路相似
- **重点关注**: 如何处理不同被试的 neural data 差异

### 8. UniBCI
- **标题**: UniBCI: Towards a Unified Pretrained Model for Invasive BCI
- **来源**: arXiv:2605.00061
- **链接**: https://arxiv.org/html/2605.00061v1
- **为什么读**: 统一预训练模型，明确区分了 SUA 和 MUA 数据集
- **重点关注**: 如何统一处理不同信号类型的 BCI 数据

## 优先级 P2（背景参考）

### 9. SUA vs MUA 对比
- **标题**: Comparison of spike sorting and thresholding of voltage waveforms for BCI
- **来源**: JNE 2015
- **链接**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4332592/
- **为什么读**: 直接比较 SUA 和 MUA 的解码性能

### 10. BCI without Spike Sorting
- **标题**: Control of a brain-computer interface without spike sorting
- **作者**: Fraser & Chase
- **链接**: https://www.semanticscholar.org/paper/3b9667673802d042503c3602d9442cd81e395ef4
- **为什么读**: 证明 MUA 足以实现 BCI 控制

### 11. Perich 2018（MC_Maze 数据来源）
- **标题**: A Neural Population Mechanism for Rapid Learning
- **作者**: Perich et al.
- **来源**: Neuron 2018
- **链接**: https://www.biorxiv.org/content/10.1101/138743v2.full-text
- **为什么读**: MC_Maze 数据的原始论文，3,139 sorted units
- **重点关注**: 实验设计、sorting 方法、行为任务

## 下载清单（需要用户手动下载）

以下论文可能需要 institutional access 或手动下载：

| # | 论文 | 下载链接 | 备注 |
|---|------|---------|------|
| 1 | NDT2 | https://papers.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf | NeurIPS 2023 |
| 2 | FALCON | https://proceedings.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Conference.pdf | NeurIPS 2024 |
| 3 | NLB | https://arxiv.org/pdf/2109.04463 | arXiv |
| 4 | LFADS | https://pmc.ncbi.nlm.nih.gov/articles/PMC6380887/ | PMC 免费 |
| 5 | Generalist Decoder | https://papers.nips.cc/paper_files/paper/2025/file/a00000e6a2208172700510bcd69d48e9-Paper-Conference.pdf | NeurIPS 2025 |
| 6 | UniBCI | https://arxiv.org/pdf/2605.00061 | arXiv |
| 7 | SUA vs MUA | https://pmc.ncbi.nlm.nih.gov/articles/PMC4332592/ | PMC 免费 |
| 8 | Perich 2018 | https://www.biorxiv.org/content/10.1101/138743v2.full-text | bioRxiv |

建议将 PDF 放入 `sua_exploration/papers/` 目录。
