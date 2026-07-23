# SUA/MUA Shared Encoder Roadmap

**状态：当前执行计划**  
**更新：2026-07-23**

## 总目标

在统一、可复现的评估口径下回答三个递进问题：

1. B3 是否是可靠的跨信号类型架构基线？
2. B15/B16 是否稳定优于 B3，且改进是否为 SUA 特有？
3. 最终应选择按信号类型重训、共享 backbone，还是联合/context-conditioned 模型？

## P0：公平 SUA 基线

### 当前状态

- 训练脚本已支持 `--seed`，并保存 teacher SHA-256、训练配置和最佳 checkpoint 信息到 `run_metadata.json`。
- 比较脚本会拒绝 teacher 或核心超参数不一致的 checkpoint，并默认保存带 artifact SHA-256 的 JSON。
- 公平 B3-v2 训练仍待运行；本机两张 RTX 3090 与 CUDA 驱动已通过 PyTorch smoke test，项目 `spint` Conda 环境正在按 `SPINT-main/environment.yaml` 创建。

### 工作

- 使用当前 teacher：
  `sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt`
- 通过 `train_variant_mc_maze.py --variant B3` 重训 `b3_mc_maze_v2`。
- B3、B15、B16 使用相同 teacher、datamodule、训练轮数、loss、decoder 和 evaluation batches。
- 修改比较脚本，使结果保存为带 checkpoint 路径和参数的 JSON，而不是只打印到终端。
- 分开记录：checkpoint validation R² 与 comparison-script R²。

### 验收

- B3-v2 checkpoint 和 JSON 结果存在。
- 结果表中不存在跨 teacher 的直接排序。
- JSON 能追溯 teacher/student checkpoint、数据路径、batch 上限和评估日期。

### 决策

- 只有完成该阶段后，才能判断 B15/B16 相对 B3 的真实增益。

## P1：多 seed 稳定性

### 工作

- 为训练脚本增加显式 `--seed`，并在 trainer/model/datamodule 初始化前固定随机性。
- 对 B3、B15、B16 运行至少 3 个完全相同的 seeds。
- 报告 task R²、identity normalized MSE、cosine、Pearson 的 mean/std 和逐 seed 值。

### 验收

- 每个变体至少 3 个有效 checkpoint。
- 同一 seed 下三种变体共享 teacher、split 和评估脚本版本。
- 结论基于重复方向，而不是单个最佳 checkpoint。

### 决策门

- B15 若仅单 seed 领先，则保留为探索结果，不进入部署主线。
- B16 若以较低硬件代价稳定接近或超过 B15，则优先作为部署候选。
- 若 B3 与 B15/B16 差异不稳定，则停止结构扩展，采用按信号类型重训 B3。

## P2：MUA 机制对照

### 当前进展

- B16 的 FALCON M2 fold 0 / seed 42 公平对照已完成：held-out `0.248 ± 0.137`，B3 为 `0.236 ± 0.102`。
- mean delta `+0.0112`，逐 session 4/6 提升；该结果保留为正向候选，不通过稳定性决策门。
- 下一步优先补 B16 的额外 seeds/folds；B15 MUA 尚未运行。
- B16 优化不直接叠加更多高阶矩；先验证 B3-preserving zero-init、variance scale/shrinkage 和 support consistency，详见 [`docs/B16_OPTIMIZATION_BRAINSTORM.md`](docs/B16_OPTIMIZATION_BRAINSTORM.md)。

### 工作

- 优先在 FALCON M2 上训练 B3/B15/B16：它与当前零样本实验一致，且 `T=100`，更适合作为第一组控制。
- M1 作为第二个 MUA task replication，不作为第一步，因为其 `T=1024` 会同时改变参数量和计算成本。
- 使用真正的 LOSO + held-out sessions 报告 MUA 结果。
- 所有对照固定 teacher、decoder、loss、训练预算、seed/fold 和 checkpoint 选择规则。

### 判读

| 结果 | 支持的解释 |
|---|---|
| B15 只在 SUA 稳定提升 | 跨 neuron 关系可能捕获 sorting split/merge 信息 |
| B16 只在 SUA 稳定提升 | trial variance 可能是 SUA sorting 可靠性信号 |
| B15/B16 在 MUA 也提升 | 属于通用 NeuronID 建模改进，而非 SUA 特有机制 |
| 两者均不稳定 | 当前单次 SUA 增益可能来自 split、seed 或评估噪声 |

## P3：SUA 泛化协议

### 当前问题

- MC_Maze 只有单 session 内部 trial validation。
- `val_dataloader()` 目前重复返回同一个 loader，`val_heldout` 名称具有误导性。
- NLB test 文件没有可直接使用的行为标签。

### 工作

- 先修正 metric 命名，避免把 internal validation 写成 held-out。
- 寻找具有多 session 和行为标签的 sorted SUA 数据，或定义 MC_Maze→MC_RTT 的外部迁移任务。
- 明确 unit、trial、session 三种 holdout 的差异，并预先固定主指标。

### 验收

- 至少有一个不与训练共享相同 session/trial 集合的 SUA 泛化结果。
- 文档能够明确回答 holdout 的对象是什么。

## P4：共享训练策略

仅在 P0–P3 证明结构收益后开展：

- MUA/SUA 共享 backbone、独立小 head；
- signal-type 或 dataset context embedding；
- MUA pretrain → SUA fine-tune；
- SUA+MUA joint training；
- 与“分别重训两个小 encoder”的成本和准确率对照。

零样本权重共享不再作为默认目标；它已经在当前 B3 设置下失败。

## 暂缓工作

- 更大型的 self-supervised foundation model；
- 在公平基线前继续添加 B17+ 结构；
- 仅依据 internal validation 进行 ASIC 定型；
- 用 MUA LOSO 数字与 SUA internal-validation 数字直接排名。

## 推荐的下一条命令

```bash
conda run -n spint python sua_exploration/scripts/train_variant_mc_maze.py \
  --teacher_ckpt "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt" \
  --variant B3 \
  --out_name b3_mc_maze_v2_s42 \
  --seed 42
```

该命令会关闭最关键的公平基线缺口，但在运行前应先确认 GPU 和训练环境，并在完成后保存结构化比较结果。

训练完成后运行：

```bash
conda run -n spint python sua_exploration/scripts/compare_neuronid_variants.py \
  --teacher_ckpt "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt" \
  --b3_ckpt "<B3-v2 best checkpoint>" \
  --b15_ckpt "sua_exploration/checkpoints/b15_mc_maze/best-epoch=016-val_heldin/r2_mean=0.9092.ckpt" \
  --b16_ckpt "sua_exploration/checkpoints/b16_mc_maze/best-epoch=017-val_heldin/r2_mean=0.9020.ckpt" \
  --output_json "sua_exploration/results/p0_s42_comparison.json"
```
