# 实验一：E0 的使用依赖与增量效用

日期：2026-09-07。状态：设计稿，未启动实验。本设计不继承其他工单的 GPU、测试集或提交授权。

## 1. 研究问题与最小成本策略

区分两个问题，不能用一个推理消融同时回答：

1. **Reliance：现有模型是否使用 E0，尤其是否使用正确的单元—E0 对应？** 用现有权重做输入干预，不训练。
2. **Utility：有 E0 的模型是否优于允许重新适应的无 E0 模型？** 用相同配方、相同初始化的匹配训练比较。

默认先在已有 M1/H1 的健康 proj_add checkpoint 上做第一层；M2 用同一 harness 增加一个现有 proj_add checkpoint。不要以 H1 已塌缩的 concat 模型判断 E0 无效。688 的 Full/No-E0 匹配训练复用第二份设计，不再重复训练一套 E0 实验。

若要得出“M1/H1 各自的 E0 有增量效用”，仍需各任务自己的第二层匹配训练；688 的重训不能替代其他任务结论。第一轮不立即开启三任务成对重训。

## 2. 冻结范围与解释对象

保留 proj_add P16 的图和四维 carrier 路径：

`h_it = f(W_local[u_it + P(e_i)] + W_carrier c_i + b)`。

P32 可作为后续复现，不与 P16 混作同一对照。窗口、时间层数、EMA/RAW、normalizer、support、unit mask、query endpoints、输出尺度和 scorer 全部固定。

必须记录 E0 provenance：

- M1 当前 E0 是 B3 Sfix e11 的 activity-only embedding；rSyn3 经直接 carrier 路径进入 decoder。
- H1 当前 E0 已融合 H-C；M2 的 B3S E0 也可包含 T4。对它们的 E0 干预测试的是**整个静态 embedding 输入端口**，不能专称纯活动统计贡献。
- 本实验保持直接 c 不变，估计的是 E0 在已有 carrier 条件下的增量作用。
- 不将“只置零直接 c，但 c 仍在 E0 内”的操作称为彻底去除 functional profile。

默认 checkpoint 选择规则：现有冻结的 endpoint EMA；M1=P16 e24；H1=P16 L200 e24；M2按已有 source-development 预声明 pick 记录锁定，不能按官方 hidden 分或此次消融损失另挑权重。保存路径、权重 SHA、bank SHA、训练 session roster。

## 3. 第一层：四种无训练干预

每个有效单元 i，保持其 x_i、c_i、mask 与位置不变，只替换输入 E0。

| arm | E0 输入 | 主要问题 |
|---|---|---|
| E-REAL | 原 e_i | 参考 |
| E-ZERO | 0 | 现模型去掉 E0 后是否退化；可能存在 OOD |
| E-MEAN | 同 session 有效单元的 E0 均值，广播给所有单元 | 保留 session 级均值，去掉单元间差异 |
| E-SHUFFLE | e_pi(i)，仅在同 session 的有效单元内置换 | 正确单元对应是否重要 |

Shuffle 用 3 个预固定随机种子（101/102/103），每个 session 的映射在整段 query 中固定；不得逐 bin 重抽、跨 session 搬运或同时置换 x/c。记录 permutation SHA，不挑最差 shuffle。总体为 **6 条推理流，而不是 6 次训练**。

重要实现约束：

- 保持 `identity_mode=proj_add`。**不要直接调用现有 `identity_mode=zero`**：该模式可能保留 concat 的宽输入参数化，不是 P16 的输入消融。
- 最干净实现是 bank.E0 替换，或在 P(E0) 后加入显式 gate；当前 P 无 bias，zero 等价。不能只改一份 E0 缓存而沿用旧 static term。
- 每个 arm 从干净 reset 开始，重建由 E0 派生的 static term/frontend cache；执行结束恢复参考，不修改磁盘原 bank。
- P、decoder 和 E0 encoder 的权重均不更新；不为了让 zero 更好而微调 bias、重拟合 normalizer 或校准 readout。
- 打乱特征列不是首选控制；那主要扰动坐标基底，不能代替单元对应控制。

## 4. 低成本评分与覆盖

1. 先在已有本地允许读取的 source/development 面执行，不开官方 hidden 标签/新 EvalAI，不开 688 formal-test。
2. 每 session 预固定最多 2,048 个 query endpoints，按 query trials 和时间块均匀抽取。所有 arms 使用同一坐标列表；不要只取开头，也不根据误差挑点。
3. 合法的连续因果窗口可以批量 GPU 推理；这是算法消融评分，不是官方 streaming latency 证明。使用同一 FP32 权重/精度，避免消融与量化混杂。
4. 对入选的主要对照再跑完整允许读取的 query 面；非入选面不扩大。先完成全 session 覆盖，再考虑重复 seed，而不是在一个长 session 上算很多相邻窗口。
5. Stage-2 all-session 权重见过的面必须标 `EXPOSED_DEPENDENCE_DIAGNOSTIC`。不能用这类干预给跨 session 泛化做显著性结论。

主要指标：逐 session 方差加权 R²、equal-session mean、pooled R²（辅报）、REAL−每个对照的 paired delta；shuffle 对每 session 先平均三次再跨 session 汇总。

诊断指标：预测标准差/目标标准差、zero-activity 输出变化、第一层 dynamic/static RMS、||P E0||、NaN/Inf。RMS 只作机制描述，不直接作为因果归因。

只以 session 为主要统计单位；不同 query windows 不是独立“样本量”。给逐 session 表和配对 session bootstrap 区间；3 个 shuffle 不是 3 个训练 seed。

## 5. 第一层的判读

- REAL > ZERO，同时 REAL > MEAN、REAL > SHUFFLE：支持现模型利用单元特异的 E0 信息；仍需重训证明必要性。
- REAL > ZERO，但 REAL≈MEAN/SHUFFLE：更像全局幅度、session 级偏置或分布扰动效应，不能宣称正确单元身份是关键。
- ZERO ≥ REAL：E0 可能冗余/有害，也可能与直接 c 冗余；值得做无 E0 重训，不能直接宣布移除一定更好。
- 所有 arms 都近常数、R²≈0：先诊断模型是否塌缩，消融无辨识力。

默认 materiality 参考：平均 ΔR²≥0.01、且多数 session 同向才称“有值得追踪的依赖”；这是工程筛选尺度，不是统计显著性阈值。不因没达到它就跳过在 688 上已计划的匹配 E0 两臂。

## 6. 第二层：严格匹配的 Full vs No-E0 训练

最省成本的执行落点是 688 B-transformer 系列中本来就要跑的两臂：

- **B-FULL**：训练和推理都输入真实 e_i 与 c_i。
- **B-NOE0**：训练和推理都把 e_i 置零，c_i 不变。

同一 seed 下两者必须从完全相同的 decoder 权重字节开始，保留同样的 P 层与计算图，用输入 gate 区分；同一 session/batch/endpoint/dropout 次序、相同有效 batch、LR schedule、更新数、EMA、loss 和最终选点规则。

不能拿一个已训练 FULL 模型临时 zero 的分数，替代 NOE0 的重新训练分数。若复用已有 FULL 训练，必须同时能复现其初始化、样本和随机轨迹；否则两臂重新起跑。

Encoder/bank 默认冻结；所有预训练、basis 和 normalizer 的训练暴露范围必须披露。不能只让 FULL 更新 encoder，然后把差异全归给 E0。

默认先 seed42；有训练能力且不是塌缩后确认 seed43。第二 seed 是 decoder 初始化复现，若同一 frozen donor 则不得称 encoder 与 decoder 全链路独立重复。

主要结论依据 trained FULL−trained NOE0：

- 两 seed 同向、平均≥0.01、逐 session 大部分同向：支持 E0 提供可重复增益；区间跨零时保持不确定性。
- 差异在 ±0.005 附近：E0 当前增益很小，可以保留为兼容接口，但不宜作独立贡献主张。
- NOE0 稳定更好：承认负结果，优先考虑更简单系统；不得只展示第一层 zero 的 OOD 跌分。

这些数值是预声明的实用判断范围，不代表 ±0.005 和 0.01 间自动显著或无效。首轮样本少时完整报告效应和不确定性，不强制追到正结果。

## 7. 交付与边界

独立建议目录：`btransform_unified_v1/results/e0_utility_v1/<UTC>/`。交付冻结 manifest、每个 arm 的输入/派生缓存哈希、抽样坐标、逐 session 结果、shuffle 映射、完整复现命令与 interpretation.md。

688 配对重训结果可由本目录引用，不复制再训。M1/H1 后续若需独立 utility 结论，另开同合同的小规模合法 session 留出配对；不自动同时开三条训练线。

执行者不独占工作区，不覆盖旧产物、不回滚别人代码、不触碰活跃任务。设计阶段不启动训练、不提交、不改论文。
