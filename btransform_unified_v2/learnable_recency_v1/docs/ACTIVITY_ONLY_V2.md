# ACT-only v2：沿用现有 activity 主干并联合训练

本路线接入 M1、M2、H1。当前交付范围是实现、CPU smoke 和 M2 实验交接；GPU 调度由另一位 agent 负责。最终数字使用 EvalAI 提交结果，本地评估沿用 FULL 的口径。不得由本交付自动启动 GPU 任务，也不得在 M2 结果审查前启动 M1/H1 正式实验。

## 机制与改动

原 SPINT 在每个通道上使用同一组可训练函数：

\[
E_i=\psi\left(\frac{1}{M}\sum_{j=1}^{M}\phi(A_{j,:,i})\right).
\]

`phi` 先独立编码每个 trial，之后跨 trial 平均，最后由 `psi` 输出通道 identity。两个函数和 decoder 一起接受 source query 的行为预测损失。target calibration 只输入无标签 neural activity；模型权重冻结后前向生成 identity，不进行梯度更新。参见 [SPINT，NeurIPS 2025，§3.2–3.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/24c8d2fe8520746f0084174b75b93ebe-Paper-Conference.pdf)。

旧 M2 clean ACT 在冻结、曾以 T4 为条件的 encoder 上把 side 输入清零。虽然 activity 路径仍连通，encoder 没有为新的纯 activity 条件和当前 decoder 联合学习。v2 **保留现有 encoder 的 early-pool activity 主干**，在每个 source batch 中重新计算 identity，使梯度能到达 `pre_pool` 和 `post_pool`。采用从头联合训练，不加载旧 `E0`、`T.npy` 或旧 encoder checkpoint。

现有主干由一层 `Linear(T,H) + ReLU` 的 `pre_pool`、跨 trial 均值和三层 Linear 的 `post_pool` 构成，后者仅在中间层使用 ReLU。M1 沿用 B3/B3S 的 activity 主干（H64），M2 沿用 B3S/hold-contrast encoder 的 activity 主干（H64），H1 沿用 C2 的 activity 主干（H32）。纯 ACT 分支移除 descriptor 拼接及依赖 descriptor 的 FiLM，保留活动路径的层数、隐藏宽度和输出维度。这不是完整保留含 side 分支的 encoder，也不是换成 SPINT 的多层 trial encoder。

Decoder 继续保留 P16 投影、local/set frontend、D4 learned recency 和 literal-zero carrier。SPINT 在本路线中主要作为 source 联合训练和 target 无梯度 activity 校准协议的参考；不复制原论文的完整模型或超参数。

STATIC 的 identity 是一张按通道位置索引的可训练表。v2 ACT 的 identity 是共享函数从 activity 计算的结果，没有可训练的 session ID 或 unit ID 表。同步重排 query 和 calibration 的通道列时，ACT identity 随之重排，输出应保持一致。STATIC 需要把固定表也同步重排。ACT 是否最终利用了 activity，仍需通过训练后的 calibration 替换诊断和效果测量判断。

## 数据与训练协议

| 项目 | M1 | M2 | H1 |
|---|---:|---:|---:|
| Query context | 100 | 50 | 300 |
| 通道数 | 64 | 96 | 176 |
| Calibration trials | 10 | 33 | 3 |
| 每个 trial 的 encoder 输入长度 | 1024 | 100 | 1024 |
| Activity 主干隐藏宽度 | 64 | 64 | 32 |
| Identity 输出维度 | 100 | 50 | 700 |
| 正式实验状态 | 接入，暂不启动 | CPU smoke 后交给调度 agent | 接入，暂不启动 |

Calibration 接口只接受 activity 和可选 trial mask。训练监督、验证标签和评分标签属于 query 数据接口，不参与 calibration identity 的构造。Source minival 使用匹配 source training session 的 support，不从 minival query 中重新取 support。

M2 主实验沿用现有 7 个 source sessions、101,171 个窗口、固定 sampler manifest：seed 42，batch 32，24 epochs，每 epoch 3165 updates，总计 75,960 updates。Source support 是最初 33 个 trials，整个训练 query context 位于 support 之后。AdamW peak LR `3e-4`、weight decay `0.01`、一轮 warmup、cosine floor `3e-5`、gradient clipping `1.0`、EMA `0.9995`、whole-unit dropout `0.1`、行为目标乘数 `5` 均沿用现有配方。首轮实验在训练和校准时均使用完整支持集，不混入 support 子集采样等额外改变。

M1 的现有 STATIC 主线包含四个 source sessions。v2 会按 M10 边界剔除与 support 重叠的完整 query context，因此不能宣称其 source sampler 与 STATIC 完全相同。M1/H1 的实际数据计数以验收记录为准，正式比较应先核对数据协议。

## 与 FULL 一致的本地评分

本地公开 calibration 数据用于开发和选择 checkpoint；**最终结果以 EvalAI 提交反馈为准**。训练梯度仍只来自 source，target identity 仍只由 activity 生成；本地评分标签用于计算指标和选择全局 checkpoint，不用于 target 梯度更新。

| 任务 | 本地 query 集合 | EMA 扫描 | 汇总与选择 |
|---|---|---|---|
| M1 | FULL 的 HO3，3,881 个窗口 | epochs 1–24 | 每 session 的 channel-variance-weighted R² 等权平均，最高分并列取最早 epoch |
| M2 | FULL 的 EXT6，15,403 个窗口 | epochs 1–24 | 每 session 的 variance-weighted R² 等权平均，最高分并列取最早 epoch |
| H1 | FULL 的公开 HO-M3，14 份记录 | epochs 1–32 | 沿用 FULL 的 grouped-seven 汇总，最高分并列取最早 epoch |

对应参考入口是 `scripts/m1_full_learnable_train.py`、`scripts/m2_projadd_learnable_score.py` 和 `scripts/h1_learnable_train.py`。保持 query、padding/mask、行为缩放和汇总规则一致。Source minival 曲线用于观察训练过程；本地选轮与 FULL 一样扫描公开 calibration 评分。

M2 本地默认使用完整 EXT6，不把额外的 post-M33 四-session 切面用作主结果或选择条件。训练数据中的 source support/query 分离检查仍保留。

| 历史 M2 路线 | Checkpoint 口径 | 本地 EXT6 equal-session mean R² |
|---|---|---:|
| FULL learned recency P16 | FULL 的 24 轮 EMA 选择，epoch 10 | 0.3633777147 |
| 旧 clean ACT，side 全零 | 同样扫描 24 轮，epoch 10 EMA | 0.1193546050 |
| STATIC，仅供已有记录参考 | 固定 epoch 24 EMA | 0.2680975604 |

STATIC 末轮 sidecar（M2 e24 = 0.2681）不得与 FULL 的 24 轮扫描混比，也不得作为此后 EvalAI 选点。正式 static / act-only / flat 提交必须先跑与 FULL 相同的完整扫描。现有记录分别位于 `results/selection_m2_projadd_learned_slope_default_ext6_s42/score_receipt.json`、`results/selection_m2_projadd_activity_only_empty_side_s42_ext6/score_receipt.json` 和 `results/selection_m2_static_final_ema_ext6_s42/score_receipt.json`。以上均为本地分数，不是 EvalAI 最终数字。

M2 实验后应查看：与 FULL 同口径的 EMA 曲线和逐 session 分数、source minival 走势、encoder 梯度与 identity 尺度、替换 calibration 后预测的变化。是否扩展到 M1/H1，应先审查 M2 的实际结果；校准有效性和稳定性不能由 smoke 或单次 seed 42 的结果代替。

## 交接

运行命令、CPU smoke 结果和本次代码指纹记录在同目录的 `ACTIVITY_ONLY_V2_HANDOFF.md`。CPU smoke 仅验证接线、数据协议和可复现执行，不代表解码质量或训练稳定性已经得到实验确认。
