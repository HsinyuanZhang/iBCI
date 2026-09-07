# 双线并行工单：逐 trial calibration memory + 独立 temporal decoder

Date: 2026-09-05 (Asia/Hong_Kong)
Status: READY_FOR_EXECUTOR_STAGE_0__PERFORMANCE_FIRST
Planning owner: Astra；实施、调度、训练交给后继 execution agent；Astra 接收审计包后复核。
Hardware allocation: 用户本轮明确提供 64 GB host RAM + 两张独立 RTX 3090；覆盖旧文档的单 GPU 约束。
Execution boundary: 本次仅创建工单，未创建实验 result root、安装环境、训练或评分。后继 agent 按本工单执行本地研究；不包含自动 EvalAI 提交、远端写入或修改历史结果。

## 0. 先纠正“两条线”的含义

不是“两个 decoder 同时训练”，而是两个独立研究问题：

| 线 | 问题 | 首个候选 | 保持不变 | 默认硬件 |
|---|---|---|---|---|
| A：calibration | 将逐 trial activity 表征过早平均，是否丢失了当前 query 可利用的信息？ | A-QMEM：query-conditioned prefix-memory read | 完整当前 champion、carrier、原 early-pool identity、支持预算 | GPU0 |
| B：decoder | 给定完整 activity signature + tuning profile，新的时间模型是否解码得更好？ | B-MAMBA：群体集合读入 → Mamba2 | 静态 calibration 接口、支持预算、query 历史与标签 | GPU1 |

优先级：正确输入/评分 → 尽快得到可解释的性能信号 → 第二 seed/合理训练时长 → 跨数据集 → 机制消融。
第一波不把 A 与 B 合并，不重启 FiLM profile 搜索，不做 PV，不安排完整论文消融矩阵。
新的 decoder 不受“小参数”限制；但必须计量实际速度，不能让编译或无限调参吞掉时间。

本工单细化并在执行顺序上覆盖 `DESIGN_DUAL_CALIBRATED_TEMPORAL_DECODERS_V1_20260905.md`：D2/D3、长历史、三 seed 内容消融后置；旧“两周/单 GPU”安排不再适用。

## 1. 授权范围与有限承诺

后继 execution agent 可以在新命名空间内实现、测试、建立独立环境，并使用两张卡完成下述限时本地实验。无需反复征求一般文件编辑、正常依赖安装、受限本地训练的权限。

- 不修改或删除历史 checkpoint/result root，不恢复已关闭的 688 FiLM 训练，不杀不属于本工单的进程。
- 不读取隐藏 test 标签；可见 development 一旦参与决策，就必须标为 development/retrospective evidence。
- 不新增 dense 行为量、query 伪标签或额外 calibration trial 作为输入。
- 不进行 EvalAI、Docker registry 或 Git push；本工单输出候选与审计包，部署另行安排。
- “完整权限”不能通过 subagent prompt 改写平台策略；继承当前环境，不反复提交提权请求。真实策略阻塞记入状态并转做独立可行工作，不尝试绕过。
- Stage 0 的事实检查由执行 agent 自行验收，不需要等待 Astra 才开始合法下一步；Astra 的角色是独立复核，而不是人为串行瓶颈。

时间目标：交接后 24 小时给第一份事实性进度/可用评分，48 小时给第一轮决策包。不是保证 48 小时内完成所有候选。每个首轮训练作业先限 6 GPU-hour，到限保存 checkpoint 和计时收据；协调者可在总 48 小时窗口内有依据地分配追加预算。超过此窗口不自动扩成多日 sweep。

## 2. 不能继承错的 baseline 与评分面

### 2.1 当前 M2 operational anchor

当前记录中的最强提交是 581919：MOVE-T4 + profile-free adapter，adapter seed 44、epoch 8（one-based）。官方 historical HO R² 为 0.3494526364。参见 `SUBMISSION_H1_M2_EPOCH_PICK_PAIR_20260905.md`。

`profile-free/EMPTY` 指 contrast 输入为零，**不指 adapter 权重为零**。训练过的头仍消费 `[MOVE-T4, 0]`，并对 early-pooled feature 做 `(1+gamma)h+beta`。A/B 都必须保留它作为冻结 calibration 的一部分。

重建依据：

- base：`streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt`；SHA256 `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`。
- canonical encoder：`tfpd_exploration/results/m2_hold_film_probe_v1/film_states.pt["p0"]`。
- trained head：`tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1/selected_head.pt["state_dict"]`；tensor-state digest `13551d3fc33d1cc296670c577c11519c04abdb42fa081f7d061191bb5610e6c5`。不要把 tensor-state digest 当成整个文件的字节 SHA。
- 重建代码参考：`tfpd_exploration/submissions/evalai_m2_movement_t4_empty_epochpick_v1/build_payload.py` 及相邻 `evalai_m2_movement_t4_empty_v1/build_payload.py`。
- 原 payload SHA256：`f2f8cd4c046a5880e9716d61981cee4aa0652d33e411212fa7be217cef05b051`。

先检查文件实际存在并匹配。不要为了复用 exporter 直接调用其全量 setup("test")；训练入口必须 source-only。旧 exporter 仅作重建逻辑参考。

### 2.2 新的 governing comparison 必须是 disjoint ext-4

代码审查发现：历史 `external_official_query` 入口可以把同一 held-out-calib 文件用于 support 和 query，默认 query_start_trial=0。因此历史六 session 的 0.3604492265 **不能直接作为新路线的 disjoint baseline**。这不撤销已经完成的官方 hidden 结果，也不是声称旧结果全部无效；它要求本工单重新建立同面比较。

现有 M33-disjoint metadata 给出一个可行面：

| 可见 held-out session | 总 trials | M33 后 query trials | 既有 clean-window 数 |
|---|---:|---:|---:|
| 2020-10-30 Run1 | 43 | 10 | 519 |
| 2020-10-30 Run2 | 41 | 8 | 490 |
| 2020-11-18 Run1 | 41 | 8 | 425 |
| 2020-11-19 Run1 | 43 | 10 | 635 |
| 2020-11-24 Run1 / Run2 | 各 33 | 各 0 | 各 0，排除并披露 |

合计 4 session、3 date、2,069 windows。几何证据：
`streaming_calibration_exp/outputs/streaming_calibration/m2_m33_disjoint_replay_correction_v1_t4_m2_final1_f1_s42_20260801_120203/split_manifest.json`。

只继承该记录的窗口边界参考，**不继承其旧 fold 的训练划分或 normalizer**。执行者重新核对当前数据与 mask，冻结具体 window IDs；若数量不同先解释差异，不能凑数。已有文件名检查未发现 held-out-minival，不能虚构一个。

在此面重新评分完整 champion，得到 `R_REF_clean`。本工单写作时此数未知。所有 A/B 的增量相对它或同面 B-TRANSFORMER；不得与 0.360449、0.349453、旧 act30 六 session 数直接相减。

ext-4 已被历史研究使用，不是 fresh test；两个 Run 属同日，不能把四 session 当四个独立被试。主读数沿用 session 等权 R²，另报三 date 等权敏感性结果。

### 2.3 训练与选点的数据契约

1. source 仍为选定 M2 的七个 held-in session；从 pinned config 构建 `setup("fit")`，显式 `include_heldout_in_fit=False`，记录文件 allowlist。
2. 支持为同一合法 first-33 trial；source query 的**完整 50-bin window**不得与支持 trial 重叠。旧 `within_post30` 不能因为传入 horizon=33 就假定自动变成 post33。
3. calibration 数组是 mask-filtered、100-bin 插值视图；query 数组是另一条时间轴并带 49-bin leading padding。用原 trial ID/raw interval 映射边界，不能直接比较两种数组的 offset。
4. held-in-minival 用作每 epoch 的 source-development 学习曲线与 checkpoint 选择。它的 support 继续取对应 held-in-calib，禁止从 query 文件重新建池。
5. 首轮 governing checkpoint 是第 1–12 epoch 中 source-minival 等 session 均值最高者；tie 在 1e-10 内取较早 epoch。保存全部 epoch，且固定报告 epoch12。A/B 同一选点权限，不复制旧 TKD 的每 epoch external 打分。
6. ext-4 只在完成 endpoint 后评分“source-selected + endpoint12”两个 checkpoint（相同者去重）。用于路线分配的外部数字也是 development evidence；不得称完全外部盲选。
7. 训练目标、反归一化、mask 和 variance-weighted per-session R² 原样绑定。当前 M2 调用链有预测 `/5`；重建时明确 target space 和转换层，避免 B 少除一次或多除一次。

若按§4.3进入24-epoch扩展，新增 governing checkpoint 固定为 epochs1–24 的 source-minival 最优者，同一 tie 规则；只新增该 checkpoint 与 endpoint24 的 ext-4 评分（已评分的去重）。保留原 selected/endpoint12 报告，不覆盖；后续资源判断可使用扩展结果，但必须标注24ep与新增选点机会。原12ep比较与共同24ep比较分开列。

## 3. A 线：逐 trial memory 的唯一首轮实现

### 3.1 不改变支持预算，不改成 continual

只保留已允许的 prefix trial 表征。当前 query 只用于读取，不写回 bank；无 query trial 完成检测、无 hidden 流分段、无伪标签、无新的标签利用。

当前 M2 形状：

```text
S            [1,33,100,96]  legal calibration activity
u=phi(S)     [1,33,96,64]   frozen Linear(100,64)+ReLU
h_bar        [1,96,64]      native chronological mean
T            [1,96,4]      source-normalized MOVE-T4
E0           [1,96,50]     FULL frozen champion identity, including trained EMPTY head
X            [B,50,96]     live neural windows
E_dynamic    [B,96,50]
```

原 early-pool `E0` 不重写为 `mean_j(post_pool(u_j,T))`；新路线不重复晚池化迁移失败的换算子问题。T4 仍由整份 M33 前缀拟合，不为每条 trial 拟合一个欠定 T4。

### 3.2 Query-conditioned centered residual read

对每个 unit 独立在 trial 轴做 attention，不对 `N×K` 个 token 做全对全 attention：

```text
s_bi  = GELU(Linear_50_to_64(X_b[:,i]))
q_bi  = Q([s_bi, E0_i, T_i])
k_ji  = K([u_ji - h_bar_i, T_i])
v_ji  = V(u_ji - h_bar_i)
w_bij = softmax_j(q_bi · k_ji / sqrt(d_head))
r_bi  = concat_heads sum_j (w_bij - 1/K) v_ji
delta = W_out(GELU(r_bi))
E_bi  = E0_i + delta_bi
y_b   = frozen_student.decode_with_identity(X_b, E_b)[:, -1, :]
```

具体首轮配置：2 heads，key/query 每 head 16 维，value 每 head 16 维；Q/K 均为 hidden64 的两层非线性 MLP；V 是无 bias 的 Linear(64,32)；W_out 是无 bias 的 Linear(32,50)，零初始化。Q/K/V 普通非零初始化，不再叠一个零 gate，以免两层零乘积断梯度。

- K 中 T 的作用必须经过非线性；单纯线性 concat 的同-unit T 项会对所有 trial 加相同 logit，softmax 后抵消，不能据此声称 tuning-conditioned selection。
- `q` 来自真实 query neural，不是 decoder 的固定 learned `rep`；不能将 50-bin query 直接喂给要求 100-bin 的 frozen pre_pool。
- 只用最后一个 bin 的输出。整窗口产生的 E 不能拿去声称窗口前部输出也是因果的；若未来改成全序列监督，每个时刻必须有自己的 causal query。
- `K=1` / 全部 memory 行相同 / 强制 uniform attention 时 residual 应为零（声明 FP32 数值容差）。这是正确性检查，不是首轮训练消融。
- 输出零初始化只保证训前等于 anchor，**不保证训后非劣**。完整 baseline 仍保留为可回退产品。

### 3.3 冻结、缓存、梯度

全部 inherited 参数冻结，包括 pre_pool、post_pool、trained EMPTY head、decoder。只训新增 Q/K/V/query projection/W_out；manifest 输出逐参数 allowlist。

保持 frozen decoder 的 eval mode，但训练时不能把 decode 包进 `no_grad()`：新 residual 需要经过冻结网络的输入梯度。正确 hook 是 `streaming_calibration_exp/src/models/components/streaming_spint.py::decode_with_identity` 外围 wrapper，不是仅替换不能看到 query 的 id_encoder。

缓存 u、E0、T；cache key 绑定 checkpoint/head/preprocessing/dtype/session/unit roster/support IDs。u 的计算算子冻结，因此可以 detach；K/V 在训练中变动，投影不能跨 optimizer update 使用 stale cache。部署冻结后才可缓存 K/V。

E0 必须通过原 native chronological FP32 路径生成。不能用新的 batched GEMM 或 torch.mean 替换它，然后要求 bitwise parity。逐 trial u 的缓存可有独立明确的算子容差，但不得改变 E0。

bank 占用约 `33×96×64×4=811,008 bytes/session`，不是 O(1) 常数状态；存储随 K、N 增长。它是 fixed-prefix memory，和 decoder 的动态 hidden state 分开计账。

### 3.4 A 首轮训练

- 一个候选 A-QMEM，seed42；REF 不重训，只重评。静态/均值/乱序等训练臂后置。
- 12 个完整 source passes；不继承 FiLM 的“每 session 抽256个 window = 一 epoch”。epoch 定义与 B 相同。
- AdamW，LR=1e-4，weight_decay=1e-2（bias/norm 排除），gradient clip=1.0；初始 effective batch32，按 session 组 batch，不静默丢尾样本。
- 5% updates linear warmup 后保持 LR 到 epoch12。第一波不动态搜索 LR、不同时换 Huber、EMA 或 carrier。
- inherited decoder 保持 eval，其 dropout 不被打开；A 新分支无 dropout。若跨线 dropout 不同，作为不同系统的训练条件披露，不把 A/B 直接相减作结构因果结论。
- 在真实 source batch 做至少两次 optimizer update，验证 W_out 首先收到梯度，随后 Q/K/V 收到有限非零梯度。零头时的前向正确不等于后续分支能训练。

A 若 null，只能说这个 frozen-anchor retrieval 没有带来可用收益；不能断言所有逐 trial calibration 都无价值。

## 4. B 线：先一个成熟 SSM，再一个必要参照

### 4.1 第一候选 B-MAMBA

固定与 A/REF 同一 `E0` 和 T，不使用逐 trial bank。前端保留完整校准，不能复用旧 TKD 仅 T4/rho 的输入。

```text
X [B,50,N]
  -> shared causal Conv1d(1,16,kernel5)+SiLU per unit
  -> concat local feature / E0_i / T_i; MLP to d_set=256
  -> per-bin set cross-attention: 8 learned slots, 8 heads
  -> one residual slot FFN, concat slots -> Linear(2048,512)
  -> 4 pre-norm residual Mamba2 blocks
  -> LayerNorm -> Linear(512,128)+GELU -> Linear(128,2)
  -> score LAST timestamp only
```

Mamba2：d_model512、d_state64、expand2、headdim64、ngroups1、d_conv4、chunk_size64。准确参数量实现后统计，预期为几百万级，不写成已经测出的精确数。保留 reference dynamics initialization 与特殊 no-weight-decay 标记。

每个50-bin窗口 reset state，causal conv 包含在该历史长度内。只在 TIME 轴运行 Mamba；不能在任意 unit 顺序上 scan，不 flatten 固定 roster。

### 4.2 B-TRANSFORMER 的地位

同一前端、slot 数、输入、readout；时间模型为4个 pre-norm causal Transformer block，width512、8 heads、FFN1024、固定 causal positional encoding。

这不是可以无限后置的“机制消融”，而是判断新 decoder 是否超过一个合理新训练参照的最低比较。调度上 B-MAMBA 先出数；它能够正常学习且未显著掉队时立即运行 B-TRANSFORMER seed42。若首个候选已严重失败，可节约该作业，但没有该参照就不作“SSM优于Transformer”的结论。

D0/D1 common frontend 初值在 seed 内完全相同，各自 temporal 参数用独立 RNG domain。参数量并非严格相等，完整报告；首轮允许这种性能比较，精确容量控制留到有信号以后。

### 4.3 训练量与选点

- seed42 先筛；effective batch32，AdamW LR3e-4，clip1.0，weight_decay1e-2，遵循 dynamics/bias/norm exclusions。
- 前端与时间主干从头训练，calibration encoder 冻结；不载入 PV 初始化，不用 teacher distillation。
- 12 full source passes；每个 epoch 覆盖冻结 manifest 的所有 eligible windows（尾 batch 不丢），只监督原约定最后 bin。记录 updates/unique targets，不能用多时刻 loss 偷增监督。
- 共同 whole-unit dropout：每个 window 以0.10概率独立丢 unit，mask 在50个时刻固定；同时屏蔽对应整行 token，并从 set attention 中 mask；若全丢则按固定规则保留一个 unit。D0/D1 同样实现，不附加随机 support sampling。
- source-only FP32 smoke 后可固定 BF16 autocast、FP32 optimizer/master dynamics；两 B 臂同一精度。A 先 FP32 以保护 anchor，跨线精度差异披露。
- LR 前1 epoch warmup，保持到 epoch12；若延长到24，固定从epoch13余弦下降至初始LR的0.1。此扩展规则现在冻结，不在看到好坏结果后新发明 schedule。

允许一次共同12→24延长：至少一个 B 臂 source-minival 的9–12 epoch平均loss较5–8降低≥3%，且计时预测在总窗口内留足6小时评分/审计。两臂均获得同样总曝光权限；12-epoch结果保留。first wave 不自动延至48。

若 source 学不动：先排查目标单位、梯度、mask与kernel。若 source进步而clean ext明显下降：标记 transfer failure，不自动扫宽度/LR掩盖它。

### 4.4 本机 Mamba 环境是实际工作，不是假定已就绪

只读核查：`spint` 中未安装 mamba-ssm/causal-conv1d；Torch 2.5.1.post303，runtime CUDA11.8，system nvcc11.5，Python3.10.15，Triton3.2.0，CXX11 ABI=True。安装前重新检查，避免将快照当永久事实。

在独立环境准备兼容的 Torch/CUDA compiler/Mamba pinned artifact；不改变共享 spint。锁 package版本/commit/wheel SHA 和可复现命令。import成功不够，必须3090上实际 forward/backward/step通过。

依赖初次 bring-up 先限90分钟；可再给一次有明确诊断的修复窗口，总计最多4小时。期间 A 与 CPU数据工作不停；B-TRANSFORMER可先训练。不静默用旧 DiagSSM、Mamba1 或手写近似替换后继续标 B-MAMBA。仍不可用则报 ENGINEERING_BLOCKED；必要时提出另立命名的 reference S4D 后继，不把安装失败写成 SSM 科学失败。

## 5. 暂缓什么，什么不能暂缓

第一波必须做：输入/标签/边界、原模型replay、零残差anchor、真实分支梯度、因果/置换/step检查、same-surface REF、候选正常学习与资源计量。

可以后置：A-only/T-only/双校准重训、matched-capacity adapter、训练时K/V错配、随机memory、support容量曲线、三seed显著性、SUA/pMUA、长历史、CPU Docker latency、D2/D3与A+B组合。

尤其：同时打乱memory key/value的trial顺序应不变，是集合不变性测试，**不是 negative control**。要破坏内容对应，应后续打破K/V配对或unit绑定。注意 fused E0 内已有T4，不能只删外部T4端口就声称做了T4消融。

第一轮正数只证明可用候选/性能信号，不证明新增信息、注意力解释、跨数据集通用创新或“第二创新点已成立”。这不是压低目标，而是防止先有小正数再透支故事。

## 6. Stage 0 最小验收：快，但不是省略正确性

| 验收 | 通过标准 |
|---|---|
| authority | 文件存在；base/head/normalizer/config/window manifest 有hash；完整trained EMPTY head保留 |
| data access | train入口只打开source allowlist；external evaluator独立打开可见ext4；hidden/test未打开 |
| disjointness | source与ext4的整个输入window均在support后；trial ID、mask、padding映射可审计 |
| REF | 完整champion在新clean面重评；不套历史六session数字 |
| A anchor | 原生E0逐位相同；W_out=0、FP32同device同kernel下baseline预测相同；明确有符号零处理 |
| A live branch | 非零人工W_out时改变合法memory/当前query能改变输出；至少两更新后Q/K/V梯度有效 |
| causality | future input perturbation不改更早合法输出；最后bin窗口契约与多输出契约不混 |
| permutation | 同步置换unit的X/E0/T/memory后预测不变；同步置换memory K/V不变 |
| B recurrence | forward/step/chunk/reset在长度1/4/49/50/63/64/65/129、非零分支和高幅输入下有限且匹配 |
| output space | last-bin、mask、目标单位、/5位置及R²实现与REF一致 |
| measurement | 20 warm-up+100 timed optimizer steps；examples/s、峰值显存、PSS/RSS、I/O、评分时间 |

误差政策先于评分固定：A原生零分支用精确比较；置换/不同矩阵形状的FP32算子与B scan/step先用 atol1e-5/rtol1e-4（标准化输出空间）并报告最大误差；高幅另报相对误差。若不通过先定位，不看decoder R²后放宽。BF16另与FP32做source-only数值核查，不能要求低精度位级一致。

所有gradient smoke与20+100步计时均在一次性model/optimizer实例上完成，不进入正式训练。正式epoch1前按seed重建模型与common frontend，重置optimizer、scheduler、sampler/dropout RNG和全部递推状态；冻结bank可复用。记录这些预检曝光但不将其混成正式更新。24ep是从已保存epoch12完整训练状态继续，不套用“重置epoch1”规则。

不用启动旧隐藏数据 exporter 来完成 smoke。旧 `DiagSSM.step()` 每步 clamp 与 `forward()` 卷积后 clamp 不同，已存在CPU反例；新实现不继承该算子。

## 7. Subagent 并行：一个协调者 + 三个明确 owner

最多4个同时活跃agent，不为每个seed另开agent。Agent数量与训练进程数不是一回事；进程由job ledger集中管理。

| Owner | 独占文件/责任 | 并行时做什么 |
|---|---|---|
| Execution coordinator | `contracts.py`, `plan.py`, package初始化、runner CLI、manifest、job ledger、结果汇总 | 先冻结接口，审核合并，发GPU lease，决定限时继续/停止；不等待每一步人类确认 |
| A implementation worker | `calibration_memory.py`、A专属测试 | 实现per-trial bank/read、零锚与梯度；GPU0作业运行时完成A推理/审计接口 |
| B implementation worker | `decoders.py`, `ssm_backend.py`、B专属测试、环境兼容方案 | 同时做D0/D1和kernel检查；GPU1训练时查学习曲线/数值，不扩新结构 |
| Shared pipeline worker | `data.py`, `training.py`, `evaluation.py`、pipeline测试 | 单次数据整理、缓存、disjoint评分、resource monitor；不重复构建两套数据管道 |

统一新命名空间：`tfpd_exploration/src/m2_dual_track_v1/`；新测试：`tfpd_exploration/tests/test_m2_dual_track_*_v1.py`；新runner：`tfpd_exploration/scripts/run_m2_dual_track_v1.py`。

协调者先用30分钟定义协议：

```text
SessionBank: raw/support trial IDs, X_store, target_store, eligible_starts,
             frozen_u(optional), E0, T, unit_mask, provenance
Batch: session_id, X[B,W,N], last_target[B,2], immutable bank reference, window IDs
Candidate.forward_last(X, bank, unit_mask) -> [B,2] in declared training target space
Candidate.trainable_parameters() -> explicit allowlist
Evaluator(candidate, scoring_manifest) -> per-session metrics + prediction digest
```

worker均可先用synthetic tensors和stub bank并行，不等完整NWB加载。共享API变更由coordinator先改契约版本、通知两worker；不可各自修改common files。

每份worker prompt必须写：**你不是唯一在仓库工作的agent；只修改你的owner文件；不得撤回他人变更；兼容共享契约；不擅自改历史root、不自行扩大数据权限、不独立抢GPU或启动第二个训练。**

专用LUNA监控不是必需的第四个子agent。优先由CPU worker实现持久化监控脚本；四槽限制下训练阶段CPU worker完成后，可将该槽交给轻量watcher。watcher只读日志/资源、通知协调者，不改超参、不自行重训。用户指定模型时再指定其模型，不靠模型名称赋予权限。

## 8. 双3090 + 64GB主存调度规则

### 8.1 GPU lease，不做两卡DDP

当前只读硬件快照：两卡各24,576MiB；GPU0约453MiB显示/桌面占用，GPU1约23MiB；二者无计算负载。执行时必须重新检查。

- GPU0 UUID：`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` → A。
- GPU1 UUID：`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` → B。
- 每个进程通过CUDA_VISIBLE_DEVICES只见一张物理卡，内部使用cuda:0；两张24GB不是一块48GB显存。
- 默认每卡一个训练进程，不用DDP/NCCL，不把相同数据重复载入两个无关流水线。
- A先结束时GPU0可借给B-TRANSFORMER或B的第二seed；B编译阻塞时GPU1可借给A确认。只有协调者修改lease，不能静态空等。

允许同卡双小臂的**测量式例外**：单作业已排除I/O瓶颈但GPU利用率仍低；先做10分钟同卡probe，两个作业总吞吐较串行实测提高≥15%、总峰值显存≤20GiB、主存预算合格且关键候选预计完成时间不恶化超过25%，才保留。否则立即回到一卡一作业。A+B同卡不作为首选。

### 8.2 64GB内存与CPU预算

当前实际RAM约62GiB，MemAvailable约54GiB；swap约1.9GiB几乎已用完。这不证明正在换页，但禁止依赖swap承载新的大缓存。机器32逻辑CPU，必须覆盖loader的默认31 workers。

| 项 | 起始政策 |
|---|---|
| 主存总应用预算 | 约≤44GiB working set/PSS；始终以MemAvailable≥12GiB为最终约束 |
| 每GPU loader | workers=0起步，此时省略prefetch_factor或设None；证实input bottleneck再升至2且prefetch_factor=1；不预设persistent workers |
| 线程 | 每训练进程Torch/BLAS2–4线程；编译MAX_JOBS=2；CPU评分最多4线程；避免线程乘法 |
| 显存 | 每卡作业实测峰值≤20GiB，保留桌面/allocator/评分余量；B若超限先减microbatch并梯度累积保持effective32 |
| 数据组织 | 单次建立只读紧凑session arrays+index；按batch gather；u/E0/T按session缓存并广播 |
| 禁止的缓存 | 全部重叠windows实体副本；全部query的[B,T,N,256]隐层；按window重复support bank |
| 磁盘 | 当前可用约152GiB；新env/cache/results合计先限40GiB；建立前估算，不能删旧ckpt腾空间 |

不要简单相加多进程RSS后误判共享页为不同内存；同时看PSS、MemAvailable和swap-in/out。MemAvailable<10GiB暂停新任务入队，<8GiB或持续换页则先停/降并发本工单的次要作业；不得杀无关进程。

两个GPU worker通过只读memmap/紧凑数组共享文件页；不要各自先全量load全部source+external。大数据初始化串行，后续训练并行。FP32 calibration缓存先保真，训练mixed precision只作用于已经验收的candidate。

### 8.3 后台运行与监控

协调者维护持久化job ledger：run ID、owner、PID及启动时间、GPU UUID、配置hash、log/checkpoint路径、状态、开始/预算/估计完成时间。使用tmux或可靠detached launcher，stdout/stderr入专属log；agent一轮结束不应终止训练。

脚本每60秒记录GPU util/显存、CPU/PSS/MemAvailable、磁盘、global_step、examples/s、loss、最新ckpt、ETA；每10分钟摘要，阶段结束立即报告。五分钟无global_step推进先判断是在编译/验证/存盘，确认无合法心跳再告警。

所有取消只针对ledger确认的本工单PID及其已验证子进程，先请求保存/正常退出，再采取有范围的终止；不能按模糊进程名pkill。失败最多一次同配置resume（optimizer/RNG/step完整恢复）；改配置必须新run ID，不覆写旧run。

## 9. 时间轴与作业队列

| 交接后时间 | GPU0 / A | GPU1 / B | CPU / 协调工作 | 必须交付 |
|---|---|---|---|---|
| 0–2h | 零锚/梯度smoke | kernel/dependency与synthetic tests | 契约、baseline字节、source/ext4 manifest；单次数据缓存 | inventory+contracts，阻塞必须具体 |
| 2–6h | 能通过即启动A seed42 | 能通过即启动B-MAMBA seed42；编译卡住先B-TRANSFORMER | REF_clean replay、吞吐预测；不让通过的线等另一线 | 第一个真实training heartbeat |
| 6–24h | 完成12ep并打分；有信号则A seed43 | B首候选12ep；随后B参照；需要时借空闲GPU0 | 固定selected/endpoint评分、保存失败、内存调度 | 第一份候选/无候选报告 |
| 24–48h | A确认或释放硬件 | 完成最低比较；满足规则才延长24ep/第二seed | 数值复检、4session/3date表、审计包 | GO_FOLLOWUP / CLOSE / INCOMPLETE分类 |

这是排程目标，不是未经计时的吞吐承诺。实测ETA公式：

`预处理/编译 + epochs × (batches_per_epoch × step_time + source_minival_time) + endpoint_scoring`。

旧0.2M TKD的30epoch约10–12分钟、FiLM几十秒，只能作历史背景；不能外推到新的几百万参数decoder。100-step profile后更新预计，不反复报虚假的固定完成时间。

## 10. 限时分流门：工程预算规则，不是统计显著性

主增量是在clean ext4上、source-selected checkpoint相对同面REF的session等权差。并列epoch12与逐date结果。以下阈值是本轮资源分配的实用量级，不是MDE估计、非劣检验或论文显著性门。

- **A优先确认**：mean delta≥+0.005、至少3/4 session正、worst delta≥−0.03 → 分配seed43。否则不自动增profile/解冻psi；若仅epoch12失败而source-selected为正，按已冻结选点报告。
- **B性能候选**：mean delta≥+0.005且无单session<-0.05灾难性退化 → 完成B参照并优先seed43。
- **B优化可行但未追平**：mean delta∈[−0.03,+0.005)，source学习正常 → 完成B参照/检查共同延长条件；不得称已经非劣。
- **B明显transfer failure**：mean delta<−0.03且source fit已改善 → 第一轮不做D2/D3/多LR sweep；记录结果并结束或用现有B参照区分前端失败。
- **任何线未完成**：编译、OOM、速度慢或训练未收敛 → INCOMPLETE/ENGINEERING_BLOCKED，不改写为科学null。

阈值不通过也完整报告曲线和all-session差异。两seed一致的小正数是“值得后续”，不是已经有强统计证据。不能从已选最优checkpoint的四session bootstrap宣称无选择偏差。

## 11. 只有出现信号才做的后继队列

1. A有效、B无效：保留原decoder+A，先第二seed；再DANDI SUA/pMUA匹配协议，最后H1 M3。H1 K小意味着可检索自由度小，是边界不是硬性不许做。
2. B有效、A无效：B作为独立性能路线；后续再做A/T内容矩阵检验双校准可移植性。可以有好decoder而没有新的calibration主张。
3. 两者有效：先分别确认；之后增加**一个**A+B组合，与已完成的A-only/B-only/REF形成四格。组合训练不自动说明增益可加。
4. B-MAMBA viable但有明确“压缩前丢时间信息”动机：另立D2 unit-time-first作业，与同calibration前端比较；不在第一轮忙着运行全部D0/D1/D2/D3。
5. 两者都null：停止扩散。保留已成立T4/activity主线与负结果，不为凑第二创新点继续造无上限附件。

后继需要新的冻结小工单；不是本轮自动开始所有数据集。SUA>pseudo-MUA是待测结果，不能预设筛选规则确保它出现。

## 12. 给Astra的验收包

执行者建立新root建议：`tfpd_exploration/results/m2_dual_track_v1/<timestamp>/`。只能创建新目录，既有同名路径不覆盖。

必须包含：

1. `execution_manifest.json`：本工单SHA、代码文件SHA、环境lock、checkpoint/head/normalizer authority、source/dev/window IDs、训练参数allowlist、输入/目标单位。
2. `stage0.json`：所有正确性结果、原生E0/REF parity、disjoint ext4计数、数据访问审计、Mamba numerical receipt。
3. `jobs.jsonl` 与 resource snapshots：GPU lease、PID、失败/重试、实际step/曝光、吞吐、峰值显存、主存、总GPU-hour。
4. 每arm的逐epoch source loss/minival、全部epoch ckpt或A的轻量head状态；best不替代last；选点由固定脚本生成。
5. `comparison.csv`：REF/A/B各session与date聚合、selected/last、delta、训练seed、R²单位、运行时间；不存在的格写NOT_RUN而不是0。
6. `HANDOFF_FOR_ASTRA_REVIEW.md`：一句判决、最强证据、最强反证、协议变更、是否发生外部选点、推荐后继和所有未完成项。

不要求第一轮写40项重复receipt模板；同一个authority引用一次。原始必要证据不能省：样本/标签边界、权重身份、选择权限、实际算力与失败记录。

## 13. 可直接交给执行 agent 的短入口

> 请执行 `tfpd_exploration/docs/WORKORDER_CALIBRATION_MEMORY_AND_TEMPORAL_DECODER_PARALLEL_V1_20260905.md`。你负责实施与协调，Astra只负责后续审核。两条线是逐trial query-conditioned calibration memory与独立SSM decoder，不是两个decoder、也不是continual query-memory。使用两个独立3090和64GB RAM，按工单建立coordinator+3个明确owner的subagent；先Stage0冻结完整M2 champion与M33-disjoint ext4评分面，GPU0跑A、GPU1跑B，资源空闲可租借。首轮性能优先，消融后置，不用PV、不重启688 FiLM、不动历史结果、不自动提交EvalAI。继承当前权限，不为正常本地工作反复问权限。24小时给事实进度、48小时给决策包；用测得的吞吐安排，而不是许诺全部矩阵完成。请先输出owners/作业队列，然后开始，不要只再写一份计划。

## 14. 本轮只读审查的关键来源

- `streaming_calibration_exp/src/models/components/streaming_encoders.py`：B3S pre_pool、逐trial累加与原生finalize。
- `streaming_calibration_exp/src/models/components/streaming_spint.py`：decode_with_identity的[B,N,50] hook与冻结权重下的输入梯度。
- `tfpd_exploration/src/m2_hold_film_probe_v1/encoder.py`：trained profile-free head的真实语义。
- `streaming_calibration_exp/src/data/falcon_datamodule.py`：heldout-calib作为query的构建路径、padding/disjoint检查、source-only fit。
- `tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py`：旧post30 helper与R²定义；只借定义，不借旧预算常量。
- `tfpd_exploration/src/m2_movement_t4_ablation_v1/physical.py`：MOVE-T4用未插值calibration bin [5,30)，不是100-bin activity tensor。
- `sua_exploration/scripts/write_m2_m33_disjoint_replay_correction_protocol.py`：ext4而非ext6的协议先例。
- `tfpd_exploration/docs/DESIGN_DUAL_CALIBRATED_TEMPORAL_DECODERS_V1_20260905.md`：SSM设计、相关工作、TF-SR负结果与clamp反例。

本工单不把Mamba本身、保存trial本身或普通attention写成已成立创新。要检验的是：双校准之上是否仍有被过早压缩的信息，以及更合适的时间解码能否把既有表征转成可用的跨session收益。
