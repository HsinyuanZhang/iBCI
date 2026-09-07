# 实验二：688 的低成本 B-transformer 系列与 M2-like carrier

日期：2026-09-07。状态：设计稿，未启动实验。本设计是新的独立分支，不覆写历史 688/FiLM/SPINT/T4 结果，也不继承旧工单的硬件或 sealed-test 授权。

## 1. 核心决策

第一轮仅做 **DANDI 000688、sub-C、CO、SUA**。先不扩 RT、sub-M、pseudo-MUA、量化、P32、其他时间核或多窗口网格。统一 decoder 采用 **proj_add P16 + 8 slots + CausalPE4**。

省成本的主要手段：冻结并缓存 calibration encoder/bank、使用现有神经时间序列缓存、按固定更新数抽样训练、减少高度重叠窗口、分阶段保留有信息的对照。不以删除困难 session、挑高分单元或改评分 mask 节省算力。

两项科学问题：

1. B-transformer 在这组本地跨 session 数据上是否有竞争力？
2. E0 与 directional carrier 分别是否有用？

与历史 SPINT/T4 的比较先按“系统级参考”报告。新 estimator、新冻结 encoder 和新训练预算同时变化时，不把全部差异归因于 B 架构。

## 2. 数据与不泄漏边界

冻结 manifest：`sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`，当前 SHA256 `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`。

- 27 source-train / 6 development-val / 6 sealed-test。
- 仅 train/val NWB 与相应缓存可读；严格按 manifest allowlist，不能 glob 后把 test 放进统一预处理。
- 保留原 max_units_exclusive=100 选择合同，不再按结果挑单位或 session。
- 6 val 有历史研究暴露；所有首轮结论称本地 development，不称官方成绩或全新盲测。
- 6 test 维持封存。若需要最终论文的独立确认，冻结方案后由用户单独授权一次本地 test；没有 EvalAI 不是反复使用 val 作 test 的理由。

### 支持集与 query

默认 **M_activity=M_carrier=30**：同一 session 中按既有奖励 trial 筛选和时间排序得到的前 30 个 trial。不能为修复 rank/缺 cue 而偷偷从第 31 个以后补标签。

Query 只使用 trial30 之后的既有合法窗口；以实际坐标证明整个输入历史不跨回支持集，保持既有 scorer 的 trial、mask、target、last-bin 定义。若旧资料存在 M30 activity/M50 carrier/q50，须另标历史协议，不移植其数字到本次 M30/q30。

保持现有 **20 ms bin、W=50（1 秒）、2D 输出、behavior scale=5**。这比重新搬入 H1 的长窗或 M1 的 W100 更省成本，也与当前 688 loader/旧参考及 M2 的 W50 接近。

## 3. Carrier：对齐 M2 的估计器，不机械照搬 trial 起点

### 3.1 新主版本：688-MOVE-T4

对每个允许的支持 trial m，用 native `target_dir` 得方向 theta_m，不从 query 速度反推方向。用 `go_cue_time=g_m` 锚定固定运动相关窗口。

**默认窗口 [g_m, g_m+0.700)**。原因来自既有只读事件审计：688 trial 中存在不同长度的等待；没有独立 movement_onset 字段；固定 post-go 700 ms 已有构造与可靠性证据。不要把预插值的 trial[5:30] 当作 M2 的原始有效 bin。

用原始 spike times 半开区间计数：

`S_mi = # {spikes_i in [g_m, g_m+0.700)}`

`R_mi = S_mi / 35`  （35 个 20-ms 等效 bin，即 counts/20-ms-bin）。

这里不依赖全 session dense bin grid。现有 loader 的 dense grid 以第一个 spike 起始，直接切该 grid 会引入 session 相关事件对齐偏差。`S/35 = Hz*0.020`，与 M2 的 counts/bin 尺度含义一致。

构建设计矩阵并复用 M2 的核心求解：

`A_m = [1, cos(theta_m), sin(theta_m)]`

`B_hat = argmin_B ||R - A B||_F²`

`B_hat[:,i] = [b_i,a_i,d_i]`

`c_raw_i = [a_i,d_i,sqrt(a_i²+d_i²),b_i]`。

**每 trial 一行、trial 等权 OLS、无 ridge、rank(A)=3**，对齐当前 M2 的 `t4_from_trial_sums`。拟合用 float64 lstsq，最终 float32；不显式求逆。可将 S 和全为35的 lengths 传入现有纯数值核心，但独立写 688 事件与方向适配，不滥用其 FALCON NWB 读取器。

源 27 sessions 的 raw profile rows 拟合 feature-wise mean/std，std≤1e-6 则置1；目标 val 使用冻结统计。这也对齐 M2，但不同于历史 688 winsorized 统计；新版本必须独立命名、独立归一化，不能沿用旧 z-score 缓存。

### 3.2 与 M2、旧 688 的区别必须照报

| 项 | 当前 M2 | 本次 688-MOVE-T4 | 历史 688 T4 |
|---|---|---|---|
| 回归样本 | 每 trial | 每 trial | 每观测方向的 trial 均值 |
| 样本权重 | trial 等权 | trial 等权 | direction 等权 |
| 神经响应 | 原始有效 trial-relative bins[5,30) | 原始 spikes、go 后700ms | whole trial |
| 响应单位 | counts/bin | counts/20ms-bin | Hz |
| 方向 | target-center 的 atan2 | native target_dir | native target_dir |
| 4维映射 | a,d,幅度,b | 同左 | 同左 |
| normalizer | source mean/std | source mean/std | source winsorized 统计 |

共同机制相同，事件锚点与窗口保留任务实际语义。新旧差异是 estimator bundle，若将来比较得分，不可称只改了窗口。

### 3.3 低成本 CPU sanity，而不是新增 GPU 网格

在27 source sessions上只计算以下三个 descriptor 候选作诊断：旧 WHOLE、上述 MOVE700、数值更贴近 M2 的 **[g+0.100,g+0.600)、counts/25**。

- 检查第一30 trial 的事件合法性、区间包含、有效样本数、方向覆盖、rank/condition number、源尺度与 split-half 稳定性。
- split-half 按预固定规则分支持 trial；某半 rank 不足标 unavailable，不伪造0或多读支持标签。使用率/秩报告本身就是结果。
- 不读取 val decoder R² 为这些窗口选点；不扫更多时间窗或 ridge。
- **本轮 GPU 主版本仍预固定 MOVE700**。MOVE500 仅回答“机械靠近 M2 会损失多少稳定性”，不能看完分数自动升级为主版本。若源证据明显支持改主版本，先在训练前形成新版本设计。
- 区间超出 trial/缺 go/方向非法时，只能在前30支持内排除并披露。主版本任何 session 无法形成合法满秩设计则停止该版本，不默默回退 whole-trial 或扩大支持集。

绝不声称 MOVE700 已在 B 上验证有效；旧 POST700 的其他网络结果只提供设计动机。

## 4. E0：使用一个已有 activity-only 源 encoder，避免隐蔽 carrier 路径

建议 donor 为已存在的严格 source-only **B0 seed42、固定 epoch_011** encoder：

`sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_b0_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt`

其 metadata 给出 side_features=none、M30、trial_length100、W50。执行前审核实际 state/config 的输出宽度，预计 E0=50；不能只凭文件名。还要审计继承的 teacher 训练 roster，证明未训练于这6个 val/6个 test；若不闭合则停止“clean source-only”命名并报告，不用隐藏的历史暴露换便利。

- 只提取校准 encoder，冻结权重；用每 session 第一30活动 trial 缓存 E0，不把已有 decoder 权重当新 B 的初始化。
- 这种选择不需要新训 encoder，并使 c-zero 确实去掉所有方向 carrier 输入；不像把 B3S 的直接 c 置零但旧 T4 仍藏在 E0 里。
- E0 encoder 看过所有27 source sessions属于明确的预训练成本。减少 B 训练更新不等于整个系统只用了少量源数据。
- 两个 B decoder seed 默认共享同一个冻结 donor，结论为 decoder seed 稳定性，不宣称整条 encoder 独立复现。
- 不把 M2/H1 “E0内含carrier”的现存细节强行抹平。本次共同的是四维 carrier 和 proj_add 接口，而非所有 E0 都有相同信息来源。

如果 donor 不可用/有禁止的暴露，不自动新训一个大 encoder；输出需要哪个 source-only donor 或新预算才能继续。

## 5. 最小训练矩阵

共用 backbone：local causal Conv1D k5/16 → proj_add P16 → token MLP width256 → 8 learned slots → 4层 CausalPE → last-bin readout。N随session变化，用实际N/valid mask；不截到“最有用”的固定单元数。

| arm | E0 | carrier | 作用 |
|---|---|---|---|
| **B-FULL** | 真 | MOVE-T4 真 | 主候选 |
| **B-NOE0** | 0 | MOVE-T4 真 | E0 的匹配重训消融，复用实验一 |
| **B-NOC** | 真、activity-only | 归一化后常数0 | carrier 条件增益控制 |

三臂保持相同模块和参数形状，只在输入口 gate；同种子共享初始 state 字节、采样顺序、dropout RNG、更新数和优化器设置。B-NOC 的0表示无单元特异 carrier 信息，不代表 raw firing coefficients 生理上为零。

第一轮不训练 B-LEGACY、P32、W100、QueryAge、FiLM 或联合 encoder finetuning。旧 SPINT/B0/T4 权重只增加低成本参考评分；支持边界、query与聚合必须重放对齐。不同输入/训练预算的历史参考注明 system-level，不伪装成 matched decoder ablation。

若未来只想证明 B 架构相对 SPINT decoder 的因果增益，需要另做共享 calibration 输入/预算的匹配 decoder 对照；本轮三臂主要证明 B 可用和 E0/carrier 作用。

## 6. 用更新数控制成本，而不是跑完整27-session epochs

### S0：CPU缓存和吞吐预检

- 优先复用现有 `sua_exploration/cache/dandi688_subc_co_v1`，对原始 spikes 仅一次计数/构建新 carrier；不重建全部 NWB。
- 缓存 binned activity、targets、trial边界、合法 endpoint index 与每session E0/c；按需取 W50 窗口，不预展开巨大 `[all_windows,W,N]` 数组。
- B训练 batch 从一个 session 的窗口组成，跨step session等概率；trial尽可能等概率，避免最长session支配。用已有合法endpoint集合，只改变抽样密度，不改物理观测边界。
- 首选抽取不同 trial/较分散时间块，减少几乎相同的相邻窗口；选择不依赖目标幅度或模型误差。各臂使用同一封存 endpoint stream。
- 用代表性的低/中/高N session做短吞吐/显存预检，不读val分数。以最慢实际时间估算总GPU小时并加30%余量。

### S1：三个arm的 seed42 筛选

- 每臂累计 **2,000 optimizer updates**，effective batch32；共6,000 updates。
- 从一开始预设完整 schedule horizon=8,000 updates，warmup200，AdamW lr1e-4→1e-5、wd0.01、clip1、unit dropout0.1、EMA0.9995；续跑不得重启schedule/optimizer/EMA/RNG。
- 模型训练目标沿用`raw=5y`、评分`pred/5`；FP32 smoke验证后才使用统一bf16训练，不混入源算法的其他损失或蒸馏。
- 只在step2000用固定 **RAW endpoint** 做筛选；短跑EMA可能滞后，不能因此把健康训练判成失败。不挑中途最高val分。
- 每个val session最多固定2,048个、按query trials/时间块均匀的窗口；全6个session都覆盖。合计≤12,288个query点，三臂完全同坐标。

预设筛选规则：

1. NaN/Inf、输入路径无梯度、near-constant预测（pred std <1% target std且无改进）属于诊断失败，不花钱补seed。
2. 至少一个 B arm 健康、source训练损失有下降且val equal-session R²>0，则允许一次同seed延长；这只是可训练性门，不是超过SPINT的门。
3. 若所有B均比同点常数基线差，停止解释性扩展，先查接入/尺度/采样；若曲线仍明显下降，只能登记为未收敛，不能宣称架构无效。
4. B-FULL与B-NOE0无论谁略优，都保留这对，以免只确认有利E0结果。B-NOC只在 carrier 增量值得确认时晋级；否则保留其pilot结果，不能声称已完成正式carrier消融。

### S2：仅确认关键配对

- B-FULL/B-NOE0 的 seed42 从2,000接续至预设8,000 updates；固定最终 EMA checkpoint，禁止val选epoch。
- 同配对seed43从头到8,000；两seed共4个最终模型。无需立刻seed44。
- B-NOC如晋级，必须与 FULL 具有相同8,000预算；要报告可重复carrier增益则也补seed43。不得用FULL8k−NOC2k作为效应。
- 仅最终checkpoint对全部6个val的完整合法query评分；不用每500step重复全val。
- 若预算超限，优先停在已声明阶段并报告不完整，不删失败arm、不降低难session权重。

### 预算建议

第一轮GPU预算建议：S1 ≤2 GPU-hours，S1+S2 总量优先控制在6 GPU-hours；CPU缓存时间另记。以上是设计预算，不是保证实测时长。

S0发现固定步数预计超预算时，必须在任何val score前给出等比例缩减的共同更新数/完整schedule新manifest，或请求更多预算；不能见分数后给某个arm独享延长。两张GPU只在用户授权且空闲时使用；没有时限理由可停止其他任务。

## 7. 评价与决策

主指标：6个session的variance-weighted R²再等权平均；pooled R²辅报。提供每session/每seed原值、FULL−NOE0、FULL−NOC（仅同预算）、预测方差和实际训练/推理成本。

在session层配对，seed先分别报告/再平均；不把密集window当成独立统计样本。2 seeds的区间精度有限，bootstrap区间只作配对session不确定性描述，不假装估计了充分seed方差。

- **E0有效**：按实验一的 trained FULL−trained NOE0 判断，允许E0无效或有害的结论。
- **Carrier有效**：仅由匹配FULL−NOC确认；CPU tuning稳定不等于decoder效用。
- **B有竞争力**：与重放的历史系统同面比较，预算与encoder/estimator差异披露。如果只相当但结构清楚/成本可接受也可保留；预设实质提升目标可沿用+0.03，但未达到不等于实验失败。
- **证据不足**：loss仍下降/两seed符号冲突/少数session支配时报告未定，不立即开展大网格。先判断需要更多训练还是更多数据/seed，两者不可混同。

由于没有官方提交，最终正式结论靠锁定的本地测试协议：模型/窗口/normalizer/选点规则全部冻结后，另经用户授权一次性测试那6个sealed sessions。否则报告仅覆盖已暴露的val。

## 8. 最小交付

建议独立目录 `btransform_unified_v1/results/688_co_costaware_v1/<UTC>/`。

交付：source/val allowlist与无test读取证明；support/query/单位/输出列序manifest；三种carrier CPU诊断与新主版本定义；encoder provenance；cache哈希；冻结训练endpoint stream和schedule；每step/最终权重与RNG；吞吐/资源预算；逐session配对结果；第一层E0干预结果（引用实验一）；明确剩余假设与下一步裁决。

不得覆盖现有 T4 normalizer、已有 parent checkpoint、原始缓存或历史分数。执行者不是唯一工作者，须隔离文件与硬件使用；设计稿本身不启动上述任务。
