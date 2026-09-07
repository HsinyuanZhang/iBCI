# FALCON B1 Spectral Functional Carrier + J-R1 完整分析与实验计划

**日期：** 2026-09-03

**状态：** 设计候选，供独立审核；本文档不授权实现、GPU 训练、Docker 构建或 EvalAI 提交

**审核状态：** 已吸收第一轮 22 项与第二轮 13 项独立审计修正，等待第三轮交叉审核；第三轮 6 项修订已落地（work order V1）

**工作名：** **B1-SFCJ**（Spectral Functional Carrier with J-R1）

**本地数据：** `SPINT-main/data/001046`

**任务纠正：** DANDI `001046` 是 FALCON **B1 vocal**，不是 H2；FALCON H2 writing 对应 DANDI `000950`

---

## 0. 执行摘要

### 0.1 两句话研究提案

FALCON B1 要从跨日期漂移的 85 通道 RA threshold-crossing activity 重建
158 频点的鸟鸣声谱，而原始 SPINT 只从少量**无标签** calibration activity 中估计
session identity，并未正式支持 B1。我们提出先建立 matched B1-SPINT 基座，再用三条
带标签 calibration trial 闭式估计每个神经通道对 source-frozen 低维声学场的编码系数，
并通过 J-R1 将 `MLP(pool(activity))` 与 `pool(MLP(activity))` 做原生锚定残差融合。

### 0.2 当前结论

1. **不能直接把现有 M1/M2/H1 配置改成 B1。** B1 的输入、输出、调用协议、loss 和
   metric 都不同，需要 B1-specific DataModule、model wrapper、decoder 和 evaluator parity。
2. **J-R1 值得原生加入，但不能成为唯一实验臂。** 它在 M2 上对 matched native control
   给出 external `+0.01312 R2`、within `+0.01200 R2`，说明 anchored residual 是有依据的
   安全融合原则；但它没有通过相对历史 POOLED 的 worst-session 实用门，因此不能被当成
   普适真理。
3. **四维载波不应因“T4 是四维”而被默认。** 一次非 governing、预处理尚未匹配正式方案的
   B1 声谱只读探索显示，前三个主成分
   只解释约 `52.89%`，前八个约 `62.74%`，前十六个约 `71.46%`。主方法应为
   `SFC9 = 8 coefficients + intercept`；`SFC4 = 3 coefficients + intercept` 是压缩消融。
4. **158 维直接 carrier 只作诊断，不作第一轮主网络。** 它回答低维投影是不是瓶颈，
   但会引入高度相关的 159 参数/unit target-session 拟合和显著更大的注入层。
5. **必须分标签层级，并排除“模板回忆”。** 原始 SPINT calibration 是 neural-only；SFC
   使用 target calibration 声谱标签，而 B1 鸟鸣在同一 date 内高度刻板。SFC 必须同时与
   neural-free 的 target-M3 模板、冻结 A0 后的全 158 维闭式输出校正、以及不依赖 source
   pretrained model 的多 lag 全 158 维直接 ridge 比较；低秩 DR-PC8 只能作压缩诊断。
6. **12 epoch 不能直接沿用。** B1 每个训练样本是一整段 trial；若按部署匹配原则把每个
   source date 的 first M3 作为 seed support 并从 task loss 排除，三折 LODO 分别只剩
   31、13、34 条 source query trial。batch size 4 时 12 epoch 约只有 36–96 个 optimizer
   step。本文改用 source-only optimizer-step horizon screen，再把选定步数锁给全部方法臂。
7. **B1 可以合法使用 growing activity memory。** evaluator 在 trial 完成后才把整段 neural
   交给 `predict`；虽然 B1 不实际调用 `on_done`，whole-trial predict 本身提供了合法边界。
   主部署律因此是 decode-before-commit，生产实现用运行和做 O(1) 增量；SFC carrier 仍冻结
   在 M3，因为 hidden query 没有可用声谱标签。
8. **单 seed 不足以承担方法门。** fold 1 只有 13 条 training queries，三个 date 在 null 下
   全同号也不罕见。A0 horizon 和六臂主矩阵都固定 paired seeds `{42,43}`；仍明确承认只有
   三个独立 date，结论属于小样本 source selection，最终转移依赖 hidden held-out dates。

### 0.3 推荐的最小主矩阵

| Arm | Activity identity | Carrier input | Pool/fusion | 标签层级 | 作用 |
|---|---|---|---|---|---|
| `A0-NATIVE` | 原生 SPINT | literal Zero9 | `MLP(mean(pre_pool), Zero9)` | neural-only | 等宽 activity-only 基线 |
| `N-SFC4` | 相同 activity | 独立拟合的 3 PC weights + intercept + zero pad | native order | labeled M3 | native-order 四维载波 |
| `N-SFC9` | 相同 activity | 8 PC weights + intercept | native order | labeled M3 | carrier 在原生 pooling 顺序中的内容对照 |
| `J0` | 相同 activity | literal Zero9 | J-R1 | neural-only | J-R1 拓扑/联合训练对照 |
| `J-SFC4` | 相同 activity | 独立拟合的 3 PC weights + intercept + zero pad | J-R1 | labeled M3 | J-R1 四维载波 |
| `J-SFC9` | 相同 activity | 8 PC weights + intercept | J-R1 | labeled M3 | **主候选** |

另设 neural-free 的 `TPL-SRC/TPL-M3`，以及四个同 M3 标签预算、无 target backprop 的
Tier-2 闭式对照：冻结 A0 后的全频 `A0-OR158`、多 lag 全频 `DR-158-ML`、单 lag 下界
`DR-158-SL`，以及仅作低秩诊断的 `DR-PC8`。其中 `TPL-M3` 是判断方法是否真的利用 query
neural、而不只是回忆三条 calibration 鸟鸣模板的必要对照。

六个训练臂构成 `native/J-R1 × Zero/SFC4/SFC9` 的完整 2×3。对每个 q∈{4,9}：
下列简式都必须在同一个部署律 `L∈{GROWING,FIXED3}` 内计算；§10 给出带 `L` 的正式定义。

```text
carrier_gain_native_q = MSE(A0-NATIVE) - MSE(N-SFCq)
carrier_gain_jr_q     = MSE(J0) - MSE(J-SFCq)
jr_gain_zero          = MSE(A0-NATIVE) - MSE(J0)
jr_gain_sfcq          = MSE(N-SFCq) - MSE(J-SFCq)
interaction_q         = carrier_gain_jr_q - carrier_gain_native_q
                      = jr_gain_sfcq - jr_gain_zero
dimension_gain_native = MSE(N-SFC4) - MSE(N-SFC9)
dimension_gain_jr     = MSE(J-SFC4) - MSE(J-SFC9)
```

以上均按“正值更好”定义。不得用 `J-SFC9 - A0-NATIVE` 一个差值同时宣称 carrier 内容、
J-R1 和标签利用都有效。也不得把训练后 J checkpoint 的 `alpha=0` forward ablation冒充
对应 N arm：前者保留了 J-R1 联合训练造成的参数共适应，只有独立匹配训练的 N arm 才能
估计 native-order carrier gain。

---

## 1. 数据集身份与本地资产审计

### 1.1 任务身份

本地 NWB 的 `experiment_description` 为：同步 Zebra finch RA neural 与 awake-singing
song data，Neuropixels recording。本地 `falcon_challenge==1.0.2` 安装包定义：

```text
FalconTask.b1 = "falcon_b1_vocal"
n_channels = 85
out_dim = 158
recommended evaluation batch size = 1
raw neural sampling interval = 1/30000 s
```

官方 FALCON 仓库把 held-out calibration 描述为小规模 few-shot recalibration 资源；本文
使用本地已发布 calibration，不读取任何隐藏 query label。官方入口：
[snel-repo/falcon-challenge](https://github.com/snel-repo/falcon-challenge)。

### 1.2 本地文件清单

`SPINT-main/data/001046` 共 9 个 NWB，约 `1,286,932,324` bytes：

| Split | Session | Trials | 用途 |
|---|---:|---:|---|
| held-in-calib | `20210626` | 11 | source train / LODO support |
| held-in-calib | `20210627` | 29 | source train / LODO support |
| held-in-calib | `20210628` | 8 | source train / LODO support |
| held-in-minival | `20210626` | 2 | source-only LODO/local score |
| held-in-minival | `20210627` | 2 | source-only LODO/local score |
| held-in-minival | `20210628` | 2 | source-only LODO/local score |
| held-out-calib | `20210630` | 3 | formal target M3 calibration |
| held-out-calib | `20210701` | 3 | formal target M3 calibration |
| held-out-calib | `20210705` | 3 | formal target M3 calibration |

本地没有 held-out query/eval labels。held-out 性能只能在设计和模型选择完全冻结后由
EvalAI 返回。

### 1.3 每个 trial 的精确几何

所有文件共享：

| Quantity | Value |
|---|---:|
| raw neural `tx` | `[27000, 85]` per trial |
| neural sampling | 30 kHz |
| `tx` values | binary threshold crossings `{0,1}` |
| trial duration | `0.89996667 s` |
| audio | 25 kHz |
| target spectrogram | `[158, 880]` per trial |
| spectrogram frame spacing | `1 ms` |
| frequency range | `292.97–7958.98 Hz` |
| evaluator-valid frames | indices `[90,790)`, exactly 700 |
| valid coordinates | `700 × 158 = 110600` per trial |

声谱时间从相对 trial 的 `10.24 ms` 到 `889.24 ms`；有效评分帧对应大约
`100.24–799.24 ms`。所有 158 个频点共享相同 temporal mask。

### 1.4 B1 evaluator 契约

B1 不是 M1/M2/H1 式逐 bin 调用。FALCON evaluator 会：

1. 收集一整个 trial 的 30 kHz neural observations；
2. 调用一次 `decoder.predict([B,85,27000])`；
3. 要求 `predict` 返回该 trial 的二维 `[158,880]`（frequency × time）预测；evaluator 随后
   按自己的 concatenate/transpose 逻辑组装成内部 time-major 表示；
4. 仅在 700 个有效时间帧上评分；
5. 对每个 trial 的 target 和 prediction **分别做全矩阵 min–max normalization**；
6. 计算 trial MSE，再先在 session 内平均，最后对 session 等权平均。

安装环境中 governing evaluator 的逻辑是：

```python
normalize_signal(x) = (x - min(x)) / (max(x) - min(x))
trial_error = MSE(normalize(target_trial), normalize(pred_trial))
session_error = mean(trial_error)
headline = equal-session mean(session_error)
```

因此本文所有 gain 都定义成：

```text
gain(reference -> candidate) = MSE(reference) - MSE(candidate)
```

**正值表示改善。** 不得把 M1/M2 的 R2 符号习惯带进 B1。

B1 还有一个与 M2 continual 不同的关键部署事实：安装版 evaluator **不会**在 B1 分支
调用 `decoder.on_done(dones)`，但它只在检测到 trial 结束、收齐该 trial 的 27000 个
neural sample 后才调用一次 `predict(whole_trial)`。因此这次 `predict` 调用本身就是一个
合法、可观测的完成边界。正确的因果事务顺序是：

```text
decode current whole trial from memory of completed past trials
-> return current prediction
-> commit current trial activity into memory for the next trial
```

H2 则显式由 `on_done(dones)` 送达边界；M2 continual 没有同等合法的完成信号。本文可以
在 B1 部署 growing activity memory，但不得把接口事实误写成“B1 也调用 on_done”，也不得
把 current trial 先写入 memory 再解码自身。

### 1.5 本地 evaluator 的三个工程风险

1. `normalize_signal` 没有 epsilon；常数 prediction 会产生非有限值。模型和本地 scorer
   必须显式拒绝 prediction range 为零或非有限，但不得改写官方 metric。
2. B1 的 raw neural 与声谱采样率不同。训练 loader 必须先通过 NWB timestamps 证明
   `i//30` 位置律；deployment 只收到 array，必须执行该冻结位置律，不能声称现场重读
   timestamps，也不能用“凭经验裁 20 点”代替 Stage 0 audit。
3. 安装版 local `hash_dataset` 对这些文件产生 undotted `20210626`，而 B1 evaluator map 使用
   dotted `2021.06.26`；完整 local `evaluate()` 会在 session lookup 处失败。设计必须直接
   复现 governing metric，并让 decoder payload resolver 接受两种合法 tag，不能把这个上游
   bug 误判成模型失败或伪称 local end-to-end parity 已过。

---

## 2. 当前 SPINT 做了什么，以及为什么 B1 尚未被支持

### 2.1 原始 SPINT identity 路径

现有 `SpintModel` 对 calibration activity 的主要算子是：

```text
calibration [B,M,T,N]
  -> transpose [B,M,N,T]
  -> shared temporal MLP fc_id_in
  -> mean over M trials
  -> fc_id_out
  -> activity identity E^A [B,N,W]
```

当前 neural window `x [B,W,N]` 转成 `[B,N,W]` 后与 `E^A` 相加，通过共享的
`fc_in` 形成 unit tokens。learned output queries 再对 unordered unit token set 做
cross-attention。这给出三项关键性质：

- unit 轴置换等变/输出置换不敏感；
- identity 来自 calibration activity，不需要 target weight update；
- 同一个 per-unit temporal encoder 跨 channel 复用。

### 2.2 当前仓库支持边界

当前 SPINT 主仓库正式配置和 `FalconLitModule` session registry 只覆盖：

```text
M1 / M2 / H1
```

缺少：

- `falcon_b1` DataModule/config；
- B1 session registry；
- 30 kHz whole-trial neural→880-frame target 对齐；
- B1-specific full-trial model output；
- official min–max spectrogram loss/metric；
- B1 decoder `predict([B,85,27000])`；
- B1 local package/Docker parity。

现有 `spint_sample.py` 的 CLI choices 也只列 `h1/m1/m2`。因此“先复用现有 SPINT 再看
carrier”在工程上不可行；必须先建立一个可独立通过 evaluator parity 的 B1 substrate。

### 2.3 H1 CarrierID 提供了什么可复用思想

H1 CarrierID 的已实现结构是：

```text
per-trial activity -> carrier_pre_pool -> mean_M
concat normalized [N,4] carrier
-> carrier_post_pool -> identity
```

它证明了 repository 内已经存在以下设计纪律：

- per-unit functional descriptor 是 deployment input，不是 target optimizer state；
- carrier 与 activity identity 必须维持 unit 对齐；
- Full 与 width-matched zero-carrier control 需要相同拓扑；
- source-frozen normalization，target 只投影/闭式拟合。

但 H1 CarrierID 是“先 activity pool，再过 carrier-conditioned nonlinear MLP”，它没有
比较 pooling 与 nonlinear operator 的次序，也不能直接复制成 B1 J-R1。

### 2.4 J-R1 的精确已测算子

令每个 calibration trial 的 activity 表征为：

```text
u_m = pre_pool(A_m)
```

原生分支：

```text
h_n = post_pool(concat(mean_m(u_m), side))
```

晚池化分支：

```text
h_p = mean_m(post_pool(concat(u_m, side)))
```

J-R1：

```text
h_J = h_n + tanh(alpha) * (h_p - h_n), alpha = IEEE +0
```

所以：

- `J-MEAN` 才是纯 `pool(MLP(per-trial))`；
- `J-NATIVE` 是 `MLP(pool(per-trial))`；
- `J-R1` 同时保留二者，以 native 为锚点学习晚池化 residual。

### 2.5 J-R1 的现有证据边界

M2 AJPF V4 的 matched external method contrast：

```text
J-R1/UNCAPPED - J-NATIVE/UNCAPPED
mean       = +0.0131178548 R2
positive   = 4/6
worst      = -0.0098362127
bootstrap  = [+0.0007254, +0.0250700]
```

within 为 `+0.01199537`, `7/7`。直接 `J-MEAN` 比 `J-NATIVE` external 低约
`-0.03223`。这支持“锚定残差可以提取少量 transferable late-branch 信息”。

但 J-R1 相对历史 POOLED 虽平均为正，某个 external session 为 `-0.07828`，因此总体
paper-success 为 false。B1 可以把 J-R1 作为有依据的融合先验，但不得预写成成功方法。

---

## 3. 问题抽象、候选空间与收敛

### 3.1 真正的问题

B1 的跨日期失败可分解成三个可能瓶颈：

1. **activity identity 不足：** 三条 target calibration activity 无法告诉模型每个新
   channel 对声学输出的功能关系；
2. **carrier 估计不足：** 158 维声谱太高，M3 下直接估计不稳定；
3. **consumer 不匹配：** 即使 carrier 有内容，`MLP(pool)` 与 `pool(MLP)` 的非交换性
   可能使载波被错误消费。

SFC 解决第 1–2 项；J-R1 只处理第 3 项。两者必须通过对照分开识别。

### 3.2 原始候选清单

在收敛前考虑过以下候选：

| Candidate | 价值 | 主要问题 | 当前路由 |
|---|---|---|---|
| activity-only B1-SPINT | 必需基座 | 无功能标签 | **保留 A0** |
| learned session embedding | 简单 | 新 target date 无 table entry | 拒绝 |
| SFC4 | 低带宽、T4-like | q=3 可能过度压缩 | **保留消融** |
| SFC9 | 表达/稳定折中 | 比四维多 5 个数 | **主候选** |
| SFC17 | 较高声学覆盖 | target M3 稳定性未知 | CPU ceiling only |
| Direct159 carrier | 不压缩声谱 | 高共线、参数/条件数大 | CPU diagnostic |
| NNMF acoustic basis | 非负、可解释 | component 不嵌套，重尾尺度敏感 | 后续 basis-family test |
| PCA acoustic basis | 稳定、q3/q8 可嵌套 | components 有符号 | **V1 basis** |
| zero-init additive carrier | 简单、从 native 起点 | 不测 pooling noncommutativity | 备选简化臂 |
| J-R1 fusion | 已有正向机制证据 | 多一个 scalar，非普适 | **主融合** |
| direct target readout ridge | 强 few-shot 标签基线 | 不是 permutation-invariant carrier | **必须对照** |
| target-M3 声谱模板 | 检测 neural-free 模板回忆 | 不使用 query neural | **必须对照 TPL-M3** |
| source-only 声谱模板 | 检测 A0 是否真的解码 | 不使用 target calibration | **必须对照 TPL-SRC** |
| target backprop fine-tune | 容量高 | 改变 few-shot 合同、过拟合风险 | V1 禁止 |
| causal growing activity memory | B1 有合法 whole-trial 完成边界 | 需训练/部署同序并防止当前 trial 自泄漏 | **主部署律** |
| continual SFC carrier | 可在线更新功能描述 | hidden query 没有声谱标签，无法重拟合 SFC | V1 拒绝 |
| output smoothing | 声谱有连续性 | 容易重复输出后处理路线 | V1 禁止 |

### 3.3 收敛后的四项贡献候选

1. **B1-SPINT substrate：** 首次把 permutation-invariant calibration identity 扩展到
   whole-trial song spectrogram reconstruction。
2. **SFC：** 用 source-frozen 声学基定义每个 unit 的跨 session 功能坐标，target M3
   仅闭式拟合。
3. **J-R1 consumer：** 在相同 carrier 下比较 `MLP(pool)` 与 `pool(MLP)`，用 native
   anchor 避免纯 late replacement 的 OOD。
4. **boundary-legal growing memory：** 在 B1 的 whole-trial 完成边界上先解码、后提交，
   让 activity identity 随已完成 trial 增长；SFC 始终冻结在 released M3 标签预算。

functional-carrier 主张要求第 2 项至少在 native 或 J-R1 的 matched reference 上过内容门；
J-R1 作为 carrier consumer 的更强主张还要求 `jr_gain_sfc9` 为正，并结合 interaction 阅读。
任何 Tier-2 neural-decoding 主张还要求 selected SFC 胜过 `TPL-M3`；否则只能说模型吸收了
calibration target template，不能说它从 query neural 解码了模板之外的信息。
单独完成第 1 项是数据集扩展；单独 `J0 > A0` 是 pooling/fusion 结果，不是 functional
carrier 结果；单独 growing 优于 FIXED3 是 memory 结果，也不能冒充 SFC 内容证据。

---

## 4. 只读预分析：目前知道什么、还不知道什么

### 4.1 声谱低维性

一次只读 exploratory probe 看了全部三个 source session，并使用了临时的 `log1p` 与
per-trial scalar min–max；这与 §6.2 的正式 preprocessing 不同，因此不把其 PCA 表格作为
receipt 或门控证据。它只提供一个非 governing 的设计提示：前三个分量约覆盖 52.9%，
前八个约 62.7%，所以 q=3 不是“明显足够”，q=8 值得作为预注册主维度。Stage 0B 必须按
正式 `log(raw)`、per-frequency source standardization 和严格 LODO 从头重算，正式结果可以
推翻这个提示。

### 4.2 初步 carrier 可重复性

另一个只读 probe 在同一临时 q=8 basis 上：

- 30 kHz tx 固定聚合为 1 ms counts；
- 5 ms centered box smoothing；
- official valid frames；
- source session 内 first M3 对 next M3；
- ridge `lambda=1`，intercept 不惩罚；
- neural/acoustic lag `0 ms`。

q=8 权重的 flattened split-half correlation 为：

```text
20210626: 0.632
20210627: 0.769
20210628: 0.627
```

per-channel cosine 的 session median 为约 `0.697 / 0.863 / 0.689`。10–60 ms lag 下
仍存在同量级稳定性，没有一个 lag 在所有 session 上明显统治。

这些数值说明 M3 拟合不是显然不可构造，但不能证明 decoder gain，也不能据此在 target
session 上挑 lag。

### 4.3 Neural-free template probe

第二轮只读审核提出这一问题后，本文在本地九个 NWB 上按 official raw min–max metric、
700 valid frames 独立复算，得到以下非 governing 数字：

| Validation date | `TPL-M3` MSE | `TPL-SRC` MSE |
|---|---:|---:|
| `20210626` | 0.00039 | 0.00115 |
| `20210627` | 0.00055 | 0.00065 |
| `20210628` | 0.00065 | 0.00062 |

上表未经四舍五入的 `TPL-M3` 是 `0.0003904354 / 0.0005532505 / 0.0006458219`，`TPL-SRC`
是 `0.0011549066 / 0.0006527209 / 0.0006214981`，与审核测量一致。三个 held-out date 内，
各自 M3 calibration trial 的两两 spectrogram MSE 独立复算范围为
`0.0002679124–0.0008027576`；这比审核摘要的近似范围稍宽，但不改变模板高度相似的判断。
这说明 B1 声谱具有强 date-local 模板性，Tier-2 的正结果可能来自
calibration target template，而不来自 query neural。Stage 0B 必须在 immutable inventory
和正式 split 上独立复算这些数字；复算前它们只用于解释为什么 `TPL-M3` 是必要对照，
不能作为任何方法的已封性能结果。

### 4.4 当前未知量

- activity-only B1-SPINT 是否能形成有竞争力的 source LODO baseline；
- exact official-metric loss 是否稳定；
- q=8 carrier 相对 J0 是否改善三折 LODO；
- q=3 是否已经足够；
- J-R1 在 from-scratch B1 训练中是否仍有益；
- target held-out date 上是否转移；
- neural-free TPL-M3 是否已经吃掉全部 M3 gain；
- label-aware multi-lag direct ridge 是否已经吃掉模板之外的剩余 gain。

---

## 5. B1-SPINT substrate 的确切设计

### 5.1 固定输入表示

每个 raw trial 是 `[27000,85]` binary tx。训练期先用 NWB timestamps 证明位置律成立；
正式 decoder 的 `predict` 收不到 timestamps，因此部署只能使用被冻结的位置算子：

1. sample i 归入 1 ms bin `i // 30`，聚合 threshold crossings；
2. 产生 `[900,85]` 1 ms count sequence；
3. 不做 target-session mean/std 重估；
4. 只允许一个 source-frozen global scalar scale；必须在 LODO train dates 上拟合并在
   validation/target date 上冻结；
5. 不把 85 个 channel 当作跨 session 固定 unit identity table。

Stage 0A 必须以 timestamps 验证每个 trial 恰好 900 bins、每个 1 ms 区间确含 30 个采样点，
再冻结该位置律。任何 round/crop/pad 都要形成显式 receipt。不得把 spectrogram 的
`10.24 ms` 起点误当作 neural trial 起点，也不得在 deployment wrapper 中假装仍能读到
NWB timestamps。

### 5.2 模型几何

V1 建议保留 SPINT 的核心 unit-set 结构，但将输入长度与输出长度解耦：

```text
neural trial       [B,900,85]
calibration        [B,M,900,85], M=3 for matched few-shot
activity identity  [B,85,900]
unit token         [B,85,D]
frequency queries  [1,158,D]
standardized-log output [B,880,158]
```

推荐固定：

```text
D = 512
cross-attention layers = 1
heads = 64
input projection = Linear(900,D) -> ReLU -> Linear(D,D)
output projection = Linear(D,880), followed by fixed inverse-standardize + exp for scoring
```

158 个 learned frequency query 直接位于 D 维；不再复用 `fc_in` 把一个 `[158,900]`
query template 投影，因为 input/output temporal lengths 已不同。

### 5.3 Activity identity

对每个 calibration trial 和 unit：

```text
u_mi = pre_pool(A_mi)            # shared temporal MLP, 900 -> H
u_bar_i = mean_m(u_mi)
E^A_i = native_post_pool(u_bar_i) # H -> 900
```

`E^A` 加到当前 trial 的 `[N,900]` waveform 后再过 `fc_in`。这保留原始 SPINT 的
“calibration 产生 waveform-space identity correction”语义。

### 5.4 Whole-unit dropout

训练可使用与 SPINT 一致的 whole-unit dropout，但所有 matched arms 必须共享：

- 同一 batch；
- 同一 sampled probability；
- 同一 `[B,N]` mask；
- activity、carrier 和 current neural branch 相同的 unit mask。

carrier 不能成为 dropped unit 的旁路。若 unit 被 mask，`u_mi`、carrier_i 和当前 x_i
都必须对该 unit 不产生可用 token。

### 5.5 训练 loss 与官方 metric

定义 official valid mask `V`，对每个 trial：

这里的 `pred` 是网络 standardized-log head 经过 source-frozen inverse standardization 和
`exp` 后的 raw-spectrum prediction，`target` 是 NWB raw spectrum：

```text
mm(x; V) = (x - min_V(x)) / (max_V(x) - min_V(x))
L_metric = mean_V((mm(pred;V) - mm(target;V))^2)
```

训练主 loss 使用该 masked per-trial metric surrogate，并同时记录：

- raw prediction range；
- normalized MSE；
- raw/log-space auxiliary MSE（仅描述，不选择 checkpoint）；
- finite/zero-range count。

不得用普通元素 MSE 训练、再只在最后换 official min–max metric 而不披露 loss mismatch。
常数 prediction 会令 official normalization 产生 NaN，而 sklearn 的
`mean_squared_error` 会直接抛错，使整个提交失败；它不只是一个可忽略的非有限样本。
若 exact min–max loss 数值不稳定，任何稳定化版本必须成为独立、source-only substrate
实验；不能在看到 SFC 结果后修改。

### 5.6 A0 parity tests

在任何方法臂之前，A0 必须通过：

1. DataModule target、mask、trial/session 顺序与 installed FALCON loader 一致；
2. B1 batch size 固定为 1；decoder `predict` 必须返回二维 `[158,880]`。安装版 evaluator
   会沿 axis 1 concatenate 后 transpose，三维 `[1,158,880]` 会组装成错误形状并失败；
3. 本地 parity 直接调用 `compute_metrics_spectrogram_distance`。当前安装版完整
   `evaluate()` 对本地 tag 有已知格式 bug：`hash_dataset` 产生 `20210626`，而 B1 map 使用
   `2021.06.26`，因此不得把本地 end-to-end `evaluate()` 宣称为可通过的门；
4. batch size 1 与训练/离线 batched decode 数值误差不超过预先冻结容差；
5. reset 不携带上一 session/trial 状态；
6. decoder `reset(dataset_tags)` 同时接受本地 undotted `20210626` 和远端可能使用的 dotted
   `2021.06.26`，并映射到同一个只读 payload；未知/重复 tag fail closed；
7. package 后 calibration features 与训练期 M3 activity identity 完全同构。

---

## 6. Spectral Functional Carrier（SFC）

### 6.1 Carrier 的语义

SFC 不直接预测声谱。它描述每个 neural channel 如何编码一个 source-defined 低维
acoustic field。

在 source-frozen acoustic basis 中：

```text
z_q(t) = B_q(spectrogram(t)), q in {3,8}
```

对每个 target session、每个 unit i，在三条 calibration trial 的 valid frames 上闭式拟合：

```text
r_i(t - lag*) = b_i + w_i^T z_q(t) + epsilon
```

其中：

- `r_i` 是 timestamp-aligned 1 ms count 的固定 5 ms rate view；
- `lag*_fold` 只用该 fold 的 training dates 选择：在每个 training date 内用 first M3 拟合
  date-specific carrier，再在该 date 的 calibration 4..N 上评分，最后对两个 training date
  等权聚合；不要求跨 date 对齐 channel，也不把一个 date 的 per-channel map 应用到另一个
  date；final lag 才以相同的 per-date-fit/within-date-score 规则在全部三个 source dates 上
  选择一次；
- ridge `lambda=1` 只惩罚 `w_i`，不惩罚 intercept；
- target 上无 backprop、无 basis 重估、无 normalizer 重估；
- query trial 不进入 carrier。

### 6.2 Source-frozen acoustic basis

V1 选择 PCA 而非 NNMF，理由不是 PCA 必然更好，而是本次核心需要一个干净的
嵌套维度消融：

```text
B_3 = first 3 components of the same fitted B_8 basis
B_8 = one source-frozen rank-8 basis
```

预处理冻结为：

1. raw spectrogram 必须 finite 且严格正；当前九个文件的最小值恰为 `1.0`，Stage 0A
   必须登记这个已含正偏移的存储事实；
2. 使用 `log(raw_spectrogram)`，不再额外 `log1p` 加第二个 pseudocount；
3. 仅用 LODO training dates、official valid frames 拟合 per-frequency mean/std；
4. 标准化后拟合 deterministic full-SVD PCA；
5. component sign 按 source-only 最大绝对 loading 为正固定；
6. 保存 mean/std/basis/sign/order SHA。

不得在三折中用全 source basis 再宣称 validation date unseen。每一 LODO fold 的 basis
必须只用另外两个 source dates；final official model 才能用全部三个 held-in dates。

网络 head 和所有闭式 baseline 的输出空间统一为 standardized log spectrum。评分前必须按
source-frozen per-frequency mean/std 逆标准化，再用 `exp` 返回 NWB raw spectrum，最后才
交给 official per-trial min–max metric。训练的 governing loss 也在这个 raw-space prediction
上计算；standardized-log MSE 只作辅助读数。禁止让某个 arm 在 log space 评分、另一个在
raw space 评分。

### 6.3 固定九维接口

全部 carrier-aware arms 使用相同的 `[B,N,9]` 网络接口和相同参数量：

```text
Zero9 = [0,0,0,0,0,0,0,0,0]
SFC4  = [b3,w31,w32,w33,0,0,0,0,0]
SFC9  = [b,w1,w2,w3,w4,w5,w6,w7,w8]
```

虽然 `B_3` 使用 `B_8` 的前三个 basis vector，SFC4 必须在 target M3 的 `z_3` design 上
**独立重拟合**其 intercept 和三维 ridge weights；它不是把 SFC9 的后五个系数截掉。SFC4
和 SFC9 还各自使用仅由 fold training dates 拟合的 coefficient mean/std，再共同 pad 到
九维网络接口。这样 §10 的 SFC4/SFC9 才是 matched dimensionality comparison。

拟合后的每个坐标再使用对应 q 的 source-only、跨 training date/unit 的 mean/std 标准化。Zero9
是网络边界上的 literal zero，不是 raw coefficient zero 经过 source normalization 后的值。

这样 `A0-NATIVE / N-SFC4 / N-SFC9 / J0 / J-SFC4 / J-SFC9` 都具有相同
`Linear(9,...)` 宽度。`A0-NATIVE` 与 `J0` 在该入口接收 literal Zero9；因此 carrier
对照不会把 SFC9 的收益解释成“参数更多”。

这里的 `A0-NATIVE` 指**函数上**的 native-order control，不是声称它与尚未扩宽的原始
B1-SPINT 拥有相同 state-dict 拓扑。必须另做一次 zero-column expansion parity：CPU
float64 参考算子要求位级相同；真实 GPU 因 GEMM K 维改变可能选择不同 kernel，不要求
bitwise，预先冻结 prediction max-abs `<=2e-6`、trial-MSE abs `<=2e-8`。之后主矩阵只使用
等宽 A0，避免参数量混杂。

### 6.4 Direct159 的位置

Direct159 定义为：

```text
carrier_i = [b_i, w_i1, ..., w_i158]
```

它只在 Stage 0/机制阶段回答：低维 basis 是否丢掉关键 unit-acoustic mapping。

第一轮不把它加入主 GPU matrix，原因：

- 158 频点高度相关；
- M3 的 2100 valid bins 不等于 2100 independent observations；
- 每 unit 159 个系数，85 units 约 13,515 个 session-specific descriptor values；
- 注入层参数量与 SFC9 不再匹配；
- 若它失败，无法区分条件数、标签效率和 consumer capacity。

V1 只在至少一个低维 SFC arm 已过内容门后的 Stage 3 把 Direct159 作为机制 ceiling 计算，
用于判断低维压缩还损失多少 unit-acoustic encoding；它不能参与 V1 产品选择。若 SFC4/9
都失败，V1 不再补跑 Direct159 来救线，更不能据此临时开启高维 GPU arm。任何高维
successor 都需要独立动机、work order 与 equal-parameter control。

### 6.5 Neural-free 模板与同标签预算闭式基线

B1 鸟鸣在同一 date 内可能高度刻板。只比较带标签方法与 neural-only A0，会把“记住三条
calibration 鸟鸣的平均模板”误写成 neural decoding。因此模板是 governing comparator，
不是可选 sanity check。

#### 6.5.1 `TPL-SRC` 与 `TPL-M3`

```text
TPL-SRC_fold 成员 = outer-training dates 全部 held-in-calib raw spectrogram
TPL-M3_date  成员 = 该 date chronological first 3 calibration raw spectrogram
```

每个 TPL 同时预注册三种 neural-free 聚合，均在 raw spectrogram 坐标上计算、再交给
原样 official per-trial min–max metric；不得先把每条 calibration target 单独 min–max
后再聚合：

```text
raw-mean : 逐坐标算术均值
log-mean : log 空间均值（几何均值）= exp(mean(log(raw)))
median   : 逐坐标中位数
```

三者数值全部报告。所有 TPL 门（Stage 1 gate、§10.5、Outcome A/E）使用三者中 **MSE
最低者**，记为 `TPL-SRC-BEST` / `TPL-M3-BEST`（合称 `TPL-*-BEST`）。只读复算（official
metric，700 帧）显示算术均值是弱化 15–40% 的稻草人：TPL-M3 raw-mean/median =
`0.000390/0.000350`、`0.000553/0.000384`、`0.000646/0.000511`；TPL-SRC
raw-mean/median = `0.001155/0.000778`、`0.000653/0.000521`、`0.000621/0.000383`。

二者都输出同一个固定 `[158,880]` 模板给该 date 的每条 query，完全不读取 query neural。
`TPL-SRC` 不使用任何 target-date calibration，属于 target Tier 1/neural-free
substrate control；`TPL-M3` 使用相同 M3 target labels，属于 Tier 2，是 selected SFC
是否提供模板回忆之外 neural 信息的必要对照。模板 body、成员 target SHA、聚合顺序和
最终 prediction SHA 必须写入 receipt。

#### 6.5.2 `A0-OR158`：冻结 source model 的全频输出校正

冻结 A0 的全部参数。对 target M3 calibration 的 A0 valid-frame、inverse 之前的
standardized-log prediction `p(t)∈R^158` 与同空间 target `y(t)∈R^158` 拟合：

```text
y(t) = A p(t) + b
A in R^(158×158), b in R^158
```

只做闭式 ridge，不反传、不创建 target optimizer state。A0 对三条 calibration trial 的
prediction 统一使用**包含这三条 trial 自身 activity 的同一个 self-contained M3 pool**；
这是可复现的 calibration-time convention，但与 hidden query 不包含自身 activity 的部署
条件并不完全同构，必须在结果中披露，不能把它包装成严格 causal query replay。

M3 时 `GROWING` 与 `FIXED3` 尚未分叉，因此只拟合一份校正；在 query stream 上必须分别
应用到 A0 的两种 memory-law prediction。与某个 selected SFC 比较时，使用和该 SFC 相同
部署律的 `A0-OR158` 预测，不能跨 memory law 取更弱对照。

#### 6.5.2b `A0-RT`：A0 + M3 residual template

冻结 A0 的全部参数。在 standardized-log 空间对 M3 calibration 拟合逐坐标残差模板

```text
c(t,f) = median_m[y_m(t,f) − p_m(t,f)]
```

同时报告 mean 版本 `c_mean(t,f) = mean_m[y_m(t,f) − p_m(t,f)]`。预测为 `p(t)+c(t)`，无 λ、
无反传。A0 对 calib trial 的预测使用 §6.5.2 同一 self-contained M3 pool 约定，并按
§6.5.2 同一部署律应用到 query stream。`A0-RT` 与 `A0-OR158`、`DR-158-ML` 同属 Tier 2
强基线。

#### 6.5.3 `DR-158-ML` 与 `DR-158-SL`

`DR-158-ML` 是 governing direct-label comparator。它把七个预注册 lag
`{0,10,20,30,40,50,60} ms` 的 85 维 5 ms box 平滑 1 ms rate view `r_i`（与 SFC 相同）
堆叠为 595 维输入，再直接闭式拟合全部 158 个 standardized-log 频点：

```text
x_ML(t) = concat[r(t), r(t-10), ..., r(t-60)] in R^595
y_158(t) = Ridge([x_ML(t),1])
```

official valid 区间从 frame 90 开始，所以这七个 nonnegative lag 不需要伪造左边界。
`DR-158-SL` 只用某一个 85 维 lag，并在相同 M3-LOO 内联合选择 lag/λ；它作为低容量下界
单独报告，不能代替 `DR-158-ML` 与 SFC 做主要公平比较。两者都不使用 source neural
network，也不受 PCA rank-8 reconstruction ceiling 限制。

#### 6.5.4 `DR-PC8`：匹配多 lag 输入的压缩诊断

`DR-PC8` 使用与 `DR-158-ML` 完全相同的 595 维 5 ms box 平滑 1 ms rate view `r_i` 输入，但只拟合 source-frozen z8，再按
§6.2 逆 PCA、逆标准化和 `exp` 回到 raw spectrum。这样它与 `DR-158-ML` 的差异只含输出
rank ceiling，不再混入 neural temporal context；它仍只作低秩诊断，不是唯一公平基线。

#### 6.5.5 Target-M3 内部 LOO 超参数合同

`A0-OR158`、`DR-158-ML`、`DR-158-SL`、`DR-PC8` 的 λ 不再通过跨 date channel mapping
或 A0 in-distribution residual 选择。对每个 validation/held-out date，严格只在其 released
M3 内做三折 leave-one-calibration-trial-out：每次用两条 calibration trial 拟合、在第三条
上按完整 inverse-to-raw official metric 评分，三条等权平均后从预注册 grid
`{1e-4,1e-3,...,1e4}` 选择 λ；`DR-158-SL` 同时选择 lag。平局选择更大的 λ，再选择更小的
nonnegative lag。随后用全部 M3 重新闭式拟合并冻结，query label 永不进入选择或 refit。

这一规则对 source validation date 和最终 held-out date 完全相同，使用已发布 calibration
标签而不读取 query。它们不得借用 SFC 的 `lambda=1` 或 SFC 的 source-selected lag。

若 `TPL-M3`、`A0-OR158` 或 `DR-158-ML` 等于或优于 selected SFC，正确结论分别是“模板
回忆已经解释结果”或“少量标签支持直接输出适配”，而不是 SFC/J-R1 的结构性能成功。
SFC 仍可能有参数量、结构或部署优势，但必须单独量化。

---

## 7. B1 中的原生 J-R1 融合

### 7.1 相同 carrier、只改变 operator order

对 arm q 的每 trial activity embedding `u_mi` 与相同 carrier `c_qi`：

```text
h_n(q) = post_pool(concat(mean_m(u_m), c_q))
h_p(q) = mean_m(post_pool(concat(u_m, c_q)))
h_J(q) = h_n(q) + tanh(alpha_q) * (h_p(q) - h_n(q))
```

这使 J-R1 只比较：

```text
MLP(pool(activity), carrier)
vs
pool(MLP(activity, carrier))
```

两个分支使用完全相同的 carrier、unit order、trial order 和 dropout mask。不能把 carrier
只放进 late branch，否则 `h_p-h_n` 同时改变 pooling placement 与信息内容。

### 7.2 初始化与梯度可达性

强制：

```text
alpha_q = IEEE +0
```

carrier-aware `post_pool` 从 A0 native weights 扩展，新增 carrier columns 初始化为精确零。
因此第一步前：

```text
prediction(A0) == prediction(N-SFC4) == prediction(N-SFC9)
               == prediction(J0) == prediction(J-SFC4) == prediction(J-SFC9)
```

但这不会造成双零死梯度：

- `h_p-h_n` 因 nonlinear/pooling noncommutativity 一般非零，alpha 第一批可得梯度；
- carrier columns 位于 native `h_n` 主路径，即使 alpha=0 也能从第一步获得梯度。

禁止把模型改写成：

```text
h = h_native + tanh(alpha) * P(carrier)
alpha = 0 and P = 0
```

该写法使 alpha 与 P 在第一步同时零梯度。

### 7.3 J0 的解释

`J0` 使用 literal Zero9，但保留每 trial late branch。因此：

- 对 q∈{4,9}，`MSE(A0)-MSE(N-SFCq)` 测 native-order carrier 内容；
- 对 q∈{4,9}，`MSE(J0)-MSE(J-SFCq)` 测 J-R1 carrier 内容；
- `MSE(A0)-MSE(J0)` 测 Zero9 条件下的 J-R1 operator/一标量 capacity 与 matched joint training；
- `MSE(N-SFCq)-MSE(J-SFCq)` 测相同 q 条件下的 J-R1 效果；
- 上述两组差值之差测 carrier × J-R1 interaction；
- `MSE(J-SFC4)-MSE(J-SFC9)` 测额外五个声学坐标的价值。

所有条目在上文公式中均以 `MSE(reference)-MSE(candidate)` 计算，避免方法名差值的正负
歧义。J0 不应被称为“完全相同参数的 A0”；它多一个 scalar alpha，且内部计算两条
identity branch。这个差异必须披露。

### 7.4 机制控制

主矩阵过门后，在冻结的 selected carrier-aware checkpoint 上做 forward-only controls；
RS/LS/intercept 可用于 N 或 J，alpha/J-MEAN 只用于 J-R1：

- `RSq`：selected carrier 的 unit rows 在 session 内固定 permutation；
- `LSq`：support spectrogram trial/time blocks 做预注册 derangement 后重新闭式拟合；
- `intercept-only`：只保留 b；
- `alpha=+0`：严格 native branch（仅 J-R1）；
- `J-MEAN`：纯 late branch，仅作机制，不作为产品候选（仅 J-R1）。

若 RSq/LSq 与正确 selected SFC 相同或更好，不得声称“unit 的声学编码关系”是增益来源。

### 7.5 B1 的 causal growing-memory 部署律

三条 released calibration trial 形成初始 activity pool，SFC 也只用这 M3 的声谱标签拟合。
之后每个 query trial 按 §1.4 的完成边界执行 decode-before-commit：第 q 个 query 的预测只
能看到 M3 和此前已经完成的 query activity；当前 q 的 activity 只能供 q+1 及以后使用。
query 声谱标签永不进入 SFC、selector、模型或 memory。

V1 的 governing law 是 `UNCAPPED` running mean；它不保存无限 raw tensor，只保存审计所需
成员摘要/可选缓存与两个运行和。source training 见到的最大 pre-decode K 必须写入 receipt；
若 official target K 超出该范围，标为 `CAUSAL_CARDINALITY_OOD`，但不得看见 target 长度后
临时选择 cap。`FIXED3` 是唯一预注册的非增长对照，V1 不扫 cap30/60/指数衰减。
三折的 `K_train_max` 是 `28/10/28`，final all-source model 的 `K_train_max=28`；source
memory-law 选择只使用各 fold 的 in-range query，超出部分保留为预注册 stress test。

训练期必须使用相同的 chronological growing law，而不是只见静态 M3、部署时再突然扩池。
每个 training date 的 first M3 是 seed support；其余 query 按原顺序逐条形成 task loss，
并在完成该条 loss 后才提交其 activity。validation 使用 §8.1 的完整 held-out chronological
query stream。额外记录同 checkpoint 的 `FIXED3` forward ablation，但它只是部署律消融，
不能替代 matched growing training。

训练与参数冻结后的部署必须明确区分：训练时 `pre_pool/post_pool` 参数每个 optimizer step
都会变化，因此每个 query forward 都从该样本所绑定的原始历史 activity stack 重新计算
全部 K 条 `u_i` 并对完整路径反传；禁止跨 optimizer step 缓存 detached/stale embedding。
这使训练单样本 identity 成本为 O(K)，但 K 路径只是小型 Linear/MLP。batch 4 可以混合
不同 date，因为每个样本携带自己的只读 activity history/pool digest，memory 不依赖模型
预测；各样本的 forward/gradient 仍完全独立。§7.5.2 的 O(1) running sum 只在模型参数冻结
后的 validation/deployment 生效。

#### 7.5.1 权威 whole-stack 形式

设已完成 pool 有 K 条 activity，`u_i=pre_pool(A_i)`，carrier `c` 在 M3 后冻结：

```text
h_native(K) = post_pool(concat((sum_i u_i) / K, c))
h_post(K)   = (sum_i post_pool(concat(u_i, c))) / K
h_J(K)      = h_native(K) + tanh(alpha) * (h_post(K) - h_native(K))
```

审计实现可以按 arrival order 重新堆叠全部 K 条 trial 并重算。这样容易对照训练时的
whole-stack operator，也最容易生成 pool-state digest 和位级哨兵；但这是工程上的权威
实现，不是算法复杂度的必要条件。

#### 7.5.2 O(1) 增量形式

生产实现缓存每条新 trial 的 `u_i` 与 `post_pool(concat(u_i,c))`，并维护两个运行和。每次
完成一个新 trial 只需：

1. 计算一次该 trial 的 `pre_pool`；
2. 计算一次该 trial 的 carrier-conditioned `post_pool`；
3. 更新两个 running sum 与 K；
4. 下一次 identity 请求时做一次除法和一次 native `post_pool`，再按 J-R1 融合。

因此新增 trial 的 identity 更新是 O(1) 状态更新，而不是随 K 反复增长的 O(K) 重算；整段
session 也不需要 O(K²)。identity 分支只是 Linear/MLP 级，主要计算仍是 whole-trial
decoder/transformer。实现还应按 `pool_state_digest` 缓存 identity：同一 pool state 的全部
内部时间位置只计算一次。B1 governing evaluator 每 trial 只调用一次 whole-trial predict，
因此不能照搬 M2 的“约 32 个 20 ms 窗口”作为 B1 计算量证据。

#### 7.5.3 等价性和状态门

正式打包前必须在相同进程、相同 arrival order、相同 float32 reduction law 下比较：

- whole-stack 与 running-sum 的 `h_native/h_post/h_J`；
- 每个完成边界前后的 K、成员顺序和 pool digest；
- prediction 与 official per-trial MSE；
- reset 后 running sums、K 和缓存均归零，再只载入该 dataset 的 M3 payload。

CPU float64 reference 目标是 identity/prediction bitwise equality；真实 GPU reduction 不保证
位级一致，必须在任何性能结果之前冻结数值容差，并同时要求 identity/prediction max-abs
`<=2e-6`、trial MSE absolute difference `<=2e-8`。不满足则生产实现保留 whole-stack
路径，不能为速度放宽科学算子。

---

## 8. Source-only split、标签预算和泄漏边界

### 8.1 三折 date-LODO

| Fold | Train dates | Validation date | Validation query | `K_train_max` | Validation pre-decode K |
|---:|---|---|---|---:|---:|
| 0 | `20210627, 20210628` | `20210626` | calib 4..11 + 2 minival = 10 trials | 28 | 3..12，全部 in-range |
| 1 | `20210626, 20210628` | `20210627` | calib 4..29 + 2 minival = 28 trials | 10 | 3..30；前 8 条 in-range，后 20 条 OOD |
| 2 | `20210626, 20210627` | `20210628` | calib 4..8 + 2 minival = 7 trials | 28 | 3..9，全部 in-range |

每个 validation date：

- 只取其 held-in-calib chronological first 3 trials 构造初始 activity identity 和 SFC；
- held-in-calib 第 4..N 条作为因果 validation query：先评分、再提交 activity；它们不产生
  参数更新，不进入该 fold 的 basis、normalizer、lag/λ 或 carrier 重拟合；完成三折后只按
  预注册 aggregate gate 选择 q/方法；
- 对应 held-in-minival 的 2 trials 按**预注册文件角色约定**接在 calibration-query stream
  后继续评分和 decode-before-commit。calib 与 minival 是不同 NWB，二者的 trial/timestamp
  都从相对零点重新开始，无法从文件内 timestamp 恢复真实跨文件先后；Stage 0A 只登记
  这个不可识别性和固定约定，不得声称证明了跨文件时间顺序；
- validation date 的全部信息不得进入训练 checkpoint。

选择 M3 是为了精确匹配三个 held-out date 各自只有三条 calibration trial 的部署条件，
而不是因为 source date 缺数据。

每个 training date 也使用 chronological first M3 作为 seed support；这三条 trial 只用于
构造初始 activity identity/SFC，不进入该 date 的 task loss。其余 held-in-calib trials 才是
source query，并按 §7.5 的 decode-before-commit law 逐条增长 activity pool：

```text
20210626: 11 - 3 = 8 query trials
20210627: 29 - 3 = 26 query trials
20210628:  8 - 3 = 5 query trials
```

所以三个 LODO fold 的训练 query 数依次为 `31 / 13 / 34`，validation query 数为
`10 / 28 / 7`，训练所见最大 pre-decode cardinality 依次为 `28 / 10 / 28`。fold 1 的
validation 后 20 条处于 `K>10` 的明确 OOD 区域，不能主导 memory-law 或 carrier 内容门；
§10 规定 in-range 选择和 OOD 描述分离。V1 不随机旋转 support、也不把
support trial 同时当 query；episodic/random-support 训练属于后续独立的数据增广问题。

### 8.2 Final source fit

三折完成、方法与 horizon 冻结后：

1. 用三个 held-in-calib dates 训练 final model；
2. 是否把 held-in-minival 六条 trial 加入 final training，必须在独立 work order 中预先
   选择；默认 **不加入**，以保持本设计的 source split 可复现；
3. 用全部三个 held-in-calib dates 拟合 final acoustic basis 和 carrier normalizer；
4. 为 held-in test dates `20210626/27/28` 也各自只用 chronological M3 构造 deployment
   activity/SFC payload；不因 held-in 可用 11/29/8 条就改变 few-shot 预算；
5. 对 `20210630/20210701/20210705` 各自三条 held-out-calib 闭式拟合 activity identity
   与 SFC；
6. 打包模型、source-frozen transforms 和全部六个 date 的 M3 payload；decoder tag resolver
   同时支持 dotted/undotted date；
7. 不读取、缓存或更新任何 hidden query label。

### 8.3 标签层级

| Tier | Target calibration 使用 | Arms |
|---|---|---|
| Tier 1 | 不使用 target labels；可用 neural，也可 neural-free | `A0-NATIVE`, `J0`, `TPL-SRC` |
| Tier 2 | released M3 calibration spectrogram labels | `TPL-M3`, `N-SFC4`, `N-SFC9`, `J-SFC4`, `J-SFC9`, `A0-OR158`, `A0-RT`, `DR-158-ML`, `DR-158-SL`, `DR-PC8` |

论文中 Tier 1 和 Tier 2 必须分栏。selected SFC 优于 A0 说明额外标签加方法的总价值；只有
selected SFC 必须先优于 neural-free `TPL-M3`，才能说明 query neural 提供了模板之外的
信息；进一步优于 `A0-OR158` 和 `DR-158-ML`，或具有清楚、量化过的效率/鲁棒性优势，
才能支撑“carrier architecture 优于直接标签适配”的更强说法。只胜 `DR-PC8` 或
`DR-158-SL` 不足以支撑。

### 8.4 Official 规则依据与封存要求

当前官方公开材料已为两类操作提供正面依据，不再把它们列为未解决的设计阻塞：

1. [FALCON README](https://github.com/snel-repo/falcon-challenge) 明确称 held-out calibration
   很小、用于 few-shot recalibration；同一 README 的官方 sklearn 示例把
   `held-out-calib` 作为 `--calibration_dir` 并以 `--mode all` 拟合。因而使用 released
   calibration 的成对 neural/target 做闭式 recalibration 与官方示例相符。
2. README 说明 evaluation 可以使用整个 available time period；
   [EvalAI challenge overview](https://eval.ai/web/challenges/challenge-page/2319/overview) 说明
   evaluation 时只给 neural observations、允许 unsupervised adaptation。因而在合法完成
   边界后累计已完成 query neural、且不读取 query labels/不更新 target optimizer 的
   GROWING 与当前公开合同相符。

正式 work order 仍必须封存当时 README、demo、challenge page 的 commit/date/body SHA，
并重新核对提交时规则是否变化。这是版本化 authority snapshot，不是让执行者重新决定
是否使用 Tier-2 标签或 GROWING；只有官方规则发生实质变化时才 fail closed。接口注释仍
不能单独替代上述公开规则文本。

---

## 9. 分阶段实验计划

### Stage 0A：数据与 evaluator contract（CPU，必须先过）

输出一份 immutable inventory，逐文件记录：

- path、size、SHA；
- session/split/trial count；
- tx shape/dtype/value range/timestamp step；
- spectrogram shape/range/frequency/time grid；
- raw spectrogram minimum `1.0` 与 `log`/inverse-output contract；
- temporal mask indices；
- trial start/stop and 27000↔900↔880 alignment；
- held-in-calib 与 minival 各自 timestamp 从零重启的事实，以及固定的
  `calib 4..N -> minival 1..2` 验证约定；不得伪称恢复了跨文件真实 chronology；
- dotted/undotted dataset-tag behavior 和安装版 `evaluate()` 已知 mismatch；
- installed `falcon_challenge` config/dataloader/evaluator SHA。
- 当前 FALCON README、官方 sklearn demo 与 EvalAI rule snapshot、commit/date/SHA，绑定
  labeled calibration 与 unsupervised hidden-query accumulation 的现有权限依据。

门：全部九个文件一致、有限、mask=700、channel=85、frequency=158；否则整条路线停止，
不在 loader 内静默修复。

### Stage 0B：SFC constructibility（CPU，非 decoder 性能门）

三折分别执行：

1. 仅 train dates 拟合 `log(raw)` + frequency standardizer + PCA8；
2. 对 validation date first M3 投影，不重估 basis；
3. source-only SFC lag grid 固定为 `{0,10,20,30,40,50,60} ms`，与 DR lag grid 相同；
   neural 输入为与 SFC 相同的 5 ms box 平滑 1 ms rate view `r_i`；
4. 对每个 lag、每个 outer-training date，分别用该 date first M3 拟合 date-specific SFC，
   在同一 date 的 calibration 4..N 上评分；再对两个 training date 等权聚合选择该 fold 的
   SFC lag。禁止把 A 日的 channel map 应用到 B 日，也禁止让 validation date 参与；
5. 在 z3/z8 上分别独立拟合 SFC4/SFC9；Direct159 移到 Stage 3 条件机制分支；
6. 无条件构造并评分 `TPL-SRC/TPL-M3`、`DR-158-ML/DR-158-SL/DR-PC8`；闭式 readout 的
   λ/lag 严格使用 §6.5.5 的每个目标 date 内 M3-LOO，不做跨 date channel mapping；
7. 报告 rank、condition、coefficient norm、split-half correlation/cosine、held-calibration
   encoding R2、label-derangement control。

冻结门建议：

```text
SFC9 finite for all 3 folds
SFC4 finite for all 3 folds
design rank = 4 for every SFC4 fit and 9 for every SFC9 fit
condition number <= 1e4 after source-frozen standardization
flattened split-half W correlation >= 0.50 in >= 2/3 source sessions
median per-channel cosine >= 0.60 in >= 2/3 source sessions
label-derangement encoding score does not exceed correct pairing
```

该门只说明 descriptor 可构造，不说明 decoder 会改善。每 fold 的 lag/λ authority、以及
预注册 q 候选一旦冻结，不得由该 fold 的 J-SFC validation 结果回头修改。

### Stage 1：A0 substrate + horizon（GPU，单方法）

只训练 `A0-NATIVE` 三折，建立 B1 model/evaluator parity。固定 paired seeds `{42,43}`，Adam、weight decay 0，
初始 LR `5e-5`；如 LR 本身需要改变，必须在独立 substrate work order 中完成，不能和
carrier 一起扫。

训练按 optimizer step 而非 epoch 比较：

```text
checkpoints = {250, 500, 1000, 2000 steps}
```

在 `3 folds × 2 seeds` 的 governing `GROWING` 律 in-range `10/8/7` coordinates 上按 equal-cell aggregate 选取
“达到全候选最低 equal-session MSE
的 1% 相对范围内的最早步数”，记为 `H*`。随后所有六个训练臂和两个 seed 固定 `H*`，
不得各自挑 checkpoint。同 checkpoint 必须同时报告 `FIXED3`，但 `H*` 选择与 Stage 1
gate 只使用 `GROWING` in-range。

同时保留 12-epoch 读数，但它只是早期训练点。以 batch size 4、丢弃不满 batch 的尾部
估算，三折每 epoch 仅约 `7 / 3 / 8` 个 full-trial batch；12 epoch 只有约
`84 / 36 / 96` step，不能默认代表收敛。

Stage 1 门：

- official metric parity 全过；
- 三折 prediction 全有限；
- 在 governing `GROWING` 律的 in-range coordinates 上，相对 neural-free `TPL-SRC-BEST` 的 equal-date mean gain 非负、
  至少 `2/3` date 为正、worst-date relative gain `>=-2%`；完整 10/28/7 与 `FIXED3` 另行披露。否则 A0
  不能证明 query neural 比 source song template 多提供信息；
- 250→H* 存在实质学习，H* 末段没有连续恶化；
- package/reset/batch-size-1 复现离线 prediction。

`H*` 冻结后，无条件在同一三折验证流上拟合/评分 `A0-OR158`；不得等待 SFC 出结果后才
决定是否加入这个强标签基线。

如果 A0 本身不能超过简单 baseline，本项目先修 substrate，不得把失败归因给 SFC。

### Stage 2：匹配六臂训练（GPU，主实验）

固定 `H*` 后，三折训练：

```text
A0-NATIVE
N-SFC4
N-SFC9
J0
J-SFC4
J-SFC9
```

每折、每个 seed∈`{42,43}`：

- byte-identical shared native substate；carrier columns 全臂精确零初始化，J arms 另有显式
  IEEE `+0` alpha；
- 相同 train trials、顺序、batch、optimizer steps；
- 相同 task loss；
- paired Python/NumPy/Torch/CUDA RNG；
- 一次 DataModule/一个 resident batch，六臂使用相同预生成 schedule；具体是否常驻由
  memory smoke 决定；
- 相同 whole-unit dropout mask；
- 相同 chronological growing-memory schedule、decode-before-commit 顺序和 pool-state manifest；
- 固定 checkpoint `H*`；
- validation stream 不参与 arm 内 checkpoint 选择；固定 `H*` 后，in-range `10/8/7`
  coordinates 进入 governing fold/seed aggregate，完整 `10/28/7` 只增加预注册 OOD stress
  panel。

不允许只从训练完的 A0 checkpoint 给 carrier/J arms 做 12-epoch warm-start；B1 主问题是
from-scratch matched architecture。若未来研究 warm-start，必须另设 matched A0 continuation。

### Stage 3：同标签 baseline 与机制控制（CPU/GPU forward-only 优先）

`TPL-SRC/TPL-M3/A0-OR158/DR-158-ML/DR-158-SL/DR-PC8` 已在 Stage 0B/1 无条件完成，
不属于本条件分支。只有任一
`N/J × SFC4/SFC9` 对相应 Zero control 通过内容门，且该 carrier arm 不劣于 A0，才执行：

- correct vs RSq；
- correct vs LSq；
- intercept-only；
- `alpha=+0`（仅 J-R1 arm）；
- J-MEAN descriptive arm（仅 J-R1 arm）；
- Direct159 CPU diagnostic。

若某个 control 需要新 trainable layer 或 matched retraining，不得伪装成 forward-only，必须
单独立项。

### Stage 4：Final train、local package 与 EvalAI

前提：Stage 0–3 全部完成且方法/维度/horizon 已冻结。

必须打包并本地运行：

1. `A0-NATIVE` Tier-1 baseline，作为 trained arm 同样按 §10.6 独立选择 `L_A0`；
2. 唯一选定的 Tier-2 SFC 方法（N/J × SFC4/SFC9 之一）及按 §10.6 独立选择的 `L_method`；
   official package 中 A0 使用 `L_A0`、方法使用 `L_method`（best-vs-best）；
3. 本地必须同时验证 `TPL-SRC/TPL-M3`、`A0-OR158` 与 `DR-158-ML`；若 official 提交预算
   允许，再提交这些 comparator。`DR-158-SL/DR-PC8` 只作本地容量/压缩诊断。

不能只提交方法而没有 matched B1-SPINT baseline，再用其他论文或其他代码栈的公开数字
做主差值。EvalAI 的 primary 是 held-out equal-session `MSE Mean`，lower is better。

---

## 10. 预注册估计量、门槛和选择规则

### 10.1 Fold 统计单位

B1 source validation 只有三个 date；各 date 的完整 causal query stream 为 10/28/7 trials，
但用于方法和 memory-law 选择的 in-range stream 为 10/8/7 trials：fold 1 只保留
pre-decode `K<=10` 的前 8 条，后 20 条 `K=11..30` 是明确的 cardinality-OOD stress panel。
trial 仍不是跨 session 独立重复。因此：

- primary unit = date/session；
- source selection headline = 三个 LODO date 的 in-range equal-session mean；
- 完整 10/28/7 stream 另报 equal-session stress headline、每 date 的全部 per-trial MSE、
  early/late 与 in-range/OOD 分层，但统计 n 仍是 3 dates，不把 45 trials 当作独立 session；
- 两个 paired seeds `{42,43}` 分别报告，再对 seed×date equal-cell aggregate；headline
  date gain 先在两个 seed 内平均；
- bootstrap 只能作为描述，不能用窄 trial-bootstrap CI 代替 session 复制。

### 10.2 内容门

对 q∈{4,9}、部署律 L∈{GROWING,FIXED3}，分别定义每 fold 的 native-order 和 J-R1
carrier gain；任何 matched contrast 的两端必须使用相同 L：

```text
absolute_native_content_gain_s,q,L = MSE_s,L(A0) - MSE_s,L(N-SFCq)
relative_native_content_gain_s,q,L = absolute_native_content_gain_s,q,L / MSE_s,L(A0)

absolute_jr_content_gain_s,q,L = MSE_s,L(J0) - MSE_s,L(J-SFCq)
relative_jr_content_gain_s,q,L = absolute_jr_content_gain_s,q,L / MSE_s,L(J0)
```

每条路径、每个 L 都只使用 §10.1 的 in-range coordinates，并分别使用同一个最低条件：

```text
mean relative_{native|jr}_content_gain >= +0.03
positive dates = 3/3
mean gain is positive in both paired seeds
worst relative_content_gain >= -0.02
all finite / same-input / no-target-update gates pass
```

一个 `(topology,q)` 只有在 `GROWING` 和 `FIXED3` **两种 L 都通过**上述内容门时，才登记为
稳健 carrier-content success 并进入主产品候选。这样 memory 对 carrier arm 和 Zero arm 的
不同影响不会被并入 carrier 主效应。若只在一个 L 通过，必须登记为
`carrier × memory-law interaction / evidence insufficient`，两律数字仍全部报告，但该 arm
不进入 V1 主产品候选，也不触发 rank/MLP/lag sweep。

`J-SFC9/GROWING` 仍是预注册主机制展示格，但不能绕过双律内容门；native 对照回答 carrier
是否无需 J-R1 也有效。若只有 native topology 的某个 q 在两律都通过，则可以选择相应
`N-SFCq`，但不能宣称 J-R1 使 carrier 生效。

### 10.3 维度和产品选择

维度只能在相同 fusion 内比较：

```text
relative_dimension_gain_native_s,L =
  (MSE_s,L(N-SFC4) - MSE_s,L(N-SFC9)) / MSE_s,L(N-SFC4)

relative_dimension_gain_jr_s,L =
  (MSE_s,L(J-SFC4) - MSE_s,L(J-SFC9)) / MSE_s,L(J-SFC4)
```

两条 topology 分别报告，不强迫共享 q；每条 topology 必须在两个 L 下各自报告维度差，
不得让 SFC4/SFC9 各用自己的 `L_arm` 后再做维度比较。稳健维度结论规则：

- 两个 L 的 mean 都 `>= +0.01` 且各自至少 `2/3` date 为正：选 SFC9；
- 两个 L 都在 `(-0.01,+0.01)`：选更小的 SFC4；
- 任一 L 的 mean `<= -0.01`，或两个 L 对维度方向不一致：保守选择 SFC4，并把 q8 额外
  维度登记为无稳定收益；
- 不按单个最好 fold 选择。

选择顺序固定为：先按 §10.2 的**双律**内容门删除不合格 `(topology,q)`。维度规则只能在
**已通过 §10.2 双律内容门的 `(topology,q)` 之间**选择：若某 topology 只有一个 q 过门，
该 q 直接冻结，维度差仅作描述性报告；若两个都未过，该 topology 退出产品候选。再按本节的双律
维度规则冻结每个仍有候选的 topology 的 q；之后才在每个剩余 arm 内用 §10.6 冻结 `L_arm`，并检查
该 arm 相对 A0 的同律总效果。这里的“不优于 A0”始终指同一个部署律下的 matched
comparison。禁止先看哪种记忆律能让 carrier contrast 最大，再反向选择记忆律。最后在
剩余合法的 `(arm,L_arm)` 候选中选择最低
seed×date equal-cell MSE。若候选间 mean relative difference `<1%`，按预注册复杂度顺序
选择：`N-SFC4 -> N-SFC9 -> J-SFC4 -> J-SFC9`。因为 N-SFC4 已在主矩阵中，最终比较不会
再把 q 与 J-R1 混在一起。

### 10.4 J-R1 与 interaction 门

```text
jr_gain_zero_absolute_s,L = MSE_s,L(A0-NATIVE) - MSE_s,L(J0)
relative_jr_gain_zero_s,L = jr_gain_zero_absolute_s,L / MSE_s,L(A0-NATIVE)

jr_gain_sfcq_absolute_s,L = MSE_s,L(N-SFCq) - MSE_s,L(J-SFCq)
relative_jr_gain_sfcq_s,L = jr_gain_sfcq_absolute_s,L / MSE_s,L(N-SFCq)

interaction_absolute_s,q,L =
  [MSE_s,L(J0) - MSE_s,L(J-SFCq)] - [MSE_s,L(A0) - MSE_s,L(N-SFCq)]
  = [MSE_s,L(N-SFCq) - MSE_s,L(J-SFCq)] - [MSE_s,L(A0) - MSE_s,L(J0)]
```

`jr_gain_zero_absolute` 和 `jr_gain_sfcq_absolute` 分开回答 J-R1 在无 carrier/有 carrier 条件下
是否有用；`interaction_absolute` 回答 carrier 是否特别需要 J-R1 consumer。interaction
只用 absolute MSE，避免不同 reference 分母破坏 2×2 恒等式。它们是机制读数，不是 carrier
成功的必要条件：J0 可以为零而 J-SFC9 正，说明 late branch 只在正确 carrier 下有用。
所有 governing interaction 都先在 in-range coordinates 上、按 L 分开计算；完整 stream 的
同名读数必须加 `_OOD_STRESS` 后缀，不能参与选择。
但若 `J0` 显著负且 `J-SFCq` 只刚好胜 J0、仍低于 matched A0/N-SFCq，则产品失败；不得用较弱的
内部对照制造正数。

产品选择完全按 §10.3 的六臂合法集合和 tie rule，不再在看到 SFC4/SFC9 结果后临时选择
一个缺少 matched N arm 的比较。

### 10.5 Tier-2 实用门

选定 SFC 方法还必须报告：

```text
vs A0-NATIVE       # 额外标签+结构的总效果
vs TPL-M3-BEST      # 相同 M3 标签、完全不读取 query neural 的模板（三聚合中 MSE 最低者；三者全报）
vs A0-OR158        # source-pretrained A0 + 相同标签的全频输出校正
vs A0-RT           # 冻结 A0 + M3 逐坐标残差模板（与 A0-OR158、DR-158-ML 同等地位）
vs DR-158-ML       # 相同标签预算、595 维多 lag neural 的全频直接闭式适配
vs DR-158-SL       # 单 lag 低容量下界
vs DR-PC8          # 仅为 rank-8 压缩诊断
```

论文级 local success 建议要求：

- SFC 在 GROWING/FIXED3 双律的 in-range 内容门都通过；
- selected SFC 对 A0 的 mean relative error reduction `>=3%`，`3/3` date 为正；
- `TPL-M3-BEST -> selected SFC` 的 mean gain 在 in-range 与完整 10/28/7 stream 上都必须
  `>=0`，且完整 stream 至少 `2/3` date 为正；否则不能声称方法解码了模板之外的 neural
  信息；
- selected SFC 对 `A0-OR158`、`A0-RT` 和 `DR-158-ML` 的 mean relative gain 都必须 `>=0`，并且对每条
  强基线的 worst-date relative gain 都必须 `>=-2%`；若 mean 输给任一强基线，则故事降为
  参数效率/机制结果，不能宣称 carrier architecture 的性能优势；
- RSq/LSq 均不能匹配 correct carrier；
- no target backprop / no query-label access。

### 10.6 Growing-memory 读数

每个训练完成的 arm 必须同时报告 governing `GROWING` 与同 checkpoint 的 `FIXED3`。用于
选择部署律的 primary 只包含各 fold 的 in-range coordinates `10/8/7`：

```text
memory_gain_s(arm) = MSE_s(arm, FIXED3) - MSE_s(arm, GROWING)
```

正值表示 causal activity accumulation 有益。由于 checkpoint 本身按 GROWING 训练，这个
差值证明的是“部署时是否实际需要更新 memory”，不是“growing training 是否优于 static
training”的完全因果估计。若要提出后一个主张，必须另训 matched FIXED3 checkpoint；V1
不为此把六臂扩成十二个独立训练臂。

完整 `10/28/7` stream 仍逐 trial 报告，但 fold 1 的后 20 条只进入
`CAUSAL_CARDINALITY_OOD` stress panel，不得改变 `L_arm`、q、topology 或 checkpoint。

每个 arm 的部署律独立按预注册规则选择：只有 GROWING 相对 FIXED3 的 in-range mean relative error
reduction（分母固定为该 arm 的 FIXED3 MSE）`>=1%`、至少 `2/3` date 为正且 worst date
`>=-1%` 时选择 GROWING；否则选择
FIXED3，复杂度 tie 也选择 FIXED3。论文中的强 growing-memory 声明还要求 `3/3` date 为正
且两个 seed 的 mean 都为正。不得用“理论上 O(1)”掩盖性能退化。

### 10.7 Official 结果阅读

只有一次或极少次数 official evaluation 时：

- baseline 和方法必须在同一预注册批次打包；
- 不按 held-out 结果回头选择 q、lag、horizon、seed；
- primary improvement（best-vs-best：A0 用独立选择的 `L_A0`，方法用 `L_method`）：

```text
official_gain = heldout_MSE(A0, L_A0) - heldout_MSE(selected SFC, L_method)
```

  本地内容门仍是 same-L matched contrast；`official_gain` 是 best-vs-best。两者都报告。

- 同时报 held-in 和 held-out；held-in 正、held-out 负即 source overfit；
- 不能把 EvalAI 的一次小差值当成统计显著，必须与三个 held-out session 的分项/方差一起读。

---

## 11. GPU 利用率与配对训练建议

Stage 2 优先使用预生成配对 schedule，而不是在六臂之间反复 snapshot/restore 全局 RNG：

1. 一个 DataModule；
2. 在训练前将 trial order、pool-state manifest、whole-unit dropout probability/mask 和其他
   stochastic choices 按 fold/seed/step 固化为只读 schedule；
3. 每臂独立读取同一 schedule，避免复杂的运行时 RNG 配对成为科学依赖；
4. 显存允许时让 `A0/N-SFC4/N-SFC9` 或 `J0/J-SFC4/J-SFC9` 三模型常驻，共享一次
   resident batch/H2D；不强制六模型同时常驻；
5. 普通 step 不做全模型 CPU SHA 或 `cuda.synchronize()`；
6. 仅在 first-step、固定 sentinel、checkpoint 做完整审计。

启动前 smoke 必须测量实际 peak memory、steps/s 和投影 wall time。若三模型也不能常驻，
六臂可独立进程顺序运行，但必须读取相同 immutable schedule；不得因看到任何一组结果而
取消另一组。GPU 利用率优化不能改变模型、batch、mask、growing-memory 或 optimizer-step
合同。

本计划没有授权占用 GPU，也没有指定 GPU0/GPU1；实际 work order 必须在启动前绑定设备，
并避开当时正在运行的其他 agent/job。

---

## 12. 实现边界与最低测试清单

### 12.1 建议 additive 路径

实现时优先新建，不修改已封 M1/M2/H1 路线的科学语义：

```text
SPINT-main/src/data/falcon_b1_datamodule.py
SPINT-main/src/models/components/b1_spint.py
SPINT-main/src/models/components/b1_sfc_jr.py
SPINT-main/src/models/falcon_b1_module.py
SPINT-main/configs/data/falcon_b1.yaml
SPINT-main/configs/model/falcon_b1*.yaml
SPINT-main/third_party/falcon_challenge/b1_spint_decoder.py
```

准确 owned paths 应由后续 work order 冻结；这里只说明模块边界。

### 12.2 必须有的 CPU tests

- 30 kHz→900 bins conservation：每 unit/trial spike sum 保持；
- NWB timestamp 只用于证明 `sample i -> i//30`，decoder positional implementation 不接受
  虚构 timestamp input；
- 900 neural bins 与 880 spectrogram timestamps 的显式对齐；
- eval mask 恰好 `[90,790)`；
- official metric 独立复现；
- constant prediction 必须在提交前 fail closed，不能把 sklearn NaN exception 降格成单条缺失；
- installed `evaluate()` 的 undotted/dotted B1 tag mismatch 回归测试；
- session-equal aggregation；
- LODO basis 不读取 validation date；
- M3 carrier 不读取第 4 条 calibration trial；
- SFC4 padding 与 SFC9 common width；
- SFC4 在 z3 上独立重拟合，不能等于 SFC9 coefficient truncation；
- standardized-log head 经逆标准化+exp 后才进入 raw-space official metric；
- `A0-OR158/DR-158-ML/DR-158-SL/DR-PC8` 都严格执行 target-date M3-LOO λ authority；
  `DR-158-SL` 在同一 LOO 中联合选择 lag；
- `TPL-SRC/TPL-M3` 完全不读取 query neural，模板成员、平均顺序和 prediction SHA 可复现；
- carrier row permutation 同步 unit permutation；
- dropped unit 的 neural/activity/carrier mask 一致；
- alpha 是 IEEE `+0`；
- first-step 六臂 CPU reference equality 与 GPU 容差门；
- J arms 的 alpha 第一批梯度可达；`N-SFC4/N-SFC9/J-SFC4/J-SFC9` 的非零 carrier columns
  第一批梯度可达（Zero9 arms 不要求 carrier-column 非零梯度）；
- causal decode-before-commit：第 q 条 prediction 的 pool digest 不含 q；
- `K_train_max=28/10/28` 与 in-range `10/8/7` query mask 正确，fold 1 后 20 条只能进入
  OOD stress receipt；
- whole-stack 与 running-sum identity/prediction 等价；
- B1 whole-trial boundary 与 reset 状态机；
- target model/optimizer state 不存在；
- query labels/access count 为零；
- decoder reset 不跨 trial/session泄漏。
- decoder `predict` 精确返回 `[158,880]`，dotted/undotted tags 映射到同一 payload。

### 12.3 必须有的 GPU smoke

- 一个 real source batch forward/backward；
- 六臂 loss finite；
- matched initial native-substate/carrier-column digest（不要求不同拓扑的整模型 SHA 相同）；
- identical dropout mask；
- model/student parameters finite；
- carrier/alpha/native encoder/decoder 预期参数组均有正确梯度；
- `J0/J-SFC4/J-SFC9` 的 alpha 不因双零初始化永久无梯度；
- 六臂的成员顺序、pool cardinality 与 decode-before-commit digest 完全配对；
- memory/time projection；
- evaluator batch-size-1 decode parity。

---

## 13. 明确禁止的做法

V1 不允许：

1. 把 001046 或 B1 写成 H2；
2. 声称 SPINT 原论文已经给出 B1 baseline；
3. 直接复用逐 20 ms 的 M1/M2 DataModule；
4. 用 target held-out query label 选 basis、lag、q、lambda、epoch、seed；
5. target backprop 或 target optimizer state；
6. 只和 neural-only A0 或低秩/单 lag direct ridge 比较，却不加 TPL-M3、A0-OR158 与
   DR-158-ML；
7. 因为 T4 是四维就把 q=3 宣称为自然真值；
8. 把 Direct159 和 SFC4 的不同参数量当作纯维度效应；
9. carrier 只进入 late branch，却把差值解释为 pooling placement；
10. alpha 与 carrier projection 同时零初始化导致死梯度；
11. 单独训练 carrier/J-R1 而省略完整 `native/J-R1 × Zero/SFC4/SFC9` 2×3；
12. 各 arm 各自按 minival 选择最好 checkpoint；
13. 默认 12 epoch 已收敛；
14. 把 45 条 validation query trial 当成 45 个独立 session；
15. 第一次 positive 后立刻扫 NNMF rank、MLP width、gate vector 或 output filter；
16. 在本设计未过本地门前提交多组 EvalAI 进行隐式 target selection；
17. 在预测 current trial 前把它的 activity 写入 pool，造成同 trial 自泄漏；
18. 声称 B1 evaluator 实际调用了 `on_done`，或把 M2 的 20 ms/window 数量当作 B1
    runtime 证据；
19. 因为 whole-stack 审计实现会重算，就宣称 J-R1 growing memory 算法必然是 O(K²)；
20. 训练时跨 optimizer step 复用 detached activity embedding，却声称与参数冻结后的 O(1)
    deployment 完全同构；
21. 只打包 held-out 三个 payload，导致 test phase 的 held-in dates 在 reset 时缺失；
22. 把接口注释当成允许 hidden-query activity accumulation 的正式竞赛规则文本。

---

## 14. 预注册结果分支

### Outcome A：SFC 内容和 J-R1 都有效

以下条件要求对 `GROWING` 与 `FIXED3` 两个 L 分别成立：

```text
carrier_gain_jr_(q,L) > 0
jr_gain_sfc_(q,L) > 0
interaction_absolute_(q,L) > 0
MSE_L(selected J-SFCq) <= min(MSE(TPL-M3-BEST), MSE_L(A0-OR158), MSE_L(A0-RT), MSE(DR-158-ML))
MSE(correct) < MSE(RSq) and MSE(correct) < MSE(LSq)
```

故事：功能 carrier 为 unseen session 提供 activity identity 缺失的 unit-acoustic mapping；
J-R1 在不放弃 native operator 的前提下提取 late conditioned structure。

### Outcome B：SFC 有效，但 native-order 已经足够

```text
carrier_gain_native_(q,L) > 0
jr_gain_sfc_(q,L) <= 0 or matched J candidate is within 1% of N-SFCq under L
```

故事：carrier 内容有效，但 pooling residual 不是独立收益；选择 matched N-SFCq，不得把 J-R1 写成
必要组件。

### Outcome C：J0 有效，carrier 无效

```text
jr_gain_zero_L > 0
carrier_gain_native_(q,L) ~= 0
carrier_gain_jr_(q,L) ~= 0
```

故事：B1 from-scratch 网络受益于 anchored operator mixture，但 M3 声学功能描述没有新增值。
关闭 SFC，不继续扫 carrier rank。

### Outcome D：SFC9 与 SFC4 相同

在 selected topology 内内容门通过，但 dimension gain 在 ±1% 内：选择其 SFC4。故事是
四个功能数字足够；这才是
真正支持“T4-like low-bandwidth carrier”的结果。

### Outcome E：Neural-free 模板或 direct readout 胜出

```text
min[MSE(TPL-M3-BEST), MSE_L(A0-OR158), MSE_L(A0-RT), MSE(DR-158-ML)]
  < MSE_L(selected SFC) < MSE_L(A0)
```

若最小值来自 `TPL-M3-BEST`，故事是三条 labeled calibration 鸟鸣模板已经解释结果，selected
SFC 没有证明 query neural 的额外价值；若来自 `A0-OR158`/`A0-RT`/`DR-158-ML`，故事是 M3 标签足以
支持更直接的闭式声学适配，SFC 没有证明结构性能优势。可讨论参数化、机制或部署成本，
但不宣称 neural-decoding 性能创新。

### Outcome F：Carrier 只依赖一种 memory law，或整体为 null/负

若某个 carrier contrast 只在 `GROWING/FIXED3` 之一过门，登记为
`carrier × memory-law interaction`，但 V1 的稳健 carrier-content 主张和主产品资格都失败；
不得把有利 L 的差值单独当成 carrier 主效应。若两律都为 null/负，而 A0 substrate 已通过，
则结论是：在 B1 M3 标签预算和当前 source-frozen field 下，functional carrier 没有可重复
增益。两种情况都不以新 rank、vector gate、postfilter 或 target fine-tune 追逐正数。

### Outcome G：carrier/J-R1 有效，但 growing memory 有害

若 selected arm 的 GROWING 没有同时满足 §10.6 的 in-range mean relative gain `>=1%`、
至少 `2/3` date 为正和 worst date `>=-1%`，就选择 FIXED3；不要求 mean 必须为负才切换。
结论是功能 carrier/融合可以有效，但现有证据不足以证明部署时累计未标注 query activity
值得其复杂度，或它存在 cardinality/stale-history 风险。O(1) 可实现性不等于统计有效性；
V1 不继续扫 cap、指数衰减或遗忘 gate。

### Outcome H：A0 substrate 失败

这不是 carrier 负结果。先修 whole-trial alignment、loss、architecture 或训练 horizon，所有
SFC/J-R1 实验暂停。

---

## 15. 两周可行性顺序（不是运行授权）

### 第 1–2 天

- Stage 0A inventory；
- independent official metric reproduction；
- timestamp alignment tests；
- 固定 source LODO manifest。

### 第 3–4 天

- Stage 0B PCA8/SFC4/SFC9 constructibility；
- 无条件完成 TPL-SRC/TPL-M3 与 DR-158-ML/DR-158-SL/DR-PC8；
- 按 fold 冻结 SFC lag/basis/normalizer；按 target date M3-LOO 冻结闭式 baseline 的 λ
  （以及 DR-158-SL lag）；
- 外部审核 Stage 0 receipt。

### 第 5–7 天

- B1 DataModule/model/decoder；
- A0 real-batch smoke；
- A0 两 seed horizon screen；
- A0-OR158；
- direct metric、predict shape、tag resolver 与 package parity。

### 第 8–11 天

- J-R1/SFC common-width implementation；
- six-arm immutable schedule smoke；
- 三折×两 seed matched training。

### 第 12–13 天

- TPL-M3/A0-OR158/DR-158-ML 对照汇总、Direct159 diagnostic 与 RS/LS controls；
- aggregate/gates；
- 方法/维度冻结。

### 第 14 天以后

- 独立 work order；
- final source train；
- local Docker decoder smoke + direct governing metric；安装版 tag bug 未修复前不宣称完整
  local `evaluate()` parity；
- baseline + selected method EvalAI submission。

实际时间以 Stage 1 smoke 的 measured wall time 为准，不把上述日历估计写进 execution receipt。

---

## 16. 最强反对意见与回应

### 反对意见

SFC 使用 released target spectrogram labels，而 SPINT 的原始卖点是 unlabeled calibration。
任何提升都可能只是“多用了标签”，J-R1 与功能 descriptor 只是复杂包装；而且三个 source
date 太少，三折正结果也可能不稳。

### 回应

这个反对意见成立，因此设计明确把方法放在 Tier 2，并加入不读取 query neural 的
`TPL-M3`、同样 M3 标签预算的 `A0-OR158` 与 `DR-158-ML`；DR-158-SL/DR-PC8 仅是容量与
低秩诊断。两个 paired seeds 和 10/28/7 条长程
validation streams 降低了单 seed/两 trial 偶然性，但统计独立单位仍只有三个 date。主张
不会是“SFC 公平击败 unlabeled SPINT”，而是：

> 在 few-shot labeled recalibration 允许的条件下，把标签压缩为每 unit 的低带宽功能坐标，
> 是否比直接 readout adaptation 更可转移，并且能否通过 permutation-invariant SPINT
> consumer 在 unseen date 使用。

三个 source date 的证据只承担方法选择，最终一般化必须依赖三个 hidden held-out date；
official 结果之前不写普适结论。

---

## 17. 审核人应重点攻击的问题

1. FALCON B1 是否允许将 released held-out-calib spectrogram labels 打包用于提交；若规则
   有额外限制，Tier 2 整体必须停机，而不是偷偷改成 unlabeled proxy。
2. 30 kHz→1 ms→880 frame 的 timestamp operator 是否精确、可复现、与 evaluator 一致。
3. official min–max metric surrogate 是否会产生 scale/range 病态解。
4. B1-SPINT whole-trial architecture 是否仍保留 SPINT 的 permutation-invariant核心，还是
   只是借用了名字。
5. LODO fold 的 acoustic basis/normalizer/lag 是否真的排除了 validation date。
6. 完整 `native/J-R1 × Zero/SFC4/SFC9` 2×3 是否真正分开 carrier、维度、J-R1 与
   interaction；`alpha=0` forward 是否被错误地当成 matched N training。
7. carrier 是否在 native/late 两个分支中完全相同，避免内容×placement混杂。
8. SFC4/SFC9 common-width 是否真的 equal-parameter。
9. M3 dense-bin fit 的有效独立样本是否被过度表述。
10. TPL-M3 是否完全 neural-free；A0-OR158/DR-158-ML 是否与 SFC 使用相同 M3/valid
    frames，并严格在目标 date M3 内 LOO 选择 λ/lag；DR-158-SL/DR-PC8 是否被错误提升为
    唯一公平基线。
11. 两个 paired seeds、10/28/7 validation streams 与 `3/3` date gate 是否仍被过度解释。
12. final official baseline/method 是否一同提交，避免跨代码栈比较。
13. B1 growing memory 是否严格 decode-before-commit，且没有把 B1 whole-trial predict
    误写成实际不存在的 `on_done` callback。
14. whole-stack 权威算子与 O(1) running-sum 部署算子是否在同一 arrival order 下等价。
15. 训练时是否对完整历史 activity 重新前向/反传，而不是跨 optimizer step 复用 stale
    embedding。
16. `[158,880]` predict、dotted/undotted tag resolver、raw/log inverse 与安装版 local
    `evaluate()` bug 是否被如实处理。
17. held-in 与 held-out 六个 M3 payload 是否全部打包。
18. 是否已取得允许 hidden-query neural accumulation 的正式规则文本，而非只依赖接口注释。
19. carrier 内容门是否在 GROWING/FIXED3 两律下分别通过，还是把 carrier×memory interaction
    错记成 carrier 主效应。
20. `K_train_max=28/10/28`、in-range `10/8/7` 与 fold-1 后 20 条 OOD 是否正确分账；OOD
    stress 是否偷偷参与了 memory law、q、topology 或 checkpoint 选择。
21. calib/minival 跨文件 chronology 是否被如实标成不可恢复，并只采用预注册拼接约定。
22. `TPL-M3` 的极强结果是否被复算、封存并作为 neural-decoding 主张的必要门，而不是只在
    附录描述。

---

## 18. 本文依据的本地实现字节

设计审核时核验过：

| File | SHA-256 | 用途 |
|---|---|---|
| installed `falcon_challenge/config.py` | `3565e4449b7fc1f55a57e7c202083bfc6094ba55e37df2956851b7865a1c6ca9` | B1 task/channel/output contract |
| installed `falcon_challenge/dataloaders.py` | `7638d839ec2085c7dde71ec9ca9676cbf1256c79fdcb327c4ac6487bfe5c9b05` | NWB B1 load law |
| installed `falcon_challenge/evaluator.py` | `2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418` | whole-trial call + official metric |
| installed `falcon_challenge/interface.py` | `781dae89351d2c5e5914c8c9b47654254edd4f5a308a3bda85b4d91a755063e6` | `on_done` API 语义；用于核对 B1 分支实际未调用 |
| `SPINT-main/src/models/components/spint.py` | `855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519` | native SPINT operator |
| `SPINT-main/src/models/components/h1_carrierid_spint.py` | `fcea04965e8fd80bf84ca290ad3d22971b37df3050f9d178e0ac1c3dc91b4618` | CarrierID precedent |
| `tfpd_exploration/src/m2_anchored_joint_postfusion_v1/adapter.py` | `536dc24273e02e2495311de7fd5a1a09ef33971fb69b4e203b0777777a3de1be` | J-R1 exact operator |

这些 SHA 是设计时审计证据，不是未来 execution closure。实现期间任一字节变化都应重新
核验相应事实，不得机械沿用本文 SHA。

---

## 19. 最终建议

最值得执行的路线不是“直接把 158 维 target 塞给 SPINT”，也不是“为了像 T4 强行四维”，
而是：

```text
先建立可信的 B1-SPINT whole-trial baseline；
用 source-frozen PCA8 把声学场固定下来；
target M3 分别闭式拟合 per-unit SFC4/SFC9；
在 common 9-D interface 中完成 native/J-R1 × Zero/SFC4/SFC9 的完整 2×3；
在 source 训练和 B1 部署中使用合法的 causal growing activity memory；
用 arrival-order running sums 将 identity 更新实现为 O(1)，whole-stack 只保留作审计权威；
用 neural-free TPL-M3、A0-OR158 与多 lag DR-158-ML 守住同标签预算公平性，
DR-158-SL/DR-PC8 只作容量与低秩诊断；
只在 source date-LODO 全过后提交 baseline + 一个 frozen method。
```

如果成功，论文故事不是“高维 carrier 更强”，而是：

> **少量配对 calibration 标签可以被压缩成跨 session 的 per-unit 功能坐标；B1 的合法
> trial 完成边界让 activity memory 因果增长，而锚定的 pooling-order residual 在保持
> native identity 解的同时消费这些坐标。**

如果 SFC4 与 SFC9 相当，四维低带宽 descriptor 会成为更漂亮的结果；如果 SFC9 明显更好，
则应诚实地说 B1 的声学任务比二维运动方向需要更高的功能维度，而不是为了统一命名牺牲
可用信息。
