# SPINT 复现与 few-shot 机制分析（FALCON 研究扩展）

> 对象：`SPINT-main`（NeurIPS 2025，UW NeuroAI Lab）——Spatial Permutation-Invariant Neural Transformer。
> 定位：作为现有 FALCON-M2 片上微调研究的**上限 baseline 与另一条 few-shot 技术路线**。
> 本文重点：① 拆解 SPINT 的 few-shot 机制原理；② 给出 M2 为主、兼顾 M1/H1 的完整复现链路；③ 设计可复现的 few-shot 实验并对接现有 m2-research。
> 环境自检：本机已确认有 GPU，M2 数据集（DANDI 000953）已下载在 `FALCON/falcon-challenge-main/data/000953/`。M1(000941)/H1(000954) 暂未下载。

---

## 0. 一句话结论

SPINT 把"跨 session 漂移"重新表述为**"记录到的神经元是一个无序、可变长度的集合"**问题，用一个对神经元排列/数量不变的 Transformer 来解码。适配新 session 时**不做任何梯度更新**，只需喂入少量**无标签**校准 trial，模型据此现算每个神经元的"身份向量"注入输入——这就是它 few-shot 的全部秘密。我已用纯 NumPy 忠实复刻其前向，验证了两条核心性质（见 §2.4）。

---

## 1. SPINT 与 FALCON 的关系

FALCON = *Few-shot Algorithms for Consistent Neural decoding*，标准化 iBCI 跨 session 鲁棒解码评测（EvalAI Challenge #2319）。评测协议：模型在 held-in sessions 上训练，面对 held-out（未来天）sessions 时只允许**少量校准数据**做适配，再做**因果流式** open-loop 解码，指标为 variance-weighted $R^2$。

SPINT 是当前 FALCON 排行榜上 M1/M2/H1 三个运动任务的 SOTA 之一。它与你现有 m2-research 的关系是**互补的两条 few-shot 路线**：

| | 现有 m2-research | SPINT |
|---|---|---|
| 适配范式 | 片外预训练 + **片上梯度微调**（LMS/RLS/LoRA、通道增益零偏） | 片外预训练 + **免梯度**，靠 calib 现算神经元身份 |
| decoder | 线性/Ridge/小 LSTM（~1.5k–115k 参数） | Cross-attention Transformer（M2 ≈ 数百万参数） |
| 目标硬件 | 低功耗数字 IC | GPU |
| calib 是否需标签 | 需要 kinematics 标签（有监督微调） | **不需要标签**（只用神经数据算身份） |
| 优势 | 极低算力/显存，可上芯片 | 对通道增删/换序天然鲁棒，恢复率高 |

所以把 SPINT 纳入研究是合理的："它能达到多好"给你的轻量方案设了一个 R² 天花板参照，同时"免梯度身份注入"这个 idea 本身可能被蒸馏进轻量方案。

---

## 2. few-shot 机制原理拆解（核心）

### 2.1 它要解决的病

跨 session 的 recording nonstationarity 有三种：(a) 记录到的**单元数量变化**（电极漂移、单元丢失/新增）；(b) 单元**顺序/通道映射变化**；(c) 单元**tuning 漂移**。传统 decoder 把输入当成固定顺序的定长向量 $\mathbb{R}^N$，(a)(b) 直接让权重矩阵对不上，(c) 让权重失配——于是跨天 R² 崩。M2 上 static decoder 的跨 session 掉幅可达 0.27–0.82 R²（见你 findings.md）。

SPINT 的核心主张：**不要把神经元当成固定索引的向量分量，而当成一个无序 token 集合**，解码器对这个集合的大小和顺序都不变。

### 2.2 三大支柱（对照 `src/models/components/spint.py`）

**支柱一：可学习的"神经元身份嵌入"（learnable ID）——few-shot 的载体。**
前向里 `calib_trialized_neural_features` 形状 `B×M×T×N`（M 条校准 trial、每条 T 帧、N 个单元）。它被转成 `B×M×N×T`，经共享 MLP `fc_id_in`（`T→H`）编码，**沿 M 条 trial 取平均**得到 `B×N×H`，再经 `fc_id_out`（`H→W`）得到 `B×N×W` 的身份图，**逐元素加到当前神经窗口 `src` 上**：

```
id = fc_id_in(calib.permute→B×M×N×T)     # B×M×N×H
id = mean over M  →  B×N×H               # 聚合多条校准 trial
id = fc_id_out(id) →  B×N×W
src = src + id                           # 把"这个神经元是谁"注入输入
```

关键点：`fc_id_in/fc_id_out` 是**跨神经元共享**的 MLP，输入是"某单元在校准 trial 里的时序活动统计"。所以身份不是查表得到的固定 embedding，而是**从校准数据现算的函数**——换了新 session、喂新的 calib，身份自动重算。**这一步不需要任何标签，也不更新任何权重。**

**支柱二：置换不变的 cross-attention（对照 `MultiLayerCrossAttention`）。**
模型有一组**可学习的协变量查询** `rep`（`1×C×W`，M2 里 C=2，即手指 x/y 速度）。每个神经元 token 走共享 read-in `fc_in`（`W→H`）成为 key/value；C 个协变量查询做 query，对 N 个神经元 token 做 cross-attention：

```
rep(query): 1×C×H  →  repeat → B×C×H
src(key/value):     B×N×H
transformer_output = CrossAttn(query=rep, key_value=src)  # B×C×H
output = fc_out(...) → B×C×W → 输出 B×W×C
```

因为 attention 对 key/value 是"加权求和"，**对 N 个神经元的顺序天然不变**；输出维度只由查询数 C 决定，**与 N 无关**——所以增删单元不改变输出形状，也不需要改网络。这就把 §2.1 的 (a)(b) 从根上消掉了。

**支柱三：dynamic dropout 训练（模拟集合变化）。**
训练时（M2 config `dynamic_dropout: true`, low=0, high=1）对神经元维随机丢弃，丢弃比例 `p` 每个 batch 从 U(0,1) 采样：

```
p ~ U(0,1); dropout_mask over N; src = src * mask
```

即模型在训练中就见过"任意比例的单元缺失"，强迫它学会只凭剩余单元的集合作解码。这是让免梯度适配在测试期真正 work 的关键正则。

### 2.3 gradient-free few-shot 的完整闭环（对照 `spint_decoder.py`）

打包与评测阶段才看得出"few-shot"落地：

1. `spint_decoder.py` 读训练好的 checkpoint，并**遍历该 session 的 `*calib*.nwb`**，对每个校准文件把神经数据切成 trial、（可选）插值/FFT，取前 `calib_n_trials` 条，算出 `calib_trial_features[session_tag]` 存进 pickle。
2. 评测时 `SpintDecoder.reset(dataset_tags)` 按 session 取出对应的 calib 特征；`predict()` 每步把当前窗口 + 该 session 的 calib 特征一起喂进模型：

```python
behavior_pred = self.local_clf(decoder_in,
    calib_trialized_neural_features=self.local_calib_trial_features...)
```

**全程 `torch.no_grad()`，没有 optimizer、没有 backward。**适配一个全新 session = 换一份 calib 特征。这就是 README 所说 "few-shot adaptation without parameter updates, gradient-free, minimal unlabeled calibration trials"。

### 2.4 我的机制验证（NumPy 复刻，已在沙箱跑通）

沙箱无法装 torch（网络白名单拦截 PyTorch 源），我用纯 NumPy **忠实复刻了 `SpintModel.forward` 的全部结构**（共享 read-in MLP、身份 MLP、多头 cross-attention、pre-norm FFN、读出），固定并复用同一套随机权重，只改神经元轴。结果：

```
[1] permutation invariance  |  max|Δ| = 0.00e+00   (输出 50×2)
[2] variable unit count     |  N: 96 -> 66，输出仍为 50×2，无需改参
PASS: 输出对神经元换序严格不变、对单元数量不敏感 -> 适配只需新 calib，无梯度步
```

脚本随本文交付（`spint_mechanism_smoketest.py`），可直接 `python3` 运行复核。这从数值上坐实了置换不变性（Δ 精确为 0）和变通道数鲁棒性——即 few-shot 的两条地基。

---

## 3. 代码结构地图

| 文件 | 职责 | 复现关注点 |
|---|---|---|
| `src/models/components/spint.py` | SpintModel 架构 | few-shot 机制全在这（§2） |
| `src/models/falcon_module.py` | Lightning 模块：train/val/test、按 session 记 R² | 损失=MSE(仅最后一帧)，held-in/held-out 分开评 |
| `src/data/falcon_datamodule.py` | 读 NWB、切 trial、构 calib 特征、SessionBatchSampler | calib 采样、interpolate/fft、同 batch 同 session |
| `third_party/falcon_challenge/spint_decoder.py` | 打包 decoder，**固化 calib 特征** | few-shot 适配的落地点 |
| `third_party/falcon_challenge/spint_sample.py` | 调 FalconEvaluator 本地/远程评测 | `--split/--phase/--batch-size` |
| `configs/{model,data}/falcon_{m1,m2,h1}.yaml` | 每任务超参 | M1/M2/H1 差异见 §4.6 |
| `setup.sh` / `environment.yaml` / `setup.py` | conda env `spint`，`pip install -e .`，依赖 `falcon-challenge` | 依赖 falcon_challenge 包 |

数据流一句话：`held-in-calib` 既作训练数据又作训练期 calib 源 → `held-in-minival` 作 val_heldin → `held-out-calib` 作 val_heldout（模拟未来天的少样本适配）。

---

## 4. 复现全链路（M2 为主）

### 4.1 环境

```bash
cd /home/xinyuan/Work_host/SPINT/SPINT-main
bash setup.sh            # mamba env create -f environment.yaml && pip install -e .
mamba activate spint
```

`environment.yaml`：Python 3.10.15 + PyTorch(CUDA 11.8) + lightning + hydra-core + omegaconf。`setup.py` 声明依赖 `falcon-challenge`——你机器上 `FALCON/falcon-challenge-main` 已是 `-e` 安装（有 egg-info），确认 `spint` 环境里也能 `import falcon_challenge`；若报缺失，在 spint 环境里 `pip install -e /home/xinyuan/Work_host/FALCON/falcon-challenge-main`。

### 4.2 数据落位（关键）

SPINT 训练配置写死 `data_dir: ${paths.data_dir}/000953/` = `<repo>/data/000953/`，但你的数据在 `falcon-challenge-main/data/000953/`。二选一：

```bash
# 方案A：软链接（推荐，不占额外空间）
ln -s /home/xinyuan/Work_host/FALCON/falcon-challenge-main/data/000953 \
      /home/xinyuan/Work_host/SPINT/SPINT-main/data/000953

# 方案B：命令行覆盖 root，让 data_dir 指向已有数据目录
python src/train.py data=falcon_m2 model=falcon_m2 \
  paths.data_dir=/home/xinyuan/Work_host/FALCON/falcon-challenge-main/data/
```

已确认 000953 内含 `sub-MonkeyN-held-in-calib / held-in-minival / held-out-calib` 三个子目录，正是 datamodule 用 `rglob('*held-in-calib*.nwb')` 等匹配的结构。注意：DANDI 公版**不含 held-out test 的标签**（在 EvalAI 服务端），所以本地只能训练 + minival，最终 test 分数要提交 EvalAI。

### 4.3 训练

```bash
python src/train.py data=falcon_m2 model=falcon_m2 trainer=gpu
# 可覆盖：model.optimizer.lr=5e-5 trainer.max_epochs=... seed=42
```

M2 关键超参（`configs/*/falcon_m2.yaml`）：`model_dim=512, num_heads=64, num_layers=1, num_id_layers=3`；`window_size=50`（=1s @20ms），`calibration_n_trials=33`，`interpolate_trials: cubic → max_trial_length=100`，`behavior_scaling_factor=5.0`，`predict_scaled_behavior + decode_last_timestep_only`，Adam lr=5e-5。checkpoint 落在 `logs/train/runs/<run_id>/checkpoints/`。

### 4.4 打包 decoder（固化 few-shot calib）

```bash
python third_party/falcon_challenge/spint_decoder.py \
  --run_dir logs/train/runs/<run_id> \
  --checkpoint epoch_<NNN>.ckpt        # 生成 local_data/spint_m2.pkl
```

### 4.5 本地评测 & EvalAI

```bash
python third_party/falcon_challenge/spint_sample.py \
  --evaluation local --model-path local_data/spint_m2.pkl \
  --split m2 --phase minival --batch-size 7   # M1=4, M2=7, H1=8

# 提交（另建 py3.6 环境装 evalai，见 README，因与 spint 依赖冲突）
docker build --build-arg TASK=m2 --build-arg BATCH_SIZE=7 \
  -t spint_m2:latest -f third_party/falcon_challenge/spint_sample.Dockerfile .
evalai push spint_m2:latest --phase few-shot-test-2319 --private
```

### 4.6 M1 / H1 差异（供后续扩展）

| 项 | M1 (000941) | M2 (000953) | H1 (000954) |
|---|---|---|---|
| 通道数 | 64 | 96 | 176 |
| 协变量 C | (见 m1 config) | 2（手指 x/y 速度） | 多维 |
| eval batch-size | 4 | 7 | 8 |
| H1 专用 | — | — | `use_calib_active_segments`（按活动段而非 trial 切 calib） |

要跑 M1/H1 需先下载对应 dandiset 到 `data/000941`、`data/000954`（本机暂无），命令把 `m2` 换成 `m1`/`h1` 即可。三者共用同一套 `SpintModel`，只是 config 不同。

---

## 5. 可复现 few-shot 实验设计

目标：量化"给多少校准数据、能恢复多少 R²"，并做消融找出 few-shot 有效性的来源。所有实验都在 M2 上跑，指标沿用 FALCON 的 variance-weighted $R^2$（held-out sessions 平均）。

**实验 E1 — 数据效率曲线（最重要）。** 扫 `calib_n_trials ∈ {1,2,4,8,16,33}`（`spint_decoder.py` 的 `--calib_n_trials`/config `calibration_n_trials`），画 R²-vs-校准trial数（也换算成秒：M2 一条 trial 约多少 bins×20ms）。产出"达到 90% 饱和 R² 所需最少 calib"这一硬结论。这直接回答你 PROJECT R2 里的"data efficiency"。

**实验 E2 — calib 选取鲁棒性。** 固定 trial 数，扫 `calib_start_trial_idx`（用 session 前段 vs 后段）+ `random_calibration` on/off，看 R² 方差。检验"免梯度适配对具体喂哪几条 calib 有多敏感"。

**实验 E3 — 身份特征消融。** `trial_feature_type: raw vs fft`、`smooth_calibration: on/off`、`interpolate_trials: cubic/linear/off`。定位身份嵌入到底吃哪种特征最有效。

**实验 E4 — 置换/掉通道鲁棒性（对应机制）。** 评测时人为打乱通道顺序、随机屏蔽 x% 通道，画 R²-vs-掉通道率。SPINT 预期近乎持平（我 §2.4 已在结构上验证不变性），这是它相对线性 decoder 的卖点，值得量化成图。

**实验 E5 — 训练期 dynamic dropout 消融。** 关掉 `dynamic_dropout` 重训，看 held-out R² 掉多少——量化"训练期模拟单元缺失"对测试期免梯度适配的贡献。

**实验 E6 — 上限 baseline 对比。** 同一 000953、同一 R² 口径下比较：SPINT（免梯度）vs static 线性/Ridge vs 你 m2-research 里的 LMS/RLS/LoRA 片上微调 vs oracle（用 held-out 标签重训的上界）。

建议每个实验固定 `seed=42` 并跑 3 个 seed 记均值±std，落 CSV 到 `m2-research/outputs/results/`，与现有 `adapt_benchmark_*.csv` 同目录、同格式，方便合并出总表。

---

## 6. 与现有 m2-research 的对接建议

1. **共享数据与评测口径**：SPINT 和你的轻量 decoder 都用 000953、都用 variance-weighted R²、都以 held-out sessions 为跨天测试集——可直接并进同一张对比表（数据效率 / R² 恢复 / 参数量 / 算力）。
2. **SPINT 作 R² 天花板**：在 `adapt_benchmark_summary.csv` 里加一行 SPINT，给 Ridge/LSTM/LoRA 的恢复率标定"离 SOTA 还差多少"。
3. **idea 迁移**："从无标签 calib 现算通道身份并注入输入"这个免梯度思路，可能以极简形式蒸馏进你的线性方案（例如用 calib 段统计估每通道增益/零偏，正对应你 findings 里的 Level-0 每通道 α/β）——SPINT 相当于它的深度学习版上界。
4. **硬件视角**：明确记录 SPINT 的参数量/显存/延迟（不可上芯片），作为"为什么仍需要轻量片上方案"的论据；两条线不是竞争而是各自占据 R²-算力权衡曲线的两端。

---

## 7. 风险与注意事项

- **数据**：本机只有 M2（000953）；M1/H1 需另下 DANDI 000941/000954。公版无 held-out test 标签，最终 test 分必须走 EvalAI。
- **依赖冲突**：`evalai` CLI 与 `spint` 环境冲突，需单独 py3.6 环境（README 已注明）。
- **显存**：M2 `model_dim=512, num_heads=64`，训练建议单卡 ≥12–16GB；如 OOM 降 `data.batch_size`（默认 32）。
- **可复现性**：`SessionBatchSampler` 用固定 `random.Random(42)`，但训练期 `random_calibration=true` + `dynamic_dropout` 仍有随机性，务必固定 `seed` 并多 seed 取平均。
- **本文的机制验证**是 NumPy 结构级复刻（权重随机、非训练值），用于证明**不变性与维度流**；数值 R² 结论仍需在 GPU 上跑真实 `SpintModel` 得到。

---

## 附：关键命令速查

```bash
# 0. 数据软链
ln -s /home/xinyuan/Work_host/FALCON/falcon-challenge-main/data/000953 \
      /home/xinyuan/Work_host/SPINT/SPINT-main/data/000953
# 1. 环境
cd /home/xinyuan/Work_host/SPINT/SPINT-main && bash setup.sh && mamba activate spint
# 2. 训练 M2
python src/train.py data=falcon_m2 model=falcon_m2 trainer=gpu
# 3. 打包
python third_party/falcon_challenge/spint_decoder.py --run_dir logs/train/runs/<id> --checkpoint epoch_<NNN>.ckpt
# 4. 本地评测
python third_party/falcon_challenge/spint_sample.py --evaluation local \
  --model-path local_data/spint_m2.pkl --split m2 --phase minival --batch-size 7
```
