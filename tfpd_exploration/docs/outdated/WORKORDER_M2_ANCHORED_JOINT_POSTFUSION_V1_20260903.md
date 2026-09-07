# Work Order — M2 Anchored Joint Post-Fusion（AJPF）V1

日期：2026-09-03  
状态：**实现授权；通过独立 no-CUDA 审计后的单次 GPU0 训练与单次 CPU 评分授权**

本工作单授权实现并在所有前置门通过后执行一次 M2 AJPF V1。它只回答一个问题：

> 在同一个强 Selected-T4 / native early-pooling checkpoint 上，以完全匹配的 source
> causal activity、固定 carrier、优化步数和 decoder 训练合同联合微调时，anchored
> post-fusion 是否能同时超过 joint-native 对照和历史强 POOLED 基线？

不得把本实验称为 from-scratch；不得把 task-only joint adaptation 与旧 frozen-decoder
`task_plus_y_plus_E` screen 的差异归因给某一个单独因素。

## 1. 权威设计与执行范围

设计文件：

```text
tfpd_exploration/docs/DESIGN_M2_ANCHORED_JOINT_POSTFUSION_V1_20260903.md
sha256 e92eafea22f473ce91eee57223436c61df6d0f881690dd8a3eccc1f24e193523
```

本工作单授权的 additive 路径固定为：

```text
tfpd_exploration/src/m2_anchored_joint_postfusion_v1/
tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v1.py
tfpd_exploration/tests/test_m2_anchored_joint_postfusion_v1.py
tfpd_exploration/results/m2_anchored_joint_postfusion_v1/training/
tfpd_exploration/results/m2_anchored_joint_postfusion_v1/score/
```

旧 PF、APFG、AOF、POOLED/CDM result root 均为只读历史证据，不得修改、删除、覆盖或
重解释。训练和评分各只能有一个 canonical attempt；失败后不得在同一根自动重试，必须
使用 successor root/work order。

GPU 边界：

- 训练只允许 physical GPU0，canonical UUID
  `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`；
- GPU1 属于并行 M1 工作，禁止查询、占用、attach、发信号、改 affinity/priority 或读取其
  进程细节；
- target/external 评分只允许 CPU；
- 不授权 EvalAI、ECR、Docker、镜像、网络下载、push 或任何外部提交。

训练环境必须精确为：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONPATH=/home/xinyuan/Work_host/SPINT
```

评分环境把 `CUDA_VISIBLE_DEVICES` 固定为空字符串，其余 CPU/路径变量不变；导入、strict
load、评分和 final revalidation 期间 `torch.cuda.is_initialized()` 必须始终为 false。

## 2. 历史不可变权威

### 2.1 Selected-T4 checkpoint

```text
path
streaming_calibration_exp/outputs/streaming_calibration/
m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt

checkpoint sha256
25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e

strict-loaded student-state sha256
2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20
```

必须 no-follow 读取、先验证 checkpoint body SHA，再反序列化；完整 Lightning state
strict-load 后验证 student state。禁止从旧 PF student-only checkpoint 猜测或部分加载。

### 2.2 Source causal-coordinate authority

只读根：

```text
tfpd_exploration/results/m2_anchored_postfusion_gate_v1
```

该根的历史运行最终失败于后续 target scorer，不改变其已经落盘的 source authority。
执行只允许在 attempt 之后 held-FD/O_NOFOLLOW descriptor-read 下列 body，并验证 body/sidecar、
regular file、mode `0444`、`nlink=1`、根 inode 不变：

```text
attempt.json          e546cb34f3efc32b8d6056e5eb6dc74877a9a548543d51a1040d149f1e5407a7
launch.json           9747223a8a5afed21e9ff59df3d0c44bea85c9a244c6095bf5afa5ace7c9dfef
source_authority.json ae1f02c97d507c28fb35165ab616100156e279fa855cb3976d61ab9ff55c0ccf
alpha_selection.json  c60a9e29928a8f559881745a727d76716b2b458c1aa3d892772f6b8b65eef908
input_authority.json  41185bceb1307cebc1afb2d7672c55bc71d601ba028a4c0d694310d8af366e24
failure.json          6e979a4b8abc9761cf6dd9b521bfc7c4a08422ba9d775530bb0cecfc72a23f1d
```

只有 `source_authority.json` 的 source topology 是本实验输入权威；历史 alpha、selection、
target input 和 failure 仅用于证明 lineage，不能进入 AJPF 选择。live rematerialization 必须
精确重算：

```text
source session count           7
M30-causal coordinate count    91717
canonical shared batches       3455
ordered coordinate sha256      06f19e267cb2029371bb4b3b3b6019314db593edfe44417d9305e7498deb7cba
ordered batch sha256           dffe4234befcc197aa47c2c6171b9c0fc9194085d96d2667d56b7308de533095
activity authority             pooled_g00m_linear
```

同时逐 session 精确复现 D-opt4 support、selected-support4 carrier、normalized side、raw-M30
audit T4、normalizer 和 coordinate count。历史 selection R2/alpha 不得加载进 optimizer。

### 2.3 强 POOLED comparator

历史根精确为两叶：

```text
tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/
  score.json
  score.json.sha256

score body sha256
455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6
```

必须 held-FD 验证 `0444`、`nlink=1`、sidecar、schema/status、130-row canonical topology，
再提取恰好 13 个 `m4_activity_only` / budget4 rows。历史 POOLED 只存在 `FIXED30` law；
不得生成 `POOLED/UNCAPPED` 别名。

## 3. 三臂与初始化

固定 arm 顺序：

```text
J-NATIVE
J-R1
J-MEAN
```

1. attempt 之前不得导入 Torch、打开 checkpoint/data、初始化 CUDA 或打开历史 result body。
2. attempt 和 launch 之后只构造一次 source PIT/DataModule，并 strict-load 一个 `25d7...`
   native module。
3. 从 strict-loaded module 深拷贝三个完全独立的模型；不得共享 Parameter、buffer、optimizer
   state 或 module object。
4. `J-NATIVE` 保留 native B3S；`J-R1/J-MEAN` 在 strict-load 后安装新的 route-owned
   `AJPFIdentityAdapter`。它只复用 `PostFusionIdentityAdapter` 的 whole-stack 算法，不得调用
   旧 `m2_postfusion_variant_screen_v1.install_variant()`，因为旧入口要求 frozen decoder 并只
   接受旧 PF arm 名。J-R1 的 alpha 必须是 float32 scalar IEEE `+0`；J-MEAN 无 alpha。
5. wrapper 安装前后，各臂 native substate SHA 必须保持
   `2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20`。
6. 第一个优化步前，同一 finite calibration/side/neural/target 下，J-R1 与 J-NATIVE 的
   identity、prediction 和 task loss 必须 bitwise 相同；post identity 和 residual delta
   必须 finite。训练表达式不能用 zero branch 截断 alpha gradient。
7. J-R1 比 J-NATIVE 多一个 scalar 参数；收据必须披露，不能声称 strict equal-parameter。

三臂只解冻并训练：

```text
student.id_encoder.*     lr 1e-4
student.decoder.*        lr 1e-5
```

所有其他 student 参数、teacher、normalizer、T4 与数据统计量冻结。每臂必须显式设置
`student._decoder_frozen=False`，再 `student.train(True)`，并证明 decoder training=true。
禁止调用 inherited `configure_optimizers()`。

每臂使用 route-owned fresh Adam，两组参数按 object identity 验证：组内无重复、组间无交集，
并集恰好等于全部且仅全部 authorized trainable parameters；teacher 参数数为 0。固定 Adam：

```text
betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=false
scheduler=None, gradient clipping=None
```

## 4. Source 训练坐标与池

所有七个 source session 都进入训练；不做 source split、epoch/arm/LR 选择。source material
只允许 POOLED/G00m linear raw activity，禁止 PIT cubic calibration tensor。PIT/DataModule 只
承担 source-only 模型构造、task/config、normalizer、source raw-session/dataset array authority；
训练入口不得调用普通 `train_dataloader()`、`val_dataloader()` 或 dataset `__getitem__`。每个
训练 batch 必须由 route-owned M30-ready canonical replay 直接按 frozen coordinates 构造，
测试必须证明没有恢复 static-prefix/M33-side loader 路径。

每 session 在 first30 上冻结一次 D-opt4 support，只用 selected support4 拟合固定 ridge T4。
carrier 固定于整个 M4/M10/M30 训练与评分；raw M30 T4 只作 audit。Hz carrier 必须以 float64
乘 `0.020` 后 cast float32，再使用 source normalizer。不得使用原 DataLoader 的 M33 side。

canonical coordinate 排序：

```text
lexical session -> query trial -> window_start
```

相同 `(session, query_trial, M30 pool state)` 的连续窗口组成 resident batch，最大 32；不得跨
session/query/state。每 epoch 精确 3455 个 shared batch group，12 epoch 共 41460。

pool cycle 与 controller 固定为：

```text
POOL_CYCLE = (4,10,30)
requested_M = POOL_CYCLE[(epoch_one_indexed - 1 + canonical_batch_ordinal) % 3]
```

所有 canonical coordinates 都预先满足 M30 因果可构造性；M4 为 support4，M10 为
support4+最近6个已完成 non-support，M30 为 support4+最近26个。若 live state 不足，必须
失败，不能 clamp、pad、repeat、跨 trial、使用当前/future trial 或跳过 coordinate。

同一个 shared group 的三臂必须收到相同 ordered activity members、selected-support4 carrier、
normalized side、neural windows、targets 和 target mask。每组只允许一次 host materialization
和一次 H2D。

## 5. Loss、随机性与显式 dropout

训练严格 12 epoch、batch<=32、seed42。每臂每 shared group 恰好一次 forward、backward、
Adam step。checkpoint 固定为 epoch12；source monitor、loss、alpha 或 external 结果都不能
选择 checkpoint。

loss 只能是 M2 source behavior 的 scaled last-bin MSE：

1. 直接调用 student path，不调用 inherited `model_step`；
2. prediction 按 inherited `predict_scaled_behavior=true` 除以 5.0；
3. 只比较 `prediction[:, -1:, :]` 与 `target[:, -1:, :]`；
4. teacher forward calls 必须为 0；
5. 无 distillation、identity/prediction consistency、smoothness、contrastive 或 auxiliary loss。

三臂 decoder 已解冻，必须共享同一 dynamic whole-unit dropout 实现。每 shared group：

1. 从 common pre-draw RNG state 只抽一次原始 dynamic probability；
2. 只生成一次 `[B,N]` scaled dropout mask；
3. 保存 post-mask Python/NumPy/Torch CPU/CUDA RNG state；
4. 三臂各自恢复该 post-mask state，应用同一 p 与 mask，再运行 transformer；
5. 三臂结束 RNG state 必须完全相同，之后只推进一次 canonical state。

不得通过 `neuron_gate` 代替 unit dropout，因为它只作用 neural、早于 identity addition；也不得
分别预乘 neural/identity，因为这会改变 `(neural+identity)*mask` 的浮点结合顺序。实现必须用
route-owned first-call `decoder.fc_in` forward-pre-hook，把 governing mask 精确施加在 identity
addition 之后、第一次 fc_in 之前；hook 只能命中第一次 fc_in，第二次 rep fc_in 必须不变，
每 arm 每 step 精确计数并在 finally 中移除。route clone 的内建 dynamic/dropout draw 关闭，
但 student/decoder train mode 保持开启；原始 dropout 配置值必须另行记录。

ordinary step 不做全模型 CPU hash。12-step smoke、每 epoch 首/末 sentinel 和 epoch 边界记录：

- common p/mask/RNG digests；
- 每臂 hook first/rep call count、internal dropout draw=0；
- encoder/alpha 与 decoder group 的 finite grad、nonzero-grad、update norm；
- loss、state digest、optimizer-state digest；
- alpha value/gradient（J-R1）。

单参数/单 batch 的合法零梯度不 fail；但 12-step smoke 和每个完整 epoch 中，每臂 encoder
组与 decoder 组都必须至少一次 nonzero group gradient 和 nonzero update，且不能出现 nonfinite。

## 6. GPU0 运行与成本门

三个模型同时常驻 GPU0，一个 source materialization、一个 resident batch、固定 arm 顺序串行
执行。禁止启动三个进程或 CUDA streams。

正式 epoch 前运行同一 iterator 的前 12 个 shared groups；这些步骤计入 epoch1，不能重复。
smoke 记录 elapsed、steps/s、arm forwards/s、peak allocated/reserved、投影 12-epoch wall time。
只有以下全部满足才继续：

- logical CUDA 只有 device0，UUID 与本工作单一致；
- GPU0 启动前 free VRAM 至少 4 GiB；
- 三模型 peak reserved 不超过 24 GiB；
- 投影总 wall time不超过 7200 s；
- 所有数值、RNG、mask、hook、gradient 与 update gate 通过。

失败不得删除 J-MEAN 或改变 batch/LR/epoch 后在同根重跑。

## 7. 训练工件与不可变生命周期

root-only opaque one-shot capability 必须绑定：canonical fresh root/parent inode、设计/工作单
SHA、当前 explicit source closure、GPU0 profile、历史 source/POOLED descriptors、checkpoint
SHA 与 exact training plan。public CLI 默认 inert，不能自行 mint capability。

训练成功根的 topology 必须精确为下列 body/sidecar 与唯一 `checkpoints/` 子目录；不得有
其他文件、目录、临时文件或 symlink：

```text
attempt.json
launch.json
source_authority.json
smoke.json
epoch_01.json ... epoch_12.json
checkpoints/J-NATIVE_epoch12.pt
checkpoints/J-R1_epoch12.pt
checkpoints/J-MEAN_epoch12.pt
manifest.json
terminal.json
```

`checkpoints/` 必须精确包含三个 body/sidecar pair：

```text
J-NATIVE_epoch12.pt
J-R1_epoch12.pt
J-MEAN_epoch12.pt
```

因此成功根顶层恰好有 18 个 JSON body/sidecar pairs 加一个 `checkpoints/` 目录，checkpoint
目录恰好有 3 个 artifact/sidecar pairs。不得发布 epoch0、epoch1–11 checkpoint、best/last
别名、training-summary 副本或 optimizer checkpoint。

每个 body/artifact 都有 basename-bound `.sha256` sidecar，regular `0444`、`nlink=1`。checkpoint
必须由 fresh model strict-load 并验证 student state/forward finite、eval repeated-forward bitwise、
state before/after不变。成功 root 不含 failure；失败保留完整 immutable prefix并发布唯一
failure，不删除已发布工件。final/failure 前重新验证 closure、root identity、历史 descriptors、
GPU identity和已发布 prefix。

failure 根只能保留上述 canonical 顺序中已经完整事务发布的 prefix（以及已经完整发布的
checkpoint pairs）加唯一 `failure.json/.sha256`；不得含 terminal 或 extras。任何半个 body/
sidecar、空临时目录或未列名 checkpoint 都是非法失败拓扑。

## 8. CPU 评分

训练 terminal 成功后才允许新建独立 `score/` root。score capability held-FD 绑定训练 terminal、
三 checkpoint/manifest descriptors、历史 POOLED score 与当前 scorer closure。评分运行必须
`CUDA_VISIBLE_DEVICES=''`、CPU、`eval()/inference_mode()`、zero optimizer/parameter/target update。

在读取任何新 arm R2 作为结果前，先从 `25d7...` strict-load 一个未训练 J-NATIVE reference，
对相同 13-session materialized inputs 运行 native G00m `FIXED30` replay。它必须逐 session精确
复现历史 `m4_activity_only` 的 prediction SHA、R2、starts、targets、window count，同时精确
重建 selected-support4 carrier、normalized side 和 normalizer authority；13/13 任一失败即停止。

随后每个新 checkpoint 固定评分：

```text
J-NATIVE / FIXED30
J-NATIVE / UNCAPPED
J-R1     / FIXED30
J-R1     / UNCAPPED
J-MEAN   / FIXED30
J-MEAN   / UNCAPPED
```

共 13×3×2=78 行。J-R1/J-MEAN 只允许 whole ordered raw-activity stack operator；禁止旧的
singleton residual identity 再平均。decode-before-commit、support永不驱逐、FIXED30 total30、
UNCAPPED 不截断。所有 78 行共享 exact starts/targets/support/carrier/query-activity evidence。

输出 external 6 与 within 7 的逐 session delta、equal-session mean/median/positive、seed42 的
10000 次 ordinary session-bootstrap 95% CI。历史 POOLED 不重跑，只从 sealed row 取值。

score 成功根的 exact body/sidecar allow-list 为：

```text
attempt.json
launch.json
producer_authority.json
input_authority.json
score.json
terminal.json
```

不得有子目录或 extras。score failure 只能是该顺序的完整 prefix 加唯一
`failure.json/.sha256`，不得含 terminal；所有 leaves 均要求 regular `0444`、`nlink=1`、
basename-bound sidecar，terminal/failure XOR。

## 9. 预注册判定

主 arm 固定为 `J-R1/UNCAPPED`，不能按 target 选择 J-NATIVE/J-MEAN、law 或 checkpoint。

```text
method_delta = J-R1/UNCAPPED - J-NATIVE/UNCAPPED
practical_delta = J-R1/UNCAPPED - historical POOLED/FIXED30
```

论文级成功要求两项 external contrast 同时满足：

- equal-session mean `>= +0.010`；
- positive count `>= 4/6`；
- worst session `>= -0.015`；
- same-input/state/RNG/scorer/lifecycle 全通过；
- no target update/selection/checkpoint selection。

`practical_delta` 是 AJPF route 加 UNCAPPED continual memory 的 total deployed-effect，不是纯
placement estimand。纯 route effect 只由同 law 的 `method_delta` 判断。

`+0.005 <= mean < +0.010` 只能写成机制性小正数；小于 `+0.005` 或少于 4/6 为 null。
任何 reference 下降造成的表面胜利无效。若 J-NATIVE 自身大幅提升而 J-R1 不胜 J-NATIVE，
结论只能是一般 joint fine-tuning 有益，不是 PF 成功。

## 10. 实现前的强制 no-CUDA 门

Terra 完成实现后、root 发 live capability 前，必须由独立 reviewer 重跑并检查：

1. design/work order/current closure SHA 与 exact-key-set；
2. attempt-first import boundary，clean `python -S` dry import 无 Torch；
3. held-FD source/POOLED descriptor topology/mode/sidecar/body/semantic adversaries；
4. strict-load-then-wrapper、三臂独立 Parameter/buffer/state；
5. decoder 真正解冻/train，teacher与未授权参数冻结；
6. two-group Adam object-ID exact union/disjoint 与 LR/hyperparameters；
7. J-R1 IEEE +0、finite post/delta、identity/prediction/loss bitwise parity、first-step alpha gradient；
8. causal linear activity、selected4 carrier、exact 4/10/30 membership/controller/count/digest；
9. explicit shared mask 的原位置 hook、single draw、zero internal draw、paired end RNG；
10. task-only last-bin、teacher calls 0、group gradient/update coverage；
11. 25d J-NATIVE `FIXED30` 对 13 个 sealed POOLED row 的 exact sentinel；
12. generalized strict scorer 的 whole-stack、FIXED30/UNCAPPED、78-row law；
13. production-shaped success/failure lifecycle、capability reuse/root/closure/env drift rejection；
14. py_compile、`git diff --check`、inert CLI。

实现 ownership 只允许新 AJPF package、inert script、focused test、上述 canonical result roots。
不得修改 `streaming_calibration_exp/**`、任何 M1 package、旧 PF/APFG/AOF package 或旧 result
root；route-owned first-call prehook 必须在新 AJPF package 内实现。审计需在实现前后分别重算
共享 `streaming_spint.py`、`streaming_encoders.py`、`streaming_calibration_module.py` 和
`falcon_datamodule.py` body SHA，任何改变都使本工作单失效。

只有这些门全部通过且 GPU0 preflight 合格，才允许本工作单所述的一次正式训练和一次 CPU
评分。任何科学合同需要修改时，本工作单失效，必须先更新设计与 SHA，不能在 live code 中
静默放宽。
