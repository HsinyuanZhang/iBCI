# RIFT：L200 质量对照与 CPU 推理优化

日期：2026-09-07。状态：L200-recency RUNNING；CPU runtime 首轮优化及最终复跑已验收。训练见 §7，CPU 结论与复现方式见 §8。

## 1. 用户授权与两条独立目标

用户要求并行交给 subagents：启动 L=200 的 RIFT 训练，用来检验效果而不是用缩窗换速度；同时优化 CPU 推理，争取超过 BT-EORT。随后限定当前只有一张空闲 GPU，先训练 recency，flat 稍后再启动。

- **质量线：RIFT-L200-recency。** 本文 L200 指 200 个原始 H1 输入 bins，在既有 RIFT 命名中对应 R200。先运行 recency；flat 准备为同配置待启动项，不在首次派发时占卡。
- **速度线：RIFT-R300 e22 EMA。** 固定上一轮已选中的 R300 模型函数与权重优化 CPU 执行，避免把 L200 缩窗带来的差异算作纯运行时优化收益。对照旧系统 BT-EORT H1 B2 e18 / L200。

本轮不授权官方提交、镜像发布或覆盖历史结果。GPU 1 的 M1 concat 任务及其后续评分保持不动。

## 2. 所有权与共享接口

| 责任 | Agent | 文件边界 |
|---|---|---|
| 真局部窗口训练 attention、dense oracle、梯度验证 | rift_temporal | temporal.py 与 temporal tests |
| H1 L200 CLI、数据合同、microbatch 预检与启动 | rift_training | h1_train.py、h1_preflight.py、相关 tests/新启动辅助脚本 |
| 固定 R300 权重的 CPU runtime、BT-EORT 对照计时 | rift_frontend | 独立 cpu_runtime 模块、CPU benchmark/helper/tests/results |
| 架构与接口裁决、验收、资源协调、交付 | 主 agent | 文档、集成检查与协调 |

各 agent 不撤销其他编辑，不改 v1/旧 submission。model.py、streaming.py 和 config.py 暂保持不动；训练器构造 RiftDecoder 后通过 temporal.set_attention_backend 选择 local/dense，后端选择不是可训练参数且不进入 state_dict。CPU 优化不依赖修改训练核心，避免冻结冲突。

## 3. L200 质量训练合同

保持 H1 D4、width256、heads8、FFN512、k5/C16、P16、8 slots、seed42、13 source sessions、C2-CAL-1 B2 bank/预算/顺序、有效 batch32、32 epochs、每 epoch 731 optimizer updates、LR/EMA/clip/dropout/目标尺度与上一轮一致。H1 raw context 从 300 改为 200，每层窗口为 `[50,50,50,49]`，满足 `5+49+49+49+48=200`。不把每层窗口设为 200。

按用户此前要求，正式新训练使用真正 local attention，不生成完整 T×T scores 后 mask；原 dense 图保留为独立正确性 oracle。需验证 local/dense 的 FP32 输出与输入/参数梯度、冷启动、因果边界、warm-up 和 recency/flat。R300 既有 state_dict/EMA checkpoint 仍可读取；新来源 SHA 单独记录，不改写旧结果的源码标识。

microbatch 在 16/32 中通过真实 GPU optimizer-step 测量选择，优先检查 32；有效 batch 仍为 32。微批数变化不能改变更新预算、尾批权重、bank/unit-mask 或 LR/EMA 计数。CPU 前检通过后才在唯一空闲设备上进行有限 CUDA/bf16 smoke 和 16/32 吞吐/显存比较，随后启动 recency。flat 暂待后续空闲设备再协调，不与 recency 共用忙卡。

后续按同一 HO-M3 开发选模面比较 L200 与既有 R300，并同时报告选中 epoch、同 epoch 曲线及 per-session 分数。此处研究问题是上下文长度对质量的影响；dense→local 和 microbatch 改动是经过等价验证的执行变化，仍需在实验元数据中披露浮点路径不同。

## 4. CPU 推理合同与验收

RIFT 源：[R300-recency e22 checkpoint](../results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/epoch_022.pt)，使用其中 EMA 而不是 raw 参数。BT-EORT 源为封存 [H1 B2 ORT 实现](../../tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/h1_exacte_ort.py)及其现有 `artifacts/ort_graphs/`，不改旧图或权重。

CPU 优化必须保持 RIFT 的 R300、D4、FP32、recency half-lives、稳定 k5 token、时间和流状态语义。不剪层、不量化、不改模型权重、不以 R200 替换 R300 后称运行时胜利。候选与原 RIFT 的逐输出误差门为 `abs(error) <= 1e-5 + 1e-5*abs(reference)`；BT-EORT 是另一个模型，不要求两者预测相等。

优先检查并优化：session-static E0 投影/融合缓存、每步 bank hash/转换、Python 状态 pack/unpack、KV 临时分配与有界环形缓冲、实际 temporal 和 set frontend 内核。不同实现可以使用不同后端，但必须在同机、同 CPU 配额和线程预算下测真实端到端推进，不比较 GPU 与 CPU，也不拿重复 predict 的缓存读取代替新 bin 消费。

报告 B1/B8（必要时 B2/B4）的 startup/reset、steady mean/median/P95、实际预测/观察调用数、线程和重复轮次；使用相同 public H1 神经流与各自合法固定 bank，包含冷启动、窗口回绕、异步结束、stream_id 重排/reset。计入数据转换、bank/状态管理、frontend、temporal、readout，不能只报最快内核。只在 RIFT 保真通过且同口径实际测得优势时称超过 BT-EORT；未达标则报告真实瓶颈与差距。

## 5. 当前资源与启动规则

2026-09-07 11:09 UTC 快照：GPU 0（RTX 3090，UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`）无 compute PID，453 MiB / 0%；GPU 1 正在运行其他 M1 concat D4，PID `1910683`，约 99% 使用率，不视为空闲。

启动前重新进行两次空闲检查；GPU 0 只派发 recency，任何新冲突均等待，不停止其他任务。CPU 优化限制线程且不初始化 CUDA。每次实际 CUDA smoke、正式 PID/命令/代码 SHA/结果路径及 CPU benchmark 都应有独立落盘证据；本页不把计划中的动作写成已完成。

## 6. 已完成的 CPU 预检与审查

2026-09-07 11:23 UTC：L200 的真实数据 [CPU preflight](../results/rift_v1/h1_r200_local_cpu_preflight_20260907T111544Z/preflight.json) 通过。输入为 `[2,200,176]`，与 R300 预检共用 endpoint 3146 / 3150；13 sessions、731 updates/epoch、C2 M3 bank 保持不变，参数与 EMA 更新成功，梯度范数 3.675262，未初始化 GPU。

正式隔离用户 site 的 conda 环境为 Torch 2.5.1.post303。true-local temporal 的 25 项测试通过，覆盖 R200 / R300、recency / flat、全 invalid 行与左侧 padding 的输出和完整输入/参数梯度对照，并直接检查实际 QK scores 的最后一维为每层窗口宽度，而非 T×T。

独立 CPU runtime 的首阶段缓存 bank tensor、raw4 和 batched stream state；reset/reorder 小几何回归已修复。针对固定 H1 EMA 权重的完整保真与 BT-EORT 对照计时尚在执行，不将早期相对 RIFT reference 的约 1.32× 初测当作最终跨系统速度结论。审查要求添加权重版本守卫和空 unit-mask 拒绝；连续 KV 缓存作为独立模块并行优化，冻结训练源码不受影响。

## 7. CUDA 验收与 recency 正式启动

两种设置均使用独立初始化、真实 R200 数据与固定有效 batch32，5 个 optimizer warmup updates 后测量 15 个 updates；不计 CPU bank 构建与 checkpoint 保存时间。

| microbatch | step median / mean | peak allocated / reserved |
|---|---:|---:|
| 32 | 0.158314 / 0.158352 s | 6351 / 7658 MiB |
| 16 | 0.158932 / 0.158952 s | 3224 / 4362 MiB |

速度相差约 0.4%，视为基本持平；选择默认 microbatch32，保持一次微批即一次有效 batch32，24 GiB 卡上显存充足。原始证据：[MB32](../results/rift_v1/h1_r200_recency_local_bench_mb32_20260907T112000Z/benchmark.json)、[MB16](../results/rift_v1/h1_r200_recency_local_bench_mb16_20260907T112100Z/benchmark.json)。两次 BF16 optimizer 短测均 finite；同初始化真实 B2 的 CUDA FP32 local/dense 输出最大绝对差 `1.79e-7`、loss 差 `1.49e-8`、参数梯度差不超过 `6.71e-8`，主审接受。

正式进程 `1915357` 于 11:28 UTC 启动，完成数据准备后 11:29:04 UTC 写入 [run_meta.json](../results/rift_v1/h1_r200_recency_local_mb32_formal_20260907T112800Z/run_meta.json)。仅绑定 `CUDA_VISIBLE_DEVICES=0`，使用隔离用户 site 的 conda Python；`context_bins=200`、`attention_backend=local`、`microbatch=32`、`epochs=32`、`variant=recency`。初始化 SHA 与 CPU 配对预检相同，短测权重没有复用。实际 argv、PID、环境与四个训练入口/核心文件 SHA 均记录在 run metadata。

主审已确认 GPU UUID 与该 PID 对应，且 [heartbeat.json](../results/rift_v1/h1_r200_recency_local_mb32_formal_20260907T112800Z/heartbeat.json) 到达 step300：loss 0.008137、grad norm 0.232525、elapsed 48.05 s、peak allocated 6367 MiB，均正常。训练完成后同进程执行全部 32 个 EMA checkpoints 的 HO-M3 开发面评分；此时仍应视 GPU0 为本任务占用。

flat 未启动、未自动抢占 GPU1；GPU1 的 M1 训练及其后续阶段保持原调度。L200 的质量结果尚未产生，不能从训练 loss 推断是否优于 R300。

后续主审确认：R200 与历史 R300 的初始化、endpoint inventory、bank roster SHA 一致，前两个 epoch 的 endpoint/bank/unit-mask 顺序 SHA 也逐一相同。两个原始输入 valid-mask inventory 的 SHA 因长度不同而正常不同。11:51 UTC 时训练已完成 11 epochs，进入 epoch12 / step8300；四个训练入口/核心文件的当前 SHA 均仍与 run metadata 一致，CPU 优化没有改变训练过程中的源码。

交付前 11:56 UTC 快照：已完成 14 epochs，epoch15 / step10350，loss 0.001655、grad norm 0.03633；PID1915357 持续健康运行，正式 GPU0 显存占用约 7824 MiB。仍未启动 flat。

## 8. CPU 首轮优化：最终冻结复跑通过

交付 [CpuRiftRuntime](../src/btransform_unified_v2/cpu_runtime.py) 与独立 [CpuRiftTemporalRuntime](../src/btransform_unified_v2/cpu_temporal.py)。前者固定注册 bank tensor 和 stream slots，免去逐 bin bank 内容 hash/转换与 adapter pack/unpack；后者将每层历史小张量列表替换为右对齐、批量更新的有界连续 K/V buffer。模型函数、权重、FP32、R300、D4、recency half-lives 均未修改。

推荐主候选显式指定 `CpuRiftRuntime(model.eval(), banks, stream_ids, temporal_backend="cached")`。`state` 后端保留为对照；没有逐 batch 选择最有利的后端。缓存具备参数版本变更拒绝、空 unit-mask 拒绝、stream reorder、独立 reset、异步 valid-mask 不推进 raw/KV 的保护。它是独立 CPU runtime，尚未替换旧提交包或发布新的 Falcon 镜像。

最终证据：[冻结源码复跑 benchmark.json](../results/rift_v1/cpu_trained_h1_final_20260907T1154Z/benchmark.json)，完成于 11:51:55 UTC。首轮探索测量也保留在 [first benchmark.json](../results/rift_v1/cpu_trained_h1_20260907T1142Z/benchmark.json)，不覆盖旧数据。主审验收和源码 SHA 见 [integration_receipt.json](../results/rift_v1/cpu_trained_h1_final_20260907T1154Z/integration_receipt.json)。

### 口径

- 同一 AMD Ryzen 9 7950X，进程绑定两个物理核 12 / 13；Torch intra/inter 为 2 / 1，ORT 为 2 / 1。宿主共享、非独占，不将观察到的波动归因到未验证的具体原因。
- 固定 RIFT R300-recency e22 EMA；BT-EORT 使用封存 B2 e18 / L200 payload 与已有 ONNX 图。此处是工程延迟比较，不是两模型预测质量比较，也不使用新 L200 训练权重。
- B1 / B8 使用同一批真实 public H1 连续 neural segments（每个 session 的 raw 0:400）；非零比例约 21.5% / 22.0%，不是合成全零输入。BT session → tag → bank 映射由 sealed payload receipt 明确解析。
- 每种配置每轮冷启动后推进 300 bins，测量后续 100 次真实推进，三轮轮换引擎顺序。计时覆盖 NumPy 输入、转换、frontend、状态、temporal、readout 与 native-scale NumPy 输出；B8 是整批 8 路一次推进的延迟，不是除以 8 后的单路耗时。
- 两个 batch 均完成 401 次 trained-weight 严格逐元素对照（400-bin cold-to-full-context、重排、单流 reset），相对冻结 RiftStreamDecoder 的最大输出差为 **0**；容差门为 `1e-5 + 1e-5 * abs(reference)`。
- BT 每轮实际 predict counter 为 400，advance / rebuild 的 B1 或 B8 ORT session 确认已建立；不是 eager fallback。

### 最终数据

下表为 mean / median / P95，单位 ms；所有轮次均保留在 JSON。

| Batch | 原 RIFT reference | RIFT state | RIFT cached（推荐） | BT-EORT ORT |
|---|---:|---:|---:|---:|
| B1 | 3.121 / 3.087 / 3.338 | 2.725 / 2.673 / 3.055 | 2.707 / 2.697 / 2.803 | 18.051 / 14.299 / 22.562 |
| B8 | 22.225 / 22.287 / 22.956 | 20.779 / 19.846 / 20.724 | 15.656 / 13.053 / 19.335 | 112.955 / 73.563 / 178.842 |

本轮统一 cached 后端相对 BT-EORT 的平均耗时比为 B1 **6.67×**、B8 **7.21×**；按中位数为 **5.30× / 5.64×**。RIFT 本来的增量结构已经在这组基准上快于 BT-EORT；本轮 runtime 优化相对原 RIFT 额外降低平均延迟约 **13.3% / 29.6%**，不能把整个跨结构速度差都归功于本轮缓存修改。

BT-EORT 和部分 RIFT 测量存在轮次波动；以上是指定共享宿主、线程预算和输入片段下的工程结果，不是官方服务器的固定加速倍数，也不是完整比赛任务的 latency 结果。

### 复现与测试

从 repository root 运行：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
taskset -c 12,13 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/benchmark_cpu_trained.py \
  --checkpoint btransform_unified_v2/results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/epoch_022.pt \
  --dest btransform_unified_v2/results/rift_v1/cpu_trained_h1_new_unique_run
```

`--dest` 必须是尚未存在的新目录。为运行封存 BT-EORT，使用其已有 vendor wheel 以 `pip install --no-deps` 仅新增了 `onnxruntime==1.19.2`；没有升级 Torch、NumPy 或其他训练依赖。

最终集成测试：正式 conda、`PYTHONNOUSERSITE=1`、无可见 CUDA、禁用 pytest 外部插件、2 threads、测试进程绑定 10 / 11 核，运行全部 `btransform_unified_v2/tests`，结果 **53 passed in 5.29s**。测试后与最终 benchmark 的 runtime / temporal / script / tests SHA 均已核对，源码保持冻结。
