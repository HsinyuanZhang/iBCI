# FALCON 推荐检查点与装载边界 — 2026-09-08

## 决策

当前可执行的推荐分为三个独立赛道。H1 已有冻结的官方提交；M1 和 M2 是在各自已声明的开发选择面上选出的候选，尚未以此清单产生新的 EvalAI 提交。候选清单的机器可读依据为 [`falcon_recommended_checkpoints_v1.json`](../results/diagnostics_v1/falcon_recommended_checkpoints_v1.json)，SHA-256 为 `d2a34bcbe84d3795c2fad533fff0419a9fbe87a6fbf645f60f759fe81b335706`。

| 赛道 | 推荐 | EMA epoch / 成绩 | 选择面与规则 | 实物与完整性 |
| --- | --- | --- | --- | --- |
| H1 | 已冻结的 RIFT R300 `582073` | e22；官方 HO equal-session mean `0.4027782688014744`，std `0.14526658466582196` | 官方冻结结果；不把这一项和 M1/M2 开发选择分数排名 | [密封 payload](../../tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/artifacts/h1_rift_r300_recency_e22.pkl)；SHA `7957ce7b52596745aab55e5955e8763cab3b2f4808c25a037688890728883b56` |
| M1 | `jointD42` | e3；`0.7342600300996954` | visible HO3、trial 0、M10、support-overlapping；各 session R2 的等权均值，EMA 最早最大值 | [epoch_003.pt](../results/rift_v1/m1_r100_joint_d_s42_formal_v1/epoch_003.pt)；SHA `d7eba8ba9cea88b1290a68ed86fa3a75d51a11256747c6232fcb1c17b341efba` |
| M2 | `concat42` | e9；`0.3900576650553506` | ext6 六 session、query trial 0、15,403 windows；有限值等权均值的最早最大 EMA | [epoch_009.pt](../results/rift_v1/m2_r50_concat_s42_formal_v1/epoch_009.pt)；SHA `78435b8e01b0beaa78057c918ec195558d9c100a692ab7a8c066766f675ac88b`；选中 EMA [包](../results/rift_v1/m2_r50_concat_s42_ext6_pick_v1/selected_ema.pt)，SHA `8d156b28a9d144ebcd39f1597c7eb1af4a4e186a3af397b2442c911b5076418f` |

H1 数值是冻结提交的官方 HO 结果；M1 与 M2 数值才是各自选择面上的 local equal-session mean。三者不用于跨数据集的官方优越性结论，也不构成机制效应结论。

## H1：冻结的官方提交优先

优先使用提交 `582073` 的密封 H1 RIFT R300 payload。它将 EMA 状态和 27 个 bank 一同密封；提交时通过同一目录的 [`h1_rift_falcon_decoder.py`](../../tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/h1_rift_falcon_decoder.py) 中 `CPUUnpickler` / `load_payload` 装载，因此推理不依赖工作区外部 encoderbank。解码器 SHA-256 为 `8ef7ff3c4292bec051e5dafab236bb6792b093708ae0ca9076bce3e777a426b2`。

冻结记录包括 [REGISTERED.json](../../tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/artifacts/REGISTERED.json)（SHA `fdaebd6964bad8f15aa2ab57b3ed0d672a550ff1ce61a78862928ac53e8cba23`）、[freeze manifest](../results/rift_v1/h1_rift_r300_official_freeze_20260907/manifest.json)（SHA `14a7014a799c43f2d1ab5f9a12e3fee9af514a10802bd41f77984bafc4d0f299`）和源 [epoch_022.pt](../results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/epoch_022.pt)（SHA `419da65ad00ff7725bca11603004f91bf64dc723a7949efcf998700bbf7c143b`）。这些文件共同绑定了提交身份、源检查点、密封实物和官方结果；恢复时应校验它们及 payload 的哈希，随后仅走 payload 自带的装载路径。

## M1：主选与近似备选

`jointD42` 的 e3 EMA 状态包含其训练得到的 B3S encoder，接口为 `live_b3s_concat`。但当前 `JointM1ConcatDecoder` 构造器会调用 `load_trainable_b3s_from_sfix`：建立 decoder 时必须保留 [B3 Sfix e11](../../tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt)，SHA `7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a`，再载入检查点的 EMA state 覆盖 encoder 参数。目标推理还须由 `install_session_memory` 提供相应的 M10 raw bank `[10,1024,64]` 与 carrier bank `[64,4]`；二者是 non-persistent buffers，不在检查点内。因此该检查点的完整 EMA 含训练 encoder 参数，但单一检查点并不自包含。对应 receipts 的 SHA-256 为：`run_meta.json` `6f364e390b984ce5672f1968223676e70e12db9d203e6f30d72d56425d5d0096`、`train_receipt.json` `9823c2dfe41ecf2e53732db23bc9374b853e88c35def42adcacea0fae3e0541f`、`score_receipt.json` `0a4d6b7d5c32532a255d1f2c9392865b266ad34c57a49096a6d2820407c4d27e`。

`frozen_concat42` e3 是可保留的近似备选：分数 `0.7341922605158908`，比主选低 `0.0000677695838046`。它需要外部冻结 B3 Sfix e11 encoder，再将 EMA 载入 concat decoder；检查点为 [epoch_003.pt](../results/rift_v1/m1_r100_concat_s42_formal_v1/epoch_003.pt)，SHA `c202daff85412470e57a81a8db199d54fe2c7ed4d79850adf39f482b9cb806d4`。这一极小差距不足以支持 joint encoder training 的收益；`jointD42` 仅因数值最大而居首。

M1 的完整候选排序可由已绑定全部检查点哈希的 [inventory](../results/diagnostics_v1/falcon_recommended_checkpoints_v1.json) 复核：`jointD42` e3 `0.7342600300996954`、`frozen_concat42` e3 `0.7341922605158908`、`jointB42` e3 `0.6630568974542209`。

## M2：RIFT concat 的装载限制

主选 `concat42` 的 e9 状态是 concat decoder 权重，装载前必须按冻结会话身份 bank 建造 decoder；`selected_ema.pt` 也不是完整的提交 payload。相关证据 receipts 的 SHA-256 为 `run_meta.json` `1ca1dd1306468eac77339677044a60e27d053c8ce78994025a4271869c4d04fa`、`train_receipt.json` `fd98ceac87a2bbbc0169d25905581844c4d7fd546f3b65db83d8ed30511c0c72`、`manifest.json` `5d1bb0bf640c3d6ef646a0a6f7366bd2d3ff6befc506325da2d0c7aa7ca38884`、`score_receipt.json` `6a985e1f3fb9392198725d6920efb7ea24b41052ddae84a335ae66ea72022e31`、`validation_audit.json` `26158347862d2e48a8baae13a72b7ec3cdf0cfc4e7a29f45541d8f0d1de8621b`。

因此，M2 推荐是一个经 ext6 选择的开发检查点，而不是可直接提交的完整运行包。`nonattn` 等机制控制只属于 ext4 匹配机制协议，已明确排除在 ext6 提交排名之外，不能以它们替换或重排这里的 M2 选择。

M2 的完整候选排序也在该 [inventory](../results/diagnostics_v1/falcon_recommended_checkpoints_v1.json) 中：`concat42` e9 `0.3900576650553506`、`jointD43` e8 `0.385522723151031`、`jointD44` e11 `0.36275330837379643`、`jointD42` e13 `0.3478275158419855`。

作为历史运行参考，旧的 `SMALL` concat e8 在 ext6 得到 `0.3990687579684978`，比当前 RIFT concat e9 高 `0.0090110929131472`；但其选择基于两个 seed、48 行的 cross-seed 记录，且架构与 RIFT 不匹配，故不改变本清单的 RIFT 主选。该旧 payload 是 [m2_small_trf_s42_ema_e08_ext6.pkl](../../tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl)，SHA `4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4`；其 [payload receipt](../../tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/artifacts/payload.receipt.json) SHA 为 `84b6c2afa8a403023b7f385c347afcd5c6927e91d5a5bcb2cc6f3909943cabc7`，实际源 [epoch_008.pt](../../tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000/S1_SMALL_COS/seed42/epoch_008.pt) SHA 为 `68c1694f73ee3932df41a7bcdf8b40b19cce43ede93eca83f8f3047070ebdd33`。

## 使用顺序

1. H1 如需复现已冻结结果，校验密封 payload、`REGISTERED.json` 和 freeze manifest 后使用自带解码器加载。
2. M1 如需开发复现实验，先选 `jointD42` e3 EMA；只有在明确需要冻结 encoder 的复现路径时再用 `frozen_concat42`。
3. M2 如需开发复现实验，使用 `concat42` e9 EMA，并提供与该运行一致的冻结 session banks。本轮没有以这些候选新提交 EvalAI。
