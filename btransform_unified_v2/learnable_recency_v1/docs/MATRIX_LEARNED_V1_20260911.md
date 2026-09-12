# 统一实验矩阵（learnable_recency_v1 主线，2026-09-11 起）

## 设计（用户 2026-09-11 裁定）

每数据集官方四格：**full + learned** / **act-only** / **static** / **flat**。宽度（P16/P32）、seed、epoch 等是格子内调参，只写本地列，不进官方表、不进论文表头。**fixed-recency 臂取消**（历史参照数字沿用）。static 为用户新建的测试（非 norm；norm 已废）。flat 训练由用户后续安排。

载体口径（每数据集最后落盘代）：
- M1 = `muscle_response16_svd4/global_rms`（carrier_official4 包；官方锚 582205=0.6260）。旧 rSyn3-refit-v1 线（0.7047/0.7075/0.6949）作废留档。
- M2 = MOVE-T4（dual-track 冻结缓存 T.npy，官方线同源，缓存哈希互锁）。
- H1 = signed_state14 27-tag banks（2026-09-09 封印，官方 582196 同源）。

## EvalAI 队伍（同一组，三人各注册一队）

挑战 2319 / phase 4599。下面三队都是我们：组里三个人各自注册了账号，不是外来对照队。看提交记录时按同一组汇总，不要当成三个独立实验室。

| 队名 | team_id | 说明 |
|---|---|---|
| HKU-ECE | 41975 | 当前默认提交队 |
| falcon_m | 41817 | 同组第二账号 |
| sustechhku | 42279 | 同组第三账号；2026-09-11 交 582356（H1 static-RIFT + diag-z，账号排队对照），2026-09-12 用户撤回 |

日额 / 并发按**各队各自**计算（每队每天 6、并发 3）。官方格回填时写清提交号所属队名。

## 矩阵（每格两数：官方提交（若有）/ 本地选择）

### M2（本地面 = EXT6 earliest-max equal_session_mean，EMA）

| 格 | 官方 | 本地 |
|---|---|---|
| full + learned | — | **0.3634**（s42 e10 P16）；s43 0.3668（e11） |
| act-only | — | **0.1194**（clean e10）⚠️ 设计待核（另一 agent 复核中，暂缓采信） |
| static | — | **0.2681**（末轮EMA e24 sidecar；earliest-max 扫轮待跑） |
| flat | **582292 = 0.3165** | **0.3611**（s42 e7，EXT6 all24 earliest-max） |

旁注（不填格）：clean-ACT 阶梯中间点 ACT-token（E0 side 含 MOVE-T4）= 0.3595 → **E0 内标签 side 贡献 +0.240**；NORM = −0.0013（已废）；concat 接口 learned = 0.3553、concat fixed-recency 参照 = 0.3901；官方 concat 线 581973=0.3901 / 582217(ACT,旧配方)=0.0930。

### M1（本地面 = HO3 earliest-max equal_session_mean，EMA）

| 格 | 官方 | 本地 |
|---|---|---|
| full + learned | **582308 = 0.6411** | **0.7078**（s42 e3；HO3 all24；ch-var 0.6208）。s43 e3 ch-var 0.5820 / legacy 0.6773，未交 |
| act-only | — | —（暂停待核） |
| static | — | **−0.1490**（末轮EMA e24 sidecar；earliest-max 扫轮待跑） |
| flat | —（582318/582314 cancelled，无官方分） | **0.7050**（s42 e2；HO3 all24；ch-var 0.61705） |

旁注：官方锚 582205 = 0.6260（concat + fixed-recency + muscle 载体，不填新格）；旧 rSyn3 线作废（learned P16 0.7075 / P32 0.6949 / 参照 0.7047）。

### H1（本地面 = HO-M3 grouped-seven earliest-max，EMA）

| 格 | 官方 | 本地 |
|---|---|---|
| full + learned | **582277 = 0.4659** | **0.4624**（s42 e16 P16）；s43 0.4516；P32 e15 0.4797 |
| act-only | — | **0.2659**（e14）⚠️ 设计待核（同上） |
| static | **582290 = 0.3135**（末轮 e32，非全扫描） | **0.2090**（末轮 sidecar；HO-M3 all32 扫轮待跑） |
| flat | —（582319/582315 cancelled，无官方分） | **0.4658**（s42 e15，HO-M3 all32 earliest-max；worst S7 0.2241）。历史参照 0.4686 不填本格 |

旁注：官方锚 582196（signed_state fixed-recency，本地对应 0.4677）；官方 ACT 582218=0.3446（旧配方）。

## 维护规则

- 官方表只保留四格与提交号+官方分。P、seed、epoch、接口宽度等只写本地列。
- **此后 EvalAI 选点必须与 FULL 同一套完整扫描**：H1 扫 epochs 1–32 的 HO-M3 grouped-seven earliest-max；M2 扫 1–24 的 EXT6 equal_session_mean earliest-max；M1 扫 1–24 的 HO3 equal_session_mean earliest-max。禁止末轮预注册、单 ckpt 评分、source-minival 选轮。提交包必须指向该扫描收据里的 `selected_epoch`。
- 每臂落袋即更新对应格；官方提交由用户执行后回填（填提交号+分数）。
- 本地数字必须与 run 目录 + selection receipt 一一对应，判定文字（非劣/差）写在旁注。
- 载体/E0/阶梯任何变更必须同步更新"载体口径"节。

## 补充（2026-09-11）

- **static 已建成**（STATIC_SOURCE_ONLY.md）：source-only 共享 `static_identity[N,16]` 通道表（无 E0/无 carrier/无目标校准），脚本 `m1/m2/h1_static_train.py`。**选轮与 FULL 相同**：各任务目标面 earliest-max（H1 HO-M3 all32；M2 EXT6 all24；M1 HO3 all24）。末轮 EMA 只作 sidecar 对照，不锁定提交轮。
- **多 seed 政策（用户 2026-09-11 二次细化）**：**仅 learned+full 允许多 seed 训练并本地选最优提交官方**；其余格（learned+act、learned+static、no-recency+full）一律 seed42 单 seed。矩阵本地列按 seed 分记。
- **选轮政策（用户 2026-09-11 三次裁定，四次确认）**：四格一律与 FULL 相同的完整目标面扫描 + earliest-max。已交例外仅 **582290**（H1 static 末轮 e32，官方 0.3135）；扫描完成后若选点不是 32，另打包装交替换该格。full + learned 的 582277 已是 H1 all32 扫描（e15），合规。

## 补充（2026-09-11 二）：flat 线已建成并接入

FULL_FLAT_CONTROL.md（PREPARED_NOT_TRAINED，17 测试）：flat = 全零 slope（`--tier fixed --half-lives none×8`），共享参数与 learned 主线同 seed 逐字节一致，选轮规则与各任务 FULL 相同（development best-epoch）。**M1 flat 改道肌肉 runner**（清单原命令绑旧 rSyn3 线，已按载体纠正替换为 `m1_projadd_muscle_train.py`，flat 参数组合 CPU smoke 已验证：0 新参、全 None 半衰期）。M2/H1 flat 沿用清单原命令。目标目录 m1_projadd_flat_p16_s42 / m2_projadd_flat_p16_s42 / h1_flat_p16_s42；历史 H1 flat 0.4686 保留为参考，不作为本次新格数字（重训后覆盖）。


## 冻结记录（2026-09-11 三）：ACT 系列暂停

用户指令：ACT 系列设计可能有问题，另一 agent 复核中。M1 ACT 未及启动已撤；M2（0.1194/0.3595）与 H1（0.2659）既有数字保留但标注待核，暂缓采信。static/full/flat 系列不受影响。


## 本地矩阵收官（2026-09-11 08:58 UTC）

三任务 flat 全部落袋：recency（学习或固定）相对 flat 的贡献在三个数据集上均 ≈0（M2 +0.0023 / M1 +0.0028 / H1 −0.0034）。全部本地臂完成，双卡空闲；余项：M1/H1 ACT v2（门控）、官方提交（用户）。

## DANDI000688 SUA／PMUA 配对设计（2026-09-11）

完整设计见 [DANDI000688 v2：SUA／PMUA 配对实验设计](../../docs/DESIGN_DANDI688_SUA_PMUA_V2_20260911.md)。这是独立本地协议：仅用全部 30 个 2015 session，18 个 source-train、6 个 development、6 个 final；不属于 FALCON 矩阵，不以旧 688 结果填入新格。

主矩阵是 `3 × 2`：`full + learned`、`ACT-v2 + learned`、`raw-set source-only + learned` 各在 SUA 与 PMUA 上单独训练和评分。当前为 **IMPLEMENTED / CPU_SMOKE_PASSED / FORMAL_NOT_RUN**；主读数是固定本地 session/endpoint、校准预算和选择规则下的配对 `SUA − PMUA` 差异。

简单对照包括两表示的 `WF-FSS`，仅 PMUA 的 WF-H0、diag-WF、CORAL-WF、AlignedFA-all/stable-WF，以及同一 Raw-PMUA checkpoint 的 diag/CORAL 输入校准，共 15 个评分格子。七个拟合 baseline 正式网格共 144 候选；WF-FSS 使用 dense velocity，Full 使用 trial angle，不能称等标签信息。NoMAD 仅作外部发表参考，不填本地表。

carrier 已核对本地最新完成的 M2 RIFT 官方提交 **582292**：正式为 **MOVE-T4 余弦 OLS**，输出 `[a,c,hypot(a,c),baseline]`。DANDI 使用 M33 neural/方向支持、GO 后 `[0.1,0.6)` 秒计数除以 25 native bins，Q50 评分；SUA/PMUA 各自重新预训练仅含 18 source 的 encoder。网络复用 M2 learned `P16 / proj_add`，24×3165 updates。缓存、代码和 smoke 证据见 [执行入口](../../dandi688_bench_v2/README.md)；未运行正式训练或 final。


## 矩阵终态（2026-09-11 ~13:00 UTC）

ACT v2 三格齐：M1 0.6642（强，距 FULL −0.044）/ M2 0.0796 / H1 0.1501——联合训练活动主干在 M1（10×1024 稠密支持）有效，在 M2（33×100）/H1（3×1024）弱。全部本地 12 格完成，双卡空闲。

## 裁决（2026-09-11）：ACT 格正式采用 v2

用户确认 learned+act 三格以 **ACT v2（联合训练主干）** 为论文数字：M1 0.6642 / M2 0.0796 / H1 0.1501。理由：无标签主张可在单次训练内完整审计、结构性无 side 通道。v1 数字（M2 0.1194 / H1 0.2659）降为附录参考，不再进主表。矩阵上表即终态。

## 统一身份编码器计划（2026-09-11，经分析修正）

目标：三任务 FULL 的身份编码器统一为裸 pre_pool→trial均值→post_pool 主干（"M2 champion 去 FiLM"），carrier 一律走独立 token 通道。

**M1 不重跑**（用户判定）：B3S 以 side=None 运行时本就是裸主干（v2 同权重测试已证同构），肌肉 carrier 已在 token 通道——现 FULL（0.7078/0.6773）即符合统一架构族；与 ACT v2（联合主干）之差 0.044 主要是冻结预训练 vs 联合主干的来源差（M1 carrier≈0）。

**M2/H1 需重跑**：其现有 FULL 的身份价值分别住在 FiLM side（M2，+0.24）与联合材料化器（H1，+0.20）里，统一后 carrier 仅走 token 通道——"干净 E0+token carrier"组合未测过。预期：M2 区间 0.08–0.36（核心赌注）、H1 区间 0.15–0.46。四臂（M2/H1 × s42/s43）待 activity_full_train.py 交付后发射。

## 方法定名与 M2/H1 重建（2026-09-11，用户裁决）

**统一方法定名 B3S**：裸 pre_pool→trial 均值→post_pool 主干（M1 方法族）——冻结预训练权重、活动校准产出 E0、carrier 一律走独立 token 通道。M1 现行 FULL 即此方法（B3S side=None + 肌肉 token carrier，0.7078/0.6773 不变）。

**M2/H1 重建**：FULL = 冻结主干（取自 ACT v2 stage-1 的选中 checkpoint：M2 e20、H1 e14）+ 任务 T token carrier + 全新 decoder（proj_add P16 + learned@default），每任务 s42/s43。原"联合训练主干 FULL"方案作废；activity_full_train.py 改为 B3S 冻结版（trunk-checkpoint 加载）。

## B3S 两阶段预训练裁决（2026-09-11）

复用 ACT v2 checkpoint 作冻结主干被否（目标函数错配：ACT 主干在无 carrier 目标下训练；M2 实测弱主干 0.08 封顶；双层选轮观感差）。改为两阶段：

- **Stage 1（预训练）**：trunk+decoder 在 carrier 在场下联合训练，每任务 s42 一次。
- **Stage 2（B3S 正式臂）**：冻结 stage-1 选中 epoch 的 trunk，重训全新 decoder（+carrier token），M2/H1 各 s42/s43 进矩阵。

与 M1 的"外部程序预训练→冻结→使用"完全同构。总臂：M2 = stage1×1 + stage2×2；H1 = stage1×1 + stage2×2。

## 裁决（2026-09-11）：B3S 裸主干线终止，方法统一到 champion 形态（T4 进编码器）

M2 B3S stage-1（trunk 联合训练 + carrier 仅 token 通道）= EXT6 **0.3130**（e5），比旧 FULL（champion 编码器，T4 经 side 进 E0）0.3634 低 0.050（约 14%，seed 差 5 倍）→ 用户判定为大幅掉分，**立即切回 champion 形态**：

- M2 learned+full = champion 编码器版（0.3668 s43 / 0.3634 s42，已有数字直接沿用）。
- H1 learned+full = C2 联合材料化器版（0.4624/0.4516，同形态沿用；H1 B3S 链已停）。
- M1 = B3S（无 side 模块，生产语义即纯活动）0.7078/0.6773 不变。
- 统一口径 = "EarlyPool 主干（pre_pool→均值→post_pool）± side 条件化（T4/载体进编码器）+ carrier token 通道"；三任务各自的 side 形态为其生产编码器形态。
- B3S 裸主干线保留为**通路消融**：其 stage-1 结果证明 token 通道可承载 MOVE-T4 价值（0.0796→0.3130，+0.233），FiLM-into-E0 通路额外 +0.05（M2 实测）。M2 stage-1 run 与 selection 收据保留为证据。

## Concat 路线执行计划（2026-09-11，用户指令）

ACT-only 与 static 不重跑。**门：M2 concat stage-1 非劣**（vs 旧 FULL 0.3634，均值≥0.3534 且 worst≥0.1389）。过线后：M2 concat full 双 seed、M1 concat full 双 seed（协议=stage-1 预训练 s42 → 冻结 trunk → stage-2 各 seed，沿用 M1 冻结编码器先例）、M2/M1 flat 各单 seed（同冻结 trunk + `--tier fixed --half-lives none×8`）。H1 全部不变。

## Concat 两阶段路线终版（2026-09-11 22:35 UTC，全部本地臂完成）

方法：`E0 = post_pool(cat(pre_pool(activity).trial均值 [N,H], carrier [N,4]))`（H1 生产形态，"T4 一起过 MLP"）；协议 = stage-1 联合预训练（s42）→ 选轮冻结 trunk → stage-2 重训 decoder；carrier 同时走 token 通道。矩阵臂一律 stage-2；flat 全部 stage-2 单 seed；learned+full 双 seed 选优。

| 格（本地） | M2（EXT6） | M1（HO3） | H1（HO-M3，原样） |
|---|---|---|---|
| learned + full | **0.3832**（s42）/ 0.3681（s43），超 champion 0.3634 +0.020 | **0.7124**（s42）/ 0.7075（s43），超 B3S 0.7078 | 0.4624 / 0.4516 |
| no-recency + full (flat) | 0.3622 | **0.7195**（全系新高） | 0.4658 |
| learned + act (v2) | 0.0796 | 0.6642 | 0.1501 |
| learned + static | 0.2681 | −0.1490 | 0.2090 |

## 论文主表草稿（2026-09-12，两位小数）

M1/M2 的 learned+full 先用统一 concat 之前的 RIFT learned 官方分；encoder 接口尚未统一，concat 新分出了再换。flat 整行暂空。† = 本地占位。

| 方法 | M1 | M2 | H1 |
|---|---:|---:|---:|
| learned + full | 0.64 | 0.34 | 0.46 |
| learned + act | 0.66† | 0.08† | 0.15† |
| learned + static | −0.15† | 0.27† | 0.31 |
| no-recency + full (flat) | — | — | — |

来源：M1 learned = 582308 official 0.6411（muscle proj_add，未统一 concat）；M2 learned = 582240 official 0.3372（champion/proj_add P16 s42 e10；s43 582244=0.3255 不取）；H1 learned = 582241 official 0.4588。ACT v2 本地 0.6642 / 0.0796 / 0.1501。M1/M2 static 本地 −0.1490 / 0.2681。H1 static = 582290 official 0.3135。

**协议效应（出处对照，双任务复证）**：stage-1 联合 vs stage-2 冻结 = M2 0.3414→0.3832（+0.042）、M1 0.6774→0.7124（+0.035；s43 0.6584→0.7075 +0.049）——trunk-decoder 共训共适应有害，预训练-冻结-重训一致更优。

**recency 结论（concat 家族内）**：M2 +0.021（首次显正，单 seed flat）、M1 −0.007（flat 0.7195 新高）、H1 −0.003——总体仍≈零，M2 的正贡献待 flat 多 seed 确认。

旧线留档：M2 champion FULL 0.3634/0.3668（FiLM side，被 concat 超越）；M1 B3S 0.7078/0.6773（被 concat 追平/超越）；B3S 裸干 M2 0.3130（通路消融）；M2 concat flat stage-1 0.3541（旧协议留档）。
