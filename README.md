# SPINT Session-Adaptive Neural Decoding Workspace / 会话自适应神经解码工作区

## 项目概览 / Project Overview

本仓库研究 `session-adaptive intracortical neural decoding` 算法。当前主线将
`streaming neural-activity path` 与紧凑的 `analytic functional carrier` 结合：系统从按时间排序的
`calibration prefix` 提取目标 session 的 functional identity，在不执行 target-session
`backpropagation`、不更新 decoder weights 的情况下完成适配。

This repository develops algorithms for session-adaptive intracortical neural decoding. The current mainline
combines a streaming neural-activity path with a compact analytic functional carrier. Functional identity is fitted
from a chronological calibration prefix, while target-session backpropagation and decoder weight updates remain
disabled.

仓库包含 research code、frozen experiment contracts、evidence-verification tools、paper source，以及未来
implementation 的背景材料。Raw datasets、checkpoints、logs、predictions 与 generated result bundles 是本地
artifact，不进入 Git。

The workspace contains research code, frozen experiment contracts, evidence-verification tools, paper sources, and
background material for possible future implementation. Raw datasets, checkpoints, logs, predictions, and generated
result bundles are local artifacts and are intentionally excluded from Git.

当前成果属于纯算法与实验研究。Quantization、RTL、latency、resource utilization 与 hardware fidelity 需要
独立验证，不是当前工作已经完成的 claim。

The current contribution is algorithmic and experimental. Quantization, RTL, latency, resource utilization, and
hardware fidelity require separate validation and are not completed claims of the present work.

## 快速入口 / Start Here

当前唯一权威的 scientific and execution handoff：

The sole authoritative scientific and execution handoff is:

[`sua_exploration/docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`](sua_exploration/docs/HANDOFF_MAINLINE_CLOSURE_20260811.md)

该 handoff 说明 selected method、evidence hierarchy、claim boundaries、remaining closure work、golden
programs 与 immutable receipts 的位置。旧的 dated handoff、review、`AGENT_BRIEF` 和 proposal 只作为
historical provenance，不能覆盖当前 handoff，也不能单独授权新实验。

It defines the selected method, evidence hierarchy, claim boundaries, remaining closure work, golden programs, and
locations of immutable receipts. Older dated handoffs, reviews, agent briefs, and proposals are historical
provenance only; they do not override the current handoff or authorize new experiments.

推荐阅读顺序 / Recommended reading order:

1. [`sua_exploration/docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`](sua_exploration/docs/HANDOFF_MAINLINE_CLOSURE_20260811.md)
   — 当前 scientific status、claim boundaries 与交接说明 / current scientific status, claim boundaries,
   and handoff.
2. [`sua_exploration/README.md`](sua_exploration/README.md)
   — functional-carrier program 的简明入口 / concise entry point for the functional-carrier program.
3. [`sua_exploration/docs/CURRENT_RESULTS.md`](sua_exploration/docs/CURRENT_RESULTS.md)
   — result ledger、evidence status 与 receipt pointers / result ledger, evidence status, and receipt pointers.
4. [`sua_exploration/ROADMAP.md`](sua_exploration/ROADMAP.md)
   — closure sequence、stop rules、cleanup 与 Git checklist / closure sequence, stop rules, cleanup, and Git
   checklist.
5. [`sua_exploration/docs/ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](sua_exploration/docs/ACTIVE_EXPERIMENT_CONTROL_BOARD.md)
   — live process 与 terminalization state / live process and terminalization state.

根 README 不记录实验分数；会变化的数字只写入 `CURRENT_RESULTS.md` 与当前 handoff。

The root README intentionally contains no experiment scores. Changing numerical results belong only in
`CURRENT_RESULTS.md` and the current handoff.

## 当前算法主线 / Current Algorithm Mainline

Selected system 将 calibration 分为两个角色：

The selected system separates calibration into two roles:

- `B3/B3T streaming activity encoder` 表示近期 neural activity；
  the B3/B3T streaming activity encoder represents recent neural activity.
- `analytic functional carrier` 表示 session-specific functional identity；
  the analytic functional carrier represents session-specific functional identity.
- `SPINT-style pretrained decoder` 消费 activity 与 carrier；
  a SPINT-style pretrained decoder consumes both activity and carrier.
- carrier 从 chronological calibration prefix 一次性拟合并缓存；
  the carrier is fitted once from a chronological calibration prefix and then cached.
- target-session inference 不构建 optimizer，不执行 `backward()`，不更新 network weights；
  target-session inference creates no optimizer, executes no backward pass, and updates no network weights.

当前主线研究的是 algorithmic adaptation，而不是硬件实现。未来如果进行 quantization 或 RTL translation，
必须从 selected FP32 contract 与 golden reference programs 出发，并另行验证 numerical fidelity 与 state
semantics。

The present mainline studies algorithmic adaptation rather than hardware implementation. Any future quantization or
RTL translation must start from the selected FP32 contract and golden reference programs, with separate validation
of numerical fidelity and state semantics.

## 核心研究问题 / Research Questions

1. 哪些 session-level functional information 能够跨 SUA、pseudo-MUA 与 native MUA representations 使用？
   Which session-level functional information remains useful across SUA, pseudo-MUA, and native MUA
   representations?
2. `analytic functional carrier` 能在多低的 target-supervision density 下保持有用的 decoding accuracy？
   How low can target-supervision density become while the analytic functional carrier retains useful decoding
   accuracy?
3. Functional carrier content、row attachment、compact consumer 与 streaming activity path 分别贡献什么？
   What are the separate contributions of functional carrier content, row attachment, compact consumer, and the
   streaming activity path?
4. 哪些 algorithm components 可以在未来被简化或高效实现，同时保持 scientific contract？
   Which algorithm components may later be simplified or implemented efficiently without changing the scientific
   contract?

项目不预设 analytic calibration 必然优于 dense direct decoder，也不把 algorithmic supervision count、manual
annotation cost、compute、latency、memory 与 energy 混为同一概念。

The project does not assume that analytic calibration must outperform a dense direct decoder. Algorithmic
supervision count, manual annotation cost, compute, latency, memory, and energy are treated as distinct quantities.

## 重要文档 / Key Documents

| document | 中文用途 | English role |
|---|---|---|
| [`sua_exploration/docs/FP32_T4_MAINLINE_PROTOCOL.md`](sua_exploration/docs/FP32_T4_MAINLINE_PROTOCOL.md) | selected FP32 method 与 calibration contract | selected FP32 method and calibration contract |
| [`sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md`](sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md) | pairing、uncertainty、noise floor 与 result-state 规则 | pairing, uncertainty, noise-floor, and result-state rules |
| [`sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md`](sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md) | 未来 implementation 可参考的 cost vocabulary；不是当前 hardware evidence | optional cost vocabulary for future implementation; not current hardware evidence |
| [`sua_exploration/docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md`](sua_exploration/docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md) | frozen RT sparse-carrier comparison | frozen RT sparse-carrier comparison |
| [`sua_exploration/docs/RT_SPARSE_T4D_VS_B2_D1024_COMPANION_PROTOCOL_20260810.md`](sua_exploration/docs/RT_SPARSE_T4D_VS_B2_D1024_COMPANION_PROTOCOL_20260810.md) | exact-query SPINT-structured companion comparison | exact-query SPINT-structured companion comparison |
| [`sua_exploration/docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md`](sua_exploration/docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md) | native MUA protocol 与 dataset-specific scope | native MUA protocol and dataset-specific scope |
| [`sua_exploration/docs/PSEUDO_MUA_T4_BRIDGE_48H.md`](sua_exploration/docs/PSEUDO_MUA_T4_BRIDGE_48H.md) | SUA-to-pseudo-MUA controlled signal-view bridge | controlled SUA-to-pseudo-MUA signal-view bridge |

Protocol 定义实验语义；terminal receipt 证明实验按合同完成；`CURRENT_RESULTS.md` 记录接受的结果；当前
handoff 决定这些结果能在论文中支持什么。

A protocol defines experiment meaning; a terminal receipt proves execution under that contract;
`CURRENT_RESULTS.md` records the accepted result; the current handoff determines what the result can support in the
paper.

## 仓库结构 / Repository Map

| path | 中文定位 | English role |
|---|---|---|
| `sua_exploration/` | functional-carrier 主线、result ledger、protocol、scripts 与 tests | functional-carrier mainline, result ledger, protocols, scripts, and tests |
| `SPINT-main/` | SPINT baseline、H1 training/evaluation 与 source-model semantics | SPINT baseline, H1 training/evaluation, and source-model semantics |
| `streaming_calibration_exp/` | reusable streaming-calibration 与 MUA experiment framework | reusable streaming-calibration and MUA experiment framework |
| `bci_paper_overleaf/` | 独立 Git repository 中的 paper source | paper source in a separate Git repository |
| `software-to-hardware/` | deferred implementation notes 与 model-export experiments | deferred implementation notes and model-export experiments |
| `encoder_rtl_handoff_v1/` | 未来 RTL work 的 background 与 golden-reference pointers | background and golden-reference pointers for possible future RTL work |
| `rtl_handoff/` | preliminary digital-design material；不属于当前 algorithm claim | preliminary digital-design material outside the current algorithm claim |
| `hardware_pe_sram/` | exploratory processing-element 与 memory notes | exploratory processing-element and memory notes |
| `planB_tempconv/` | historical low-cost temporal-decoder branch | historical low-cost temporal-decoder branch |
| `docs_archive/` | 已移出 active navigation 的历史材料 | historical material removed from active navigation |

## 证据链 / Evidence Pipeline

项目使用 fail-closed evidence chain：

The project uses a fail-closed evidence chain:

```text
frozen protocol
  -> source/config/data manifest
  -> checkpoint-selection receipt
  -> one-shot target evaluation
  -> terminal aggregate
  -> independent verifier
  -> result ledger and paper claim
```

关键边界 / Important boundaries:

- launch receipt 或 preflight PASS 不是 accuracy result；
  a launch receipt or preflight PASS is not an accuracy result.
- partial fold 不是 terminal aggregate；
  a partial fold is not a terminal aggregate.
- held-out system score 不自动等于 matched causal attribution；
  a held-out system score is not automatically a matched causal attribution.
- no target-session backpropagation 不代表没有 offline source training；
  no target-session backpropagation does not mean no offline source training.
- 更少的 algorithmic target values 不自动等于更低的 manual annotation cost 或 energy；
  fewer algorithmic target values do not automatically imply lower manual annotation cost or energy.
- negative result 关闭 tested implementation，而不是整个 method family；
  a negative result closes the tested implementation, not the entire method family.

Generated evidence 保存在被 `.gitignore` 排除的 result 与 pilot-artifact roots 中。即使大型 artifact 不上传
GitHub，论文数字也必须能追溯到 terminal artifact 与 content hash。

Generated evidence is stored under ignored result and pilot-artifact roots. Even though large artifacts are not
pushed to GitHub, paper numbers must remain traceable to terminal artifacts and content hashes.

## 开发与 Git 规则 / Development and Git Hygiene

- 使用 relevant protocol 指定的 environment 与 focused tests；
  use the environment and focused tests named by the relevant protocol.
- 保留 shared multi-agent worktree 中的 unrelated changes；
  preserve unrelated changes in the shared multi-agent worktree.
- 禁止 blanket staging，例如 `git add .`；只 stage reviewed paths；
  never use blanket staging such as `git add .`; stage reviewed paths explicitly.
- 不提交 datasets、checkpoints、predictions、logs、caches 或 generated result bundles；
  do not commit datasets, checkpoints, predictions, logs, caches, or generated result bundles.
- 不删除 active run directories、immutable receipts、terminal aggregates 或 canonical best checkpoints；
  do not delete active run directories, immutable receipts, terminal aggregates, or canonical best checkpoints.
- archive 旧文档前检查 current docs、scripts、tests、receipts 与 paper source 的 basename/stem references；
  before archiving an old document, inspect basename and stem references from current docs, scripts, tests,
  receipts, and paper sources.
- 每次 GitHub push 后核验 remote commit；
  verify the remote commit after every GitHub push.

## 论文与未来实现 / Paper and Future Implementation

Paper source 只消费 result ledger 与 current handoff 中经过 terminal、scope-checked 的 claim。未来如果对 selected
algorithm 进行 quantization 或 RTL translation，应从 FP32 method contract 与 golden reference programs
开始，而不是从 historical experiment logs 重新推导算法。

The paper source consumes only terminal, scope-checked claims from the result ledger and current handoff. Any future
quantization or RTL translation should begin from the FP32 method contract and golden reference programs rather than
re-deriving the algorithm from historical experiment logs.

Future implementation 需要独立验证 numerical fidelity、resource use、latency 与 state semantics；当前算法结果
不隐含这些 hardware claims。

Future implementation requires separate validation of numerical fidelity, resource use, latency, and state
semantics. None of those hardware claims are implied by the present algorithm results.
