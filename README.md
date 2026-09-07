# SPINT Session-Adaptive Neural Decoding Workspace / 会话自适应神经解码工作区

## 项目概览 / Project Overview

本仓库研究 `session-adaptive intracortical neural decoding`。当前统一研究实现位于
[`btransform_unified_v1/`](btransform_unified_v1/)，以 B-transformer 的因果 neural window、局部卷积、
slot 聚合、时序 transformer 与 session-static identity 接口为共同骨架。各任务仍有不同的数据几何、
校准物和评分合同；不能将某一任务的结果、数据暴露或部署验证外推到另一任务。

This repository develops session-adaptive intracortical neural decoding. Its current unified research implementation
lives in [`btransform_unified_v1/`](btransform_unified_v1/) and uses a B-transformer backbone: a causal neural
window, local convolution, slot aggregation, temporal transformer, and a session-static identity interface. Dataset
geometry, calibration material, and evaluation contracts remain task-specific; evidence from one task does not
transfer automatically to another.

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

请从统一实现和当前交接文档开始，而不是从历史分数或旧工单反推当前配方：

Start with the unified implementation and the current handoffs, rather than inferring a live recipe from historical
scores or work orders:

1. [`btransform_unified_v1/docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md`](btransform_unified_v1/docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md)
   — 统一 B-transformer 接口、任务差异和历史比较的治理边界 / unified interface, task differences, and comparison governance.
2. [`btransform_unified_v1/docs/HANDOFF_DANDI688_BTRANSFORM_TEAMMATE_20260907.md`](btransform_unified_v1/docs/HANDOFF_DANDI688_BTRANSFORM_TEAMMATE_20260907.md)
   — DANDI 000688 的当前研究入口；固定 B-transformer 与 M2-like T4 计算定义，开放训练、预算、划分与 PMUA 对照设计 / current 688 research handoff.
3. [`btransform_unified_v1/docs/superpowers/specs/2026-09-07-h1-m2-ort-next-push-design.md`](btransform_unified_v1/docs/superpowers/specs/2026-09-07-h1-m2-ort-next-push-design.md)
   — H1/M2 ORT CPU deployment design and per-task FP32-equivalence gates; it is not a blanket completion claim.
4. [`tfpd_exploration/docs/HANDOFF_EXPERIMENT_CLOSEOUT_20260906.md`](tfpd_exploration/docs/HANDOFF_EXPERIMENT_CLOSEOUT_20260906.md)
   — 已关闭/收尾中的 historical decoder record / historical decoder close-out.

过时文档（不可授权新实验、不可单独引用分数）：

Outdated documents (do not authorize new experiments or stand alone as score sources):

- [`docs_archive/`](docs_archive/) — 早期分析、CPU brief、根目录 dump
- [`tfpd_exploration/docs/outdated/`](tfpd_exploration/docs/outdated/) — 2026-09-04 及更早的 tfpd 支线
- [`SPINT-main/docs/outdated/`](SPINT-main/docs/outdated/) — H1 CarrierID 时期协议
- `sua_exploration/docs/` — 已关闭的 T4 / functional-carrier 程序；保留作历史主线，不是下一架构

根 README 不记录实验分数。

The root README intentionally contains no experiment scores.

## 当前算法主线 / Current Algorithm Mainline

当前主线是 `btransform_unified_v1` 的统一 `proj_add` identity 接口与因果 B-transformer 骨架，而不是把
历史 SPINT、C2、T4 或单一 benchmark 的赢家称为全局“最强”。H1、M1、M2 的选择、训练和部署证据必须在各自
冻结的评分面与 receipt 上读取；历史分数仅作带来源、暴露和训练集范围说明的比较材料。

DANDI 000688 是独立且开放的研究线。其优先问题包括在固定的 M2-like 四维 T4 定义下检验 B-transformer
与 calibration information 的效用。**PMUA 是负责人重点推荐、应尽早开展的高优先级主线**：它必须作为有明确
表示差异和信息预算的对照开展，且不依赖先取得 SUA 正结果；也不能把 SUA 结论无条件迁移到 pseudo-MUA/native MUA。详见 [688 交接指南](btransform_unified_v1/docs/HANDOFF_DANDI688_BTRANSFORM_TEAMMATE_20260907.md)
和 [cost-aware design](btransform_unified_v1/docs/DESIGN_688_BTRANSFORM_COST_AWARE_V1_20260907.md)。

The current mainline is the unified `proj_add` identity interface and causal B-transformer backbone in
`btransform_unified_v1`, not a claim that any historical SPINT, C2, T4, or single-benchmark winner is globally best.
H1, M1, and M2 selection, training, and deployment evidence must be read from their respective frozen scoring
surfaces and receipts. Historical scores are comparison material only when their source, exposure, and training-set
scope are preserved.

DANDI 000688 is a separate, open research line. It tests B-transformer and calibration-information utility under a
fixed M2-like four-dimensional T4 definition, with **PMUA** treated as an explicit representation- and
information-budget-matched comparator rather than an automatic transfer of SUA conclusions.

## E-ORT CPU deployment / E-ORT CPU 部署

**E-ORT** names an inference-implementation optimization, not a new method contribution: **fast exact-E** first
removes only provably redundant computation while retaining causal boundary/state semantics, then **ONNX export**
packages that operator, and **ONNX Runtime CPU** executes the exported graphs. Each dataset must close its own
FP32-equivalence and runtime gate; passing M1 does not certify H1 or M2. The current H1/M2 plan explicitly forbids
automatic submission and records task-specific geometry, references, and acceptance checks in the
[ORT next-push design](btransform_unified_v1/docs/superpowers/specs/2026-09-07-h1-m2-ort-next-push-design.md).

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
| [`btransform_unified_v1/docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md`](btransform_unified_v1/docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md) | 统一模型、任务接口与比较治理 | unified model, task interfaces, and comparison governance |
| [`btransform_unified_v1/docs/HANDOFF_DANDI688_BTRANSFORM_TEAMMATE_20260907.md`](btransform_unified_v1/docs/HANDOFF_DANDI688_BTRANSFORM_TEAMMATE_20260907.md) | 688 当前开放研究与 PMUA 对照边界 | current open 688 research and PMUA-comparator boundaries |
| [`btransform_unified_v1/docs/DESIGN_688_BTRANSFORM_COST_AWARE_V1_20260907.md`](btransform_unified_v1/docs/DESIGN_688_BTRANSFORM_COST_AWARE_V1_20260907.md) | 688 cost-aware 起点和可解释对照 | cost-aware 688 starting point and interpretable controls |
| [`btransform_unified_v1/docs/superpowers/specs/2026-09-07-h1-m2-ort-next-push-design.md`](btransform_unified_v1/docs/superpowers/specs/2026-09-07-h1-m2-ort-next-push-design.md) | H1/M2 E-ORT CPU 等价门与提交边界 | H1/M2 E-ORT CPU equivalence gates and submission boundary |
| [`tfpd_exploration/docs/HANDOFF_EXPERIMENT_CLOSEOUT_20260906.md`](tfpd_exploration/docs/HANDOFF_EXPERIMENT_CLOSEOUT_20260906.md) | 历史 decoder 收尾与 receipt 指针 | historical decoder close-out and receipt pointers |

Protocol 定义实验语义；terminal receipt 证明实验按合同完成；`CURRENT_RESULTS.md` 记录接受的结果；当前
handoff 决定这些结果能在论文中支持什么。

A protocol defines experiment meaning; a terminal receipt proves execution under that contract;
`CURRENT_RESULTS.md` records the accepted result; the current handoff determines what the result can support in the
paper.

## 仓库结构 / Repository Map

| path | 中文定位 | English role |
|---|---|---|
| `tfpd_exploration/` | 当前 decoder / B-transformer 实验与官方提交包 | current decoder / B-transformer experiments and official packs |
| `btransform_unified_v1/` | 当前统一 B-transformer、跨任务研究设计与 deployment 入口 | current unified B-transformer, cross-task research designs, and deployment entry points |
| `sua_exploration/` | 历史 SUA / T4 研究、688 数据管线与可复用对照材料 | historical SUA/T4 research, 688 data plumbing, and reusable comparator material |
| `SPINT-main/` | SPINT baseline、官方数据与 source-model semantics | SPINT baseline, official data, and source-model semantics |
| `streaming_calibration_exp/` | 可复用 streaming-calibration 框架 | reusable streaming-calibration framework |
| `docs_archive/` | 过时文档归档 | outdated document archive |
| `bci_paper_overleaf/` | 独立 Overleaf Git repository 中的 paper source；不随外层仓库提交 | paper source in a separate Overleaf Git repository; never commit it through the outer repository |
| `software-to-hardware/` | deferred implementation notes | deferred implementation notes |
| `planB_tempconv/` | historical low-cost temporal-decoder branch | historical low-cost temporal-decoder branch |

Operator setup、historical archives、RTL handoffs 和 exploratory hardware workspaces 可以保留在本地，但由
`.gitignore` 排除，不属于当前 algorithm repository 或 reproducibility surface。

Operator setup, historical archives, RTL handoffs, and exploratory hardware workspaces may remain local, but are
excluded by `.gitignore` and are not part of the current algorithm repository or reproducibility surface.

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

本仓库包含多个同名 `src` package；从 repository root 一次性收集所有 tests 会造成 module
shadowing。请从对应 subtree 运行 focused tests，并禁用无关的外部 pytest plugin：

This repository contains sibling packages named `src`; collecting every test from the repository
root causes module shadowing. Run focused tests from the owning subtree and disable unrelated
third-party pytest plugins:

```bash
cd sua_exploration
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.. \
  /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest -q tests/<focused_test.py>

cd ../streaming_calibration_exp
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.. \
  /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest -q tests/<focused_test.py>
```

这是一项已知的 repository-layout 限制；root-level blanket `pytest` 目前不是受支持的验证入口。

This is a known repository-layout limitation; a blanket root-level `pytest` invocation is not a
supported verification entry point.

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
