# Task Plan — 历史 ASIC 审计

> 本文件记录较早的 few-shot ASIC 分析任务，不是当前项目总计划。当前主线计划见 [`sua_exploration/ROADMAP.md`](../sua_exploration/ROADMAP.md)。

## Goal
Analyze the repository's few-shot learning path at tensor/operator level and assess a research ASIC deployment architecture with explicit compute, memory, bandwidth, precision, and validation requirements.

## Phases
- [complete] Phase 1: Inventory repository, papers, existing analysis, and identify the exact few-shot implementation.
- [complete] Phase 2: Trace training and inference code paths, tensor shapes, losses, and episodic data movement.
- [in_progress] Phase 3: Derive parameter, MAC, activation, storage, and bandwidth formulas plus representative numerical cases.
- [pending] Phase 4: Map the workload to a practical ASIC partition and identify algorithm/hardware co-design changes.
- [pending] Phase 5: Write and verify the detailed Chinese report with source references and explicit assumptions.
- [complete] Phase 6: Synchronize the audited QAT-B LOSO0 result into `software-to-hardware/` documentation.
- [complete] Phase 7: Define the frozen software/hardware contract, parallel-work boundaries, and remaining sign-off gates.
- [complete] Phase 8: Verify documentation consistency against reports, checkpoints, and executable validation paths.

## Key Decisions
- Treat repository code as the implementation source of truth and papers as intent/context.
- Separate deployment inference, on-chip few-shot adaptation, and full training because their hardware costs differ substantially.
- Label every estimate as exact, formula-based, measured, or assumed.
- Treat QAT-B epoch 14 as the current LOSO0 hardware-candidate artifact, while keeping fold/seed scope and unresolved bit-exact/control-ablation caveats explicit.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| Unified `exec_command` could not create `/bin/bash` process | 2 | Use the available filesystem Read/Bash MCP tools and `apply_patch` instead. |
| First absolute-path `apply_patch` returned without creating files | 1 | Reapply with workspace-relative paths. |
| Unsafe checkpoint deserialization request rejected | 1 | Switched to `weights_only=True`; the active Python lacks PyTorch, so use source/config derivation and label checkpoint inspection as not run. |
| `apply_patch` later could not see existing workspace files | 2 | Used exact replacement editing after reading the target files.
