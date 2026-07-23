# Progress — 历史 ASIC/MUA 工作日志

> 本文件保留旧阶段工作记录。当前 SUA/MUA 执行计划见 [`sua_exploration/ROADMAP.md`](sua_exploration/ROADMAP.md)。

## 2026-07-10
- Started detailed few-shot computation and data-flow analysis for research ASIC deployment.
- Confirmed workspace inventory and selected a code-first, inference/adaptation/training-separated analysis method.
- Logged an execution-tool startup failure and switched to working filesystem tools.
- Inventoried implementation files and located the canonical few-shot model, data preparation, and decoder code paths.
- Completed repository/model inventory and began exact tensor/operator tracing from the M2 source and configuration.
- Traced calibration preprocessing, causal windowing, session batching, packaged deployment flow, and calculated exact affine/MHA MAC counts.
- Analyzed the Plan-B frozen TCN/CORAL alternative and summarized its six-session few-shot curve.
- Completed end-to-end data-flow tracing; started memory, throughput, and numerical-format derivations.

## 2026-07-14
- Audited the completed 25-epoch QAT-B LOSO0 run and independently re-evaluated its epoch-14 checkpoint.
- Confirmed corrected backward coverage, separate weight/scale optimizer groups, live scale export, low saturation, and exact fake/integer output for the selected artifact.
- Identified scope/caveats to preserve in documentation: one-fold validation selection, shadow stability amber, unmatched fixed-scale control, and transient epoch-9/13 bit-exact failures.
- Began synchronizing the audited result and software/hardware parallel-development contract into `software-to-hardware/`.
- Added `B3_QAT_B_hardware_handoff.md` with the candidate artifact, audited metrics, frozen datapath contract, configurable model fields, remaining risks, release-package schema, and RTL acceptance order.
- Updated the software-to-hardware README, quantization baseline, decisive-experiment history, and EarlyPool network spec to reflect the 2026-07-14 QAT-B result and parallel-development boundary.
- Verified all documented QAT-A/QAT-B/evaluation CLI options against each script's `--help` output.
- Rechecked the frozen epoch-14 JSON metrics and retained the independently generated `eval_best/eval_paths_report.json` as the referenced evidence artifact.
- Re-ran `test_b3_qat_backward.py`: all four layers had finite/nonzero gradient coverage, all layer weights changed after one optimizer step, and exact forward remained aligned (PASS).

## 2026-07-21
- Implemented 8 new encoder variants (B7-B14) exploring {count conditioning, fixed projection, sparse hash, streaming hash, ensemble, ternarization}.
- Discovered all prior streaming_calibration_exp results used wrong protocol (minival instead of LOSO, no heldout evaluation). Re-ran key variants with correct LOSO+heldout protocol.
- Established B3-D64 as the encoder stopping point: heldout R²=0.236 matches SPINT paper's 0.26 ± 0.13 claim using only 1.6% of encoder params.
- Concluded B7-B14 direction: all aggressive encoder simplifications monotonically degrade heldout generalization. Detailed analysis in `streaming_calibration_exp/ARCHITECTURE_EXPLORATION.md`.
- Identified the real bottleneck as the decoder (3.47M params, 82.9M MAC/frame with softmax/LayerNorm), not the encoder.
- Next steps: multi-fold validation, joint training (unfreeze decoder), or decoder compression.
