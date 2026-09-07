# M2 dual-track shuffled 12-epoch review and bounded 24-epoch continuation

Date: 2026-09-05
Status: **REVIEW_COMPLETE__RECOMMEND_PAIRED_24EP_CONTINUATION__NO_JOB_LAUNCHED_BY_REVIEWER**
Reviewer: Astra / root
Scope: read-only review plus this new document; no training, new result root, historical rewrite, or EvalAI submission.

Parent: [parallel calibration-memory / temporal-decoder work order](WORKORDER_CALIBRATION_MEMORY_AND_TEMPORAL_DECODER_PARALLEL_V1_20260905.md).
Evidence root: `tfpd_exploration/results/m2_dual_track_v1/20260905_101500/`.
This is an additive decision record, not an amendment of the original 12-epoch evidence.

## 1. Decisions

1. Preserve the completed 12-epoch findings. B-TRANSFORMER is near REF but below the routing threshold; B-MAMBA has a material transfer deficit. Neither completed run becomes INCOMPLETE merely because more optimization might help.
2. Recommend **one paired seed-42 continuation of both shuffled B arms from epoch 12 to epoch 24**, under the parent's already-declared cosine tail. Both pass the actual source-validation-loss continuation gate, not merely a training-loss surrogate.
3. Do not allocate B seed43 yet. Resolve the declared optimization extension before spending a second seed on the current +0.001768 point estimate.
4. Keep A as a modest candidate, not a demonstrated memory mechanism or second innovation. Preserve source-picked and visible-development-picked views separately.
5. Require a concise, linked audit package. `comparison.csv` is the summary, not the whole evidence package.
6. No new B topology, LR sweep, 48-epoch extension, A+B combination, or EvalAI follows automatically.

This document records the recommended next action for the execution agent; its creation itself executes nothing. Normal local work remains under the user's existing permissions; do not introduce repeated permission prompts.

## 2. Verified 12-epoch scoreboard

All numbers below use the same M33-disjoint clean ext-4 surface and REF:

`R_REF_session = 0.3582396424`, `R_REF_date = 0.3279795072`.

| Arm | Seed / sampler | Source-picked epoch | Ext-4 session mean | Delta vs REF | Ext-4 date mean |
|---|---|---:|---:|---:|---:|
| A-QMEM | 42 / original | 8 | 0.3634059245 | +0.0051662821 | 0.3313483408 |
| A-QMEM | 43 / original | 5 | 0.3655350000 | +0.0072953576 | 0.3272158295 |
| B-TRANSFORMER | 42 / shuffled | 9 | 0.3600075106 | +0.0017678682 | 0.3302380000 |
| B-MAMBA | 42 / shuffled | 12 | 0.3021181567 | -0.0561214858 | 0.2672630000 |
| B-MAMBA | 42 / sequential, historical attempt | 3 | 0.1544415950 | -0.2037980474 | 0.1470163906 |

Important endpoints:

- B-TRANSFORMER epoch12: `0.2719287282`, delta `-0.0863109142`.
- B-MAMBA shuffled selected=epoch12; do not count the duplicate score as replication.
- Shuffled versus sequential Mamba selected scores differ by `+0.1476765617`. This is a substantial recipe correction, not a controlled estimate of the sampler alone: epoch choices differ and the corrected run has its own implementation/trajectory.
- At the selected endpoints B-MAMBA is `-0.0578893539` below B-TRANSFORMER. This is evidence against the current 12-epoch Mamba recipe, not a theorem about SSM capacity.
- Transformer endpoint deterioration does not invalidate legitimate source epoch-picking. It does expose optimization/generalization sensitivity and makes the declared decay test informative.

The original source-selected view is retained for the fixed analysis. The already-completed A visible-ext4 scan additionally reports:

| Arm | Visible-ext4-picked epoch | Ext-4 session mean | Delta vs REF |
|---|---:|---:|---:|
| A-QMEM seed42 | 4 | 0.3724013640 | +0.0141617216 |
| A-QMEM seed43 | 2 | 0.3746668013 | +0.0164271589 |

These are usable visible-development product-selection numbers, not fresh-test evidence. Never compare visible-picked A with source-picked B and attribute the difference solely to architecture. Reporting a legal selection surface does not make it statistically independent of selection.

## 3. The actual extension gate passes

Parent §4.3 requires source-minival mean loss over epochs9–12 to decrease by at least 3% relative to epochs5–8 for at least one B arm, plus the time-budget condition. **Training loss is not the gate.**

Read directly from each shuffled arm's `metrics.jsonl`, using only `event == "epoch"` and the stored `minival.native_mse`:

| Arm | Mean validation MSE e5–8 | Mean validation MSE e9–12 | Relative reduction | Train-MSE reduction, descriptive |
|---|---:|---:|---:|---:|
| B-MAMBA | 0.0000907256740642 | 0.0000821163618769 | **9.489389%** | 34.038832% |
| B-TRANSFORMER | 0.0000970754563241 | 0.0000869494234783 | **10.431095%** | 19.836260% |

Reduction is `1 - mean_late / mean_early`. The stored native MSE is aggregated by the current scorer; keep that aggregation, rather than substituting a differently weighted `1 - mean(R2)` after observing the curves.

Measured epoch time (includes source-minival, before checkpoint overhead): Mamba `69.0225 s`, Transformer `69.4695 s`. Twelve additional epochs are about `13.80 / 13.89 min` per arm, roughly `27.70 GPU-min` in total, before resume verification, writes, and external scoring. Two independently leased GPUs can overlap this work. This is an estimate, not a guaranteed completion time; check the current ledger and remaining budget first.

## 4. Resume is feasible, but the current CLI does not implement it

The reviewer loaded **the actual two epoch_012.pt files on CPU**. Both contain:

- schema `m2_dual_track_v1_full_ckpt_v1`;
- `epoch_one_based = 12`, `global_step = 37980`;
- model state, optimizer state, Python/NumPy/Torch/CUDA RNG;
- sampler completed epoch12 and manifest digest `0c1808baa4d6722742a3705963c60b55fd7fcc7f3b2d03c2c416e3e0d2937a95`.

Mamba file: 103,420,282 bytes, 69 optimizer parameter states.
Transformer file: 124,950,074 bytes, 77 optimizer parameter states.

However, the inspected `launch.py::_train_b_loop` rebuilds the model/optimizer, sets `global_step=0`, and loops from epoch1. `_restore_rng` exists but has no call site. The CLI has `--epochs` but no `--resume` argument. **Do not run `train --epochs 24` and call it continuation.**

Minimum implementation prerequisite, owned by the existing execution coordinator/shared-training owner:

1. Add an explicit resume source. Resume from each arm's **epoch12**, not its selected epoch. Preserve the original root/metrics/summaries; put continuation artifacts in a new named child/successor run with parent hashes.
2. Restore model, AdamW moments/steps, global step, all active RNG domains, and sampler position. Rebuild the deterministic LR law from its bound config; retain dynamics/no-decay parameter grouping. Validate optimizer parameter-name correspondence, not only number of groups.
3. If the old sampler manifest contains only12 epochs, generate one shared24-epoch manifest with the same deterministic generator and prove its first12 epoch batch lists exactly equal the frozen original. Store parent and new digests; never overwrite the old manifest or falsely require its whole-file digest to remain equal after extending it.
4. Begin at epoch13/global_step37980. Use the existing declared epoch13–24 cosine function, ending at `0.1 * 3e-4 = 3e-5`; do not invent a new schedule, rerun warmup, reset Adam, or average weights in the same cell.
5. Prove interrupted-versus-uninterrupted continuation parity on a disposable source-only/synthetic instance, including the next sampled batch, dropout behavior, optimizer step, and LR. Test both architectures. Bind a numerical contract appropriate to the existing BF16 kernels; do not claim cross-GPU bitwise parity without checking it.
6. Disposable smoke/profile work must not advance formal optimizer/RNG state. Preserve max-epoch reporting and best/source-picked history from epochs1–12 when selecting over1–24.

A failure to implement or verify resume within a bounded engineering attempt is `EXTENSION_ENGINEERING_BLOCKED`; it does not erase the completed12-epoch verdict. Do not silently substitute a fresh24-epoch run. Bound initial resume implementation/debug to two hours before reporting the concrete issue.

## 5. Selection and stopping at24

Record three non-interchangeable views:

1. Original12: source-pick from1–12 and endpoint12, immutable.
2. Extended24: source-pick from1–24 using the original tie rule, and endpoint24. Score on the same ext4 windows; retain12-vs24 exposure differences.
3. Visible-development product view: after the24 trajectory is complete, a deterministic scan may select over the declared1–24 set on visible ext4. Report candidate count, all scanned scores, tie rule, per-session/date results, and the selection label. Do not feed these scores into training or an epoch25 decision. If comparing to A's12-epoch scan, explicitly disclose different candidate/exposure budgets.

Apply the parent's +0.005 / per-session safety routing rule to the extended **source-selected** view before assigning a replication seed. The visible-product view is supplemental; it is not a retrospective rewrite of that rule.

- Neither B passes: close this B recipe after24; retain A and the B-Transformer comparator. No automatic new seed or topology.
- Transformer passes and Mamba does not: prioritize Transformer seed43 for performance, make no SSM-positive claim.
- Mamba passes: compare to the matched Transformer as well as REF. An SSM-specific follow-up needs a benefit over Transformer, not merely over REF.
- A visible-only winner may be nominated for product consideration with its selection disclosure, but does not automatically trigger new training or submission.

These are resource-routing decisions, not non-inferiority tests or population-level significance. Four sessions are three dates, and the two runs on one date are correlated. Keep session-equal and date-equal results alongside each other.

## 6. The short audit package

The existing CSV is enough to communicate status, not enough to audit it. The execution agent should provide one short `HANDOFF_FOR_ASTRA_REVIEW.md` linking, rather than duplicating:

1. code/config/data/checkpoint hashes, including the exact sampler correction and all attempted runs;
2. source/ext4 separation and the same-surface REF receipt;
3. per-epoch validation loss/R2, per-session/date selected and endpoint scores;
4. this gate calculation and the complete-state/resume-parity receipt;
5. wall time, GPU minutes, memory, job ledger, and actual12/24 exposure;
6. separate source-selection and visible-development-selection manifests;
7. completed, deferred, and forbidden actions, plus the proposed final disposition.

Do not require a large new ablation matrix before finishing this bounded optimization test. Do not mark current per-arm `summary.json` fields such as `ext4_scored=false` as the authoritative scoring status when a separate `ext4_score.json` already exists; the handoff should identify the scoring receipt and reconcile that stale metadata additively.

## 7. Reviewer evidence inventory

- `comparison.csv` under the evidence root.
- `arms/B-MAMBA/seed42_shuffled/{metrics.jsonl,summary.json,epoch_012.pt}`.
- `arms/B-TRANSFORMER/seed42_shuffled/{metrics.jsonl,summary.json,epoch_012.pt}`.
- `tfpd_exploration/src/m2_dual_track_v1/{launch.py,training.py,sampler.py}`.
- `tfpd_exploration/scripts/run_m2_dual_track_v1.py`.
- Parent work order §2.3, §4.3, §6, §10, §12.

This is a focused continuation review, not a fresh full re-audit of every model/data operator. Earlier completed structural checks are referenced, not claimed to have been rerun here.
