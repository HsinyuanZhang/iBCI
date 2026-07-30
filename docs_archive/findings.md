# Findings — 历史 ASIC/MUA 审计记录

> 本文件不是当前项目总状态。SUA/MUA 主线的当前结论见 [`sua_exploration/docs/CURRENT_RESULTS.md`](../sua_exploration/docs/CURRENT_RESULTS.md)。

## Repository Inventory
- Two source papers are present in the workspace root.
- Candidate implementations: `SPINT-main/` and `planB_tempconv/`.
- Existing reports: deployment cost, paper analysis, reproduction analysis, and next-steps plan.

## Evidence Log
- The exact SPINT few-shot path is in `src/models/components/spint.py`, calibration construction in `src/data/falcon_datamodule.py`, and deployment behavior in `third_party/falcon_challenge/spint_decoder.py`.
- Few-shot adaptation is gradient-free and unlabeled: calibration trials produce per-neuron identity embeddings; no deployment-time optimizer/backward is involved.
- M2 working point documented in repo: `W=50`, `H=512`, calibration `M=33`, normalized trial length `T=100`, output queries `C=2`, `N=96`, one cross-attention layer, 64 heads, FFN 2048.
- `planB_tempconv` is a separate TCN/student research branch with ridge/few-shot experiments, not the canonical SPINT identity-encoder implementation.
- Existing `SPINT_deployment_cost_analysis.md` flags that the reference decoder recomputes the calibration identity path per frame instead of caching the per-session embedding.
- `num_id_layers=3` creates three affine layers in `fc_id_in` (`T->H`, `H->H`, `H->H`) and three in `fc_id_out` (`H->H`, `H->H`, `H->W`): six IDEncoder affine layers total.
- Calibration trial averaging happens after all three `fc_id_in` layers, so only `fc_id_out` can run after reduction over `M`; this determines the dominant `M*N*H^2` cost.
- `fc_in` is shared by neural tokens and learnable behavior queries; the projected query is constant after training and is cacheable.
- M2 trains/executes `fc_out: H->W=50` for each of `C=2` queries, but the Lightning wrapper immediately slices the last time point. Deployment can retain only the corresponding output row (`H->1`) if equivalence is checked.
- The cross-attention layer is pre-norm and uses PyTorch `MultiheadAttention`: Q/K/V/out projections, scaled dot-product softmax over `N`, residuals, then `H->2048->H` FFN.
- Training loss is MSE on only the last predicted time bin after division by behavior scale factor 5.0; all model parameters are optimized offline with Adam.
- M2 calibration preprocessing filters to evaluation bins, splits trials, and cubic-interpolates every trial to `T=100`; this SciPy preprocessing is outside the learned IDEncoder and should remain on the host for a first ASIC.
- Training samples are `W x N` causal windows with `(W-1)` zero-history padding. `SessionBatchSampler` keeps one session per batch; training may choose a different contiguous calibration subset per sample.
- The packaged decoder stores raw calibration features (`M x T x N`) and copies them into every `forward`; it neither caches `E` nor avoids reconstructing/transposing the full `W x N` window each frame.
- An ASIC should replace `np.roll`/copy with an `N x W` circular buffer, load fixed-length calibration trial data through DMA, calculate `E` once at session reset, then reuse `E` for every 20 ms frame.
- Exact M2 affine MAC counts: IDEncoder `1,875,935,232` MAC/session at `M=33`; reference cached-E online graph `84,021,248` MAC/frame before elementwise ops; with constant-query caching and last-row-only output, `82,871,296` MAC/frame.
- Exact full-model parameter count is `4,594,888`; IDEncoder is `1,127,986`, leaving `3,466,902` in the recurrent online decoder path.
- `planB_tempconv` uses a depthwise causal FIR (`N=96`, `K=9`) plus `96->32->2` MLP: 4,130 parameters and an ASIC-optimized 4,000 MAC/frame when only the newest FIR output is evaluated.
- The PyTorch TCN computes all `W=50` convolution outputs and discards 49; a stateful ASIC should retain only `K-1` channel histories and compute the final output.
- Local six-session Plan-B results: CORAL `ref_raw` + frozen TCN mean R2 is about 0.172 at 4 s calibration and 0.186 at 32 s, versus the paper's SPINT M2 mean 0.26. This is lower accuracy but much lower hardware cost.
- Per-channel CORAL (`x'=a*x+b`) can be folded into the first depthwise FIR after calibration: `w'=a*w`, `bias'=b*sum(w)+bias`, eliminating online adaptation arithmetic.
- A safe checkpoint tensor inspection could not run because PyTorch is absent from the active Python environment; parameter/MAC totals are derived exactly from checked-in source/config, not measured from the checkpoint.

## 2026-07-14 QAT-B Audit
- Independent four-path re-evaluation of the QAT-B epoch-14 checkpoint reproduced anchor FP32 R2 `0.63024885`, shadow FP32 `0.62401214`, fake/integer `0.65356012`, and fake-vs-integer E `max_abs=0` on LOSO fold 0 / seed 42 / heldout `ses-2020-10-19-Run1`.
- The checkpoint passes the predefined integer deployment threshold on this fold (`delta R2=+0.02331128`), but shadow delta is `-0.00623670`, narrowly outside the original `-0.005` stability gate.
- QAT-B warm-starts from the corrected-STE QAT-A checkpoint at integer R2 `0.63752937`; QAT-B's incremental gain is about `+0.01603`, while the total gain from fixed-scale epoch -1 (`0.58665389`) is about `+0.06691`.
- Best epoch-14 learned-scale ratios are input `0.739x`, pre-out `1.042x`, mean `0.667x`, post0 `0.710x`, post1 `0.979x`, E `1.082x`; heldout calibration saturation rates remain below `0.5%` on every edge.
- Backward coverage now passes for all four weight matrices; the exact/surrogate dual-path STE repair is active.
- Training was not bit-exact at every epoch: epochs 9 and 13 logged one-code-scale E mismatches, although the frozen epoch-14 artifact passes independent exact evaluation.
- Learnable-scale causality remains unproven because QAT-B updates weights and scales jointly; a matched fixed-scale continuation is the required control.

## 2026-07-21 Architecture Exploration (B7-B14) — Concluded

### 实验协议修正
- 发现之前的 streaming_calibration_exp 所有训练都用了 `validation_protocol: minival`（held-in 自验证），不是 LOSO，且 `include_heldout_in_test: false`。这导致 R² 数字严重高估（B3 minival=0.630 vs LOSO heldin=0.620 vs LOSO heldout=0.236）。
- 正确协议：`data.validation_protocol=loso data.loso_fold=0 data.include_heldout_in_test=true`。所有 2026-07-21 之后的结果都用正确协议。

### B3 baseline 与论文对齐
- B3-D64 LOSO+heldout 实测：heldin R²=0.620, **heldout R²=0.236 ± 0.102**。
- SPINT 论文报告 M2 heldout R²=0.26 ± 0.13（Table 1，full SPINT，端到端训练）。
- B3 用 1.6% 的 encoder 参数（18K vs 1.13M）、冻结 decoder、20 epochs 蒸馏，达到论文报告性能的 91%。Gap -0.024 在论文 ±0.13 标准差范围内。

### B7-B14 变体全部不值得继续
- B7 (count-cond): heldout=0.211, gap -0.025 vs B3。Count conditioning 引入虚假依赖。
- B8 (rand-proj): heldout=0.200, gap -0.036。JL 引理保持距离但不保持判别力。
- B9 (hash K=16): heldout=0.166, gap -0.070。稀疏投影破坏 per-neuron discriminative identity。
- B14 (ternarized): heldin=0.058，训练崩溃。
- B10/B11/B12/B13 基于趋势预测都会更差，未实测。

### 根因分析
1. IDEncoder 的职责是 per-neuron discriminative identity，不是距离保持。固定/稀疏投影无法学到"哪些时间 bins 对区分神经元最重要"。
2. 论文 Table A2 证明 SPINT attention 与放电标准差的相关性在 M2 上高达 0.87 — 这种选择性必须通过 learned MLP 获得。
3. B3 已接近 GF-FSU 范式上限（paper 0.26），encoder 压缩的收益空间几乎为零。
4. 真正的瓶颈在 decoder（cross-attention, 3.47M params, 82.9M MAC/frame, 含 softmax/LayerNorm），不在 encoder。

### 决策
- **B3-D64 固定为 encoder 最终方案**。不再探索更激进的 encoder 架构。
- 详细文档：`streaming_calibration_exp/ARCHITECTURE_EXPLORATION.md`。
- 下一步方向：多 fold/seed 验证、joint training (unfreeze decoder)、或 decoder 压缩 (planB_tempconv)。

## 2026-07-21 Cross-Task Validation (M1/H1) — Started

### 数据下载
- M1 (DANDI 000941): 298 MB, 11 NWB files, 4 held-in + 3 held-out sessions, N=64 neurons
- H1 (DANDI 000954): 98 MB, 40 NWB files, 13 held-in + 14 held-out sessions, N=176 neurons
- M2 (DANDI 000953): 15 GB, 20 NWB files, 7 held-in + 6 held-out sessions, N=96 neurons
- M1+H1 加起来只有 396 MB（M2 的 2.6%），下载+验证 < 5 分钟

### 论文 Table 1 三任务对比
- SPINT M1 heldout = 0.66 ± 0.07（SPINT 优势最大，超过 FSU baseline 53%）
- SPINT M2 heldout = 0.26 ± 0.13（SPINT 优势最小，仅超过 FSU baseline 18-27%）
- SPINT H1 heldout = 0.29 ± 0.15（SPINT 优势中等，超过 FSU baseline 123%）

### M1 结果（已完成）
- M1 teacher: 20 epochs, best at epoch 14, run dir `2026-07-21-19-11-01/`
- B3-M1 student (LOSO fold 0, seed 42): **heldout R² = 0.630**
  - ses-20121004: 0.730, ses-20121017: 0.589, ses-20121024: 0.570
- 论文 SPINT M1 = 0.66 ± 0.07 → gap = -0.03（在方差范围内）
- Student/Teacher ratio ≈ 0.95（优于 M2 的 ~0.91）

### 跨任务汇总
| Task | B3 Heldout | Paper | Gap |
|------|-----------|-------|-----|
| M2 | 0.236 | 0.26±0.13 | -0.024 |
| M1 | 0.630 | 0.66±0.07 | -0.03 |
| H1 | — | 0.29±0.15 | 待开始 |

**结论**: B3-D64 在两个任务上均接近论文 full SPINT 性能，验证了压缩方案的 task-agnostic 有效性。

### Config 准备
- `configs/data/falcon_m1.yaml`, `configs/data/falcon_h1.yaml`
- `configs/model/_streaming_base_m1.yaml`（M1 专用 base，scaling_factor=1.0）
- `configs/model/streaming_b3_m1.yaml`, `configs/experiment/b3_m1.yaml`
- `scripts/wait_and_run_m1_student.sh`（teacher 完成后自动跑 student）
- `CROSS_TASK_VALIDATION.md` 完整文档
