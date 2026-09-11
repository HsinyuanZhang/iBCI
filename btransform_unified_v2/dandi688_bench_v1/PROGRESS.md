# PROGRESS — dandi688_bench_v1（exp1_narrow 全臂波次）

- 更新：2026-09-10 04:45（GLM5.3 执行；GPU0 全程，UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`，`CUDA_VISIBLE_DEVICES=0` 钉死，`PYTHONNOUSERSITE=1` spint env，torch 2.5.1+cu121）。
- 上位：`docs/README.md` / `btransform_unified_v2/docs/DESIGN_688_LOCAL_BENCHMARK_PROTOCOL_V1_20260909.md`；协议 `exp1_narrow`（9 train + 4 exam，0713/0714/0715/0716）；合同：12ep、seed42、batch32、EMA e8–e11、5931 upd/ep、preflight→train→score。

## 本波次执行记录（2026-09-09 23:41 → 09-10 04:37，~4.9h，预算 6h 内无 BUDGET_HIT）

| 臂 | 决定 | 依据 / 结果目录 | GPU | equal_session_mean_r2 |
|---|---|---|---|---|
| t4 | 跳过 | 已有 seal score receipt（`train_exp1_narrow_t4`） | （09-09 早前会话） | 0.8729 |
| floor | 跑 | preflight_floor_v1 已过 → train+score | GPU0 | **0.3247** |
| f_labelfree | 跑 | 本波次建 `cache_f_labelfree`（198 arrays 全验）→ preflight → train+score | GPU0 | **0.8257** |
| ts4 | 跑 | preflight_ts4_v1 → train+score | GPU0 | **0.8656** |
| equiv_zero | 跑 | cache_equiv_zero + preflight 已在盘 → train+score | GPU0 | **0.3454** |
| vstate | 跑 | `cache_vstate_exp1_narrow`（兄弟会话建、独立验 198 arrays + protocol 绑定）→ preflight → train+score | GPU0 | **0.8681** |
| vstate_concat | 跑 | 同上 cache → preflight → train+score | GPU0 | **0.8688** |
| f0 / z0 / u1_m10 / u1_m30 | 跳过 | 已有 seal score receipt | （09-09 早前会话） | 0.8726 / 0.2791 / 0.8826 / 0.8739 |
| t4_concat / z_vstate_srcbank | 跳过 | 派单裁定（附属） | — | — |

- 收尾产物：`results/exp1_narrow_gate_report_v1/gate_report.json`（seal 0444，sha256 `6082e7a910709efa3c70718460dd0f5be93a3daee744f974ec9ba8cc38a36c91`）——含预注册门、嵌套三分解、附属读数、vstate 系列、逐臂 per-session 读数与 provenance。
- 全部 receipt（train/score/preflight）均 0444 + sha256 sidecar，逐一枚验通过。

## 逐 session exam 读数（R²，exp1_narrow exam 面）

| 臂 | 0713 | 0714 | 0715 | 0716 | mean_mse |
|---|---|---|---|---|---|
| t4 | 0.8128 | 0.8913 | 0.8799 | 0.9077 | 0.0526 |
| f_labelfree | 0.7811 | 0.8075 | 0.8228 | 0.8913 | 0.0727 |
| floor | 0.4990 | 0.1981 | 0.4553 | 0.1463 | 0.2862 |
| ts4 | 0.7943 | 0.8900 | 0.8733 | 0.9049 | 0.0555 |
| equiv_zero | 0.4135 | 0.3292 | 0.3315 | 0.3074 | 0.2756 |
| vstate | 0.8087 | 0.9053 | 0.8424 | 0.9159 | 0.0545 |
| vstate_concat | 0.7816 | 0.9066 | 0.8686 | 0.9184 | 0.0540 |

## 判读（exp1_narrow exam 面，equal-session mean R²）

**嵌套阶梯三分解（预注册，主对比轴）**：activity 独立贡献 = f_labelfree − floor = **+0.5010**；carrier 标签增量 = t4 − f_labelfree = **+0.0472**；校准总价值 = t4 − floor = **+0.5483**（前两项之和 = 总价值，算术律自洽）。跨 session 能力的主体来自 label-free 的 activity 校准（E0 通路），carrier/标签在其上再贡献约 +0.05。

**附属消融**：equiv_zero 通路价值 = 0.3454 − 0.3247 = **+0.0207**（参数量对齐的随机投影通路 ≈ floor——通路/容量本身几乎不增益）；equiv_zero 信息价值 = 0.8729 − 0.3454 = **+0.5276**；z0（旧双零语义）carrier-only 读数 = 0.8729 − 0.2791 = +0.5939；f0 direct-carrier 边际 = 0.8729 − 0.8726 = **+0.0003**；ts4 内容对照 = 0.8729 − 0.8656 = **+0.0073**。

**预注册门（val/exam 面同口径，判读记录、不自动消费）**：gate1 t4−f0 = +0.0003 < +0.03 未过；gate2 t4−ts4 = +0.0073 < +0.03 未过；gate3 非劣（对 0.5750−0.01=0.5650）：t4 = 0.8729 ≥ 0.5650 **过**（且超过参照 +0.30）。

**vstate 系列（附属探索）**：增益门 vstate − t4 = 0.8681 − 0.8729 = **−0.0048** < +0.03 未过（与 t4 持平，不构成增益）；标签信息总增量 vstate − f_labelfree = **+0.0424**；融合读数 vstate_concat − vstate = **+0.0007**（concat ≈ proj_add，M2 的 concat 先验未在 688 显现）；archived u1_m10 = 0.8826（u1_m10 − t4 = +0.0097，与本波 vstate 量级交叉一致）。z_vstate_srcbank 未跑（附属降级）。

## 运维备注

- **多臂共卡实测失败**：5 臂同卡并发时每臂 ~2–3 steps/s（单臂 ~25 steps/s），聚合吞吐反降（多 CUDA context 切换），已回退 GPU0 串行队列（每臂 ~47 min train + ~2 min score）。
- **GPU1 全程有外来 compute 进程**（持续 ~2.3GB / 95% util），30 分钟空闲条件从未满足，从未向 GPU1 派臂。
- **并行会话协调**：兄弟会话同期在建 variant cache；为避免同 dest 并发写，本会话停掉自建 vstate build、复用对方建成物并独立校验（digest 198/198 匹配、estimator.protocol=exp1_narrow、b_mode=mean_k）。对方 f_labelfree 重复 build 撞非空 dest 干净退出，无损坏。
- 环境：`spint` conda env 必须 `PYTHONNOUSERSITE=1`（否则误用 user-site torch 2.12+cu130，驱动 12020 不兼容、CUDA 不可用）。
