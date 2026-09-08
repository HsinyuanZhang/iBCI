# EvalAI 推荐点：实验完成后的证据清单（初稿，2026-09-08）

本文件只整理未来可建议的提交点；本集群**绝不 POST、提交或操作 EvalAI**。候选必须具备模型/权重 view、epoch、checkpoint SHA、selection surface/rule、候选范围、完整 receipt、local score/per-session 明细及官方结果（若已有）。全部齐全才可标为 `RECOMMENDED`；任何 mechanism B/C/shuffle/mean/nonattention arm 都不会因中间分数自动成为候选。

可用的汇总入口是 `scripts/paper_program_v1/summarize_submission_candidates.py`。它只读取 receipts、写入显式 `--output`，不访问网络或提交。例：

```bash
python3 btransform_unified_v2/scripts/paper_program_v1/summarize_submission_candidates.py \
  --output /tmp/evalai_candidate_summary.json
```

## 主要候选与保留项

| Dataset | Model/cell | 当前状态 | 已绑定选择 | 推荐前仍须读取的最终 receipt |
|---|---|---|---|---|
| M1 | `M1-RIFT-R100-D4-CONCAT-E100-RECENCY-V1` concat seed42 | `ELIGIBLE_CANDIDATE_FINAL_RECOMMENDATION_PENDING_COMPARISONS` | selected EMA e3；HO-trio equal-session mean `0.7341922605`；earliest maximum equal-session EMA。e3 checkpoint SHA `c202daff85412470e57a81a8db199d54fe2c7ed4d79850adf39f482b9cb806d4`。候选范围须以完整 score receipt 的 epoch map 为准。 | joint 最终 receipt 与其预先声明的比较/选择 contract；之后才能判断 baseline 是否仍为推荐点。 |
| M2 | `M2-RIFT-R50-D4-CONCAT-E50-RECENCY-V1` concat seed42 | `ELIGIBLE_CANDIDATE_FINAL_RECOMMENDATION_PENDING_REMAINING_MECHANISM_SEEDS` | frozen ext6 六 session、EMA epochs `1..24`、six-session unweighted arithmetic R² mean、earliest maximum。完整曲线选择 EMA e9=`0.3900576650553506`；checkpoint SHA `78435b8e01b0beaa78057c918ec195558d9c100a692ab7a8c066766f675ac88b`。D42 joint ext6 已完成但 e13=`0.3478275158419855`，低于 concat e9。 | 剩余 B/D mechanism seed receipts、three-seed paired ext4 mechanism summary、预先声明的 concat-versus-joint 比较/选择 contract，以及最终优先级决定和任何获授权的 packing receipt。单个 D42 ext6 不外推到 seed43/44。 |
| H1 | R300 recency EMA e22, submission 582073 | `REFERENCE_FROZEN` | official result已归档；冻结 payload/selection 已绑定。 | 无新追分或新推荐动作；只在汇总中保留其 official reference 和不可与 profile-only effect 混同的限制。 |
| 688 | external teammate lane | `EXTERNAL_PENDING_INTERFACE` | 本集群无本地候选。 | 仅接收队友提供的 strict-manifest、weight/selection、score/official receipts；在此之前不得列为推荐。 |

## 已知 evidence fields

### M1 concat seed42

- dataset/model/cell: M1 / RIFT R100 D4 concat E100 recency;
- weight view: EMA; epoch: 3; checkpoint path: `results/rift_v1/m1_r100_concat_s42_formal_v1/epoch_003.pt`;
- selection face/rule: visible HO-trio, earliest maximum equal-session EMA;
- local aggregate: `0.7341922605158908`; per-session R²: `20121004=0.7862409151`, `20121017=0.7271330438`, `20121024=0.6892028227`;
- full score receipt SHA: `8edee971ea3b2e47d5e8ec8c99d81f7467e212aae9d1389f3b1687e183e40288`; train receipt SHA: `12314de708d991fed4c6e49a3c518264099ebadf81e07eddf36c57965d44e5fc`;
- official result: none; this is not an instruction to submit it.

### M2 concat seed42

- dataset/model/cell: M2 / RIFT R50 D4 concat E50 recency;
- weight view: EMA; selected epoch: 9; checkpoint: `results/rift_v1/m2_r50_concat_s42_formal_v1/epoch_009.pt`, SHA `78435b8e01b0beaa78057c918ec195558d9c100a692ab7a8c066766f675ac88b`;
- selection face/rule: frozen ext6 six-session query cache, EMA epochs `1..24`, earliest maximum unweighted equal-session arithmetic R² mean;
- complete local selection result: `0.3900576650553506` across `15,403` windows. The same run's old e7 weight has ext6 `0.37425822760099076`; e9 increases this by `0.015799437454359855`;
- receipts: score SHA `6a985e1f3fb9392198725d6920efb7ea24b41052ddae84a335ae66ea72022e31`; manifest SHA `5d1bb0bf640c3d6ef646a0a6f7366bd2d3ff6befc506325da2d0c7aa7ca38884`; selected EMA SHA `8d156b28a9d144ebcd39f1597c7eb1af4a4e186a3af397b2442c911b5076418f`; full-curve validation SHA `26158347862d2e48a8baae13a72b7ec3cdf0cfc4e7a29f45541d8f0d1de8621b` passed;
- joint D42 ext6 的完整 24 epoch 曲线已选择 EMA e13=`0.3478275158419855`，比 concat e9 低 `0.04223014921336511`。D42 validation audit 已通过，但这是单一 seed 的 ext6 结果，不构成对 seed43/44 或其它 mechanism arms 的 ext6 外推，也不超越 concat 候选；ext4 mechanism receipt 同样不能替代缺失的跨-seed ext6 证据。

### H1 frozen reference

- dataset/model/cell: H1 / RIFT R300 recency EMA e22 / submission 582073;
- official held-out mean/std: `0.4027782688 / 0.1452665847`;
- official receipt SHA: `52e88d57b25c47f1648b73d4b8dad431a01fe49c4ea017153652de0266cc54d8`;
- priority: retain as frozen official reference; no score chasing, re-packing, or resubmission recommendation.

## Promotion rule

Promote a candidate to `RECOMMENDED` only after a final, immutable receipt binds all required fields above and any relevant completed comparison does not invalidate its priority. Do not choose an epoch from a partial curve, a mechanism ablation, or an ext4 result where ext6 policy applies.
