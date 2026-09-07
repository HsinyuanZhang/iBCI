# Independent audit — SUA selected-T4 factorized attention-logit residual

**Audited:** 2026-08-02 (Asia/Hong_Kong)  
**Scope:** the fixed seed-42 validation-only causal screen defined in
[`SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md`](SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md).
This is an independent read-only re-audit: no data files, formal sessions, or training jobs were
opened or started by this audit.

## Verdict

**The primary gate is a robust FAIL.**  The prescribed stop is justified: do not spend additional
GPU time on rank/seed/confidence variants, a formal SUA test, or INT8 follow-up for this particular
attention-logit-residual branch.

The failure is not an aggregation or checkpoint-selection artefact.  Re-running the fail-closed
aggregator in memory gave an exact match to the frozen `aggregate.json` for every substantive field
(all fields except its creation timestamp); its dedicated test suite passed **11/11**.

| Aligned comparison | Mean paired ΔR² | Positive validation sessions | Required mean ΔR² | Gate result |
|---|---:|---:|---:|---|
| aligned − ordinary T4 continuation | -0.003142 | 3/6 | +0.030000 | fail |
| aligned − residual-only shuffled T4 | +0.006402 | 5/6 | +0.030000 | fail |
| aligned − parameter-matched additive control | +0.003756 | 4/6 | +0.030000 | fail |

The per-session paired deltas are exactly those recorded in
[`aggregate.json`](../results/sua_t4_factorized_logit_residual_v1/aggregate.json).  In particular,
aligned does show a small descriptive advantage over a row-shuffled new residual path, but it loses
to retaining ordinary T4 alone and is an order of magnitude below the predeclared practical target.
That pattern cannot support an attention-selection improvement claim.

## What was verified

- **Frozen score rule:** each arm was rederived from the same eight fixed epochs (5--12), each
  epoch’s six-session mean was recomputed from its leaves, and each session’s paired score was
  recomputed across those eight checkpoints.  No per-arm argmax entered the endpoint.
- **Provenance:** live SHA-256 values match the frozen protocol, preflight receipt, selected T4
  anchor, teacher, split manifest, all three new result JSONs, and the ordinary-T4 continuation
  result.  The complete new-arm checkpoint paths and their file hashes were independently checked
  by the aggregator.
- **Frozen substrate:** for every scored checkpoint of every new arm, the checkpoint contained
  exactly the two added tensors
  `t4_logit_residual.query_factors` and
  `t4_logit_residual.unit_projection.weight`; every inherited selected-T4 tensor was bitwise equal
  to the anchor.  The live model code additionally freezes decoder and identity-encoder parameters
  before exposing those two tensors to the optimizer.
- **Mechanism controls:** the shuffled arm leaves the ordinary identity encoder’s aligned T4 input
  unchanged and permutes only rows entering the new residual.  The additive arm uses the same two
  tensors and produces a post-attention per-query scalar, not an attention-logit modification.
- **Isolation:** all arms carry the identical strict 27/6/6 manifest; their test-file lists are
  empty, fit-loader formal-test receipts are false, and the held-out evaluation seals state that no
  formal evaluation was performed.  Aggregation itself never opens a DANDI data file.
- **Resource claim:** rank 8 has 48 trainable parameters.  At the declared N=64 reference it adds
  2,048 B FP32 calibration-static state, 2,048 calibration-only factor MACs, and 1,024 online
  query/unit factor MACs; no neuron-axis quadratic term is introduced.

## Documentation caveat (non-invalidating)

The generic `training` subsection of the new-arm `run_metadata.json` says
`freeze_decoder: false` and `freeze_encoder_base: false`.  Those flags are inherited generic CLI
settings and are misleading for this specialized module.  They do **not** describe the executed
residual model: the specialized adapter calls `freeze_backbone_for_residual_pilot()`, the training
logs report exactly 48 trainable parameters, the checkpoint receipts state a frozen backbone, and
the scored checkpoint tensors prove the anchor state did not change.  The next metadata schema
should either override those generic fields or label them as inactive, but this bookkeeping defect
does not change the numerical or causal conclusion above.
