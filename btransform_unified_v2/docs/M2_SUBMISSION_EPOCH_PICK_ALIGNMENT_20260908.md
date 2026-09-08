# M2 submission epoch-pick alignment audit — 2026-09-08

## Scope and decision

This is a read-only audit of the two finished submissions. It records that
future submission epoch picking is authorized to use **ext6**. It does not
alter the separate M2 B/D mechanism protocol, whose matched development
surface remains ext4.

## Finished payload facts

| Submission | Submitted payload and weight view | Selected checkpoint | Selection surface and rule |
|---|---|---|---|
| 582047 | `evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl`, SHA `4db109e7…`; its payload metadata and payload receipt both say `view: EMA`. The payload contains one `state_dict`; no raw-weight payload or averaging strategy is recorded. | `…/S1_SMALL_COS/seed42/epoch_008.pt`, SHA `68c1694f…`; payload receipt calls the submitted weight SHA `2c710376…`. | Six visible held-out-calibration sessions, query start trial 0; equal-session arithmetic mean; EMA epoch 1–24 candidates, earliest maximum. Selection e8 = `0.3990687579684978`. |
| 582128 | `evalai_m2_rift_r50_concat_cached_v1/artifacts/m2_rift_r50_concat_e7.pkl`, SHA `3a99d8b7…`; it has `ema_state_dict`, not `raw_state_dict`. The packing script explicitly loads `epoch_007.pt`, loads checkpoint `ema`, invokes `ema.apply_to(model)`, and serializes the resulting EMA state dict. No weight average beyond EMA is recorded. | `btransform_unified_v2/results/rift_v1/m2_r50_concat_s42_formal_v1/epoch_007.pt`, SHA `90d7d9b9…`. The checkpoint contains both `raw_state_dict` and EMA `shadow`; only EMA was submitted. | ext4 development: Run1/Run2 2020-10-30, Run1 2020-11-18, Run1 2020-11-19; equal-session arithmetic mean, 24 EMA candidates; earliest maximum selected e7 = `0.38257682030627616`. |

Primary evidence: `payload.receipt.json` and `selection.json` for 582047;
`pack_and_verify.py`, the RIFT payload, `score_receipt.json`, and e7
checkpoint for 582128. The finished official records are
`OFFICIAL_582047.json` (HO mean 0.3903053864) and `OFFICIAL_582128.json`
(HO mean 0.3379315881).

## ext6 definition

The exact six sessions are:

`ses-2020-10-30-Run1`, `ses-2020-10-30-Run2`,
`ses-2020-11-18-Run1`, `ses-2020-11-19-Run1`,
`ses-2020-11-24-Run1`, `ses-2020-11-24-Run2`.

The small-model receipt records their per-session R2 values and window counts
at the selected EMA e8. Its selection script freezes candidates to EMA,
epochs 1 through 24, and computes `np.mean([per[s] for s in SIX])`, with an
earliest-maximum sort key. The selection JSON has 48 rows because it includes
seeds 42 and 43; the submitted row is seed 42 EMA e8.

## Alignment contract for a later independent scorer

1. Bind an immutable run manifest to cell, seed, all 24 checkpoint byte
   hashes, query-bank hashes, six session names/window counts, and the intended
   view (`EMA`).
2. Score all and only epochs 1–24 with the same frozen ext6 query builder.
   Do not reuse the current RIFT ext4 receipt for a submission pick.
3. For each epoch compute the unweighted arithmetic mean of the six session
   R2 values. Record per-session values, count, prediction/hash provenance,
   candidate checkpoint SHA, and view.
4. Select the earliest epoch attaining the maximum finite ext6 mean. Serialize
   the selected EMA state only; raw-state and any post-hoc averaging require a
   separately declared, pre-specified contract.
5. Recheck all source/query/checkpoint hashes immediately before packing.

## Availability and missing pieces

The RIFT concat formal directory contains all 24 `epoch_001.pt` through
`epoch_024.pt`; each checkpoint has raw and EMA state, so its EMA candidates
are available for an ext6 rescore. The existing `score_receipt.json` is
ext4-only, so it supplies **no** ext6 RIFT score curve.

For 582047, the selected small payload is proved EMA e8 and the ext6 receipt
contains the 24-epoch seed42 EMA curve. This audit found no file evidence that
the currently queued/new M2 architectures already have an ext6 query-scoring
runner or an ext6-scored candidate receipt. That is the required next artifact;
no score was run by this audit.

## Non-goals

No submission, score, query data, training code, queue, or mechanism protocol
was changed. The ext4 B/D matched mechanism evidence remains ext4-only and
must not be relabeled as ext6.
