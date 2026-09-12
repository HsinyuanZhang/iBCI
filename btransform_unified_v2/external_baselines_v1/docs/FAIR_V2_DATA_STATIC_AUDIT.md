# FAIR V2 raw-support and static-RIFT availability audit

Scope: read-only audit on 2026-09-11.  This document does not authorize a
model run, fitting, checkpoint conversion, or data export.

## Conclusion

All three target raw 20-ms neural support surfaces can be rebuilt without a
dummy tensor, but the existing `activity_data.py` calibration route is not the
right input for the proposed fair control because it resamples M1/H1 to cubic
1024 bins.  M2 support is reconstructed from the six public NWBs on the canonical query
timeline, with exactly trial IDs `0..32` and their native `eval_mask` bins.  M1 requires retaining the first ten raw calibration trials per one
of the three HO3 recordings before any cubic interpolation.  H1 requires the
first three `eval_mask`-valid `TrialNum` runs from each of the fourteen HO-M3
recordings, at original bin count.

The available static-RIFT checkpoints are source-only static-identity models.
They have raw, unnormalized 20-ms input and no target calibration interface.
They provide a useful frozen frontend/temporal/readout target for a *new*
raw-support preprocessing wrapper, but no existing checkpoint validates a
target `diag-z` or CORAL transform.  Such a transform must be inserted before
`frontend.local_conv`; it must not be added after the frontend or inferred to
be neutral because the model contains trainable LayerNorms.

## Exact raw data routes

### M2: EXT6, six public held-out recordings, first 33 native trials

* Authoritative builder: `btransform_unified_v2/scripts/rift_v1/prepare_m2_joint_ext6_m33.py`, especially `run()` and `tfpd_exploration.src.m2_dual_track_v1.data._calib_bundle`.
* Canonical public query reader: `tfpd_exploration/scripts/run_m2_small_s1_visible_ext6_epoch_pick_v1.py`, loaded by that builder's `builder_module()`.
* Support identity rule: `support_trial_ids = list(range(33))`.  The former
  `calib_activity` output is `[33,100,96] float32`, but it is a
  mask-filtered cubic interpolation view, not raw support.
* The six sessions and their raw NWB paths/hashes were sealed before deletion
  in `external_baselines_v1/results/m2_full_v1/receipt.json` under
  `target_support_provenance`, and in the deleted-root builder's prior schema
  `m2_joint_ext6_raw_m33_v1`.  `activity_data.py:load_m2_data()` confirms the
  same M33 rule.
* Query input remains the exact EXT6 cache at
  `tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query`, accessed by
  `learnable_recency_v1/scripts/m2_static_score.py:query_surface()` or
  `activity_data._load_m2_query_modules().score_reader.query_surface(...)`.
  It supplies `X_store.npy`, `eligible_starts.npy`, 49-bin pad, and local
  target only for scoring.  The frozen score receipt records 15,403 total
  query rows and per-file hashes at
  `learnable_recency_v1/results/selection_m2_static_final_ema_ext6_s42/score_receipt.json`.
* Source seven recordings are at
  `tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train/*`;
  `m2_static_train.load_static_surface("source_train")` reads raw `X_store`,
  starts and source target.  It is query-only and does not consume M33.

Do not use the absent
`btransform_unified_v2/results/rift_v1/m2_joint_ext6_raw_m33_v1` as a required
input.  FAIR V2 maps the first 33 NWB trial intervals through the matching
`eval_mask` onto `X[49:]`; it retains native bins and verifies the NWB
timestamp length equals the canonical query cache length.  The historical
M33 cubic view remains only provenance for the trial budget.

### M1: HO3 recordings, raw 20-ms support before cubic resampling

* Raw source/query plane: `learnable_recency_v1/scripts/m1_static_data.py`.
  `_source_path`, `_heldout_path`, `_nwb_loader`, and `_load_item` use
  `falcon_challenge.dataloaders.load_nwb(FalconTask.m1)`, keep raw float32
  neural bins, prepend exactly 99 literal-zero bins, and expose `X`, `starts`,
  `pad=99`, and `Y`.
* Source roster: 20120924/26/27/28 in
  `SPINT-main/data/000941/sub-MonkeyL-held-in-calib`; the formal static receipt
  fixes all four file hashes and 213,336 source query windows in
  `learnable_recency_v1/results/m1_static_s42/run_meta.json`.
* HO3 roster: 20121004/17/24 in
  `SPINT-main/data/000941/sub-MonkeyL-held-out-calib`; the exact query starts,
  file hashes and 1,305/1,295/1,281 row counts are fixed in
  `m1_static_s42/score_receipt.json`.
* Existing external-baseline support route is
  `external_baselines_v1/data.py` -> `activity_data._m1_activity_item`.
  It identifies trial boundaries from NWB timestamps/eval_mask and takes
  `starts[:10]`, then cubic-resamples each of those ten trials to 1024 bins.
  For FAIR V2, reuse only its raw trial/eval-mask selection logic and retain
  each selected raw 20-ms segment; do **not** call `interp1d`.

"HO3" here denotes the three held-out recordings.  The historical
external-baseline support budget inside each recording is the first ten
calibration trials, not three trials; the future implementation must state
that distinction in its receipt.

### H1: HO-M3, first three raw trial runs

* Raw source cache/query source: `btransform_unified_v1.adapters._h1_source_cache()`;
  `learnable_recency_v1/scripts/h1_static_train.py:_load_data` obtains all 13
  source sessions from its `train` section.  It uses raw 176-column neural,
  `query_starts`, `eval_mask`, and end-anchored `endpoint_context(...,300)`.
* Raw HO query route: `h1_static_train.py:_load_ho`, reading all fourteen NWBs
  in `SPINT-main/data/000954/sub-HumanPitt-held-out-calib` with
  `load_nwb(FalconTask.h1)` and scoring `score_mask` endpoints.
* Exact existing HO-M3 support selector: `external_baselines_v1/data.py:load_h1_data`.
  It reads `TrialNum` and `eval_mask`, takes the first three distinct valid
  trial values in observed order, then selects raw bins with
  `eval_mask & (TrialNum == trial_id)`.  The present code cubic-resamples these
  segments to `[3,1024,176]`; FAIR V2 must retain the selected native-bin
  segments instead.  For public held-out H1 it verifies `len(values) == 3`.
* Source roster is thirteen held-in H1 records; target roster is fourteen
  held-out records grouped as S6..S12.  Existing static formal metadata binds
  the source cache SHA `51ff9e...bcc91b4` and H1 300-bin endpoint mask SHA in
  `learnable_recency_v1/results/h1_static_s42/run_meta.json`.

## Completed static-RIFT assets

All three use `StaticLearnableRiftDecoder` in
`learnable_recency_v1/src/learnable_recency_v1/static_model.py`:

* model replaces E0 projection with `static_identity[units,16]` and keeps
  literal-zero `static_carrier[units,4]`;
* its input is raw 20-ms neural (`normalization: none` for M1; direct raw cache
  for M2/H1), frontend starts at `frontend.local_conv`;
* source-only training has no TaskBank, E0, carrier or target-calibration
  support route.

| Task | Completed run | Available fixed-final checkpoint | Selection/status evidence |
|---|---|---|---|
| M1 | `learnable_recency_v1/results/m1_static_s42` | `epoch_024.pt`, SHA `562e0febc2818e2fc6952e383a5dea7fadbc6493034cee3557a315e1538cf4df` | `train_receipt.json` says formal completed; `score_receipt.json` is fixed final EMA and explicitly `no held-out tuning`. |
| M2 | `.../results/m2_static_learned_slope_s42` | `epoch_024.pt`, SHA `8313e7bb202d43def2885b7e7c4c74dc8466af81aaccb18b437333ae86801612` | `selection_m2_static_final_ema_ext6_s42/score_receipt.json`; 15,403 EXT6 rows, fixed final/no held-out tuning. |
| H1 | `.../results/h1_static_s42` | `epoch_032.pt`, SHA `b79adeda51955c14bb60e4643835a5d8d6862cdb2480134fd9fd42a04a765b51` | `train_receipt.json` and `local_ho_static_report.json`: fixed final EMA e32; HO labels never select epoch. |

The fixed-final EMA checkpoints above are the intended paired frontend for
FAIR V2.  `STATIC_SOURCE_ONLY.md` also describes all-epoch earliest-max scans
for other protocol variants; those scans are not required to use these fixed
final checkpoints.

## Minimal implementation boundary and risks

1. Construct target raw support from the routes above, fit target-only
   `diag-z` or CORAL parameters on those raw bins, and apply the transform to
   every raw query bin before the frozen static model.  Freeze all checkpoint
   parameters and call `eval()`.
2. Feed transformed `[B,context,units]` plus the existing input-valid mask to
   `StaticLearnableRiftDecoder.forward`, or use `StaticRiftStreamDecoder` for
   continuous inference.  Keep query padding literal zero in transformed
   feature space, as in the existing causal contract.
3. Do not adapt `static_identity`, token/slot modules, temporal stack,
   `final_norm`, or readout.  No target Y may enter fitting.
4. The transform is not guaranteed harmless: `static_model.py` applies a
   learned convolutional frontend followed by `token_norm`, `slot_norm`, and
   `final_norm` LayerNorms.  LayerNorm is not a substitute for a fixed channel
   coordinate transform; diagonal scaling/CORAL can change learned filters and
   cross-unit attention.  Record transform ordering, dtype, ridge/shrinkage,
   support samples and all source/target hashes.
5. Existing static checkpoints store raw and EMA state; identify the precise
   state view chosen by the new protocol before writing a CPU loader.  The H1
   script's `_rebuild`, M1 trainer rebuild path, and M2 `m2_static_score.py`
   are the appropriate reference constructors.  Do not use an old RIFT
   package payload as a shortcut because its identity/frontend contract can be
   different.

## Existing standard/legacy RIFT evidence

* Canonical architectural documentation: `btransform_unified_v2/README.md` and
  `learnable_recency_v1/docs/DESIGN.md` distinguish RIFT from legacy BT-EORT.
* Same-task selection conventions are documented in
  `learnable_recency_v1/docs/STATIC_SOURCE_ONLY.md`,
  `MATRIX_LEARNED_V1_20260911.md`, and `FULL_FLAT_CONTROL.md`: M1 HO3 all24,
  M2 EXT6 all24 equal-session earliest-max, H1 HO-M3 grouped-seven all32
  earliest-max.  Those are protocol references, not evidence that the static
  fixed-final runs already obeyed them.
* Standard learned RIFT receipts are present under
  `learnable_recency_v1/results/h1_{activity_only,flat_p16,learned_slope_default*,norm_only}_*/ho_m3_selection.json`
  and M2 selectors under `.../results/selection_m2_static_*` plus
  `scripts/rift_v1/m2_ext6_epoch_pick.py`.  Legacy official package receipts
  and selected checkpoint/payload bindings are under
  `tfpd_exploration/submissions/evalai_*rift*/artifacts/`; they should only be
  used to establish historical architecture/selection provenance, never as a
  FAIR V2 checkpoint substitution.
