# M2 channel-coordinate audit for alignment controls

## Decision

**AlignedFA and linear-CORAL may use the M2 96-column axis as a common fixed-electrode coordinate.  They must not describe a column as the same stable sorted neuron across days.**  This is an electrode/channel correspondence assumption, not a neural-identity correspondence claim.

## Direct local evidence

1. Every NWB in the seven source sessions and six public held-out calibration sessions under `SPINT-main/data/000953/sub-MonkeyN-*-calib/` has exactly 96 `units` rows.  In each inspected file:
   - `units/id == [0, ..., 95]`;
   - `units/electrodes == [0, ..., 95]`;
   - `general/extracellular_ephys/electrodes/id == [0, ..., 95]`;
   - `units/electrodes` is one-to-one (96 unique values) and is identical across all 13 files.
   The electrode table labels all rows `group_name = M1_array` and `location = M1`.

2. The installed FALCON M2 loader obtains `nwbfile.units.to_dataframe()`, then `bin_units` iterates that DataFrame row order and writes one output column per row.  It does not sort, match, or infer unit identity.  Because the raw M2 files have the one-to-one identity `units`-row-to-electrode mapping above, its output `[T,96]` column `j` is the count stream for electrode/channel ID `j` in this data release.  The local loader source is the installed `falcon_challenge.dataloaders.bin_units`; the M2 branch of `load_nwb` first constructs `units = nwbfile.units.to_dataframe()`.

3. The existing M2 cache records `channels: 96` and `unit_roster: "m2_96_contiguous"`, e.g. `tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train/ses-2020-10-19-Run1/provenance.json`.  The existing source cache covers seven sessions; `btransform_unified_v2/results/rift_v1/m2_joint_ext6_raw_m33_v1/manifest.json` records all six public held-out sessions with activity shape `[33,100,96]` and immutable raw-NWB paths/hashes.

## Permitted interpretation

- A common vector coordinate `c in 0..95` is supported by the NWB electrode-region mapping and the unchanged loader row order.
- It is appropriate for a **channel-space** linear alignment control to align source and target columns by that fixed coordinate.
- This evidence supports the M2-only “shared 96-channel array coordinate” precondition.  It does not establish functional or cellular equivalence.

## Prohibited interpretation / material risk

`units/id` is reset to `0..95` in every NWB and therefore is a per-file row identifier, not longitudinal single-unit tracking evidence.  A sorted/spiking unit may be lost, changed, or have different tuning while retaining the same electrode/channel index.  No source inspected here contains a cross-day unit-match table, waveform-match identity, or persistent cell identifier.  Do not state that AlignedFA/CORAL matches the same neurons, or evaluate an improvement as proof of stable neuron identity.

## Other tasks: metadata to audit before reusing this assumption

- **M1:** inspect each raw `000941/sub-MonkeyL-*-calib/*.nwb` file at `units/id`, `units/electrodes`, and `general/extracellular_ephys/electrodes/id`, and verify the `FalconTask.m1` `load_nwb` row order.  The existing static route is `learnable_recency_v1/scripts/m1_static_data.py`.
- **H1:** inspect each raw `000954/sub-HumanPitt-*-calib/*.nwb` at the same NWB fields and verify the `FalconTask.h1` loader's `units.to_dataframe()` / binned-unit ordering.  The existing activity route is `learnable_recency_v1/scripts/activity_data.py` (`load_h1_data`).

## M1 follow-up audit

**Decision: run M1 channel-space FA/CORAL if desired, with the same fixed-electrode-only wording as M2.**

All seven files reached by `learnable_recency_v1/scripts/m1_static_data.py` (four source in `000941/sub-MonkeyL-held-in-calib` and three held-out in `000941/sub-MonkeyL-held-out-calib`) have 64 `units` rows.  In all seven, `units/id`, `units/electrodes`, and `general/extracellular_ephys/electrodes/id` are each `[0, ..., 63]`; the unit-to-electrode mapping is one-to-one and identical across the files.  `FalconTask.m1` calls `units.to_dataframe()` and passes that row order to `bin_units`, which writes one output column per iterated row without cross-day matching.  Thus the released `[T,64]` M1 data supply fixed electrode/channel coordinate `0..63`; they do not supply same-cell tracking.  Reused `units/id` values cannot be treated as persistent unit identifiers.

## H1 follow-up audit

**Current execution decision: run H1 FA/CORAL only as an exploratory positional-row-assumption screen.  Its receipt must state that physical channel correspondence is unverified.**

All 27 current `000954/sub-HumanPitt-held-in-calib` and `sub-HumanPitt-held-out-calib` files have 176 `units` rows with `units/id == [0, ..., 175]`, and the H1 loader likewise calls `units.to_dataframe()` followed by `bin_units` in row order.  That establishes a common *array shape and delivered row index* only.  Unlike M1/M2, these H1 NWBs contain neither `units/electrodes` nor `general/extracellular_ephys/electrodes`; therefore the raw records inspected provide no NWB metadata that binds row `j` on one date to a physical electrode/channel `j` on another date.  The files do not contain a cross-day unit match or persistent-cell identifier either.

Using H1 row position for FA/CORAL adds an unverified ordering assumption beyond the shared-shape fact.  The current local screen may proceed under that explicitly named **positional-row assumption**.  Its receipt must state `physical_channel_correspondence_unverified: true`.  It must not be described in a paper, submission, or EvalAI result as demonstrated physical-channel alignment.  A separately documented acquisition/channel map, stable across dates and shown to be the loader's unit-row order, would be required for that stronger claim.
