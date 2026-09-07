# H1 installed-loader semantics audit — 2026-09-06

## Scope and authority boundary

This is a static, read-only inspection of the locally installed `falcon_challenge`
H1 loader and the repository's source adapter/cache-builder code. I did not
load an NWB/HDF5 file, cache payload, checkpoint, model, or GPU, and I did not
repeat the separate cache-content inspection.

The installed file examined was
[`dataloaders.py`](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py),
at `/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py`.
Its SHA-256 was `7638d839ec2085c7dde71ec9ca9676cbf1256c79fdcb327c4ac6487bfe5c9b05`.

These findings prove the behavior of this local installed file only. They do
not prove a remote evaluator version or the NWB producer's scientific
semantics. The raw-NWB facts explicitly attributed below were independently
observed by root, not read by this audit.

## Exact installed H1 route

The H1 branch is [lines 93–106](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:93).
It:

1. obtains `units = nwbfile.units.to_dataframe()`;
2. returns the stored `OpenLoopKinematicsVelocity.data[:]` as `kin`;
3. reads `eval_mask.astype(bool)`, with a legacy `~Blacklist` fallback;
4. derives timestamps from `OpenLoopKinematics`;
5. calls `bin_units(units, bin_size_s=0.02, bin_timestamps=timestamps)`;
6. derives a trial-change flag from positive `TrialNum` differences; and
7. returns `(binned_units, kin, trial_change, eval_mask)`.

Therefore H1 does **not** read a pre-binned `binned_spikes` acquisition.
It re-bins `units.spike_times`; the separate pre-binned branch is H2
([lines 107–118](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:107)).
It also does **not** calculate velocity: its target is the stored
`OpenLoopKinematicsVelocity` array at
[line 95](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:95).

## Binning, timestamps, axes, and mask

`bin_units` is [lines 13–76](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:13).
For H1 its default `is_timestamp_bin_start=False` means supplied timestamps
are bin ends ([lines 41–44](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:41)).
It forms edges `[t[0]-0.02, t[0], t[1], ...]`, uses `np.histogram` on each
unit's spike times, and returns `spike_arr[bins, units]`
([lines 68–75](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:68)).

The literal normal-branch timestamp code is
`OpenLoopKinematics.offset + arange(len(kin)) * OpenLoopKinematics.rate`
([line 98](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:98)).
Root's independent first-minival HDF5 inspection reports
`OpenLoopKinematics starting_time=0.0`, data attribute `offset=0.0`,
`conversion=1`, and `rate=0.02`; velocity, TrialNum, and mask reportedly
have the same `0.0/.02` time metadata. Thus this dataset and this installed
loader use the stored `rate` value as a 0.02-second sampling **interval** in
this expression. It must not be “corrected” here to `1 / rate` based on a
generic convention. That report also says the file contains
`units.spike_times/index/id`, rather than pre-binned neural data, and labels
the seven target axes `tx, ty, tz, rx, g1, g2, g3`.

The loader reads `eval_mask.astype(bool)` at
[line 97](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:97).
The installed evaluator subsequently applies it as `preds = preds[eval_mask]`
before regression scoring
([lines 727–737](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/evaluator.py:727)).
It is therefore an evaluation-row selector; the loader does not assert that
all earlier bins in a W700 history are eval-valid.

For gaps not equal to 0.02, `bin_units` warns and constructs/drops an
auxiliary proximal interval; for shorter gaps it uses the supplied short
interval ([lines 21–26](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:21),
[45–66](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:45)).
Whether any source session exercises those branches was not tested here.

The output's column order is DataFrame row order: `spike_arr[:, idx]` is
filled while iterating `units.iterrows()`
([lines 69–75](/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/falcon_challenge/dataloaders.py:69)).
There is no local unit sort, quality filter, deduplication, channel remap, or
normalization in this function. Counts are allocated as `uint8`; this audit
does not establish whether any source bin would overflow that representation.

## Repository source adapter and cache builder

The repository adapter directly receives the four loader outputs. It casts
neural and velocity to float32 and flattens/casts the mask to bool—without a
normalizer, rounding, clipping, permutation, or timing shift
([`_load_nwb`](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_temporal_decoder_quick_product_v1/data.py:48)).
It rejects anything except neural `[:,176]` and velocity `[:,7]`
([lines 107–128](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_temporal_decoder_quick_product_v1/data.py:107)).

The cache builder stores those returned arrays directly as `neural`,
`velocity`, and `eval_mask`; it converts only `query_starts` to int64 and
adds bank tensors
([`encode`](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_optimized_v2/cache.py:42)).
There is no neural/velocity z-score, centering, clipping, rounding, row
reordering, or temporal shift in that builder. Query construction is the
separate selection operation: a start is retained when its endpoint is
eval-valid, not when every history bin is eval-valid
([`_query_starts`](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_temporal_decoder_quick_product_v1/data.py:82)).

## Bounded conclusion

The local route is now statically clear: it pairs freshly histogrammed
per-unit spike counts with directly stored seven-dimensional velocity rows,
using the documented 0.02-second H1 timestamp progression, and the source
adapter/cache adds no coordinate transform after loader return.

This does not prove physical/scientific alignment of spike bins and target
rows, the producer's coordinate definitions, DataFrame-unit ordering intent,
or remote-evaluator identity. A separate HDF5 `searchsorted` reconstruction,
without `pynwb/load_nwb/bin_units`, would be the distinct kind of check needed
to compare this installed loader's re-binning with cached neural arrays. It is
not performed or proposed by this frozen-state audit.
