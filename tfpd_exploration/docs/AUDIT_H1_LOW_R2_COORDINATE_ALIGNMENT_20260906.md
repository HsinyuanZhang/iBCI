# H1 low-R² coordinate-alignment audit — 2026-09-06

## Scope and disposition

This is a read-only coordinate and indexing audit prompted by the low live
source-development scores (roughly 0.21 for both formal arms).  It examines
only the data-to-window-to-target path used by the H1 QueryAge formal run.  It
does **not modify** the live workers, checkpoints, cache, training protocol,
or any official/outer data. It does read the frozen cache once on CPU to check
its structural invariants, and it does not establish a model-quality
explanation.

The audited path is:

```text
NWB --falcon load_nwb--> neural[t, 176], velocity[t, 7], eval_mask[t]
                         |
                         +--> start s, neural[s : s + 700]
                         +--> target velocity[s + 699] × 20 for training
                         +--> prediction / 20 for native-unit scoring
```

## Findings that are proved by the local implementation

| Question | Evidence | Finding |
| --- | --- | --- |
| Does a W=700 query window target its last neural bin? | `h1_optimized_v2.data.collate_runtime_target` stacks `neural[start:start + WINDOW]` and reads `velocity[start + WINDOW - 1]`; `h1_optimized_v4.paired_train.collate` uses the same `start + 699` target. | **Yes.** The training coordinate is `(s..s+699) -> velocity[s+699]`. |
| Does source-development scoring use the same endpoint? | `h1_optimized_v2.score.evaluate` defines selected endpoints as `query_starts + WINDOW - 1`, creates a right-aligned window ending at each endpoint, and scores `velocity[ends]`. The formal-prefix scorer independently validates and uses `starts + 699`. | **Yes.** There is no observed one-bin shift between the ordinary training/selection paths and the formal scorer. |
| Are query starts legal for a full unpadded window? | Cache inspection found all 23,212 train starts and all 2,908 minival starts satisfy `0 <= s`, `s + 699 < len(neural)`, `s + 699 < len(velocity)`, and `eval_mask[s+699]`. | **Yes.** Selected W700 windows are complete and have eval-valid endpoints. |
| Is the first-three-trial exclusion applied to source training? | `load_session_arrays(... skip_first3=True)` finds the last eval-valid bin of the first three ordered calibration trials; `_query_starts` starts no earlier than that index plus one. Cached train starts begin exactly at `first3_end + 1` for every one of the 13 sessions. | **Yes.** The first three calibration trials are not part of a train window history. |
| Is a minival first-three exclusion incorrectly imposed? | The minival route calls `skip_first3=False`, explicitly documenting minival as query-only. Its starts begin at zero, while selected endpoints begin at 699. | **No.** That is intentional and does not introduce selected-window left padding. |
| Is unit/channel order permuted in this path? | The loader converts the neural array with `np.asarray(..., dtype=np.float32)` only. Window construction slices its second axis directly. Cached banks and masks have matching `176` units and every cached `unit_mask` has 176 active units. | **No local permutation is present.** It preserves the order emitted by `falcon_challenge.dataloaders.load_nwb`. |
| Is velocity scaling internally consistent? | Training targets are `native_velocity * 20`; model forecasts are divided by 20 before all audited R² calculations. The collator asserts the reversible bridge at `1e-7` absolute tolerance. | **Yes.** This is a scale bridge, not an endpoint-coordinate change. |

The relevant implementation locations are linked here for direct review:

- [NWB loading, trial extraction, W700 start construction, and the first-three rule](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_temporal_decoder_quick_product_v1/data.py:48)
- [The explicit source train/minival split and native-to-runtime target bridge](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_optimized_v2/data.py:12)
- [Selected versus complete endpoint selection, zero-left-padding, native-unit scoring](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_optimized_v2/score.py:13)
- [Formal scorer endpoint validation and `start + 699` construction](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/formal_prefix_score.py:49)

## Reproducible frozen-cache check

The only cache content read for this audit was
[`source_cache.pt`](/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt),
an existing 122 MiB artifact. Its SHA-256 at audit time was
`51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4`.
The corresponding recorded input authority is
[`source_cache_authority.json`](/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache_authority.json).

It was loaded with `torch.load(..., map_location="cpu", weights_only=False)`
in a process constrained to CPU cores 12–13 with every common BLAS/OpenMP
thread limit set to one. No model class was instantiated and no forward pass
was made. The retained shell invocation was:

```bash
sha256sum tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt
taskset -c 12-13 env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python -
```

The Python stdin program called `torch.load` on the absolute cache path and
performed the following material assertions:

```python
for split in ("train", "minival"):
    for row in cache[split].values():
        neural = np.asarray(row["neural"])
        velocity = np.asarray(row["velocity"])
        mask = np.asarray(row["eval_mask"], dtype=bool)
        starts = np.asarray(row["query_starts"])
        ends = starts + 699
        assert neural.ndim == 2 and neural.shape[1] == 176
        assert velocity.shape == (len(neural), 7)
        assert mask.shape == (len(neural),)
        assert np.all(starts >= 0)
        assert np.all(ends < len(neural)) and np.all(ends < len(velocity))
        assert np.all(mask[ends])
        assert tuple(row["bank"]["E0"].shape) == (176, 700)
        assert tuple(row["bank"]["T"].shape) == (176, 4)
        assert int(row["bank"]["unit_mask"].sum()) == 176
```

The same one-time read printed each train session's `first3_end` and first
query start. For all 13 train rows, `query_starts.min() == first3_end + 1`.
This is an empirical result of that cache inspection on 2026-09-06, rather
than a claim inferred only from the constructor's intended rule.

The cache-level shape and count check was made without forwarding a model:

| Split | Sessions | Selected windows | Eval-valid bins |
| --- | ---: | ---: | ---: |
| held-in calibration train | 13 | 23,212 | 132,476 |
| held-in minival selection | 13 | 2,908 | 20,325 |

All arrays were `neural[T,176]` float32, `velocity[T,7]` float32, and
`eval_mask[T]`; each cached static bank was `E0[176,700]`, `T[176,4]`, with a
fully active `unit_mask[176]`.  These facts rule out a visible mismatch such
as using a 175/177-channel input or a 699/700 endpoint in the audited path.

## Complete-stream padding is explicit and is not used for selected windows

The generic score helper has a separate complete-stream utility.  For an
endpoint earlier than 699, it creates a 700-bin input by left-padding only the
missing history with zeros and retaining the real final bin.  This is relevant
to all-bin complete scoring, because that mode uses every `eval_mask` bin,
including early bins.  It is not used by a selected minival window: those
endpoints are `s + 699` with `s >= 0`, so the helper takes all 700 real bins.

The formal-prefix cold-history helper follows the same principle: it replaces
only *left* history with zero and asserts that the final input bin remains
equal to the original endpoint bin.  Thus, padding is a documented
cold-start/complete-stream condition, not evidence of an accidental selected
training-target offset.

## What is not proved here

The local source route delegates original NWB interpretation to
`falcon_challenge.dataloaders.load_nwb(FalconTask.h1)`.  This audit did not
independently reconstruct that library's binning, timestamp alignment,
velocity derivation, trial ordering, or channel ordering from raw NWB objects.
Accordingly, it cannot prove that the original NWB producer's neural and
velocity timestamps are scientifically aligned; it only proves that the
current pipeline preserves the loader's arrays and consistently pairs the
last bin of each local W700 input with the corresponding returned velocity row.

Likewise, this audit does not test whether the known-source cache is an
appropriate generalization proxy, whether prefix dropout harms optimization,
or whether QueryAge/ROUTE has enough capacity.  Those are distinct hypotheses
and require an authorized experiment, not a coordinate conclusion.

## Bottom line

Within the inspected code and cached source arrays, there is no demonstrated
off-by-one target mapping, selected-window padding error, first-three-trial
history leak, active-unit shape mismatch, local channel permutation, or
runtime/native velocity-scale mismatch that explains the low R².  The most
important remaining coordinate uncertainty is upstream of this repository:
the semantics of `load_nwb` relative to the original NWB binning.  It remains
an explicitly unproven possibility rather than a diagnosed defect.
