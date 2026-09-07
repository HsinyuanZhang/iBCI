# H1 QueryAge comparison surfaces: existing-reference audit

This is a read-only inventory and derivation from immutable existing artifacts.
It contains no new model forward, selection, fitting, or artifact creation.

## Three distinct comparators

| Comparator | What it is | Existing evidence strength | Do not call it |
| --- | --- | --- | --- |
| Original-H1 | The as-shipped Original-SPINT deployment, replayed through its image-owned public wrapper | Exact 20,325-point native archive with prediction, target, session, and endpoint coordinates | C2 or a matched-support/calibration experiment |
| C2 epoch 15 | Immutable historical H1 C2 held-out-selected deployment package with fixed readout | Hash-bound 20,325 and 2,908 aggregate receipts only; no pointwise prediction archive | As-shipped Original-SPINT or a pointwise archive comparator |
| Current H1 QueryAge | Fresh formal candidate | Its own frozen 2,908 selection and 20,325 complete surfaces only after its separately authorized lifecycle | Evidence already available at the time of this audit |

Original and C2 must not be silently swapped merely because the Original
archive makes coordinate slicing convenient. C2 is the historical strong-C2
deployment reference; Original is the as-shipped SPINT reference. Neither
changes the descriptive-only, unequal-calibration/training-exposure caveat.

## Original-H1: exact 20,325-point archive

Authoritative files:

- [receipt.json](/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/family_runtime_v1/original_h1_frozen_same20325_v1/receipt.json), SHA-256 `539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48`.
- [original_h1_minival_native_float64.npz](/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/family_runtime_v1/original_h1_frozen_same20325_v1/original_h1_minival_native_float64.npz), SHA-256 `f1bb6739415ea66bb86bf4285035f131333e350072ae364139b81488912ba81c`.
- The recorded cache authority SHA is `92739d9f20d8184e2a24ddc2e4a00545c0b90170796c33f9d189cf90e8c6e41f`; the original archive replay used cache SHA `51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4`.

The receipt schema is `original_h1_as_shipped_frozen_minival_reference_v1`,
status `PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY`, with 20,325 scored
endpoints and 20,920 public calls.  Its immutable archive fields are
`prediction` and `target`, each `[20325,7]` float64, plus `session_id`
`[20325]` Unicode and `end` `[20325]` int64.  It is therefore the only one of
these two historical comparators from which a validated coordinate subset can
be derived without re-evaluation.

Complete-surface R² values (native-float64 archive arithmetic):

| Measure | Value |
| --- | ---: |
| Pooled | `0.9607844356539761` |
| Equal-session mean | `0.9604391953018591` |
| Worst session | `ses-19250113T120811`: `0.9531916118790326` |

| Session | R² |
| --- | ---: |
| ses-19250101T111740 | 0.9636908040825720 |
| ses-19250101T112404 | 0.9557921656535027 |
| ses-19250108T110520 | 0.9628082013110142 |
| ses-19250108T111022 | 0.9579010388164350 |
| ses-19250108T111455 | 0.9687221998636062 |
| ses-19250113T120811 | 0.9531916118790326 |
| ses-19250113T121303 | 0.9631100037158139 |
| ses-19250115T110633 | 0.9568868013280474 |
| ses-19250115T111328 | 0.9592579749424263 |
| ses-19250119T113543 | 0.9568750123628318 |
| ses-19250119T114045 | 0.9610300531498883 |
| ses-19250120T115044 | 0.9579108735459883 |
| ses-19250120T115537 | 0.9685327982730098 |

### Native FP64 archive versus historical FP32 outputs

“native_float64” is the archive/scoring representation, not a claim that the
released Original-SPINT model emitted float64. The frozen original scorer
requires a native FP32 public prediction, then writes prediction and target
arrays as float64 before calculating and archiving the reported metric; see
[score_original_h1_frozen.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/family_runtime_v1/score_original_h1_frozen.py:51),
[score_original_h1_frozen.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/family_runtime_v1/score_original_h1_frozen.py:221),
and [score_original_h1_frozen.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/family_runtime_v1/score_original_h1_frozen.py:233).
The stated metrics are consequently reproducible FP64 calculations over
float64 casts of the original native-FP32 outputs/targets.

## Derived Original 2,908-point frozen-selection subset

The derivation was read-only: for every one of 13 minival sessions, form
`end = query_starts + 699`, join the resulting `(session_id,end)` coordinates
to the immutable Original archive, and require one and only one match. All
2,908 keys occurred exactly once; the joined archive `session_id/end` columns
equal the queried coordinates, and its float64 targets exactly equal the
current immutable cache native targets cast to float64. The coordinate-list
SHA-256 is `eab09367df01f042441a35a1c344103131469fac86d649aa204888f590ffcb8b`;
the derived target-byte SHA-256 is
`e1b61a0f4f15342875ae88612e26ecef75e7eee937e755d704863d36744ee3a1`.

This is the same endpoint law used by the formal QueryAge scorer
[formal_prefix_score.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/formal_prefix_score.py:46):
`query_starts + (WINDOW - 1)`, with `WINDOW=700`, and it requires exactly
2,908 endpoints [formal_prefix_score.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/formal_prefix_score.py:18).

| Measure | Derived Original selection value |
| --- | ---: |
| Pooled R² | `0.9636925865316174` |
| Equal-session mean R² | `0.9635082159353363` |
| Worst session | `ses-19250113T120811`: `0.9486096001446648` |

Per-session derived selection R²: `0.9553767147288249`,
`0.9583816221157042`, `0.9639449384762370`, `0.9606465868611276`,
`0.9724780017639636`, `0.9486096001446648`, `0.9635707121622148`,
`0.9690584686664402`, `0.9633283183530751`, `0.9754483338952135`,
`0.9604282566877249`, `0.9635882840041992`, and `0.9707469692999807`, in the
same sorted session order as the complete table above.

## C2 epoch-15 fixed-deployment reference

Authoritative complete receipt:

- [c2_epoch15_complete_stream.json](/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/c2_epoch15_same_surface_v1/c2_epoch15_complete_stream.json), SHA-256 `96a9a496141e2bae533facb009903417f19a78ea970b0d3dc11c8742835d5c98`.
- Package SHA-256 `91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a`; checkpoint SHA-256 `ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215`.

It is `COMPLETE_FIXED_REFERENCE_SCORE` over 20,325 points: pooled R²
`0.8884989023208618`, equal-session mean `0.8879631849435660`, worst
`ses-19250120T115537` at `0.8669828176498413`. The direct historical selection
receipt is [c2_epoch15_selection_stream.json](/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/c2_epoch15_same_surface_v1/c2_epoch15_selection_stream.json), SHA-256 `94e309f139100c6d56b9b1a7af08731b2b038e6831194642296c8a45f33eb8b8`, with 2,908-point pooled R² `0.8715679049491882`.

C2 uses the same coordinate law (`query_starts + 699`) in
[c2_reference.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_optimized_v2/c2_reference.py:107), and its complete/selection receipts bind the same full cache authority. But those receipts contain scalar/per-session metrics only—no C2 prediction/target/session/end NPZ. The archive-only comparison code therefore imports C2 only as an aggregate and explicitly records `c2_prediction_npz_available=False` [compare_h1_frozen_quality.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/family_runtime_v1/compare_h1_frozen_quality.py:90). C2’s 2,908 scalar is a recorded direct evaluation, not a subset sliced from a complete archive.

## Comparison caveats

The Original selection scalar above is safely derived from an exact archive on
the same cache coordinates. It does not replace C2 as the historical strong
C2 comparator. C2’s complete and selection scores remain separately useful,
but lack pointwise evidence for coordinate-level residual, segment, or paired
comparison. Original used its as-shipped two-trial calibration while family
work uses explicit frozen three-trial M3 banks; C2 has its own fixed historical
deployment/readout. These are descriptive references, not matched training,
support-budget, calibration, selection-exposure, or non-inferiority evidence.

## Frozen Original wrapper: reset/calibration source audit

A constrained read-only source inspection of image
`sha256:f719c4228c345f9a1d6aa7c1e10d63ad7b9aa1f551dc95272b0b7a612d61fac6`
verified `/third_party/falcon_challenge/spint_decoder.py`
(`e4ae9ce5f51d5050a021b339c4d6b272ce10d580be4ab8c115ac436d1bd12763`)
and `/src/models/components/spint.py`
(`855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519`).
The inspection used only stdlib text/AST reads in a networkless, read-only
container; it did not import model code, load the pickle, read data, run
inference, or modify the image.

The actual `SpintDecoder.reset()` maps only each supplied session path's
`stem` through `FalconConfig.hash_dataset`, then uses that key to select an
already payload-resident `calib_trial_features` dictionary entry and clears
neural history.  `predict()` accepts neural observations, rolls the neural
buffer, and calls the decoder with that stored calibration tensor.  The
wrapper's constructor/reset/predict/observe paths contain no velocity, label,
target, query-file, or NWB read; its only relevant file read is the supplied
decoder payload in the constructor.  This resolves runtime target-access
concerns for the reference path.  The receipt's observed Original calibration
shape is two trials `[2,1024,176]`, whereas family QueryAge uses immutable
M3-derived identity/carrier banks.  Payload calibration provenance and
historical training/selection exposure remain unknown; this is not evidence
of training disjointness, unfairness, or grounds to drop the Original goal.
