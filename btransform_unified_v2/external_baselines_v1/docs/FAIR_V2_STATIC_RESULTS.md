# FAIR V2 frozen static-RIFT controls

The static controls retain the source-trained, frozen EMA checkpoint for each
task.  `identity`, `diag_z`, and CORAL only transform the raw neural input
before `frontend.local_conv`.  Every transform fits pooled held-in raw support
and the target recording's declared raw calibration support; no transform reads
target behavior labels.  Diagonal z-score is
`(x - mu_target) / sd_target * sd_source + mu_source`; CORAL is
`(x - mu_target) Ct^-1/2 Cs^1/2 + mu_source`, with 0.1 diagonal shrinkage and
ridge `0.001`.  Padding remains literal zero.

M1 and M2 predictions were run through the fixed-window reference runner and
then aggregated in separate `report_rescored.json` files.  The raw run reports
remain untouched.  M1 identity reproduced the historical flattened result
within float32 batching tolerance: `-0.1489977445` versus `-0.1489979657`.

H1 uses the separate `fair_v2/static_chunked.py` runner.  It is an exact
finite-receptive-field execution of the frozen reference model: local
convolution width 5 plus RIFT windows `[75, 75, 75, 74]` gives 300 native rows,
which equals the checkpoint's fixed context.  Each 512-row chunk prepends 299
native rows and retains only its new suffix scores.  The report records
fixed-window parity samples at recording starts, middles, ends, and chunk
boundaries for every arm.  The largest H1 native-score error was
`1.20e-7`; the identity stream parity maximum was `1.64e-7`.  A full 2,313-row
reference-window comparison for `19250126T113454` also passed with maximum
prediction difference `9.31e-9`.

H1's main aggregation is `grouped_seven`, which combines the 14 recordings
into seven FALCON sessions.  The grouped-seven standard / legacy R² means are:

These are local public held-out implementation and protocol diagnostics only.
They do not establish an advantage or a failure for frozen RIFT on the final
EvalAI held-out evaluation, which is the decision surface for this work.

| arm | standard mean | legacy mean |
| --- | ---: | ---: |
| identity | 0.209015 | 0.209987 |
| diag_z | 0.256093 | 0.257010 |
| CORAL | 0.276458 | 0.277351 |

The complete artifacts, frozen checkpoint hash, code snapshot, prediction
hashes, support provenance, metrics, and parity receipt are under
`fair_v2/results/static_h1_v2/`.
