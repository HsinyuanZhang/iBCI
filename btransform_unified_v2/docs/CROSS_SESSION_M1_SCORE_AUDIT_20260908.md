# M1 cross-session audited scoring

Use this entrypoint only after the matching arm has completed training and its
`preflight.json` says `PASSED`. It is a score-only wrapper: it checks the
training receipt and checkpoint hashes before reading target data, verifies the
rematerialized source arrays against the training receipt, and writes a new
score receipt with hashes for the wrapper and all score dependencies.

```bash
export PYTHONPATH=/home/xinyuan/Work_host/SPINT/btransform_unified_v2/src:/home/xinyuan/Work_host/SPINT/btransform_unified_v1/src:/home/xinyuan/Work_host/SPINT
python3 scripts/cross_session_v1/m1_score.py \
  --dest results/cross_session_m1_v1/ses-20120924/Z_NONE/s42 \
  --target ses-20120924 --arm Z_NONE --seed 42 --device cuda:0
```

Replace `Z_NONE` in both destination and `--arm` with `B_ACTIVITY_ONLY` or
`D_JOINT` for the other arms. The wrapper reports exactly two target surfaces:
the source-selected EMA checkpoint and the fixed final epoch-24 EMA checkpoint.
It never selects using target labels.

`D_JOINT` reads target M10 EMG and projects it by NNLS using the source cache's
sealed `basis_dictionary` and `basis_scale`, then normalizes its carrier with
the sealed source `normalizer_mean` and `normalizer_scale`. It does not reload
source EMG or perform source NNMF/ridge/normalizer fitting during score. `B`
uses target M10 activity only, and `Z` uses only target query windows.

The output schema is:

| File | Arrays |
| --- | --- |
| `target_selected_predictions.npz` | `target`, `prediction`, `window_start_padded`, `output_index_padded`, `output_index_query_relative`, `prefix_bins` |
| `target_epoch24_predictions.npz` | Same schema for fixed epoch-24 EMA |
| `score_receipt.json` | score-source hashes, frozen rSyn3 hashes, checkpoint hashes, metrics, and coordinate schema |

Coordinates are derived from the actual target `FalconDataset`: `prefix_bins`
is `trial_start_indices[target][0]`; `output_index_padded` is the final raw
position of each scored window; `output_index_query_relative` subtracts that
actual prefix. Thus each prediction row, target row, and query-relative output
position are aligned by row index without relying on a hard-coded padding
constant.
