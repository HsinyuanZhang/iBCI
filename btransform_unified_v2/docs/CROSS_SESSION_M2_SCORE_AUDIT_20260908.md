# M2 cross-session score audit

`scripts/cross_session_v1/m2_score.py` is the post-training score wrapper for
the sealed cross-session M2 Z/B/D runs.  It is separate from the active trainer,
data reader, and model, so adding score audit output cannot alter a formal run.

It takes the same run identity arguments as the trainer:

```bash
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
$PY scripts/cross_session_v1/m2_score.py \
  --dest results/cross_session_v1/m2_z_s42_v1 --arm Z_NONE --seed 42 \
  --device cuda:0 --cpu-threads 2
```

Before opening EXT4, the wrapper requires matching `cell`, `arm`, and `seed` in
`run_meta.json` and `train_receipt.json`; the formal schemas and completed
receipt status; agreement between receipt, run metadata, and current source
hashes; agreement between receipt and metadata source-cache hashes; a complete
24-row source-trial-validation curve; and a finite declared selection metric
that equals its curve row.  It also verifies the selected package tensor by
tensor against `ema_epoch_{selected:03d}.pt`.  This is content equivalence, not
an unwarranted historical file-byte checksum: the score receipt records current
package SHA-256 values before and after scoring, and requires the package bytes
not to change during scoring.

The scorer then creates a fresh target-surface model, installs its live arm
memory, and invokes the trainer's strict EMA loader.  That loader requires an
exact trainable-parameter key set, finite tensors, matching shapes, and only
the fixed temporal recency-slope persistent buffer plus verified parameter
aliases.  Both the source-selected EMA and the mandatory e24 EMA are scored.

For each score endpoint and EXT4 session, it writes:

```
<dest>/score_arrays/<selected_source_trial_val|e24>/<session>.npz
```

Each compressed NPZ contains `prediction` in native covariate scale, `target`
in the same scale, and `eligible_starts`.  Batches are iterated in increasing
row-index order and the wrapper rejects any concatenated coordinate sequence
that differs from the cache's `eligible_starts` array.  The receipt includes
the prediction, target, and query-coordinate SHA-256 values, artifact path,
artifact SHA-256, per-session R2, and row count.  It also verifies the expected
EXT4 roster, counts, strictly increasing coordinates, window bounds, and
target geometry before scoring.

The score receipt preserves the established fields
`selected_ema_ext4` and `predeclared_e24_ext4_sensitivity`, so downstream
summary code can read their `per_session`, equal-session mean, pooled R2, and
window count fields while gaining the persisted row-level audit arrays.
