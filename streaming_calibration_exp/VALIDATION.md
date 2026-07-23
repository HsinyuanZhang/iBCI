# Validation protocol notes

## Protocols

`FalconDataModule` supports three held-in validation protocols via `data.validation_protocol`:

| Protocol | Train sessions | Val held-in sessions | Use case |
|---|---|---|---|
| `minival` | all 7 held-in calib | all 7 held-in minival | legacy screening only |
| `loso` | 6 held-in calib | 1 held-out held-in minival | structure selection (recommended) |
| `rotation_5_2` | 5 held-in calib | 2 held-in minival | fixed rotation alternative |

For `loso`, set `data.loso_fold` in `[0, 6]`. For `rotation_5_2`, set `data.rotation_id`.

Validation queries always use each session's own fixed first-33 calibration support
(`random_calibration=false` on val/test splits).

## Held-out isolation

- `data.include_heldout_in_fit: false` (default): training/validation epochs do **not**
  evaluate the six held-out sessions.
- `data.include_heldout_in_test: false` (default): post-fit `trainer.test()` evaluates only
  the fold validation held-in session(s). Held-out is reserved for `final_heldout_eval`.
- Final `trainer.test()` with `include_heldout_in_test=true` evaluates held-in + held-out once.
- Checkpoint selection monitors `val_heldin/r2_mean` only.

## Artifacts

Each run now writes:

- `metrics_summary.csv` (design + aggregate test rows)
- `metrics_per_session.csv`
- `checkpoint_manifest.json` (+ copied `checkpoints/best.ckpt`)

Matched baseline for `R2_delta_vs_matched_baseline`:

```text
outputs/streaming_calibration/b0_baseline/metrics_per_session.csv
```

Generate with:

```bash
python scripts/export_b0_baseline.py
```

## Implication

- `minival` is not sufficient for final structure lock.
- Use LOSO (7 folds) or `rotation_5_2` before claiming Gate 2.
- Six held-out sessions remain final-only comparison after candidate lock.
