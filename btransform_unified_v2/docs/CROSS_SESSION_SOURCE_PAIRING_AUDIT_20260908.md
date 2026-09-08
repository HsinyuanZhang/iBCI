# Cross-session source-pairing audit

`scripts/cross_session_v1/audit_source_pairing.py` is the post-completion,
metadata-only evidence check for the 33-cell cross-session Z/B/D experiment.
Run it only after canonical `program.json` lists exactly the 33 primary cells
as `COMPLETED`:

```bash
python scripts/cross_session_v1/audit_source_pairing.py \
  --root results/cross_session_v1 \
  --out results/cross_session_v1/source_pairing_audit.json
```

The tool is fail-closed.  It accepts exactly four M1 LOSO folds, one M2
`source7_ext4` fold, and six H1 LODO folds, each with `Z_NONE`,
`B_ACTIVITY_ONLY`, and `D_JOINT` at seed 42.  It rejects incomplete programs,
duplicate cells, missing arms, unexpected folds, missing evidence, and an
existing output path.  Every completed program cell must also supply an
existing absolute `run` directory; no derived-path fallback is accepted.

It reads JSON metadata only and SHA-256 binds every JSON file it reads.  It
does not open, map, or hash numerical source/target arrays; the array hashes
already recorded in receipts and run metadata are its evidence.  It does not
read target predictions, target coordinates, target R2 values, or compute any
statistical summary.

Per M1 fold it requires exact Z/B/D equality of every `actual_source_arrays`
hash field, `batch_order_sha256`, nonempty `data_contract`, nonempty `recipe`, and
`source_code_sha256`, and exact B/D equality of `initialization_hash`.

Per H1 fold it requires exact Z/B/D equality of `source_manifest_sha256`, the
explicit nonempty scientifically shared recipe fields, `source_code_sha256`, and
source-file hashes.  It also requires 32 ordered epoch rows in every arm and
checks equality of `sampler_endpoint_keep_sha256` at every epoch.  B/D
`initial_parameter_sha256` must match.  The source-validation selection remains
the existing source-session EMA criterion; this audit does not alter it.
The H1 run schema intentionally stores `recipe_binding` at the top level of
both `run_meta.json` and `train_receipt.json`, while `train_receipt.json`'s
embedded `meta` is the exact `run_meta.json` object with that one key removed.
The audit requires both recorded bindings to be nonempty SHA-256 values and
recomputes the binding as SHA-256 of the compact, sorted JSON encoding of the
binding-free metadata.  All three values must match.  Bindings may differ by
arm because arm-specific metadata and initialization are part of the recipe;
the output preserves the verified value for each arm.

For M2 it requires exact equality of the whole `split_contract`, including its
24-epoch sampler manifest, and of `source_hashes`.  Its manifest must contain
exactly nonempty batch lists for epochs `1` through `24`, and the source-cache
roster must exactly match the seven `HELDIN_SESSIONS` declared by the existing
M2 plan module.  For every source session it
checks equality of the recorded `X_store.npy`, `target_store.npy`,
`eligible_starts.npy`, and `mapping.json` hashes.  It additionally requires
the B/D `calib_activity_sha256` values to match.  This is the generalised,
fail-closed form of the prior `m2_source_pairing_audit.json` evidence.

The output has `status: "PASS"`, per-fold evidence, SHA bindings for all input
JSON files read, the audit script's own SHA binding, its metadata-only scope, and the remaining root verification
boundary: `summarize_figure.py` remains responsible for target-score and
coordinate validation and for H1's date-level reporting aggregation.  To avoid
duplicating the M2 manifest's large batch-index payload, the audit strictly
compares the full object but emits only its canonical JSON SHA, internal
manifest digest, per-epoch batch counts, and core split declarations.
