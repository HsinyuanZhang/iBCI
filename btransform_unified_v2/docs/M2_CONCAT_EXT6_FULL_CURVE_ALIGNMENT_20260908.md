# M2 concat ext6 full-curve validation

The formal concat run was rescored on the frozen six-session ext6 query cache for all 24 EMA checkpoints. The read-only validation artifact is `results/rift_v1/m2_r50_concat_s42_ext6_pick_v1/validation_audit.json`; it passed every seal and completeness check.

The curve contains six sessions and 15,403 query windows at every epoch. Each candidate report has the exact manifest checkpoint SHA-256, finite per-session R2, the full six-session arithmetic mean, and the expected six window counts. The manifest's live source hashes, all query assets, and the sealed BT oracle links all matched. The earliest maximum was EMA epoch 9 at `0.3900576650553506`.

`selected_ema.pt` is an exact materialization of epoch-9's EMA shadow applied to the formal checkpoint's raw model state. Its hash matches the selection receipt, its keys and tensors match the independently rematerialized state dict, and the receipt declares that raw state was not serialized.

For aligned local-ext6 comparison within the same RIFT concat run, the former epoch-7 weight scores `0.37425822760099076`; the selected epoch-9 weight improves it by `0.01579943745435984`. The sealed BT submission 582047 uses a different SMALL concat family and has local ext6 selection value `0.3990687579684978`; the selected RIFT concat is `0.0090110929131472` lower on that common local selection surface. This is an alignment result for candidate selection only. The official held-out result of 582047 (`0.39030538635614676`) remains a completed external record for that separate submitted model, not an outcome for the newly selected RIFT artifact.

No EvalAI interaction, official-test access, DANDI 688 access, or new scoring occurred during this validation.
