# External GF FALCON CPU packages — frozen v1 archive

All six packages are `FROZEN_PROTOCOL_INVALID_FOR_COMPARISON` and `submission_eligible: false`. Their original single-source `ridge=1` protocol lacks the official preprocessing control, so they are neither fair RIFT comparisons nor submission candidates. Do not push, register, or submit them. Their retained payloads, manifests, predictions, audits, and image IDs are provenance records. The replacement plan is [fair_v2](../fair_v2/).

Six self-contained archived build contexts remain under `../submissions/`. Each contains only its numeric `payload.npz`, `manifest.json`, the NumPy runtime, and entry point; it does not include source query labels, raw NWB files, or training pickles.

| Task | CORAL | AlignedFA |
| --- | --- | --- |
| M2 | [package](../submissions/m2_coral_v1/README.md) | [package](../submissions/m2_aligned_fa_v1/README.md) |
| M1 | [package](../submissions/m1_coral_v1/README.md) | [package](../submissions/m1_aligned_fa_v1/README.md) |
| H1 | [package](../submissions/h1_coral_v1/README.md) | [package](../submissions/h1_aligned_fa_v1/README.md) |

[READINESS.json](../submissions/READINESS.json) is the cross-package final status record. Each package also has `container_smoke.json`, `streaming_audit.json`, and `container_replay_audit.json`.

[replay_streaming.py](replay_streaming.py) runs the host full raw-bin replay against saved held-out predictions. [replay_containers.py](replay_containers.py) mounts the same query streams read-only into each final CPU image and checks image-resident runtime parity; both use append-then-stack retention and reset once per session.

`build_packages.py` rejects a normal invocation and requires `--rebuild-frozen-archive` for archival maintenance. That explicit path preserves the frozen status and does not restore eligibility.
