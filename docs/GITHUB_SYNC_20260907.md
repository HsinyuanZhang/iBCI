# GitHub synchronization record — 2026-09-07

This record describes the deliberately scoped repository synchronization
prepared on 2026-09-07. It is a source and research-record snapshot, not a
claim that every local filesystem artifact has been mirrored to GitHub.

## Included categories

- Project source, launch and utility scripts, tests, and build definitions.
- The `E-ORT` naming record: fast exact-E execution followed by ONNX export
  and ONNX Runtime execution.
- Configuration and experiment specifications.
- Markdown and other human-readable research protocols, handoffs, decisions,
  terminal receipts, and small text manifests needed to understand or rebuild
  the tracked work.
- The existing DANDI 000688 handoff V2 material remains included; local ignore
  rules do not discard already tracked handoff content.

## Deliberately excluded categories

- Raw datasets; checkpoints and learned weights; cached tensors; replay and
  prediction arrays; large generated JSON replays; runtime logs; outputs; and
  generated environments.
- Binary packages and libraries, including vendored runtime wheels, as well as
  TensorBoard event files, process markers, and transient watcher files.
- The array-heavy local-only
  `sua_exploration/prit_feasibility/gate_results_task.json` artifact.
- The independently versioned Overleaf repository and the nested
  `prit_upstream` checkout. They retain their own Git histories and are not
  absorbed into this repository.

The Overleaf repository was independently synchronized at commit
`43e6dc2069dac9dc44b7287702395d22ab991728`; this outer-repository record does
not absorb that repository as a subdirectory or claim its files are included
in this commit.

## Snapshot boundary

The staging allowlist is generated from the active ignore rules and reviewed
before any commit. Files above five MiB and binary files require an explicit
exception; no broad `git add .` operation is used. The outer branch and the
default branch are updated only after the selected file list and remote
ancestry have been reviewed.

This is a point-in-time snapshot. Updates produced by active tasks after the
final snapshot boundary are not evidence that this record omitted them; they
belong to a later reviewed synchronization unless explicitly added before the
commit is sealed.
