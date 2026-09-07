# M2 AOF-S static package V8 host-validation result

Date: 2026-09-03

Status: `STOP_NO_CANONICAL_EXECUTION`

This is a local engineering result, not an EvalAI submission and not a new held-out scientific score.

## Outcome

V8 corrected the two host-validator defects exposed by V7:

1. the streaming-calibration import path is established before unpickling the payload, so `src.models` resolves to the required implementation;
2. the validator explicitly imports NumPy before executing the source bridge.

A fresh CPU-only process successfully audited the unchanged V2 payload:

`65b8001156a7cdb58efd4bbff9547d88443cfa983546558009771f3d4c18ab20`

This closes the V7 `ModuleNotFoundError` and the bounded diagnostic `NameError`.

Two attempts to run the complete host `run_production_validation` in a fresh `/tmp` directory advanced beyond payload audit into source-bridge model loading, but neither produced a Python traceback, a wrapper completion/exit marker, or durable minival output. Because no inspectable terminal evidence exists, V8 was not authorized to reserve or execute a canonical result root.

## Preserved positive evidence

V8 does not invalidate the V7 container result. The frozen payload was already built into an offline image and completed container M2 minival with network disabled. V8 deliberately did not rebuild that image or rerun Docker.

The exact V7 five-body failure graph remains immutable and is bound by V8. No V7 result or artifact was modified.

## Freeze

- V8 focused no-data/no-Docker tests: `5 passed`.
- V8 explicit candidate closure: 82 leaves.
- V8 closure SHA-256: `285fe2c17d515baa0b17fd02954b64fb7b991f3bbc80bfd2ca1826794c15b6eb`.
- Corrected shared host validator SHA-256: `2b3ae452367710dbdc0fd7575f0963f93291af86eca7bc140c4dc026d33bdc90`.
- Canonical V8 result root: absent.
- Canonical V8 artifact parent: absent.
- Network, EvalAI submission, Docker execution, and GPU access: none.

## Decision

Stop the AOF-S host-validator successor sequence here. The offline container constructibility question is already answered positively by V7, while further host-only debugging cannot create a new scientific result. Research execution returns to the preregistered AJPF joint-training experiment.
