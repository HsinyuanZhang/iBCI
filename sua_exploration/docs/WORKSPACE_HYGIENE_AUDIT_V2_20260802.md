# Workspace Hygiene Audit V2 — 2026-08-02

## Outcome

This was a non-destructive, low-risk workspace hygiene pass. It removed only
ignored, reproducible Python and pytest caches outside the protected SUA and
experiment-output areas. No source, protocol, manifest, receipt, result,
checkpoint, NWB/data, M1, EvalAI, or active T4 M30 Experiment A material was
modified or deleted. No `reset`, `checkout`, commit, push, or `.gitignore`
change was performed.

The cache deletion released **3,670,016 bytes** (**3,584 KiB**, or **3.50
MiB**) as measured by `du -sx -B1` before and immediately after deletion.

## Scope and safeguards

The following locations were explicitly excluded from deletion:

- all of `sua_exploration/`, including active Experiment A scripts, tests,
  results, and documentation;
- `sua_exploration/data/`, `sua_exploration/checkpoints/`,
  `sua_exploration/results/`, `sua_exploration/cache/`, and both EvalAI trees;
- `SPINT-main/data/` and `SPINT-main/local_data/`;
- `streaming_calibration_exp/logs/` and `streaming_calibration_exp/outputs/`;
- the Git database and all other data, artifacts, logs, manifests, receipts,
  and result-bearing locations.

Before deletion, each candidate was required to be a real directory named
`__pycache__` or `.pytest_cache` and to match an existing Git ignore rule.
There were no standalone `*.pyc` or `*.pyo` files outside those cache
directories. A process inspection found no repository training or pytest
processes at audit time. Empty `.codex/` was discovered but deliberately
retained as local tool state rather than treated as a temporary directory.

## Initial inventory

At the start of the audit, `git status --short --ignored` reported:

| Category | Count |
| --- | ---: |
| Modified tracked paths | 29 |
| Untracked paths | 264 |
| Ignored paths | 123 |

The workspace occupied **147,817,644,032 bytes** (**137.67 GiB**) by `du`.
The material large directories were retained, rather than inferred to be
disposable:

| Area | Size at inventory | Disposition |
| --- | ---: | --- |
| `sua_exploration/checkpoints/` | 59.76 GiB | retained — checkpoints |
| `streaming_calibration_exp/logs/` | 31.99 GiB | retained — runtime evidence |
| `SPINT-main/data/` | 15.21 GiB | retained — dataset |
| `streaming_calibration_exp/outputs/` | 13.12 GiB | retained — experiment output |
| `sua_exploration/data/` | 9.85 GiB | retained — dataset |
| `SPINT-main/logs/` | 4.09 GiB | retained — runtime evidence |
| `sua_exploration/results/` | 2.52 GiB | retained — results |
| `sua_exploration/cache/` | 0.28 GiB | retained — protected SUA state |
| `sua_exploration/evalai_m1_threeway/` | 0.24 GiB | retained — EvalAI material |
| `software-to-hardware/runs/` | 0.11 GiB | retained — run artifacts |

## Deleted reproducible caches

All 32 paths below were ignored by Git before deletion. Sizes are the
pre-deletion allocated sizes in KiB; their total is 3,584 KiB.

| Path | KiB |
| --- | ---: |
| `.pytest_cache/` | 168 |
| `SPINT-main/.pytest_cache/` | 36 |
| `SPINT-main/src/__pycache__/` | 32 |
| `SPINT-main/src/callbacks/__pycache__/` | 28 |
| `SPINT-main/src/data/__pycache__/` | 72 |
| `SPINT-main/src/models/__pycache__/` | 52 |
| `SPINT-main/src/models/components/__pycache__/` | 48 |
| `SPINT-main/src/utils/__pycache__/` | 64 |
| `SPINT-main/tests/__pycache__/` | 20 |
| `SPINT-main/third_party/__pycache__/` | 12 |
| `SPINT-main/third_party/catalyst/__pycache__/` | 20 |
| `SPINT-main/third_party/falcon_challenge/__pycache__/` | 52 |
| `docs_archive/__pycache__/` | 12 |
| `planB_tempconv/__pycache__/` | 28 |
| `planB_tempconv/models/__pycache__/` | 28 |
| `planB_tempconv/scripts/__pycache__/` | 20 |
| `rtl_handoff/golden/__pycache__/` | 48 |
| `rtl_handoff/tools/__pycache__/` | 8 |
| `software-to-hardware/__pycache__/` | 364 |
| `streaming_calibration_exp/.pytest_cache/` | 80 |
| `streaming_calibration_exp/scripts/__pycache__/` | 60 |
| `streaming_calibration_exp/src/__pycache__/` | 48 |
| `streaming_calibration_exp/src/callbacks/__pycache__/` | 20 |
| `streaming_calibration_exp/src/data/__pycache__/` | 136 |
| `streaming_calibration_exp/src/metrics/__pycache__/` | 136 |
| `streaming_calibration_exp/src/models/__pycache__/` | 148 |
| `streaming_calibration_exp/src/models/components/__pycache__/` | 492 |
| `streaming_calibration_exp/src/utils/__pycache__/` | 72 |
| `streaming_calibration_exp/tests/__pycache__/` | 1,208 |
| `streaming_calibration_exp/third_party/__pycache__/` | 12 |
| `streaming_calibration_exp/third_party/catalyst/__pycache__/` | 20 |
| `streaming_calibration_exp/third_party/falcon_challenge/__pycache__/` | 40 |

Post-deletion verification found no remaining cache directories in the eligible
area. During the collaborative audit, a root `.pytest_cache/` containing only
pytest sentinel files was regenerated (32 KiB), as were three eligible
`streaming_calibration_exp/src/` bytecode directories (236 KiB in total).
After confirming that no repository pytest or training process was running,
those regenerated caches were removed as well. The deletion neither alters
source state nor prevents the directories from being recreated automatically by
Python or pytest.

## Explicitly retained cache-like state

Even though these paths look regenerable, they reside in SUA/EvalAI or active
experiment areas and were intentionally not touched:

| Protected path | KiB |
| --- | ---: |
| `sua_exploration/src/data/__pycache__/` | 16 |
| `sua_exploration/evalai_m1_threeway/__pycache__/` | 40 |
| `sua_exploration/evalai_t4_m2/__pycache__/` | 40 |
| `sua_exploration/.pytest_cache/` | 44 |
| `sua_exploration/mc_maze/__pycache__/` | 308 |
| `sua_exploration/tests/__pycache__/` | 1,380 |
| `sua_exploration/scripts/__pycache__/` | 2,072 |

This is a protection decision, not a statement that the contents are
non-regenerable. In particular, no path in the active T4 M30 Experiment A v3
scripts/tests/results/docs was changed.

## `.gitignore` audit

No update is needed. Existing rules already precisely cover the cache types
cleaned here: `**/__pycache__/` (line 21), `*.py[cod]` (line 22), and
`.pytest_cache/` (line 24). Existing rules also segregate local datasets,
runtime logs, results, checkpoints, and model artifacts. Adding broad rules
for source, protocol, manifests, receipts, or result documentation would be
unsafe and was deliberately avoided. The pre-existing modified `.gitignore`
was not changed by this pass.

## Space measurement

| Measurement | Before deletion | Immediately after deletion | Change |
| --- | ---: | ---: | ---: |
| Workspace `du -sx -B1` | 147,817,644,032 bytes | 147,813,974,016 bytes | -3,670,016 bytes |
| Workspace `du -sx -k` | 144,353,168 KiB | 144,349,584 KiB | -3,584 KiB |

At audit close, after the report was written and the regenerated root cache was
removed, the workspace measured **147,814,010,880 bytes** (144,349,620 KiB).
That is an exact net decrease of **3,633,152 bytes** (3,548 KiB) from the
initial inventory. It is 36 KiB higher than the immediate post-deletion
measurement because other collaborative working-tree writes occurred while the
new audit report was being added. The direct cache-only release remains the
separately measured 3,670,016 bytes above.

## Closing Git snapshot

`git status --short --ignored` at close reported 29 modified tracked paths,
268 untracked paths, and 91 ignored paths. This report is one of the untracked
paths. The counts are a snapshot of a shared, actively changing working tree;
no pre-existing modified or untracked file was changed by this hygiene pass.

## Suggested follow-up commits

No commit was created. When the active work is ready, use focused logical
commits rather than staging the whole dirty tree:

1. Review and commit the independently intended `.gitignore` change separately
   from code or experiment work.
2. Split SPINT-main and streaming-calibration source/config/test changes by
   coherent feature or validation contract; use `git add -p` to avoid mixing
   unrelated edits.
3. Keep SUA protocol/docs work distinct from executable implementation work,
   and preserve the active Experiment A v3 branch of work for its owning agent.
4. Add this audit as a small documentation-only commit only after confirming it
   does not overlap the active documentation work. Do not stage ignored
   datasets, outputs, checkpoints, EvalAI artifacts, receipts, or results by
   accident.
