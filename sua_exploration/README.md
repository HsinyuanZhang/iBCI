# SUA/MUA analytic functional-carrier program

This project studies session calibration for streaming intracortical neural decoding. The selected method combines
a compact B3/B3T activity encoder with a fixed-width analytic functional carrier such as T4. Target-session
calibration is chronological, closed form, and uses no optimizer, backward pass, or decoder weight update.

**Current authoritative handoff:**
[`docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`](docs/HANDOFF_MAINLINE_CLOSURE_20260811.md)

That handoff is the single entry point for scientific conclusions, remaining work, claim boundaries, golden
programs, and evidence locations. Older `HANDOFF_*`, `AGENT_BRIEF_*`, dated reviews, and historical roadmap
sections are provenance only and do not authorize new experiments.

## Current result in one table

| dataset / scope | principal result | status and boundary |
|---|---|---|
| external subject-M SUA | T4/Zero4/TS4 `0.356828/−0.057766/−0.115319`; T4−Zero4 `+0.414594` | terminal, 3/3 seeds and 15/15 session means positive |
| deterministic pseudo-MUA | T4/Zero4/TS4 `0.306073/−0.086878/−0.164314`; T4−Zero4 `+0.392951` | terminal controlled signal-view bridge, not native threshold-crossing MUA |
| native M2 | matched SPINT/T4 `0.293110/0.382906`; delta `+0.089796`, 7/7 sessions positive | matched seed-42 development evidence; fresh three-seed full gate remains unavailable |
| RT dense carrier | Full/B2 `0.441950/0.145148`; delta `+0.296802`, 15/15 folds positive | terminal; B2 is the same-pipeline SPINT-structured reference |
| RT sparse endpoint T4d | at 12/15 complete folds, T4d−Zero4 `+0.294121`, 12/12 positive; T4d−Full `+0.005379` | running early evidence only; final 45-cell verifier and exact-query B2 companion remain |
| H1 | H-C−H-S `+0.056287`; organizer-held `0.274939` versus paper-LR SPINT `0.261492` | separate dense-covariate compact-consumer evidence; not a sparse-label result |
| M1 | Original/T4/D4 `0.648591/0.644766/0.643993`; matched carrier content `−0.00652` | current negative boundary |

The main scientific statement is intentionally narrower than “T4 beats ridge.” The evaluated Ridge50 is more
accurate on external subject-M but consumes dense per-bin velocity targets; T4 uses one direction scalar per
calibration trial. This is an accuracy/supervision-density trade-off, not a universal compute, energy, latency, or
annotation-cost claim.

## Method boundary

The canonical carrier is

```text
T4_i = [a_i, c_i, m_i, b_i]
m_i = sqrt(a_i^2 + c_i^2)
rate_i(theta) ~= b_i + a_i cos(theta) + c_i sin(theta)
```

- B3/B3T supplies the temporal neural-activity representation.
- T4 supplies session-specific functional identity.
- Calibration fits the carrier once and caches it for streaming inference.
- Offline source training and evaluation scoring may use dense behavior targets.
- “No backpropagation” applies to target-session calibration, not offline source training.
- “Sparse labels” applies to the target-session carrier estimator, not the whole pipeline.

## Evidence hierarchy

Use evidence in this order:

1. immutable terminal receipt and independent verifier;
2. terminal aggregate reconstructed from raw per-session or per-fold R²;
3. organizer-held result with explicit comparator and provenance boundary;
4. complete development matrix;
5. partial or pilot evidence, which must be labelled nonterminal.

Do not promote a launch receipt, preflight PASS, source-only constructibility gate, partial fold, or same-checkpoint
diagnostic into an accuracy claim.

## Active closure

Only RT can still change the main paper:

1. finish the fixed 15-fold `T4d/Full/Zero4` matrix;
2. pass the independent 45-cell terminal verifier;
3. forward-evaluate all 15 sealed B2 checkpoints on the exact Stage-2 query windows;
4. update the paper and concise result ledger with terminal-only statistics;
5. clean historical documents/checkpoint aliases, run focused checks, and make path-scoped Git commits.

H1 CI64 is a secondary consumer-width experiment already in flight. It may finish or be handed to another
maintainer, but it cannot delay or reopen the SUA/M2/RT mainline. Quantization is deferred.

## Documentation

| file | role |
|---|---|
| [`docs/HANDOFF_MAINLINE_CLOSURE_20260811.md`](docs/HANDOFF_MAINLINE_CLOSURE_20260811.md) | sole current status and scientific handoff |
| [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) | concise result ledger and receipt pointers |
| [`docs/ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](docs/ACTIVE_EXPERIMENT_CONTROL_BOARD.md) | live processes and terminalization state only |
| [`ROADMAP.md`](ROADMAP.md) | remaining closure sequence and stop rules |
| [`docs/FP32_T4_MAINLINE_PROTOCOL.md`](docs/FP32_T4_MAINLINE_PROTOCOL.md) | selected FP32 method contract |
| [`docs/MEASUREMENT_PROTOCOL_V4.md`](docs/MEASUREMENT_PROTOCOL_V4.md) | uncertainty, pairing, and claim rules |
| [`docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md`](docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md) | frozen RT sparse matrix contract |
| [`docs/RT_SPARSE_T4D_VS_B2_D1024_COMPANION_PROTOCOL_20260810.md`](docs/RT_SPARSE_T4D_VS_B2_D1024_COMPANION_PROTOCOL_20260810.md) | exact-query B2 companion contract |

Historical method proposals remain available only where scripts, tests, or receipts still cite them. They are not
part of the active navigation.

## Repository layout

```text
sua_exploration/
├── README.md
├── ROADMAP.md
├── docs/                    # current contracts plus historical evidence
├── mc_maze/                 # shared encoder and carrier implementation
├── scripts/                 # experiment, aggregation, and verifier entry points
├── tests/                   # focused contract tests
└── results/                 # ignored generated evidence; do not commit

SPINT-main/
├── src/                     # SPINT/H1 training implementation
├── scripts/                 # H1 producers, evaluators, and verifiers
└── pilot_artifacts/         # ignored immutable receipts/checkpoints
```

## Workspace rules

- Never use `git add .` in this multi-agent worktree.
- Never commit checkpoints, raw predictions, logs, `results/`, or `pilot_artifacts/`.
- Do not delete an active run root or a file referenced by a script, test, receipt, current document, or paper.
- Remove duplicate checkpoints only after verifying byte identity and retaining the canonical best checkpoint.
- Every published number must be traceable to a terminal aggregate, receipt, and SHA.
