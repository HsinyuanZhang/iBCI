# TFPD Stage-1 seed-42 matched-pair contract — DRAFT, NOT EXECUTABLE

**Status:** draft for review only. Not an execution authority: no GPU launch, target
scoring, or data access is permitted under this file. It becomes executable only after
(1) Stage-0 receipt `PASS_ALL_GATES` exists, and (2) the operator (user) explicitly
freezes this contract.
**Parent:** `HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md` §8.2–§8.8.

## 1. Cells (seed 42, concurrency gated by GPU availability)

| arm | model | init | carrier input |
|---|---|---|---|
| TFPD-BL-T4 | `BilinearTaskFrameDecoder` | standard | aligned T4 |
| TFPD-BL-Z4 | `BilinearTaskFrameDecoder` | standard (fresh seed offset, same schedule) | `zeros_like` |
| TFPD-PV-T4 | `LearnedPopulationVectorDecoder` | standard | aligned T4 |
| TFPD-PV-Z4 | `LearnedPopulationVectorDecoder` | standard | `zeros_like` |
| SPINT-shape-T4 | SPINT-structured decoder, standard init, task-only joint (no teacher copy) | standard | aligned T4 via additive port |

The standard-initialized SPINT-shaped arm separates **architecture** from **teacher
initialization** (handoff §8.3). The sealed teacher-initialized A2 T4 remains the
historical deployment reference and is scored from receipts, never retrained.

## 2. Shared deployment contract (inherited from A2, read-only reuse)

- strict-27 sub-C source roster / 6-session validation split from the audited A2
  authority bytes (source manifest SHA `4607e979…`, teacher SHA `9b4a94ca…` are
  provenance references only — the teacher checkpoint is NOT loaded);
- M30 chronological support, query strictly after rewarded trial 30;
- strict-27 source-only behavior normalizer (semantic SHA `f062506c…`);
- 12-epoch schedule, task MSE only, no distillation terms;
- within/external scorer on the A2 development surfaces; formal sub-C test sessions
  remain sealed; sub-M is a development external subject only.

## 3. Primary decision rule (frozen before launch)

- **Primary:** external absolute T4 of TFPD-BL-T4 minus SPINT-shape-T4 ≥ +0.03
  (predeclared practical lift), with within-sub-C floor −0.03.
- **Secondary (mechanism, non-rescuing):** TFPD-BL T4−Z4 contrast; TFPD-PV arms
  establish the simple-baseline floor.
- Reporting: sealed A2 T4 external mean is quoted as the historical reference.

## 4. Same-checkpoint diagnostics (no retraining)

Aligned T4 / exact zero / frozen matched-swap carrier / activity-destroyed
(counts replaced by shuffled bins) — the four diagnostics of handoff §8.6.

## 5. Sequencing

1. Stage-0 receipt minted (this package, CPU).
2. Data-adapter smoke: strict-27 loader read-only parity check (row counts, query
   boundaries, normalizer identity) without training.
3. One-cell CPU dry-run of each model on a single source session.
4. GPU cells only after 2–3 pass, on GPUs freed from swap-v2 successor work.
5. Seeds 43/44 only if the seed-42 practical gate passes.

## 6. Explicitly out of scope

Tier-2 closed-form target adaptation; distillation-loss lattices; hypernetwork or
state-space extensions (conditional on the bilinear premise passing); any reopening of
sealed A2/RT/H1 endpoints.
