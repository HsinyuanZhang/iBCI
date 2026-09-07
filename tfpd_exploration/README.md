# tfpd_exploration

Task-Frame Population Decoder (TFPD) program — the teacher-free clean-sheet successor
direction defined by
[`../sua_exploration/docs/HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md`](../sua_exploration/docs/HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md).

## Premise

The carrier (T4) is a **condition on the population read-in operator**, not an additive
identity waveform. The first test is an explicit low-rank bilinear activity-carrier read-in;

```
z_t = (1/sqrt(N)) * sum_i (U a_it) (.) (V g(c_i)),    y_hat = D(z)
```

with a learned-population-vector transparent baseline as the strong simple comparator.
No teacher output, no teacher-identity matching, no SPINT algebra. SPINT remains a
comparator and an optional disclosed initializer only.

## Reading order

当前有效文书见 [`docs/README.md`](docs/README.md)。Stage-0 bilinear 合同与更早的
workorder 已移到 [`docs/outdated/`](docs/outdated/)，全部过时。

Current documents live in [`docs/README.md`](docs/README.md). The Stage-0 bilinear
contract and earlier workorders are in [`docs/outdated/`](docs/outdated/) and are
outdated.

1. `docs/HANDOFF_EXPERIMENT_CLOSEOUT_20260906.md` — current close-out.
2. `docs/WORKORDER_M2_B_SMALL_TRANSFORMER_STABILIZATION_INTERIM_V1_20260905.md` — S1-SMALL-COS.
3. `docs/outdated/TFPD_STAGE0_SYNTHETIC_GATE_CONTRACT_20260815.md` — historical Stage-0 contract.

## Stage discipline

- Stage 0 (this package, CPU only): the five frozen gates. No real data, no GPU.
- Stage 1 (separate contract, needs explicit GO): seed-42 matched T4/Z4 source pair on the
  A2 strict-27 roster, standard-initialized SPINT-shaped T4 comparator, learned-PV baseline.
- Tier-2 closed-form target adaptation is a separate deployment claim and never rescues a
  Tier-1 forward-only result.

## Hard rules

- This package must not inherit `StreamingCalibrationLitModule`. Reuse of audited
  data/normalizer/query/scorer infrastructure is read-only, at Stage 1, via imports that
  never mutate sealed modules under `sua_exploration/mc_maze/` or `SPINT-main/src/`.
- Arms are produced by *inputs*, not by architecture flags: aligned carrier, zero carrier
  (`torch.zeros_like`), and frozen matched-swap carrier all feed the same trained weights.
- sub-M is a repeatedly used *development* external subject; no general cross-subject claim
  may be made from it.
- Environment: `spint` conda env with `PYTHONNOUSERSITE=1`; focused tests from this subtree
  with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Layout

```
docs/     frozen Stage-0 contract (and future Stage-1 draft)
src/tfpd/ models + synthetic generator (plain torch, CPU-capable)
scripts/  gate runner / receipt writer
tests/    focused pytest suite mirroring the frozen gates
results/  immutable Stage-0 receipts (ignored by Git)
```
