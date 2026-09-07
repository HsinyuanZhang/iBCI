# cebra_exploration

Exploration of **CEBRA** (Schneider, Lee & Mathis, *Nature* 617:360-368, 2023) in relation to the
encoding-signature carrier paper (`bci_paper_overleaf/paper_6pp.tex`).

## Read in this order

1. **[`docs/PROJECT_VISION.md`](docs/PROJECT_VISION.md)** — why this project exists, how it is
   partitioned, what it is allowed to conclude, and where it goes next. **Start here.**
2. **[`docs/COORDINATOR_VERIFIED_FINDINGS.md`](docs/COORDINATOR_VERIFIED_FINDINGS.md)** —
   authoritative. Every fact established by *running code*. Several findings overturn claims made in
   the track documents, and **this file takes precedence over them**.
3. [`docs/CODE_INTERFACE.md`](docs/CODE_INTERFACE.md) — the comparator API, the gates, how to run it.
4. [`docs/BACKGROUND_BRIEF.md`](docs/BACKGROUND_BRIEF.md) — the shared brief every agent received:
   paper summary, dataset scope discipline, environment, hard rules.

Track documents (`TRACK_A_*`, `TRACK_B_*`, `TRACK_C_*`) are primary sources. The brainstorms in
particular contain claims later falsified — check the findings before acting on them.

## The question in one table

CEBRA and our method are two answers to the same question: **when the recorded unit set changes,
what absorbs the change?**

| | CEBRA `MultiSessionSolver` | CEBRA `UnifiedSolver` | Ours |
|---|---|---|---|
| absorbs a changed unit set | a per-session encoder, **learned** | one shared model over **concatenated** units | an identity token, **closed-form** |
| aligns sessions | the contrastive objective | the contrastive objective | the token's content |
| weights shared across sessions | **none** (F2b) | one model | the entire decoder |
| **can an unseen session be served?** | only by training a new encoder | **no — input width is fixed to the sum of training sessions** | yes: a linear solve plus one forward pass |

*Corrected after red team.* An earlier version of this table claimed "parameter growth per session:
linear versus zero". That is false against `UnifiedSolver`, whose parameter count does not grow with
session count. The axis that actually separates the methods is **whether a new session can be served
at all** — see `COORDINATOR_VERIFIED_FINDINGS.md` F2b addendum.

## Status, 2026-08-13

| Track | Question | Outcome |
|---|---|---|
| **A** | Can we use CEBRA's datasets? | **Closed, negative.** No CEBRA dataset can exercise cross-session calibration on a motor task. Area2_Bump is one monkey and one recording — its `session` argument selects trial type, not a recording day. Rat hippocampus has four subjects with differing unit counts, but it is CA1 place-cell activity on a linear track — the wrong task and species, *not* an absence of behavioural labels (it does expose a direction label). |
| **B** | Can we adapt CEBRA's method as a comparator? | **Skeleton complete and gated; nothing scored.** The original arms were structurally void (F8) and were rebuilt around a joint multi-session fit plus a frozen-source variant requiring a patch to the vendored source. A latent-recovery positive control is now a required gate with a proven-failing negative control. 22 tests pass. |
| **C** | Can our method borrow CEBRA's ideas? | **Complete.** Two independent brainstorms, mutual review, coordinator audit. The durable output is three ranked explanations of the H1 sparse failure (F11, F13, F14) — the strongest emerged from the audit, not from either brainstorm. |

**No citable number exists.** Every arm is unrun on real data.

## Environment

CEBRA 0.6.1 is **vendored** at `third_party/cebra/` (provenance and our patch in
`third_party/CEBRA_PROVENANCE.txt`) rather than pip-installed, so the architecture can be modified.
It runs on the existing `spint` conda env — **no new env**.

```bash
source cebra_exploration/scripts/cebra_env.sh
CUDA_VISIBLE_DEVICES= "$CEBRA_PY" -s scripts/run_cebra_comparator.py --dry-run
```

`PYTHONNOUSERSITE=1` is mandatory: a user-site `torch 2.12.0+cu130` shadows the env and reports
`cuda False` (too new for driver 535.309.01); with user site disabled you get `torch 2.5.1.post303`
with CUDA on 2× RTX 3090. The only dependency added to `spint` was `literate-dataclasses`,
installed `--no-deps`.

## Layout

```
docs/         vision, findings, interface, brief, track outputs
src/          cebra_comparator.py — dataset-agnostic core plus per-dataset adapters
scripts/      cebra_env.sh, run_cebra_comparator.py, probe_*.py (throwaway)
tests/        test_cebra_comparator.py — 22 tests including the positive-control gate
results/      receipts and outputs
third_party/  vendored CEBRA 0.6.1 + provenance
```

## Ground rules

- **CPU only.** The GPUs are contended by other agent sessions.
- **No scoring runs** without coordination. Build, unit-test, audit, report.
- Never modify sealed modules under `sua_exploration/mc_maze/` or `SPINT-main/src/`.
- H1 discovery goes through `h1_sparse_event_endpoint.index_heldin_calib`, never a glob.
- RT is `dandi_000688/sub-C`, **not** `data/000129/sub-Indy`.
