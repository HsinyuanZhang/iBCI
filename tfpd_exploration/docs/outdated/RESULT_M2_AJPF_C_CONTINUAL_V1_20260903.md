# Result — M2 AJPF-C: Continual-Law Matched Joint Training V1

Date: 2026-09-03
Status: **completed; primary tier failed its gate, pre-declared sensitivity tier passed BOTH gates**
Roots: training `results/m2_ajpf_c_v2` (training.json, 0444), score `results/m2_ajpf_c_v4`
(attempt/score/terminal, 0444). Failed predecessors preserved: `m2_ajpf_c_v1`
(unsatisfiable V1 exposure law, nothing trained), `m2_ajpf_c_v3` (runner stage-guard
bug, no science touched).
Workorders: `WORKORDER_M2_AJPF_C_CONTINUAL_V1_20260903.md` + V2 exposure addendum.
Motivating probe: `RESULT_M2_CONTINUAL_CHUNK_PROBE_V1_20260903.md`.

## What was trained

Four modules, paired two-per-law, 12 epochs from the Selected-T4 checkpoint
`25d7bc72…` (strict-load state `2a340745…`), reusing the AJPF V1 machinery by
import (authorization, two-group Adam 1e-4/1e-5, shared governing dropout
mask, packed single-H2D residents, task-only last-bin MSE).  The ONLY
route-owned change: source-training pools are built by the same boundary-free
chunk law as continual deployment — zero trial metadata, seed = first-30
block, D-opt4 selected-support4 carrier, capacity 30, support protected.

| Law | Meaning | Groups/epoch | Coordinates | Wall |
|---|---|---:|---:|---:|
| chunk100 | every tumbling 100-bin window commits | 2,322 | 57,882 | 317 s |
| chunk100e | commits only if mean rate ≥ running median (label-free) | 1,967 | 54,495 | 277 s |

Learned alphas: `C-R1 = -0.109563`, `Ce-R1 = -0.071840` (both left zero;
direction consistent with every prior learned gate).  Sentinel nonzero
encoder/decoder updates passed for all arms; epoch losses converged (≈6e-5).

## Score (13 sessions, official-window surface, deployment law per module)

Equal-session mean R2:

| Module / law | External (6) | Within (7) |
|---|---:|---:|
| pooled / static (sealed anchor; reproduced ≤1e-7) | 0.29099 | 0.67722 |
| C-NAT / chunk100 | 0.28747 | 0.64699 |
| C-R1 / chunk100 | 0.28080 | 0.63818 |
| **Ce-NAT / chunk100e** | **0.37022** | **0.69273** |
| **Ce-R1 / chunk100e** | **0.36953** | **0.69424** |

Pre-registered contrasts, external:

| Contrast | Mean | Pos | Worst | 95% CI | Gate |
|---|---:|---:|---:|---|---|
| **C-R1 − pooled/static (PRIMARY)** | −0.0102 | 2/6 | −0.1008 | [−0.0561, +0.0357] | **FAIL** |
| C-R1 − C-NAT | −0.0067 | 2/6 | −0.0139 | [−0.0119, −0.0010] | — |
| **Ce-NAT − pooled/static (sensitivity)** | **+0.0792** | **6/6** | **+0.0038** | **[+0.0468, +0.1045]** | PASS |
| **Ce-R1 − pooled/static (sensitivity)** | **+0.0785** | **6/6** | **+0.0003** | **[+0.0454, +0.1036]** | **PASS (candidate AND paper)** |
| Ce-R1 − Ce-NAT | +0.0015 | 5/7 within | — | within CI [+0.0003, +0.0027] | — |

## Reading

1. **The primary tier failed.** Training matched to commit-all chunk windows
   does not recover the static baseline (−0.010), and its gate is slightly
   harmful.  Raw windows with idle bins remain a poor activity
   representation even when the model is trained on them.
2. **The sensitivity tier passed everything.** Energy-gated commits
   (label-free median rule, pre-declared before training) plus matched
   training produced **+0.0785 external over the static champion with every
   session positive (worst +0.0003)** — far beyond the growing-pool V4
   effect (+0.032 on its own surface) and beyond anything measured on this
   official-window surface.  The same operator-correction lesson that cost
   PF 0.08 R2 works in reverse: matching train to deploy law transformed the
   probe's 0.228 (AJPF weights under chunk100e) into 0.370.
3. **Tier discipline is binding.** chunk100e was chosen with probe evidence
   that had seen external R2, which is exactly why it was pre-registered as
   the SENSITIVITY tier.  No paper-level continual claim may be made from
   this cell alone; a successor study must pre-register the energy-gated law
   as primary on a fresh surface (e.g. B1/H2, or a new fold).  What this
   cell DOES license without upgrade: **building the official TTA
   deployment candidate** (a deployment is not a claim tier).
4. **Deployment legality.** The chunk100e decoder consumes only the observed
   neural stream (no labels, no boundaries); on the official continual
   contract it is legal test-time adaptation, to be declared
   `IsTestTimeAdaptive=true`.

## Predicted official score

The harness's local→official offset is −0.0013 (act30: 0.2910 → 0.2897).
Predicted official held-out R2 for the Ce-R1 TTA candidate:
**≈ 0.368–0.370** (vs team best 0.295, `act30_full`).

## Follow-up boundary

No further M2 training from this cell.  A submission packaging cell (online
chunk-memory decoder, cached first-30 seed identity, per-100-bin gated
commits, incremental identity recompute) is the natural next step, gated on
the user.  A scientific promotion of the energy-gated law requires a fresh
pre-registered study surface.
