# Result — Continual-Legal Growing-Memory Probe V1 (chunk laws × AJPF checkpoints)

Date: 2026-09-03
Status: **completed immutable probe; diagnostic evidence for the continual-deployment route**
Receipt: `tfpd_exploration/results/m2_continual_chunk_probe_v1/replay.json` (0444 + sidecar, 117 rows)
Checkpoints: read-only strict-load of the AJPF V2 epoch-12 bodies (`40c46a95…`, `fd7beac0…`, states `7b5be4f6…`/`47736531…`, SHA-verified). AJPF/A0 result roots untouched.
Lineage anchor: `pooled/static` reproduces the sealed `act30_dopt4` rows at max |ΔR2| = `8.87e-08` (13/13).

## Question

Can the growing-memory effect survive under the official continual M2 contract
(`reset` + per-bin `predict`, no boundaries, no labels)?  The sealed A0 chunk
experiment showed raw-window commits poison the *pretrained* model; this probe
asks whether the *jointly trained* AJPF checkpoints (co-adapted to growing
trial pools) tolerate boundary-free chunk commits, and whether label-free
energy gating of commits helps.

Laws (identical first-30 seed + D-opt4 selected-support4 carrier, capacity 30,
support4 protected):

- `static` — no commits (the scored act30 deployment);
- `chunk100` — A0 geometry: tumbling 100-bin windows of the observed raw
  stream (idle bins included), committed on completion, decode uses chunks
  strictly before the endpoint;
- `chunk100e` — energy gate: a window commits only if its mean multi-unit
  rate ≥ the running median of candidate rates (label-free, causal).

## Results (equal-session mean R2)

| Arm / law | External (6) | Within (7) |
|---|---:|---:|
| pooled / static (anchor) | **0.29099** | **0.67722** |
| pooled / chunk100 | 0.20524 | 0.31075 |
| pooled / chunk100e | 0.26217 | 0.37564 |
| J-NATIVE / static | 0.25791 | 0.65214 |
| J-NATIVE / chunk100e | 0.20896 | 0.36182 |
| J-R1 / static | 0.27437 | 0.66378 |
| J-R1 / chunk100 | 0.15813 | 0.34769 |
| J-R1 / chunk100e | 0.22801 | 0.37494 |

Key contrasts, external:

| Contrast | Mean | Pos | 95% CI |
|---|---:|---:|---|
| **J-R1 − J-NATIVE under chunk100** | **+0.0158** | **6/6** | [+0.0076, +0.0235] |
| **J-R1 − J-NATIVE under chunk100e** | **+0.0191** | **6/6** | [+0.0097, +0.0275] |
| J-R1 − J-NATIVE under static (reference) | +0.0165 | 6/6 | [+0.0128, +0.0206] |
| pooled/chunk100 − pooled/static | −0.0858 | 0/6 | fully negative |
| pooled/chunk100e − pooled/static | −0.0288 | 0/6 | fully negative |
| J-R1/chunk100e − pooled/static | −0.0630 | 1/6 | fully negative |

## Findings

1. **The anchored gate is law-robust.** J-R1 beats its matched J-NATIVE
   control by +0.013…+0.019 with 6/6 (external) and 6–7/7 (within) positive
   sessions and fully positive bootstrap CIs under *every* pool law measured —
   including badly poisoned chunk pools. The gate extracts its residual
   regardless of pool composition. This is the strongest robustness statement
   in the AJPF line so far.
2. **Chunk rows poison every model — jointly trained ones included.** Raw
   100-bin windows (with inter-trial idle, no time normalization) cost the
   pretrained model −0.086 and the jointly trained J-R1 even more (−0.116
   under chunk100). The AJPF co-adaptation was to *trial-shaped* rows; window
   rows are out-of-distribution for it too.
3. **Label-free energy gating recovers ~2/3 of the poison** (pooled:
   0.205 → 0.262, still −0.029 below static). Idle bins are the largest
   single poison source, but misalignment / time-normalization mismatch
   remains.
4. **Deployment verdict without retraining: negative.** The best
   continual-legal combination measured (J-R1/chunk100e, 0.2280) is far below
   the static anchor (0.2910). No submission is justified from existing
   checkpoints under a growing law.

## Consequence (pre-registered direction)

The poison is a train/deploy law mismatch, the same failure class that cost
the PF line 0.08–0.09 R2 before its operator correction.  The principled fix
is **AJPF-C**: joint fine-tuning from the same Selected-T4 checkpoint in which
the *source* training pools are built by the same boundary-free chunk law as
deployment (no trial metadata at all in training pool construction), so the
train graph equals the deploy graph.  Primary law pre-registered as
`chunk100` commit-all (zero hyperparameters, the A0-frozen geometry);
`chunk100e` is the pre-declared sensitivity arm, trained separately — arm or
law selection on external R2 is not permitted to upgrade any claim beyond its
pre-registered tier.
