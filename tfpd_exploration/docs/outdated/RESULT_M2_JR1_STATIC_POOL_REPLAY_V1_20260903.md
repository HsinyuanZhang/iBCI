# Result — J-R1 Static-Pool Official-Surface Replay V1

Date: 2026-09-03
Status: **completed immutable replay; submission-relevant negative on net, method-positive on contrast**
Receipt: `tfpd_exploration/results/m2_jr1_static_pool_replay_v1/replay.json` (0444 + sidecar)
Provenance: read-only consumption of the AJPF V2 training manifest (checkpoint bodies
`40c46a95…`/`fd7beac0…`, student states `7b5be4f6…`/`47736531…`, both SHA-verified
before strict load); AJPF V4 result roots untouched.

## Question

AJPF V4's positive held-out result (+0.0131 `J-R1 − J-NATIVE`, bootstrap CI fully
positive) was measured under the decode-before-commit growing-pool protocol. The
official EvalAI M2 surface is a static first-30 calibration pool with cached
identities. Would a J-R1 static deployment beat the officially scored static
baselines?

## Method

One CPU process, one 13-session materialization, three arms, identical inputs:

- `pooled` — pretrained Selected-T4 checkpoint `25d7bc72…` (lineage anchor);
- `J-NATIVE` — AJPF epoch-12 jointly fine-tuned native control;
- `J-R1` — AJPF epoch-12 jointly trained anchored gate (learned
  `alpha = -0.1838170737028122`).

Static identity per session under the sealed `act30_dopt4` law (greedy D-opt k=4
support, ridge λ=0.1 selected-support4 T4, first-30 label-free pool), computed
once per session; evaluator-semantics decode over the official window convention
(batch 1, decode-only).

**Lineage anchor:** the `pooled` arm reproduces the sealed `act30_dopt4` screen
rows with max |ΔR2| = `8.87e-08` across all 13 sessions. The replay harness is
the one already validated bitwise by the APFG-S local validation.

## Results

Equal-session mean R2 (external = 6 official-style held-out sessions):

| Arm | External | Within |
|---|---:|---:|
| pooled (sealed anchor 0.2910 / 0.6772) | 0.29099 | 0.67722 |
| J-NATIVE | 0.25791 | 0.65214 |
| J-R1 | 0.27437 | 0.66378 |

Paired contrasts (mean, positive sessions, 95% session-bootstrap CI, seed 42):

| Contrast | External | Within |
|---|---|---|
| **J-R1 − J-NATIVE** | **+0.01647, 6/6, CI [+0.0128, +0.0206]** | +0.01164, 6/7, CI [+0.0070, +0.0156] |
| J-NATIVE − pooled | −0.03309, 1/6, CI [−0.0670, +0.0012] | −0.02508, 2/7 |
| **J-R1 − pooled (net deployed)** | **−0.01662, 2/6, CI [−0.0507, +0.0182]** | −0.01344, 2/7 |

## Reading

1. **The anchored-gate effect survives — and strengthens — on static pools.**
   Unlike the frozen-weight APFG (whose gate effect vanished at pool size 30,
   −0.0008), the jointly adapted J-R1 delivers +0.0165 over its own matched
   control with **6/6 external sessions positive** and a fully positive CI.
   Decoder co-adaptation made the gate pool-robust.
2. **But the joint fine-tuning route itself costs −0.0331 on static pools.**
   Twelve epochs of warm-start adaptation shifted the whole representation
   (trained under M∈{4,10,30} cycling with decode-before-commit) away from the
   static-deployment optimum. The gate recovers +0.0165 of that hole and still
   nets **−0.0166 below the pretrained static champion** (2/6 positive, CI
   crossing zero).
3. **Predicted official score for a J-R1 static candidate: HO ≈ 0.273–0.275**,
   below `act30_dopt4` (0.2897) and `act30_full` (0.295). The replay-to-official
   offset for this harness is −0.0013 (0.2910 → 0.2897).

## Decision implication

Do **not** spend the last rationed EvalAI submission on a J-R1 static
candidate: the pre-submission replay predicts a descriptive loss against both
already-scored static baselines. The V4 method result is real but its home
deployment is the **growing-pool protocol** (where J-R1 beats POOLED outright,
0.3310 vs 0.2991). That protocol is illegal on official M2 (continual, no
trial boundaries) but **officially legal on B1/H2** (`continual=False`,
`on_done` delivered). The natural follow-up is therefore not an M2 submission
but the B1 extension, where V4's deployment surface can be realized inside the
official contract.
