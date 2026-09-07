# Work Order: M2 AJPF-C (Continual-law matched joint training) V1

Date: 2026-09-03
Status: pre-registered design; implementation authorized (additive, CPU); GPU0 launch only after no-data tests + user green-light
Owner: this session's route (Luna's AJPF roots consumed strictly read-only)

## 1. Question and motivation

The continual-legal probe (`RESULT_M2_CONTINUAL_CHUNK_PROBE_V1_20260903.md`)
established: (a) the anchored gate residual is law-robust (+0.013…+0.019,
6/6 external, CI fully positive under every pool law); (b) boundary-free
chunk rows poison all existing checkpoints because none was trained on them.
AJPF-C closes that gap by making the training pool law *identical* to the
deployment law — the operator-correction principle that recovered 0.08–0.09
R2 on the PF line.

Deployment target: the official continual M2 contract (`reset` +
per-bin `predict`; no boundaries, no labels; test-time-adaptive declared
TRUE on the submission attributes).

## 2. Frozen arms (four modules, one paired run per law)

From the same strict-loaded Selected-T4 checkpoint `25d7bc72…`
(student state `2a340745…`):

| Module | Identity operator | Trainable | Law |
|---|---|---|---|
| `C-NAT` | native early-pool | student encoder + decoder | chunk100 |
| `C-R1` | `h_n + tanh(alpha)(h_p - h_n)`, alpha init IEEE +0 | + 1 scalar | chunk100 |
| `Ce-NAT` | native early-pool | student encoder + decoder | chunk100e |
| `Ce-R1` | anchored gate | + 1 scalar | chunk100e |

`chunk100`: tumbling 100-bin raw windows committed on completion, decode uses
chunks strictly before the endpoint, capacity 30, support4 protected.
`chunk100e`: same geometry; a window commits only if its mean multi-unit rate
≥ the running median of past candidate rates (label-free, causal).
Primary estimand = `C-R1` under chunk100 (zero-hyperparameter law);
chunk100e is the pre-declared sensitivity tier.  No law/arm selection on
external R2 may upgrade a tier.

## 3. Source training contract (adapted from AJPF V1 §5, law-matched)

- Seed pool per session: first-30 calibration block (deployment-identical);
  carrier: D-opt4 selected-support4 ridge T4, frozen.
- **Pool construction uses zero trial metadata**: chunk states derive only
  from raw bins (origin = session bin 0; source training includes early
  windows so deployment-time early-session states are in-distribution).
- Coordinates: all source windows (both surfaces' convention: window start
  ≥ 49) grouped by exact (session, chunk-state digest) identity, batches ≤ 32.
- Budget: seed 42; 12 epochs; batch 32; fresh Adam; encoder/alpha LR 1e-4,
  decoder LR 1e-5; weight decay 0; no scheduler; task-only last-bin MSE
  (scale 5.0); zero teacher forwards.
- Pairing: the two modules of one law share each resident batch, RNG
  restore per module, shared governing dropout mask (explicit-mask
  injection, per AJPF V1 §5.2); the two laws run as separate paired runs.
- `_decoder_frozen` explicitly cleared; parameter groups verified by name
  (`id_encoder.*` incl. alpha @1e-4; `decoder.*` @1e-5; teacher excluded).

## 4. Scoring and gates

Score all four epoch-12 modules plus the no-train anchors on the same
13-session official-window materialization with **their own training law**
as the deployment law (C modules under chunk100, Ce under chunk100e), plus
`pooled/static` sealed anchor reproduction (≤1e-7).

Primary external contrast (chunk100 tier):
`C-R1/chunk100 − pooled/static` — deployment promise: mean ≥ +0.005 and
≥4/6 positive sessions justifies building an official TTA candidate
(declared `IsTestTimeAdaptive=true`); ≥ +0.010 and ≥4/6 and worst ≥ −0.015
qualifies as a paper-level continual result.
Method contrast (secondary): `C-R1 − C-NAT` under matching law.
Sensitivity tier: same contrasts for Ce modules, reported, never promoting.

Honest-failure pre-registration: if C-NAT/chunk100 fails to approach
pooled/static within −0.010, the chunk-law mismatch is not fixable by
matched training at this budget — the continual route closes and B1
(non-continual) remains the growing-memory deployment surface.

## 5. Boundaries

- GPU0 only, after `nvidia-smi` confirms no foreign owner; GPU1 untouched.
- Additive package `tfpd_exploration/src/m2_ajpf_c_v1/` + runner + tests;
  no edits to any sealed/shared production file; AJPF/A0/probe roots
  read-only.
- No-data/no-CUDA tests before any GPU launch; public CLI inert.
- One 12-epoch run per law; no LR/epoch/law retuning after seeing any
  external number; continuation requires a successor note.
- No EvalAI submission from this cell; a submission candidate is a separate
  packaging cell gated on the primary contrast and the user.
