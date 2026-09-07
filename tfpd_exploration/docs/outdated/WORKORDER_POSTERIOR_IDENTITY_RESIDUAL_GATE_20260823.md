# PIRG: Posterior Identity Residual Gate

## Objective

`POSTERIOR_IDENTITY_RESIDUAL_GATE_V1` is a performance-first, source-only
successor to the negative Posterior Carrier consumer result. It keeps the
sealed Cell-D system and asks one narrow question: can posterior directional
precision regulate how much the existing calibrated identity residual is
used, without replacing the successful OLS point carrier or live activity
path?

This is not teacher training, distillation, posterior-carrier retraining, or
a replacement decoder. Cell-D is the frozen starting system. The only live
trainable quantity is one scalar gate amplitude.

## Held system

- seed 42 sealed Cell-D final-four SWA:
  `626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd`;
- ordinary source OLS T4 normalizer and OLS point M30 carrier;
- B3S M30 calibration activity path, coupled decoder, and all Cell-D shapes;
- existing U(0,1) dynamic whole-unit dropout law during source training;
- strict-27 source roster, B32, dense valid-bin supervised MSE, Adam
  `lr=1e-4` without weight decay, and no target optimizer/backward/update.

No posterior mean carrier, empirical-Bayes replacement, posterior normalizer,
posterior sampling, attention-logit bias, activity gate, architecture change,
or extra ablation arm is permitted in this first cell.

## Sole intervention

For a source session and the M4/M10/M30 prefix chosen by the deterministic
three-epoch rotation, use only its already fitted posterior directional
credibility `r` to form:

```text
z_i = clamp(log(r_i) - mean_j log(r_j), -4, 4)
g_i = 1 + 0.5 * tanh(alpha) * tanh(z_i)
```

`alpha` is one scalar `float32` parameter initialized exactly to zero. The
model computes the ordinary Cell-D identity from the unchanged normalized OLS
point T4, then calls:

```text
decode_with_identity(neural, identity * g.unsqueeze(-1))
```

The live neural activity tensor is passed unchanged. At `alpha=0`, `g` is
bitwise all ones and the one decoder call must be bitwise Cell-D-equivalent;
the derivative with respect to `alpha` remains nonzero for a nonuniform
credibility fixture. Equal credibility makes `z` exactly zero and cancels
the gate at every alpha. The gate is bounded in `[0.5, 1.5]`.

## Source training and cache contract

Three logical epochs use `M(epoch, session_index) = (4,10,30)[(epoch +
session_index) mod 3]`. Thus every strict source session consumes each
prefix budget once. The existing audited posterior fits may build their
27×3 source bank before optimizer batches. PIRG caches only
session×logical-epoch `(budget, credibility, z, gate-control evidence)` at
the epoch boundary. No inverse, posterior fit, posterior mean, normalizer,
or sampling may happen in the B32 optimizer loop.

The source receipt records three epoch rows: the exact rotation, alpha before
and after, gate min/mean/max, finite loss, only-alpha gradient/update proof,
throughput, and cache counters. A final alpha artifact and CPU reload/digest
replace the 48-epoch/SWA machinery. Attempt, source authority, per-epoch,
final artifact, terminal, and honest failure pairs remain immutable.

## Quick matched evaluation scaffold

After a reviewed successful source terminal, the additive score route compares
frozen Cell-D against trained PIRG on the fixed V3 3-within plus 3-external
sessions, same materialized inputs, last-bin variance-weighted R², and
equal-session aggregation. It evaluates M30, M10, and M4 only; both models
are eval/no-dropout/no-gradient and no target state changes.

The physical scorer is not mock-only. It composes the reviewed V3
`QuickScreenPhysicalBackend` for the fixed 3+3 held-FD parser, session batches,
M30 calibration, selected target arrays, ordinary `point_side[M]`, and device
attestation from the immutable V3 stage; PIRG does not recreate those paths
under its fresh source stage. It then strict-loads a second frozen Cell-D copy
plus the PIRG final-alpha artifact and runs only the new identity-residual
forward. For every budget it reads `point_side[M]` and *only*
`posterior_view[M].credibility`; it does not read posterior mean, normalized
posterior carrier, or a sample. All twelve cells are newly forwarded—V3 score
rows are provenance/input replay only, never reused as PIRG results.

The V3 parser may internally construct a posterior view in order to expose its
same-prefix directional precision. That construction is recorded honestly as
substrate provenance; PIRG's consumer algebra reads only `credibility` and its
ordinary OLS point-side identity input never receives `raw_t4`, normalized
posterior T4, a sample, or a logit bias.

The screen is descriptive, not formal:

- M30 within and external safety: each mean delta must be at least `-0.02`.
- Short-prefix promise: pooled M4 mean delta at least `+0.03` and at least
  `4/6` positive paired sessions.

No extra controls or post-hoc arm selection may rescue this screen.

## Engineering boundary

Public CLIs are static/dry and import no Torch. Real source/CUDA/remote work
requires a future in-process root-reviewed capability. The physical adapter
must compose the audited strict-27 posterior source adapter for credibility
only and reuse the V3 Torch-only 5070Ti surface; it may not monkeypatch
globals, copy source NWBs, or rewrite existing routes. This workorder
authorizes no staging, source access, checkpoint read, GPU initialization, or
launch before independent review. A future source stage may carry the five
literal source-adapter authority leaves (not the source NWBs) and exactly
three sealed Cell-D initialization pairs: the Cell-D terminal receipt, its
final-four SWA, and the `D_swa.governing_last_bin` baseline receipt, each with
its canonical `0444` SHA sidecar. Their paths, body SHA-256 values, modes, and
sidecar bytes are explicit in the PIRG identity and staging table. The matched
Cell-D loader descriptor-validates those bytes and their terminal/SWA/baseline
semantics before the PIRG route opens source data or initializes CUDA.

The source train and the later evaluator have separate explicit closures. The
source closure contains only the strict-27 adapter, Cell-D, and one-scalar
training route; it excludes V3 evaluation code and held target assets. The
broader score closure adds the exact V3/attribution composition seam and its
fixed evaluation authority leaves. A source terminal is validated against the
first closure; a future scoring attempt is validated against the second before
it may resolve evaluation input. This keeps the score route runnable without
making source training depend on target-facing machinery.

After a source terminal is accepted, the score closure is staged as an
additive code/metadata overlay on the already reviewed source stage. It must
not replace any source-closure byte, copy the final-alpha result, or copy or
re-stage evaluation NWBs. The scorer reads the completed source artifact at
that stage and composes V3's immutable held evaluation assets in place.
