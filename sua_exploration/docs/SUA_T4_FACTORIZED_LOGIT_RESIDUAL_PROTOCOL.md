# SUA selected-T4 factorized attention-logit residual protocol

**Frozen:** 2026-08-02 (Asia/Hong_Kong)  
**Status:** seed-42 validation-only causal screen authorized after M1 E4 Gate-A failed. This
protocol does not authorize formal-test access, extra seeds, rank search, confidence/waveform
features, value residuals, backbone unfreezing, or quantization.

## 1. Hypothesis and fixed substrate

Ordinary selected T4 already enters the successful B3S identity `E`, but it may not have a direct,
low-cost route for changing which units the frozen SPINT behavioral queries attend to. The test
retains the complete selected T4 encoder and coupled decoder and adds only

```text
u_i = tanh(W_u T4_i)             # cached at calibration, rank r
bias[j,i] = q_factor[j] dot u_i  # shared over frozen teacher heads
```

to the teacher attention logits. The query factors are exactly zero initialized. The fixed
substrate is seed-42 ordinary T4 checkpoint `epoch_011.ckpt`, SHA-256
`cf533e7cd97801d53985383b37f3fa1ff72fb6c385eaeb8c81273c3fa128273d`, with teacher SHA-256
`9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d`.

## 2. Data and endpoint

- DANDI 000688, subject C, CO, sorted SUA.
- Immutable strict split manifest: 27 train / 6 validation / 6 unopened formal-test sessions,
  units `<100`.
- Activity calibration: chronological rewarded trials `[0,30)`.
- T4 labels/rates: chronological rewarded calibration pool `[0,50)`.
- Evaluation begins at trial 50 for every arm.
- Training is exactly 12 epochs, fixed learning rate `1e-4`, seed 42, no early stopping.
- Score is the unweighted mean validation R² over protocol epochs 5--12. No per-arm checkpoint
  argmax is permitted.
- Formal-test NWBs remain unopened. Validation is development evidence only.

## 3. Frozen arms

The existing `t4_continuation_m50_s42` artifact is the no-new-path control. It has the same selected
anchor, activity/T4 budgets, split, evaluation start, seed, and epoch window.

Run exactly three new arms:

1. **aligned-logit:** aligned T4 drives the new attention-logit residual;
2. **shuffled-logit:** only the rows entering the new residual are deterministically permuted by
   seed 42; the ordinary identity encoder still receives aligned T4;
3. **additive-control:** the same two factor tensors and parameter count produce a per-query
   post-attention additive scalar, without changing attention selection.

All new arms freeze the selected decoder and identity encoder. The optimizer whitelist is exactly
`t4_logit_residual.query_factors` and `t4_logit_residual.unit_projection.weight` (48 parameters at
`r=8`, two output queries). Rank is fixed to 8.

## 4. Mandatory preflight contracts

Before launch:

- zero-init residual decode is bitwise equal to the inherited coupled decode using the same restored
  production substrate;
- cached and on-the-fly paths are bitwise equal;
- aligned and shuffled identities are equal and only residual row attachment differs;
- joint unit permutation leaves the aligned model output invariant;
- aligned and additive arms have identical trainable parameter names/counts;
- checkpoint reconstruction validates teacher, selected-anchor, topology, residual factor, mode,
  rank and shuffle seed;
- no GPU run starts if the strict manifest, anchor SHA, teacher SHA, output freshness or formal-test
  isolation check fails.

At `N=64,r=8,C=2`, the frozen receipt is:

- per-unit persistent state: `64*8=512` values = `2,048 B` FP32 or `512 B` INT8 data;
- calibration-only factor MAC: `64*4*8=2,048`;
- online query-factor MAC: `2*64*8=1,024`;
- shared bias additions across 64 teacher heads: `8,192`;
- no `N^2` neuron attention and no full-width `N*512` residual cache.

## 5. Decision

Primary seed-42 practical gate requires aligned-logit to exceed each of:

- T4 continuation;
- shuffled-logit;
- additive-control;

by at least `+0.03` mean validation R², with positive paired delta on all six validation sessions.
Report paired session deltas, bootstrap intervals, per-epoch scores, factor hashes, and the exact cost
receipt. If any comparison fails, stop the branch: no output-projection unfreeze, rank sweep, extra
seed, confidence input, formal test, or INT8. A pass authorizes only a separately reviewed
three-seed validation confirmation before any formal or quantization work.
