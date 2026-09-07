# Handoff: M1 Calibration-Aware Behavior Bottleneck

## 0. Decision

Keep the calibration-aware behavior bottleneck for label-scarce closed-form M1 adaptation.

Do not use it as a post-hoc hard projection of a trained full SPINT decoder. That integration failed across three source-only decoder folds.

The publishable result is not "an MLP reduces a high-dimensional target." The useful mechanism is narrower:

> A source-trained behavior bottleneck optimized through the exact short-calibration deployment rule can improve gradient-free target-session adaptation, even when PCA and an ordinary reconstruction autoencoder do not.

The best frozen configuration is a 16 -> 64 -> 8 -> 64 -> 16 MLP with 3,224 parameters. A new target session fits only one closed-form M10 neural-to-latent ridge. It performs zero target backward passes and zero target optimizer steps.

## 1. Frozen protocol

- Dataset/task: M1, 16-dimensional EMG behavior.
- Sessions: the exact four public held-in-calibration sessions.
- Support: chronological M10 only.
- Query: strict post-M10 samples.
- Governing aggregation: variance-weighted R2 within each session, then equal session weighting.
- Outer target session: excluded from all manifold fitting and candidate selection.
- Inner source validation session: excluded from the candidate manifold fit.
- Target adaptation: frozen encoder/decoder plus one affine closed-form ridge.
- Target backward steps: 0.
- Target optimizer steps: 0.
- Formal/held-out surfaces: not opened; all results are development evidence.

DirectRidge is the exact common baseline: equal-session R2 = 0.3623304528.

## 2. Low-cost comparison matrix

| Method | Equal-session R2 | Delta vs DirectRidge | Positive sessions |
|---|---:|---:|---:|
| DirectRidge, lag 0 | 0.362330 | 0 | -- |
| Source-frozen PCA | 0.362477 | +0.000146 | 2/4 |
| Source-frozen predictive PCA | 0.365392 | +0.003062 | 2/4 |
| Reconstruction MLP, hidden 32 | 0.336115 | -0.026215 | 0/4 |
| Reconstruction MLP, hidden 64 | 0.339611 | -0.022719 | 0/4 |
| Calibration-aware MLP, q=4 | 0.426643 | +0.064312 | 4/4 |
| Calibration-aware MLP, q=8, seed 0 | 0.427117 | +0.064787 | 4/4 |
| Calibration-aware MLP, q=12 | 0.427367 | +0.065036 | 4/4 |
| Calibration-aware MLP, q=8, hidden 128 | 0.428224 | +0.065894 | 4/4 |
| Calibration-aware MLP, q=8, seed 1 | 0.433273 | +0.070942 | 4/4 |
| Calibration-aware MLP, q=8, seed 2 | 0.429422 | +0.067091 | 4/4 |
| q=8 three-seed prediction ensemble | **0.434177** | **+0.071847** | **4/4** |

The q=8 ensemble session scores are 0.446973, 0.440282, 0.424966, and 0.424490. Its paired gains are +0.073600, +0.073618, +0.067621, and +0.072550.

Fifteen of sixteen EMG outputs have a positive mean delta. PECmaj is the only mean-negative output (-0.016162; 1/4 sessions positive), so the result is broad but not universal across output channels.

## 3. What generated the gain

The ordinary autoencoder objective only reconstructs source behavior. It loses 0.023 to 0.026 R2, so neither nonlinear capacity nor dimensionality reduction alone explains the positive result.

The calibration-aware objective differentiates through the same operation used at deployment:

1. Encode source M10 behavior into a latent vector.
2. Compute the fixed closed-form ridge projection from source M10 neural activity to that latent vector.
3. Predict source post-M10 query latents.
4. Decode those predicted latents back to raw behavior.
5. Optimize query behavior loss plus a source-only reconstruction regularizer.

PCA is approximately null and predictive PCA is only +0.0031. The large gap between those controls and the calibration-aware MLP is evidence that the deployment-matched objective is the active ingredient.

q=4 already captures most of the benefit. q=12 and hidden width 128 add little. Therefore the evidence does not support increasing width or latent dimension further.

## 4. Full-decoder integration test

The same frozen three-seed projector was applied to three independently trained source-only full SPINT decoders without retraining them.

| Fold | Raw full decoder | Hard projected | Delta | Positive outputs |
|---:|---:|---:|---:|---:|
| 0 | 0.649942 | 0.654646 | +0.004704 | 10/16 |
| 1 | 0.678635 | 0.633244 | -0.045391 | 3/16 |
| 2 | 0.682011 | 0.656828 | -0.025183 | 5/16 |
| Equal-fold mean | **0.670196** | **0.648239** | **-0.021957** | 1/3 folds positive |

An earlier fold-0 epoch-23 checkpoint showed +0.014790 from projection, but the three-fold test demonstrates that this was not a stable system-level gain.

Interpretation: a short-calibration ridge benefits from a strong output-manifold prior because it is underdetermined and noisy. A full SPINT decoder already learns useful output-specific residual structure. A hard 8-dimensional projection removes part of that structure and therefore harms the stronger system.

Do not select a target-tuned blend coefficient to rescue this result. That would use the evaluation target to hide the negative three-fold outcome.

## 5. Recommended use and next experiment

Use the q=8 calibration-aware ensemble when the deployment constraint is M10 plus a closed-form target adapter. It is a compact, teacher-free, target-gradient-free improvement with a clean control chain.

Do not add the hard projector after an existing full decoder.

If a future GPU experiment attempts to bring the idea into the full network, it must be trained end to end and preserve raw-output residual capacity. The preferred design is a dual-objective decoder:

- keep the original 16-dimensional output head and governing raw-behavior loss;
- add an auxiliary q=8 calibration-aware latent head during source training;
- decode the latent head back to 16 dimensions only as an auxiliary consistency/regularization target;
- deploy the unchanged raw 16-dimensional head;
- compare against the exact same full-decoder graph without the auxiliary loss.

This tests whether the manifold is useful as source-training regularization without forcing predictions onto it at inference. It is a new GPU cell, not a low-cost continuation, and should only run if improving the full SPINT decoder is the next priority.

## 6. Evidence files

Primary positive result:

- `sua_exploration/results/behavior_autoencoder_v1/m1_calibration_aware_q8_ensemble.json`
- SHA-256: `836f0af06973813135fd7ff9e5e7a22df0113089d29ab99a7fe4da0f4406eb9d`

Primary full-decoder rejection:

- `sua_exploration/results/behavior_autoencoder_v1/m1_source_decoder_threefold_projection.json`
- SHA-256: `b013539357eaa79389b469d703890f02566ae694f2efa2d7481cb3034955d4f5`

Important controls:

- PCA: `4343c42688e0f5fea7dc8a089816d493782b7a44e0e1408d5fc203403184463f`
- Predictive PCA: `6aecf46dca5bf59ca4d6ccf86f2553050ab4c62be1120defc266dd438fcf4b99`
- Reconstruction MLP h32: `da4e76b75fa7c98ed2edeac8bbe0e68e04b39feaae845d1d3c9006923d493464`
- Reconstruction MLP h64: `ed6c62c17df1989100f73cf359df94fc6df5d4952a9df115fe0eb94fc9b5b664`
- Earlier non-governing fold-0 projection: `9d72783c64df666760f18c867daca3250bd4e7a8818ec4f8bf53f21fb5e2ba3a`

Focused implementation tests: 8 passed. The three-fold projection completed on GPU0 and wrote a fresh result; no target training or checkpoint mutation occurred.
