# SPINT published FALCON movement baselines

## Scope and non-comparability rule

This document transcribes **published, private held-out EvalAI** results from Le et al., *SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding*, NeurIPS 2025.  It is not a record of this repository's local public-calibration development results.  Do not merge these numbers with local results in one aggregate table or report a local result as a reproduction of them.

The source is [the official NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/24c8d2fe8520746f0084174b75b93ebe-Paper-Conference.pdf), Table 1 on p. 6.  The paper states that all Table 1 results are mean ± **standard deviation** of R² across held-out sessions (p. 6, Table 1 caption), not standard error.  The benchmark's original paper independently defines OR as an oracle trained with unreleased held-out data and says its held-out table reports standard deviations for held-out data ([Karpowicz et al., NeurIPS 2024, p. 8](https://papers.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf)).

| Method | Class | M1 held-out R² | M2 held-out R² | H1 held-out R² |
|---|---|---:|---:|---:|
| Wiener Filter (WF) | OR | 0.53 ± 0.04 | 0.26 ± 0.03 | 0.21 ± 0.04 |
| RNN | OR | 0.75 ± 0.05 | 0.56 ± 0.04 | 0.44 ± 0.13 |
| NDT2 Multi | OR | 0.78 ± 0.04 | 0.58 ± 0.04 | 0.63 ± 0.08 |
| NDT2 Multi | FSS | 0.59 ± 0.07 | 0.43 ± 0.08 | 0.52 ± 0.04 |
| WF | ZS | 0.34 ± 0.06 | 0.06 ± 0.04 | 0.16 ± 0.03 |
| RNN | ZS | −0.60 ± 0.45 | −0.07 ± 0.23 | 0.09 ± 0.18 |
| CycleGAN + WF | FSU | 0.43 ± 0.04 | 0.22 ± 0.06 | 0.12 ± 0.06 |
| NoMAD + WF | FSU | 0.49 ± 0.03 | 0.20 ± 0.10 | 0.13 ± 0.10 |
| SPINT | GF-FSU | 0.66 ± 0.07 | 0.26 ± 0.13 | 0.29 ± 0.15 |

## Protocol interpretation under this project's definition

Here, **gradient-free** means *no target-day reverse-mode/backpropagation*.  It does not prohibit a target-day closed-form fit, SVD, CCA, or another non-backprop calibration operation.  `Target labels` means behavior labels from the target day.  `Source gradients` do not disqualify a method: source supervised training is allowed.

| Published method | Target-day neural calibration | Target labels | Target-day backprop / trainable update | Closed-form target calibration | Source labels / gradients | Evidence and use |
|---|---|---:|---:|---:|---|---|
| SPINT (GF-FSU) | few unlabeled trials | No | No | No; ID inference is a forward pass | labels and end-to-end MSE training / yes | SPINT p. 5 §3.5 says target uses few unlabeled trials and no gradient-descent updates; p. 4 says IDEncoder and cross-attention are end-to-end MSE-trained. **Direct main-table comparator.** |
| WF ZS | none | No | No | N/A on target day | source labels / no backprop: ridge | SPINT p. 6 says WF is fit on one held-in session then evaluated zero-shot. FALCON p. 31–32 gives the ridge matrix solution. **Direct main-table comparator; stricter target-data condition than GF-FSU.** |
| RNN ZS | none | No | No | No target fit | source labels / yes, supervised 1-layer LSTM training | SPINT p. 6 calls it a simple LSTM, fit on one held-in session and evaluated zero-shot; FALCON p. 32 calls it a supervised PyTorch baseline. **Direct main-table comparator; stricter target-data condition.** |
| CycleGAN + WF (FSU) | target-day neural calibration | No for its alignment objective | Yes; target-day generator/discriminator training | No | source day-0 decoder labels; GAN training | SPINT p. 6 says it trains a GAN on day K to transform activity to day 0. **Unlabeled-target but not gradient-free; separate comparison group.** |
| NoMAD + WF (FSU) | target-day neural calibration | No | Yes; target-day alignment network training | No | source day-0 labels for decoder; dynamics/alignment training | SPINT p. 6 says a day-K alignment network is trained. FALCON p. 34 states only neural data are available on day K, then the feedforward alignment network is trained. **Unlabeled-target but not gradient-free; separate group.** |
| NDT2 Multi (FSS) | target calibration | Yes | Yes; supervised model training | No | labels / gradient training | SPINT p. 6: held-in plus held-out few-shot calibration “with supervision.” **Only a label-assisted reference.** |
| WF OR | unreleased/private target data | Yes | No reverse-mode; ridge fit | Yes | labels / closed-form ridge | SPINT p. 6: private held-out labelled data upper bound. FALCON p. 6 says held-out oracle uses calibration plus redacted data; p. 31–32 specifies the ridge matrix solution. **Oracle upper bound only.** |
| RNN OR | unreleased/private target data | Yes | Yes; supervised LSTM trained on target-session oracle data | No | labels / gradient training | SPINT p. 6 defines OR as private held-out labelled-data training; FALCON p. 6 defines held-out oracle data use and p. 32 gives the 1-layer supervised LSTM. **Oracle upper bound only.** |
| NDT2 Multi OR | unreleased/private target data | Yes | Yes; neural-network training | No | labels / gradient training | SPINT p. 6 defines OR; FALCON p. 33 says this model is trained with held-in calibration, held-out calibration, and held-out redacted data. **Oracle upper bound only.** |

## Naming and definitions verified

- **OR means oracle**, not “Oracle Recalibration.”  SPINT p. 6: WF, RNN, and NDT2 OR use private held-out labelled data as upper bounds.  FALCON p. 8 expands it as “oracle models trained with unreleased data on held-out split.”
- **ZS means zero-shot/static.**  SPINT p. 6 says WF and RNN were fit on a single held-in session and evaluated zero-shot on held-out sessions.  No target-day labels, fitting, or updates occur.
- **FSU means few-shot unsupervised.**  It permits target-day neural data, but the two published FSU comparators explicitly train target-day alignment networks; they are not gradient-free under this project's definition.
- **FSS means few-shot supervised.**  Target labels are used.
- **CCA is not a published FALCON Table 1 row in SPINT.**  SPINT p. 2 names CCA only as related-work linear alignment.  It supplies neither a FALCON score nor a target-day label/optimization protocol, so CCA must not be represented as a published same-protocol SPINT comparator.  A newly implemented AlignedFA+WF or linear-CORAL+WF should therefore be reported as a new local control, not as a reproduction of a SPINT/CCA number.

## What should and should not be re-run

The above are already published private-evaluation baselines and should be cited rather than re-run locally: SPINT, WF ZS, RNN ZS, WF OR, RNN OR, NDT2 Multi OR/FSS, CycleGAN+WF, and NoMAD+WF.  New local work may report AlignedFA+WF and linear-CORAL+the-same-WF as explicitly labeled controls.  A direct WF paired control is useful for those controls, but must not be labeled “published WF ZS reproduction” unless it exactly recreates the publication protocol and receives official EvalAI evaluation.
