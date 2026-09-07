# H1 frozen carrier method binding — 2026-09-08

## Decision

The official H1 RIFT submission **582073** is exactly bound to a sealed four-dimensional H-C tensor, but the available frozen artifacts do **not** prove that this tensor was constructed by the `q=16`, `lambda_H=100`, PCA/whitening, output-SVD, empirical-Bayes-shrinkage formula currently stated in `bci_paper_overleaf/paper_4pp.tex`. The paper must not attribute that full formula to submission 582073 without a missing construction-level provenance record.

This is not evidence against the formula or the final bank arrays. It is a provenance boundary: the frozen deployment payload contains final arrays, while the audited chain lacks the source-solver receipt needed to bind its estimator parameters to those arrays.

## Frozen submission chain

The freeze manifest identifies submission 582073 as H1 RIFT R300 recency EMA-22, cached CPU, context 300, seed 42, `proj_add`, and payload SHA `7957ce7b52596745aab55e5955e8763cab3b2f4808c25a037688890728883b56`.

`pack_and_verify.py::_build_payload` loads the sealed 582044 B2 payload, assigns `banks = bt["bank_by_dataset_tag"]`, and writes those same banks into the RIFT payload. It does not call an H-C estimator. Thus the correct upstream bank authority is the C2-CAL-1 B2 payload, not any later B2 diagnostic.

The RIFT decoder converts each payload row into `TaskBank(E0, T, unit_mask)`. `E0` is `[176,700]`; `T` is the direct `[176,4]` H-C carrier; all 27 official tags are required. The upstream B2 receipt records every tag's array hashes. For example, `S0_set_1` has E0 SHA `5875f0c64a4e3be2cbbb833c63446951b2826e5ee5a0451464bbddfc6572f30c` and H-C SHA `85d4801852554164421218d030404d07d8bcfea7795296e1ae28ffc82d17e2a2`.

## What C2 proves

The frozen C2 materializer consumes a sealed M3 payload containing per-session activity `[3,1024,176]` and already-computed H-C `[176,4]`. It applies a frozen `1024→32` pre-pool with ReLU, averages the three trials, concatenates the 32 activity coordinates and the four supplied H-C coordinates, then applies frozen `36→32→32→700` post-pool layers. It verifies the C2 checkpoint SHA `ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215`.

Consequently, the RIFT bank's E0 includes H-C-derived information, while `T` supplies the same four H-C coordinates through the direct path. This is why direct-carrier ablations hold E0 fixed and cannot establish total H-C removal.

## Paper formula comparison

The current 4-page manuscript describes H1 as source-centering/scaling, `q=16` PCA, ridge `lambda_H=100`, a source right-singular basis `U`, and empirical-Bayes shrinkage. Those parameter claims are visible in the paper, but they are absent from the frozen RIFT payload, B2 payload receipt, C2 M3 payload, and C2 materializer. The materializer has no PCA, ridge, shrinkage, or normalizer solver; it only consumes the precomputed H-C tensor.

Therefore the audited artifacts establish that 582073 consumes a dense, labelled-support-derived H-C carrier, but leave the exact construction parameters of that carrier **unbound**. A similarly named or unrelated B2 implementation cannot fill this gap.

## Minimum evidence to close the gap

A valid closure requires one of the following:

1. A sealed C2 H-C construction receipt/source snapshot that records source mean and scale, PCA basis, output basis, ridge/shrinkage settings, and input/output array hashes that reproduce every one of the 27 H-C payload hashes; or
2. A paper revision that treats the `q=16`, `lambda_H=100` formula as a separately evidenced H1 carrier family rather than the proven estimator behind 582073.

The companion JSON receipt contains all paths, SHA-256 values, and exact status. This audit did not load data, score a model, open EvalAI, or modify payloads, code, paper text, or results.
