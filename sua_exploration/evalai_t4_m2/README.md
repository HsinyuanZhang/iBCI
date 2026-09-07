# E8 T4 M2 EvalAI candidate

This directory is an isolated, M2-only submission build. It must not import
changes from the concurrent M1 workstream.

The frozen candidate is the canonical seed-42 model from
`m2_spint_t4_mainline_fp32_v1`, selected at epoch 2 by held-in minival R2. Its
checkpoint SHA-256 is
`25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`.
It uses the chronological first 33 calibration trials and the train-only T4
normalization sealed in that run's split manifest.

The exporter reduces calibration to one `E[N,50]` identity tensor per known M2
session. The online runtime therefore executes the exact coupled decoder path
with cached identity, without shipping Lightning or the calibration encoder.
The payload remains a T4 model: each cached identity is produced by the frozen
T4 encoder from that session's first-33 neural trials and calibration target
labels.

All preparation gates below have passed:

1. direct T4, cached-identity, and decoder-only outputs are bit-exact locally;
2. host public minival reproduces the frozen seed-42 held-in artifact;
3. the derived Docker image reproduces host minival and the remote filesystem
   contract;
4. the image/payload/checkpoint hashes and private Test Phase choice are frozen
   in a receipt before any second EvalAI submission is created.

The frozen image is
`spint-t4-m2:e8-seed42-epoch002-dcc449a-r3`, immutable ID
`sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260`.
The exact candidate is positive in a fully disjoint local M33/q33 replay:
`T4-B0=+0.06420` (3/4 eligible sessions positive) and
`T4-TS4=+0.09558` (4/4 positive). This is development evidence, not a hidden
test or a conventional significance result.

See `PREPARATION_RECEIPT.md`, `candidate_m33q33_receipt.json`, and the final
`SUBMISSION_RECEIPT.md`. The guarded helper registered the frozen image exactly
once as private submission `578221`; it must not be executed again.

Official private M2 result: held-out mean R2 `0.30324395`, held-in mean R2
`0.58760827`, normalized latency `0.04290260`. Relative to original SPINT
submission `578218`, the observed deltas are `+0.11676523`, `+0.01936560`, and
`-0.06945843`, respectively.
