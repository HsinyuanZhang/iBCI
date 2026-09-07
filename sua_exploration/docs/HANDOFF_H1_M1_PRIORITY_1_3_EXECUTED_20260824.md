# Handoff: H1/M1 Priority 1–3 — Executed Results

Date: 2026-08-24  
Status: **TERMINAL COMPLETE AND AUDITED**  
Final terminal: `sua_exploration/results/h1_m1_priority_v1/terminal_v2.json`  
Final terminal SHA-256: `d4c5261cf137e2c676f23e7ac3cf0a175367dc6747e95c4d966f8c6521191d9e`

## 1. Decisions first

1. **M1 M10 DirectRidge is now complete.** It is a valid same-label-budget operational reference, not a carrier mechanism baseline. Equal-session held-in post-M10 R2 is `0.36233`.
2. **H1 per-DoF attribution is now complete on a fully local controlled five-date contrast.** Carrier benefit is strongly anisotropic: `rx` is the only DoF with a clearly positive five-date interval. This diagnostic is `CI64-FULL - CI64-C0`; it is not a reconstruction of the historical `H-C - H-S` mainline.
3. **The H1–M2 shared physical-subspace proposal is stopped.** H1 and M2 do not expose two auditable common physical coordinates. Numerical covariance similarity cannot repair the metadata/semantics failure.
4. **The effective-dimension premise is confirmed but qualified.** M1 nominal 16-D EMG has mean PR `3.934`; H1 strict query nominal 7-D behavior has PR `3.137` and needs four PCs for 90% variance. Neither problem behaves like its nominal output count.

All new work used public held-in-calibration recordings only. No minival, held-out, EvalAI, or formal surface was opened. Target backward steps, optimizer steps, and parameter updates were all zero.

## 2. M1 M10 DirectRidge

### 2.1 Frozen protocol

- Four public held-in-calib sessions: `20120924`, `20120926`, `20120927`, and `20120928`.
- For each target session, the target's first ten chronological trials supply dense paired 20-ms neural and 16-D EMG bins.
- Query begins at trial 11 and uses only eval-valid bins.
- Lag convention: neural `X[t-lag]` predicts EMG `Y[t]`; any pair crossing a trial or the M10 boundary is rejected.
- Frozen lag grid: `0..10` bins (`0..200 ms`).
- Frozen ridge grid: `0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10`, with penalty `lambda * N` on slopes only.
- For each target, lag and lambda are selected using only the other three sessions. Each source-validation session is fit on its own M10 support and scored on its own post-M10 query. Target support and query do not participate in hyperparameter selection.
- The selected affine ridge model is fit once on target M10 support and evaluated once on target post-M10 query.
- Label-cost disclosure: M10 means ten trials containing dense per-bin 16-D EMG, not ten scalar labels.

### 2.2 Result

All four nested folds independently selected `lag=0` and `lambda=0.3`.

| Target session | Post-M10 R2 |
|---|---:|
| 20120924 | 0.37337 |
| 20120926 | 0.36666 |
| 20120927 | 0.35735 |
| 20120928 | 0.35194 |
| **Equal-session mean** | **0.36233** |
| Equal-session median | 0.36200 |
| Pooled-bin R2 | 0.36484 |

This result is stable across sessions but strongly output-dependent. Equal-session mean R2 by EMG output:

| Output | R2 | Output | R2 |
|---|---:|---|---:|
| APL | 0.60569 | BCPs | 0.58940 |
| DLTa | 0.54344 | DLTp | 0.20085 |
| ECRB | 0.48890 | ECU | 0.43233 |
| EDC | 0.53526 | FCR | 0.47268 |
| FCU | 0.39534 | FDI | 0.16996 |
| FDPr | 0.27820 | FDPu | 0.11171 |
| Hypoth | 0.46944 | PECmaj | 0.04409 |
| TCPlat | 0.49144 | Thenar | 0.32721 |

This is useful as an operational floor/reference. It is not directly comparable to the official M1 EvalAI number because the surface, target sessions, and deployment protocol differ.

## 3. Effective output dimension

### 3.1 M1

Post-M10 query covariance, per session:

| Session | Participation ratio | PCs for 90% variance |
|---|---:|---:|
| 20120924 | 3.840 | 6 |
| 20120926 | 3.935 | 6 |
| 20120927 | 4.012 | 6 |
| 20120928 | 3.949 | 6 |
| **Equal-session mean** | **3.934** | **6 in every session** |

The M1 decoder emits 16 named EMG channels, but the behavioral covariance is effectively about four-dimensional. Per-output attribution remains necessary because low PR does not make the individual outputs interchangeable.

### 3.2 H1

Across the exact strict post-M4 query used by the five-date diagnostic:

- nominal outputs: 7;
- participation ratio: `3.13676`;
- variance fractions of the first four PCs: `0.4683, 0.2629, 0.1322, 0.1124`;
- PCs needed for 90% variance: `4`.

Across all 13 held-in-calib recordings using their eval-valid behavior arrays, equal-session mean PR is `3.07659` and median PR is `3.05560`.

H1 is therefore neither a seven-independent-axis problem nor a disguised 2-D problem.

## 4. H1 per-DoF attribution

### 4.1 Why this contrast

The historical five-date `H-C - H-S` receipts retain aggregate R2 but not raw per-DoF predictions, and the original mainline checkpoint locations are no longer complete locally. The complete local controlled contrast is:

- `CI64-FULL`: width-64 model with the frozen M4 carrier;
- `CI64-C0`: the same width-64 graph with carrier clamped to model-bound zero.

Both use seed 42, epoch 49, the exact same source-date LODO protocol, the same first-four-trial support, and identical strict query windows beginning at trial 5. This isolates carrier content within the CI64 family. It must not be relabeled as the historical mainline comparison.

The re-forward reproduced every historical pooled date score. Maximum absolute parity error was `4.94e-14`.

### 4.2 Aggregate and per-DoF result

Pooled over all 74,108 strict query windows:

- `CI64-FULL = 0.42050`;
- `CI64-C0 = 0.38260`;
- pooled difference = `+0.03790`.

The historical equal-date pooled difference is `+0.03798` with 4/5 dates positive.

Per-DoF values below use the five dates as the inference unit:

| DoF | Mean date ΔR2 | Positive dates | 10k bootstrap 95% CI |
|---|---:|---:|---:|
| tx | +0.07345 | 4/5 | [-0.02227, +0.19787] |
| ty | +0.03419 | 4/5 | [-0.01988, +0.08315] |
| tz | +0.03445 | 3/5 | [-0.04410, +0.11300] |
| **rx** | **+0.15197** | **5/5** | **[+0.06401, +0.25327]** |
| g1 | +0.02675 | 4/5 | [-0.01347, +0.06796] |
| g2 | +0.02445 | 3/5 | [-0.01096, +0.06616] |
| g3 | +0.02023 | 3/5 | [-0.04633, +0.08188] |

The carrier effect is not a uniform consequence of “seven dimensions.” It is dominated by `rx`. The grasp dimensions already have relatively strong baseline decodability and gain only about `+0.02–0.03`; the translation dimensions are weakly decoded and their carrier gains are heterogeneous. This supports anisotropic reporting and rejects any story based only on nominal output count.

## 5. H1–M2 shared physical-subspace gate

### 5.1 Metadata facts

H1 NWB authority:

- object: `OpenLoopKinematicsVelocity`;
- axes: `tx, ty, tz, rx, g1, g2, g3`;
- unit: `arbitrary`;
- meaning: human translation, rotation, and grasp coordinates.

M2 NWB authority:

- object: `finger_vel`;
- axes: `index, mrs`;
- unit: `AU`;
- meaning: nonhuman-primate index and MRS finger-group velocities.

There are zero exact name-and-unit axis matches. H1 `tx/ty` are not M2 `index/mrs`; they describe different physical quantities.

### 5.2 Decision

**STOP.** A shared fixed physical output subspace is not identified by these datasets. A learned target/session-specific rotation would use target labels to manufacture the correspondence and would defeat the intended claim. A normalized covariance match is also insufficient because covariance has no semantic axis authority.

This STOP applies to the proposed shared physical-coordinate claim. It does not forbid a multi-task model with separate task-specific output heads or a representation shared before task-specific behavior readout.

## 6. Integrity and receipts

All result bodies and canonical basename sidecars are regular immutable mode-0444 files. The final audit rehashed every body, checked all sidecars, rechecked target-exclusion/parity/STOP assertions, and bound the final implementation bytes.

| Artifact | SHA-256 |
|---|---|
| M1 DirectRidge + output rank | `18f00a42c144ee289cf8652b5357c9ff87c9e49316dc00994bc34467ea1da993` |
| H1 per-DoF | `fa12cd48ad6681b51887e1b4b5872f92e0ade283284c689ac19774c515e90764` |
| H1–M2 subspace gate | `421e4181ffcea2722795adae05ef9a9552ef213683d6de2742d747820756156f` |
| Original terminal | `4ed6eac63e4e598939b5cad1979df53d36b7a78ce2c7f50a24265ee8ea60250a` |
| Independent audit | `8280974af7d7622058fc86b0f356f620d027a91c5bb5ae0575ac36cada6f4144` |
| **Final terminal v2** | **`d4c5261cf137e2c676f23e7ac3cf0a175367dc6747e95c4d966f8c6521191d9e`** |

Implementation closure SHA-256: `175083057da269f2decaba1b40fe42d4005a7fa060cc88912651458137adfbe4`.

Validation:

- focused pure numerical/leakage tests: `6 passed`;
- H1 historical pooled-score parity: PASS, max absolute error `4.94e-14`;
- immutable result and sidecar rehash: PASS;
- M1 target exclusion from hyperparameter selection: PASS;
- H1/M2 semantic gate: STOP as predeclared;
- formal/minival/held-out access: false;
- target backward/optimizer/update count: zero.

## 7. Recommended next use

1. Add DirectRidge to the M1 comparison table with an explicit dense-M10-label-cost footnote.
2. Use the H1 per-DoF table as controlled anisotropy evidence, with the contrast named exactly `CI64-FULL - CI64-C0`.
3. Do not pursue H1–M2 shared physical coordinates unless a new external metadata authority supplies a real coordinate mapping. Do not infer that mapping from target labels.
4. Any later H1 method should report `rx` separately and should not claim that gains scale simply with nominal dimension.
