# EvalAI daily pair — H1 and M2

Date of registration: 2026-09-04 (Asia/Hong_Kong)

Status: `BOTH_FINISHED__H1_READOUT_REMOVAL_POSITIVE__M2_MOVE_T4_POSITIVE`

Both submissions are private, use public calibration data only, and declare
`IsHeldOutZeroShot=false`, `IsTestTimeAdaptive=false`, and
`IsPretrained=false`.

## M2 — movement-window T4

- EvalAI submission: `581899`
- Submitted: `2026-09-04T15:50:17.695285Z`
  (`2026-09-04 23:50:17` Asia/Hong_Kong)
- Candidate: M33 activity identity with T4 estimated from calibration-neural
  bins `[5,30)`, i.e. `[100,600)` ms after each trial boundary
- FiLM: none
- Online adaptation: none; the runtime serves cached `E[N,50]` identities
- Sparse label budget: the same first-33 target-direction labels as the prior
  M33 line
- Payload SHA-256:
  `f3e64950b00193949f6993d1a9022ba98fb65b759198427d666483d0c6bc49c1`
- Image ID:
  `sha256:374426038abfc17785148b17b39b081b09df354c5f44802b756811f8b9c8300e`
- Local external evidence: `0.3254516464` versus whole-trial T4
  `0.2991329173`, delta `+0.0263187291`, positive on `6/6` sessions
- Deployment-payload checks: `13/13` session coverage; cached decoder versus
  direct decoder max absolute difference `0.0`; container prediction finite

Official terminal result (`finished` after the normal two-stage EvalAI host
evaluation):

| Metric | 581899 |
|---|---:|
| Held Out R2 Mean | **0.3275179374** |
| Held Out R2 Std. | 0.1049180338 |
| Held In R2 Mean | **0.5738219776** |
| Held In R2 Std. | 0.0305381066 |
| Normalized Latency | 0.0445166578 |

The first worker snapshot temporarily reported `failed` after both prediction
batches had completed and exposed an empty result array.  It contained no
traceback and matched the known EvalAI two-stage race previously observed for
581658.  The same submission was automatically re-evaluated by the host scorer
and changed to `finished` at `2026-09-04T16:21:59.753907Z`; no replacement was
submitted.  The finished result above is governing.

Relative to submission 581801 (`0.320281` held-out, hold/reach FiLM), MOVE-T4
without FiLM is approximately **+0.00724**.  Relative to the prior static T4
score `0.303244`, it is approximately **+0.02427**.

This submission tests the carrier-correction explanation directly.  Dense
behavior was used only to choose the source-side window geometry; it is not
read by the deployed carrier materializer.

## H1 — EP-FiLM without MAT7 readout

- EvalAI submission: `581900`
- Submitted: `2026-09-04T15:51:27.527702Z`
  (`2026-09-04 23:51:27` Asia/Hong_Kong)
- Candidate: the immutable all-source C1/M3 EP-FiLM state from submission
  `581866`, with every MAT7 output map replaced by the exact 7-D identity map
- Rationale: `581866` reached held-out R2 `0.2675402`, while its additive
  stack included an output readout already implicated in an approximately
  `-0.043` official-chain loss.  This submission removes that readout without
  changing the decoder, activity/carrier/profile payloads, or FiLM state.
- Source payload SHA-256:
  `df71cb9329a87b5073242044b2933391ced7bf866d1f481e994f71dbd97ba2d7`
- Derived payload SHA-256:
  `523d3d2e55a8fd4620a3f0a94dea6a99ae6ff53cefe807c3ff809a30b1d2479d`
- Image ID:
  `sha256:0408012c36a0a15a6a1fc21de4760847d7f64e2ac64139002cf7adc30b06f6b9`
- Payload coverage: `27/27` sessions
- Container smoke: pass; finite output; substrate model and FiLM state hashes
  unchanged before/after inference

Official terminal result (`finished`, execution time `0.126374` s):

| Metric | 581900 |
|---|---:|
| Held Out R2 Mean | **0.2980939037** |
| Held Out R2 Std. | 0.1421943751 |
| Held In R2 Mean | **0.4619134437** |
| Held In R2 Std. | 0.0334382099 |
| Normalized Latency | 0.0359405464 |

Relative to the otherwise identical 581866 package with MAT7, removing the
readout improves official held-out R2 by **+0.0305537146** (`0.2980939037 -
0.2675401891`) and held-in R2 by `+0.0104360821`.  This confirms that MAT7 was
a material negative component of the H1 additive stack.  It does not attribute
the remaining score to profile content; the source-only EMPTY/ROWSHUFFLE
diagnostic continues to govern that mechanism claim.

This is an accuracy/decomposition probe, not evidence that H1 profile content
is useful.  The source-only V5 diagnostic already established that EMPTY and
row-shuffled profiles retain the EP-FiLM gain.

## Registration integrity

The uploaded ECR manifest digests match the complete local Docker image IDs
for both submissions.  Durable push-state files prevent accidental duplicate
registration:

- `tfpd_exploration/submissions/evalai_m2_movement_t4_v1/artifacts/evalai_push_state.json`
- `tfpd_exploration/submissions/evalai_h1_epfilm_no_readout_v1/artifacts/evalai_push_state.json`

Both terminal result artifacts have been captured under their respective
submission directories.
