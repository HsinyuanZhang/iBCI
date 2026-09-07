# T4 M2 EvalAI preparation receipt

**Frozen:** 2026-08-01T16:22:17+08:00  
**Status:** `PREPARATION_FROZEN / SUBSEQUENTLY_SUBMITTED_AS_578221`  
**Challenge:** EvalAI 2319, FALCON  
**Target:** private Test Phase `few-shot-test-2319`, phase ID `4599`  
**Task:** native FALCON M2  
**Formal quota consumed during preparation:** none  
**Subsequent outcome:** private submission `578221` finished positive; see
`SUBMISSION_RECEIPT.md`.

## 1. Decision

The canonical all-held-in seed-42 T4 candidate passed preparation and was later
submitted exactly once as private submission `578221`. It finished with
official held-out mean R2 `0.30324395`, versus `0.18647872` for original SPINT
submission `578218`. This document preserves the pre-submission freeze; the
official outcome is recorded in `SUBMISSION_RECEIPT.md`.

The candidate is worth submitting because its exact checkpoint, replayed with
chronological support `[0:33]` and query beginning at trial 33, has:

| Arm | Four eligible held-out sessions, mean R2 |
|---|---:|
| matched epoch-34 B0 | 0.23142761 |
| T4 seed 42 | **0.29563015** |
| TS4 seed 42 | 0.20004525 |

- `T4-B0 = +0.06420254`, positive on 3/4 eligible sessions;
- `T4-TS4 = +0.09558490`, positive on 4/4 eligible sessions.

This is positive development evidence, not a significance result. Only four
local sessions contain query trials after a 33-trial prefix; the two 2020-11-24
files contain exactly 33 trials and were correctly marked zero-query rather than
assigned a fabricated score. The machine-readable pairing receipt is
`candidate_m33q33_receipt.json` (SHA-256
`37e15bd09af2b5cd83725423510de40444291ceac8d175e78290c7a642f1114d`).

The old historical M33 value that scored from trial 0 remains withdrawn. It is
not used anywhere in this preparation decision.

## 2. Isolation from concurrent M1 work

All new submission source files are under `sua_exploration/evalai_t4_m2/`.
The exact-candidate replay wrote new M2-only artifacts named
`e8_t4_m2_submission_candidate_m33q33_v1_*` and
`e8_ts4_m2_submission_control_m33q33_v1_*`.

This preparation did not edit the concurrently dirty M1/shared files, including:

- `SPINT-main/configs/train.yaml`;
- `SPINT-main/src/data/falcon_datamodule.py`;
- `SPINT-main/src/models/falcon_module.py`;
- `SPINT-main/src/train.py`;
- `SPINT-main/third_party/falcon_challenge/spint_sample.Dockerfile`.

The Docker build derives from the already validated immutable base image rather
than rebuilding from those working-tree files. Every replay and container audit
in this preparation used CPU explicitly; neither RTX 3090 was allocated.

## 3. Frozen candidate provenance

| Item | Frozen value |
|---|---|
| source screen | `m2_spint_t4_mainline_fp32_v1` |
| arm / seed | all-held-in T4 / 42 |
| selected epoch | 2 |
| selection metric | held-in minival R2 only; no hidden or held-out query score |
| checkpoint | `streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt` |
| checkpoint SHA-256 | `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e` |
| frozen decoder teacher | M2 epoch 34 |
| teacher SHA-256 | `fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec` |
| variant | B3S, coupled frozen SPINT decoder, `side_dim=4` |
| training sessions | all seven M2 held-in sessions |
| calibration support | chronological first 33 trials per session |
| target-label access | one target direction per calibration trial |
| calibration backward/optimizer | none |
| T4 normalization | fit only from the seven held-in training sessions |
| normalization SHA-256 | `d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e` |

The seed-42 candidate was selected canonically, not by choosing the best hidden
or held-out seed. Hidden EvalAI query data have not been read.

## 4. Calibration and deployment mechanism

For each of the 13 known M2 calibration sessions, the exporter computes T4
features from the first 33 public calibration trials and their public target
directions, runs the frozen T4 identity encoder, and stores one identity tensor
`E` with shape `(96, 50)`.

At EvalAI runtime the online path is:

```text
neural window [B,50,96]
  -> transpose and add cached session identity E[96,50]
  -> frozen SPINT read-in / cross-attention / read-out
  -> 2-D behavior
```

There is no optimizer, backward pass, T4 refit, calibration-trial buffer, or
raw target-label state inside the online runtime. It stores only the cached
identity for the active session. Consequently the reported online latency will
not include the one-time offline T4 fit/export cost; no end-to-end calibration
latency claim should be made from the EvalAI normalized-latency field.

This is not zero-shot: T4 uses labeled calibration data from the held-out days.
It is also not runtime test-time adaptation under EvalAI's literal metadata
definition, because no recalibration runs in the evaluation container.

## 5. Export and numerical-equivalence gates

The payload contains the frozen decoder plus 13 cached identities:

| Item | Receipt |
|---|---|
| payload | `artifacts/t4_m2_seed42_identity.pkl` |
| bytes | 18,650,875 |
| SHA-256 | `dcc449a15bc478f3380c95add964fc344522a25bc9938c0a5563bf5a75ae0c96` |
| sessions | 13: seven held-in and six held-out calibration sessions |
| identity shape | `(96,50)` for every session |
| direct T4 vs cached-identity max absolute error | `0.0` |
| direct T4 vs standalone decoder-only max absolute error | `0.0` |
| export receipt SHA-256 | `f062293a7f3899529161db65935608aaa4b9daa49db22a4fdd997591f2e9c169` |

All 13 session checks were exact. The compatibility runtime explicitly maps
NumPy 2's `numpy._core` pickle module names to NumPy 1's `numpy.core` names in
the validated base image. The first pre-fix local image tag without this map is
not a submission candidate.

## 6. Exact-candidate M33/q33 gate

The final candidate and seed-matched TS4 checkpoint were reloaded for test-only
CPU inference. Every eligible 50-bin history begins fully after the raw query
boundary.

| Session | B0 | T4 | TS4 | T4-B0 | T4-TS4 |
|---|---:|---:|---:|---:|---:|
| 2020-10-30 Run1 | 0.265599 | 0.372221 | 0.293256 | +0.106622 | +0.078965 |
| 2020-10-30 Run2 | 0.404346 | 0.439683 | 0.403893 | +0.035337 | +0.035790 |
| 2020-11-18 Run1 | 0.239725 | 0.384115 | 0.226851 | +0.144391 | +0.157264 |
| 2020-11-19 Run1 | 0.016041 | -0.013499 | -0.123820 | -0.029540 | +0.110321 |

The negative absolute R2 on 2020-11-19 is retained. T4 still beats the shuffled
label-content control there but does not beat B0. No session was deleted based
on score.

The T4 and TS4 replay artifacts seal the exact resolved configuration, split
manifest, checkpoint manifest, and per-session metrics hashes in
`candidate_m33q33_receipt.json`. B0 was recomputed at q33 from the exact epoch-34
teacher; the stale `R2_delta_vs_matched_baseline` column in the generated T4
CSV points to the historical q0 baseline and is explicitly not used.

## 7. Public minival gates

### Host FALCON evaluator

| Metric | Value |
|---|---:|
| Held In R2 Mean | 0.6531808376 |
| Held In R2 Std. | 0.0524009652 |
| Normalized Latency | 0.0308761847 |

The training artifact reports a seven-run mean of `0.64727443`. Recomputing the
host predictions separately over the seven runs gives `0.64683870`; the FALCON
evaluator instead concatenates Run1/Run2 by date and averages four date-level
scores, producing `0.65318084`. The apparent `+0.00591` difference is therefore
an aggregation-definition effect, not a cached-identity implementation gain.

### Final r3 container

| Metric | Value |
|---|---:|
| Held In R2 Mean | 0.6531808227 |
| Held In R2 Std. | 0.0524010110 |
| Normalized Latency, local CPU staging | 0.0798364657 |

Host and container R2 differ by approximately `1.5e-8`. A prior remote-path
simulation also showed every per-bin prediction differs from the host path by
at most `2.05e-8`.

Do not interpret the staging latency as the future official hidden-test
latency, or as including offline calibration.

## 8. Remote-path simulation

The exact final r3 image ran with `EVALUATION_LOC=remote`, public minival data
at `/dataset/evaluation_data/m2/minival`, and a writable submission volume.
It found all seven files and wrote:

| Output | Receipt |
|---|---|
| file | `submission.pkl` |
| bytes | 11,435 |
| SHA-256 | `02b45ab143abac0639a50c73cbd0a05469c9537c1a0342e5b53ebd0d17145f93` |
| task key | `m2` |
| session count | 7 |
| shape per session | `(194,2)` |
| normalized latency in this run | 0.0793489457 |

The local command ended with status 124 only because it deliberately
interrupted the official evaluator's fixed 300-second post-write sleep after
the complete file appeared. No container was left running.

## 9. Frozen Docker image

| Item | Frozen value |
|---|---|
| final tag | `spint-t4-m2:e8-seed42-epoch002-dcc449a-r3` |
| immutable image ID | `sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260` |
| uncompressed size | 9,474,513,979 bytes |
| EvalAI maximum | 42,949,672,960 bytes |
| base image | `spint-m2:e8-epoch027-76f0fb2` |
| immutable base image ID | `sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8` |
| added model payload | 18.65 MB |

The final labels distinguish the T4 artifact from its deployment base:

- `ai.eval.model.sha256` and `ai.eval.payload.sha256` both equal the T4 payload
  hash `dcc449a...`;
- `ai.eval.checkpoint.sha256` equals `25d7bc...`;
- `ai.eval.base.image.id` equals the validated original-SPINT image ID;
- `ai.eval.base.model.sha256` equals the original epoch-27 packaged-model hash.

Tracked build/preflight source hashes:

| File | SHA-256 |
|---|---|
| `export_t4_payload.py` | `4cd6914869463881811ad987315ecc2e036b97529f285c27929ccf365a01f077` |
| `t4_spint_decoder.py` | `fd1d5b203d9c8daccdda2e212c7efd7720f43717dceef876bd136e99bf7224dc` |
| `decode.py` | `beda41a330e15ea6327cd229e6a3f5458c2c2660425c98da9b2d34a30848f782` |
| `Dockerfile` | `2218fc41183b8ee2fc6ca535c015292374ea493fdf3718d31d0a50d6c5448408` |
| `submit_evalai.py` | `950eff328037a7627dcb57374b936b501bdf327b63e2e8740841df23dd3747f2` |

Only the `r3` tag and complete image ID above are authorized by the guarded
helper. The earlier `r1`/`r2` local tags must not be submitted.

## 10. EvalAI authenticated preflight

The configured token is accepted and remains mode `600`. Authenticated API
checks returned:

| Field | Value |
|---|---|
| participant team | `HKU-ECE`, ID `41975` |
| phase | Test Phase, ID `4599` |
| slug | `few-shot-test-2319` |
| active / paused | `true / false` |
| visibility requested | private |
| submissions today | 1 of 6 |
| submissions this month | 1 of 50 |
| submissions total | 1 of 100 |
| active concurrent submissions | 0 of 3 |
| only existing submission | original SPINT `578218`, finished |

Proposed required phase metadata:

| Attribute | Value | Reason |
|---|---:|---|
| `IsHeldOutZeroShot` | false | uses first-33 held-out calibration trials and target directions |
| `IsTestTimeAdaptive` | false | identities are frozen before EvalAI runtime; no container-time recalibration |
| `IsPretrained` | false | no data outside the FALCON datasets |

The method description explicitly discloses the labeled first-33 calibration
policy and the absence of query labels/backpropagation.

## 11. Guarded submission command

Read-only preflight:

```bash
/tmp/spint-e8-evalai-py38/bin/python \
  sua_exploration/evalai_t4_m2/submit_evalai.py
```

Formal push and private registration, only after explicit authorization:

```bash
/tmp/spint-e8-evalai-py38/bin/python \
  sua_exploration/evalai_t4_m2/submit_evalai.py \
  --execute \
  --confirm-image-id \
  sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260
```

The helper fails closed on image-ID, payload-label, phase, size, and quota
drift. It handles Docker 7's missing legacy `aux.Tag` event, verifies the ECR
manifest digest equals the frozen local image ID, records a resumable ignored
push state, attaches the required metadata, and refuses duplicate registration.

This was true at the preparation freeze. The command was subsequently executed
after explicit user authorization. `artifacts/evalai_push_state.json` now
records the single successful push and registration as submission `578221`.
Do not execute it again.

## 12. Interpretation after a future hidden result

The primary endpoint is private held-out R2. Held-in R2 and normalized latency
must be reported alongside it.

The operational comparison to submission `578218` is useful, but its numerical
difference is not a pure T4 ablation: `578218` packages the original epoch-27
SPINT decoder, while this T4 candidate freezes the epoch-34 teacher decoder.
The matched local q33 B0 comparison above uses epoch 34 and is the appropriate
local T4-effect reference. A strict official causal claim would require a
separate matched epoch-34 B0 hidden submission; this preparation does not
silently equate the two checkpoints.

Likewise, T4 has additional supervised label information relative to B0.
`T4-B0` is an end-to-end value-of-labeled-calibration contrast, while
`T4-TS4` is the channel-attachment/content control.
