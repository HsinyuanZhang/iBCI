# E8 FALCON EvalAI submission receipt

**Prepared:** 2026-08-01T14:16:07+08:00  
**Formal submission created:** 2026-08-01T15:29:43+08:00  
**Challenge:** EvalAI 2319, *Few-shot Algorithms for Consistent Neural decoding (FALCON)*  
**Target phase:** `few-shot-test-2319`  
**Visibility:** private  
**Task:** native FALCON M2 (`DANDI 000953`)  
**Submission:** `578218` (participant team `HKU-ECE`, team ID `41975`)  
**Current status:** `FORMAL_FINISHED / OFFICIAL_RESULT_RECEIVED` — the private M2 hidden-test result has been returned and archived.

## 1. Scope and interpretation

This receipt freezes the first E8 candidate: the repository's original packaged SPINT M2
decoder. It is an external-benchmark anchor, not a T4/K4 submission. A successful hidden
test result would establish the frozen SPINT decoder's performance on the organizer-held
private query set; it would **not** by itself validate a T4/K4 improvement.

The candidate was selected before opening any hidden result. The packaged pickle contains
13 calibration-session tags. Each calibration tensor has shape `(33, 100, 96)`: 33
chronological calibration trials, 100 bins per trial, and 96 M2 channels. The decoder uses a
50-bin online window, behavior scaling factor 5, and no calibration smoothing.

## 2. Frozen artifact identity

| Item | Frozen value |
|---|---|
| packaged decoder | `SPINT-main/local_data/spint_m2_epoch27.pkl` |
| packaged-decoder SHA-256 | `76f0fb2092c81b5fed94cf947f859f685b03be095ea48d2e85a64cde421be3b0` |
| source run | `SPINT-main/logs/train/runs/2026-07-07-16-05-16` |
| source checkpoint | `checkpoints/best_ckpt/epoch_027.ckpt` |
| checkpoint SHA-256 | `4d0dfd5b56e635d2d3dc15bb698628ff8a0d7298d48e4fe097589c9ef32b8a66` |
| tensor provenance | all 31 packaged state-dict tensors exactly match the named checkpoint; no other inspected checkpoint matched |
| FALCON package | `falcon_challenge==1.0.2` |
| image tag | `spint-m2:e8-epoch027-76f0fb2` |
| immutable image ID | `sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8` |
| uploaded manifest digest | `sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8` |
| uploaded UUID tag | `0fb44635-1e66-4798-a72a-fc8fdf6742d8` |
| uncompressed image size | 9,457,273,463 bytes |

Build-input hashes:

| File | SHA-256 |
|---|---|
| `third_party/falcon_challenge/spint_sample.Dockerfile` | `0c239a3ba879d0bdb50f1af0378014cd011cf87ee51b8112c58acde4005adced` |
| `third_party/falcon_challenge/spint_sample.py` | `3d17b8097a2804f99380f922500a4a41c7f6c4e5a81a6504a010c6a55ce4260e` |
| `third_party/falcon_challenge/spint_decoder.py` | `3aacad1c22ea2b41ec776627de938c7973417361cf6d01da1e0daf15be0c7144` |
| `environment.yaml` | `67752e340c20e5aeefc98fb78bc2d0bcbc5d69b81d14bbfcb3cc830b1d612848` |

The image embeds the packaged-decoder SHA in label `ai.eval.model.sha256`. An independent
`sha256sum /data/decoder.pkl` inside the completed image returned the same digest.

## 3. Staging results

### 3.1 Host local minival

Command contract: M2, phase `minival`, batch size 7, `falcon_challenge==1.0.2`.
The repository root must be on `PYTHONPATH`; the README's bare script invocation otherwise
fails before model loading because `third_party` is not importable.

Result on the seven public M2 held-in minival files:

| Metric | Value |
|---|---:|
| Held In R2 Mean | 0.5238485709 |
| Held In R2 Std. | 0.1309910220 |
| Normalized Latency | 0.0767972656 |

### 3.2 Container local minival

The host Docker daemon has no NVIDIA container runtime, so the container audit used the
decoder's supported CPU fallback. This changes latency, not decoded values.

| Metric | Value |
|---|---:|
| Held In R2 Mean | 0.5238486007 |
| Held In R2 Std. | 0.1309909704 |
| Normalized Latency (CPU container) | 0.4565695391 |

The container and host R2 means differ by only `2.98e-8`. This passes the numerical-parity
gate. The CPU latency is a local runtime diagnostic and must not be presented as the hidden
EvalAI latency.

### 3.3 Exact remote-path simulation

The image was also run with its default `EVALUATION_LOC=remote`, public minival data mounted
at the server contract path `/dataset/evaluation_data/m2`, and a writable `/submission`
volume. It found all seven files and emitted:

| Output | Receipt |
|---|---|
| `/submission/submission.csv` | 11,435 bytes |
| output SHA-256 | `273c9e744224721d925e34842f3931156ccd29ad5ea2b22316fac392d43750a4` |

This validates the remote filesystem and output interface. It is still a local simulation,
not a leaderboard submission.

## 4. Packaging corrections made for E8

1. Added `MODEL_FILE` as a Docker build argument so the image can bind the audited
   `spint_m2_epoch27.pkl` instead of relying on an ambiguous `spint_m2.pkl` alias.
2. Pinned `falcon_challenge==1.0.2`, matching the successful host evaluation.
3. Added `SPINT-main/.dockerignore`. The original build sent about 20.9 GB of datasets,
   checkpoints, and logs to Docker; the audited context is about 160.5 MB.
4. Installed the EvalAI CLI in an isolated Python 3.8 environment at
   `/tmp/spint-e8-evalai-py38`. `evalai==1.3.18` cannot be installed cleanly under the
   host's Python 3.10 because it pins `lxml==4.6.2`, for which that interpreter has no wheel.

These changes affect packaging and reproducibility only. They do not alter decoder weights
or predictions.

## 5. Formal phase and submission receipt

The public EvalAI phase page reports that challenge 2319 remains open through 2099-05-31.
The Test Phase evaluates private test data and currently permits at most 6 submissions/day,
50/month, 100 total, and 3 concurrent submissions. The FALCON repository specifies the
phase pattern `few-shot-<test/minival>-2319`, hence the frozen target is:

```bash
evalai push spint-m2:e8-epoch027-76f0fb2 \
  --phase few-shot-test-2319 \
  --private
```

Authenticated preflight confirmed all of the following before the push:

| Field | Receipt |
|---|---|
| challenge participation | already participating in challenge `2319` |
| participant team | `HKU-ECE`, ID `41975` |
| phase | Test Phase, ID `4599`, exact slug `few-shot-test-2319` |
| phase visibility/data | public phase using private test data |
| requested submission visibility | private |
| API time limit at submission | 3,600 seconds |

The single private push completed successfully. The uploaded manifest digest exactly equals
the frozen local image ID. EvalAI then accepted the corresponding submission registration
with HTTP 201:

| Formal field | Value |
|---|---|
| submission ID | `578218` |
| server timestamp | `2026-08-01T07:29:43.100855Z` |
| local timestamp | `2026-08-01T15:29:43.100855+08:00` |
| status at creation | `submitted` |
| observed lifecycle | `submitted` → `queued` → `running` → `finished` |
| final API `started_at` | `2026-08-01T07:50:53.951278Z` |
| final API `completed_at` | `2026-08-01T07:50:54.123367Z` |
| final API `execution_time` | 0.172089 seconds |
| latest status | `finished` |
| public | `false` |
| participant-team ID | `41975` |
| challenge-phase ID | `4599` |
| ignored / flagged | `false` / `false` |
| stdout / result / stderr | present / present / absent |

During polling, EvalAI first exposed `started_at` values for the queue/container lifecycle,
then replaced that field at completion with the final scoring interval above. Therefore the
reported 0.172089 seconds is not an end-to-end container runtime or decoder latency. The
official benchmark latency quantity is `Normalized Latency`, reported below.

## 6. Official private-test result

EvalAI returned the following JSON for `test_split_m2`:

| Official metric | Value |
|---|---:|
| Normalized Latency | 0.1123610309530884 |
| Held Out R2 Mean | 0.18647872031978285 |
| Held Out R2 Std. | 0.1621446155579683 |
| Held In R2 Mean | 0.5682426700992199 |
| Held In R2 Std. | 0.031164881860054585 |

The official stdout is 260 bytes with SHA-256
`71eec22847985448063ae1b4b36ed1449edcdef417bb4afff5fd2e6c1eb3a0ea`. The official JSON
result file is 222 bytes with SHA-256
`baa6432cdc74b6354e564a66a7c54b4acef48b7849870445a490a104c20c27dd`. EvalAI exposed no
stderr file. A machine-readable receipt is preserved at
`results/e8_falcon_evalai_m2_spint_epoch27_v1/submission_578218_receipt.json` (SHA-256
`5f4499810758078dff0020ae1c3bfb14fe404d30446a077048febe620111e8ee`). Per the repository's
artifact policy, `sua_exploration/results/` remains local/ignored; the complete official
metrics and remote-file hashes are mirrored in this tracked documentation receipt.

Interpretation:

- The original frozen SPINT M2 decoder is healthy on the private held-in split
  (`R²=0.56824`) but substantially weaker and heterogeneous on held-out sessions
  (`R²=0.18648 ± 0.16214`). This is exactly the failure mode for which held-out few-shot
  calibration matters.
- This result is an external anchor for original SPINT, not evidence that T4 or K4 improves
  it. No T4/K4 model was present in this image.
- The submission is private. The returned score is official, but this receipt does not claim
  a public-leaderboard rank or superiority over another method.
- The local minival result and official private result use different evaluation data. Their
  numerical difference must not be treated as a paired improvement.

## 7. EvalAI CLI compatibility audit

The configured token was accepted, and its local file mode was tightened to `600`. The
legacy `evalai==1.3.18` push path required two compatibility corrections for the host's
Docker 29 daemon:

1. The isolated CLI environment was upgraded to `docker==7.1.0`, because the package's old
   Docker SDK negotiates API 1.35 while this daemon requires at least API 1.44.
2. The temporary CLI code was made to read `Size` when Docker 29 omits the legacy
   `VirtualSize` image attribute.

Both failed attempts stopped before a submission existed and consumed no formal-submission
quota. The corrected push uploaded the image once. Docker 7 reports successful completion as
a final digest/status record rather than the `aux.Tag` event expected by EvalAI 1.3.18, so
the legacy client skipped only its final registration call despite exiting successfully.
After confirming that the phase submission list was still empty, the already-uploaded UUID
tag was registered exactly once through the same EvalAI submission endpoint and multipart
payload used by the CLI. The server returned HTTP 201 and submission ID `578218`; no second
image upload occurred.

These are deployment-client corrections only. They do not modify the container manifest,
decoder, model state, predictions, phase, or private/public selection.

## 8. Exact continuation

1. Treat submission `578218` as closed; do not create a duplicate submission.
2. Interpret the result only as the external hidden-test anchor for the original frozen SPINT
   M2 decoder. It is not a T4/K4 comparison.
3. Do not select or resubmit a different checkpoint in response to the hidden score unless a
   separately declared experiment explicitly authorizes it.
4. Any later T4/K4 EvalAI comparison must freeze its candidate and submission decision before
   exposing another hidden result, use the same M2 phase, and report held-in, held-out, and
   normalized latency together.

E8 is **formally finished**. The main scientific signal is the held-out generalization gap,
not a T4/K4 gain.
