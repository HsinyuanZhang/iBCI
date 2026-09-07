# EvalAI M1 original/T4/D4 three-way preparation receipt

**Frozen:** 2026-08-02T13:23:34+08:00  
**Phase:** FALCON Test Phase `few-shot-test-2319` (`4599`)  
**Team:** HKU-ECE (`41975`)  
**Visibility:** private

## Scientific scope

This is an operational M1 system comparison on the organizer-held private test set.
Original SPINT uses the epoch-19 M1 model. T4 and D4 use the same frozen epoch-19 decoder,
with identities produced by the fold-1/seed-42 side-feature encoders and cached for the seven
public calibration sessions. T4 and D4 are directly matched to each other. Original SPINT was
trained on all held-in sessions whereas the T4/D4 encoder heads were trained on the three
fold-1 source sessions, so original-versus-side-arm differences are not pure feature-only
ablations.

All arms use chronological first-10 public calibration trials. T4 uses their target-direction
labels; D4 uses their `obj_id` labels. No hidden query label, optimizer step, backpropagation,
or online calibration fit is present in the submitted runtime.

## Frozen candidates

| Arm | Image | Immutable image ID | Payload SHA-256 | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- |
| original | `spint-original-m1:e9-epoch019-052e9ea` | `sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd` | `052e9eab7be2af8bacd5298e348d414cce60dad767d6a0880b58dd39b27279f6` | `c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2` |
| T4 | `spint-t4-m1:e9-f1s42-b5cc6d2-1f2e4f9` | `sha256:7897eb6adbb8451d9ea0f2fe5f07850baedff685d89ead3e7e23ee34d786cf13` | `1f2e4f96ac160abcd56dc5a79190bb3063099d6efe1039826817745556093957` | `b5cc6d28d9cb17782a234c3b744f827795c6bfaccb766855bebd351efd3529fb` |
| D4 | `spint-d4-m1:e9-f1s42-28530a7-a45471b` | `sha256:16aba3807945ca309c3a1041c9fdba7724e8321fc327962ce0fd2beba46e04a8` | `a45471b29a4b847f510a0a21695b2fccbe9615a8dbec285d3dffe3c4be2f6910` | `28530a7cd61f08edc9cf8f41cf5c7391f6150de72fc196ab3e38814bb698b11d` |

The common epoch-19 decoder checkpoint SHA-256 is
`c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2`.
Every image is below EvalAI's 40-GiB limit. Each image label and an independent in-container
`sha256sum /data/decoder.pkl` reproduce the payload hash above.

## Cached-identity audit

T4 and D4 payloads each contain exactly seven identities with dataset tags:
`20120924`, `20120926`, `20120927`, `20120928`, `20121004`, `20121017`, and
`20121024`. Each state has shape `[64,100]`. For every session, direct side-feature inference,
cached-identity inference, and manual decoder-only inference are bit-exact (`max_abs=0`).

The train-only normalization remains frozen to source sessions `20120924`, `20120927`, and
`20120928`. Loading the three held-out calibration files during export does not refit it.

## Public M1 minival

| Arm | Host R² mean | Container R² mean | Absolute host/container difference |
| --- | ---: | ---: | ---: |
| original | 0.7321254362 | 0.7321254353 | 0.0000000009 |
| T4 | 0.7270099695 | 0.7270099645 | 0.0000000051 |
| D4 | 0.7140224453 | 0.7140224504 | 0.0000000050 |

The public minival ordering is unfavorable to D4. It is recorded before the hidden result and
does not alter the user's authorized submission decision.

Container CPU normalized latencies were original `0.37903`, T4 `0.12069`, and D4 `0.11513`.
These are staging diagnostics, not official GPU benchmark latencies.

## Remote-path simulation

With public minival mounted at `/dataset/evaluation_data/m1/minival`, every image found the
four sessions and wrote a valid M1 prediction payload containing four `(1039,16)` arrays plus
normalized latency:

| Arm | Bytes | SHA-256 |
| --- | ---: | --- |
| original | 266,422 | `6c4a74c4fee9359a866f4aac997a0a37ea58fd6ae707359c1a4efa96a6f45e98` |
| T4 | 266,422 | `554b43bf4eff81440d73018a3e17b3b459893cc5a71881107eaf6e5ec6b07a0e` |
| D4 | 266,422 | `494871b7b363bcd4c06d1fae1073f2b6c27311c9ef33e2e84cac26c5629b5a46` |

The simulation commands ended with status 124 only because they interrupted the evaluator's
fixed 300-second post-write sleep after the complete output existed.

## Submission gate

Authenticated preflight found the phase active and unpaused. Before these submissions, quota
usage was `0/6` today, `2/50` this month, `2/100` total, and `0/3` concurrent. The candidate
hashes and methods are frozen above before opening any new hidden result.

