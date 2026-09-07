# H1 RIFT official freeze — 2026-09-07

## Decision

Freeze the H1 baseline at `h1_rift_r300_recency_e22_cached`.  Do not run further
H1 score-chasing work or submit another H1 candidate.  The active effort moves to
M2 and M1.

Existing H1 diagnostics and the already-planned L200 work may be completed for
understanding or operational closeout.  They must not replace this frozen baseline
automatically; any replacement requires an explicit new decision and a separately
recorded submission identity.

## Frozen submission identity

| Field | Frozen value |
| --- | --- |
| EvalAI submission ID | `582073` |
| Candidate / arm | `h1_rift_r300_recency_e22_cached` |
| Method | `H1 RIFT R300 recency e22 cached` |
| Challenge / phase | `2319` / `4599` |
| Submitted at | `2026-09-07T12:12:12.548011Z` |
| Payload SHA-256 | `7957ce7b52596745aab55e5955e8763cab3b2f4808c25a037688890728883b56` |
| Image manifest digest | `sha256:3d9f1c35371778078225c8b28895fc32dd526b094b0c182daae78f19768fa5df` |
| Submitted image tag | `905418259932.dkr.ecr.us-east-1.amazonaws.com/few-shot-algorithms-for-consistent-neural-decoding-falcon-2319-participant-team-41975:22c00e42-a088-40fa-a525-1d8957d34742` |

The local registration and push-state records agree on the submission ID, candidate,
payload hash, and image digest.  `REGISTERED.json` reports host pack verification
with maximum absolute difference zero for B1 and B8; this is a package check, not an
official-score receipt.

## Model configuration

The frozen model is RIFT with cached CPU KV runtime: H1, context window `300`,
recency bias, seed `42`, `proj_dim=16`, batch cap `8`, and behavior scale `20`.
The candidate metadata describes `D4/P16/width256`, EMA epoch `22`, trained over 32
epochs and 13 sessions.  Its frontend is the accepted v1 `proj_add` identity mode.
It is not BT-EORT, ONNX Runtime, or SPINT.

## Official metrics

The completed official result for submission `582073` was retrieved on
2026-09-07 by an authenticated read of the existing submission, followed by
its result artifact. These values independently confirm the user's report.

| Metric | Value | Provenance |
| --- | ---: | --- |
| Held Out R² mean / std. | `0.4027782688014744` / `0.14526658466582196` | Official result artifact, `test_split_h1` |
| Held In R² mean / std. | `0.6681165838896308` / `0.025051352172649453` | Official result artifact, `test_split_h1` |
| Normalized Latency | `0.14299749625157748` | Official result artifact, `test_split_h1` |

The submission GET response, original result JSON, and retrieval SHA-256 are
archived in `results/rift_v1/h1_r300_official_receipt_20260907/`. The API reports
`status=finished`. This read did not create a submission or alter the frozen
payload, checkpoint, or container.

## Integrity locks

No payload, weight, or container package is copied into this freeze record.  The
manifest at `results/rift_v1/h1_rift_r300_official_freeze_20260907/manifest.json`
contains the corresponding file paths and SHA-256 values.

Core source hashes at freeze time:

| File | SHA-256 |
| --- | --- |
| `tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/Dockerfile` | `f3f1722b4c98252d362d6b674989f0dce32e1ba2b397d544a4f61051e7d6639a` |
| `tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/decode.py` | `fcc01ded048ef18040f05bef6d470e98c54f9277652e1927daea60ff80434056` |
| `tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/h1_rift_falcon_decoder.py` | `8ef7ff3c4292bec051e5dafab236bb6792b093708ae0ca9076bce3e777a426b2` |
| `btransform_unified_v2/src/btransform_unified_v2/model.py` | `482fe07122332f3ad5610e1348a98f139fe3a113a947c4d9a1d47787ea71efca` |
| `btransform_unified_v2/src/btransform_unified_v2/cpu_runtime.py` | `90729daded828d02da8ef0931aa87ef17dc3fa86dfe63de33d408967b6d3605c` |
| `btransform_unified_v1/src/btransform_unified_v1/bank.py` | `015776661ff2782cacdee1d49c0440a83fd52721b5481cdbca7be1e73f0e6dce` |

## Scope after freezing

M2 and M1 are now the active optimization and submission tracks.  Preserve the
frozen H1 identity for comparison and reproducibility.  Do not alter the sealed
package, retrain it, or initiate an H1 submission as part of this freeze.
