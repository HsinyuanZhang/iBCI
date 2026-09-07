# M1 Runtime V3 — local validation record

## Scope and immutable local candidate

This is a **local diagnostic** record, not an official latency promise, image
acceptance, push, registry action, or submission. The existing user submission
package and image were not changed.

| Item | Bound value |
|---|---|
| Candidate tag | `spint-m1-runtime-v3:local-20260906-r2` |
| Candidate image digest | `sha256:613944b53d6039cdb6831e97e56e0327d79d2df37f9627b886b4ba1e4ab604e6` |
| Copied payload SHA-256 | `5e44fc37965640138bf977da42a988e7636daa996d1eb9b50a75fa5628f9d7b3` |
| Container decoder SHA-256 | `55bf8862553abfa9cf1c7743521f6f6b13141fb0d988e0ac829809bad10ea6a4` |
| Container V3 runtime SHA-256 | `92e1b66f2178fa61653d0e331399dab77f60311e06762780ad21b80e8b9445ac` |
| Container Torch | `2.5.1.post303` |

The payload is the frozen seven-bank package payload. The source fixture tests
only the legal source subset—20120926, 20120927, 20120928, and a fourth lane
replaying 20120926—not the full official evaluator bank roster. The wrapper
still resolves banks exclusively from the copied payload by Falcon dataset tag;
the seven-bank payload is retained unchanged.

`container_parity.py` and `container_benchmark.py` are mounted local diagnostic
scripts, not files copied into the candidate image. Their hashes must therefore
not be confused with the container decoder/runtime file hashes above.

### Inherited header disclosure

The vendored `m1_trf_falcon_decoder.py` retains an inherited top-of-file prose
header that inaccurately describes a `256→128→7` output, `/20` scaling, and no
cross-window KV reuse. That prose was inherited from a generic decoder and is
**stale**; it was not edited after the image build so as not to mutate the
validated candidate. The actual bound M1 contract is `out_dim=16`, native
`BEHAVIOR_SCALE=1.0` (divisor one), finite `W=100/k=5`, and V3's checked
current-query memory cache. The executable constants, the vendored
`v3_runtime.py`, the copied payload state, and the parity records—not that
inherited header—are the operative specification.

## Local container parity

The mounted, source-only fixture
`fixtures/source_b4_public_parity.npz` was generated from cached raw source
observations and the frozen package public API. It contains no labels and no
outer data. The candidate passed:

- B4 256-call continuous long roll: maximum native absolute error
  `1.430511474609375e-06`, satisfying
  `|error| <= 1e-5 + 1e-5 * |reference|` elementwise;
- B1 startup/public path;
- reset to different source banks; and
- a final B2 partial evaluator batch, with inactive lanes frozen and the
  returned result an owning native `float32` copy.

The host complete selected-EMA V3 validation remains the stronger source
correctness record: [complete selected replay](../../results/decoder_validation_v2/20260905_190000/m1/runtime_v3/complete_selected/complete_selected_v1.json)
replayed all 31,252 source endpoints across the three source sessions,
consumed all intervening raw gaps, passed independent pre/post NPZ audits, and
had maximum native error `1.430511474609375e-06`.

## Public latency observations

All candidate calls used cpuset `8-11`, two Torch/BLAS threads, Torch interop
one, workers zero, C-contiguous native `float32` inputs, and the actual public
decoder method. The candidate runs below are **three fresh containers × 2,048
calls** after 128 warmup calls:

| Candidate run | Mean ms | P95 ms | P99 ms | Cold construct/reset ms | First return ms |
|---|---:|---:|---:|---:|---:|
| r1 | 8.424 | 9.288 | 10.447 | 129.980 | 9.756 |
| r2 | 8.260 | 8.361 | 8.518 | 136.428 | 9.801 |
| r3 | 8.301 | 8.440 | 8.573 | 130.427 | 9.752 |

The historical submitted-image comparison is intentionally unequal in duration:
the original image was sampled at **128 calls**, not 2,048. Its same-fixture
B4/t2 result was mean `80.058 ms`, P95 `80.347 ms`, P99 `80.540 ms`, and first
return `90.749 ms`. It returned a non-owning view, whereas the candidate
requires a copied public result. This comparison is useful as a diagnostic of
the original whole-window adapter, but it does not make a deployment claim.
On these unequal-duration local samples, the candidate's approximately
`8.26–8.42 ms` mean is about `9.5–9.7×` lower than the original image's
`80.058 ms` 128-call mean; this is descriptive only, not an official promise.

The candidate's measured cgroup `memory.current` peak values were raw bytes:
`230,809,600`, `232,157,184`, and `232,374,272` bytes. These are approximately
`220.12`, `221.40`, and `221.61` MiB respectively (MiB = 1,048,576 bytes), and
include process/container allocation rather than only the V3 owned cache.
The V3 B4 persistent runtime-owned cache is separately `4,080,160` bytes.
The original-image 128-call comparison reported `248,119,296` raw cgroup bytes,
approximately `236.62 MiB`; its duration and output-copy semantics remain
different from the candidate measurement.

## Reproduction commands

The local-only image was built with:

```bash
docker build \
  --build-arg BASE_IMAGE=spint-original-m1:e9-epoch019-052e9ea \
  --build-arg PAYLOAD_SHA256=5e44fc37965640138bf977da42a988e7636daa996d1eb9b50a75fa5628f9d7b3 \
  -t spint-m1-runtime-v3:local-20260906-r2 \
  tfpd_exploration/runtime_candidates/m1_runtime_v3_falcon
```

The strict mounted parity invocation was:

```bash
docker run --rm --cpuset-cpus=8-11 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 -e NUMEXPR_NUM_THREADS=2 \
  -v "$PWD/tfpd_exploration/runtime_candidates/m1_runtime_v3_falcon:/work:ro" \
  spint-m1-runtime-v3:local-20260906-r2 python /work/container_parity.py
```

Dependencies are inherited from `spint-original-m1:e9-epoch019-052e9ea` and
include its Falcon challenge runtime plus actual Torch `2.5.1.post303`. The
candidate adds only its copied payload, vendored decoder, vendored V3 runtime,
and decode entrypoint.

## Remaining uncertainty

The local tag and the prior `local-20260906` tag are preserved; neither was
pushed or retagged. Local source-fixture parity and local latency do not prove
all official evaluator tags, evaluator lifecycle details, container cold-start
conditions, host contention behavior, a two-hour wall-clock outcome, or an
official latency threshold. Those remain separate acceptance work.
