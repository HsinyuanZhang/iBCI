# M2 AOF-S static package V7 local-build result

Date: 2026-09-03

Status: `LOCAL_BUILD_FAILED_AFTER_CONTAINER_MINIVAL`

This result is an engineering/runtime result only. It is not an EvalAI submission and does not provide a new held-out scientific score.

## Immutable V7 result

Canonical result root:

`tfpd_exploration/results/m2_aof_scalar_static_package_v7/local_build`

The root contains exactly five immutable receipt bodies and their SHA-256 sidecars:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `4d3a0761de6e0982f36248f8fd4e90dca3ed59195032339748c5238c92a79c21` |
| `predecessor_authority.json` | `e9ab1a29ea4311f2ed381748006c0dd86c601912753840fb1d8beedfd628ccab` |
| `input_authority.json` | `fcdd9a49ab38131e4cec8c105d961ca2c2fbe5b9a76f4dd565d9cd7a216972ab` |
| `build.json` | `ed1ac4ce7a0027d5b6086c67db220c907a305c50f78c127eb0e9c4ef830199ac` |
| `failure.json` | `12a6f12041cd964a32620f56a1483ec9b4fe7288adae43b1193db39b7919e298` |

The admitted V7 execution closure contained 75 leaves and had SHA-256:

`25e80bc4bb5ffbe176c8c898b0f6e2c2f5a38e5fb740dfb3cbd66e9362c5642c`

## What succeeded

V7 corrected the container-side import path while preserving the V2 payload, model, frozen scalar, staged eight-file Docker context, and offline execution law.

The formal route successfully:

1. reused and rehashed the frozen V2 payload;
2. built the local Docker image without network access;
3. ran the container with `network=none` and `pull=false`;
4. completed the M2 local minival inside the container;
5. sealed the container prediction, target, and stdout digests in `build.json`.

The container image ID was:

`sha256:e1451cb0db6913708fd779b78392b92ed7702f0038ce2bcd72960e4035cfc3c3`

The container minival prediction and target SHA-256 values were respectively:

- `38f7c206a492fb5b85266c17985b950ca841c662dee5f1dcf6a0094d643ec8db`
- `546e5a4a0b2260cdd4c7124d4d668fe0f136f84a81f7aa472e7dbe39a3115530`

Thus the V6/V7 container-layout and container-import questions are closed positively.

## Why the formal route failed

The failure occurred only when the host validator began `audit_payload`. The host process used a path order in which top-level `src` resolved to `tfpd_exploration/src`. Unpickling the frozen decoder requires `src.models` from `streaming_calibration_exp`, so the host validator raised:

`ModuleNotFoundError: No module named 'src.models'`

This happened before the host source bridge and host minival. It is a host validation namespace-order defect, not a payload, model, Docker, or container-minival failure.

## Bounded host diagnostic

A fresh `/tmp` diagnostic process placed `streaming_calibration_exp` before the repository paths in `PYTHONPATH`. That change allowed payload audit/unpickling to pass and advanced into the seven-source-session bridge. The next first failure was:

`NameError: name 'np' is not defined`

at `validate_local.py` while constructing the source neural array. The validator uses `np` throughout the bridge but does not import NumPy. No canonical result or artifact root was changed by this diagnostic.

The diagnostic establishes that the remaining issues are two narrow host-validator defects:

1. streaming path/bootstrap must precede payload unpickling in a fresh process;
2. NumPy must be explicitly imported before the source bridge.

## Successor constraint

Any successor must bind this exact ten-leaf V7 failure graph with held-directory/no-follow descriptor validation. It must preserve the already successful V7 payload, Dockerfile, staged Docker context, image build law, and offline container minival. It may change only the host validation execution seam needed to establish the streaming namespace before unpickling and to import NumPy explicitly.

The successor remains local-only. It must not submit to EvalAI, access a network, modify the V7 roots, or reinterpret local minival as a held-out scientific result.
