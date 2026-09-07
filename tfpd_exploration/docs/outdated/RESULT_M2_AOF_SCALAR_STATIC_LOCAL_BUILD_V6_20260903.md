# AOF-S V6 local-build incident — 2026-09-03

Status: `LOCAL_BUILD_FAILED` during the offline container minival payload-load stage. No host minival, network operation, EvalAI submission, or GPU use occurred.

V6 successfully repaired V5's oversized/wrong Docker context and missing `third_party` package. It staged an exact eight-file, read-only Docker context, built the local image with `--network=none --pull=false`, and imported the nested runtime modules from the expected paths:

```text
spint-m2:aof-scalar-static-v6-local-v1
sha256:aa5cb12bb6a5ef5d56cb8fe73a73f27ddb48ac9418464f463a0d1e4c7af0a666

aofs_static_decoder -> /workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py
laws               -> /workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py
filtering           -> /workspace/third_party/falcon_challenge/filtering.py
```

The formal offline minival then failed while `CPUUnpickler` loaded the already sealed V2 payload:

```text
ModuleNotFoundError: No module named 'src'
```

The pinned base image does contain its historical model package at `/src`, but V6 sets `WORKDIR /workspace`, so Python's empty-path entry resolves to `/workspace`; V6's `PYTHONPATH` also omitted `/`. Consequently the immutable decoder object in `decoder.pkl`, whose historical pickle globals refer to `src.models...`, could not resolve the base image package.

A network-disabled, noncanonical diagnostic run of the exact built V6 image changed only the process environment to:

```text
PYTHONPATH=/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1:/workspace:/
```

With `/` present, the same sealed payload and same image completed the full local M2 minival normally:

```text
Held In R2 Mean: 0.6345120221376419
Held In R2 Std.:  0.0813317620899876
Normalized Latency: 0.06854311295008447
```

These diagnostic values are constructibility/runtime evidence only. They are not a new held-out result and were not submitted to EvalAI.

Immutable V6 result root:

```text
tfpd_exploration/results/m2_aof_scalar_static_package_v6/local_build
```

Exact bodies:

```text
attempt.json                 e8fd7fc1cadf82e3029fffa28ba616d33440c94724ccea771d35bd08a3749dac
predecessor_authority.json   aa3fc7297ba0eb6a72a989d025f5b1deae9a7ce654c177cbe2fcac3a13078fd1
input_authority.json         347e89e88ab10dcaa187d82e61f1cdbe795f81b733f6f4f377eef9cc46d5010b
failure.json                 d035048742e2ec8485e0ed43ef4d827fa1581d4a0fd5b2d60a9ee7d831f0ec6e
```

The root has exactly eight immutable leaves: those four bodies and their basename-bound `.sha256` sidecars, all mode `0444` and link count one. The reviewed V6 execution closure was:

```text
cfe32b783ef9acc3a4429b21afa9080b0a6f65378b53018aa7eff4c39e59f7c9
```

Its external closure authority was:

```text
tfpd_exploration/docs/AUTHORITY_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V6_20260903.json
sha256 3dc7048e0b40f168cb76eaafbb33d15b1f37ff7b6360271406fa7b41749b4db4
```

Required successor scope: additive V7 only. Preserve the sealed V2 payload, AOF-S science, beta, exact eight-file staged context, no-network Docker law, V6 artifact-parent discipline, and all evaluator arguments. Change only the container import path so the pinned base image's `/src` package is resolvable, preferably by appending literal `/` to the frozen `PYTHONPATH`. V6 must not be retried or overwritten.
