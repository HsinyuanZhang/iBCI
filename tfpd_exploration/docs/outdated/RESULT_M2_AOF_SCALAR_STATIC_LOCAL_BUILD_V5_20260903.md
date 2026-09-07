# AOF-S V5 local-build incident — 2026-09-03

Status: `LOCAL_BUILD_FAILED` during the offline container minival import stage. No host minival, performance scoring, network operation, EvalAI submission, or GPU use occurred.

V5 successfully repaired V4's missing artifact-parent lifecycle defect. It created only its exact route-owned artifact root, reused the sealed V2 payload, and built the local image:

```text
spint-m2:aof-scalar-static-v5-local-v1
sha256:e2e6e6e948c5b0c6524ebd73db66b75b5f4bcbc5072ed180d4e511a894841f9c
```

The nested decoder source path was also correct. The offline container then failed while importing `third_party.falcon_challenge.filtering`: the pinned base image provides `falcon_challenge` in site-packages, but does not provide the repository's top-level `third_party` package. A separate network-disabled diagnostic invocation confirmed:

```text
falcon_challenge -> /opt/conda/envs/spint/lib/python3.10/site-packages/falcon_challenge/__init__.py
third_party      -> ModuleNotFoundError
```

This is a container dependency-layout defect, not an AOF-S model, payload, β, decoder, or accuracy result. The frozen repository dependency already belongs to the reviewed closure and consists of exactly:

```text
SPINT-main/third_party/__init__.py
SPINT-main/third_party/falcon_challenge/__init__.py
SPINT-main/third_party/falcon_challenge/filtering.py
```

Immutable V5 result root:

```text
tfpd_exploration/results/m2_aof_scalar_static_package_v5/local_build
```

Exact bodies:

```text
attempt.json                 c684705696cd93790133821938a7e7b67939542f583cb377a2c4ab971e888bf5
predecessor_authority.json   3a448fb162ba175c99f5f53d15de42cdaa49925a543091f2c0d5d96789269d0b
input_authority.json         1a10ec8411edcdcd99d339f5ffd9c642beef28e39d1b7b6ffd7bb69834a76aa2
failure.json                 33607a6ea9b44625127a32e2103a3692f76702d960d0ec432e4fb3bafd6a28d2
```

The root has exactly eight immutable leaves: those four bodies and their basename-bound `.sha256` sidecars, all mode `0444` and link count one. The reviewed V5 execution closure was:

```text
daac9e48b71f5c570ebb059ef9b1e3a15058ad5bee5759e4b6d4f4e5d2bc6a73
```

Required successor scope: additive V6 only. Preserve the V2 payload, AOF-S science, β, nested decoder layout, offline evaluator command, and V5 artifact-parent discipline. Change only the Docker build context/layout needed to copy the three frozen `third_party` leaves to `/workspace/third_party/...`. V5 must not be retried or overwritten.
