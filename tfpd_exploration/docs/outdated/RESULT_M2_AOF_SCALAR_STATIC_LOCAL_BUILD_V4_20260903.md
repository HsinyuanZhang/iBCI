# AOF-S V4 local-build incident — 2026-09-03

Status: `LOCAL_BUILD_FAILED` before Docker build, container minival, host minival, or any performance scoring.

V4 correctly repaired the V3 container source layout, but its production lifecycle assumed that the route-owned artifact parent already existed. After publishing immutable `attempt.json` and `predecessor_authority.json`, `_reuse_build` called `mkdir()` for:

```text
tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/artifacts/local_build_v4
```

The intermediate `artifacts/` directory did not exist, so Python raised `FileNotFoundError`. No Docker command ran, no payload was rebuilt, no prediction or target file was produced, no network or EvalAI operation occurred, and `CUDA_VISIBLE_DEVICES` was empty. This incident therefore says nothing about AOF-S accuracy, decoder validity, or the V4 Docker layout.

Immutable V4 result root:

```text
tfpd_exploration/results/m2_aof_scalar_static_package_v4/local_build
```

Exact successful prefix and failure bodies:

```text
attempt.json                 fa653f6ebb0a8395e2b30152dea3c486665726f16551cfb7f763449c7b178a4d
predecessor_authority.json   229fda31ff4fc22a67d57cec0e902b3fd4ede73977507d2f5ad81ed00ade08ba
failure.json                 5e2260f80e2ddb5e33e5f5e8b22b6371efdc91b9229a02a3c5b0682b6d95dc59
```

The root has exactly six immutable leaves: those three bodies and their basename-bound `.sha256` sidecars, all mode `0444` and link count one. The reviewed V4 execution closure was:

```text
736ef936059c59d873040f24af23eda5df79547768a41cbe1e82ff2be9920610
```

Required successor scope: additive V5 only. Preserve the V2 sealed payload and V4 scientific/container contract; change only route-owned artifact-parent creation, use fresh V5 result/artifact roots and schemas, and bind this exact V4 failure graph. V4 must not be retried or overwritten.
