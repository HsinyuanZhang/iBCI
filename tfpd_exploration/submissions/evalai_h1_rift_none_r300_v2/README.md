# H1 RIFT R300 ablation package template

Builds one fresh local payload from a completed 32-epoch formal H1 ablation run and its sealed 27-tag bank directory. It is prepared for later host/runtime verification and Docker work; it does not claim host, container, registry, or EvalAI success and performs no external action.

```bash
python3 pack_and_verify.py --arm ACTIVITY_ONLY --banks <sealed-bank-dir> --run-dir <formal-run-dir> --dest <fresh-package-dir>
```

## Local package ready

Immutable identity:

- image: `h1-rift-none-r300-e15:v1`
- image ID: `sha256:9a8af356f7dd4b9e7e2b3b188eacc238b9499396891d5b40d9d11dd60bdfa0f5`
- payload SHA-256: `35d720ffc7b7d604be2d98573b09c6aea4661ca9765499c4e1185beb64a95671`
- selected epoch: 15; local HO-M3 grouped-seven mean `0.002879623722468462` (not official)

