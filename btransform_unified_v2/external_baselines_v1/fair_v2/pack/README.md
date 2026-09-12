# Fair v2 EvalAI pack path

New pack pipeline for sealed fair_v2 linear and same-capacity static-RIFT
controls. Frozen v1 packages under `../submissions/*_v1/` are not mutated.

This directory exports numeric payloads, builds local CPU images, and writes
host/container replay audits. It does not docker-push, register, or submit.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/external_baselines_v1/fair_v2/pack/pack_all.py
```

Later submit helper: `submit.py`. Never invoke it with `--execute` from packing.
