# M1 RIFT NONE R100 static package

`build` accepts only a completed formal M1/NONE seed-42 run. It validates the
training and score receipts, source bindings, all 24 checkpoint hashes, and the
HO3 earliest-maximum selection before sealing the selected EMA into a static
RIFT concat payload. Every official tag carries literal `float32` zero
`E0[64,100]` and `T[64,4]` arrays.

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/pack_and_verify.py \
  --stage build --run-dir btransform_unified_v2/results/final_ablation_official_v1/formal_m1_none_s42_v1
```

The build requires fresh `artifacts/` targets and writes
`build_receipt.json` with status `BUILT_NOT_HOST_VERIFIED`.

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/pack_and_verify.py \
  --stage host --run-dir btransform_unified_v2/results/final_ablation_official_v1/formal_m1_none_s42_v1
```

`host` dynamically follows `build_receipt.json`, verifies the selected static
EMA decoder against independent full-R100 references and cached KV runtimes on
the public padded stores after stripping 99 query bins, and writes
`host_verify.json` with status `HOST_VERIFIED`. It also seals an unchanged
40-bin raw fixture and its complete expected prediction trace for the runtime
smoke command.

```bash
docker build --build-arg PAYLOAD_SHA256="$(sha256sum tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/m1_rift_none_r100_selected.pkl | awk '{print $1}')" \
  -t m1-rift-none-r100-cached:v1 tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1
```
