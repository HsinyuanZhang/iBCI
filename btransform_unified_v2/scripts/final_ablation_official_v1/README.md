# Fixed-control submission helper

`submit_fixed_control.py` accepts only `m1_none`, `m2_activity_only`, `m2_none`, `h1_activity_only`, and `h1_none`, all bound to EvalAI challenge `2319` and phase `4599`. The default action is read-only preflight: it verifies local Docker image metadata and makes live EvalAI `GET` requests for phase, challenge, submissions, and quota. It never tags, logs in, pushes, or sends `POST` unless `--execute` is supplied.

All five profiles use labeled public local data for selected-epoch choice. A NONE identity input therefore does not imply strict zero-shot, and `IsHeldOutZeroShot=False` is intentional.

A manifest must supply the fixed `profile`, payload/host/build/container receipt paths and SHA-256 values, immutable `image_tag`/`image_id`, method name/description, budget disclosure, and state path. It must bind the profile's real build and host receipt statuses and an exact future ROOT container receipt:

```json
{"schema":"root_final_ablation_container_verify_v1","status":"PASSED","image_id":"…","payload_sha256":"…","bytes_exact":true,"smoke_exit_code":0}
```

This is formal readiness evidence written only after ROOT completes the real container check; runtime smoke artifacts and smoke weights are rejected.

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/final_ablation_official_v1/submit_fixed_control.py \
  --profile m1_none --manifest /absolute/path/to/manifest.json
```

Only explicit execution can push/login/tag/register. It requires the exact immutable image ID and payload SHA:

```bash
…/submit_fixed_control.py --profile m1_none --manifest /absolute/path/to/manifest.json \
  --execute --confirm-image-id sha256:… --confirm-payload-sha256 …
```

Before POST it uses the authoritative phase-4599 quota response (daily, monthly, and total must all be positive), current concurrent occupancy, immutable ECR image binding, and exact Docker labels. A same-method submission with an unknown or different URI fails closed. POST exceptions first re-list submissions and recover only an identical method/URI match.
