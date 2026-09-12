# Official static M1/M2 — packs ready, wait to submit

Both CPU images are packed and host-verified. **Do not submit, push, or change tokens.**

HKU-ECE flats **582318** (M1 muscle flat e2) and **582319** (H1 flat e15) were cancelled 2026-09-12 by user; no official scores. Do not submit these static packs unless re-authorized.

Confirm the active `~/.evalai/token.json` team is HKU-ECE (41975) before `--execute`.

HKU-ECE (41975), falcon_m (41817), and sustechhku (42279) are the same group: three people each registered a team. Treat their EvalAI records as ours, not as external labs.

## Gate

- HOLD: 582318 and 582319 cancelled; no official scores
- Do not submit unless the user re-authorizes

## Submit commands (filled, not run)

M1:

```bash
/usr/bin/python3 tfpd_exploration/submissions/evalai_m1_rift_r100_static_e19_v1/submit.py \
  --manifest tfpd_exploration/submissions/evalai_m1_rift_r100_static_e19_v1/artifacts/evalai_candidate.json \
  --execute \
  --confirm-image-id sha256:03c320d462254209f50a4c0bb8002a0c43c8df062ad272cc66275ae1ee298d3d \
  --confirm-payload-sha256 e13af98508ba5bb72ac4e8f997bf502147c54dfa92ac6ebe1ed1f5da0a00b5e0
```

M2:

```bash
/usr/bin/python3 tfpd_exploration/submissions/evalai_m2_rift_r50_static_e15_v1/submit.py \
  --manifest tfpd_exploration/submissions/evalai_m2_rift_r50_static_e15_v1/artifacts/evalai_candidate.json \
  --execute \
  --confirm-image-id sha256:86a49a9324238f033b3f47113a0da91ecd130f7e698e2999f682353d294a2687 \
  --confirm-payload-sha256 b67f380b19d603512da8e764d9678a74a384ed22316d8bd02f8c23543ba6408a
```

## Packs

| task | dest | image_tag | image_id | payload_sha256 | eligible? | local score |
|---|---|---|---|---|---|---|
| M1 | `tfpd_exploration/submissions/evalai_m1_rift_r100_static_e19_v1/` | `m1-rift-r100-static-e19:v1` | `sha256:03c320d462254209f50a4c0bb8002a0c43c8df062ad272cc66275ae1ee298d3d` | `e13af98508ba5bb72ac4e8f997bf502147c54dfa92ac6ebe1ed1f5da0a00b5e0` | no — HOLD, user cancelled pending flats | HO3 e19 equal_session_mean −0.142517 |
| M2 | `tfpd_exploration/submissions/evalai_m2_rift_r50_static_e15_v1/` | `m2-rift-r50-static-e15:v1` | `sha256:86a49a9324238f033b3f47113a0da91ecd130f7e698e2999f682353d294a2687` | `b67f380b19d603512da8e764d9678a74a384ed22316d8bd02f8c23543ba6408a` | no — HOLD, user cancelled pending flats | EXT6 e15 equal_session_mean 0.283517 |

H1 official static is not packed (scan still running). Not packed: fair_v2 static-RIFT+diag-z / CORAL / WF. Not used: epoch_024.

## Host-verify tolerance

Gate is `1e-5` for both stream-vs-offline and stream-vs-cached. Offline windows mark left-pad bins invalid; Falcon `predict` bins are valid observations.

- M1 worst stream-vs-offline `1.43e-6`, stream-vs-cached `8.34e-7`
- M2 worst stream-vs-offline `4.77e-8`, stream-vs-cached `3.63e-8`

Container smoke passed on CPU (`CUDA_VISIBLE_DEVICES=''`). No EvalAI POST.
