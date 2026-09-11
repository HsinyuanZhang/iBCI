# M1 RIFT ACTIVITY_ONLY R100/e3 local package

This directory contains the completed local package for the independently
trained, audited `B_ACTIVITY_ONLY` M1 run. Its recorded state is
`LOCAL_PACKAGE_READY_NOT_SUBMITTED`: it has not been pushed, registered, or
submitted to EvalAI, and it has no official submission ID or result.

The immutable local package identity is:

- image: `m1-rift-activity-only-r100-e3:v1`
- image ID: `sha256:beae4356d303957cd9e33788c6702161fed497d9d579e44822c3cdfd4e2171ec`
- payload SHA-256: `650136399bd9f8582c88d7f721adcff6a70b2999f6c3c815e59a778fa8b451fc`

[The local pack receipt](artifacts/root_local_pack_receipt.json) records the
fresh 24-epoch / 159,960-update B training, selected checkpoint e3, host
verification, and successful container smoke. The host check sealed E0 from
the B checkpoint's EMA B3S encoder with literal zero side columns and sealed
literal zero direct T for all seven official M1 tags. It passed live-EMA/static
identity (`0` maximum absolute difference), packed/full parity
(`8.344650268554688e-7`), B8 cached/full parity
(`7.152557373046875e-7`), query-pad/full-stream checks, and the container
smoke; 33 container file SHA-256 values equal their host copies.

The public channel replay value `0.5618270536263784` only confirms local epoch
selection. It is not an EvalAI official metric.

`pack_and_verify.py --stage build` is intentionally overwrite-protected.
Do not rerun `build` against this completed directory; use the recorded
artifact and receipts for reproduction and review. The existing local image is
also already verified.

The ready immutable candidate manifest is
[artifacts/evalai_candidate.json](artifacts/evalai_candidate.json). The
guarded helper is [submit.py](submit.py). It verifies the local image labels,
payload and host receipt, exact EvalAI phase identity, server-side phase-4599
daily/month/total quota, and current concurrent occupancy before any write.
The following command is read-only; it does not push, register, or submit.
Its EvalAI client dependencies are the existing user-site packages, so use
`/usr/bin/python3` as shown:

```bash
/usr/bin/python3 tfpd_exploration/submissions/evalai_m1_rift_activity_only_r100_v1/submit.py \
  --manifest tfpd_exploration/submissions/evalai_m1_rift_activity_only_r100_v1/artifacts/evalai_candidate.json
```

No external push, registration, or submission has been claimed here. After a
reviewable preflight, the user-operated guarded execute command requires both
complete immutable values:

```bash
/usr/bin/python3 tfpd_exploration/submissions/evalai_m1_rift_activity_only_r100_v1/submit.py \
  --manifest tfpd_exploration/submissions/evalai_m1_rift_activity_only_r100_v1/artifacts/evalai_candidate.json \
  --execute \
  --confirm-image-id sha256:beae4356d303957cd9e33788c6702161fed497d9d579e44822c3cdfd4e2171ec \
  --confirm-payload-sha256 650136399bd9f8582c88d7f721adcff6a70b2999f6c3c815e59a778fa8b451fc
```
