# Fair v2 M2 static_rift_diag_z

Same-capacity static-RIFT CPU image. Network weights are the sealed frozen EMA checkpoint; only the pre-local_conv neural frontend changes. No WF smoothing. Payload has numeric frontend maps and EMA weights only.

```bash
docker build -t fair-v2-m2-static_rift_diag_z:cpu .
```
