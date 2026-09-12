# Fair v2 H1 static_rift_coral

Same-capacity static-RIFT CPU image. Network weights are the sealed frozen EMA checkpoint; only the pre-local_conv neural frontend changes. No WF smoothing. Payload has numeric frontend maps and EMA weights only.

```bash
docker build -t fair-v2-h1-static_rift_coral:cpu .
```
