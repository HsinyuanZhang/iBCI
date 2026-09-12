# External GF H1 coral

Self-contained CPU EvalAI context. Numerical inference uses NumPy only.
It contains the frozen numeric payload and no source query labels, raw NWB files, or training pickles.

## Build

```bash
docker build -t external-gf-h1-coral-v1:cpu .
```

## Frozen archive

`FROZEN_PROTOCOL_INVALID_FOR_COMPARISON`: this original single-source, `ridge=1` package lacks the official preprocessing control. It is not a fair RIFT comparison and is not eligible for EvalAI submission. Preserve this image, payload, manifest, predictions, and audits for traceability; do not push or register it.

See `../READINESS.json` for archive status and `../../fair_v2/` for the replacement plan.
