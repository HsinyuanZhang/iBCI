# Task Plan: M2 concat SMALL e8 ORT exact-E pack (no submit)

## Goal
Pack and verify a CPU-only Docker image wrapping sealed M2 SMALL concat (EvalAI 581973, payload SHA 4db109e7) with B-transformer unified ORT exact-E. Do not train, do not EvalAI-submit, do not overwrite 581973 dest, do not kill GPU jobs.

## Current Phase
Phase 5 complete

## Phases

### Phase 1: Requirements & Discovery
- [x] Confirm dest absent; payload SHA; wheel; base image Python 3.10
- [x] Locate real M2 streams and 7 source-train tags
- [x] Document concat vs proj_add frontend difference
- **Status:** complete

### Phase 2: Dest layout + concat runtime
- [x] Copy decoder + payload (hardlink after SHA)
- [x] Write m2_concat_exacte_fast.py (concat tokens, no e0_proj)
- [x] Write m2_concat_exacte_ort.py (new B=1..7 graphs)
- [x] Write decode.py, Dockerfile, submit.py (register:false)
- **Status:** complete

### Phase 3: Host gates
- [x] Packed exact-E vs itself/smoke on real M2 stream (max_abs ≤ 1e-5)
- [x] Export graphs B=1..7
- [x] Host ORT vs packed: smoke + advance ≥100 + B=7 (≤ 1e-5)
- **Status:** complete

### Phase 4: Docker build + container smoke
- [x] Build from spint-m2:e8-epoch027-76f0fb2, bake ORT 1.19.2 wheel original filename
- [x] Container smoke prints CONTAINER_SMOKE_PASS
- [x] Write payload.receipt.json + evalai_candidate.json (register:false)
- **Status:** complete

### Phase 5: Delivery
- [x] Reply dest, image tag+id, payload sha, all max_abs, smoke status, hold reason
- **Status:** complete

## Key Questions
1. Is e8 pkl present with sealed SHA? Yes.
2. Can 7 real tags be formed from source_train cache? Yes.
3. Are unit masks all-true (ORT unmasked export)? Yes.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| New dest only; never write 581973 | User: refuse overwrite of sealed cell |
| Concat frontend: cat(local16,E0 50,T4); graph inputs e0_static+t4_static | Not proj_add; do not reuse P32 ONNX |
| submit.py guarded, register:false | Pack+verify only |
| Planning files stay in dest; dockerignore excludes them | Avoid baking notes into image |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |
