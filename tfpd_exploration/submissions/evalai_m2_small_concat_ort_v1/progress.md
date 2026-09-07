# Progress Log

## Session: 2026-09-07

### Phase 1: Requirements & Discovery
- **Status:** complete
- Payload SHA 4db109e7 confirmed; base Python 3.10.15; 7 source_train streams; GPU0/1 left running.

### Phase 2: Dest layout + concat runtime
- **Status:** complete
- Wrote concat fast/ORT (e0_static+t4_static, no proj_add), decode.py, submit.py (register:false), host_pack_verify.py.

### Phase 3: Host gates
- **Status:** complete
- fast vs packed smoke 5.97e-9; advance 2.42e-8
- ORT vs packed smoke 5.96e-8; advance 5.96e-8; B=7 8.38e-8
- Graphs B=1..7 exported (new concat ONNX, not P32)

### Phase 4: Docker + smoke
- **Status:** complete
- Image `spint-t4-m2:small-concat-ort-e8-ext6-w0-4db109e7` sha256:7ff9beb109daffc583093bd3abc68ba1d8f68fd382129f933b6cd37e1fdae9be
- CONTAINER_SMOKE_PASS; register:false; evalai_opened:false
- 581973 dest untouched

### Phase 5: Delivery
- **Status:** complete

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| fast vs packed smoke | real Run1_20201019 50 bins | ≤1e-5 | 5.966e-9 | PASS |
| fast vs packed advance | 170 bins | ≤1e-5 | 2.421e-8 | PASS |
| ORT vs packed smoke | 50 bins | ≤1e-5 | 5.960e-8 | PASS |
| ORT vs packed advance | 170 bins | ≤1e-5 | 5.960e-8 | PASS |
| ORT vs packed B=7 | 7 tags × 170 | ≤1e-5 | 8.382e-8 | PASS |
| container smoke | ORT decoder | CONTAINER_SMOKE_PASS | CONTAINER_SMOKE_PASS | PASS |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5 complete |
| Where am I going? | Hold; no submit |
| What's the goal? | Pack+verify M2 concat SMALL e8 ORT image; no submit |
| What have I learned? | See findings.md |
| What have I done? | Packed and verified; register false |
