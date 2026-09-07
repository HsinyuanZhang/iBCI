# Task Plan: H1/M2 ORT port for next push

## Goal
Port the sealed M1 ORT runtime recipe onto H1 P16 L200 e24 and M2 P32 e7, close the FP32 equivalence gate on CPU, and leave pack-ready artifacts for the next EvalAI push. Do not train M2 P-widths while GPUs are occupied.

## Current Phase
Phase 2 — parallel CPU ports

## Phases

### Phase 1: Requirements & Discovery
- [x] Confirm M1 ORT verdict and template scripts
- [x] Confirm GPU0 = M1 depth-2, GPU1 = H1 B2 HO sweep
- [x] Confirm sealed H1/M2 payloads and images
- **Status:** complete

### Phase 2: Parallel CPU ORT ports
- [x] H1 Claude dest `nongrok_20260907T044956Z` host PASS
- [x] M2 Claude dest `nongrok_20260907T044947Z` host PASS
- [x] Grok dests marked NOT_NEXT_PUSH
- **Status:** complete

### Phase 3: Compare and pack-readiness
- [x] Next-push candidates = Claude `nongrok_*` dests
- [ ] Container replay before authorize
- **Status:** in_progress

### Phase 4: M2 P-width (blocked)
- [ ] Wait until a GPU is actually free
- [ ] Do not start P16/P32 retrain; P64 only after ORT wrap
- **Status:** pending

## Decisions
- SHA restore not required for B2 (prior). ORT uses sealed payloads as-is.
- No auto-submit.
- Do not occupy GPU0/GPU1.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| | | |
