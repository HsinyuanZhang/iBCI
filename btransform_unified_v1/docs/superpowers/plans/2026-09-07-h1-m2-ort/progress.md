# Progress

## 2026-09-07 12:41 +0800
- User asked to port M1 ORT to H1/M2 and try M2 P16/P32/P64 if GPU free.
- GPU not free. Queued P-width. Dispatching H1 and M2 ORT CPU subagents.

## 2026-09-07 12:43 +0800
- Spec written. H1 ORT agent `4c6015fb-2b63-42b2-92cc-80e690ef1857`, M2 ORT agent `ef9e4655-0291-40bb-bfb1-f5aaac769833`. Both CPU-only.

## 2026-09-07 12:45 +0800
- User: exclude Grok subagents. Wrote `.cursor/rules/no-grok-subagents.mdc`.
- Re-dispatched H1/M2 ORT on `claude-sonnet-5-thinking-high`, dest prefix `nongrok_`. Ignore inherit/Grok dests.

## 2026-09-07 12:49 +0800
- Grok M2 dest `20260907T044715Z` finished. Marked NOT_NEXT_PUSH. Waiting for Claude `nongrok_*` dests.

## 2026-09-07 12:51 +0800
- H1 host gate (Grok dest `20260907T044733Z`) PASS: max_abs 3.54e-8, 0 violations, B4 mean 22.8 ms. Marked NOT_NEXT_PUSH.
- Claude dests started: `h1_projadd_runtime_v1/nongrok_20260907T044956Z`, `m2_projadd_runtime_v1/nongrok_20260907T044947Z` (scripts only so far).

## 2026-09-07 12:57 +0800
- Claude M2 dest `nongrok_20260907T044947Z` host gate PASS. Marked NEXT_PUSH_CANDIDATE. H1 Claude gate still running.

## 2026-09-07 13:06 +0800
- Claude H1 dest `nongrok_20260907T044956Z` host gate PASS. Marked NEXT_PUSH_CANDIDATE. Both Claude ORT ports closed. Container replay still pending.
