# H1 / M2 ORT next-push + M2 P-width queue

Date: 2026-09-07  
Series: `btransform_unified_v1`  
Status: **in progress** (CPU ORT ports via **non-Grok** subagents; no GPU train)

**Subagent rule:** never Grok / `inherit` when the parent is Grok. Use Claude or GPT slugs.

## Decision

Reuse the sealed M1 Runtime-line recipe (fast exact-E cores → ONNX → ORT CPU, intra2 official-protocol replay) on the **already-submitted** H1 / M2 winners. Next EvalAI push may use ORT **only if** the same FP32 gate closes (`max_abs` vs current exact-E P0 reference ≤ 1e-5, zero violations). Do not auto-submit.

M2 P16 / P32 / P64 **retrain is queued, not started**. GPU0 is owned by M1 depth-2 (`m1_projadd_depth2_series.py` pid 1873248). GPU1 is H1 C2-CAL-1 B2 HO-M3 sweep (pid 1870315). Do not steal either.

## Locked references (read-only)

| Task | Official cell | Payload | Image |
|---|---|---|---|
| M1 template | P16 e24 ORT intra2 | `evalai_m1_projadd_exacte_v1/.../m1_projadd_p16_s42_ema_e24.pkl` | `spint-m1:projadd-p16-s42-ema-e24-w0-3e260ebd` |
| H1 | 582025 P16 L=200 e24 | `evalai_h1_projadd_exacte_v1/.../h1_projadd_stage2_s42_ema_e24_L200.pkl` | `spint-t4-h1:projadd-s42-ema-e24-L200-w0-7a478342` |
| M2 | 582009 P32 e7 | `evalai_m2_projadd_p32_exacte_v1/.../m2_projadd_p32_s42_ema_e07_ext6.pkl` | `spint-t4-m2:projadd-p32-s42-ema-e7-ext6-w0-26b779f8` |

M1 sealed verdict (do not rerun as a gate): container minival B4, intra2, `max_abs=2.5e-6`, R² 0.9630 bit-identical, steady 9.96 ms/call, Normalized Latency 0.163.

Template scripts: `btransform_unified_v1/results/m1_projadd_runtime_v1/20260907T020150Z/scripts/{m1_exacte_fast,m1_exacte_ort,p1b_export_and_check,p1_equivalence_gate,p1_container_ort_replay}.py`  
Host ORT: same dest `ort_venv` (onnxruntime 1.19.2).

## What changes per task

Same operator semantics as M1: 5-position advance + per-wave rebuild, SDPA last-query, k=5 left-boundary recompute, PE window-relative, no cross-window temporal KV. Geometry only:

| | L | N | P | token_in | out |
|---|---|---|---|---|---|
| H1 | 200 | 176 | 16 | 20 | 7 |
| M2 | 50 | 96 | 32 | 36 | 2 |

New dests only:

- `btransform_unified_v1/results/h1_projadd_runtime_v1/<utc>/`
- `btransform_unified_v1/results/m2_projadd_runtime_v1/<utc>/`

Do not overwrite packed e24 / P32 dests or historical result roots.

## M2 P-width (when a GPU actually frees)

| Width | Verdict |
|---|---|
| P16 | Already have. Officially weaker than P32. Do not retrain. |
| P32 | Already official HO **0.359** (582009). Do not retrain. Wrap with ORT. |
| P64 | New cell, `token_in=68`, higher CPU cost. Concat official **0.390** still beats P32. **Not first in queue.** Only after ORT wraps P32 and a GPU is free without evicting M1/H1. |

## Out of scope

- EvalAI register/submit
- GPU0 / GPU1 occupancy
- Overwriting 582025 / 582009 artifacts
- Calling ORT “C2-identical”
