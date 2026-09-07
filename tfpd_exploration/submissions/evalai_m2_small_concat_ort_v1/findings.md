# Findings & Decisions

## Requirements
- Pack + verify only. No EvalAI submit. No train. No GPU-job kill.
- Source dest read-only: evalai_m2_small_trf_ext6_epochpick_v1 (581973)
- New dest: evalai_m2_small_concat_ort_v1 (create; refuse overwrite)
- Payload SHA: 4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4
- Geometry: kind=small, concat token_in=70, W=50, N=96, 8-slot, 4-layer CausalPE, out=2, scale=/5, official B=7, 13 tags
- ORT: graphs B=1..7, intra_op=2, inter_op=1, wheel 1.19.2 original filename
- Base: spint-m2:e8-epoch027-76f0fb2 Python 3.10
- Labels: ai.eval.task=m2, exact_e=true, payload.sha256, method says concat SMALL e8 ORT not proj_add not 581971

## Research Findings
- Dest did not exist at start (create OK).
- Sealed decoder SharedSetFrontend uses torch.cat([local, e0, t4], dim=-1); no e0_proj / grouped P.
- Fast advance recipe still applies: one conv over W=50, set-attn on 5 changed positions (0..3 + last), last-layer last-query.
- M2 real streams: tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train
- Source-train tags (7): Run1_20201019, Run1_20201020, Run1_20201027, Run1_20201028, Run2_20201019, Run2_20201020, Run2_20201027
- Templates: M1 exacte_fast/ort (proj_add geometry — adapt frontend); H1 B2 pack pattern for Docker/submit/receipt
- Do not copy P32 ONNX graphs.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| ORT inputs e0_static [B,1,N,50] + t4_static [B,1,N,4] | Concat identity; P(E0) does not exist |
| RT_PACKED_DECODER points at copied trf_falcon_decoder.py | Fast/ORT wrap sealed classes |
| Hardlink payload after SHA check | Bit-identical sealed cell; no rewrite |
| OMP=2 in image (not 581973's OMP=1) | Match H1/M1 ORT recipe intra_op=2 |
| Container smoke via m2_concat_exacte_ort.py | Must print CONTAINER_SMOKE_PASS |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
|       |            |

## Resources
- Source decoder: tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py
- Wheel: btransform_unified_v1/results/m1_projadd_runtime_v1/20260907T020150Z/scripts/vendor/onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
- Host ORT venv: .../20260907T020150Z/ort_venv/bin/python
- H1 pack: btransform_unified_v1/scripts/pack_h1_c2_cal1_b2_ort_v1.py
