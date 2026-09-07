# SUA formal one-shot readiness audit — 2026-08-05

**Status:** **NO-GO for another opening of the existing sub-C formal scope.**

**Audit scope:** read-only inspection of development-side receipts, frozen aggregates, validators, and test code. This audit did not enumerate, hash, resolve, or open a formal raw session. It did not run a model, scorer, CUDA task, or EvalAI submission.

## Executive decision

Fresh C1 is internally closed development evidence. Its best **provisional scientific candidate** is full-descriptor `shared_t4`: shared B3S / full T4 `[a,c,m,b]`, source objective `0.5 L(SUA)+0.5 L(pseudo-MUA)`, target activity support first 30 trials, labelled T4 fit first 50 rewarded trials, score from trial 50 onward, and analytic/forward target-session calibration with no target-session backward or update.

It is **not yet a final one-shot candidate**. First, a pre-existing P3 formal-scope receipt is already `status=started` (SHA-256 `013d21a738dc604071f276f3b04362d178e8fa8d9cb1801b28c989a91610e07a`). The project bridge audit defines that scope as consumed even though fresh C1 did not open raw files. It cannot be rerun, renamed, or treated as statistically fresh. Second, encoder-only QAT v2 is still running and C2 has no completed source-only unique-lambda gate or frozen C2 prelaunch.

The correct independent-confirmation route is a **new, separately authorized external subject/data scope** after explicit candidate freeze. Existing documentation identifies an external subject with enough sessions as preferable and the small three-session subject only as a smoke-test substrate.

## 1. Closed evidence

| Item | Result | Authority |
| --- | --- | --- |
| Fresh C1 program integrity | PASS: 12 cells, 34 sealed source items, six fresh separate controls, correct teacher hash, no formal raw access. | `results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json`, SHA `8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85`. |
| Fresh C1 aggregate | PASS: all 12 cells closed; all five frozen gates pass; `c1_pass=true`. | `results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/finalization/aggregate.json`, SHA `32ebde0b145c63c09c1bdbc67e48582db3e2ad588e70cb5a1d52914352607e31`. |
| Post-run provenance | PASS: two host runtime stamps, sealed status/hash closure for all 12 cells, no historical C1 artifact, formal paths unresolved. | `results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/finalization/publication_readiness.json`, SHA `fcd1e45702640178efd0c351f435406916f94bc51fa0da294480f2a5c3d31bfd`. |
| Isolation | PASS: separate source-train-only normalizers/cache namespaces; development does not enter optimizer, backward, or train loader. | Representative `closure/shared_t4_s42/run_metadata.json`, sealed per cell by the finalizer. |
| Reporting rule | PASS: fixed no-argmax epochs 5--12 scoring window. | Fresh C1 aggregate and `eval_paired_view_c1_epoch_window.py`. |
| T4 attachment | Strong: shared T4 minus shared TS4 is `+0.285474` R2 SUA and `+0.369468` pseudo-MUA; 3/3 seed and 6/6 session signs positive. | Fresh C1 aggregate. |
| Compact SUA mechanism | AC4 is sufficient relative to full T4 on old M30 SUA development and correct AC4 attachment is necessary. | Component aggregate SHA `965bfc9ed7c20d37e99cff838e0a17da67da45fc81fddfeb1d649b03ddbc2932`; AC4-RS4 aggregate SHA `97f369cf67f9d88da5d343fc4bc4b72f23284e96e0db70c0a5591af5580cd0a1`. |

This audit reran the production C1 prelaunch verifier without content-rehashing data: PASS. It also reran the focused C1 PTQ, QAT, and finalization suites with third-party pytest plugin autoload disabled: **20/20 passed**. The post-run closure separately attests content rehashing on each runtime host.

## 2. Candidate choice

### Provisional winner: shared full-T4 C1 FP32

`shared_t4` is the only current candidate with fresh 12-cell closure, passed paired SUA/pseudo-MUA non-inferiority, strong attachment gates in both views, no deployment-state increase relative to one separate T4 model, and a specified supervised backprop-free target-session calibration boundary. Its development absolute R2 is `0.574378` SUA and `0.546109` deterministic pseudo-MUA. Its shared-minus-separate deltas (`+0.008351` SUA, `+0.012543` pseudo-MUA) establish non-inferiority, not robust raw-accuracy improvement.

| Seed | sealed terminal FP32 checkpoint SHA-256 |
| ---: | --- |
| 42 | `ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f` |
| 43 | `05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22` |
| 44 | `a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6` |

| Option | Decision | Why |
| --- | --- | --- |
| Separate ordinary T4 | Not the C1 winner | It supports earlier single-view T4, but cannot make fresh shared-weight/pseudo-MUA claims. |
| AC4 `[a,c]` | Do not substitute now | Strong SUA M30 mechanism/compression, but no independent pseudo-MUA sufficiency and a different M50 C1 budget. Swapping now would select on reused development data. |
| PTQ | Not deployable under frozen hardware rule | Near-lossless accuracy, but saturation `0.022727 > 0.005`; legacy float diagnostic also fails. |
| QAT v2 | Pending | Authorized source-only fixed-8-epoch recovery; no all-three-seed aggregate yet. |
| C2 | Pending branch decision | No unique source-only lambda result or C2 prelaunch. |

The C1 statistic is a deterministic **epoch-5--12 score average**, not a development argmax and not an online eight-model ensemble. PTQ/QAT use terminal `epoch_011` as one deployable file. A new endpoint must predeclare either the fixed reporting estimator or the terminal checkpoint; it must not call the window statistic one physical checkpoint or pick an epoch after the new score. This is a future-receipt specification gap, not a C1 leak.

## 3. Open gates

### PTQ and QAT

PTQ used all three terminal C1 checkpoints and source-only scale fitting over 27 sessions × two equally weighted views. Decoder bytes remained FP32-identical, INT32 overflow was zero, mean INT8−FP32 R2 was `+0.000803` SUA and `+0.005195` pseudo-MUA, and an independent 81,300-code audit had zero final integer-code mismatches. It is nevertheless `ptq_pass=false` because maximum saturation was `0.022727` against `0.005`. Authority: `results/t4_paired_view_c1_encoder_int8_ptq_v1_20260805/aggregate.json`, SHA `40dfaaa671ec95854d2c876cbed2246dcbe29b14cdb4f1a53724a60f974ee0c9`.

QAT v2 is encoder-only and freezes decoder FP32. It seals all three seeds/both views, source-only 27-session training, no development training or epoch selection, final epoch 8, degradation threshold `-0.01 R2`, saturation `0.005`, zero overflow, zero final `E_q` code mismatch, and decoder-byte identity. Authority: `results/t4_paired_view_c1_encoder_qat_prelaunch_v2_20260805/receipt.json`, SHA `ec986e775fa3ca500247442d404df832fb0b1f460d9bb342e7cc1b717720bcdb`. At audit time only active seed logs existed. Thus a quantized deployment claim is **NO-GO pending all three reports and one frozen aggregate**. An FP32-only scientific endpoint is logically separable but must say so.

### C2

C1 passed entrance gates but neither shared-vs-separate contrast established strict exploratory improvement. C2 is permissible only after a development-blind source-only procedure uniquely freezes one lambda. No C2 result root, lambda-selection receipt, or C2 prelaunch was found. C2 is not automatically required for an external FP32 C1 confirmation; it must, however, be resolved before a new one-shot endpoint: explicitly close it if no unique lambda is selected, or complete its one sealed three-seed matrix before final candidate choice.

## 4. Leakage and interpretation

Passed boundaries: strict 27 source / 6 reused-development / 6 sealed-formal partition; separate source-only normalizers; T4 labels/rates only from chronological first 50 rewarded trials; score starts at 50; activity support first 30; zero target-session optimizer/backward/update. Offline source training does use joint encoder/decoder backpropagation. The defensible phrase is **supervised, backprop-free held-out-session calibration**, not wholly backprop-free learning.

Claim-limiting blockers rather than fresh C1 implementation leaks are: (1) the six development sessions informed many earlier design decisions and are not pristine confirmation; (2) the old started P3 receipt consumes the formal scope; (3) T4 uses target-direction labels, so comparisons with neural-only B3/SPINT are unequal-information; (4) pseudo-MUA is not native threshold-crossing MUA; and (5) window-statistic and terminal-checkpoint endpoints must remain distinct.

## 5. Minimum future external-subject prelaunch

1. Candidate-freeze receipt: bind full shared-T4 C1, objective, labels, calibration/query boundary, seeds, source map, teacher/checkpoint hashes, normalizer rule, and one reporting convention; rule out post-endpoint C2 rescue.
2. Compatibility receipt before target scoring: task, rewarded directions, unit/electrode regime, binning/window policy, variable-N behavior, and frozen train/validation/test roles.
3. Matched controls: predeclare neural-only/no-functional-descriptor and row-attachment controls; preserve matched shared/separate views when testing granularity.
4. No target score may select lambda, budget, normalizer, component, checkpoint, or quantization mode; log zero target optimizer/backward/update counters.
5. For a hardware claim, complete all-seed QAT and bind final integer packages; otherwise name endpoint FP32-only.
6. Fresh scope ID, single-use authorization, write-once root, output-hash closure, no retry and no new arm.

## 6. GO / NO-GO

| Decision | Status |
| --- | --- |
| Cite fresh C1 as development-held-out evidence | **GO** |
| Call C1 native-MUA proof | **NO-GO** |
| Call PTQ strict deployment pass | **NO-GO** |
| Call QAT deployment result | **NO-GO** |
| Execute C2 now | **NO-GO** |
| Open the existing sub-C formal scope | **NO-GO, terminal** |
| Prepare a new external-subject confirmation | **CONDITIONAL GO** after candidate freeze, C2 resolution, compatibility receipt, and controls |

## Authoritative references

- `docs/C1_POSTRUN_MECHANISM_CLAIM_AUDIT_20260804.md`
- `docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md`
- `docs/HELDOUT_BP_FREE_CALIBRATION_METHOD_CLOSURE_20260804.md`
- `docs/SUA_THREE_PATHS_ROOT_REVIEW_20260804.md`
- `docs/CURRENT_RESULTS.md`

The earlier formal-scope receipt is identified by SHA rather than by a raw-data path. This audit records no formal raw-session path, name, size, or hash.
