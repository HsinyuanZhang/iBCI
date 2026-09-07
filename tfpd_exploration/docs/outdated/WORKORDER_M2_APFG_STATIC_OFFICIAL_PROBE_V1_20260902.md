# Work Order: M2 APFG Static Official Probe V1

Date: 2026-09-02
Status: authorized as an exploratory packaging cell; single attempt; no network submission without explicit user confirmation
Scope: build and locally validate one EvalAI M2 candidate image deploying the frozen APFG scalar gate; CPU-only; inference-only

## 1. Origin and interface ruling

The Luna-verified build request asked for an EvalAI M2 candidate "M2-APFG-U"
deploying the frozen APFG gate under UNCAPPED growing activity memory with
decode-before-commit. Two sealed facts rule that configuration out:

1. **No legal completed-trial boundary exists.** The official continual M2
   evaluator (`falcon_challenge==1.0.2`, evaluator SHA
   `2b848f84...`, sealed audit `results/cdm_p1_m2_v1/audit.json`) calls only
   `reset(dataset_tags)` and `predict(neural)` per 20 ms bin; `on_done` fires
   only for non-continual tasks and `observe` is never called; behavior
   targets never reach the decoder. Verdict recorded there:
   `NOT_EVALUABLE_OFFICIAL_CONTRACT` (best label-free boundary detectors
   reached AUC 0.51-0.53 against a 0.65 floor).
2. **The trial-free chunk alternative already failed.**
   `results/m2_a0_chunk_noninferiority_v2/external` measured phase0-chunk
   minus true-trial at mean `-0.0322`, worst `-0.0678` (external), failing its
   pre-registered noninferiority gate. That cost is ~7x the entire local APFG
   external effect (`+0.004612`, 4/6), so a chunk-memory APFG submission is
   dominated.

The user selected the **static-pool approximation**: the offline calibration
law byte-identical to the officially scored `dopt4_static_act30` arm, plus the
APFG gate applied offline at build time. The deployed identity is
`E = native + tanh(alpha) * (post - native)` with
`alpha = -0.20759029686450958` (the `learned_refit_alpha` of the immutable V2
terminal, sidecar-verified on disk at test/export time).

Because the pool is static, this candidate is **not** the requested "U"
configuration; it is named `apfg_static_act30_dopt4` (display: M2-APFG-S).

## 2. Immutable bindings

- checkpoint `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`
  (strict SHA-guarded load through
  `sua_exploration/evalai_t4_m2/export_t4_payload.py::load_frozen_model_and_data`);
- gate adapter `tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py`,
  installed once on the in-memory frozen encoder, alpha frozen and
  `requires_grad` disabled after identity export;
- V2 same-surface graph: terminal
  `9ea88d3a4e9048c484a4e67575a3b1c9a1323fbf0558d739acb065217f2a7b63`, score
  `5dd7c3e911ca00709e028590af93c065a38f5ea1bd1369dc27fabda87b595d6e`;
- native-arm sealed anchor `ridge_activity30_m4` (external equal-session mean
  `0.2909923623168425`, within `0.6772200181830803`) and the prior official
  deployment receipt (identity bytes must match 13/13);
- native replay tolerance vs the sealed GPU screen: `5e-3` absolute per
  session (the historical CPU-replay convention).

## 3. Required sentinels (fail closed)

1. **Build time:** for every one of the 13 sessions, the APFG-ZERO identity
   (exact IEEE `+0.0`, adapter route) equals the native identity bitwise
   (SHA-256 of the `[96,50]` float32 array). The learned identity must differ.
2. **Validation:** on the same CPU process, same inputs, same batch size 1,
   the ZERO-identity decoder run must equal the NATIVE-identity decoder run
   exactly, per session, on prediction bytes, last-bin bytes, R2, target
   bytes, query starts, and window count.
3. **Validation:** the native run must reproduce the sealed screen rows within
   tolerance, proving the deployment serves the officially scored calibration.
4. **Contract:** predict-time imports of selection/gate law modules are
   blocked and must fail the run; weights and identities fingerprint-frozen
   across the replay; `on_done` a no-op; no optimizer/gradient state; the real
   host `FalconEvaluator` minival run must complete with `on_done` never
   invoked.

## 4. Pre-registered interpretation (binding)

- This cell is an **exploratory probe**, not a promotion attempt. The local
  external point estimate (+0.004612, CI crossing zero) did not pass the
  pre-registered +0.010 / 4-of-6 gate, and the RESULT V2 decision closed the
  Post-Fusion architecture axis for the paper.
- The static pool further differs from the local V2 growing pools, so the
  official held-out score of this image is not an estimate of the V2 contrast.
  It must be read descriptively against the officially scored static
  baselines of the same calibration law: `act30_dopt4` HO R2 `0.2897` and
  `act30_full` HO R2 `0.295`.
- Exceeding PF-MEAN-class or weaker baselines establishes nothing. A
  descriptive win over `act30_dopt4` would be consistent with a small
  transferable gate effect; a descriptive loss would be consistent with
  pool-size sensitivity of the transferred scalar. Neither outcome may be
  described as validating or refuting APFG beyond those words.
- Any future higher-capacity or growing-pool deployment remains governed by
  RESULT V2 section 6: a new study selected on source-grouped data only.

## 5. Cell boundary

- The cell publishes: package `submissions/evalai_m2_apfg_static_v1`
  (payload + validation artifact + receipt), local validation receipt,
  attempt/terminal receipts under `results/evalai_m2_apfg_static_v1`, and a
  built docker image tagged `spint-t4-m2:apfg-static-s42-<payload8>`.
- The cell does NOT push to EvalAI. The push is one operator-owned network
  step (`submit_evalai_apfg_static.py`), executed only after the user
  confirms, with the challenge/phase/team constants (2319 / 4599 /
  few-shot-test-2319 / HKU-ECE) and `IsTestTimeAdaptive=false`.
- No GPU access, no checkpoint mutation, no alpha retraining, no retry of a
  failed stage without a successor note.
