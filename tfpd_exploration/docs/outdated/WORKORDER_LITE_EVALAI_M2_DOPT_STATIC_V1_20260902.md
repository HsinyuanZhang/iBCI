# Work order (lite): EvalAI D-opt-4 static M2 submission package v1

Date: 2026-09-02.  Cell: `EVALAI_M2_DOPT_STATIC_V1_PACKAGING`.
Status: packaging + local validation only.  No network submission is performed
by this cell; the push is the operator's single explicit step (Section 7).

## 1. System under deployment

The sealed M2 static lineage: frozen checkpoint `25d7bc72...` of
`m2_spint_t4_mainline_fp32_v1` (seed 42, epoch 2, B3S encoder + frozen coupled
SPINT decoder, T4 normalization `d17f5f4c...`), deployed as:

1. **Calibration phase (offline, build time, per session)**: greedy forward
   D-optimal selection of `k=4` trials from the finite-angle candidates
   inside the first 30 labelled calibration trials — the frozen law of
   `tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py:17-35`
   (`greedy_forward_d_optimal_indices` from
   `sua_exploration/mc_maze/d_optimal_calibration_design.py:188-219`);
2. ridge `lambda=0.1` T4 fit on the selected four trials
   (`fit_ridge_t4`, `tfpd_exploration/src/calibration_budget_comparators_v1.py:56-95`,
   sealed import; side assembly `physical.py:38-71`);
3. B3S activity pool = exactly the four selected trials (the B3S encoder
   mean-pools them internally);
4. frozen-checkpoint `compute_identity` -> one cached `E[96,50]` per session.

This is the sealed screen's `ridge_static_m4` cell, deployment-shaped.

**Runtime (EvalAI container)**: `reset(dataset_tags)` selects the cached
identity; `predict` decodes one 50-bin window with the frozen decoder;
`on_done` is an explicit no-op.  Zero online updates, no calibration data, no
selection law, no optimizer inside the runtime.  Structure mirrors the
previously deployed and officially scored images (`sua_exploration/evalai_t4_m2`,
`sua_exploration/evalai_t4_m2_activity_budget`); the only change is the
offline T4 selection law (chronological full block -> D-opt-4) and the static
four-trial activity pool.

## 2. Contract facts (sealed audit `results/cdm_p1_m2_v1/audit.json`)

- Continual M2 evaluator calls only `reset(dataset_tags)` and
  `predict(neural_observations)`; `on_done` fires only for non-continual
  tasks; `observe` is never called by the evaluator loop.
- Labelled calibration NWBs (`trial_target_angles`) belong to the official
  calibration phase; the query stream does not.  The D-opt selection therefore
  runs at build time from the public calibration files, exactly like the prior
  deployed images, and the container receives only dataset tags at runtime.

## 3. Artifact tree

`tfpd_exploration/submissions/evalai_m2_dopt_static_v1/`

| File | Role |
|---|---|
| `laws.py` | verbatim mirror of the frozen D-opt law (+ sealed imports for the ridge laws) with file:line provenance |
| `export_dopt_static_payload.py` | offline calibration phase -> payload + receipt |
| `dopt_static_decoder.py` | the `BCIDecoder` runtime (cached identity; on_done no-op) |
| `decode.py` | container entry point |
| `Dockerfile` | image build (base `spint-m2:e8-epoch027-76f0fb2`) |
| `validate_local.py` | local contract validation + reference scoring |
| `artifacts/t4_m2_seed42_dopt4_static_identity.pkl` | the payload (frozen decoder + 13 cached identities) |
| `artifacts/*.receipt.json` | export/local-validation receipts |
| `submit_evalai_dopt_static.py` | guarded push helper (read-only by default; operator executes) |

Receipts: `tfpd_exploration/results/evalai_m2_dopt_static_v1/{attempt,terminal}.json`
(0444 + sha256 sidecars).  Tests:
`tfpd_exploration/tests/test_evalai_m2_dopt_static_v1.py` (no-data, no-CUDA).
Driver: `tfpd_exploration/scripts/run_evalai_m2_dopt_static_v1.py`.

## 4. Local validation gates (all must pass)

1. Payload audit: exact key set, 13 identities `(96,50)` float32, arm binding
   `dopt4_static_m4`, metadata is pure JSON provenance (no calibration arrays).
2. Export gates: per-session D-opt selection, activity bytes, and normalized
   ridge T4 bytes bitwise-equal to the sealed screen `ridge_static_m4` rows;
   cached-identity deployment bit-exact vs the direct T4 path and the
   standalone decoder (max abs 0.0 on all 13 sessions).
3. Evaluator-semantics replay (reset+predict only, batch 1, per bin) on the
   six local held-out query sessions (`external_official_query` surface) and
   the seven held-in post-30 surfaces (`within_post30`): per-session
   variance-weighted R2 within `5e-3` of the sealed static_m4 rows; the
   external equal-session mean is the sanity anchor
   `0.22271999429945652`.
4. Contract assertions: selection-law imports poisoned during the replay;
   frozen-weight and identity fingerprints byte-identical before/after; no
   gradient/optimizer/calibration state; `on_done` no-op.
5. Host `FalconEvaluator` (local, minival, batch 7; evaluator module sha
   `2b848f84...` as sealed in the audit) runs end-to-end; the instrumented
   decoder proves the loop never calls `on_done`.
6. Container: image builds from the validated base; local-path minival run
   reproduces the host metrics; remote-path simulation writes
   `/submission/submission.csv` with the 7 minival session keys and enters the
   evaluator's expected 300 s post-write wait.

## 5. Surfaces (disclosure)

- `external_official_query` / `within_post30` are the sealed screen's local
  surfaces (held-out calibration NWB query streams / held-in post-30 windows).
- The official EvalAI hidden test surface is different (secret query files,
  evaluator masking and date-grouped aggregation); the local numbers are a
  mechanism check, not a prediction of the official score.
- The host minival number is an aggregation the evaluator defines (four
  date-level R2 scores averaged); it is reported for contract-comparison
  against the prior deployed images (E8 anchor held-in 0.5238; M4/M10/M30
  activity30 arms 0.6364/0.6354/0.6378), not against the screen.

## 6. Expected local outcome

Static-M4 identities are weaker than the activity30 arms: expect held-in
minival well below the 0.635-0.638 band of the prior ridge images and the
external replay at the sealed static anchor `~0.2227`.  This is the known
cost of the strictly static four-trial pool; it is the point of the
deployment (a causal, four-label, no-online-update baseline), not a defect.

## 7. Operator submission commands (the only network step)

The account/token are the operator's; this cell never pushes.

1. Preflight (read-only):

```bash
/tmp/spint-e8-evalai-py38/bin/python \
  tfpd_exploration/submissions/evalai_m2_dopt_static_v1/submit_evalai_dopt_static.py
```

2. Formal push + private registration (challenge 2319, phase
   `few-shot-test-2319`/4599, team HKU-ECE), only after explicit
   authorization, with the frozen image ID from the terminal receipt:

```bash
/tmp/spint-e8-evalai-py38/bin/python \
  tfpd_exploration/submissions/evalai_m2_dopt_static_v1/submit_evalai_dopt_static.py \
  --execute --confirm-image-id <terminal.json docker.image_id>
```

The helper fails closed on image-ID, payload-label, phase, size and quota
drift, pushes to the challenge ECR repository with a UUID tag, verifies the
uploaded manifest digest equals the frozen local image ID, attaches the
required metadata (`IsHeldOutZeroShot=false`, `IsTestTimeAdaptive=false`,
`IsPretrained=false`), and refuses duplicate registration.  Plain-CLI
equivalent (SPINT-main/README.md): `evalai push <tag> --phase
few-shot-test-2319 --private`.

3. After the official result returns, archive the submission id/receipt next
   to this work order and do not create duplicate submissions.

## 8. Boundaries

- Read-only on every sealed root (screen results, audit, checkpoints,
  previously deployed image trees); all new files are additive under
  `tfpd_exploration/`.
- No GPU anywhere (CUDA_VISIBLE_DEVICES empty for every stage).
- No parameter updates, no target gradients, no query labels.
