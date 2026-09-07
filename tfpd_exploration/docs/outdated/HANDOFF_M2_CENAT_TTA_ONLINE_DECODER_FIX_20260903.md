# HANDOFF: M2 Ce-NAT chunk100e TTA online decoder — two bugs found, fix applied but UNVERIFIED

Date: 2026-09-03 (late)
Handoff from: the session that built and pushed submission 581749
Status: **CLOSED 2026-09-03.** Section 5 completed; replacement submission **581763**.
See `SUBMISSION_581763_CENAT_CHUNK100E_TTA_20260903.md`.
Working dir: `/home/xinyuan/Work_host/SPINT`

## 1. TL;DR

Submission **581749** (Ce-NAT chunk100e TTA) failed on the official evaluator with a
**deterministic decoder bug** — it will NEVER flip to finished; do not wait on it.
Two real bugs were found in
`tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1/cenat_chunk_decoder.py`:

1. **Wave-2 IndexError (fixed, verified by simulation no-longer-crashing)**: the
   official test phase loads 13 eval NWBs in two waves (batch 7 → waves of 7 + 6).
   `predict()` iterated `range(self.batch_size)` (=7) while wave 2 `reset` creates
   only 6 slots → `IndexError` at ~351 s (exactly the official execution time).
   Fix applied: loop over `n = len(self.slots)`; raise ValueError on row/slot
   mismatch; return `n` rows (the evaluator handles per-wave row counts — the scored
   act30 decoders do exactly this).
2. **`np.roll` wrap corruption in batch mode (fix APPLIED but NOT VERIFIED)**: the
   per-slot decode loop rolled the shared `observation_buffer` once per slot, so with
   n slots every column was rolled n times but written once.  `np.roll` wraps, so
   every slot's 50-bin decode window drifted/wrapped → near-constant predictions →
   session R2 ≈ −0.02 in batch mode.  Batch-1 parity could never catch this (1 slot =
   1 roll = correct).  Fix applied: ONE roll per predict step before the loop; write
   all rows; decode all slots; then update continual states.

**Remaining work = section 5 checklist** (verify fix → rebuild image → container
minival → new push).  The payload/weights are UNCHANGED — no re-export needed.

## 2. Verified-good foundation (do not redo)

- Science: AJPF-C continual result — local external R2 0.36953 (Ce-NAT/chunk100e)
  vs sealed static champion 0.29099 (+0.0785, 6/6, worst +0.0003).
  `docs/RESULT_M2_AJPF_C_CONTINUAL_V1_20260903.md`; training root
  `results/m2_ajpf_c_v2/training.json`; score root `results/m2_ajpf_c_v4/score.json`
  (all 0444+sidecar).  External audit verdict: conditional GO, submit **Ce-NAT**
  (not Ce-R1), `IsTestTimeAdaptive=true`.  Audit brief:
  `docs/AUDIT_BRIEF_M2_AJPF_C_CONTINUAL_V1_20260903.md`.
- Payload: `submissions/evalai_m2_cenat_chunk100e_v1/artifacts/
  t4_m2_seed42_cenat_chunk100e_tta.pkl` SHA `33181fa3c1543b9af777a17571d9c98d427b
  7e65d49375716d624e80c99603f3`.  Contents: frozen decoder module, id-encoder
  STATE DICT (rebuilt container-side by `PayloadIdEncoder` — op-for-op mirror of
  `SideFeatureEarlyPoolEncoder`: per-trial pre_pool Linear+ReLU, arrival-order
  running sum, mean, side concat, 3-layer post_pool), 13 per-tag seed pools
  [30,100,96] + sides [96,4].  Export receipt binds training receipt → checkpoint
  `c5672a2b…` → V4 score → payload SHA.  Payload does NOT need re-export for the
  decoder-code fix.
- Batch-1 parity (pre-roll-fix decoder): worst |Δpred| 5.2e-08 vs V4 scorer rows —
  the payload weights and the local scorer semantics are proven.  After the roll
  fix, batch-1 parity must be re-run to confirm the fix preserves it (the fix keeps
  1-roll-per-step semantics for n=1, so it should).
- Official submission history: 581713 (static APFG probe) **finished HO 0.2865**
  (predicted ≈0.289 — prediction band confirmed).  3 historical H1 submissions
  exist: 578473 (0.2099), 578474 (0.2615), 578689 CarrierID (0.2749) — all finished.

## 3. Root-cause analysis of the 581749 official failure

- Evidence: status `failed`, execution 351.9 s, stdout 0 bytes, stderr `[]`
  (files at `evalai.s3.amazonaws.com/.../submission_581749/`).  581658's failed
  snapshot had the same empty-files signature and later flipped — BUT 581658's
  image did not contain this bug; 581749 crashes deterministically at the start of
  evaluator wave 2.  **Do not wait for a flip.**
- The official test phase (from 581658's post-flip stderr): 13 eval NWBs under
  `/dataset/evaluation_data/m2/eval/` (7 held-in + 6 held-out), evaluator waves of
  `batch_size=7` → wave 1 = 7 files, wave 2 = 6 files.

## 4. THE INDEXING TRAP (read before touching the simulation)

Two coordinate systems are in play and mixing them produces garbage R2 that looks
like a decoder bug but isn't:

- Padded stream (local datasets, `dataset.neural_data`): rows 0–48 are the W50 zero
  pre-history; real bin k = padded row k+49.
- A local scorer window with start `s` (padded) covers real bins `[s-49, s]` and
  predicts the target at real bin `s`  (`targets[s+49]` in padded indexing).
- The online decoder at real bin `k` decodes the window of the last 50 real bins
  `[k-49, k]` → predicts target at real bin `k`.
- Therefore: local start `s` ↔ online real bin `k = s` → **compare
  `online_predictions[s]` where `s` iterates the padded start list.**

Current state of the two consumers:

- `validate_local.py` parity uses `online_predictions[starts]` — CORRECT (this is
  how the 5.2e-08 batch-1 parity passed).
- `test_phase_simulation.py` currently has `padded_pred[starts - 49, slot]` — WRONG
  (a half-thought correction during debugging; with the roll bug present, both
  indexings produced garbage, so the error went unnoticed).  **Section 5 step 1:
  change it back to `padded_pred[starts, slot]`.**

## 5. Take-over checklist (in order)

Env for every command (CPU unless stated):

```text
env -i HOME=$HOME PATH=$PATH CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=/home/xinyuan/Work_host/SPINT \
  /home/xinyuan/miniconda3/envs/spint/bin/python <script>
```

The rollout package dir (subsequent `cd` persists — use ABSOLUTE paths; relative
paths from a changed cwd caused three silent patch no-ops in this session):

```text
PKG=/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1
```

1. **Fix the sim indexer** in `$PKG/test_phase_simulation.py`:
   `padded_pred[starts - 49, slot]` → `padded_pred[starts, slot]`.
2. **Re-run the full-stream simulation** (launched-but-unverified attempt exists in
   `/tmp/cenat_sim6.log`; its numbers are invalid because of step 1):
   expect per-session R2 ≈ V4 `Ce-NAT/chunk100e` rows
   (external ≈ 0.5356/0.5044/0.3778/0.2440/0.3662/0.1934 by session) with
   worst_abs_diff ≤ ~1e-6; total wall ≈ 270 s locally.  The official eval worker
   took 351.9 s — same magnitude, so runtime is not a failure risk.
3. **Re-run** `$PKG/validate_local.py --execute` (parity batch-1 + minival).
   Parity must stay ≤1e-7.  Re-interpret minival: the previously disclosed held-in
   R2 = −0.032 was measured with the ROLL-CORRUPTED decoder and may rise to
   something sane; re-disclose whatever it now shows (do not assume the earlier
   "first-chunk poison" explanation — it may have been corruption).
4. **Rebuild the image** (payload unchanged; only `cenat_chunk_decoder.py` changed):
   `docker build --build-arg PAYLOAD_SHA256=33181fa3… -t
   spint-t4-m2:cenat-chunk100e-s42-<sha8> -f Dockerfile .` inside $PKG; verify the
   `ai.eval.method` label matches the submit helper's METHOD_LABEL byte-for-byte.
5. **Offline container minival** (the act30/cenat pattern: `docker run --rm
   --network none -v …/SPINT-main/data:/dataset/evaluation_data:ro …`), confirm it
   completes offline with sane output shapes.
6. **Ask the user, then push a NEW submission** (581749 is dead; platform quota
   healthy: day 3/6, month 9/50, total 22/100; the user's self-ration of 2 was
   already exceeded in spirit by the failed 581749 — a replacement push needs one
   explicit user confirmation):
   ```text
   /tmp/spint-e8-evalai-py38/bin/python submit_evalai_cenat_chunk100e.py --execute \
     --pin-image-id <new image id> --confirm-image-id <same> \
     --confirm-payload-sha256 33181fa3…   # payload UNCHANGED
   ```
7. **Update docs**: `DEPLOYMENT_M2_CENAT_CHUNK100E_TTA_V1_20260903.md` (sentinel
   table: add the two bugs + fixes + new sim/parity numbers; correct the minival
   narrative), `SUBMISSION_581749_…md` (add "permanent failure, root cause" note),
   and write a new submission receipt for the replacement ID.

## 6. File-by-file state

| File | State |
|---|---|
| `cenat_chunk_decoder.py` | **Both fixes applied**: wave-slot loop fix + single-roll-per-step. `PayloadIdEncoder` (container-side encoder rebuild, strict state-dict load) in place. NOT yet re-validated end-to-end. |
| `test_phase_simulation.py` | Batch-7/6 two-wave full-stream simulation works end-to-end (270 s); indexer is WRONG (section 4 step 1). |
| `validate_local.py` | Parity + contract + minival stages; `laws` path needs the APFG package on sys.path (already added in `bootstrap_runtime_imports`). |
| `export_cenat_payload.py` | id-encoder exported as state dict (`id_encoder_state`); sealed checkpoint `artifacts/Ce-NAT_epoch12.sealed.pt` (0444+sidecar); idempotent. |
| `submit_evalai_cenat_chunk100e.py` | Preflight/execute helper; METHOD_LABEL byte-aligned to the image; reads payload SHA from `receipt["chain"]["payload_sha256"]`; `IsTestTimeAdaptive=true`. |
| `artifacts/local_validation_receipt.json` | Stale (pre-roll-fix decoder) — regenerate at checklist step 3. |
| tests `tfpd_exploration/tests/test_m2_ajpf_c_v1.py` | 9 passed (no-data, AJPF-C route). |

## 7. Trap log (things that silently burned time this session)

1. **Relative-path patching**: three `python3 - <<PY` patches used repo-relative
   paths while the shell cwd had moved into the package dir → FileNotFoundError OR
   (worse) "patched" printed while writing nowhere.  Always patch with absolute
   paths and re-`sed` the region to confirm.
2. **`np.roll` wraps**: rolling a shared ring buffer per-slot corrupts every other
   slot; per-step state updates must happen exactly once.
3. **Pool-length commit detection**: at capacity 30, a commit does not change
   `len(pool)` (append+evict cancel).  The decoder now carries an authoritative
   `accepted_count`; traces must read the counter delta.
4. **Padded vs real indexing** (section 4).
5. **Stale `__pycache__`** in the package dir — removed once; keep it removed or
   set PYTHONDONTWRITEBYTECODE=1 (already in the env law).
6. **EvalAI "failed" snapshots**: empty result + empty logs can be either the
   pre-flip snapshot (581658) or a deterministic crash (581749).  Execution time is
   the discriminator hint only after you can reproduce locally.

## 8. Contacts / ownership

- AJPF science roots (`m2_anchored_joint_postfusion_v*`) and A0/chunk roots belong
  to agent **Luna** — strictly read-only for this route.
- EvalAI: challenge 2319, phase few-shot-test-2319 (id 4599), team HKU-ECE, private.
  CLI env: `/tmp/spint-e8-evalai-py38/bin/{python,evalai}`.
- Checkpoint law: Selected-T4 `25d7bc72…` (state `2a340745…`); AJPF-C Ce-NAT
  `c5672a2b…` (state `f1d89a60…`).
