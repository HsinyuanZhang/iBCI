# Audit Brief: M2 AJPF-C Continual Result (for independent review)

Date: 2026-09-03
Prepared for: independent agent audit — (A) cheating/leakage, (B) compute cost
Author of the result under audit: this session's route (checkpoints/receipts below)
Status: **submission decision ON HOLD pending this audit**

## 1. One-paragraph summary

A continual-legal, label-free test-time-adaptive deployment was developed for
the official M2 contract: the decoder seeds its activity pool with the
session's public first-30 calibration block and, during the eval stream,
commits every tumbling 100-bin neural window whose mean multi-unit rate
exceeds the running median of past candidate rates ("chunk100e"), recomputing
its cached identity after each commit.  Four modules were jointly fine-tuned
from the sealed Selected-T4 checkpoint so that the TRAINING pool law equals
this DEPLOYMENT law (zero trial metadata in training pool construction).
Local replay on the official-window surface: `Ce-R1/chunk100e` external mean
R2 **0.36953** vs the sealed static champion 0.29099 (**+0.0785, 6/6
sessions positive, worst +0.0003**); predicted official held-out score
≈ 0.368–0.370 vs team best 0.295.  The pre-registered PRIMARY tier
(commit-all chunk100) FAILED its gate; the passing arm is the pre-declared
SENSITIVITY tier — a known limitation (§4.3).

## 2. Immutable evidence graph

| Artifact | Path | Content |
|---|---|---|
| Motivating probe | `results/m2_continual_chunk_probe_v1/replay.json` (+ RESULT doc) | 117 rows; gate law-robustness; chunk poison; energy gate recovers 2/3 |
| Failed root V1 | `results/m2_ajpf_c_v1/` | fail-closed before any training (exposure law unsatisfiable) |
| Training (good) | `results/m2_ajpf_c_v2/training.json` + `checkpoints/*.pt` | both laws, 12 epochs, sentinel updates, checkpoint SHAs + strict-reload state SHAs |
| Failed root V3 | `results/m2_ajpf_c_v3/` | runner stage-guard bug; no science touched |
| Score (good) | `results/m2_ajpf_c_v4/{attempt,score,terminal}.json` | 65 rows; pooled/static anchor ≤1e-7 vs sealed; gates |
| Result docs | `docs/RESULT_M2_CONTINUAL_CHUNK_PROBE_V1_20260903.md`, `docs/RESULT_M2_AJPF_C_CONTINUAL_V1_20260903.md` | narratives + numbers |
| Workorders | `docs/WORKORDER_M2_AJPF_C_CONTINUAL_V1_20260903.md` + V2 addendum | pre-registration incl. gates and tier discipline |
| Code (all additive) | `src/m2_ajpf_c_v1/{plan,chunk_law,source_stream,train,score}.py`, `scripts/run_m2_ajpf_c_v1.py`, `tests/test_m2_ajpf_c_v1.py` | reuses AJPF V1 machinery by import only |

Everything above is 0444+sidecar where published; Luna's AJPF/A0 roots were
consumed strictly read-only.

## 3. Audit axis A — cheating / leakage checks

### A1. What the training saw (claim: 7 held-in sessions ONLY)

- Materials: `source_stream.build_session_material(data_module.train_dataset, …)`
  over the 7 train sessions (`ses-2020-10-10-19/20/27/28-*`).  The 6 external
  val sessions enter only in `score_all` via `val_heldout_dataset`.
  **Auditor action:** grep `train_dataset`/`val_heldout_dataset` usages in
  `train.py`/`score.py`; confirm no external session name appears in
  training receipts (`training.json` epoch losses are per-arm scalars only).
- Loss = task-only last-bin MSE on those sessions' targets; no teacher
  forwards (AJPF step receipts record `teacher_forward_calls=0` by construction).
- Checkpoint selection: fixed epoch 12, no metric-based selection
  (`training.json.checkpoint_selection` equivalent: fixed_epoch law in code).

### A2. Decode-time causality (claim: no future/current-trial information)

- Chunks commit at bin `b`; a decode at endpoint `t` uses commits with
  `b < t` strictly (`chunk_law.committed_before`, searchsorted-left).  Unit
  tests cover this.  Targets never enter the pool (pool rows are raw neural
  windows only).  **Auditor action:** `tests/test_m2_ajpf_c_v1.py::test_chunk_law_tumbling_nonoverlap_and_causality`.
- Energy gate uses only past candidate rates (running median; unit-tested
  determinism).  No eval_mask, no behavior, no trial metadata anywhere in
  `chunk_law.py`.

### A3. Scoring fairness (claim: same harness as the officially scored baselines)

- `pooled/static` anchor reproduces the sealed `act30_dopt4` rows to
  **≤1e-7** on all 13 sessions (`score.json` lineage field) — the same
  harness whose act30 images scored 0.2897/0.295 officially.
- All arms decode the same windows/targets; only weights + pool law differ.
- **Auditor action:** recompute any session's R2 from `score.json` rows;
  re-run `run_m2_ajpf_c_v1.py --stage score --training-json …/v2/training.json`
  is one-shot-blocked (terminal present) — instead re-derive from the probe
  script pattern or verify checkpoints' SHAs then spot-replay one session.

### A4. The two honest limitations (pre-disclosed, not hidden)

1. **Tier discipline.** chunk100e was declared the SENSITIVITY tier because
   the probe that motivated it had already read external R2.  The passing
   result therefore cannot, by the workorder's own rule, be upgraded to a
   paper-level claim from this cell.  A fresh pre-registered surface is
   required for any promotion.
2. **Effect size is large (+0.079) and uniform (6/6).**  This is much larger
   than every prior memory/gate effect on this line (+0.005…+0.032).  The
   train/deploy law match is the claimed mechanism (probe: same law with
   AJPF weights = 0.228 → matched weights = 0.370), but the auditor should
   specifically attempt to break it:
   - does `Ce-NAT` (no gate parameter) also pass? YES (+0.0792, 6/6) — so the
     gain is NOT the scalar gate; it is the energy-gated memory + matched
     training.  The gate adds +0.0015 within (CI barely positive).
   - is there any channel by which the energy gate encodes target-correlated
     information beyond movement richness?  It is a threshold on mean spike
     rate — behaviorally correlated by construction (movement ↑ rate), but
     the SAME correlation is available to any TTA method and to the official
     baseline's calibration phase; no labels are read.  This is the
     intended TTA mechanism, not leakage — but it IS the point an auditor
     should stress-test on principle.
   - external sessions are short (few hundred–2.2k windows): verify the
     commits actually accrue there (probe receipt has per-session
     `commits_used`; e.g. within sessions 75+).  If external commits were
     near-zero, the gain would have to come from fine-tuning alone — but
     C-NAT (fine-tuned, chunk100) is −0.010, so fine-tuning per se does not
     explain it.

### A5. Statistical honesty

- 6/6 positive with worst +0.0003; bootstrap CI fully positive; seed-42
  10,000-resample session bootstrap (same law as all prior cells).
- Single training run, no reruns, no epoch/LR selection after external
  reads (v1→v4 root chain is all fail-closed infrastructure, auditable).

## 4. Audit axis B — compute cost

### B1. One-time training cost (already spent, GPU0)

| Item | Cost |
|---|---|
| chunk100 law (2 modules × 12 epochs) | 317 s GPU0 |
| chunk100e law (2 modules × 12 epochs) | 277 s GPU0 |
| exposure | 57,882 / 54,495 coordinates per epoch (≈63% of AJPF's 91,717 — availability-limited) |
| Reference: AJPF V2 (3 arms, trial pools) | same 12-epoch contract, comparable wall |

**Total added GPU cost ≈ 10 minutes.**  No hyperparameter sweeps were run.

### B2. Deployment-time cost per submission (the recurring cost)

- Per 20 ms bin: one decode forward (identical to the scored act30 images).
- Every 100 bins: gate evaluation (median recomputed per candidate (np.median; two-heap O(log k) only in a future deployment)) + on commit, one
  identity recompute (forward over ≤30×[100,96] through the encoder MLPs —
  CPU cost to be measured in Docker minival) — amortized ≪ 1 ms/bin; worst-case single-bin latency spikes
  by the identity forward every ~2 s.
- Memory: pool ≤30 rows (1.15 MB) + running median state — negligible.
- **No increase in model size** (same SPINT decoder; the TTA state is data,
  not parameters).

### B3. Honest compute caveats

- The energy-gated law was the SECOND law trained; a skeptic may count both
  laws as the cost of the result (10 min total — still trivial).
- The probe (30 min CPU) and two failed roots are the audit-trail overhead,
  not science compute.

## 5. Suggested auditor commands

```text
# receipts are 0444; verify sidecars
for f in tfpd_exploration/results/m2_ajpf_c_v2/*.json \
         tfpd_exploration/results/m2_ajpf_c_v4/*.json; do sha256sum $f; done

# unit tests (no data, no CUDA)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  tfpd_exploration/tests/test_m2_ajpf_c_v1.py -q

# causal-law reading
sed -n '1,60p' tfpd_exploration/src/m2_ajpf_c_v1/chunk_law.py

# training-data surface: confirm only train_dataset in train path
grep -n "train_dataset\|val_heldout" tfpd_exploration/src/m2_ajpf_c_v1/{train,score,source_stream}.py
```

## 6. Decision pending on this audit

- If audit passes: build the TTA submission package (online chunk-memory
  decoder, `IsTestTimeAdaptive=true`), local sentinel + minival, then ask
  the user for the final push (1 submission remains).
- If audit finds a leakage channel: the result is withdrawn, roots stay
  immutable, and the energy-gated law joins the closed routes.
