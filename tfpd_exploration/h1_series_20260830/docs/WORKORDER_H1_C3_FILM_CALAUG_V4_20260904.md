# Work Order: H1 C3 — CAL-AUG M3-Aware Recipe + EP-FiLM Joint Training (held-in honest selection) V4

Date: 2026-09-04  (supersedes WORKORDER_H1_C2HI45_EPFILM_V4_20260904.md, which
froze the E45 checkpoint under V3 FiLM internals; user directed full migration
of the cal-aug training recipe with FiLM added)
Status: **preregistered before any data access or CUDA init; pending GPU0 launch**
Lineage: V2 (LODO 2×2, EP-FiLM +0.0239 on frozen C1) → V3 (581866, all-source
EP-FILM on frozen C1, official HO 0.2675, additive stack dominated) →
**V4/C3 (this): the cal-aug line's proven C2 recipe with zero-init EP-FiLM
folded into joint training, honest held-in-only epoch selection, one submission**

## 1. Goal

Train one new arm **C3** = the C2 recipe (deterministic balanced identity
prefix cycle `(M7,M5,M4,M3)`, V1 plan/normalizer/batch-order bindings, seed 42,
batch 32, Adam 5e-5, weight decay 0, FP32, 50 epochs, dynamic dropout, last-bin
MSE after prediction `/20`) on the `H1CarrierIdSpint` architecture **plus the
V2-validated EP-FiLM design** (648 parameters, H=32, early pooling, profile4 =
low/high speed state contrast, carrier4 context) integrated at zero init, so
FiLM starts as the exact identity modulation and trains jointly with the
decoder.  All 50 epoch checkpoints saved.  **Epoch selection is the cal-aug
line's team-standard dual selection** (HI-M3 held-in surface and HO-M3
held-out development surface — the user has determined that held-out epoch
selection is officially sanctioned standard practice in FALCON, as exercised
by the sealed C2 submissions); the **deployment candidate is the HO-selected
epoch**, exactly the practice that produced C2-HO-E15.  Package the selected
epoch as a cached-identity deployment (no MAT7 readout),
`IsTestTimeAdaptive=false`, one submission.

## 2. Authority bindings (all local except three sealed npz binaries)

- Recipe authority: the cal-aug line's sealed V2 contract constants
  (predecessor commit `84c7aaec…`, V1 initial-state SHA `bc6dc8a0…`, V1
  dropout-count discipline, V1 source authority / schedule / batch order / M7
  schedule / source tensor / plan / normalizer / carrier-cache SHAs) —
  imported from `h1_cal_aug_m3_aware_dual_selection_v2_contract.py` (Git,
  branch `exp/h1-cal-aug-m3-aware-dual-selection-v2`).
- V1 predecessor root: the pinned local checkout
  `/tmp/ibci-h1/deployment-v1` at HEAD `5dd9bb4a…`.  Its JSON authorities are
  present, but the sealed binaries `plan.npz`, `carrier_cache.npz`,
  `schedule.npz` are Git-excluded and MUST be transferred from the cal-aug
  line owner (sidecar SHAs already pin them locally; `verify_sidecar` fails
  closed until the bodies arrive).
- Data: local held-in NWBs (`SPINT-main/data/000954`), never held-out.
- C3 integrity (self-referential, recorded in our own receipts): fresh
  substrate initial state must equal the V1 initial state SHA bit-exact with
  FiLM zero-init appended without consuming the tracked RNG stream; 206650
  optimizer steps; the dropout probability digest must equal the V1 digest
  `c1dd24d6…` (the FiLM identity path draws from the same dynamic-dropout
  structure, and the zero-init anchor restores the python RNG state, so the
  digest binding survives); per-epoch checkpoints SHA-recorded; no
  validation/selection/early stop interleaved with training.
- C3 is a new arm: bit-exact equality with the teammate's C2 run is NOT
  expected (different GPU) and NOT required; what binds C3 to their recipe is
  the contract above, not their checkpoint.

## 3. FiLM integration contract

- **Design fidelity (user requirement: no misalignment with the locally
  validated design).**  The FiLM module, conditioning inputs, and parameter
  budget are imported verbatim from `h1_calibration_profile_film_v1.core` /
  `plan` — the exact design V2 validated on the local LODO 2×2
  (EP-FiLM `+0.0239`, 4/5 dates, classification `FILM_EARLY_REPLICATION_ONLY`).
  Proof is executed, not asserted: `tests/test_h1_c3_film_fidelity.py`
  verifies constants, builder shapes/zero-init/param count, operator
  bit-equality with `film_identity(late=False)` at M=3, zero-init
  bit-equality with `H1CarrierIdSpint.carrierid_identity_projection` at M=3
  and M=7, and bit-equality of all three profile builders.  **ALL PASS**
  (commit `2740c35`).
- **Conditioning granularity** follows V2: the profile is computed **per
  scheduled support block** (the block's three trials — the same trials as
  the block's M3 carrier), not per session; the deployment payload's block is
  the session's public earliest-M3, so train and deploy share one profile
  construction.
- Zero-init `film_out` (identity modulation) at step 0: the C3 forward with
  zero FiLM must equal the plain C2 architecture forward bit-exact
  (`torch.equal`), anchored on the first scheduled batch under eval() with
  the python RNG state restored afterwards.
- Optimizer: single Adam, lr `5e-5`, weight decay `0`, one schedule over
  substrate + FiLM parameters together — the C2 recipe unchanged; the V3
  FiLM-only lr `3e-4` does not apply to joint training.
- FiLM is NOT isolated in C3 (decoder trains too).  Isolation evidence for the
  paper remains V2/V3; C3 is a deployment-strength combination arm.

## 4. Selection and gates (team-standard dual selection)

1. After training integrity passes, evaluate all 50 checkpoints plus the
   C1-e49 baseline on **both** surfaces with the cal-aug line's evaluation
   code reused verbatim: HI-M3 (held-in-calib earliest-M3 identity; held-in
   minival scoring) and HO-M3 (held-out-calib earliest-M3 identity and
   scoring), `select_epoch` tie-breaks unchanged.
2. **Deployment candidate = the HO-selected epoch**
   (`val_ho_m3_grouped/r2_mean` primary), the team-standard practice that
   produced C2-HO-E15; the HI-selected epoch is recorded as the honest
   companion arm.
3. Reference points sealed in advance: **C2-HO-E15 official HO `0.3760`
   (`581814`)** is the like-for-like anchor — same selection practice, no
   FiLM — so the submission tests whether the V2-validated FiLM adds gain on
   top of the C2 recipe under identical selection.  Secondary context:
   C2-HI-E45 `0.3240`, C2-E49 `0.3050`, C1 `0.2841`, V3 `0.2675`.  Local
   surface numbers do not guarantee official transfer.
4. Packaging parity gate (V3 pattern): container CPU smoke PASS; host/container
   minival R2 parity within `1e-4` (tight); zero-init FiLM replay == native
   identity bit-exact on the deployment path.

## 5. Packaging

Clone the V3 packaging with the readout stage removed: payload = selected C3
checkpoint + cached per-session identities (offline from each session's three
public calibration trials only) + FiLM state; decoder entry cloned from
`h1_epfilm_spint_decoder.py` without readout apply; base image = existing H1
image chain with payload swapped; `IsHeldOutZeroShot=false`,
`IsTestTimeAdaptive=false`, `IsPretrained=false`.

## 6. Boundaries

- Held-out calibration recordings are used **only** for the HO-M3
  development/model-selection surface, exactly as in the cal-aug line's
  sealed, officially accepted C2 practice (user-determined team standard);
  declared via `IsHeldOutZeroShot=false`.  No hidden-test query labels are
  read and no test-time adaptation exists.
- One training run (50 epochs, uninterrupted), one selection pass, one
  packaging, one submission; no sweeps, no recipe changes beyond the FiLM
  integration specified in §3.
- GPU: the workorder originally pinned GPU0; at launch time GPU0 was occupied
  by another agent's dandi688 seed jobs, so the run uses the idle **GPU1**
  after an ownership/idleness preflight (deviation recorded in the attempt
  receipt per multi-agent discipline).  CPU thread discipline;
  `PYTHONNOUSERSITE=1` spint env.
- Push to ECR and submission registration are executed by the user in their
  terminal; authorization covers at most one submission.
- The 2026-09-03 stop rule closed the FiLM-on-C1 line (V3 result −0.0166 vs
  C1); this V4/C3 is a user-directed new arm on the cal-aug recipe,
  preregistered here before launch; it does not reopen V3.
