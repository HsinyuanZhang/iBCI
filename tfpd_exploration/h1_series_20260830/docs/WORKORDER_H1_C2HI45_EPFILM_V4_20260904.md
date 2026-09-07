# Work Order: H1 EP-FILM on C2-HI-E45 Substrate (held-in honest protocol) V4

Date: 2026-09-04
Status: **draft pending user authorization + C2-HI-E45 checkpoint transfer from
the cal-aug line owner** (user-directed combination of EP-FiLM with the honest
C2-HI-E45 arm; preregistered here before any training or data access)
Lineage: V2 (LODO 2×2, EP-FiLM +0.0239) → V3 (581866, all-source EP-FILM on
frozen C1, official HO 0.2675, below C1) → **V4 (this): FiLM on the cal-aug
line's honest best substrate**

## 1. Goal

Deploy **EP-FiLM on the C2-HI-E45 substrate** — the cal-aug line's C2 decoder
(M3-aware prefix-cycle retraining) at epoch 45, selected on the held-in surface
only, official HO R2 `0.3240` (submission `581813`, registered outside
HKU-ECE).  No MAT7 readout layer (officially −0.043 on the C1 chain), cached
identities, `IsTestTimeAdaptive=false`.  One training run, one packaging, one
submission.  This is the FiLM×cal-aug combination; its official value is
whether calibration-profile FiLM adds gain on top of the strongest honestly
selected substrate.

## 2. Substrate authority

- C2-HI-E45 checkpoint SHA-256:
  `83d56e8ce82a41ec711e0b2ec5e57d66488fae18252189123a107d5523d9352f`
  (epoch `045`, zero-based; pinned by the cal-aug line's sealed
  `selection/c2_hi.json`).  The bit-exact file must be transferred from the
  cal-aug line owner; cross-machine retraining is NOT attempted (FP32 +
  cuDNN nondeterminism across hosts breaks the SHA chain to 581813).
- Import receipt cloned from the `h1_c1_all_source_epoch49_v1` pattern; the
  extracted model-state SHA-256 and parameter count are pinned in the V4 plan
  module **before** any training runs.
- Architecture: `H1CarrierIdSpint`, h=32, W=700, `10,947,836` parameters —
  identical class to the C1 substrate; only the weights differ.

## 3. Zero-init parity gate (stronger than V3's; held-in data only)

Before training: the untrained (zero-init) FiLM on the C2-E45 substrate must
reproduce the cal-aug line's sealed `evaluation/hi/epoch_045.json`
per-recording R2 values on the 13 held-in minival recordings within tolerance.
Their numbers are imported from Git; **no held-out data is opened by this
gate**.  This verifies simultaneously: checkpoint integrity, our identity and
carrier construction matching their deployment pipeline, and FiLM zero-init
exactness.  Failure = stop, no training, no packaging, no submission.

## 4. Training (one run, GPU0, minutes-scale)

V3 training internals verbatim (import from the sealed V1 plan module): FiLM
params only, 12 epochs, lr 3e-4, batch 32, seed 42, zero-init `film_out`,
all held-in dates, M3 labeled calibration prefix, profile4 = low/high speed
state contrast, carrier4 context, frozen decoder, no target adaptation.  The
EP-ZERO anchor pass (zero-init FiLM == native identity, bit-exact) is kept as
the per-run sanity.

Post-training gates (all held-in, same criteria as V2/V3):

1. LODO gain: EP-FILM vs EP-ZERO positive under the V2 leave-one-date-out
   criterion.
2. In-sample overfit gate: in-sample session-mean delta ≤ 2× the LODO gain.

Any failed gate = no submission; the result is sealed as a negative local
receipt.

## 5. Packaging

Clone the V3 packaging with the readout stage **removed**: payload = C2-E45
checkpoint + all-source film state + cached per-session identities (built
offline from each session's three public calibration trials only); decoder
entry cloned from `h1_epfilm_spint_decoder.py` without readout apply;
container CPU smoke and host/container parity as V3 (tight threshold `1e-4`).
Base image: the existing H1 image chain with the payload swapped.

## 6. Official-outcome expectation (honest)

Primary comparison: **C2-HI-E45 official HO `0.3240`** (`581813`).  The
submission is meaningful only if it officially exceeds that.  Secondary
context: C2-E49 `0.3050` (no selection), C1 `0.2841`, V3 EP-FILM `0.2675`.
`C2-HO-E15` (`0.3760`) is **not** a comparable target: its epoch was selected
on held-out-calibration recordings that are part of the official test set
(their own preregistration labels that surface "not untouched held-out
generalization"); V4 does not chase it.  Local C1-substrate FiLM evidence
(`+0.024` LODO) does not guarantee transfer; V3 demonstrated that local LODO
gains can fail to materialize officially.

## 7. Boundaries

- Held-out calibration recordings and any local copies of official test
  recordings are **not opened** by this line, for selection or validation.
- One checkpoint, one training run, one packaging, one submission; no sweeps;
  no method changes (FiLM structure/lr/epochs frozen to the V1/V3 contract).
- Push to ECR and submission registration are executed by the user in their
  terminal; authorization covers at most one submission.
- The 2026-09-03 stop rule closed the FiLM-on-C1 accuracy line (V3 result:
  −0.0166 vs C1).  This V4 is a user-directed new line on a different
  substrate, preregistered here; it does not reopen V3.
