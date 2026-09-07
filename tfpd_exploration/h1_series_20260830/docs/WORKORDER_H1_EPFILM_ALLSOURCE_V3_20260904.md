# Work Order: H1 EP-FILM All-Source Refit + Deployment Packaging V3

Date: 2026-09-04
Status: **draft pending H1-line owner coordination + GPU0 timing** (V2 explicitly
withheld all-source refit authorization; this successor requests it for deployment
packaging, user-directed)
Lineage: `h1_calibration_profile_film_v1` (failure) → `v2` (source-LODO 2×2,
`FILM_EARLY_REPLICATION_ONLY`, EP-FiLM +0.0239 4/5 dates) → **V3 (this)**

## 1. Goal

Produce the deployment artifact for **EP-FILM** (native early-pooling + CP-FiLM,
H=32, 648 params) trained on **all held-in dates** (no LODO holdout) and package
it as an H1 cached-identity submission image — the same contract class as
581792 (H1-M3RC / LP-R3), `IsTestTimeAdaptive=false`.

## 2. Training (one run, GPU0, minutes-scale)

- Reuse the V2 runner's training internals (`h1_calibration_profile_film_v2/
  evaluate.py` → `run_v1` internals: FiLM-only param group, 12 epochs, lr 3e-4,
  batch 32, seed 42, zero-init film_out) with the LODO outer-loop REMOVED:
  training episodes cover **all held-in dates/sessions**.
- Arms: **EP-FILM only** (LP arms are not needed for deployment; the V2 2×2
  already answered the science).  Keep one EP-ZERO anchor pass (zero-init FiLM
  identity == native identity, bit-exact) as the per-run sanity.
- Data contract unchanged: M3 labeled calibration prefix, profile4 = low/high
  speed state contrast, carrier4, frozen decoder, no target adaptation.
- Receipts: same 0444+sidecar discipline; record all film state SHAs per date
  removed==none, plus the all-source film state SHA.

## 3. Packaging (clone the M3RC pattern)

- Base image: the M3RC chain (`h1-m3rc-c1:evalai-v1` lineage) with the payload
  swapped to the all-source EP-FILM identities; decoder + interface identical to
  581792 (H1: 176ch in, 7-dim out; cached E[N,50] per session tag; zero online
  state; `IsTestTimeAdaptive=false`).
- Identity build: per session tag, EP-FILM identity from that session's public
  calib prefix (M3) + carrier + profile4 — **the same build-time computation the
  V2 scorer used for the EP-FILM arm** (reuse its functions verbatim).
- Chain: V2 training receipt → all-source checkpoint SHA → payload SHA → image
  id → parity receipt (per-session cached-identity replay must equal the V2
  EP-FILM local evaluation numbers on the overlapping surface where defined).

## 4. Local validation before push

1. Zero-init anchor: untrained FiLM == native identity, bit-exact.
2. Trained all-source FiLM: local per-date R² within tolerance of the V2 LODO
   EP-FILM readings on their own dates (the LODO models and the all-source
   model differ only in training-date coverage; large divergences are a bug).
3. Container offline minival on H1 minival files; `on_done` never called
   (H1 is non-continual — on_done IS legal here but the cached-identity
   deployment does not use it); latency finite; runtime ≪ 6 h.

## 5. Official-outcome expectation (honest)

Local LODO: EP-FILM 0.4299 vs LP-R3 0.4460 (−0.016 locally).  The local→official
offset on H1 is large (LP-R3: 0.4460 → 0.2410).  EP-FILM's official number is
therefore expected in the LP-R3 neighborhood ±unknown; the submission's value is
the official 2×2 quadrant completion (FiLM×pooling on hidden data), not a
predicted score improvement.  It does not replace LP-R3 as the sealed deployment
candidate unless it officially exceeds it.

## 6. Boundaries

- Coordination: H1 line owner's sign-off on the all-source refit (this workorder
  is that request); GPU0 after ownership check; GPU1 untouched.
- One training run, one packaging, one submission; no sweeps; no method changes
  (FiLM structure/lr/epochs frozen to the V2 contract).
- No claims beyond: "official 2×2 completion on hidden data".
