# Result — M2 Trial-Shaped TTA Diagnostic V1: the online-update route is closed

Date: 2026-09-03
Status: **completed; pre-registered gate FAILED catastrophically (0/7 sessions)**
Receipt: `tfpd_exploration/results/m2_trial_shape_tta_diag_v1/diag.json` (0444+sidecar)
Checkpoint: Ce-NAT epoch12 `c5672a2b…` (strict-load verified); static reference = same weights, seed-only identity.

## Design (pre-registered, single configuration, no sweep)

Per the honest cut: keep the B3S encoder unchanged; test whether ANY legal
label-free online identity update helps.  Query = trials 34+ of each held-in
calibration recording (seed = trials 1–33, the deployed calibration policy);
decoder-driven bin-by-bin; commits are **trial-shaped fragments** cut from the
raw 20 ms stream by silence gating (active bin ≥ 0.25 × median seed-row rate,
gaps ≤ 25 bins tolerated, length 20–300 bins, linear resample to 100), pushed
after the decode at their closing bin.  Capacity: seed 33 + accumulating
pushed rows (no eviction; commit counts disclosed).  Pass gate: mean Δ ≥
+0.01 with majority positive over the 7 held-in sessions.

## Result

| Session | Static (seed-33) | TTA | Δ | Commits |
|---|---:|---:|---:|---:|
| 10-19-Run1 | 0.69203 | −1.02656 | −1.71859 | 2 |
| 10-19-Run2 | 0.72836 | −1.05412 | −1.78249 | 1 |
| 10-20-Run1 | 0.66358 | −0.82275 | −1.48633 | 1 |
| 10-20-Run2 | 0.67674 | −0.89552 | −1.57226 | 1 |
| 10-27-Run1 | 0.68984 | −1.05541 | −1.74526 | 1 |
| 10-27-Run2 | 0.67591 | −0.91837 | −1.59427 | 1 |
| 10-28-Run1 | 0.71849 | −1.20028 | −1.91877 | 1 |

**Mean Δ = −1.688; 0/7 positive; gate FAILED.**  A single pushed fragment
collapsed session R2 from ≈ +0.7 to ≈ −1.0 immediately.

## Scale sanity check (rules out an implementation artifact)

Resampled trial-34 row mean 0.1125 vs calib seed-row mean 0.1269 (ratio 0.89);
max values 6.45 vs 6.87.  The pushed rows are magnitude-matched to the seed
rows.  The collapse is the encoder/decoder's true sensitivity to any
identity-row perturbation, not a scaling bug.

## Conclusion (per the pre-registered decision rule)

> This MLP is a **static identity encoder**.  It is not an online pool
> updater.  Every row-shape family tested — raw tumbling 100-bin windows
> (chunk100/100e), energy-gated chunks, silence-gated trial-shaped fragments
> resampled to 100 — degrades decoding when committed online, from −0.03
> (matched-training chunks) to −1.7 (trial-shaped fragments).  Its proven
> competence is turning the calibration 30–33 trials into one static identity;
> its failure mode under any online update is immediate and large.

## Closures this implies

1. **M2 continual/TTA deployment: closed.**  All three row families and both
   training regimes (unmatched AJPF, matched AJPF-C) fail locally, and the
   official 581763 (HO 0.233, below every static champion) confirms the
   negative on hidden data.  Static champions stand: act30_full 0.295 /
   act30_dopt4 0.2897 (official), static APFG probe 0.2865.
2. **V4's local +0.0131 method contrast** stands as a same-recording
   continuation result only; the official test cannot be reached by any
   current update law.
3. The only remaining growing-memory surface in FALCON is **B1/H2
   (non-continual, real on_done boundaries)** — and even there, this
   diagnostic warns that pushed-row identity updates are the fragile part;
   any B1 attempt should first reproduce the static identity exactly and add
   updates only behind the same pre-registered gates.

## Provenance

- Diagnostic script: `tfpd_exploration/scripts/run_m2_trial_shape_tta_diag_v1.py`
- Supersedes the raw-window CAMR diagnostics (never executed; superseded
  before launch by the strategy correction).
- 581763 (HO 0.233) remains the last M2 submission; user-rationed quota spent.
