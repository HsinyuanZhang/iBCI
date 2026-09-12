# Fair v2 M2 aligned_fa_stable_wf

Self-contained CPU EvalAI context. Numerical inference uses NumPy only.
The payload is numeric: no source query labels, raw NWB files, or training pickles.

## Protocol

Fair-v2 AlignedFA STABLE-SUBSET POSTERIOR variant + WF (not the author-form all-electrode posterior). Shared WF: 240 ms causal exponential smooth (FALCON 12-tap, tau=240ms, extent=1) on each full unpadded raw 20-ms recording from zero state, no trial reset; per-session z-score from that session's native calibration bins after the same filter; then aligner; 10-bin causal history; unpenalized-intercept ridge readout. No interpolated support. Source-selected K=40, stable fraction=1.0. Fits passed convergence checks. Solver settings and normalized input differ from the original raw-count MATLAB experiment. Local standard R2=0.081052 (not official). New linear baselines used source-only hyperparameter selection (80/20 chronological per source recording, 21-native-bin gap). Query labels did not enter fit or selection. Existing official RIFT candidates used public calibration query labels for epoch pick; that is a different selection budget. Fixed-final RIFT rows are local diagnostics only and do not replace existing official RIFT candidates.

## Build

```bash
docker build -t fair-v2-m2-aligned_fa_stable_wf:cpu .
```

Do not docker push or EvalAI-submit from this directory without the parent queue.

