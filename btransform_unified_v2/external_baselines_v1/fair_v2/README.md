# Fair gradient-free controls, v2

This directory replaces the frozen single-session, `ridge=1` v1 experiment.
The six v1 images remain archived and are ineligible for submission. This
version fits a common kind of linear readout on **all held-in recordings**,
uses native neural calibration bins, and chooses hyperparameters exclusively
on source validation. Results and their receipts are generated under `results/`.

## Fixed protocol

All linear arms first smooth every full, unpadded raw 20-ms recording with
FALCON's normalized 12-tap causal exponential kernel (`tau=240 ms`,
`extent=1`). There is one zero-initialized filter state per recording and no
trial reset. Each recording's neural mean and standard deviation are fit on
its selected calibration bins **after the same filtering**. Support is taken
from the first 33 trials for M2, first 10 for M1, and first 3 valid `TrialNum`
trials for H1. No interpolation is used. Trial IDs and eval masks select neural
support; target behavior never enters normalizer or alignment fitting.

Training pools the canonical source supervision used by the existing static
RIFT controls: M2 has 7 recordings / 101,171 windows; M1 has 4 / 213,336;
H1 has 13 / 23,212. Each source recording is split chronologically 80:20 by
eligible endpoint. A 21-native-bin separation between the final fit endpoint
and the first validation endpoint prevents overlap of the linear arms' causal
receptive fields. Validation uses an equal-recording mean of sklearn's
per-output-centered variance-weighted R². The final readout is then refit on
all eligible source rows. Intercepts are unpenalized; behavior is not z-scored.

| Arm | Representation | Total history bins | Source-selected parameters |
| --- | --- | ---: | --- |
| `wf_zs_h0` | Session neural z-score | 1 | Ridge alpha |
| `diag_z_wf` | Session neural z-score | 10 | Ridge alpha |
| `coral_wf` | Session z-score, then CORAL to the reference session | 10 | Ridge alpha and covariance shrinkage |
| `aligned_fa_wf` | Session z-score, per-session FA, stable-row rotation, all-electrode posterior | 10 | Ridge alpha, FA dimension and stable fraction |
| `aligned_fa_stable_wf` | Same FA/rotation, posterior inferred only from stable electrodes | 10 | Ridge alpha, FA dimension and stable fraction |

The grids are `alpha ∈ {1e2,1e3,1e4,1e5}`, CORAL shrinkage
`∈ {0,0.1,0.5,1}`, FA dimension `∈ {10,20,40}`, and stable fraction
`∈ {0.5,0.75,1}`. CORAL's numerical ridge is fixed at `1e-3`.
CORAL and FA align every non-reference source recording to the same latest
source reference (`max(train)`), then pool **all** source recordings for the
readout. The reference recording is not the only supervised training source.

FA uses covariance sufficient-statistics EM, three deterministic starts,
per-sample log-likelihood tolerance `1e-6`, and a normalized-feature private
variance floor of `1e-6`. The declared iteration budget is 10,000 with one
50,000-iteration retry at the identical tolerance. Non-convergence is an
error, not a successful fit. Stable-row selection first screens both loading
norms at `0.01`, then uses iterative orthogonal Procrustes pruning. All fitted
parameters and convergence diagnostics are retained.

The all-electrode posterior is the form used by the Degenhart author release.
The stable-only posterior is a separately named robustness variant requested
for this experiment; it is not mislabeled as an exact restoration of the
published algorithm. Solver settings and normalized input also differ from
the original raw-count MATLAB experiment. The derivations and primary-source
links are in [the implementation audit](../docs/OFFICIAL_WF_AFA_AUDIT.md).

`wf_zs_h0` matches the official demo's default history length and causal
filter, but deliberately uses a consistent calibration-statistic surface and
the source-only selection protocol above. It is a **FALCON-style local
reproduction**, not a literal rerun of the official demo. That demo uses a
different alpha grid/default K-fold score and switches from smoothed training
statistics to raw session statistics at deployment. Published private-test
WF-ZS scores cannot be treated as scores on this public calibration face.

## Metrics and paired neural-network controls

Every report includes standard variance-weighted R² and the historical
flattened R², with per-recording and pooled values. H1 additionally groups its
14 held-out recordings into 7 sessions; this grouped-seven result is the H1
main result. CORAL's reported increment is relative to `diag_z_wf` on the same
metric and aggregation. A numerical improvement over the frozen v1 WF is not
evidence for cross-channel alignment.

The static-RIFT controls use the existing fixed-final EMA checkpoint for each
task and compare `identity`, `diag_z`, and CORAL with identical frozen network
parameters. They map target raw support to the pooled source raw-support
distribution **before** the existing convolutional frontend. They do not add
the WF smoothing to a network trained on raw inputs. These controls preserve
model capacity and isolate the input-calibration change.

The RIFT comparison distinguishes a target-independent fixed-final EMA rule
from the historical local-held-out earliest-maximum rule. The latter inspected
public calibration query labels for epoch selection and is reported as a
separate selection regime. Extracting a fixed-final result from the existing
curve is a retrospective comparison rule, not a claim that no one previously
examined the curve. Neither regime used private official-test labels.

H1 retains the explicit assumption that corresponding unit-row positions are
used across dates; physical channel correspondence is unverified. The local
public target queries overlap the neural calibration recordings. Their
behavior labels are used for scoring only in this source-selected v2 run.

## Run

From the workspace root, use a fresh output directory:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/external_baselines_v1/fair_v2/run.py \
  --task m2 \
  --dest btransform_unified_v2/external_baselines_v1/fair_v2/results/m2_v2
```

Use `m1` or `h1` and its own fresh destination for the other tasks.
`protocol.json` binds code/configuration before training, and `selection.json`
is sealed before the runner opens target evaluation data. The final receipt
binds source inputs, calibration provenance, fitted models, candidate scores,
selected hyperparameters, target predictions, and both metric definitions.
This runner does not push images, register methods, or submit to EvalAI.
