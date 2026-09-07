# Incident: H1 Calibration-Profile FiLM V1

Date: 2026-09-04  
Status: `FAILED_ANCHOR_POLICY_BEFORE_ANY_FILM_SCORE`

V1 completed fold-19250108 source training and sealed both 648-parameter FiLM
checkpoints.  It then opened the first outer-date record and failed before
publishing a fold or any FiLM R2 because LP-ZERO did not match the historical
LP-R3 prediction byte SHA in that process.

The immutable root is retained at
`tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v1/`.
Its body SHA256 values are:

- attempt: `1dd84e148fd03d3da472f3340662adcfbf36e7d3db366e7477075595cacf9d8f`;
- training: `e4691925908f8e7663155bee780cfeda3c2058da854bb7f36804239030af87a5`;
- EP-FILM checkpoint: `25851d58844486126f67e97b5b84ef15728e8ca571dc515dd7e2fdc0b388920b`;
- LP-FILM checkpoint: `298abb04f2018a09daf0aafbc5a845c7427a1bda774649f21ad8a8f14270e165`;
- failure: `51265f15650c6f4044d80966df94a87011c0bee79a2a9d19712feaf507b93796`.

A bounded clean-process reproduction using the same C1 and LP-R3 states and
the same first outer record subsequently reproduced both historical prediction
SHA values and both R2 values exactly.  EP-ZERO matched
`cba11d3a...13448`; LP-ZERO matched `153842a5...dc2e`; both R2 deltas were
exactly zero.  Frozen-state evidence in the V1 training receipt also shows both
substrates unchanged.

Therefore the incident does not support a pooling or FiLM implementation
failure.  It shows that a historical CUDA prediction byte SHA is too strict as
a cross-process scientific anchor for the late-pool matrix shape.  V2 may
change only the anchor policy: same-process repeat predictions remain
byte-exact; model/input/target authorities remain exact; historical R2 must be
within `1e-7`; the historical prediction-SHA equality is disclosed but does
not gate.  All training, profile, pooling, optimizer, fold, and decision laws
remain unchanged.

