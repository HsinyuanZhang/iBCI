# Carrier support audit

The requested 50/75/100 support resampling must run the frozen production
estimator on the original calibration units. It cannot be reconstructed from a
final carrier tensor: that would estimate the sampling distribution of a new,
unspecified procedure.

M2 is the only task whose estimator is fully identified from the read-only
source audit. `champion.fit_move_t4` uses chronological trial sums and the
native `t4_from_trial_sums` solve. Its source normalizer is fit across seven
held-in sessions. The available cache preserves `T.npy` and trialized activity,
but omits the neural time series, trial-change locations and direction angles
that the production function needs. Hence all M2 subsample statistics, timing,
and design condition values are deliberately absent.

M1, H1, and 688 have their production entries identified in the JSON receipt,
but lack an isolated legal support materialization in the current diagnostic
scope. This is a blocker report, not negative evidence about carrier quality.
It avoids silently changing formulas, adding a ridge to a rank-deficient solve,
or touching training/query labels.
