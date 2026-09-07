# M2 frozen RIFT baseline diagnosis

## Result

The completed frozen-encoder RIFT baseline underperformed the declared historical
same-ext4 BT concat reference. The selected EMA checkpoint is epoch 11 at
**0.3408126083** equal-session R². Endpoint 24 is **0.3366115095**. The
historical p1a_v2b M2 concat reference is **0.452135** at ext4 epoch 19, a
selected-score difference of **-0.1113223917**. This reference is contextual;
it did not supply data, weights, or a selection override for RIFT.

The selected RIFT score is uneven across ext4: 2020-10-30 Run1 = 0.4942992
(519 windows), Run2 = 0.5147304 (490), 2020-11-18 Run1 = 0.2694301 (425), and
2020-11-19 Run1 = 0.0847908 (635). Endpoint 24 is 0.4670687, 0.4213736,
0.3040534, and 0.1539503 respectively. The latter two dates account for the
visible limitation, especially 2020-11-19; no extra date scan was run.

## Audit finding

No obvious training, scale, or scoring contract failure was found in the formal
record. It trained 7 source sessions for 24 epochs, 3,165 updates per epoch
(75,960 total) using the frozen 24-epoch manifest digest
`a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a`.
Training MSE compares decoder-raw predictions to `native_covariate * 5`.
Scoring divides predictions by 5 before native-covariate R². The ext4 receipt
contains all 24 EMA checkpoints, each with 2,069 windows, and selects the
first maximum. Gradients use `source_train` only; ext4 is opened only by the
post-training scoring stage. This establishes a valid failed result, not an
explanation of the regression.

The cache contract is usable for a joint encoder experiment. Each session has
`calib_activity.npy` as float32 `[33,100,96]`, `T.npy` as the MOVE-T4 carrier
`[96,4]`, a padded raw query timeline `X_store.npy [T,96]`, and legal window
starts. R50 examples are exactly `X_store[start:start+50]`, with
`valid[start+offset >= 49]`. The existing frozen bank derives `E0 [96,50]` through the cache-generation
provider. Audit evidence now distinguishes that provider from the absent
original champion: the observed cache path loads the base and installs its
EMPTY-FiLM adapter, with native side parameters `[MOVE-T4, 0]` (`[96,8]`). It
must not be described as plain B3S with a direct `side4` post-pool route. The
frozen bank is valid as a fixed full-E0 input for matched decoder experiments,
but it must not be treated as a differentiable encoder input. A new joint path
must consume the raw 33 trials directly.

## Joint-encoder follow-up audit

The earlier activity-only-pretraining restriction was too strong. Source-supervised
FULL labels may be used for a common B3S initialization. The target calibration
law is stricter: the encoder must receive original M33 activity with
`side=zeros`, and the direct route must be zero for B. Reusing the frozen cache
FULL E0 while only zeroing a direct input is invalid.

The declared common champion path is absent from this checkout, so its
source-seven manifest and lack of ext4 exposure are not yet verified. The
champion's `load_base` followed by `install_empty_film`, the selected-head
path, and the observed cache provider must be reproduced together before any
claim of native encoder parity; a simple `post_pool(mean, side4)` construction
has not established that parity. Also, `champion.native_e0_and_u` is an inference helper: it calls `encoder.eval()`
under `torch.no_grad()`. It cannot be used for joint gradient training. A
future wrapper must implement equivalent differentiable
`reset_stream/push_trial/finalize_identity` operations and prove encoder
gradient flow. No joint wrapper was written. The matched RIFT concat cell is
now the prioritized next experiment.
