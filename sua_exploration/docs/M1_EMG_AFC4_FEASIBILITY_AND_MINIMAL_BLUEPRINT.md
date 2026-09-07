# M1 source-frozen EMG AFC4: CPU feasibility and minimal implementation blueprint

**Status:** source-only CPU gate passed; no decoder, GPU, formal endpoint, or
EvalAI evaluation is authorized by this document.

## Decision

The circular T4 basis is not an adequate general form for M1's 16-D EMG output.
The following strictly source-only audit establishes that a source-frozen EMG
subspace supplies a well-defined four-coordinate analytic functional carrier
(AFC4), without claiming an M1 decoding gain.

Input scope was exactly the four native M1 `held-in-calib` source sessions
`20120924/26/27/28`.  No `held-out-calib`, minival, formal, EvalAI, external
SUA, or GPU path was opened.  The immutable audit is
`results/m1_emg_afc4_feasibility_v1/audit.json`
(`SHA-256 1eafaea9956434c38a803135ed0576f86b84e40f0648c8b832165dddfd13f57c`).

For every source-LOSO fold, the other three sessions alone determined EMG
mean, scale, PCA order, and PCA orientation.  On the left-out source session,
only calibration trials `[0,10)` fitted the per-channel affine rates.  Later
neural trials were never used as a target, score, fit input, or selection
input; the audit only verified the structural query availability `[10,end)`.

| interface | source-frozen EMG energy, mean (min) | M10 affine rank | max M10 condition | positive split-half W correlations | gate |
|---|---:|---:|---:|---:|---|
| q=2, `[w1,w2,||W||,b]` | 0.4796 (0.4560) | 3/3 in all 4 | 4.176 | 4/4 | pass |
| q=3, `[w1,w2,w3,b]` | 0.6331 (0.6078) | 4/4 in all 4 | 4.771 | 4/4 | **pass; selected** |

The predeclared q=3 preference threshold was a mean source-frozen energy gain
of at least 0.05.  Its observed gain over q=2 was **+0.1536**, so the only
selected implementation candidate is:

```text
M1-EMG-AFC4-Full(q=3) = [w1, w2, w3, b]
```

This is evidence that the *analytic functional-carrier procedure* can use a
non-circular, source-frozen task basis.  It is not evidence that AFC4 improves
M1 decoder R², replaces T4, or generalizes to a hidden test.

## Frozen estimator and sign convention

For source sessions `S` in one LOSO fold, concatenate source movement-window
trial-mean EMG vectors `e_t in R^16` and compute:

```text
mu_S    = mean_S(e)
sigma_S = population_std_S(e), replacing only zero entries by 1
z_t     = P_S ((e_t - mu_S) / sigma_S)
```

`P_S` is the first three right singular vectors of standardized source EMG,
ordered by decreasing singular value.  Each row has a canonical orientation:
find its largest-absolute loading (ties select the lowest index), and multiply
the entire row by `-1` if that loading is negative.  The target/deployment
session can project calibration EMG through this frozen `mu_S`, `sigma_S`, and
`P_S`; it cannot refit, rotate, reorder, or sign-align PCA.

For each neural channel `i`, using exactly the target session's first 10
movement-window raw-spike rates `r_i(t)`, solve the fixed closed-form ridge
model with an unpenalized intercept:

```text
r_i(t) = b_i + W_i z_t + epsilon_i
lambda = 1.0  (fixed, never selected on target or query)
```

The four-channel-row carrier is `[W_i1,W_i2,W_i3,b_i]`.  The side-feature
mean/std for a decoder must be calculated only from **source-session Full
descriptors** under the same fold, then frozen.  This makes all comparator
arms share one raw descriptor definition and one normalizer.

## Minimal joint-source decoder matrix — not yet implemented or launched

One future internal M1 source-LOSO experiment may compare exactly these four
arms using the same B3S decoder, source sessions, teacher/checkpoint policy,
trial windows, optimizer, epoch count, and four-wide side API:

| arm | raw target-session input | normalized side matrix supplied to decoder | question |
|---|---|---|---|
| `Full` | q=3 AFC4 from first 10 EMG/neural calibration trials | `normalize_source(Full_raw)` | Is EMG-aligned functional identity useful? |
| `Zero4` | no target descriptor fit | exact `[0,0,0,0]` | Is any side-path benefit merely width/fusion? |
| `RS4` | same `Full_raw` as Full | deterministic within-session permutation of complete normalized rows | Is correct channel-to-functional-row attachment necessary? |
| `B4` | same `Full_raw` as Full | `[0,0,0,Full_normalized[:,3]]` | Does baseline firing rate alone account for Full? |

Rules that keep this a valid minimal attribution matrix:

1. `Zero4`, `RS4`, and `B4` inherit the **Full arm's source-frozen PCA and
   source feature normalizer**.  They must not independently recompute a
   target normalizer or a second PCA.
2. `RS4` shuffles all four normalized coordinates as a row with a recorded,
   deterministic `(session, fold, seed)` schedule.  It does not shuffle scalar
   columns independently and does not rerun the neural/EMG fit.
3. `B4` zeroes the first three coordinates **after** source normalization and
   retains only the normalized intercept coordinate.  It is an attribution
   control, not an equal-information null.
4. `Full−Zero4`, `Full−RS4`, and `Full−B4` are the only predeclared primary
   contrasts.  q=2 is retired from the GPU matrix by the frozen CPU choice;
   it is not reopened after observing a decoder score.
5. The only permissible local query is the already audited held-in-calib
   post-support interval, beginning at trial 10.  Historical minival and
   `query_start_trial=0` overlap endpoints must never be reused.  Use a
   reviewed subset such as `[10,210)` for source selection and keep any later
   report window disjoint; exact selection/report ownership must be frozen
   before implementation.

### Required implementation seams

The next implementation should be a new isolated M1 path, not an edit to the
active RT/K4 or N4/NS4 code:

```text
streaming_calibration_exp/src/data/falcon_emg_afc4_features.py
    - source PCA fit / canonical sign-and-order receipt
    - target M10 closed-form ridge fit
    - source-only mean/std normalizer and Full/Zero4/RS4/B4 transforms

streaming_calibration_exp/src/data/falcon_datamodule.py
    - one explicit `afc4_emg` feature-group dispatch only
    - M1-only guards: M=10, no held-out-in-fit, deterministic first prefix

streaming_calibration_exp/configs/model/
    - one B3S `side_dim: 4` M1 AFC4 model inheriting the existing frozen path

streaming_calibration_exp/configs/experiment/
    - one base Full config and three matched control configs
```

Before any GPU submission, CPU tests must prove source-only PCA fitting,
canonical component signs/order, zero target normalizer reads in controls,
complete-row RS4 marginal preservation, B4 masking after normalization,
M10-only target fit, and no read of a query EMG/neural value during descriptor
construction.

## DirectRidge reference — separate baseline, never a carrier selector

DirectRidge answers a different question: how much can a session-specific
decoder obtain when it is allowed to fit a direct neural-to-EMG map from the
same M10 dense calibration data?  Its reference protocol should be:

```text
source sessions only: freeze neural-bin preprocessing, lag and lambda grid;
target first M10: use exposed paired 20-ms neural and EMG bins only;
target fit: select no hyperparameter; refit once with source-frozen lambda;
target output: predict the source-frozen standardized 16-D EMG (and invert
               source standardization only for reporting); 
query: strictly post-support held-in-calib interval, never minival/overlap.
```

DirectRidge has no side carrier, no pretrained B3S decoder, and no right to
choose q, a PCA rotation, AFC4 lambda, checkpoint, or a query window.  It is a
same-label-budget operational reference, not a competitor in the AFC4
mechanism contrast or evidence that a compact carrier is unnecessary.

## What existing SUA evidence already resolves, and the genuinely missing work

No SUA experiment needs repeating to motivate this M1 branch:

- Full T4, AC4, B4, and the matched AC4-RS4 attachment control are already
  complete on SUA.  In particular AC4 was close to T4 and AC4-RS4 collapsed,
  establishing that functional coefficients and channel attachment can matter
  in that separate task/view.
- The external `sub-M` three-arm program is already an independent completed
  cross-subject T4/Zero4/shuffle evidence chain.  It is not an M1 EMG test and
  must not be reopened or used as a tuning set for AFC4.
- The historical native-M1 T4 results cannot answer this question: their
  endpoints overlapped support.  The audited M1 post-M10 held-in-calib windows
  (366--404 trials per session) are structurally available but have not been
  evaluated with any valid M1 decoder comparison.

Therefore the smallest missing evidence is only:

1. a CPU preflight of the new M1 AFC4 loader/config/control provenance;
2. one frozen joint-source **M1** Full/Zero4/RS4/B4 matrix on a legal,
   post-M10 held-in endpoint; and
3. one separately reported DirectRidge reference on exactly that support/query
   contract.

Nothing in this document authorizes a formal/EvalAI run, a new SUA AC4/T4
experiment, an external `sub-M` rerun, an RT-core change, or any GPU work.
