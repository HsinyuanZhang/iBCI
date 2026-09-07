# Design: H1 Calibration-Profile FiLM V1

Date: 2026-09-04  
Status: `FROZEN_BEFORE_SOURCE_OOF`

## 1. Question

M2 submission 581801 established that a small, zero-initialized FiLM module can
use a four-dimensional behavioral carrier plus a four-dimensional calibration
state contrast and improve official held-out R2.  It did **not** establish late
pooling: M2 averages `pre_pool` features first, applies FiLM, and then applies
`post_pool`.

H1 supplies two independently sealed facts:

- frozen C1 is the native early-pooling substrate;
- LP-R3 is a successful late-pooling substrate, improving frozen C1 by
  +0.040018 R2 over five date-LODO folds.

This experiment asks whether the **same FiLM mechanism** transfers to H1, while
keeping pooling position as a separate factor rather than silently changing the
method between datasets.

## 2. Global method: CP-FiLM

The cross-dataset method is named **Calibration-Profile FiLM (CP-FiLM)**.

For each unit and each calibration member:

```text
f_i = pre_pool(activity_i)
q   = concat(carrier4, task_profile4)
[gamma, beta] = Linear(8, 8) -> ReLU -> Linear(8, 2H)(q)
f'_i = (1 + gamma) * f_i + beta
```

The final linear layer is initialized to exact zero.  H1 uses `H=32`, so the
module has exactly 648 trainable parameters.  M2 uses the same module with
`H=64`, hence 1,224 parameters.  The base encoder, `pre_pool`, `post_pool`, and
decoder are frozen.

The descriptor schema is also shared:

```text
task_profile4 = [high-low mean, high/low log-ratio,
                 low-state std, high-state std]
```

Each column is robust-z normalized across units.  The frozen mask is
`[1, 1, 0, 0]`, matching the official M2 winner: the two dispersion columns are
recorded but zeroed before FiLM.

The semantic state partition is necessarily task-specific, not a network
change.  M2 uses known hold versus reach trials.  H1 uses low-speed versus
high-speed bins from the same three labeled calibration trials.  A future M1
route may use phase/EMG states while retaining the same four-column algebra and
the same FiLM module.

## 3. H1 task profile

For one exact contiguous M3 support block:

1. take only eval-valid, finite 20-ms bins whose `TrialNum` is one of the three
   support trials;
2. compute seven-dimensional speed `s(t)=||velocity(t)||_2`;
3. freeze low state as `s <= q25(s)` and high state as `s >= q75(s)`;
4. require distinct finite quartiles and at least 16 bins in each state;
5. for each of 176 neural channels compute high-minus-low mean, the difference
   of `log1p` nonnegative state means, low-state population std, and high-state
   population std;
6. robust-z each column across the 176 channels using
   `max(1.4826*MAD, 1e-6)`.

No target label outside chronological first M3 may enter the profile.  This is
a labeled few-shot Tier-2 method, like H-C itself; it is not label-free TTA.

The source-only constructibility audit was completed before launch.  All 144
contiguous M3 blocks from the 13 public held-in calibration recordings were
finite and constructible.  Each block supplied 2,012--2,723 valid bins, each
quartile state supplied at least 503 bins, and the smallest high-minus-low
speed threshold gap was 0.006356.  These are geometry disclosures, not model
performance or an outer-date gate.

## 4. Pooling is a crossed factor

The identical CP-FiLM module is evaluated under two operators.

### Early/native

```text
f_bar = mean_i(f_i)
f_bar' = FiLM(f_bar, q)
identity = post_pool(concat(f_bar', carrier4))
```

This is the direct architectural replication of the successful M2 FiLM.

### Late

```text
f_i' = FiLM(f_i, q)
identity = mean_i(post_pool(concat(f_i', carrier4)))
```

This composes CP-FiLM with the sealed H1 LP-R3 substrate.  FiLM is affine and
uses a shared per-unit context, so it commutes with the trial mean; the only
remaining operator difference is whether the nonlinear `post_pool` is before or
after the mean.

## 5. Frozen 2 x 2 experiment

Five outer dates are evaluated in the existing order
`19250108/13/15/19/20`.

| arm | substrate | pooling | trainable parameters |
|---|---|---|---:|
| EP-ZERO | sealed fold C1 | early | 0 |
| EP-FILM | sealed fold C1 | early | 648 |
| LP-ZERO | sealed fold LP-R3 | late | 0 |
| LP-FILM | sealed fold LP-R3 | late | 648 |

Both FiLM arms use the same source sessions, query batches, epoch order, and
deterministic 50/50 first-M3/non-first-contiguous-M3 schedule.  Activity,
carrier, and task profile always come from the same selected M3 block.  Target
scoring always uses chronological first M3.  The two FiLM modules start from an
identical state and are optimized for 12 fixed epochs with Adam, learning rate
`3e-4`, batch 32, weight decay 0, seed 42, and the existing stride-4 source
query surface.  There is no epoch, seed, mask, rank, or learning-rate sweep.

Only the two FiLM modules receive gradients.  Both frozen substrates and the
shared decoder body must retain their pre-training state hashes.

## 6. Mandatory anchors

Before interpreting any FiLM result:

- EP-ZERO must reproduce every prior FROZEN-C1 target prediction digest and R2;
- LP-ZERO must reproduce every prior LP-R3 target prediction digest and R2;
- each FiLM arm at initialization must be bitwise equal to its own zero arm in
  identity and prediction on the first source update;
- target optimizer/backward/update counts must be zero;
- target task profile must use only chronological first M3.

Failure of an anchor invalidates the run rather than producing a scientific
null.

## 7. Predeclared readings and gates

Per outer date, compute:

```text
early_film_gain = EP-FILM - EP-ZERO
late_film_gain  = LP-FILM - LP-ZERO
pooling_at_zero = LP-ZERO - EP-ZERO
pooling_with_film = LP-FILM - EP-FILM
interaction = late_film_gain - early_film_gain
```

Each FiLM increment passes independently when its equal-date mean is at least
`+0.005 R2`, at least 4/5 dates are nonnegative, and the worst date is at least
`-0.010`.

Decision labels:

- both pass: `FILM_POOLING_ROBUST`;
- early only: `FILM_EARLY_REPLICATION_ONLY`;
- late only: `FILM_LATE_SUBSTRATE_ONLY`;
- neither: `H1_FILM_NULL_KEEP_LP_R3`.

The global M2-to-H1 replication claim requires the early increment to pass.
The late increment is the separate composability test.  Absolute LP-FILM versus
EP-FILM is descriptive because their sealed substrates already differ.

For a future H1 all-source product, select LP-FILM only if the late increment
passes.  Otherwise retain sealed LP-R3.  EP-FILM may replace LP-R3 only under a
separate predeclared all-source/product comparison; it cannot be chosen here
from absolute outer-date score alone.

## 8. Interpretation boundaries

This experiment can establish that one small carrier-plus-state FiLM mechanism
works across datasets and determine whether it is compatible with both pooling
locations.  It cannot attribute an effect specifically to H1 speed quantiles,
prove the unused std columns useful, or claim pooling invariance if only one arm
passes.

The strongest objection is that H1's state partition differs semantically from
M2 hold/reach.  The response is deliberately narrow: CP-FiLM standardizes the
descriptor algebra, dimensions, normalization, mask, zero-init module, and
injection interface; it does not pretend heterogeneous tasks share identical
behavioral labels.  Pooling is crossed explicitly rather than hidden as a
dataset-specific architecture choice.

## 9. Deferred alternatives

Rejected from this cell: profile-only FiLM without carrier, full four-column
profile, learned state thresholds, per-date target fitting, unfreezing either
identity branch, decoder fine-tuning, output-space fusion, continual target
memory, and any EvalAI packaging.  These would either break the M2 recipe or
confound the pooling test.
