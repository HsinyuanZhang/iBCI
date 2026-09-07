# H1 CarrierID: carrier-quality diagnostic program

## Status and scope

This program separates a deliberately leaky **diagnostic** from two possible
source-legal feature designs.  It neither changes the sealed CarrierID H-C
checkpoint nor authorizes a GPU experiment.

The source-only feasibility receipt is
[`H1_CARRIERID_QUALITY_SOURCE_ONLY_CPU_PREFLIGHT_v1.json`](../pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUALITY_SOURCE_ONLY_CPU_PREFLIGHT_v1.json)
(SHA-256 `0c6ece5a6d24cbcf97d34eb0ba38b5e62eacbd759bb2cae1e896920fc5396d52`,
mode `0444`).  It opened exactly the 11 fold-0 source held-in recordings;
target, minival, formal, EvalAI, CUDA, Trainer, checkpoint creation, and
selection were all absent.

The explicitly leakage-marked query-carrier diagnostic has now also been run once on the already
opened fold-0 development target. Its immutable receipt is
[`H1_CARRIERID_QUERY_ORACLE_LEAKAGE_DIAGNOSTIC_v1.json`](../pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUERY_ORACLE_LEAKAGE_DIAGNOSTIC_v1.json)
(SHA-256 `b45acbbfd9aae74f898e3aa7652b22b025677f15f04d16e77f3dca0d4b315431`,
mode `0444`). The ordinary support carrier scored `0.5255107931`; replacing only the carrier with
one fitted on the first four query trials scored `0.5226522069`, for an oracle-minus-support delta
of `-0.0028585862`. Per-recording deltas were `-0.0039902708` (6,735 samples) and
`+0.0004078519` (2,230 samples). This is a negative sizing diagnostic: a query-local M=4 block under
the same frozen estimator and frozen H-C consumer did not expose useful headroom.

It is not a mathematical no-go for a decoder-loss-trained estimator. The diagnostic still uses the
same fixed PCA/ridge/U/EB estimator, fits only four query trials, and feeds a consumer trained on
ordinary support-carrier distributions. It therefore establishes only that this frozen consumer did
not benefit from a test-time query-local replacement; it does not upper-bound a different
source-learned analytic operator.

The follow-up distance audit rules out the trivial explanation that the two carriers were nearly
identical. Its immutable sidecar is
[`H1_CARRIERID_QUERY_ORACLE_DISTANCE_AUDIT_v1.json`](../pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUERY_ORACLE_DISTANCE_AUDIT_v1.json).
For the two recordings, shrunk-carrier median per-channel cosine is `0.6815/0.4243`, flattened cosine
is `0.6050/0.4213`, and relative Frobenius change is `0.8270/1.0509`. Subtracting the frozen source
prior `mu` changes these conclusions negligibly. Thus the intervention materially moved the carrier
while pooled R2 changed by only `-0.00286`.

This supports local insensitivity or distribution mismatch in the frozen consumer, not estimator
saturation. The earlier disposition that used the test-time replacement as a mandatory estimator
headroom gate is withdrawn. Estimator-side and consumer-side hypotheses remain unresolved until a
fresh support-carrier consumer and a fresh query-local-carrier consumer are trained from the same
initialization/source schedule and evaluated under their matched carrier distributions. That entire
comparison remains leakage-diagnostic-only and cannot enter a main result.

A paired prediction audit further shows that the consumer is not strictly invariant but attenuates
this carrier direction. The pooled prediction-change RMS is `0.0002510`, equal to `6.06%` of target
centered RMS and `11.34%` of the support-carrier prediction RMS; flattened prediction correlation is
`0.99380`. Per-recording correlations are `0.99492/0.98507`. Preserve
[`H1_CARRIERID_QUERY_ORACLE_PREDICTION_DISPLACEMENT_AUDIT_v1.json`](../pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUERY_ORACLE_PREDICTION_DISPLACEMENT_AUDIT_v1.json)
(SHA-256 `0b0940d093778ac95d46fdbadeff1b15bbf5c5126c8102f747e8d4d71ec82d09`,
mode `0444`). This supports local attenuation but still cannot distinguish limited consumer capacity
from a carrier-distribution mismatch learned by the frozen post-pool path.

The follow-up must preserve three distinctions:

| Item | Target labels used? | Purpose | Eligible for selection/main result? |
| --- | --- | --- | --- |
| Query-carrier oracle | Yes, intentionally | Diagnose how much the sealed H-C can use an unrealistically local carrier | No — `LEAKAGE_DIAGNOSTIC_ONLY` |
| C1 confidence | Only ordinary support trials 1–4 | Add an analytic carrier-uncertainty input | Potential future source-trained candidate |
| Q1 split quality | Only support trials 1–4 | Measure whether a carrier inferred on trials 1–2 agrees with trials 3–4 | Potential future source-trained candidate |
| P1 source-date prior | Source recordings only | Test whether channel-quality has a stable cross-date component | Comparator only until Q1 has evidence |

## 0. Query-carrier oracle: completed leakage diagnostic, not a result

Implementation: [`h1_carrierid_oracle_carrier_diagnostic.py`](../scripts/h1_carrierid_oracle_carrier_diagnostic.py).

The script requires `--leakage-diagnostic-only`; otherwise it refuses target
access.  Before opening target data it binds:

- the immutable sealed H-C epoch-49 checkpoint and saved resolved config;
- the immutable existing terminal receipt, including H-C checkpoint/config
  SHA-256 and strict query-window SHA-256; and
- the reconstructed source-only transform and normalizer to the checkpoint
  metadata.

For each target recording it leaves the ordinary first-four-trial support
identity untouched, takes the next four query `TrialNum` values, fits the
existing frozen four-dimensional carrier estimator from their neural rates and
velocity labels, and replaces **only** H-C's carrier input.  It then evaluates
the ordinary support-carrier H-C and query-oracle H-C using the same sealed
checkpoint, same identities, same query-window hash, and same per-recording
sample counts.

The receipt reports support versus oracle R² per recording and pooled, all
sample counts, query-fit trial values, query-label-bin counts, and explicit
fit/evaluation overlap.  It records:

```text
status = LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT
model_selection = false
paper_main_result = false
target optimizer steps = 0
target backward steps = 0
```

This is an approximate *carrier-channel diagnostic ceiling*, not a formal
upper bound.  The H-C checkpoint was trained with ordinary support carriers;
changing its target-time carrier is not equivalent to an honestly trained
quality-feature model.  In particular, an observed difference such as `0.0102`
is an observation, not a mathematical ceiling on an honest source-trained
method.

The oracle is intentionally not run by CPU preflight.  It opens target query
labels, so it must be invoked explicitly as a diagnostic after review; it
never runs as a training or selection dependency.

## 1. Legal candidate A: C1 analytic carrier confidence

The existing M=4 EB carrier fit already returns a channel-wise posterior
shrinkage weight `w_i ∈ [0,1]`.  C1 would supply:

```text
ordinary carrier_i = C_i( support trials 1–4 )      # existing [4]
confidence_i       = z_src( w_i )                   # new [1]
CarrierID input_i  = [ ordinary carrier_i, confidence_i ]  # [5]
```

`z_src` is a scalar normalizer fitted over source recordings only.  The model
change is tightly bounded: CarrierID's carrier dimension changes from 4 to 5,
adding only 32 weights to the first carrier-post linear layer.  It requires a
new source-trained checkpoint; the sealed H-C cannot be retrofitted and must
not be compared through target-time mutation.

CPU evidence is weak for C1 as a priority direction.  Across 11×176 source
channel examples, confidence is non-degenerate (mean `0.8014`, SD `0.1571`),
but its pooled Pearson association with Q1 agreement is only `0.0284`.  The
per-recording associations range from `−0.1785` to `+0.1928`.  Thus C1 is
constructible but does not currently establish that EB shrinkage weight is a
useful proxy for forward-transfer quality.  It should not outrank Q1.

## 2. Legal candidate B: Q1 support-split forward-transfer quality

Q1 avoids query leakage.  It uses the same four support trials that ordinary
H-C already receives:

```text
fit 1–2:   frozen source PCA + ridge → raw carrier c(1,2)_i
score 3–4: frozen source PCA + ridge → raw carrier c(3,4)_i
q_i = (cosine(c(1,2)_i − μ_src, c(3,4)_i − μ_src) + 1) / 2

ordinary carrier_i = EB_M4( trials 1–4 )            # unchanged
CarrierID input_i  = [ ordinary carrier_i, z_src(q_i) ]  # [5]
```

Trials 3–4 score the stability of the 1–2 estimate; they do **not** replace
the ordinary M=4 carrier and are never query trials.  The quality value is
therefore legal at deployment under the existing M=4 support budget.

The source-only audit shows Q1 is numerically viable: all source values are
finite, its mean is `0.6554`, SD `0.2533`, and range is `0.0032–1.0000`.
The fixed pre-GPU kill rule is to stop Q1 if any feature is non-finite, its
source normalizer is degenerate, or fewer than 95% of channels in any source
recording have a defined split cosine.  This receipt passes that constructible
screen, but it is not an accuracy result and does not authorize training.

## 3. P1: source-date per-channel prior

The CPU audit also constructs a static vector

```text
p_i = mean_source_records q_i
```

and tests it with leave-one-source-date-out (five source dates, eleven source
recordings).  The held-date prior is always estimated from the other source
dates.  Its source-date cross-validation is positive but only moderate:

- pooled LODO channel Pearson: `0.2617`;
- median held-recording LODO Pearson: `0.3049`;
- every held source date has a positive mean held-recording correlation
  (`0.2479–0.3389`).

The immutable candidate vector is
[`H1_CARRIERID_QUALITY_SOURCE_DATE_PRIOR_v1.npy`](../pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUALITY_SOURCE_DATE_PRIOR_v1.npy)
(SHA-256 `ee1f2c28ec0f76eb1bc24cdfa9d94fae4f1282dc5459a34b0ed7175aca71f01a`).

P1 is not a reason to revive an unrestricted electrode lookup table: it is a
specific Q1-derived source-date prior.  It should appear only as a fixed
comparator to Q1, and should be dropped before GPU if its pooled LODO Pearson
or median held-recording LODO Pearson is non-positive, or if fewer than three
of five held dates have positive mean correlation.

## Pre-GPU decision discipline

1. Do not use the query oracle to choose C1, Q1, P1, an epoch, or a width.
2. Do not concatenate Q1 and P1 in the first candidate; that would conflate a
   session-specific support statistic with a static source prior.
3. If a future source-trained Q1 model is authorized, compare only a fixed
   `carrier_dim=5` Q1 arm against a matched `carrier_dim=4` H-C source fit,
   with fixed seed/epoch and no target-driven selection.
4. C1 is lower priority than Q1 because its present source association with
   split forward-transfer quality is close to zero.
5. A failed CPU kill rule ends its branch.  No target score may rescue it.
