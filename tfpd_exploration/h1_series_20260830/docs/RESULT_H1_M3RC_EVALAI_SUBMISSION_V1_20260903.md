# H1-M3RC EvalAI Submission V1

Date: 2026-09-03  
Status: `FINISHED_NEGATIVE_OFFICIAL_RESULT`

## Decision

Submit one exactly-M3 readout-calibrated H1 candidate on top of the sealed
all-source C1 checkpoint.  The official H1 calibration geometry is not an
estimate: all 14 locally available held-out-calibration NWBs contain exactly
three legal calibration trials.  The package therefore requires `M=3` and
fails closed on any other support count.

Three trials do not mean three regression rows.  Their eval-valid 20 ms bins
provide thousands of labeled rows per session for the selected `MAT7` affine
readout (49 slopes plus seven intercepts).  No fourth trial, hidden query
label, target backward pass, target optimizer step, or continual target
update is used.

## Source-only evidence

Five-date outer OOF passed in every fold:

- mean gain over frozen C1: `+0.127455` R2;
- positive dates: `5/5`;
- worst-date gain: `+0.056844`;
- selected family in every outer fold: `MAT7`.

The final all-source selector, run before any held-out-calibration file was
opened, chose `MAT7` with ridge `0.0`.

## Package and container authority

- package SHA-256:
  `731725de2bacda34aa015c095d7208aff8f4dad4f3892d6efeeb0af28d4629b7`;
- checkpoint SHA-256:
  `0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`;
- model-state SHA-256:
  `bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85`;
- source carrier authority SHA-256:
  `8ea4bb1174c00ab713843cd7561562d43f81509eaaea6ea12ee80cd4eba95de7`;
- final image ID:
  `sha256:f057efc7350f94d6482ddd6ba69801025b6ad05fe208c371db15d802f1f12346`;
- frozen successful H1 base image ID:
  `sha256:93ddcdb0213c43518ef67a6cc4ec32e0ee6ef416750f738ff4e5a845bc326f4b`;
- private ECR tag:
  `20335282-c5fa-47f0-889d-bc65dd387a39`.

The exact ECR tag was pulled back before registration.  Its image ID,
package/checkpoint/M3 labels, embedded payload SHA and CPU smoke all matched
the local candidate.  The host lacks the NVIDIA Docker runtime, so local
container-GPU execution was unavailable; the same payload passed the host
GPU0 smoke and the container CPU smoke.  This environment limitation is not
recorded as a model failure.

The official `falcon_challenge==1.0.2` evaluator ran inside the final
container on all 13 held-in minival sessions:

- Held In R2 Mean: `0.9723302821318308`;
- Held In R2 Std.: `0.005659663642035413`;
- normalized latency: `0.15256737602828338`.

## EvalAI registration and terminal result

- challenge: `2319`;
- phase: `4599` / `few-shot-test-2319`;
- participant team: `41975`;
- submission ID: `581792`;
- visibility: private;
- submitted at: `2026-09-03T13:01:35.541171Z`;
- terminal status: `finished`;
- completed at: `2026-09-03T13:42:33.412173Z`;
- stderr: empty;
- official result JSON SHA-256:
  `92e47daecb45408ea2d2e60200e48822f916ee08a88edba2836bab83681f2359`;
- official stdout SHA-256:
  `c9d87f2d101b0607fec3bfac63187a153039b51b652e5099d5b841c072b8eb3f`.

Official `test_split_h1` metrics:

| Metric | Value |
|---|---:|
| Held Out R2 Mean | `0.24095023000695576` |
| Held Out R2 Std. | `0.24518023764038085` |
| Held In R2 Mean | `0.44481513502587705` |
| Held In R2 Std. | `0.03926750543702225` |
| Normalized Latency | `0.01855312017053397` |

Official result object:
`https://evalai.s3.amazonaws.com/media/submission_files/challenge_2319/phase_4599/submission_581792/0d3d3dd1-5102-42b0-8a77-84b3d95432.json`.

Official stdout:
`https://evalai.s3.amazonaws.com/media/submission_files/challenge_2319/phase_4599/submission_581792/dfb8dd13-932c-4bdd-848a-7e739e67fe2.txt`.

## Matched official comparison

The relevant sealed C1 anchor is submission `581748`:

- C1 Held Out R2 Mean: approximately `0.284139`;
- C1 Held In R2 Mean: approximately `0.4587`.

H1-M3RC therefore changes the official metrics by approximately:

- Held Out: `-0.04319` R2 versus C1;
- Held In: `-0.01388` R2 versus C1.

Submission `581747` is the matched T0 anchor with Held Out R2
`0.2411106`.  H1-M3RC is effectively tied with T0 (`-0.0001604`) and loses
the previously realized C1 improvement.

## Scientific decision

This is a valid finished score, not a packaging or runtime failure.  The
source-only OOF gain of `+0.127455` did **not** transfer to the official
hidden surface.  In particular, the held-out session standard deviation
increased to `0.24518`, which is consistent with unstable per-session
readout fits under the exactly-three-trial contract.

The dense 20 ms rows inside the three calibration trials are strongly
correlated and must not be interpreted as thousands of independent labelled
samples.  The unregularized `MAT7` fit selected on source dates is therefore
not supported as an H1 deployment method.  No positive H1-M3RC claim may be
made from the source OOF or local minival numbers, and this result may not be
used to claim that labelled readout calibration improves C1.

The negative result does not invalidate the existing C1 result: C1 remains
the best official configuration in this comparison and retains a Held Out
gain of about `+0.04303` over T0.  Any successor readout experiment must be a
new, prospectively frozen design with trial-level shrinkage or a deploy-time
stability guard; it may not silently tune against submission `581792`.

## Post-terminal attribution audit

The following analysis was performed only after the official result was
known.  It is diagnostic evidence, not a preregistered gate and not authority
for retroactively changing submission `581792`.

No gross deployment mismatch was found in the first audit pass:

- the sealed C1 configuration has `smooth_calibration=false`, so the new
  decoder did not omit an active causal smoothing operator;
- source screening and deployment both use a zero-left-padded `W=700`
  history, the last-bin output, division by `20`, the same exactly-M3 activity
  identity, and the same M3 H-C carrier;
- the package strict-loads the complete C1 state before applying the frozen
  readout, and local official-interface evaluation completes normally.

The strongest observed difference is statistical.  Expressing each fitted
`MAT7` map in the original output units gives:

| Diagnostic | 13 held-in/source sessions | 14 held-out-calibration sessions |
|---|---:|---:|
| map condition number, median | `2.34` | `46.19` |
| map condition number, maximum | `3.80` | `591.23` |
| off-diagonal Frobenius norm, median | `0.60` | `3.18` |
| full-map Frobenius norm, median | `2.66` | `4.49` |

A separate leave-one-calibration-trial-out diagnostic fit `MAT7` with
`ridge=1` on two trials and scored the third, rotating across the three
trials.  Its per-session mean R2 ranges were:

- held-in/source: `0.9131` to `0.9671`, median `0.9498`;
- held-out calibration: `-0.3226` to `0.5025`, median `0.1614`.

The groups are completely separated on this post-terminal diagnostic.  Thus
the thousands of calibration bins do not supply thousands of independent
constraints: only three independent trajectories support the map, and the
held-out sessions do not even exhibit a stable trial-to-trial affine
relationship.  This is a direct explanation for the large official
held-out variance and the loss of the C1 gain.

The original ridge grid is also weak in the relevant units.  With roughly
`2,200--2,700` standardized bin rows, changing `ridge` from `0` to `1` barely
changes the fitted map; even `100` is only modest shrinkage.  Consequently,
this experiment did not provide a meaningful test of a strongly regularized
or hierarchical M3 readout.

An honest successor would have to be newly preregistered and would keep C1 as
an exact `g=+0` branch.  It would select any correction using trial-level
cross-fitting, scale regularization by effective trial count rather than raw
bin count, and fall back to C1 when the three calibration trials disagree.
Because these safeguards were motivated after observing `581792`, a later
official point would be exploratory engineering evidence, not a clean
confirmatory result.
