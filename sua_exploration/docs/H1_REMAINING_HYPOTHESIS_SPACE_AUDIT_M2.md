# H1 M=2 generalized-T4 carrier: remaining-hypothesis-space audit

**Decision: STOP.** Under the frozen H1 two-trial target-calibration budget, there is no remaining statistically distinct CPU-only target-side identifiability test that can justify a generalized T4 carrier for joint source training with SPINT. Therefore no new test is recommended or authorized from this audit.

This is not a claim that H1 behavior carries no neural information. It is a claim about the prerequisite needed for the stated deployment goal: a target-specific, channel-attached, four-dimensional object whose dependence on the correct target neural--kinematic pairing is reproducible from exactly two support trials and survives strict later-trial/date-LODO evaluation.

## Evidence reviewed

The complete handoff and the authoritative scripts/immutable receipts below were read. All receipts restrict themselves to the 13 public held-in-calib recordings, exclude formal held-out and minival/query data, and forbid target backpropagation and GPU construction.

| Family | Program and authoritative receipt | What it rules out |
| --- | --- | --- |
| Signed AFC4 | `streaming_calibration_exp/scripts/audit_h1_afc4_m2_date_lodo.py`; `sua_exploration/results/h1_afc4_m2_date_lodo_v1/cpu_feasibility_receipt.json` | Target per-channel affine encoding direction `x_i <- y`. None of six dates reached split-trial median direction cosine 0.5. |
| Invariant AFC4 | `streaming_calibration_exp/scripts/audit_h1_invariant_carrier_m2_date_lodo.py`; `sua_exploration/results/h1_invariant_carrier_m2_date_lodo_v1/cpu_feasibility_receipt.json` | Affine magnitude/baseline/relevance as a functional carrier. Modulation rank was stable, but correct cross-trial rate gain was positive on 0/6 dates and exceeded rotation-null q95 on only 3/6. |
| LFMC4 | `streaming_calibration_exp/scripts/audit_h1_lfmc4_m2_date_lodo.py`; `sua_exploration/results/h1_lfmc4_m2_date_lodo_v1/cpu_feasibility_receipt_v5.json` | A source-learned bounded nonlinear kinematic basis with target per-channel cross-moments. Split-trial moment cosine and positive correct transfer gain were both 0/6; rotation-null separation was 3/6. |
| Population decoder direction | `streaming_calibration_exp/scripts/audit_h1_population_decoder_carrier_m2_date_lodo.py`; `sua_exploration/results/h1_population_decoder_carrier_m2_date_lodo_v1/H1_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json` | The most favorable population `y <- X` alternative, subsequently expressed as four-dimensional channel coefficient rows. Correct later-trial R2 was positive on 4/6 dates and beat rotation-null q95 on 6/6, but trial-1/2 coefficient-row attachment stability was at least 0.5 on 0/6 dates. |

The governing handoff read in full is `sua_exploration/docs/HANDOFF_AFC4_H1_KINEMATIC_CARRIER.md`. Its §6.4 already states the decisive recurring observation: every predeclared per-channel functional estimator required to transfer between the two calibration trials had non-positive date-level evidence. The separate population receipt adds that a weak population association does not turn into a reproducibly channel-attached carrier.

## Why no fifth M=2 test is defensible

Any candidate carrier that actually changes a target channel token must obtain target-specific paired information from the empirical relation of that channel's neural sequence and the two kinematic support sequences.

- A conditional mean, lagged/spectral response, kernel regressor, CCA-like construction, or neural--kinematic similarity score is a reparameterized per-channel `x_i <- y` relation or finite cross-moment. Changing basis, nonlinearity, time-frequency representation, or estimator name would recycle signed/invariant AFC4 or LFMC4 rather than create a new hypothesis.
- A fitted population decoder followed by a channel projection is the fourth tested family. It is the only one with a strong correct-pairing-vs-rotation result, but the rows needed to attach the result to channel tokens are unstable on every date. Relaxing the attachment statistic, using a wider basis, or passing it through a richer fusion network would hide rather than solve that failure.
- A tempting apparent exception is a source-trained population manifold plus a target closed-form global alignment (for example a 4-by-4 Procrustes rotation). If the alignment is fitted from paired target data, it is a more constrained population `y <- X` decoder and must still demonstrate a stable per-channel attachment; the fourth receipt supplies the directly relevant negative evidence. If it is fixed from sources instead, the target carrier no longer depends on the correct target pairing, so a rotation null cannot establish its target-side functional identity. It is merely an additional source token prior / width-or-fusion change.

Thus a fifth construction can obtain an apparent positive result only by one of two invalid routes: reusing a stopped estimator family under another parameterization, or letting source trajectory/channel priors stand in for new target-side identifiability. Neither is evidence that a jointly trained SPINT residual can exceed original SPINT because of a correctly attached target functional carrier.

## What would reopen the question

The smallest meaningful change is **new calibration budget**, not a new M=2 estimator. A separately authorized M=4 study could use trials 1--4 as support, compare independently fitted 1--2 versus 3--4 channel carriers, and reserve trials 5+ as strict chronological query. M=4 is common to all 13 held-in recordings and retains at least three later trials even for the short recordings. It is a different deployment condition and must be compared with an equally budgeted neural-only SPINT baseline; it cannot be presented as a rescue or a result for original M=2 SPINT.

Before any such program, it needs a new explicit authorization and a frozen source-only protocol: source dates must still exclude the outer date from all basis/normalizer/model choices; query labels must remain absent from selection; the paired target support must be compared with deterministic within-support label rotations while query labels remain correct; and passage must require both correct-later-trial gain over null and stable channel attachment on at least 4/6 dates. Alternatively, independent repeated target calibration under a new data collection protocol, with verified channel correspondence and enough later query trials, could supply the missing information. More source training, wider token fusion, a different target-side network, or GPU budget does not.

## Single recommended next action

Keep the H1 M=2 supervised carrier branch closed and finish/report the already authorized original-SPINT neural-only reproduction. Do not implement or run a fifth M=2 carrier test. Revisit only after an explicit decision to change the target calibration/data budget as above.
