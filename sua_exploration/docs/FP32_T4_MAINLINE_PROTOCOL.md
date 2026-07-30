# FP32 T4 Mainline Protocol

**Status:** M2 complete and positive; SUA running; encoder-INT8 watcher armed  
**Frozen:** 2026-07-30  
**Order:** finish the matched SUA matrix; if both SUA T4 contrasts are strictly
positive and the protocol audit passes, immediately run encoder PTQ, followed
automatically by encoder QAT if required.

## Claims under test

### M2

On local FALCON M2 held-out-calibration sessions, T4 must outperform the locally
reproduced full SPINT checkpoint when both receive the same chronological first
33 calibration trials.

- SPINT/B0 receives spikes from trials `0:33`.
- T4 receives the same spikes and target labels from trials `0:33`.
- TS4 receives the same inputs, with T4 features permuted over units.
- Calibration-time inference is frozen and has no optimizer or backward pass.
- T4/TS4 are trained on all seven held-in M2 sessions. Held-out sessions are
  loaded only for test.
- The result is a local replay, not a hidden EvalAI benchmark result.

The existing B0 artifact has `data.random_calibration=true` because that option
applies to its training split. `FalconDataModule` hard-codes
`random_calibration=false` for `val_heldout_dataset`, so the reported B0
held-out metrics use the same first-33 prefix as T4.

Matrix: `T4/TS4 × seeds 42,43,44`; B0 is the fixed locally reproduced full-SPINT
checkpoint. Runner:
`streaming_calibration_exp/scripts/run_m2_spint_t4_mainline.sh`.

### SUA

On DANDI 000688 sub-C CO validation sessions, T4 must outperform a local
reproduction of the original SPINT identity encoder, not merely B3/F0.

- B0 is a trainable, non-aliased copy of SPINT's original `fc_id_in/fc_id_out`.
- B0, T4, and TS4 all use activity trials `0:30` during training and evaluation.
- T4/TS4 labels are also restricted to trials `0:30`; the previous
  activity-30/side-pool-50 mismatch is forbidden.
- Evaluation uses only windows after trial 30.
- All arms use the same strict 27-train/6-validation split and the same
  12-epoch budget with the fixed epoch-5..12 averaging rule.
- Six formal test-session files remain unopened.

Matrix: `B0/T4/TS4 × seeds 42,43,44`. Runner:
`sua_exploration/scripts/run_sua_spint_t4_mainline.sh`.

## Acceptance rule

For the primary contrast (`T4−SPINT` on M2; `T4−B0` on SUA), all conditions
must pass:

1. mean paired delta R² is at least `+0.03`;
2. all three seed-level means are positive;
3. all six session-level means are positive;
4. the hierarchical seed/session bootstrap 95% interval has lower bound above zero;
5. the exact two-sided Wilcoxon test over six session-averaged paired deltas is
   at most `0.05`.

The same rule is applied to `T4−TS4` as evidence that label-unit alignment,
rather than extra parameters alone, carries the gain. Because related held-out
sessions were inspected in earlier internal-LOSO experiments, passing is strong
local replication evidence, not a prospectively untouched confirmatory test.

Strict aggregators:

- `streaming_calibration_exp/scripts/aggregate_m2_spint_t4_mainline.py`
- `sua_exploration/scripts/aggregate_sua_spint_t4_mainline.py`

They reject incomplete matrices or drift in trial count/order, session split,
side-feature pool, epoch budget, checkpoint rule, or formal-test isolation.

## INT8 dependency

Do not use quantization to rescue a failed FP32 claim.

The automatic launch threshold is deliberately distinct from the stronger
paper-claim threshold above: launch when the strict SUA aggregate has
`T4−B0 > 0` and `T4−TS4 > 0`, with the complete protocol audit passing.  This
tests quantizability as soon as T4 has a positive signal; it does not promote a
sub-`+0.03` result to the main FP32 claim.

Per the 2026-07-30 scope decision, this repository does not repeat decoder
quantization already completed on another platform.  The local experiment is:

1. Quantize the frozen **T4/B3S identity encoder only**.  The four Linear layers
   are W8A8 with INT32 accumulators and integer requant; the decoder remains
   FP32.
2. Quantize the normalized four-dimensional T4 tensor at the same activation
   scale used by pooled activity at the real `post0 [68→64]` input.  Concatenate
   in the integer domain; a floating-point side-feature bypass is forbidden.
3. Fit/select PTQ scales using only the 27 training sessions.  Evaluate the
   frozen candidate on the six validation sessions with the same chronological
   first-30 support and `trials[30:]` windows.  Do not open formal-test NWBs.
4. PTQ passes only if end-to-end `T4 INT8 encoder + FP decoder`
   `ΔR² ≥ −0.01`, every activation-edge saturation rate is at most `0.5%`,
   INT32 overflow is zero, and the STE encoder output equals the independent
   integer engine exactly.
5. Any PTQ failure automatically launches fixed-budget encoder QAT from the
   same FP32 checkpoint and split.  QAT may use labels from the 27 training
   sessions, but validation cannot choose the epoch; the final fixed epoch is
   evaluated.

Automation:

- `sua_exploration/scripts/watch_and_launch_t4_encoder_int8.sh`
- `sua_exploration/scripts/run_t4_encoder_int8_after_positive.sh`
- `sua_exploration/scripts/eval_t4_encoder_int8_dandi688.py`
- `sua_exploration/scripts/train_t4_encoder_qat_dandi688.py`
- `sua_exploration/scripts/aggregate_t4_encoder_int8.py`

The permitted claim from this local work is **T4 encoder INT8 + FP decoder**.
Any full-model quantization statement must cite the separate decoder result as
independent evidence instead of attributing decoder quantization to this run.
