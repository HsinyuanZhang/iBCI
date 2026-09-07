# Native M2 Precision-CDM V2 Screen V1

## Purpose

Test Precision-CDM V2 on native FALCON M2 with the frozen seed-42 T4 model. This is a local, trial-aware engineering screen. It is not an official submission because the official evaluator does not expose completed-trial boundaries for online state updates.

## Matrix

- Surfaces: seven held-in sessions and six held-out calibration sessions, both scored strictly after trial 30.
- M4: D-optimal four finite-direction trials from the first 30.
- M10: first ten finite-direction trials from the first 30.
- M30: full-rewarded-trial fixed-ridge T4 and an exact Precision no-op copy.
- M4/M10 arms: static T4, activity-only, ordinary independent-activity CDM, Precision-CDM V2.
- Query windows, targets, frozen model, source T4 normalization, and R2 are identical across arms.

## M2 rate adapter

The frozen M2 T4 model was normalized in spike-counts per 20-ms bin, while the
CDM state machine consumes complete rewarded-trial native counts in Hz.  This
local mechanism screen therefore rebuilds every M4/M10/M30 initializer from
the same complete rewarded-trial Hz table used by later online updates.
Immediately before frozen-model identity computation, every raw T4 coordinate
is multiplied by 0.020 and then passed through the immutable M2 T4 normalizer.
No normalizer is refit.  Absolute rows are consequently a local CDM-adapter
screen, not a byte-identical replay of the production filtered-calibration T4
baseline; all four arms within a budget nevertheless share the exact same
initializer, query windows, targets, and frozen model.

B3S activity is reconstructed independently per raw trial from only the
calibration-visible bins inside that trial, then interpolated to 100 bins.
This prevents the frozen dataset's filtered calibration time axis from being
misrepresented as contiguous native rewarded-trial counts.

## Causality and scope

- The state used for a trial is read before that trial is completed.
- Metric target behavior never enters activity, pseudo-direction, precision, or carrier state.
- Query-trial activity and count updates use only neural data and frozen-model predictions.
- A completed raw trial shorter than 50 bins still advances the independently
  validated activity FIFO, but it cannot create complementary carrier
  predictions.  Its carrier transition is therefore the typed
  `velocity_shape` rejection; padding and cross-trial history are forbidden.
- No gradient, backward call, optimizer step, or parameter update is allowed.
- Seeds 43/44 run only if seed42 gives a positive, credibly broad
  Precision-minus-activity-only M4 or M10 signal and M30 remains an exact
  no-op. Improving ordinary CDM alone is insufficient because activity-only
  is the simpler accepted comparator.
