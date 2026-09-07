# M2 Official Activity-Budget Payload V1

## Decision

Prepare, but do not yet submit, two official-compatible cached-identity M2
candidates:

- fixed ridge-T4 M10 with first-30 unlabelled B3S activity;
- D-opt fixed ridge-T4 M4 with first-30 unlabelled B3S activity.

The M4 candidate consumes four trial labels in its T4 fit but reads finite
target-direction metadata across the first-30 candidate pool for D-opt
selection. This must be disclosed as cue-budgeted M4.

Both candidates use the already submitted seed42 T4 decoder/checkpoint and the
same 13 public calibration session tags. Each identity is computed offline and
stored as `[96,50]`. The official runtime reuses the accepted stateless cached
decoder: no trial detector, online memory, target labels, gradients, optimizer,
or recalibration exist inside the EvalAI container.

Payload export is allowed after the local paired result is accepted. Image
build, remote-path simulation, quota preflight, push, and formal registration
are separate gates. No formal submission may be created merely because an
export succeeds; the local same-query and three-seed results must first show
that the selected budget is worth consuming an official quota slot.
