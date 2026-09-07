> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md`](DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md).
> Preserved as append-only audit evidence; content unchanged.

# External sub-M score-only v2 — C1 data-adapter parity protocol v3

Status: `STATIC_PROTOCOL_AUDITED_V3_DATA_ADAPTER_PARITY_EXECUTION_NOT_AUTHORIZED`

This is an append-only correction to the scorer-adapter parity evidence. It does
not modify, reissue, or delete score-only v1/v2, parity v1, or parity v2. The
existing parity-v2 draft, receipt, and seal remain byte-preserved evidence, but
v3 classifies that evidence as
`INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING`.

The reason is specific: v2 prepared one C1-owned batch and fed that same batch
to both labels. It could check a thin forward wrapper, but it could not test
the sub-M scorer's own data-adapter owners. It therefore cannot authorize a
score-only release. No claim in v3 changes the old v2 bytes or turns them into
an authorization.

No operation in this delivery opens a checkpoint, normalizer NPZ, or NWB;
imports the runtime helper; runs a model forward or R2; touches CUDA; or
constructs an external sub-M path. The v3 runner's only usable mode is a
metadata-only dry run.

## Fixed, already-consumed C1 fixture

| Field | Fixed value |
|---|---|
| session | C1 development `sub-C_ses-CO-20151103`, first literal `val` manifest row only |
| view | SUA |
| checkpoint | `shared_t4`, seed 44, terminal `epoch_011.ckpt` |
| checkpoint pin | `a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6`, 64,769,167 bytes |
| NWB pin | `7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7`, 62,145,872 bytes |
| behavior normalizer | `821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd` |
| T4 normalizer | `32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701`; semantic pin `ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7` |
| support/query boundary | rewarded `trials[0:50]` / query `trials[50:]` |
| identity activity | C1 `first_n30` selected from the first-50 pool |
| finite batch observation | whole first post-50 batch plus its rows `0:16`; `batch_size=128`, `shuffle=False`, `num_workers=0` |

The C1-side authority is the source-pinned loader chain
`load_session_with_trials` → `select_calibration_trial_indices(first_n30)` →
`build_calib_trials_for_indices` → `attach_side_features` →
`make_subset_dataset`.

The adapter-side authority is independently run from the score-only v1 pinned
owners: `load_dandi688_session`, `list_datamodule_rewarded_trials`,
`load_unit_side_features`, and `MCMazeSessionDataset`. The extra
`list_datamodule_rewarded_trials` call is an existing owner function, not a
new reimplementation: `SessionRecord` deliberately has no retained trial
list, while a chronology comparison needs the exact rewarded/usable ordered
list that the loader used.

## The calibrated-identity correction

The current score-only v1 owner calls
`load_dandi688_session(calibration_n_trials=50,
exclude_calibration_trials_from_windows=True)`. That is correct for its score
boundary and its T4 label/rate pool, but its returned `calib_trials` is 50
rows. C1's activity identity is not 50 rows: it is the first 30 rows selected
by `select_calibration_trial_indices(..., 30, 50, "first")`.

V3 therefore imposes the explicit
`LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30` gate:

1. The adapter owner loader still receives 50. Its post-50 query
   `valid_starts` remain an adapter-produced fact and must match C1.
2. The adapter owner side-feature loader still receives `pool_size=50`; T4
   fitting/label pooling is not changed or refit.
3. Independently obtained adapter neural data and its owner-provided rewarded
   chronology are passed to C1's existing `build_calib_trials_for_indices`
   with the C1-selected first 30 indices. V3 does not copy interpolation,
   padding, or selection algorithms.
4. The adapter-owned `MCMazeSessionDataset` is constructed with that rebuilt
   30-row tensor, its own post-50 starts, and independently loaded T4 rows.

This deliberately does **not** redefine C1 to use 50 calibration rows merely
to make a comparison easier. If the independently built tensors differ after
the explicit rebuild, parity fails closed.

## Required observer order

The shared observer receives two independently prepared objects. Before it can
load a model or execute one forward it requires exact equivalence of:

- the complete ordered rewarded chronology common to both owner APIs: original
  NWB trial row, start bin, stop bin, and target direction;
- C1 selection indices `[0, …, 29]`, support/query boundary 50, and all
  post-50 `valid_starts`;
- full neural, behavior, rebuilt calibration, T4 side-feature, and
  electrode-id arrays (including absent `None` electrode ids for this T4
  fixture), with dtype, shape, contiguous bytes, SHA-256, and exact values;
- the complete first `DataLoader` batch and an explicit first-16-row trace of
  neural/history, behavior, calibration, side features, electrode ids, and
  session labels.

Only after that gate succeeds can a future, separately authorized execution
call the one `_forward_and_observe` helper for both datasets. That helper is
the sole fixed-model body: C1 unpacking, `decoder_key_features`, student
forward, C1 `decode_last_behavior`, target extraction, and a CPU TorchMetrics
observer. Prediction and target arrays must be byte/value-exact between the two
invocations; independent R2 values must agree within `1e-6`.

The future execution is CPU-only, deterministic, one-threaded, and must use an
exclusive-create output root sealed at mode `0444`. It remains a C1
consumed-development compatibility check, never an external sub-M endpoint
score and never a score authorization.

## Source and authorization gates

The metadata writer binds the final four score-only v2 sources and their
three-file prelaunch closure, the preserved v1/v2 parity bundles, the v3 source
set, and the v1 owner sources named above. Any SHA drift is `FAIL_CLOSED`;
existing sealed artifacts must not be regenerated to repair a drift.

The current v3 runner raises `PARITY_EXECUTION_NOT_AUTHORIZED` before
importing the helper. A future append-only review must bind this v3
draft/receipt/seal, a one-time execution authorization, the exact runtime
environment, a successful sealed parity receipt, and a separate root review.
Until then, no checkpoint/NWB/forward operation is permitted.

