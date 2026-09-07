> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md`](DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md).
> Preserved as append-only audit evidence; content unchanged.

# External sub-M score-only v2 — C1 scorer-adapter parity protocol v1

Status: `STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED`

This is an append-only design and static-audit protocol.  It creates neither a
sub-M result nor a parity result.  In particular, this delivery does not open
an NWB, load a checkpoint or normalizer, import a model, execute a forward
pass, create a prediction, update TorchMetrics, or use a GPU.  The only future
purpose of the protocol is to establish that the external-subject v2 scorer
adapter means exactly the same thing as the already-used C1 evaluator before
it is allowed to see a sub-M endpoint.

The development session below is already consumed C1 development material. It
is never a sub-M asset and it is not a source-training or formal-test session.

## Fixed, non-selectable parity fixture

| Field | Frozen value |
|---|---|
| view | `sua` |
| terminal C1 arm / seed / epoch | `shared_t4` / `44` / `epoch_011.ckpt` |
| checkpoint SHA-256 / bytes | `a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6` / `64769167` |
| C1 run metadata SHA-256 | `d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7` |
| consumed C1 development session | `sub-C_ses-CO-20151103` |
| consumed session SHA-256 / bytes | `7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7` / `62145872` |
| session selection authority | first `val` entry of the C1 27/6 manifest; no discovery, resampling, or fallback |
| behavior normalizer | source-train-27-only `be50f588491c004f721e.npz`, SHA-256 `821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd` |
| SUA T4 normalizer | source-train-27-only `dd3da1f59700c1b96ab8.npz`, SHA-256 `32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701`; semantic SHA-256 `ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7` |
| fixture query batch | positions `0:16` within the first `batch_size=128`, `shuffle=False`, `num_workers=0` C1 query DataLoader batch |

The session and terminal checkpoint are fixed by the existing C1 manifests and
the v2 score-only constants.  They cannot be changed because of a parity
failure.  A failure is a failed adapter, not a license to search another
session, seed, epoch, normalizer, device, batch size, or numerical tolerance.

## Exact semantic reference

The future adapter must reuse these existing functions; it must not copy their
preprocessing, T4 attachment, frozen-model construction, or metric logic into
another implementation.

| Required semantics | Existing owner that the adapter must call |
|---|---|
| session/trial decoding, 20-ms spike binning, valid 50-bin history | `scripts/eval_adaptation_dandi688.py:load_session_with_trials` |
| calibration tensor construction | `scripts/eval_adaptation_dandi688.py:build_calib_trials_for_indices` |
| post-support query dataset and `MCMazeSessionDataset` order | `scripts/eval_adaptation_dandi688.py:make_subset_dataset` |
| T4 feature/unit-row attachment | `scripts/eval_adaptation_dandi688.py:attach_side_features` |
| terminal checkpoint/teacher model construction | `scripts/select_gradient_free_protocol_dandi688.py:load_frozen_model` |
| reference forward and TorchMetrics stream | `scripts/eval_adaptation_dandi688.py:eval_r2` |
| external sealed-array independent recomputation | `mc_maze/subm_co_score_only_v2.py:recompute_torchmetrics_r2_cpu` |

The C1 wrapper `scripts/eval_paired_view_c1_epoch_window.py` establishes that
the C1 path selects its view-specific T4 normalizer and applies the fixed
`pool_size=50`, 20-ms bins, 50-bin history, trial length 100 and padding -1.
The static prelaunch receipt seals the hashes of all of the files above,
including `multisession_datamodule.py` and `unit_side_features.py`.

There is an implementation blocker: `eval_r2` returns only a scalar and does
not expose a public per-batch trace.  A future authorized implementation must
add one observation-only shared trace hook beside the reference evaluator (or
another shared public helper) and have both paths call it.  A separately
reimplemented loader/forward/R² loop is not acceptable for parity.  This
document and its prelaunch receipt do not remove that blocker.

## Parity comparisons and fail-closed limits

The parity run is one **CPU-only** process using the exact software lock
recorded in its execution receipt: `device=cpu`, no CUDA initialization,
`CUDA_VISIBLE_DEVICES=""`, `torch.use_deterministic_algorithms(True)`, TF32
disabled, autocast disabled, and one intra-op / inter-op thread.  It is not a
performance result.  The later sub-M matrix keeps the v2 authorization-bound
one-device policy; it is a separate execution binding.

Chronology is evaluated before any model operation:

1. C1's rewarded-trial list order is recorded as an index/time digest.
2. `support` is exactly `trials[0:50]`; no selection is made from later trials.
3. The C1 identity configuration remains its fixed `first_n30` selection from
   that support pool.  The remaining support trials stay excluded from query.
4. `query` is exactly `trials[50:]`, and the fixed finite comparison batch is
   the first 16 valid query windows from that ordered query DataLoader.

Every item in the following table must pass.  `FAIL_CLOSED` is the only other
outcome; tolerances are upper bounds, not diagnostics to tune.

| Ledger item | Required proof | Fail-closed threshold |
|---|---|---|
| trial chronology / support / query | exact ordered digest of trial ids, start/stop bins, support indices `[0..49]`, query trial indices, and selected query rows | byte-identical digest and lengths |
| spike bins and history | shape, dtype, and digest of binned neural rows plus each 50-bin query-history tensor | exact equality; `max_abs=0` |
| target windows | shape, dtype, ordering and digest | exact equality; `max_abs=0` |
| unit ordering | ordered unit identity/index digest and `n_units` | exact equality; adapter row map must be identity `[0,...,n_units-1]` |
| T4 row attachment | T4 side-feature shape/dtype/digest and paired `(unit_index, row)` digest | exact equality; no permutation, broadcasting, or refit |
| normalizers | behavior and T4 NPZ file SHA-256, semantic hash, loaded array dtype/shape/value digest | exact file/semantic/array equality; fitting calls = 0 |
| model input | neural, calibration, side-feature and electrode-id tensors after C1 collation | exact equality; `max_abs=0` |
| prediction | raw and `/5.0` last-bin output shape/dtype/value digest | exact equality; `max_abs=0`, `max_rel=0` |
| R² stream | ordered TorchMetrics update tensor digests, version strings, and final scalar | reference-adapter absolute difference `<= 1e-6` and finite |
| sealed-array R² | CPU `recompute_torchmetrics_r2_cpu(predictions, targets)` | each reported scalar differs by `<= 1e-6` |

The value requirement is intentionally stronger than a close-enough output
test: with a CPU-only deterministic fixture and a common forward helper, the
prediction tensors should be bit-identical.  The `1e-6` scalar allowance is
only the already-sealed v2 `R2_RECOMPUTE_ATOL` for TorchMetrics serialization
and recomputation, never an allowance for different predictions.

## Future single-use execution receipt

No receipt with the following schema exists yet.  A root-reviewed, one-time
execution may create exactly one fresh directory and the immutable files:

```text
subc_scorer_adapter_parity_execution_v1/
  input_manifest.json
  c1_reference_trace.json
  adapter_trace.json
  reference_prediction_target.npz
  adapter_prediction_target.npz
  receipt.json
  seal.json
```

All writes use exclusive creation, fsync, SHA-256, then mode `0444`; an
existing root is an error, and no retry/overwrite/mutable append is permitted.
`seal.json` lists every path, SHA-256, bytes, and mode.  The receipt must use:

```json
{
  "schema": "dandi_000688_subc_scorer_adapter_parity_receipt_v1",
  "status": "PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION"
}
```

only after every table row passes.  Otherwise its status must be
`PARITY_FAILED_FAIL_CLOSED`, retain the first failing ledger item, and be
ineligible for a root pin.  It must bind the protocol prelaunch receipt SHA,
all source hashes, fixed checkpoint and normalizer pins, consumed-session pin,
CPU/software/device lock, exact batch selector, trace/NPZ hashes, every
comparison result, zero fitting/backward/optimizer counters, and an
independent reviewer signature.  It must additionally state
`subm_nwb_paths_constructed=false`, `subm_nwb_files_accessed=false`,
`subm_endpoint_scores_computed=false`, and `external_score_authorized=false`.

The current `subm_co_score_only_v2.py` deliberately has
`SCORER_ADAPTER_PARITY_RECEIPT_PIN = None`; no parity design file, draft, or
prelaunch receipt is accepted as this pin.  A later append-only runner revision
may add the exact receipt path/SHA only after an independently reviewed
`PARITY_CONFIRMED...` execution receipt exists.  Root must still separately
provide the real Ed25519 public-key pin; this protocol does not authorize the
external score run.

## Current blockers

1. No observation-only shared C1 batch-trace hook exists, so a future adapter
   cannot prove raw prediction/tensor parity without either changing the C1
   reference in a shared way or duplicating logic.  Duplication is forbidden.
2. No consumed-sub-C execution receipt exists to pin in v2.
3. `ROOT_ED25519_PUBLIC_KEY_PIN` remains `None` in v2.
4. Therefore the v2 external runner remains `NOT_AUTHORIZED_FOR_SCORING`, and
   this document authorizes neither a checkpoint load nor any sub-M access.
