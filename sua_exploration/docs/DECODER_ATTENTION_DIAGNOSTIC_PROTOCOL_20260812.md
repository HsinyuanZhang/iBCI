# A12 descriptive coupled-attention audit protocol

**Frozen contract:** 2026-08-13 (Asia/Hong_Kong), v4 streaming-receipt revision
**Status:** metadata preflight and implementation only. It authorizes neither a
GPU run, training, formal-test access, nor a causal claim. A real CPU forward
is separately root-reviewed and must bind an immutable metadata preflight.

## Scope

The only permitted comparison is the canonical sorted-SUA B3S M30 pair:

- the SHA-qualified **historical T4** mainline reference; and
- the independently trained **v10 Z4** component-attribution arm.

This is a same-seed, same-development-session, same-unit paired comparison,
not a claim that T4 and Z4 were freshly trained in a shared run directory.
The historical/v10 lineage asymmetry is disclosed in both preflight and
forward receipts. There is no M2, H1, AC4, pseudo-MUA, synthetic-output, or
same-checkpoint carrier-ablation arm.

The exact grid is seeds `42,43,44`, epochs `5..12`, six development validation
sessions, `M=30` support trials, `W=50` neural bins, `T=100` calibration bins,
and four B3S side-feature components. The sealed formal-test sessions
`20151113`, `20151116`, `20151117`, `20151119`, `20151120`, and `20151201` are
never opened.

The complete descriptive matrix has all 24 seed/epoch paired receipts. Any
aggregate missing one or more of those coordinates is a **partial pilot** only
and must not be described as a completed cross-seed result.

The v4 runner computes the same descriptive sufficient statistics and tensor
SHA digests in streaming form, one query batch at a time. It does not retain
the full Q/K/V/attention tensor pool in memory. This is an execution-memory
repair only; the checkpoint, session, window, metric, and interpretation
contracts are unchanged from the v2 metadata scope.

## Provenance and leakage boundary

`a12_descriptive_attention_preflight.py` validates the strict 27/6/6 manifest,
all 48 T4/Z4 epoch-checkpoint byte SHA-256 values, checkpoint run metadata,
source normalizer, T4/Z4 same-unit roster, canonical development-session paths,
and current implementation hashes. Its official receipt is O_EXCL-written,
fsync'd, `0444`, and accompanied by a SHA sidecar.

### Frozen CPU batching and v3 invalidation

The descriptive CPU forward uses the fixed DataLoader batch size
`B_cpu=512`. This is an engineering choice made without target metrics: an
interrupted pre-receipt capacity smoke ran for more than seven minutes at that
size without an OOM. It was interrupted, is explicitly invalid, and wrote no
scientific forward receipt. It is not evidence for any attention or decoding
result.

The exact batch contract (size, non-metric selection basis, and no-override
rule) is included in the v4 metadata preflight, in every forward receipt, and
in the aggregate. The runner rejects any other `--batch-size`; a different
DataLoader partition cannot silently enter a result. Synthetic tests confirm
that streaming accumulation over one already-captured ordered tensor sequence
is partition-invariant. A separate tiny real-model fixture shows that CPU
BLAS may change raw float32 bits across DataLoader partitions at ulp scale;
this is why the operational batch size is frozen and recorded rather than
treated as an interchangeable performance knob.

The immutable v3 preflight remains preserved at its old path, but is stale:
it lacks this frozen batch contract and no longer matches the implementation
bindings. A successor may be minted only at
`results/a12_descriptive_attention_audit_v4/official_metadata_preflight.json`;
the old receipt must never be overwritten or used for a forward.

The real runner accepts only that immutable official receipt. For each paired
forward receipt it binds:

- official preflight file/body SHA-256 and all current implementation bindings;
- the exact T4 and Z4 checkpoint SHA-256 and run-metadata SHA-256;
- source-train-only T4 normalizer authority, including the Z4 rule “mask after
  ordinary T4 standardization”;
- each development session, same-unit proof, M30 support-trial hash, query
  trial hash, query-window start hash, input-shape/batch evidence, and input
  digest; and
- model state SHA-256 before/after each arm and capture-on/off output parity.

Activity/identity support and the ordinary-T4 direction-derived carrier use
only chronological usable rewarded trials `[0:30]`. Query windows derive only
from `[30:end]`. The shared `load_session_with_trials` loader necessarily
loads and standardizes behavior, including query velocity, to construct its
legacy evaluator record. A12's inputs-only wrapper then records the following
separate facts in each forward receipt:

- `query_behavior_loaded_by_shared_session_loader=true`;
- `query_behavior_passed_to_model=false`;
- `query_behavior_used_for_attention_metrics=false`; and
- `query_behavior_used_for_selection_or_updates=false`.

There is no training, backward pass, decoder update, target fitting,
whole-identity zeroing, or within-checkpoint carrier replacement.

## Actual forward path

The runner imports `src` exclusively from `streaming_calibration_exp` and loads
the checkpoint wrapper through `select_gradient_free_protocol_dandi688.py`'s
actual B3S `load_frozen_model` route. It uses the B3S/T4-aware evaluation
helpers for source-train normalization, M30 trialized calibration, per-unit
T4/Z4 side features, and the standard `MCMazeSessionDataset` query windows.

The tensor contract is exact:

```text
neural      [B,W,N]
calibration [B,M,T,N]
side        [B,N,4]
```

For this v4 protocol, the batch dimension is additionally fixed to
`B<=512`, with full batches of `512` except for the final remainder batch of a
session. The recorded `cpu_forward_batch_size=512` is an execution-provenance
constant, not a model hyperparameter.

In particular, a `[W,N]` session window is stacked directly into `[B,W,N]`;
it is never transposed to `[B,N,W]` before the B3S model's own `neural.permute`
inside `decode_with_identity`.

Only the ordinary coupled, one-layer B3S decoder is accepted. Its real
`CrossAttentionLayer.forward` runs unchanged for the prediction. A separate
reporting call obtains per-head maps with `average_attn_weights=False`, so
capture-on/off prediction tensors must be exactly equal. The receipt records
the actual fused projection order `[Q,K,V]`; head contribution norms use the
**V** projection and the corresponding `out_proj` input-column block, never K.

The process requires `CUDA_VISIBLE_DEVICES=''`, `PYTHONNOUSERSITE=1`, an active
CPU-only Torch runtime, `model.eval()`, frozen parameters, and no-grad
execution. `--run-forward` additionally requires an explicit root-review
environment token; the default invocation is a metadata-only dry run that does
not import Torch, NWB, or model/data helpers.

## Descriptive metrics and interpretation

For each session the receipt reports only descriptive, paired metrics:

- normalized attention entropy and entropy-effective support;
- realized per-head contribution L2 norm from attention-weighted V;
- pairwise head-map cosine redundancy;
- window-to-window attention variability; and
- covariate-query attention variability.

The aggregate preserves each same-session T4/Z4 pairing and reports T4, Z4,
and signed `T4 − Z4` values (positive means the T4 value exceeds the paired
Z4 value). It has no causal gate, no checkpoint selection,
and no inferential claim from three seeds. Attention maps may describe routing
patterns; they do not establish that routing causes any performance or
consumer-saturation effect.
