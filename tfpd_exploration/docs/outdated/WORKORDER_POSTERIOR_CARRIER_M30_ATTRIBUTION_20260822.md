# Posterior Carrier M30 Attribution Quick Screen

## Status and scope

This is a **non-governing, zero-training, M30-only engineering attribution
screen**.  It follows the strongly negative V3 quick-screen result.  It is
not a new benchmark, does not select a checkpoint, and cannot make or rescue a
formal performance claim.

The route is additive.  It must not modify or reuse the V1/V2/V3 output roots,
the Posterior full-training route, the sealed Cell-D artifacts, shared models,
or any target dataset cache.  A public CLI is dry only; staging/evaluation
requires a reviewed, in-process capability after an independent code audit.

## Immutable V3 predecessor

The only input surface is the completed V3 selected 3+3 screen:

| Item | Literal |
|---|---|
| V3 stage root | `/home/xinyuan/Work_host/posterior_carrier_quick_screen_stage_v3` |
| V3 result root | `tfpd_exploration/results/posterior_carrier_quick_screen_v3` |
| V3 closure | `3b56ec450374e0b67d13fbcb8f68d00463b1bba837a3cd57990f3fb2fa4756cd` |
| V3 attempt | `fd85e83bbc9180c9d7ca27720de3063c76be20cf6889797570234cfd82f413d2` |
| V3 input authority | `bad14dadec2c4e525c4cffbaf689b75ce1cebfa9bccf27bc022256574176386e` |
| V3 score | `07bb30a018ad33a63d89ed3cd4fcc0293d7150b59262f0db84ae9ae5ad6183b5` |
| V3 terminal | `51103a5703ff6933362f11913afd379c3b580c59abac62630194bf2a44626949` |

The four V3 M30 cells which this route is allowed to reuse have independently
recomputed canonical JSON body digests.  These are literals, not caller
supplied receipt fields:

| Surface | A: sealed point M30 | E: posterior aligned M30 |
|---|---|---|
| within | `1bad74aae7e4e871ed4de0eaf293c1caa4c4fc396adbcefcbafa31130ae5621d` | `dafdd37de50d23b964bd2b75a2db8a061808334d80955dab93ae32727ff91cbc` |
| external | `f619a29d2aec9ac44531263a0e0b1926dc931884e18b0a509018e501d180431f` | `275732f0cc3b384d8d14bc5e11aac77bde214da17753ca113d55928fc0c9f49c` |

There is one subtle provenance rule behind that table.  A successor A/E
`CellEvidence` has extra carrier-digest fields and a new input-replay SHA,
which did not exist in V3.  Before it can be copied, the route reconstructs
the original V3 cell body from the current copied cell/session/model-state
facts: V3 score-cell mode, V3 input-authority SHA, model SWA/state fields,
and the exact session rows.  It then canonicalizes and hashes that original
body and requires the corresponding literal above.  Thus changing a copied
R2, prediction digest, model state, model SWA, or claimed reusable digest
cannot turn into a valid A/E reuse merely by rebuilding an outer receipt.

The V3 root must contain exactly the four 0444 body/sidecar pairs
`attempt`, `input_authority`, `score`, and `terminal`; it must contain no
failure pair.  The successor validates the full V3 identity, same input
authority, the original score/terminal chain, the V3 FP32 final-four proof,
and every V3 closure byte before any evaluation input is resolved.

The selected sessions are frozen and ordered:

```
within:   sub-C_ses-CO-20151103, sub-C_ses-CO-20151106, sub-C_ses-CO-20151112
external: sub-M_ses-CO-20140307, sub-M_ses-CO-20150611, sub-M_ses-CO-20150626
```

No M10, H1, formal, zero, wrong-pair, target adaptation, sampling, refitting,
or full-window metric is allowed.

## Fixed metric and systems

The score remains V3's last-bin-only query metric: variance-weighted two
coordinate R2 per session followed by equal weighting of the three selected
sessions.  Inputs, last-bin target/mask/count records, B3S M30 calibration,
both sealed SWAs, behavior normalization, source-only carrier normalizers,
and remote Torch-only 5070Ti device attestation are held fixed.

For each of the two surfaces, exactly the following five M30 systems occur in
the listed order:

| ID | Consumer weights | Raw carrier estimator | Normalizer | Credibility |
|---|---|---|---|---|
| A | sealed Cell-D | OLS point | sealed Cell-D ordinary point | none |
| B | sealed Cell-D | posterior mean | sealed Cell-D ordinary point | none |
| C | Posterior SWA | OLS point | frozen posterior distribution | uniform / exact cancellation |
| D | Posterior SWA | posterior mean | frozen posterior distribution | uniform / exact cancellation |
| E | Posterior SWA | posterior mean | frozen posterior distribution | original nonuniform precision |

`A` and `E` may be copied only after exact V3 score/input/terminal
verification.  `B`, `C`, and `D` must perform new read-only forwards in the
successor route.  No model parameter, optimizer, dropout state, or target
state may change.

The raw OLS point values for C must be captured from the same no-cache
M30 construction before normalization and checked against its retained raw-T4
axis proof.  Reconstructing raw values by algebraically inverting a normalized
tensor is forbidden.  If that exact cache cannot be obtained, the route must
fail before any new score is published.

## Attribution, not a governing gate

For each surface the receipt reports exact paired per-session deltas and
mean/median/positive count for:

- `B - A`: posterior-mean estimator change under the sealed consumer;
- `D - C`: posterior-mean estimator change under the Posterior consumer;
- `E - D`: nonuniform precision-bias effect;
- `C - A`: Posterior consumer plus its frozen posterior-normalizer contract
  versus the sealed consumer plus its frozen ordinary-normalizer contract.

The final contrast is intentionally labelled **consumer-system**, not literal
parameter-only weight causality: the predeclared systems use their respective
frozen normalizers.  The receipt emits deterministic descriptive labels
`bias_rescue`, `estimator_harm`, and `training_weight_harm` based on the signs
and breadth of these paired deltas, but all labels are explicitly
non-governing and non-inferential.

## Deferred physical route

Future roots are fresh:

```
/home/xinyuan/Work_host/posterior_carrier_m30_attribution_stage_v1
tfpd_exploration/results/posterior_carrier_m30_attribution_v1
```

The physical route composes the frozen V3 backend without monkeypatching it.
It uses only this exact reviewed Torch-only engineering device authority:

```text
schema=posterior_carrier_quick_screen_remote_torch_only_device_v3
CUDA_VISIBLE_DEVICES=0; CUDA_DEVICE_ORDER=PCI_BUS_ID; visible_device_count=1
logical_device=cuda:0; name=NVIDIA GeForce RTX 5070 Ti Laptop GPU
torch=2.13.0+cu130; CUDA=13.0; cuDNN=92000; TorchMetrics=1.9.0
compute_capability=[12,0]; torch_total_memory_bytes=12346195968
uuid=null; bdf=null; nvidia_smi_memory_total_mib=null
nvml_status=UNAVAILABLE_DRIVER_LIBRARY_MISMATCH; nvidia_smi_called=false
engineering_only=true; current_local_authority=false
route_role=remote_5070ti_non_governing_quick_screen
variance_weighted_r2_parity_gate=PASS__SYNTHETIC_MANUAL_REFERENCE
```

The identity, durable score body, and physical attestation all exact-compare
this complete mapping.  It never calls `nvidia-smi`; null NVML fields are
honest facts rather than placeholders.
Attempt publication precedes any evaluation-path resolution.  Result
publication is an O_EXCL/fsync/0444 atomic score+terminal group; a post-attempt
failure can publish only an honest failure terminal.

Before a launch, the route must prove all of the following:

1. V3 input replay is byte- and descriptor-bound, and newly materialized
   records exactly equal V3's six records.
2. Every new control uses an exact raw-before-normalization carrier and the
   selected frozen normalizer; uniform credibility produces the wrapper's
   exact no-bias path.
3. Each new forward is eval/no-grad/no-dropout, finite, repeat-bitwise, and
   preserves the strict model-state digest.
4. Both models and all source authorities are loaded before evaluation, but no
   target optimizer/backward/update/refit/sampling/formal/H1 path is reachable.
