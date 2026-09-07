> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md`](DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md).
> Preserved as append-only audit evidence; content unchanged.

# External sub-M score-only v2 — C1 scorer-adapter parity protocol v2

Status: `STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED`

This append-only v2 protocol supersedes parity-prelaunch v1 **only as a
non-authorizing source/prelaunch binding**.  The v1 draft, receipt, and seal
remain unchanged evidence, but are now classified
`STALE_NON_AUTHORIZING_SUPERSEDED`: they recorded an earlier
`subm_co_score_only_v2.py` source hash and predated the final signed-v2
four-file source/prelaunch closure.  This protocol does not reclassify, delete,
or overwrite v1.

No operation in this delivery opens a checkpoint, source normalizer, sub-C
NWB, sub-M NWB, model, or endpoint; invokes a forward pass/R²; or uses CUDA.
The new helper and runner are source only and remain blocked until a future,
independent execution authorization exists.

## Final v2 binding

The v2 parity prelaunch must bind exactly these final score-only sources:

| File | SHA-256 |
|---|---|
| `mc_maze/subm_co_score_only_v2.py` | `213e5495b4fa8776967ea07bf57743db18619b74aea593bac1d17d5944308e4d` |
| `scripts/run_dandi688_subm_co_score_only_v2.py` | `dc271dbe0b7ede31865f5c35890444b969296096f9107f960183079cc63a76c7` |
| `scripts/write_dandi688_subm_co_score_only_prelaunch_v2.py` | `d8f4b37a091233c421bd052238c165c194bb31318ed2c8172e0049a444d2f58c` |
| `tests/test_dandi688_subm_co_score_only_v2.py` | `944e5a35ca7c61d18655a4aa4e3c661252cd8cc9e97e687034a3c28f4d3ff339` |

It also binds the complete stored v2 prelaunch bundle:

| Artifact | SHA-256 |
|---|---|
| `prelaunch_authorization_draft.json` | `2a8455fff85e8a0644e5dbb1f9932f126db55b856842cc3207d3f59bf7c59eae` |
| `receipt.json` | `44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868` |
| `prelaunch_artifact_seal.json` | `2b8a3591eb7b3d18b2587f38526cbe139c7b7d40686d8a041b992f07f13635ea` |

Any mismatch is `FAIL_CLOSED`; it is not a reason to regenerate or patch an
existing sealed artifact.

## One fixed, already-consumed fixture

| Field | Fixed value |
|---|---|
| session | C1 development `sub-C_ses-CO-20151103`, first literal `val` manifest row only |
| session pin | `7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7`, `62145872` bytes |
| model | `shared_t4`, seed 44, terminal `epoch_011.ckpt` only |
| checkpoint pin | `a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6`, `64769167` bytes |
| signal view | SUA |
| source normalizers | behavior `821e98bc…4235cd`; T4-side `32d32a7f…261701`; semantic `ac515609…e8a7a7` |
| chronology | rewarded support exactly `trials[0:50]`; C1 identity selection `first_n30`; query exactly `trials[50:]` |
| finite batch | rows `0:16` of the first C1 post-50 `DataLoader(batch_size=128, shuffle=False, num_workers=0)` batch |

The helper calls C1-owned loaders and preprocessing rather than recreating a
parallel loader: `load_session_with_trials`,
`select_calibration_trial_indices`, `build_calib_trials_for_indices`,
`make_subset_dataset`, and `attach_side_features`.  It uses C1’s
`_unpack_loader_batch` and `decode_last_behavior` at forward time.  It exposes
the same prepared tensors to the reference invocation and to a future adapter
invocation; neither path parses, reorders, bins, normalizes, or attaches T4
rows independently.

The reference forward body lives once in the new shared observation helper.
It retains the C1 `decoder_key_features → student(... ) →
decode_last_behavior` sequence.  The future adapter receives the resulting
prepared C1 batch and can only return a raw decoder output; trace creation,
last-bin scaling, target extraction, digesting, and TorchMetrics update are
again shared once.  A second loader/forward/metric implementation is
forbidden.

## Observation ledger and hard stop rules

The eventual parity run is CPU-only: `CUDA_VISIBLE_DEVICES=""`, CPU model,
deterministic algorithms enabled, TF32/autocast disabled, and one intra-op and
inter-op thread.  It is a compatibility execution, not an external score run.

The helper emits the following for both labels (`c1_reference` and
`future_adapter`):

- ordered trial/support/query selection digest;
- each prepared neural/history, calibration, side-feature, electrode-id and
  target tensor’s dtype, shape, byte hash and value hash;
- `n_units`, identity row map, and T4 row attachment digest;
- raw decoder output, scaled last-bin prediction and target arrays;
- exact query row selector plus Torch/TorchMetrics versions and final R².

`FAIL_CLOSED` applies if any input digest/shape/dtype/order differs, if either
prediction differs (`np.array_equal` required; `max_abs=0`, `max_rel=0`), if a
value is nonfinite, or if the absolute R² difference / independent CPU
TorchMetrics recomputation exceeds final-v2 `R2_RECOMPUTE_ATOL = 1e-6`.

The one future execution root is exclusive-create only.  It must seal
`input_trace.json`, two trace JSONs, reference/adapter prediction-target NPZs,
`receipt.json`, and `seal.json`, each fsync’d, SHA-256 indexed and mode `0444`.
The receipt schema is the existing v2-required
`dandi_000688_subc_scorer_adapter_parity_receipt_v1`; its only passing status
is `PARITY_CONFIRMED_CONSUMED_SUBC_DEV_SESSION`.  A failure receipt preserves
the first failure but cannot be root-pinned.  Both outcomes must explicitly
state that no sub-M path was constructed/accessed, no external endpoint score
was computed, and no external score authorization was granted.

## Deliberate blocker

The score-only v2 core still has both
`SCORER_ADAPTER_PARITY_RECEIPT_PIN = None` and
`ROOT_ED25519_PUBLIC_KEY_PIN = None`.  This v2 protocol/prelaunch does not
relax either.  The new runner's execute mode therefore fail-closes before it
can load the fixed checkpoint or development session.  A separate approved
turn must bind this prelaunch, add a valid one-time authorization, run the
parity fixture once, independently review its seal, and only then append a
root pin in a later score-runner revision.
