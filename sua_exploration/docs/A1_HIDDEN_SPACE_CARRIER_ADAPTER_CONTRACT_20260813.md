# A1 — hidden-space carrier adapter production contract

**Date:** 2026-08-13  
**Screen ID:** `a1_hidden_space_carrier_v2`  
**Status:** production code and CPU evidence ready. The CPU preflight is non-authorizing; the one-cell runner separately requires explicit root-go and GPU authorization. No formal-test access is permitted.

## Scientific question and minimal matrix

The ordinary W interface is the sealed A2 B3S path. A1 retains the matched activity identity but moves the T4 carrier to a bias-free hidden residual:

```text
h_W = fc_in(x + E_A)
h_H = fc_in(x + E_A) + P(T4),     P ∈ R^(512×4), bias = 0
```

The logical matrix is exactly:

| Logical cell | Evidence | Fresh training? |
|---|---|---:|
| `W/Z4` | sealed A2 within-subject receipt | no |
| `W/T4` | sealed A2 within-subject receipt | no |
| `H/Z4` | exact structural alias of `W/Z4` | no |
| `H/T4` | A1 hidden adapter | **yes** |

`H/Z4` is not a separately initialized, trained, or scored family. CPU production-path proofs require its zero carrier port, forward output, loss, shared gradients, P gradient, shared optimizer state, and checkpoint round trip to match `W/Z4` exactly. Its score points to the immutable `W/Z4` receipt SHA.

`H/TS4` is an attachment-only control of the **same trained H/T4 checkpoint**. The scorer loads the aligned checkpoint, switches only the unit permutation in evaluation mode, and refuses such a switch in training mode. It is not a fifth logical cell and creates no training run.

## Source and deployment contract

Only `H/T4`, seed 42, is eligible for the development pilot. Source training uses the existing MC-Maze teacher and jointly trains the student decoder, matched B3S activity encoder, and P:

- `task=CO`, `variant=B3S`, `side_dim=4`, coupled decoder, `fixed_slot_count=0`;
- `freeze_decoder=false`, `freeze_encoder_base=false`;
- `task_only`, `lambda_y=0`, `lambda_E=0`;
- chronological first 30 rewarded trials for activity and T4;
- 12 epochs, no early stopping, immutable score rule: unweighted mean of epochs 5–12;
- source train/validation rosters exactly 27/6 from the strict manifest.

Target-session calibration and scoring remain forward-only: T4 is fitted from the first 30 target-session direction labels, query begins at trial 30, no target backward pass occurs, no target velocity label updates weights, no decoder/encoder/P/normalizer/checkpoint is updated, and the source-only T4 normalizer is reused.

## Sealed A2 reuse gate

The preflight refuses W-cell reuse unless all historical immutable bodies, SHA sidecars, modes, and internal bindings pass. It verifies:

- terminal aggregate SHA `5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc`;
- official preflight SHA `8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd`;
- contract/config/implementation binding digests, teacher and manifest bytes;
- exact W/Z4 and W/T4 within-subject receipt bodies for the selected seed;
- literal six-session development roster;
- source metadata including `freeze_decoder=false`, `freeze_encoder_base=false`;
- all eight epoch-5–12 checkpoint hashes for each W arm;
- each checkpoint's 31 decoder + 8 identity-encoder tensor optimizer coverage;
- M30 chronology, query-after-30 policy, source-only T4 normalizer SHA, zero target updates, and no formal data access.

The A1 adapter is additive. The shared `streaming_spint.py` remains byte-identical to sealed A2 SHA `141129622ee2c187c053a0139f3926ab1880871d51dc88d0f8b93cc1b734ad9a`.

## Frozen routing gate

For each of the six development sessions,

```text
d[s] = (R(H/T4,s) - R(H/Z4,s)) - (R(W/T4,s) - R(W/Z4,s)).
```

Because `H/Z4` is the exact `W/Z4` alias, this is numerically `R(H/T4,s)-R(W/T4,s)`, but the aggregate must retain the complete interaction form.

The seed-42 routing pilot passes only if:

1. mean `d[s] ≥ +0.03`;
2. median `d[s] > 0`;
3. at least 4/6 session deltas are positive;
4. the same-checkpoint `H/TS4` attachment control is complete.

This is a development routing gate, not a confirmatory significance claim. Seed 43/44 expansion requires a later reviewed decision; it is not automatic.

## Receipt lifecycle

The official CPU preflight, pretraining launch record, score receipt, and aggregate are written with `O_CREAT|O_EXCL`, file `fsync`, mode `0444`, a separate immutable `.sha256` sidecar, and parent-directory `fsync`. Readers reject missing pairs, changed bytes, symlinks, or modes other than `0444`.

The preflight itself sets `authorizes_gpu=false`. The runner additionally requires explicit root-go plus GPU authorization and accepts only `CELL=H/T4`, `SEED=42`. It never resolves or opens the six formal sub-C NWBs.

## Data isolation

These formal sessions may appear only as an exclusion-name list and must never be resolved, loaded, or scored:

- `sub-C_ses-CO-20151113`
- `sub-C_ses-CO-20151116`
- `sub-C_ses-CO-20151117`
- `sub-C_ses-CO-20151119`
- `sub-C_ses-CO-20151120`
- `sub-C_ses-CO-20151201`

## Production entrypoints

- model: `streaming_calibration_exp/src/models/a1_hidden_carrier_module.py`
- student adapter: `streaming_calibration_exp/src/models/components/streaming_spint_hidden_carrier_adapter.py`
- dedicated trainer: `sua_exploration/a1_hidden_carrier/trainer.py`
- dedicated scorer: `sua_exploration/a1_hidden_carrier/scorer.py`
- A2 anchor verifier: `sua_exploration/a1_hidden_carrier/a2_anchors.py`
- immutable artifacts: `sua_exploration/a1_hidden_carrier/artifacts.py`
- preflight: `sua_exploration/scripts/a1_hidden_space_carrier_preflight.py`
- runner: `sua_exploration/scripts/run_a1_hidden_space_carrier_one_cell.sh`
- aggregate: `sua_exploration/scripts/aggregate_a1_hidden_space_carrier.py`

This contract creates no authority to open formal test sessions.
