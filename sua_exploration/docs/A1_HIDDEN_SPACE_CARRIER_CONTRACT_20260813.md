# A1 — independent hidden-space carrier CPU contract

**Date:** 2026-08-13
**Status:** independent CPU reference/oracle contract. **It authorizes no GPU run, no training, no data access, no formal-test open, and no official receipt.**
**Scope:** the minimal A1 `W/H × Z4/T4` attribution design only.

This document is intentionally additive. It does not modify, replace, or bind any existing production trainer, model, config, runner, A2 artifact, B1 artifact, or main handoff. Its executable companion is [a1_hidden_adapter_reference.py](../mc_maze/a1_hidden_adapter_reference.py), with synthetic-only tests in [test_a1_hidden_adapter_reference.py](../tests/test_a1_hidden_adapter_reference.py). Those files do not import production A1 code, open a dataset, read a checkpoint, or write a receipt.

The purpose is to state the narrow A1 CPU contract that a later production integration must meet. Passing these reference checks is necessary evidence about algebra and construction parity; it is not a performance result and is not an authorization to run a pilot.

## 1. Frozen scientific question

The existing W-add path enters the student decoder as

```text
h_W = fc_in(x + E_A(Z4)).
```

The only new family is the hidden-space T4 carrier path

```text
h_H = fc_in(x + E_A(Z4)) + P(T4),
P: R^4 → R^512.
```

`E_A` is the **matched A2 B3S activity identity**: `variant=B3S`, `side_dim=4`, calibrated identity, with the A2 Z4/activity representation and its frozen source-only normalizer authority. It is not a newly invented zero-feature surrogate, a different identity encoder, or a target-refit representation.

`P` is exactly one bias-free weight matrix of shape `[512, 4]`. It must be directly created as exact zeros without sampling an initializer or otherwise advancing the construction RNG. Therefore:

- `P.bias` does not exist;
- `P(0)` is an exact all-zero `[B,N,512]` tensor both at initialization and after training;
- at initialization, `P(T4)` is exactly zero for every T4 input;
- constructing H immediately after the same seed leaves the CPU RNG state exactly where constructing W leaves it.

The location is deliberate: `P(T4)` is added **after** `fc_in` and **before** the student decoder's remaining attention/transformer/readout operations. A waveform-side addition, a decoder-key addition, a logit residual, a bias term, a widened side feature, or a second decoder is outside this contract.

## 2. Source-training and target-session semantics

During source training only, the teacher is frozen. The trainable student set is jointly:

```text
student decoder + matched E_A + P
```

The source loss is exactly `task_only`; this contract adds no teacher/distillation auxiliary loss. In particular, zero-initializing P is not permission to freeze P, freeze the student decoder, or train H in a different optimizer topology from the matched source setup.

Target-session behavior is forward-only and fixed-weight:

- target direction labels may construct the session-local T4 carrier from the chronological first 30 rewarded trials;
- no target backward pass occurs;
- no target velocity label is used for a weight update;
- no decoder, E_A, P, normalizer, optimizer, or checkpoint is updated;
- the target direction carrier fit is not decoder adaptation.

The reference code does not perform either source training or target scoring. It provides only synthetic mathematical gates that a production integration must satisfy.

## 3. Minimal attribution matrix

The logical matrix is exactly four cells:

| Logical cell | Definition | Provenance | Is a new source-training family? |
|---|---|---|---|
| `W/Z4` | `fc_in(x + E_A(Z4))` | sealed A2 reuse | no |
| `W/T4` | sealed ordinary A2 B3S/T4 W-add result | sealed A2 reuse | no |
| `H/Z4` | exact structural alias of `W/Z4` | alias, not a run | no |
| `H/T4` | `fc_in(x + E_A(Z4)) + P(T4)` | new A1 family | **yes** |

`H/Z4` is not a separately initialized model, a separately trained cell, a separately scored checkpoint, or an opportunity to alter the activity identity. Its score, hidden state, prediction, shared gradients, and shared optimizer state are required to be exactly those of `W/Z4`; its `P` input is an exact zero tensor. This structural alias is what makes the main interaction an attribution rather than a generic comparison of two independently trained families.

The matrix is deliberately minimal. `H/T4` is the only new family. If an attachment control is later reviewed, `H/TS4` must be a deterministic attachment-only evaluation of the already trained `H/T4` family (the same source checkpoint and all other bindings, but unit-shuffled T4 input to P). It must not be silently promoted into another source-training family, and it is not a fifth term in the primary interaction.

## 4. Primary estimand

For a matched session/seed score table, the only primary statistic is

```text
I = (H_T4 - H_Z4) - (W_T4 - W_Z4).
```

The production scorer may calculate this scalar only from already bound scores. It may not load data, select a checkpoint, construct a carrier, train, update a target model, or create a receipt. The reference function `score_only_primary_interaction` rejects a missing cell, a non-finite score, or an `H/Z4` value that is not exactly equal to `W/Z4`.

Because the alias is exact, `I` is numerically equal to `H_T4 - W_T4`. That numerical simplification must not replace the interaction in an aggregate, plot, or claim: the interaction is what records the Z4 structural control and prevents a generic H-family lift from being mislabeled as T4-specific benefit.

## 5. A2 sealed-W reuse: exact deny-by-default checklist

`W/Z4` and `W/T4` may be reused only if **every** item below is exact. A missing field, a type change, a changed path/hash/value, an altered source roster order, a refit, or a target update means **do not reuse**. The independent oracle has no artifact I/O; a later integration must collect the bound evidence from its own immutable inputs and pass it to `validate_a2_reuse_evidence`.

### 5.1 Immutable A2 anchors

| Binding | Exact value |
|---|---|
| A2 terminal aggregate SHA-256 | `5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc` |
| A2 official preflight SHA-256 | `8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd` |
| A2 contract SHA-256 | `f8d2f1f9e2420423584f18e667df27bc209c5ae5d1730232d703886244ea5cb2` |
| A2 config SHA-256 | `68aa9b599b5d30c690af8639ece6f3d2e7d63e47c5ef3f7fedcd853cae6ed73c` |
| A2 implementation-bindings SHA-256 | `7e46115fc3bafa1010d454c80215d6cf99987973464b0720997366d92b2887b3` |

### 5.2 Teacher, manifest, backbone, and W-cell identity

| Binding | Exact requirement |
|---|---|
| teacher path | `checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt` |
| teacher SHA-256 | `9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d` |
| teacher during source training | frozen |
| train/validation manifest | `configs/subc_co_27_6_strict_train_val_manifest.json` |
| manifest SHA-256 | `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9` |
| backbone | `B3S`, `side_dim=4`, `identity_mode=calibrated`, `decoder_mode=coupled` |
| sealed `W/Z4` cell | `source_arm=source_z4`, `side_feature_group=z4`, B3S, side_dim 4 |
| sealed `W/T4` cell | `source_arm=source_t4`, `side_feature_group=t4`, B3S, side_dim 4 |

The reused source seed must match the A1 source seed. Its sealed epoch-5–12 checkpoint-bundle SHA-256 must be the corresponding entry below; the integration must also verify the individual checkpoint hashes inside the bundle against the sealed A2 receipt.

| Seed | `W/Z4` (`source_z4`) bundle SHA-256 | `W/T4` (`source_t4`) bundle SHA-256 |
|---:|---|---|
| 42 | `c1d5482640eea00b649810b0107527cdcfb2c1d0b5c0cbbb886887a21eb27a45` | `16daf346541c7594ec9bb097ddae369899dc59f16bfe4f5bfd2e993565409260` |
| 43 | `dfe39ea79db1c4a856e4932d18ddfd26542688f67950fc1ec53ee1cf399e6085` | `dbcc34ada3aa616a933d55840f7f5c31687f8fe02c993dae798fd6a3f0711e65` |
| 44 | `25ffc59a551f83d5b7c146a9a3f8df98488550beeda45780e70e563ec3229c01` | `200c39adac1e7d3a37865b833ce15c582b6ec9bb50ee8d9a2d11e9191562bab2` |

### 5.3 T4 normalizer and exact source roster

The T4 normalizer is source-only, with authority `strict_subc_source_train_27_only`, value SHA-256
`293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0`, and no target-domain refit. The ordered source roster is exactly:

```text
sub-C_ses-CO-20131003
sub-C_ses-CO-20131022
sub-C_ses-CO-20131023
sub-C_ses-CO-20131031
sub-C_ses-CO-20131101
sub-C_ses-CO-20131203
sub-C_ses-CO-20131204
sub-C_ses-CO-20131219
sub-C_ses-CO-20131220
sub-C_ses-CO-20150309
sub-C_ses-CO-20150311
sub-C_ses-CO-20150312
sub-C_ses-CO-20150313
sub-C_ses-CO-20150319
sub-C_ses-CO-20150629
sub-C_ses-CO-20150630
sub-C_ses-CO-20150701
sub-C_ses-CO-20150703
sub-C_ses-CO-20150706
sub-C_ses-CO-20150707
sub-C_ses-CO-20150708
sub-C_ses-CO-20150709
sub-C_ses-CO-20150710
sub-C_ses-CO-20150713
sub-C_ses-CO-20150714
sub-C_ses-CO-20150715
sub-C_ses-CO-20150716
```

No target-domain behavior or side normalizer refit is permitted. The historical source-only behavior normalizer authority remains `strict_subc_source_train_27_only`; normalizer identity is not allowed to drift because the new H/T4 branch is convenient to construct.

### 5.4 M30, chronology, query, epoch, loss, and isolation

All of the following must match exactly:

| Topic | Exact requirement |
|---|---|
| activity calibration support | first 30 rewarded trials, chronological (`selection_mode=first`) |
| T4 label pool | first 30 rewarded trials, chronological |
| evaluation start | trial index 30 |
| query rule | usable rewarded trials `[30:]`; every 50-bin window is wholly contained in a trial strictly after chronological rewarded trial 30 |
| trial filter / geometry | rewarded `R`; 20 ms bins; window 50 bins; trial length 100 bins |
| source schedule | total 12 epochs; no early stopping; exactly epochs 5–12 |
| epoch score | unweighted mean session R² over source epochs 5–12 |
| source loss | `task_only` |
| formal data | no formal-test NWB opened and no test files evaluated |
| target update | `backward_gradients=false`, `decoder_weight_updates=false`, `target_velocity_labels_used_for_weight_updates=false` |
| target carrier provenance | `target_session_carrier_fit_performed=true`, `target_direction_labels_used_for_carrier=true` |

The six sealed formal sub-C sessions stay excluded. Their names may be retained only as an exclusion list; they may never be opened, scored, or used as a data source for A1.

## 6. Mandatory CPU-only oracle gates

The independent synthetic test suite requires all of the following on CPU `float32` tensors:

| Gate | What is proven |
|---|---|
| constructor parity | H construction consumes no RNG beyond W construction |
| P shape/bias/init | exact `[512,4]`, no bias, direct all-zero parameter |
| P-zero invariant | `P(0)` is exact zero |
| forward lattice | for every `B ∈ {1,2}` and `N ∈ {1,7,64}`, W/Z4 and H/Z4 hidden outputs and predictions are `torch.equal` |
| task-only step | with the same synthetic batch, loss and all shared gradients are bit-equal and `P.grad` is a materialized exact zero tensor |
| optimizer parity | after at least three AdamW-like source steps, all shared parameters and shared optimizer state are bit-equal |
| checkpoint round trip | an H state dict and optimizer round trip preserves hidden state, prediction, and optimizer state exactly |
| nonzero-carrier sensitivity | one backward pass with nonzero T4 gives `grad_P > 0`; an update changes both hidden state and prediction |
| negative controls | a biased P spec fails, and a nonzero H/Z4 carrier port fails the structural-alias check |
| A2 reuse | the validator accepts only the exact evidence tree above and rejects a changed normalizer, target update, bundle, or any other frozen field |

The tests are intentionally an independent oracle. They do not claim that an arbitrary production adapter has passed merely because it has a similarly named class.

## 7. Production integration boundary and source-binding hygiene

The preferred future integration is an **additive A1-only subclass or wrapper** that owns the H path and its direct-zero P parameter. It should leave the shared W-add base byte-for-byte unchanged. This preserves the old W-add implementation binding for unrelated work while giving A1 its own explicit binding surface.

Do not create P with `nn.Linear(..., bias=False)` followed by `zeros_`: the default `nn.Linear` constructor samples its initial weight before it is zeroed, so it advances the global RNG and fails this contract's construction-parity rule. Directly create the zero `[512,4]` parameter and apply a bias-free linear operation instead.

Adding an optional hidden-carrier argument or an H-only branch to a shared base file is material even when the default behavior is numerically unchanged: a SHA-bound W-add consumer observes a different source file. If a production implementation must modify the shared base rather than use an additive wrapper/subclass, then:

1. the complete B1 lattice requires a new binding and a fresh rerun under a successor preflight; preserving only one pair is insufficient;
2. all affected A12 work requires its own successor binding/preflight before any further receipt is accepted;
3. the old B1/A12 artifacts remain historical and must not be silently merged with successor-bound output.

This is a source-integrity consequence, not a criticism of fail-closed behavior. A B1 or A12 stop caused by the SHA mismatch is the correct outcome until the source surface is settled.

## 8. Development receipt schema, not a receipt writer

`a1_development_receipt_schema()` returns a descriptive schema only. A later development receipt must be explicitly non-authorizing and include:

- the full A2 reuse evidence tree, validated before W reuse;
- all four logical matrix cells, with `H/Z4.structural_alias_of="W/Z4"`;
- source-training and target-session policy facts;
- the primary interaction written in its full 2×2 form;
- an immutable-integrity record.

If a later reviewed process is authorized to write a **development** receipt, its body and its separate `.sha256` sidecar must each use `O_CREAT|O_EXCL`, `fsync`, and mode `0444`; acceptance must verify both paths and the digest. The schema specifically prohibits official receipt minting, formal-test access, and target backward/update behavior. This contract creates no file and mints no receipt.

## 9. Frozen Stage-P routing proposal (not an authorization)

The recommended development-only order is seed 42 on the six validation sessions. Let

```text
d[s] = R(H/T4, s) - R(W/T4, s).
```

With the exact H/Z4 alias, this is algebraically the primary interaction per session, but the aggregate must still report the four-cell interaction from section 4. The frozen routing threshold is:

1. `mean_s d[s] ≥ +0.03`;
2. `median_s d[s] > 0`;
3. at least 4 of 6 session deltas are strictly positive;
4. the H/TS4 attachment control is completed from the same H/T4 source family before seed 43/44 expansion is considered.

The gate is mechanically attainable: six deltas of `+0.04` pass all three numerical requirements. It can fail, for example, with three positive and three negative deltas. Thus no impossible discrete p-value rule is hidden in the screen. It is **not** a six-session confirmatory statistical test: 4/6 positive signs has one-sided null probability `22/64 = 0.34375`, so it cannot support a conventional significance claim. Stage P is routing only; it authorizes neither GPU execution nor automatic expansion. Any seed 43/44 decision remains a separate reviewed authorization after the attachment-control result is visible.

## 10. Current decision

**GPU GO: NO.** The only completed evidence described here is a synthetic CPU oracle. Before a future review can consider a pilot, the production integration must satisfy the exact A2 reuse validator, restore or preserve source-binding hygiene through an additive A1 wrapper/subclass, prove all parity gates against its actual code path, and obtain separate authority.
