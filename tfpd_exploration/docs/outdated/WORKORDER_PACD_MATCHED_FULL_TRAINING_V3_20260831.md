# Work Order: PACD Matched Full Training V3 — 2026-08-31

## 1. Purpose

V3 is a narrow evidence-policy successor to PACD full V2. It exists only
because the sole V2 P0 attempt stopped before its first optimizer update when
the paired whole-unit dropout mask retained zero units and therefore produced
an exactly zero identity-encoder gradient.
V3 does not change the PACD hypothesis, model, task loss, optimizer, RNG
replay, data order, or update budget.

This work order authorises code construction and independent no-data review.
It does not by itself authorise a live launch.

## 2. Immutable V2 predecessor

V3 must descriptor-read, through one held directory FD with `O_NOFOLLOW`, the
exact eight-leaf graph at:

`tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p0_fullfull_seed42`

Every leaf must be a regular mode-`0444` file, every sidecar must use the
canonical basename, no extra leaf may exist, and the following body digests
must match exactly:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `f0bc06ee590a68c669819ee4a2893e1f5c3af9408fd4fb2b198dacafaedb4bf6` |
| `launch.json` | `25de41fbd71fe6fed6f691db7acb018db03a47ba59ed6aa1a6f08995fa73410a` |
| `source_authority.json` | `452ed16be3c76f5e793ff9e21a58dad33d30e26bdfb1d4ace1cbab3e96e30071` |
| `failure.json` | `c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0` |

The validator must additionally require:

- V2 P0 arm and schema;
- implementation closure
  `74b9495597fa8ca1b2a6731d78c2fb10108e47e2fc4cc00540b34d68c377763c`
  at launch and final failure;
- failure kind `PACDError` and detail `encoder gradient is zero`;
- zero epoch receipts, zero checkpoints, no SWA, and no terminal;
- `target_access=false`;
- the exact V1 predecessor failure binding carried by V2.

This validation occurs before V3 root freshness/reservation and again before
V3 terminal or failure publication.

## 3. Unchanged scientific contract

V3 inherits the full numerical contract from V2:

- seed 42;
- 48 epochs;
- 33,925 paired optimizer steps per epoch;
- batch size 32 and `num_workers=0`;
- canonical Cell-D initial artifact and strict-loaded state;
- Adam construction and inherited learning-rate schedule;
- P0 M30/M30, P1 M30/M4, and P2 M30/M10;
- ordinary M30 T4 in both branches;
- paired Python/Torch/CUDA RNG replay;
- the same whole-unit dropout probability and mask in each pair;
- loss `0.5 * L_anchor + 0.5 * L_short`;
- exactly one optimizer step per accepted paired batch;
- the same source roster, sampler order, final-four checkpoints, and SWA rule;
- no target data, target loss, target gradient, or target update.

The network graph and inference interface remain unchanged.

## 4. Correct gradient evidence rule

Strictly positive encoder and decoder norms are not symmetric per-step
correctness invariants. In the real Cell-D graph, a whole-unit dropout mask
with `retained == 0` deliberately cuts the calibration identity path and makes
all identity-encoder gradients exactly zero, while the decoder query/representation
path still receives a finite nonzero task gradient. V3 recognises only this
explicitly observed stochastic state; it does not broadly waive zero encoder
gradients.

For every paired step V3 must:

1. compute and record finite anchor and combined encoder/decoder gradient
   norms;
2. reject any non-finite norm;
3. require a strictly positive combined decoder norm before
   `optimizer.step()`;
4. require a strictly positive combined encoder norm unless the two recorded
   paired masks are equal and their common evidence has `retained == 0`;
5. reject a zero encoder norm when one or more units were retained;
6. preserve the exact optimizer-step and RNG-transition invariants;
7. record the zero-encoder event with the typed reason
   `all_units_dropped_valid_zero`, never as an unlabeled waiver.

For every complete epoch V3 must publish:

- min/mean/max for all four gradient-norm streams;
- exact counts of zero encoder-gradient steps and their typed all-units-dropped
  justifications;
- zero decoder-gradient steps equal to zero;
- at least one strictly positive combined encoder gradient in the epoch;
- at least one strictly positive combined decoder gradient in the epoch;
- the existing sentinel branch/cosine evidence and finite-state evidence.

If an entire epoch has no encoder connection or no decoder connection, the
run fails closed before the next epoch. This rule prevents a globally
disconnected route while allowing valid batch-level sparsity.

The accepted V2 source smoke already establishes that the real sealed Cell-D
encoder path can carry nonzero gradient on source batches. A no-data CPU
reproduction additionally establishes that fixed `p=1.0` gives zero gradients
for all eight Cell-D encoder parameters, nonzero decoder gradients, and no Adam
state before the rejected update. V3 must retain this real Cell-D regression,
reject zero encoder gradient with `retained>0`, and reject an all-epoch-zero
encoder coverage trace.

## 5. Implementation discipline

V3 must be additive and reuse the single V1/V2 full lifecycle through the
immutable execution-profile seam. It may add only the smallest
backward-compatible gradient-evidence policy hook needed by the shared paired
operator/epoch aggregator. V1 and V2 default behavior remains their historical
strict policy; no runner, model, optimizer loop, or paired forward may be
copied.

Public CLI execution remains impossible. A live run requires a fresh opaque,
one-shot in-process capability that binds the exact V2 predecessor, current V3
closure, arm, canonical root, runtime factory, and selected GPU0 profile.

## 6. Canonical V3 roots and order

- P0: `tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42`
- P1: `tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p1_m4_seed42`
- P2: `tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p2_m10_seed42`

Only P0 may launch first. P1 and P2 remain forbidden until P0 reaches a valid
48-epoch terminal with four checkpoints, SWA, manifest, exact source-only
facts, and no failure. Existing V1/V2 roots are immutable evidence and must
never be overwritten, resumed, or retried.

## 7. Required no-data tests

Before launch review, tests must prove:

- exact held-FD acceptance and adversarial rejection of the V2 failed graph;
- V1/V2 historical strict policy remains unchanged;
- a real Cell-D `retained==0` finite-zero encoder/nonzero decoder step reaches
  exactly one update under the V3 policy;
- the same zero encoder norm with `retained>0` fails before update;
- a non-finite gradient or all-zero combined trainable gradient fails before
  update;
- epoch-level encoder/decoder positive-coverage gates;
- real Cell-D inactive-lazy topology remains safe;
- P0 prediction/identity equality, paired mask equality, and RNG equality;
- V3 success and failure lifecycle topology under a typed synthetic runtime;
- closure reconstruction, inert CLI, and absence of Torch/CUDA import in dry
  mode.

## 8. Decision rule

V3 is a launchable engineering successor only after independent root review.
It creates no new scientific claim. The first scientific decision remains the
matched P0/P1/P2 score comparison after all authorised full terminals exist.
