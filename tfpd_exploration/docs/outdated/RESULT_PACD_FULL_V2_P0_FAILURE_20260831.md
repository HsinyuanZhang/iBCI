# PACD Full V2 P0 Failure Record — 2026-08-31

## Outcome

The sole authorised `PACD_MATCHED_FULL_TRAINING_V2` P0 launch failed closed
before the first optimizer update completed. This is not a training result and
must not be interpreted as evidence for or against PACD.

- Canonical root: `tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p0_fullfull_seed42`
- Status: `CELL_FAILED`
- Target access: `false`
- Epoch receipts: `0`
- Checkpoints: `0`
- SWA artifacts: `0`
- GPU after failure: GPU0 idle; GPU1 untouched

The immutable root contains exactly the attempt, launch, source-authority, and
failure body/sidecar pairs. Every leaf is a regular mode-`0444` file and every
body digest matches its canonical sidecar.

| Body | SHA-256 |
|---|---|
| `attempt.json` | `f0bc06ee590a68c669819ee4a2893e1f5c3af9408fd4fb2b198dacafaedb4bf6` |
| `launch.json` | `25de41fbd71fe6fed6f691db7acb018db03a47ba59ed6aa1a6f08995fa73410a` |
| `source_authority.json` | `452ed16be3c76f5e793ff9e21a58dad33d30e26bdfb1d4ace1cbab3e96e30071` |
| `failure.json` | `c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0` |

The launch and final implementation closure are identical:
`74b9495597fa8ca1b2a6731d78c2fb10108e47e2fc4cc00540b34d68c377763c`.
The attempt also binds the accepted V1 predecessor failure SHA
`65b125b982506c82a9daa9eb32d88fed1afca7355155abeab2c224f31b4c3d08`.

## Exact failure

The source roster, canonical initial checkpoint, CUDA binding, and immutable
launch receipts were established. The first paired step then stopped at:

```text
paired_anchored_calibration_dropout_v1/core.py:298
PACDError: encoder gradient is zero
```

The strict-positive combined encoder-gradient assertion is evaluated before
`optimizer.step()`. Consequently this attempt contributed no valid optimizer
update and cannot be resumed.

## Interpretation and successor rule

The failure is currently classified as an evidence-policy failure, pending a
real Cell-D reproduction. A zero encoder gradient on one stochastic batch can
be legitimate; finiteness, graph connectivity over a fixed audit set, and
nonzero task/decoder learning must not be conflated with a requirement that
every branch have a strictly positive norm on every update.

Any successor must:

1. bind this exact eight-leaf V2 failed graph through held no-follow
   descriptors;
2. prove with the real Cell-D graph whether the observed zero encoder gradient
   is a valid batch-level event or a disconnected training path;
3. preserve the model, loss, paired RNG replay, calibration prefixes, optimizer,
   batch order, and update budget;
4. record finite encoder/decoder gradient norms and aggregate coverage without
   requiring a strictly positive encoder norm at every individual step unless
   disconnection is independently proved;
5. use a fresh successor root; never overwrite or retry this V2 root.

P1 and P2 remain unlaunched until a successor P0 reaches a valid terminal.
