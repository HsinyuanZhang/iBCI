# M1 joint B/D pair validation — 2026-09-08

This note binds the completed M1 seed-42 B/D comparison. It is a local,
visible-HO3 development result and does not authorize an official submission,
additional training, or a generalization claim.

## Formal runs and validation

Both arms use `M1-RIFT-R100-D4-JOINT-B3S-CONCAT-V1`: a full R100/D4 concat
decoder with a live B3S encoder and decoder source-trained on four source
sessions. The source calibration is M10. The held-out face is visible HO3,
query trial 0, with 3,881 windows; support may overlap the query recordings.

| Arm | Formal run | Selected EMA | Selected equal-session mean | Epoch-24 equal-session mean |
|---|---|---:|---:|---:|
| B activity-only | `results/rift_v1/m1_r100_joint_b_s42_formal_v1` | e3 | `0.6630568974542209` | `0.5799890397983798` |
| D joint | `results/rift_v1/m1_r100_joint_d_s42_formal_v1` | e3 | `0.7342600300996954` | `0.6675785440106274` |

The B42 validation audit is
`results/rift_v1/m1_r100_joint_b_s42_formal_v1/validation_audit.json`,
SHA-256 `a01e762028186891d19e9ec0717c3d752239e8291960f3f360ff8b905e7b3a6e`.
It passed the formal 24-checkpoint, source, Sfix, raw/EMA finite, guard, and
selection checks. The already completed D42 audit is
`results/rift_v1/m1_r100_joint_d_s42_formal_v1/validation_audit.json`,
SHA-256 `7fa2c2092b34670cd368ee1795eb555d5ed7671334ce4b549abeddd5dd41a4f7`.

The receipt triplets are bound by SHA-256 as follows:

| Arm | `run_meta.json` | `train_receipt.json` | `score_receipt.json` |
|---|---|---|---|
| B | `bc5b7b542ea99e53d2d2877efe708757d4b05848ff2d2a6fa454739588269e5e` | `53e4a3c86eb23080e13a121e46401b6e189572f5ea1f7a47c164715b797bac4b` | `7ebb53eee59166e9db3d35a06171a692155392d334cf06eea5f19741085d4606` |
| D | `6f364e390b984ce5672f1968223676e70e12db9d203e6f30d72d56425d5d0096` | `9823c2dfe41ecf2e53732db23bc9374b853e88c35def42adcacea0fae3e0541f` | `0a4d6b7d5c32532a255d1f2c9392865b266ad34c57a49096a6d2820407c4d27e` |

## Paired summary

`results/diagnostics_v1/m1_joint_pair_summary_v1.json` has SHA-256
`b85d061fca7d13aed62b22be10bdb296428ffcca4c07364e1cf462017572d415`.
The helper verified all 48 B/D checkpoint files, the B/D/frozen-concat shared
contracts, and the selected checkpoint bindings.

At independently selected EMA epochs (B e3, D e3), D-minus-B is
`+0.07120313264547451` equal-session mean. The per-session differences are
`+0.03791812718721643`, `+0.07305975200941073`, and
`+0.10263151873979659` for 20121004, 20121017, and 20121024. At fixed epoch
24, D-minus-B is `+0.08758950421224765`; all three fixed-epoch session
differences are positive.

This is one paired seed-42 comparison. The three sessions are repeated
measurements within that seed, not three independent training replicates. It
does not provide a cross-seed CI, p-value, significance statement, private
split generalization result, or official-test result. D e3 exceeds frozen
concat e3 by only `+0.00006776958380461107`; this does not support a separate
joint-encoder benefit claim.
