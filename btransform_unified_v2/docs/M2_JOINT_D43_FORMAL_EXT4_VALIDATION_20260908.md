# M2 joint D43 formal ext4 validation

The read-only validation artifact at `results/rift_v1/m2_r50_joint_d_s43_formal_v1/validation_audit.json` passed. The run is formal `D_JOINT`, model seed `43`, sampler seed `42`, with 24 training epochs and 75,960 updates. All live source and cache hashes, all 24 checkpoint hashes, and the complete ext4 EMA score map matched their receipts.

The control surface is the fixed four-session ext4 set with 2,069 windows per epoch: 2020-10-30 Run1 `519`, 2020-10-30 Run2 `490`, 2020-11-18 Run1 `425`, and 2020-11-19 Run1 `635`. It is mechanism-control evidence and must not be relabeled as ext6.

The earliest maximum is D43 EMA epoch 18 at `0.4202184879917157`; epoch 24 is `0.3969813828305111`. The selected per-session R² values are `0.4721844296597112`, `0.36685289320749426`, `0.44544801965276615`, and `0.396388609446891` in the session order above.

For the within-model-seed B43/D43 paired comparison, B43 independently selects epoch 19 while D43 selects epoch 18. The independent-pick D minus B equal-session delta is `0.10702026564802603`; the four within-seed session deltas are `0.007156476987068894`, `-0.04438542055401362`, `0.2793157817461912`, and `0.1859942244128575`. At fixed epoch 24, the D minus B equal-session delta is `0.11501540413062017`, with session deltas `0.0007719710849061201`, `0.051485130464621554`, `0.24946120741284628`, and `0.15834330756010673`.

These four sessions are repeated measurements within model seed 43. They do not create additional independent training seeds, and the result does not report an ext6 outcome or a cross-seed conclusion.
