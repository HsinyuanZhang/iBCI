# M1 RIFT joint B3S concat protocol

`m1_joint_train.py` independently trains `B_ACTIVITY_ONLY` and `D_JOINT` on
the frozen M1 source-four R100 face.  Both arms use a trainable B3S
`SideFeatureEarlyPoolEncoder` over exactly ten M10 calibration trials per
session.  B supplies zero side and direct carrier; D supplies rSyn3 through
both paths.  The full-concat RIFT decoder and B3S encoder are jointly trained.

The formal recipe is B32 AdamW, LR 1e-4, 6,665-update warmup, cosine schedule,
EMA .9995, 24 epochs/159,960 updates.  HO M10 is visible development-calib
only, may overlap query recordings, and is never an official test surface.
