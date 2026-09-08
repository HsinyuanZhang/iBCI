# FALCON training and evaluation contracts — 2026-09-08

This is a read-only protocol audit. Each entry distinguishes **source-config evidence** from a **completed-receipt**. A running arm has a fixed configuration but no result claim. No 688 material is read or included.

## M1 R100 concat seed42 — completed

**Source-config evidence:** [m1_concat_train.py](../scripts/rift_v1/m1_concat_train.py) SHA `68d9d35727101a90bc9113ab1ff0a6554c3cd6340ee926b1d58a3f4c207d0668` fixes four source sessions (`ses-20120924`, `ses-20120926`, `ses-20120927`, `ses-20120928`), M10 support, 213,336 source windows, batch 32, 24 epochs × 6,665 updates, AdamW, MSE, peak LR `1e-4`, one-epoch warmup/cosine floor `1e-5`, weight decay `.01`, clip `1.0`, EMA `.9995`, and post-training HO-calib M10 EMA selection.

**Completed-receipt evidence:** [run_meta.json](../results/rift_v1/m1_r100_concat_s42_formal_v1/run_meta.json) SHA `46184396bea473343b89cd66d6f7c31cbc9bf7e1bbd4d87c4467cdb58d9c188f`; `train_receipt.json`; `score_receipt.json`. Query surface is three visible HO-calib sessions, 3,881 windows; pick is earliest maximum equal-session mean EMA e3; official test is false.

## M1 R100 joint D/B — configuration frozen, results pending

**Source-config evidence:** [m1_joint_train.py](../scripts/rift_v1/m1_joint_train.py) SHA `15affe894d757f0acbed302e3226f6e9299190dad33a1d359c18a991c7c2dfc7` fixes exactly the same four source sessions, M10, windows, batch, epochs, optimizer, MSE/LR/EMA schedule, and HO-calib M10 selection contract as concat; the arm field is the declared D or B joint arm. Root `64961` running does not make these fields unknown; it only leaves train/score results pending.

## M2 R50 joint B/D — ext4 mechanism

**Source-config evidence:** [m2_joint_train.py](../scripts/rift_v1/m2_joint_train.py) SHA `8c1307cab905df8093ddc5ff9df9b518f775006d1ff90bb2a7c139969925a139` fixes seven source sessions, M33 support, context 50/depth 4/local attention, batch 32, 24 epochs × 3,165 updates, AdamW, MSE, peak LR `3e-4`, cosine floor `3e-5`, weight decay `.01`, clip `1.0`, EMA `.9995`; source gradients are source-train only. Ext4 has four visible target sessions and is the post-training earliest-best equal-session-mean EMA selection surface.

**Completed-receipt evidence:** `results/rift_v1/m2_r50_joint_{b,d}_s42_formal_v1/{run_meta,train_receipt,score_receipt}.json`; no official test. B42/D42 results are completed; C42/B43 completion and D43 running status do not change this protocol.

## M2 ext6 candidates

Ext6 is a six-session submission-pick surface only. concat e9 `0.3900577` and D42 e13 `0.3478275` are candidates; they do not alter the four-session ext4 mechanism contract.

## H1 actual frozen R300 recency e22

**Source-config evidence:** [h1_train.py](../scripts/rift_v1/h1_train.py) SHA `2ef5daee9de79ac6f5dfc12c6830d3d8b926dbade3055f447ab7fdaefdd3b18e` and [run_meta.json](../results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/run_meta.json) SHA `94248c4267b1b5caac8230ccaf8bc8055cd3889392e091bc4ecdd07a226d5696` bind the actual 13 source-session roster, R300 context, batch/microbatch 32, 32 epochs × 731 updates, AdamW/MSE, peak/floor LR `1e-4/1e-5`, weight decay `.01`, clip `1.0`, one warmup epoch and EMA `.9995`. Source training banks are `V1 build_cal1_banks C2-CAL-1 B2`; this receipt does **not** establish M3 as the source-training support. Deployment banks use the separately bound public M3 carrier construction.

The current bound helper makes the training support precise: [h1_c2_cal1_b2_l200_p16.py](../../btransform_unified_v1/scripts/h1_c2_cal1_b2_l200_p16.py) SHA `7154deb38b6f1d2f536b7020153fb82fa48c07ab8b138d7336d1961eecee83e8` builds each source bank from an allowed start with activity budgets `M∈{7,5,4,3}`. [c2_protocol.py](../../btransform_unified_v1/src/btransform_unified_v1/c2_protocol.py) SHA `44d2e53f1dd8297d0b95572bd5127468c9a441894b26130088611a5cef6b5819` fixes the C2 prefix cycle `7,5,4,3`; `h1_train.py` selects the corresponding bank at each step. Carrier construction uses M4 `fit_frozen_carrier(first4)` for activity M7/M5/M4 and uses `fit_deployment_carrier(first3)` for activity M3. The helper also binds selected `q=12`, `lambda=10` and source RMS `6.8260113140959355e-06`; its source basis is rebuilt from the 13-source first-four-trial face. This differs from the deployment/target M3 carrier contract, which is always first-three public trials.

This is a current-helper and actual-R300-bank-roster binding, not a claim to have recovered every historical helper revision: the R300 run metadata does not independently hash every helper file. That limitation does not alter the recorded R300 run contract or the separate deployed M3 binding.

**Completed-receipt evidence:** [train_receipt.json](../results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/train_receipt.json) SHA `64f9ff37ef3ee6b6341bb7ab538915d50ff7a31bdb1cc005661d74415e9d29bb`; HO-M3 seven-group earliest-best EMA e22 selection; frozen official arm is R300 recency e22, never an R200 support run.
