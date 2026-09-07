"""V3 all-source EP-FILM constants: the V2 contract with the LODO loop removed.

Training hyperparameters are inherited verbatim by import from the sealed V1
plan module (``h1_calibration_profile_film_v1.plan``); nothing in this file
redefines them.  This module only pins the new all-source authorities that the
V3 workorder attaches to the unchanged training internals.
"""
from __future__ import annotations

from h1_calibration_profile_film_v1.plan import (  # noqa: F401  (verbatim contract)
    BATCH_SIZE,
    DATE_ORDER,
    EPOCHS,
    EXPECTED_GPU0_UUID,
    FILM_PARAMETERS,
    LEARNING_RATE,
    PREDICTION_DIVISOR,
    SEED,
    TRAIN_STRIDE,
    WEIGHT_DECAY,
)

SCHEMA = "h1_calibration_profile_film_v3"
ARM = "EP-FILM"
ANCHOR_ARM = "EP-ZERO"
RESULT_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v3"
WORKORDER_RELATIVE = "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_EPFILM_ALLSOURCE_V3_20260904.md"
DATA_RELATIVE = "SPINT-main/data/000954"

# Sealed all-source substrate: the decoder class deployed by the M3RC/581792
# chain, imported and pinned by its immutable import receipt.
ALL_SOURCE_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/artifacts/h1_c1_all_source_epoch49_v1"
ALL_SOURCE_CKPT_NAME = "epoch_049.ckpt"
ALL_SOURCE_CKPT_SHA256 = "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06"
ALL_SOURCE_IMPORT_RECEIPT_SHA256 = "f1791b5d7f8fd3d76bd7cc612f72af95fc9ff79838d7b60745eeeb0c4213ff0d"
ALL_SOURCE_MODEL_STATE_SHA256 = "bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85"
ALL_SOURCE_MODEL_PARAMETERS = 10_947_836

# Pinned legacy checkout that owns the all-source plan producer code.  The
# runner takes the path as an argument and requires the HEAD below.
LEGACY_HEAD = "5dd9bb4a7377a5431b7dbac4f1378e529130eb1a"
LEGACY_PLAN_RELATIVE = (
    "tfpd_exploration/h1_series_20260830/results/"
    "h1_cal_aug_all_source_m3_deployment_v1/source_authority/plan.json"
)
LEGACY_PLAN_SHA256 = "a92b57350f2dcb04027bb6d848e5582d84e1bef4bf507d6844963ccad3c87bd5"
LEGACY_CLOSURE_RELATIVES = (
    "SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1_exec.py",
    "SPINT-main/src/h1_hc_date_lodo_regen_v1.py",
    "SPINT-main/src/h1_m4_cce_contract.py",
    "SPINT-main/src/data/h1_m4_eb_pilot.py",
)

# The sealed all-source plan selection (teammate authority, mirrored by the
# local M3RC package constants).
ALL_SOURCE_DOMAIN = "h1_all_source_13"
ALL_SOURCE_SELECTED = {"q": 12, "lambda": 10.0}
SOURCE_SELECTION_SHA256 = "3a9f59f75da95aab056850f70e4b5c47afc093125e02e789f0c2f7db59ce6067"
SEALED_TRANSFORM_SHA256 = "1c566312152d0203b282fd62a415694d9ddf5845a0208c291e4240e2b9b3ccd7"
SEALED_SOURCE_NORMALIZER = 6.8260113140959355e-06
SOURCE_NORMALIZER_RELATIVE_TOLERANCE = 1.0e-12

# Cached family-A deployment payload (M3RC package == EvalAI 581792 class).
M3RC_PACKAGE_RELATIVE = "SPINT-main/local_data/h1_m3rc_evalai_v1/decoder.pt"
M3RC_PACKAGE_SHA256 = "731725de2bacda34aa015c095d7208aff8f4dad4f3892d6efeeb0af28d4629b7"
M3RC_CALIBRATION_AUTHORITY_RELATIVE = (
    "tfpd_exploration/h1_series_20260830/results/"
    "h1_m3_readout_calibration_evalai_package_v1/calibration_authority.json"
)
M3RC_CALIBRATION_AUTHORITY_SHA256 = "46aeb22d9bdbb5925d980ae3229257ab38c094eff49d52b3fa2222412cd384a0"

# V2 sealed root (read-only authority: the contract this run inherits).
V2_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v2"
V2_SCORE_SHA256 = "8b56b105bf1cd965ffa9d74e323dbce7b5d2e2410a8abfb88b5eaa2151494813"
V2_TERMINAL_SHA256 = "f6432d0832d0173313dc9678078e6e4d348928218396b602d7320ec1b7dcb38d"
V2_INITIAL_FILM_STATE_SHA256 = "ff2263b5705a499caeb5fe807f9889479fdbc31f806b42de82b11d351ba210cf"

# Roundoff-scale carrier deviation tolerated between the locally rebuilt plan
# and the sealed family-A plan cached in the M3RC deployment payload.
CARRIER_RELATIVE_DEVIATION_BOUND = 1.0e-9

IMPLEMENTATION_RELATIVES = (
    "tfpd_exploration/h1_series_20260830/src/h1_calibration_profile_film_v1/plan.py",
    "tfpd_exploration/h1_series_20260830/src/h1_calibration_profile_film_v1/core.py",
    "tfpd_exploration/h1_series_20260830/src/h1_calibration_profile_film_v1/evaluate.py",
    "tfpd_exploration/h1_series_20260830/src/h1_calibration_profile_film_v3/__init__.py",
    "tfpd_exploration/h1_series_20260830/src/h1_calibration_profile_film_v3/plan.py",
    "tfpd_exploration/h1_series_20260830/src/h1_calibration_profile_film_v3/evaluate.py",
    "tfpd_exploration/h1_series_20260830/scripts/run_h1_calibration_profile_film_v3.py",
)


__all__ = ("ARM", "ANCHOR_ARM", "SCHEMA")
