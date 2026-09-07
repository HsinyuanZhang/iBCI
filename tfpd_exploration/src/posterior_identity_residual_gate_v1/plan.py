"""Static, standard-library-only plan for PIRG."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


CELL = "POSTERIOR_IDENTITY_RESIDUAL_GATE_V1"
PHASE = "PIRG_SOURCE_ONLY_ALPHA_GATE_V1"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
BUDGETS = (4, 10, 30)
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_BASELINE_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_IDENTITY_RESIDUAL_GATE_20260823.md"

# These are execution prerequisites for the frozen Cell-D initialization, not
# an additional training input.  ``matched_score_physical.load_sealed_cell_d_material``
# remains the sole semantic validator; this table makes the three exact
# body+sidecar pairs transport-visible so a fresh PIRG stage can satisfy that
# already-reviewed loader before it opens source data or initializes CUDA.
#
# Keep each canonical basename sidecar literal in the table rather than
# reconstructing an undocumented transport convention downstream.
SEALED_CELL_D_INIT_ASSETS = (
    (
        "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json",
        SEALED_CELL_D_TERMINAL_SHA256,
        0o444,
        "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442  terminal_receipt.json\n",
    ),
    (
        "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt",
        SEALED_CELL_D_SWA_SHA256,
        0o444,
        "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd  swa_final4.pt\n",
    ),
    (
        "tfpd_exploration/results/sparsification_score_v1/sparsification_score_receipt.json",
        SEALED_CELL_D_BASELINE_SHA256,
        0o444,
        "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f  sparsification_score_receipt.json\n",
    ),
)


def sealed_cell_d_init_assets_payload() -> list[dict[str, object]]:
    """Return the fixed, transport-ready Cell-D initialization asset table.

    This pure helper intentionally does not inspect a historical result root.
    A future transport must descriptor-read these named leaves and the frozen
    matched loader validates their full terminal/SWA/baseline semantics.
    """
    rows: list[dict[str, object]] = []
    for relative, sha256, mode, sidecar in SEALED_CELL_D_INIT_ASSETS:
        canonical_sidecar = f"{sha256}  {Path(relative).name}\n"
        if mode != 0o444 or sidecar != canonical_sidecar:
            raise ValueError("PIRG sealed Cell-D initialization table drift")
        sidecar_bytes = sidecar.encode("ascii")
        rows.append({
            "relative_path": relative,
            "sha256": sha256,
            "body_mode": "0444",
            "sidecar_relative_path": f"{relative}.sha256",
            "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "sidecar_mode": "0444",
            "sidecar_contents": sidecar,
        })
    return rows

# The physical source adapter is deliberately reused rather than rebuilt.  A
# reviewed stage therefore carries only these small immutable authorities; the
# 9.1-GB strict-27 NWBs stay at the typed external source-data root.  The
# manifest has no sidecar/mode-0444 claim, while the other four paired leaves
# do and are checked by the adapter itself through held descriptors.
SOURCE_ADAPTER_AUTHORITY_ASSETS = (
    (
        "tfpd_exploration/results/admission_arms_v1/preflight_armA.json",
        "2632c6a6a4cfb8a4c0fb2b23e0cc8205ea323240b376903c60d5b27110f59e43",
        0o444,
    ),
    (
        "tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority_receipt.json",
        "d023dd632c4717443f1f55e924a09be1747fc58d5c30a8fb6fa38f4b7b117184",
        0o444,
    ),
    (
        "tfpd_exploration/results/sparsification_theta_authority_v1/theta_authority.pt",
        "cef39dc8220aa253214963a32e5457dede1045e64b158e37fc267a6fb4146319",
        0o444,
    ),
    (
        "sua_exploration/results/misleading_identity_swap_v2_source_authority_dev/strict27_m30_source_lineage_v3.json",
        "7375a37c8c5e59cb7e6786e0c67e930f918d59ff4dc769742391ad6dc97579fa",
        0o444,
    ),
    (
        "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json",
        "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9",
        None,
    ),
)

# The score route composes this immutable engineering substrate in place.  The
# source-only route never imports it; literals live here so both dry plans can
# describe their dependency boundaries without importing a Torch module.
V3_QUICK_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_carrier_quick_screen_stage_v3"
V3_QUICK_INPUT_AUTHORITY_SHA256 = "bad14dadec2c4e525c4cffbaf689b75ce1cebfa9bccf27bc022256574176386e"
V3_QUICK_SCORE_SHA256 = "07bb30a018ad33a63d89ed3cd4fcc0293d7150b59262f0db84ae9ae5ad6183b5"
V3_QUICK_TERMINAL_SHA256 = "51103a5703ff6933362f11913afd379c3b580c59abac62630194bf2a44626949"


def budget_for(epoch: int, session_index: int) -> int:
    if type(epoch) is not int or not 0 <= epoch < EPOCHS:
        raise ValueError("PIRG epoch must be an exact integer in [0, 3)")
    if type(session_index) is not int or session_index < 0:
        raise ValueError("PIRG session index must be a nonnegative exact integer")
    return BUDGETS[(epoch + session_index) % len(BUDGETS)]


def dry_plan() -> dict[str, Any]:
    """Return a pure contract.  No file, data, Torch, or network access."""
    return {
        "status": "DRY_NO_TORCH_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "cell": CELL,
        "phase": PHASE,
        "seed": SEED,
        "source_only": True,
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "budget_rotation": "(4,10,30)[(epoch + source_session_index) % 3]",
            "loss": "dense_valid_bin_supervised_mse",
            "optimizer": "Adam(alpha_only; weight_decay=0)",
        },
        "held_cell_d": {
            "swa_sha256": SEALED_CELL_D_SWA_SHA256,
            "initialization_assets": sealed_cell_d_init_assets_payload(),
            "ols_point_carrier": True,
            "ordinary_ols_normalizer": True,
            "b3s_activity_path": True,
            "decoder": True,
            "whole_unit_dynamic_dropout": "U(0,1) existing law",
        },
        "only_new_parameter": {"name": "alpha", "shape": [], "initial_value": 0.0},
        "gate": "1 + 0.5*tanh(alpha)*tanh(clamp(log(r)-mean(log(r)),-4,4))",
        "forbidden": {
            "posterior_mean_carrier": True,
            "posterior_normalizer": True,
            "posterior_sampling": True,
            "attention_logit_credibility_bias": True,
            "activity_gate": True,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "formal_opened": False,
        },
        "quick_score": {
            "surfaces": ["within_fixed3", "external_fixed3"],
            "budgets": [30, 10, 4],
            "query": "last_bin_variance_weighted_r2_equal_session",
            "m30_safety_delta_min": -0.02,
            "m4_promising_mean_delta_min": 0.03,
            "m4_promising_positive_count_min": 4,
        },
        "execution": "requires future in-process root-reviewed capability; not available from this CLI",
    }
