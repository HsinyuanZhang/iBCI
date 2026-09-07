"""C1-specific M3->M7 frozen-weight H1-CAC Stage-1 evaluator.

This route consumes the five immutable date-LODO C1 checkpoints uploaded at
one fixed Git commit.  It reconstructs no learned object at target time: the
source-selected H-C plan bytes and normalizer are immutable imported
artifacts, every model is frozen, and only label-free activity state changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .core import array_sha256, require
from .evaluate import DATA_RELATIVE, _evaluate_arm, _session_material
from .plan import (
    ARM_ORDER,
    CHUNK_LENGTH_BY_OUTER_DATE,
    DATE_ORDER,
    MAX_MEMBERS,
    SCHEMA,
    STAGE1_SUPPORT,
    WINDOW,
    decide,
)


ARTIFACT_RELATIVE = (
    "tfpd_exploration/h1_series_20260830/artifacts/"
    "h1_c1_date_lodo_epoch49_v1"
)
SOURCE_COMMIT = "5b21de415afc35a5f4ad63dd2e8a459d925dbbf7"
COMMON_CONFIG_SHA256 = "d9a6f5163b9887439de47febdc6c13f3803f9b94a545646488993db212ae1453"
MODEL_PARAMETERS = 10_947_836
INITIAL_STATE_SHA256 = "bc6dc8a0543c760811f770206c7ee22ae35eaf970c6dad0ec259a84172e4d04b"


@dataclass(frozen=True)
class C1Authority:
    checkpoint_sha256: str
    terminal_sha256: str
    terminal_state_sha256: str
    source_authority_sha256: str
    plan_sha256: str
    plan_npz_sha256: str
    normalizer_sha256: str


C1_AUTHORITIES = {
    "19250108": C1Authority(
        "904216a596fb71abc72ce2fcdba9ae207eeb258be5fa32fb2aaf533ab42dd047",
        "083a6016d4fc053be34beb2f21bc6b9b24343c134f62ba257ea69fec05a1eec3",
        "db41c05f70af87b58bf05e441f5821ad269a002bd119d34a94f4bc8d77b0d836",
        "8cdecf55d406ebf1175009140b44a1e5cf7d515a8d054c2416695834f4a9fd2b",
        "3f525e5f2b2fb59aadebe8efc4375671d0737a2995a399cea44b381beaa20f1e",
        "1c9af578fb333ebe77acad00f6fdf2da4a6c40fa3dbc1099f308d2bf27d725ad",
        "cfe5d7c8f9307ecb2e0962001272e9c37f5309809b63521182df9c3933b6b1ee",
    ),
    "19250113": C1Authority(
        "2a114c5f2edddef9df19167e1d756ab80c1cfa04ae20b46343cb7a848f252d35",
        "b4d7e301f917477c2e790c7823219ce62f99b0114f41005d1cbc610051da60cb",
        "b2c9f02fac46a2e8edffb9ab534cf965509556053a6e9b08d4007dc685d09448",
        "078a57fcf79ebbb161f9d15daabb09100fb22b3dee2a0668ea92c509256c6cf5",
        "6e8c84abcec1996b2d89c4b810bf23430c5b60e3bed6ad51f70e69aaed50c9d4",
        "d632a373d712b3f5c9a7882a4f50fd24f58af244e178fd4132bbc1d0309b4e7c",
        "9bb7e3628ebb01b09157f6e76144c06ca34175fbcc777f14afb4c49f65c6cc38",
    ),
    "19250115": C1Authority(
        "0a15fbed098a6488496f8ec7caa5e9ed2ed04c165db80b9ee817c456698af932",
        "4aba995c4ba30d45d01dd8ba8a3feeda534d1bb9ae67037bdd29519bbd54e19d",
        "16d5765437d93fcf804e0967cf9956848bdfd372e2138a880e38926e15ad9696",
        "67bedcb96a6ba82d6730bb8718df2058fb3fd56c12c7013ead64fa90b1f6f406",
        "304b0b09d334322e531b4fab95ff8936b9a64a335588d209b70568d95d46f98b",
        "d9e96af3effcdefe2c4782be5edb6138aa1ef9254020b69947c2c4ce834f8ea3",
        "560137bda5d945ba1fa9487b37d5b7981b3385727a8b2dc877a5978b4d1741a4",
    ),
    "19250119": C1Authority(
        "276a789edae2419704f4649465a477511b7571185441267c1de1073374fecc42",
        "377874fc77795c238203e2e45d9e538e7f7b6b6e16b9f5c7d30286822aa01153",
        "8523215e59ed189889f9a4e7eb25fc8834aaa35efd39fba8ab41488973e83ccb",
        "ef89ada380c6329480e6d6ef71715c88bb81c81c124dc4362b48dbd0fd1c891e",
        "edbe067fdc2a3d668a5c01af20e5f2f2a4862823b6777d359e29427c81708647",
        "173f6c0ba4b276f46b485e3dda5c7656daaca4adeaa427a198219a647310db8c",
        "23e2c6055388b723580afc0fc774ec24f68d4b223eed09d336f486e4c9f124b6",
    ),
    "19250120": C1Authority(
        "5d7632c604ad92e52a6ae2bfbc77a875ed7b7a12b5ebc44bbfbb35aa1691d186",
        "6a754210892c7a714c7ea2fdc9e44b21d34489c10226d6be137b3f3f0d8fa0de",
        "bda95cbd8f9b71db0f3e48012153b1789f3280eec33e8036202e442af4e96786",
        "cd088871268054223a1f14666ffd493d7711e627e92bfe02af8dc782696a6f5d",
        "e7d43cbf1ef87bbb9294d14aab4bbe16c480035fd959b114bd52eb64e218315b",
        "c9f70f536cf0d0a9fd71b35928d1dae4432a4792bdd7617df0f97b8abdc44221",
        "acb4c2caab3cb7fac6ff2c205c40c9b0a30e599b77137c5c1bb12182e7b2cbb9",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked authority: {path}")
    info = path.stat()
    require(stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
            f"authority must be immutable 0444/nlink1: {path}")
    observed = _sha256(path)
    require(observed == expected, f"authority SHA drift: {path}")
    side = path.with_name(path.name + ".sha256")
    require(side.is_file() and not side.is_symlink(), f"authority sidecar missing: {side}")
    side_info = side.stat()
    require(stat.S_IMODE(side_info.st_mode) == 0o444 and side_info.st_nlink == 1,
            f"authority sidecar must be immutable 0444/nlink1: {side}")
    require(side.read_text(encoding="ascii") == f"{observed}  {path.name}\n",
            f"authority sidecar drift: {side}")
    return observed


def _load_plan(directory: Path, authority: C1Authority, outer_date: str) -> tuple[Any, float, dict[str, Any]]:
    from src.data.h1_m4_eb_pilot import array_sha256 as source_array_sha256

    plan_path = directory / "source_authority" / "plan.json"
    npz_path = directory / "source_authority" / "plan.npz"
    normalizer_path = directory / "source_authority" / "normalizer.json"
    authority_path = directory / "source_authority" / "authority.json"
    _verify_file(authority_path, authority.source_authority_sha256)
    _verify_file(plan_path, authority.plan_sha256)
    _verify_file(npz_path, authority.plan_npz_sha256)
    _verify_file(normalizer_path, authority.normalizer_sha256)
    source = json.loads(authority_path.read_text(encoding="utf-8"))
    body = json.loads(plan_path.read_text(encoding="utf-8"))
    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    require(
        source.get("schema") == "h1_hc_date_lodo_regen_v1_date_source_authority"
        and source.get("status") == "PASS_H1_HC_DATE_LODO_REGEN_V1_SOURCE_AUTHORITY"
        and source.get("outer_date") == outer_date
        and source.get("plan_sha256") == authority.plan_sha256
        and source.get("normalizer_sha256") == authority.normalizer_sha256
        and source.get("target_recordings_opened") == 0
        and source.get("target_bytes_read") == 0,
        f"{outer_date} source-plan authority drift",
    )
    require(
        body.get("schema") == "h1_hc_date_lodo_regen_v1_plan"
        and body.get("outer_date") == outer_date
        and body.get("arrays_file_sha256") == authority.plan_npz_sha256,
        f"{outer_date} source plan schema/date drift",
    )
    with np.load(npz_path, allow_pickle=False) as payload:
        require(set(payload.files) == {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"},
                f"{outer_date} source plan array set drift")
        values = {name: np.asarray(payload[name], dtype=np.float64) for name in ("mean", "scale", "pcs", "U", "mu")}
        q = int(payload["q"].item())
        ridge_lambda = float(payload["lambda"].item())
        tau2 = float(payload["tau2"].item())
    require(
        q == int(body["q"])
        and ridge_lambda == float(body["lambda"])
        and tau2 == float(body["tau2"])
        and {name: source_array_sha256(value) for name, value in values.items()} == body["array_sha256"],
        f"{outer_date} source plan array/scalar drift",
    )
    require(
        normalizer.get("schema") == "h1_hc_date_lodo_regen_v1_normalizer"
        and normalizer.get("formula") == "s_src=sqrt(mean(C_src_raw**2)); C_norm=C_raw/max(s_src,1e-12)"
        and float(normalizer.get("floor")) == 1.0e-12
        and np.isfinite(float(normalizer.get("s_src")))
        and float(normalizer.get("s_src")) > 0.0,
        f"{outer_date} source normalizer drift",
    )
    plan = SimpleNamespace(
        outer_date=outer_date,
        source_sessions=tuple(body["source_sessions"]),
        source_input_sha256=tuple(body["source_input_sha256"]),
        mean=values["mean"], scale=values["scale"], pcs=values["pcs"], q=q,
        ridge_lambda=ridge_lambda, U=values["U"], mu=values["mu"], tau2=tau2,
    )
    return plan, float(normalizer["s_src"]), {
        "source_authority_sha256": authority.source_authority_sha256,
        "plan_sha256": authority.plan_sha256,
        "plan_npz_sha256": authority.plan_npz_sha256,
        "normalizer_sha256": authority.normalizer_sha256,
        "q": q,
        "ridge_lambda": ridge_lambda,
        "normalizer_s_src": float(normalizer["s_src"]),
        "source_sessions": list(plan.source_sessions),
    }


def _load_model(directory: Path, authority: C1Authority, outer_date: str, device: str) -> tuple[Any, str, dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint

    terminal_path = directory / "c1" / "terminal.json"
    config_path = directory / "c1" / "config.json"
    checkpoint_path = directory / "c1" / "epoch_049.ckpt"
    _verify_file(terminal_path, authority.terminal_sha256)
    _verify_file(config_path, COMMON_CONFIG_SHA256)
    _verify_file(checkpoint_path, authority.checkpoint_sha256)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(
        terminal.get("schema") == "h1_cal_aug_prefix_cycle_v1"
        and terminal.get("status") == "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_ARM_EPOCH49"
        and terminal.get("outer_date") == outer_date
        and terminal.get("arm") == "c1"
        and terminal.get("epoch_zero_based") == 49
        and terminal.get("checkpoint", {}).get("sha256") == authority.checkpoint_sha256
        and terminal.get("config_sha256") == COMMON_CONFIG_SHA256
        and terminal.get("initial_state_sha256") == INITIAL_STATE_SHA256
        and terminal.get("terminal_state_sha256") == authority.terminal_state_sha256
        and terminal.get("target_recordings_opened") == 0
        and terminal.get("target_bytes_read") == 0
        and terminal.get("target_optimizer_steps") == 0
        and terminal.get("target_backward_steps") == 0,
        f"{outer_date} C1 terminal contract drift",
    )
    require(
        config.get("schema") == "h1_cal_aug_prefix_cycle_v1_common_config"
        and config.get("arm") == "c1"
        and config.get("c1_cycle") == [7, 5, 4]
        and config.get("terminal_epoch_zero_based") == 49
        and config.get("warm_start") is False,
        f"{outer_date} C1 config drift",
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    require(
        isinstance(metadata, Mapping)
        and payload.get("schema") == "h1_cal_aug_prefix_cycle_v1_checkpoint"
        and metadata.get("outer_date") == outer_date
        and metadata.get("arm") == "c1"
        and metadata.get("epoch_zero_based") == 49
        and metadata.get("terminal_state_sha256") == authority.terminal_state_sha256
        and metadata.get("target_recordings_opened") == 0
        and metadata.get("target_bytes_read") == 0
        and metadata.get("target_optimizer_steps") == 0
        and metadata.get("target_backward_steps") == 0,
        f"{outer_date} C1 checkpoint metadata drift",
    )
    model = H1CarrierIdSpint(**config["base"]["model_kwargs"])
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            f"{outer_date} C1 strict load drift")
    state = state_hash(model.state_dict())
    require(state == authority.terminal_state_sha256, f"{outer_date} C1 state SHA drift")
    require(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS,
            f"{outer_date} C1 model parameter-count drift")
    model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state, {
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": authority.checkpoint_sha256,
        "terminal_sha256": authority.terminal_sha256,
        "config_sha256": COMMON_CONFIG_SHA256,
        "terminal_state_sha256": authority.terminal_state_sha256,
        "global_step": int(terminal["global_step"]),
        "strict_load_missing_keys": [],
        "strict_load_unexpected_keys": [],
        "model_parameter_count": MODEL_PARAMETERS,
    }


def _first_valid_bin(record: Any, trial_value: float) -> int:
    indices = np.flatnonzero(
        record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == float(trial_value))
    )
    require(indices.size > 0, f"{record.session_name}: fourth trial lacks an eval-valid bin")
    return int(indices[0])


class C1M3StrictTargetDataset:
    """Exact post-M3 query surface with one frozen M3 H-C per session."""

    def __init__(self, records: Mapping[str, Any], plan: Any, s_src: float, *, outer_date: str) -> None:
        from src.data.h1_m4_eb_pilot import fit_deployment_carrier, interpolate_trial_identity
        from src.h1_m4_cce_contract import NORMALIZER_FLOOR, canonical_sha256

        self.records = dict(records)
        self.outer_date = outer_date
        self.support: dict[str, Any] = {}
        self.window_indices: list[tuple[str, int]] = []
        support_receipts: dict[str, Any] = {}
        for session, record in self.records.items():
            require(len(record.trial_values) >= STAGE1_SUPPORT + 1,
                    f"{session}: C1 Stage 1 requires a post-M3 query trial")
            values = tuple(float(value) for value in record.trial_values[:STAGE1_SUPPORT])
            query_trial = float(record.trial_values[STAGE1_SUPPORT])
            boundary = _first_valid_bin(record, query_trial)
            identity = np.ascontiguousarray(
                np.stack([interpolate_trial_identity(record, value) for value in values]), dtype=np.float32,
            )
            carrier = np.ascontiguousarray(
                fit_deployment_carrier(record, plan, values)["carrier"]
                / max(float(s_src), NORMALIZER_FLOOR),
                dtype=np.float32,
            )
            require(identity.shape == (3, 1024, 176) and carrier.shape == (176, 4),
                    f"{session}: C1 M3 identity/carrier geometry drift")
            require(np.isfinite(identity).all() and np.isfinite(carrier).all(),
                    f"{session}: C1 M3 identity/carrier is nonfinite")
            self.support[session] = SimpleNamespace(carriers={"full": carrier})
            for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
                if bool(record.eval_mask[start + WINDOW - 1]):
                    self.window_indices.append((session, int(start)))
            support_receipts[session] = {
                "support_trials": list(values),
                "query_trial": query_trial,
                "query_first_valid_bin": boundary,
                "identity_sha256": array_sha256(identity),
                "normalized_carrier_sha256": array_sha256(carrier),
            }
        require(bool(self.window_indices), f"{outer_date}: C1 M3 target surface is empty")
        self.window_indices_sha256 = canonical_sha256(self.window_indices)
        self._manifest = {
            "schema": f"{SCHEMA}_c1_m3_strict_target_v1",
            "outer_date": outer_date,
            "sessions": list(self.records),
            "support_members": STAGE1_SUPPORT,
            "query_starts_at_fourth_trial": True,
            "window_indices_sha256": self.window_indices_sha256,
            "samples": len(self.window_indices),
            "support": support_receipts,
        }

    def __len__(self) -> int:
        return len(self.window_indices)

    def manifest(self) -> dict[str, Any]:
        return self._manifest


def _evaluate_date(root: Path, outer_date: str, *, device: str) -> dict[str, Any]:
    import torch
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.h1_m4_cce_contract import state_hash

    authority = C1_AUTHORITIES[outer_date]
    directory = root / ARTIFACT_RELATIVE / outer_date
    plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    model, state_before, model_receipt = _load_model(directory, authority, outer_date, device)
    records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=outer_date)
    dataset = C1M3StrictTargetDataset(records, plan, s_src, outer_date=outer_date)
    chunk_length = CHUNK_LENGTH_BY_OUTER_DATE[outer_date]
    material = {
        session: _session_material(record, chunk_length=chunk_length, support=STAGE1_SUPPORT)
        for session, record in records.items()
    }
    results = [
        _evaluate_arm(
            model=model, dataset=dataset, material=material, arm=arm, device=device,
            support=STAGE1_SUPPORT,
        )
        for arm in ARM_ORDER
    ]
    require(len({row["target_sha256"] for row in results}) == 1,
            f"{outer_date} four-arm target authority drift")
    require(len({row["n_windows"] for row in results}) == 1,
            f"{outer_date} four-arm window-count drift")
    state_after = state_hash(model.state_dict())
    require(state_before == state_after, f"{outer_date} C1 frozen state changed")
    traces = {
        session: {
            "phase_origin": row["origin"],
            "B-TRIAL7": row["B-TRIAL7"].receipt(),
            "C-FIX7": row["C-FIX7"].receipt(),
            "D-EMED7": row["D-EMED7"].receipt(),
        }
        for session, row in material.items()
    }
    for row in results:
        row.pop("_prediction")
        row.pop("_target")
    return {
        "outer_date": outer_date,
        "sessions": list(records),
        "chunk_length": chunk_length,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "target_surface": dataset.manifest(),
        "state_traces": traces,
        "results": results,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


def _selected_fixed_chunk_readout(decision: Mapping[str, Any]) -> dict[str, Any]:
    per_date = decision["per_date"]
    content = float(decision["equal_date_mean"]["B-TRIAL7_minus_A-STATIC"])
    fixed = float(decision["equal_date_mean"]["C-FIX7_minus_A-STATIC"])
    values = [float(per_date[date]["C-FIX7_minus_A-STATIC"]) for date in DATE_ORDER]
    recovery = fixed / content if content > 0.0 else None
    passed = bool(
        decision["gate_1_content"]["pass"]
        and recovery is not None
        and recovery >= 0.50
        and sum(value >= 0.0 for value in values) >= 3
        and min(values) >= -0.010
    )
    return {
        "status": "POST_STAGE0_SELECTION_INFORMED_NON_GOVERNING_READOUT",
        "selection_history_disclosed": True,
        "selected_because_stage0_C_exceeded_stage0_D": True,
        "mean_gain": fixed,
        "recovery_fraction": recovery,
        "positive_dates": sum(value >= 0.0 for value in values),
        "worst_date_delta": min(values),
        "same_thresholds_as_primary_detector": {
            "recovery_fraction_min": 0.50,
            "positive_dates_min": 3,
            "worst_date_min": -0.010,
        },
        "pass": passed,
        "may_not_replace_primary_D_in_this_result": True,
    }


def run(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    require(device == "cuda:0" and torch.cuda.is_available(), "H1-CAC Stage 1 requires logical cuda:0")
    date_rows = [_evaluate_date(root, outer_date, device=device) for outer_date in DATE_ORDER]
    original = decide(date_rows)
    selected = _selected_fixed_chunk_readout(original)
    if not original["gate_1_content"]["pass"]:
        verdict = "STOP_C1_M3_ACTIVITY_CONTENT_DID_NOT_TRANSFER"
    elif original["gate_2_deployable_recovery"]["pass"]:
        verdict = "PASS_C1_M3_PRIMARY_ENERGY_CAC_FOR_PACKAGE_REVIEW"
    elif selected["pass"]:
        verdict = "C1_M3_CONTENT_POSITIVE_FIXED_CHUNK_ONLY_SUCCESSOR_CONFIRMATION_REQUIRED"
    else:
        verdict = "PASS_C1_M3_ACTIVITY_CONTENT_BUT_STOP_DEPLOYABLE_CAC"
    return {
        "schema": f"{SCHEMA}_stage1_c1_m3_score",
        "status": "COMPLETE_H1_CAC_STAGE1_C1_M3_FROZEN_WEIGHT_SCORE",
        "stage": 1,
        "surface": "five_date_lodo_strict_post_m3_common_query",
        "device": device,
        "source_commit": SOURCE_COMMIT,
        "date_order": list(DATE_ORDER),
        "arm_order": list(ARM_ORDER),
        "date_results": date_rows,
        "primary_preregistered_decision": original,
        "selection_informed_fixed_chunk_readout": selected,
        "verdict": verdict,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = (
    "ARTIFACT_RELATIVE", "C1_AUTHORITIES", "C1M3StrictTargetDataset", "SOURCE_COMMIT", "run",
)
