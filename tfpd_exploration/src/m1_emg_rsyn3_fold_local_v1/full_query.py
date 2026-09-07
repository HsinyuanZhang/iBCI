"""Inference-only rescoring of sealed fold-local checkpoints on all M1 windows."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from . import carrier_bank
from . import gpu as stage1_gpu
from . import plan
from . import receipts as fold_receipts
from . import stage0 as fold_stage0
from .stage1 import ensure_streaming_path


class FullQueryError(RuntimeError):
    """Fail closed for inference-only full-query rescoring."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullQueryError(message)


def verify_pilot_arm_anchor(repo_root: Path, arm: str) -> dict[str, object]:
    """Verify the sealed pilot_r3 arm root and return the anchor literals."""
    _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
    root = Path(repo_root) / plan.STAGE1_ARM_ROOT_RELATIVE[arm]
    terminal_path = root / "terminal.json"
    _require(terminal_path.is_file(), "sealed terminal.json missing")
    terminal_bytes = terminal_path.read_bytes()
    terminal = json.loads(terminal_bytes)
    _require(isinstance(terminal, dict), "sealed terminal is not an object")
    _require(terminal.get("status") == "COMPLETE", "sealed terminal is not COMPLETE")
    _require(terminal.get("arm") == arm, "sealed terminal arm drift")
    bodies = terminal.get("bodies")
    _require(isinstance(bodies, dict), "sealed terminal bodies missing")
    body_shas: dict[str, str] = {}
    for name in plan.FULL_QUERY_PILOT_BODIES:
        path = root / name
        _require(path.is_file(), f"missing sealed body {name}")
        body = path.read_bytes()
        digest = plan.sha256_bytes(body)
        _require(name in bodies, f"sealed terminal missing body {name}")
        _require(bodies[name] == digest, f"sealed body digest drift {name}")
        sidecar = (root / f"{name}.sha256").read_text()
        _require(sidecar == f"{digest}  {name}\n", f"sealed sidecar drift {name}")
        body_shas[name] = digest
    table = json.loads((root / "arm_table.json").read_bytes())
    _require(isinstance(table, dict), "sealed arm_table is not an object")
    _require(table.get("arm") == arm, "sealed arm_table arm drift")
    _require(list(table["query"]) == [10, 210], "sealed arm_table query drift")
    static = json.loads((root / "score_static.json").read_bytes())
    cdm = json.loads((root / "score_cdm_a.json").read_bytes())
    _require(isinstance(static, dict) and isinstance(cdm, dict), "sealed score bodies")
    return {
        "schema": "m1_emg_rsyn3_fold_local_full_query_anchor_v1",
        "arm": arm,
        "pilot_root_relative": plan.STAGE1_ARM_ROOT_RELATIVE[arm],
        "checkpoint_relative": f"{plan.STAGE1_ARM_ROOT_RELATIVE[arm]}/epoch_011.pt",
        "checkpoint_sha256": body_shas["epoch_011.pt"],
        "terminal_sha256": plan.sha256_bytes(terminal_bytes),
        "arm_table_sha256": body_shas["arm_table.json"],
        "score_static_sha256": body_shas["score_static.json"],
        "score_cdm_a_sha256": body_shas["score_cdm_a.json"],
        "sealed_query": [10, 210],
        "sealed_static_r2": static["governing_r2"],
        "sealed_cdm_a_r2": cdm["governing_r2"],
        "sealed_static_prediction_sha256": static["prediction_sha256"],
        "sealed_cdm_a_prediction_sha256": cdm["prediction_sha256"],
        "sealed_n_windows": static["n_windows"],
    }


def load_student(repo_root: Path, checkpoint_path: Path, torch_module):
    ensure_streaming_path(repo_root)
    payload = torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require("epoch" in payload, "checkpoint payload has no epoch counter")
    _require(
        int(payload["epoch"]) == plan.FIXED_LAST_EPOCH_INDEX,
        f"checkpoint payload epoch {payload['epoch']} != fixed-last {plan.FIXED_LAST_EPOCH_INDEX}",
    )
    _require("state_dict" in payload, "checkpoint payload has no state_dict")
    from . import module as stage1_module

    lit = stage1_module.make_module(repo_root)
    lit.setup("fit")
    lit.load_state_dict(payload["state_dict"], strict=True)
    return lit


def window_json(window) -> list:
    start, end = window
    return [int(start), None if end is None else int(end)]


def build_arm_table(arm: str, anchor: Mapping, scores: Mapping[str, Mapping[str, Mapping]]) -> dict:
    _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
    rows: list[dict[str, object]] = []
    for name, window in plan.FULL_QUERY_WINDOWS:
        _require(name in scores, f"missing window scores {name}")
        pair = scores[name]
        _require("static" in pair and "cdm_a" in pair, f"score pair {name}")
        static = pair["static"]
        cdm = pair["cdm_a"]
        static_r2 = static["governing_r2"]
        cdm_r2 = cdm["governing_r2"]
        rows.append({
            "window": name,
            "query": window_json(window),
            "static_r2": static_r2,
            "cdm_a_r2": cdm_r2,
            "delta_cdm_minus_static": cdm_r2 - static_r2,
            "n_windows_static": static["n_windows"],
            "n_windows_cdm_a": cdm["n_windows"],
            "contract_relaxation": static.get("version_b_contract_relaxation"),
        })
    anchor_name = plan.FULL_QUERY_ANCHOR_WINDOW
    _require(anchor_name in scores, "missing anchor window scores")
    anchor_static = scores[anchor_name]["static"]
    anchor_cdm = scores[anchor_name]["cdm_a"]
    sealed_static_r2 = anchor["sealed_static_r2"]
    sealed_cdm_a_r2 = anchor["sealed_cdm_a_r2"]
    rescored_static_r2 = anchor_static["governing_r2"]
    rescored_cdm_a_r2 = anchor_cdm["governing_r2"]
    abs_diff_static = abs(rescored_static_r2 - sealed_static_r2)
    abs_diff_cdm_a = abs(rescored_cdm_a_r2 - sealed_cdm_a_r2)
    tolerance = plan.FULL_QUERY_ANCHOR_TOLERANCE
    n_anchor = int(scores[plan.FULL_QUERY_ANCHOR_WINDOW]["static"]["n_windows"])
    n_late = int(scores[plan.FULL_QUERY_LATE_HALF_WINDOW]["static"]["n_windows"])
    n_official = int(scores[plan.FULL_QUERY_OFFICIAL_EXTENT_WINDOW]["static"]["n_windows"])
    plus = n_anchor + n_late
    return {
        "schema": "m1_emg_rsyn3_fold_local_full_query_arm_table_v1",
        "arm": arm,
        "target_session": plan.FOLD0_TARGET_SESSION,
        "fixed_last": f"epoch_{plan.FIXED_LAST_EPOCH_INDEX:03d}",
        "checkpoint_sha256": anchor["checkpoint_sha256"],
        "rows": rows,
        "anchor_reproduction": {
            "window": plan.FULL_QUERY_ANCHOR_WINDOW,
            "sealed_static_r2": sealed_static_r2,
            "rescored_static_r2": rescored_static_r2,
            "abs_diff_static": abs_diff_static,
            "sealed_cdm_a_r2": sealed_cdm_a_r2,
            "rescored_cdm_a_r2": rescored_cdm_a_r2,
            "abs_diff_cdm_a": abs_diff_cdm_a,
            "tolerance": tolerance,
            "within_tolerance": bool(abs_diff_static <= tolerance and abs_diff_cdm_a <= tolerance),
            "static_prediction_sha256_match": (
                anchor_static["prediction_sha256"] == anchor["sealed_static_prediction_sha256"]
            ),
            "cdm_a_prediction_sha256_match": (
                anchor_cdm["prediction_sha256"] == anchor["sealed_cdm_a_prediction_sha256"]
            ),
            "n_windows_match": (
                int(anchor_static["n_windows"]) == int(anchor["sealed_n_windows"])
                and int(anchor_cdm["n_windows"]) == int(anchor["sealed_n_windows"])
            ),
        },
        "n_windows_additivity": {
            "q10_210_plus_q210_end": plus,
            "q10_end": n_official,
            "equal": plus == n_official,
        },
        "official_extent_window": plan.FULL_QUERY_OFFICIAL_EXTENT_WINDOW,
        "late_half_window": plan.FULL_QUERY_LATE_HALF_WINDOW,
        "inference_only": True,
        "target_optimizer_steps": 0,
        "formal_benchmark_verdict": False,
        "ls4_enabled": False,
    }


def execute_arm(
    repo_root: Path,
    *,
    arm: str,
    gpu_index: int,
    gpu_authorized: bool,
    pack: bool = False,
) -> tuple[dict[str, str], str | None, str | None]:
    repo_root = Path(repo_root)
    _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
    _require(type(pack) is bool, "pack flag")
    plan.verify_bound_documents(repo_root)
    fold_stage0.verify_sealed_roots(repo_root)
    fold_receipts.refuse_sealed_roots(repo_root / plan.FULL_QUERY_ARM_ROOT_RELATIVE[arm])
    (repo_root / plan.FULL_QUERY_ROOT_RELATIVE).mkdir(parents=True, exist_ok=True)
    anchor = verify_pilot_arm_anchor(repo_root, arm)
    bank = carrier_bank.build_fold0_carrier_bank(repo_root)
    gpu_info = stage1_gpu.require_launch(
        gpu_authorized, gpu_index, pack=pack, pack_limit=plan.FULL_QUERY_PACK_LIMIT,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def launch_builder() -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_fold_local_full_query_launch_v1",
            "arm": arm,
            "gpu": gpu_info,
            "anchor": anchor,
            "windows": [[name, window_json(window)] for name, window in plan.FULL_QUERY_WINDOWS],
            "pack": pack,
            "pack_limit": plan.FULL_QUERY_PACK_LIMIT,
            "inference_only": True,
            "target_optimizer_steps": 0,
            "variant": "B3",
            "side_dim": 0,
            "ls4_enabled": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": True,
        }

    def body_publisher(artifact) -> dict[str, str]:
        return _rescore(repo_root, arm=arm, artifact=artifact, bank=bank, anchor=anchor)

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_fold_local_full_query_terminal_v1",
            "status": "COMPLETE",
            "arm": arm,
            "bodies": dict(shas),
            "anchor_reproduced_within_tolerance": True,
            "inference_only": True,
        }

    return fold_receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        attempt_payload={
            "schema": "m1_emg_rsyn3_fold_local_full_query_attempt_v1",
            "arm": arm,
            "gpu_index": gpu_index,
            "pack": pack,
            "fold": 0,
            "pilot_root_relative": plan.STAGE1_ARM_ROOT_RELATIVE[arm],
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        relative=f"full_query_v1/{plan.ARM_SLUG[arm]}",
    )


def _rescore(repo_root: Path, *, arm: str, artifact, bank, anchor) -> dict[str, str]:
    import lightning.pytorch as pl
    import torch

    from . import datamodule as stage1_data
    from . import score as stage1_score

    ensure_streaming_path(repo_root)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(plan.SEED, workers=True)
    lit = load_student(repo_root, repo_root / str(anchor["checkpoint_relative"]), torch)
    device = "cuda:0"
    lit.to(device)
    lit.eval()
    student = lit.student
    session = plan.FOLD0_TARGET_SESSION
    carrier = carrier_bank.carrier_for_arm(bank, session, arm)
    shas: dict[str, str] = {}
    scores: dict[str, dict[str, Mapping]] = {}
    for name, window in plan.FULL_QUERY_WINDOWS:
        data = stage1_data.make_datamodule(repo_root, bank, arm, query_window=window)
        dataset = data.val_heldin_dataset.base
        static = stage1_score.score_static(
            student, dataset, session_id=session, carrier=carrier, device=device,
        )
        cdm = stage1_score.score_cdm_fifo(
            student, dataset, session_id=session, carrier=carrier, device=device,
        )
        relaxation = getattr(data, "version_b_contract_relaxation", None)
        static = {
            **static,
            "window": name,
            "query": window_json(window),
            "version_b_contract_relaxation": relaxation,
        }
        cdm = {
            **cdm,
            "window": name,
            "query": window_json(window),
            "version_b_contract_relaxation": relaxation,
        }
        shas[f"score_static_{name}.json"] = artifact.publish_json(f"score_static_{name}.json", static)
        shas[f"score_cdm_a_{name}.json"] = artifact.publish_json(f"score_cdm_a_{name}.json", cdm)
        scores[name] = {"static": static, "cdm_a": cdm}
        del data
    shas["anchor.json"] = artifact.publish_json("anchor.json", anchor)
    table = build_arm_table(arm, anchor, scores)
    shas["arm_table.json"] = artifact.publish_json("arm_table.json", table)
    _require(table["anchor_reproduction"]["within_tolerance"], "pilot anchor did not reproduce")
    return shas
