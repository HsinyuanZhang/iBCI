"""CPU contracts for fold-local EMG-rSyn3 full-query. No live GPU in this module."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan


ROOT = Path(__file__).resolve().parents[2]
FULL_QUERY_CLI = ROOT / "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_full_query.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
_CLI_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(ROOT),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_pilot_arm(
    repo_root: Path,
    *,
    arm: str = "Z-Fix",
    ckpt: bytes = b"ckpt-bytes",
    query: list[int] | None = None,
    status: str = "COMPLETE",
    static: dict[str, object] | None = None,
    cdm_a: dict[str, object] | None = None,
) -> dict[str, str]:
    root = repo_root / plan.STAGE1_ARM_ROOT_RELATIVE[arm]
    root.mkdir(parents=True)
    static = static or {
        "governing_r2": 0.123,
        "prediction_sha256": "aa" * 32,
        "n_windows": 26517,
    }
    cdm_a = cdm_a or {
        "governing_r2": 0.456,
        "prediction_sha256": "bb" * 32,
        "n_windows": 26517,
    }
    bodies = {
        "epoch_011.pt": ckpt,
        "score_static.json": json.dumps(static).encode("utf-8"),
        "score_cdm_a.json": json.dumps(cdm_a).encode("utf-8"),
        "arm_table.json": json.dumps({"arm": arm, "query": list(query or [10, 210])}).encode("utf-8"),
    }
    shas: dict[str, str] = {}
    for name in plan.FULL_QUERY_PILOT_BODIES:
        payload = bodies[name]
        (root / name).write_bytes(payload)
        digest = _sha256(payload)
        (root / f"{name}.sha256").write_text(f"{digest}  {name}\n")
        shas[name] = digest
    terminal = json.dumps({"status": status, "arm": arm, "bodies": dict(shas)}).encode("utf-8")
    (root / "terminal.json").write_bytes(terminal)
    shas["terminal.json"] = _sha256(terminal)
    return shas


def test_full_query_plan_literals() -> None:
    for _name, window in plan.FULL_QUERY_WINDOWS:
        assert window in plan.FULL_QUERY_ALLOWED_WINDOWS
    windows = dict(plan.FULL_QUERY_WINDOWS)
    assert windows[plan.FULL_QUERY_ANCHOR_WINDOW] == (plan.QUERY_START, plan.QUERY_STOP_EXCLUSIVE)
    for arm, relative in plan.FULL_QUERY_ARM_ROOT_RELATIVE.items():
        assert relative.startswith(f"{plan.FULL_QUERY_ROOT_RELATIVE}/")
        assert relative == f"{plan.FULL_QUERY_ROOT_RELATIVE}/{plan.ARM_SLUG[arm]}"
    assert set(plan.FULL_QUERY_ARM_ROOT_RELATIVE.values()).isdisjoint(
        set(plan.STAGE1_ARM_ROOT_RELATIVE.values()),
    )
    assert plan.FULL_QUERY_OWN_TOKEN in plan.OWN_TOKENS
    assert plan.OWN_TOKEN in plan.OWN_TOKENS
    assert plan.FULL_QUERY_PACK_LIMIT == 3
    assert "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_full_query.py" in plan.OWNED_PATHS
    assert "tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_full_query.py" in plan.OWNED_PATHS
    payload = plan.dry_cli_payload()
    assert payload["prospective_roots"]["full_query"] == plan.FULL_QUERY_ROOT_RELATIVE
    assert len(payload["full_query_windows"]) == 3
    assert payload["full_query_windows"] == [
        [name, [start, end]] for name, (start, end) in plan.FULL_QUERY_WINDOWS
    ]


def test_full_query_own_token_is_recognized_and_pack_limit_three() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import gpu as stage1_gpu

    own_full = "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_full_query.py --arm Z-Fix"
    own_stage1 = "tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage1.py --arm S-Fix"
    rows = (
        f"0, {plan.GPU0_UUID}, 453, 0\n"
        f"1, {plan.GPU1_UUID}, 987, 54\n"
    )

    def two_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return (
                f"{plan.GPU1_UUID}, 11, python, 958\n"
                f"{plan.GPU1_UUID}, 12, python, 958\n"
            )
        return rows

    receipt = stage1_gpu.assert_target_gpu_launchable(
        1, pack=True, cmdline_runner=two_own,
        pid_cmdline=lambda pid: own_full,
        pack_limit=3,
    )
    assert receipt["own_occupants"] == 2
    assert receipt["pack_limit"] == 3

    def three_own(arguments):
        query = " ".join(arguments)
        if "compute-apps" in query:
            return (
                f"{plan.GPU1_UUID}, 11, python, 958\n"
                f"{plan.GPU1_UUID}, 12, python, 958\n"
                f"{plan.GPU1_UUID}, 13, python, 958\n"
            )
        return rows

    with pytest.raises(stage1_gpu.GpuError, match="pack limit 3"):
        stage1_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=three_own,
            pid_cmdline=lambda pid: own_full,
            pack_limit=3,
        )

    with pytest.raises(stage1_gpu.GpuError, match="pack limit 2"):
        stage1_gpu.assert_target_gpu_launchable(
            1, pack=True, cmdline_runner=two_own,
            pid_cmdline=lambda pid: own_full,
        )

    def mixed_cmdline(pid: str) -> str:
        if pid == "11":
            return own_stage1
        return own_full

    mixed = stage1_gpu.assert_target_gpu_launchable(
        1, pack=True, cmdline_runner=two_own,
        pid_cmdline=mixed_cmdline,
        pack_limit=3,
    )
    assert mixed["own_occupants"] == 2


def test_full_query_datamodule_rejects_undeclared_window() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import datamodule as stage1_data

    with pytest.raises(stage1_data.DatamoduleError, match="not pre-declared"):
        stage1_data.make_datamodule(Path("/nonexistent"), {}, "Z-Fix", query_window=(0, 100))


def test_full_query_anchor_verifies_and_rejects_tampering(tmp_path) -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.full_query import (
        FullQueryError,
        verify_pilot_arm_anchor,
    )

    static = {
        "governing_r2": 0.123,
        "prediction_sha256": "aa" * 32,
        "n_windows": 26517,
    }
    cdm_a = {
        "governing_r2": 0.456,
        "prediction_sha256": "bb" * 32,
        "n_windows": 26517,
    }
    good_root = tmp_path / "ok"
    shas = _write_pilot_arm(good_root, static=static, cdm_a=cdm_a)
    anchor = verify_pilot_arm_anchor(good_root, "Z-Fix")
    assert anchor["schema"] == "m1_emg_rsyn3_fold_local_full_query_anchor_v1"
    assert anchor["arm"] == "Z-Fix"
    assert anchor["pilot_root_relative"] == plan.STAGE1_ARM_ROOT_RELATIVE["Z-Fix"]
    assert anchor["checkpoint_relative"] == f"{plan.STAGE1_ARM_ROOT_RELATIVE['Z-Fix']}/epoch_011.pt"
    assert anchor["checkpoint_sha256"] == shas["epoch_011.pt"]
    assert anchor["terminal_sha256"] == shas["terminal.json"]
    assert anchor["arm_table_sha256"] == shas["arm_table.json"]
    assert anchor["score_static_sha256"] == shas["score_static.json"]
    assert anchor["score_cdm_a_sha256"] == shas["score_cdm_a.json"]
    assert anchor["sealed_query"] == [10, 210]
    assert anchor["sealed_static_r2"] == static["governing_r2"]
    assert anchor["sealed_cdm_a_r2"] == cdm_a["governing_r2"]
    assert anchor["sealed_static_prediction_sha256"] == static["prediction_sha256"]
    assert anchor["sealed_cdm_a_prediction_sha256"] == cdm_a["prediction_sha256"]
    assert anchor["sealed_n_windows"] == static["n_windows"]

    flip_root = tmp_path / "flip"
    _write_pilot_arm(flip_root, static=static, cdm_a=cdm_a)
    ckpt_path = flip_root / plan.STAGE1_ARM_ROOT_RELATIVE["Z-Fix"] / "epoch_011.pt"
    mutated = bytearray(ckpt_path.read_bytes())
    mutated[0] ^= 0xFF
    ckpt_path.write_bytes(bytes(mutated))
    with pytest.raises(FullQueryError):
        verify_pilot_arm_anchor(flip_root, "Z-Fix")

    query_root = tmp_path / "query"
    _write_pilot_arm(query_root, static=static, cdm_a=cdm_a, query=[10, 400])
    with pytest.raises(FullQueryError):
        verify_pilot_arm_anchor(query_root, "Z-Fix")

    failed_root = tmp_path / "failed"
    _write_pilot_arm(failed_root, static=static, cdm_a=cdm_a, status="FAILED")
    with pytest.raises(FullQueryError):
        verify_pilot_arm_anchor(failed_root, "Z-Fix")


def test_full_query_arm_table_anchor_and_additivity() -> None:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.full_query import (
        build_arm_table,
        window_json,
    )

    sealed_static = 0.4
    sealed_cdm = 0.5
    static_sha = "ss" * 32
    cdm_sha = "cc" * 32
    counts = {"q10_210": 100, "q210_end": 50, "q10_end": 150}
    anchor = {
        "checkpoint_sha256": "dd" * 32,
        "sealed_query": [10, 210],
        "sealed_static_r2": sealed_static,
        "sealed_cdm_a_r2": sealed_cdm,
        "sealed_static_prediction_sha256": static_sha,
        "sealed_cdm_a_prediction_sha256": cdm_sha,
        "sealed_n_windows": counts["q10_210"],
    }

    def scores_for(*, static_r2: float, n_end: int) -> dict[str, dict[str, dict[str, object]]]:
        out: dict[str, dict[str, dict[str, object]]] = {}
        for name, _window in plan.FULL_QUERY_WINDOWS:
            n_windows = n_end if name == "q10_end" else counts[name]
            out[name] = {
                "static": {
                    "governing_r2": static_r2 if name == plan.FULL_QUERY_ANCHOR_WINDOW else 0.1,
                    "n_windows": n_windows,
                    "prediction_sha256": static_sha,
                },
                "cdm_a": {
                    "governing_r2": sealed_cdm if name == plan.FULL_QUERY_ANCHOR_WINDOW else 0.2,
                    "n_windows": n_windows,
                    "prediction_sha256": cdm_sha,
                },
            }
        return out

    matching = build_arm_table("Z-Fix", anchor, scores_for(static_r2=sealed_static, n_end=150))
    assert [row["window"] for row in matching["rows"]] == [name for name, _ in plan.FULL_QUERY_WINDOWS]
    for row, (name, window) in zip(matching["rows"], plan.FULL_QUERY_WINDOWS, strict=True):
        assert row["query"] == window_json(window)
        assert row["window"] == name
    assert matching["anchor_reproduction"]["within_tolerance"] is True
    assert matching["anchor_reproduction"]["abs_diff_static"] == 0.0
    assert matching["n_windows_additivity"]["q10_210_plus_q210_end"] == 150
    assert matching["n_windows_additivity"]["q10_end"] == 150
    assert matching["n_windows_additivity"]["equal"] is True
    assert matching["formal_benchmark_verdict"] is False
    assert matching["official_extent_window"] == "q10_end"
    assert matching["official_extent_window"] == plan.FULL_QUERY_OFFICIAL_EXTENT_WINDOW

    drifted = build_arm_table("Z-Fix", anchor, scores_for(static_r2=sealed_static + 1e-3, n_end=150))
    assert drifted["anchor_reproduction"]["within_tolerance"] is False
    assert drifted["anchor_reproduction"]["abs_diff_static"] == pytest.approx(1e-3)

    unequal = build_arm_table("Z-Fix", anchor, scores_for(static_r2=sealed_static, n_end=149))
    assert unequal["n_windows_additivity"]["q10_210_plus_q210_end"] == 150
    assert unequal["n_windows_additivity"]["q10_end"] == 149
    assert unequal["n_windows_additivity"]["equal"] is False


def test_full_query_public_cli_is_dry_and_execute_gpu_errors() -> None:
    dry = subprocess.run(
        [PYTHON, str(FULL_QUERY_CLI)],
        check=True, capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    payload = json.loads(dry.stdout)
    assert payload["cli"] == "run_m1_emg_rsyn3_fold_local_full_query.py"
    assert payload["public_gpu_capability"] is False
    gpu = subprocess.run(
        [PYTHON, str(FULL_QUERY_CLI), "--execute-gpu"],
        capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    assert gpu.returncode != 0
    assert "cannot mint a GPU capability" in (gpu.stderr + gpu.stdout)
    execute = subprocess.run(
        [PYTHON, str(FULL_QUERY_CLI), "--execute", "--gpu-index", "1", "--arm", "Z-Fix"],
        capture_output=True, text=True, cwd=str(ROOT),
        env=_CLI_ENV,
    )
    assert execute.returncode != 0
