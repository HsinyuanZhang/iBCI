#!/usr/bin/env python3
"""Isolated 5070 Ti transport for the sealed RT L-D fold-0 experiment.

This module changes only the resource-release predicate of the reviewed v8
runner.  It deliberately reuses v8/V4 fit and outer-scope validation and the
existing v4 terminalizer.  It neither reads an intermediate score nor owns a
scientific configuration.

The three modes are intentionally distinct:

``--mode dry-run``
    Validate the staged source, sealed hashes, Python environment, dataset,
    teacher, run-root layout, and all composed Hydra arms.  No GPU query, file
    creation, fit, or evaluation occurs.
``--mode preflight``
    Repeat dry-run and additionally require exactly one empty 5070 Ti GPU,
    sufficient disk, and no other RT L-D fit/evaluator/transport process.
    This remains read-only.
``--mode run``
    Repeat preflight, atomically reserve a new append-only run root, execute
    A0/G-Full/G-XLS in the sealed order, validate every fit/evaluation with v8,
    then invoke the existing v4 terminalizer.  A readiness receipt contains no
    accuracy value and is written outside the exact three-arm run root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"
V8_RUNNER = ROOT / "sua_exploration/scripts/rt_ld_device_handoff_v8.py"
V8_PLAN = ROOT / "sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v8.json"
V8_REVIEW = ROOT / "sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_ROOT_REVIEW_v8.json"
V4_TERMINALIZER = ROOT / "sua_exploration/scripts/rt_ld_fold0_terminalize_v4.py"
V4_TERMINAL_PLAN = ROOT / "sua_exploration/results/rt_ld_fold0_terminalize_v4/RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v4.json"
V4_TERMINAL_REVIEW = ROOT / "sua_exploration/results/rt_ld_fold0_terminalize_v4/RT_LD_FOLD0_TERMINALIZER_ROOT_REVIEW_v4.json"
OPTIMIZER_DRIFT = ROOT / "sua_exploration/results/rt_ld_remote_5070_v1/drift/RT_LD_OPTIMIZER_COVERAGE_DRIFT_v1.json"

EXPECTED_SESSION_COUNT = 15
EXPECTED_TEACHER_SHA256 = "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
DEFAULT_MIN_FREE_GIB = 20.0
RESULT_BASE_RELATIVE = Path("sua_exploration/results/rt_ld_remote_5070_v1")

# These are the exact reviewed scientific sources and sealed decision plans.
# Path relocation is permitted; byte drift is not.  V4's own v2-to-current
# drift validator is also called below, so this is an independent transport
# binding rather than a replacement for the scientific validator.
SEALED_BINDINGS: dict[str, str] = {
    "sua_exploration/scripts/rt_ld_device_handoff_v8.py": "17bd2ea7dfc4668b1307729e2f33a5a102ca5b9b72458a086d8d4e6fd9a1c6db",
    "sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v8.json": "8fe55a15fa0e820454a0d0d35dcd40081b2088b41a427fa44f63217370f08df9",
    "sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_ROOT_REVIEW_v8.json": "390773d2fe1c486b3d29f53a641640f1f91d9bb4071ede8c5353d2316e115767",
    "sua_exploration/scripts/rt_ld_device_handoff_v4.py": "095cb634a74283648622326726b125f7a624337d84b4c96be0bf0c9803b88e81",
    "sua_exploration/results/rt_ld_device_handoff_v4/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v4.json": "c88aca88f10c57254e973dfe5e52d4a683bba7f9ac0ec8c6fa361992ff2ae9f8",
    "sua_exploration/results/rt_ld_device_handoff_v4/RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v4.json": "06ea4daf25490dbf6ead812a7b7979486f2b65a0aef38e8c225110de9bc7288a",
    "sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json": "0e3bdb78038f975dce693f5c85d82b48fe8e97e697e645549144990152b09dfd",
    "sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_FOLD0_LAUNCH_PLAN_v2.json": "1e5bbb2e20f4d30290a20e715ebac46751dbf4871e6bddf8212679bc89b26888",
    "sua_exploration/scripts/rt_ld_fold0_terminalize_v4.py": "d8b8bc5abaf50b791b685465fff61abf128690b132971421654d465924d2a709",
    "sua_exploration/results/rt_ld_fold0_terminalize_v4/RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v4.json": "50f0f16b90aa363df63139a24342397c84c995f10f79fe1844c7c0deecb483aa",
    "sua_exploration/results/rt_ld_fold0_terminalize_v4/RT_LD_FOLD0_TERMINALIZER_ROOT_REVIEW_v4.json": "b6ef74de8d5255bfda77951eab18868badc578a98ae582767d754e7be4ea4da5",
    "sua_exploration/results/rt_ld_remote_5070_v1/drift/RT_LD_OPTIMIZER_COVERAGE_DRIFT_v1.json": "6426e52d0cf1a1ad2bde05061d8247d495d42ae9c5a5cb906876c877ee1ca531",
    "streaming_calibration_exp/configs/data/rt_ld_nested_loso_m24.yaml": "35c84595fe89bb757ab9356bf9e802d459dab77a788069c3315d1ba1025ee608",
    "streaming_calibration_exp/configs/experiment/rt_ld_a0_full_m24_fold0_seed42.yaml": "93343caf8e0f72f3bdc20bca02053263d8c3741e7008c109a635eb41832e848a",
    "streaming_calibration_exp/configs/experiment/rt_ld_g_full_m24_fold0_seed42.yaml": "8b676d85cd238a573042aa4f2f1e736e730fd4bbba6f4ab64c8e2b33366f06e1",
    "streaming_calibration_exp/configs/experiment/rt_ld_g_xls_m24_fold0_seed42.yaml": "f4d185df42f1f831a6195e86cbf6d3c144be6a237a6078573442ff865487b36c",
    "streaming_calibration_exp/configs/model/rt_ld_b3s_t4.yaml": "dc234a5dd0481b1277a095d162848ba3dcaeebd7cd96d38b4feff5feeb2d115b",
    "streaming_calibration_exp/src/data/rt_ld_adapter.py": "07c058091e7fbd5f1520cff89cbfa8cdf1446404130a78b96581ddeb25c94a88",
    "streaming_calibration_exp/src/data/rt_ld_datamodule.py": "4e6e1b5d27a0fe881aa236d8f6b4b40d4ba32d0e0308ec7d34db99e0bf5a96ca",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py": "043765595b67aa587dc358f11e4c424120f3bd4b60abee4ecded2a18049cfc4e",
    "streaming_calibration_exp/src/models/components/streaming_spint.py": "141129622ee2c187c053a0139f3926ab1880871d51dc88d0f8b93cc1b734ad9a",
    "streaming_calibration_exp/src/models/rt_ld_streaming_module.py": "e057c6f8b5ae7e197b435d1cf20320853e5d427c3d6cbd72a54352ea4c08f4a9",
    "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py": "316d1ea77cc83715bdee820887d49eb782fabcc2745098ea44c29f3dc649707f",
    "streaming_calibration_exp/tests/test_rt_ld_gain.py": "e10eda4c4da43d55a6af7c96250a01e9f835db4ba1a4466d835a4615f15d89f0",
}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sealed module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V8 = _load(V8_RUNNER, "rt_ld_v8_remote_sealed")
TERMINALIZER = _load(V4_TERMINALIZER, "rt_ld_terminalizer_v4_remote_sealed")
ARM_SPECS: Mapping[str, Mapping[str, str]] = V8.ARM_SPECS
ARM_ORDER: tuple[str, ...] = tuple(V8.ARM_ORDER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_stage_root(stage_root: Path) -> Path:
    supplied = stage_root.expanduser().resolve()
    if supplied != ROOT or stage_root.is_symlink() or not supplied.is_dir():
        raise ValueError("--stage-root must be the real repository root containing this wrapper")
    return supplied


def _validate_bindings(root: Path, bindings: Mapping[str, str] = SEALED_BINDINGS) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in bindings.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"sealed regular file required: {relative}")
        actual = _sha(path)
        if actual != expected:
            raise ValueError(f"sealed hash drift: {relative}")
        observed[relative] = actual
    return observed


def _validate_optimizer_drift(
    receipt: Mapping[str, Any], base_drift: Mapping[str, Any], supplement: Mapping[str, Any]
) -> None:
    """Validate the v2 anchor plus the append-only RT-LD optimizer repair.

    V4 remains immutable.  Its replacement rows are still required; the new
    supplement supersedes only the already-drifted RT-LD test and adds the
    previously unchanged RT-LD Lightning module.  Every other v2 artifact is
    checked byte-for-byte, so this is a strict union rather than a bypass of
    the older drift validator.
    """
    if supplement.get("schema") != "rt_ld_optimizer_coverage_drift_v1":
        raise ValueError("RT-LD optimizer drift schema mismatch")
    if supplement.get("status") != "APPEND_ONLY_REPLACEMENT_BINDING":
        raise ValueError("RT-LD optimizer drift is not append-only")
    if supplement.get("v2_cpu_preflight_receipt_sha256") != _sha(V8.V4.V2_RECEIPT):
        raise ValueError("RT-LD optimizer drift v2 anchor mismatch")
    if supplement.get("v4_code_drift_supplement_sha256") != _sha(V8.V4.DRIFT):
        raise ValueError("RT-LD optimizer drift v4 anchor mismatch")

    base_rows = base_drift.get("artifact_drift")
    rows = supplement.get("artifact_drift")
    expected_new = {
        "streaming_calibration_exp/src/models/rt_ld_streaming_module.py",
        "streaming_calibration_exp/tests/test_rt_ld_gain.py",
    }
    if not isinstance(base_rows, Mapping) or set(base_rows) != set(V8.V4.DRIFT_PATHS):
        raise ValueError("base v4 drift path set mismatch")
    if not isinstance(rows, Mapping) or set(rows) != expected_new:
        raise ValueError("RT-LD optimizer drift path set mismatch")

    for relative, v2_sha in receipt["artifacts"].items():
        actual = _sha(ROOT / relative)
        if relative in rows:
            row = rows[relative]
            if not isinstance(row, Mapping):
                raise ValueError(f"RT-LD optimizer drift row invalid: {relative}")
            if row.get("v2_sha256") != v2_sha or row.get("replacement_sha256") != actual:
                raise ValueError(f"RT-LD optimizer replacement mismatch: {relative}")
            if relative in base_rows and row.get("supersedes_v4_replacement_sha256") != base_rows[relative].get("replacement_sha256"):
                raise ValueError(f"RT-LD optimizer supersession mismatch: {relative}")
        elif relative in base_rows:
            row = base_rows[relative]
            if not isinstance(row, Mapping) or row.get("v2_sha256") != v2_sha or row.get("replacement_sha256") != actual:
                raise ValueError(f"base v4 drift binding mismatch: {relative}")
        elif actual != v2_sha:
            raise ValueError(f"unexpected relative-to-v2 drift: {relative}")

    fresh = supplement.get("fresh_run_contract", {})
    if (
        fresh.get("required_new_run_root_basename") != "fold0_seed42_v3"
        or fresh.get("reuse_completed_a0_from_v2") is not False
        or fresh.get("rerun_all_three_arms_in_frozen_order") is not True
        or fresh.get("failed_v2_run_root_must_not_be_resumed_or_overwritten") is not True
    ):
        raise ValueError("RT-LD optimizer fresh-run contract mismatch")


def _validate_scientific_contract() -> None:
    """Compose v8/V4 identity with the strict append-only optimizer repair."""
    V8._plan_self(_json(V8_PLAN))
    V8.V4._plan_self_check(_json(V8.V4.PLAN))
    receipt = V8.V4._validate_v2_anchor()
    _validate_optimizer_drift(receipt, _json(V8.V4.DRIFT), _json(OPTIMIZER_DRIFT))
    # The transport-aware compose below checks these same fixed identifiers
    # with the pinned interpreter plus the only two permitted path overrides.
    # Do not compose a second time here: on a laptop this doubles a costly
    # framework import without strengthening the contract.
    terminal_plan = TERMINALIZER._check_plan()
    if terminal_plan.get("frozen_gate") != 0.03:
        raise ValueError("v4 terminal gate drift")
    if tuple(ARM_SPECS) != ARM_ORDER or ARM_ORDER != (
        "rt_ld_a0_full_m24_fold0_seed42",
        "rt_ld_g_full_m24_fold0_seed42",
        "rt_ld_g_xls_m24_fold0_seed42",
    ):
        raise ValueError("sealed three-arm order drift")
    v8_review = _json(V8_REVIEW)
    terminal_review = _json(V4_TERMINAL_REVIEW)
    if v8_review.get("status") != "PASS_ROOT_REVIEW__V8_WATCHER_MAY_BE_ARMED_AFTER_TERMINALIZER_V2_REVIEW":
        raise ValueError("v8 root review status drift")
    if terminal_review.get("status") != "PASS_ROOT_REVIEW__V8_WATCHER_AND_V4_TERMINALIZER_MAY_BE_ARMED":
        raise ValueError("v4 terminalizer root review status drift")


def _validate_python_path(python: Path) -> Path:
    if not python.is_absolute() or not python.exists() or not python.is_file():
        raise ValueError("--python must name an existing absolute executable")
    resolved = python.resolve()
    if Path(sys.executable).resolve() != resolved:
        raise ValueError("invoke this wrapper with the exact --python interpreter")
    if not os.access(resolved, os.X_OK):
        raise ValueError("pinned Python is not executable")
    return resolved


def _python_probe(python: Path, *, require_cuda: bool) -> dict[str, Any]:
    code = (
        "import json,sys,torch,lightning,hydra,omegaconf;"
        "print(json.dumps({'executable':sys.executable,'torch':torch.__version__,"
        "'lightning':lightning.__version__,'cuda_available':torch.cuda.is_available(),"
        "'cuda_count':torch.cuda.device_count(),"
        "'cuda_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    result = subprocess.run(
        [str(python), "-c", code], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"pinned Python import probe failed: {result.stderr[-1000:]}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("pinned Python probe did not return JSON") from error
    if Path(str(payload.get("executable", ""))).resolve() != python.resolve():
        raise ValueError("pinned Python subprocess identity drift")
    for package in ("torch", "lightning"):
        if not isinstance(payload.get(package), str) or not payload[package]:
            raise ValueError(f"missing pinned Python package: {package}")
    if require_cuda:
        if payload.get("cuda_available") is not True or payload.get("cuda_count") != 1:
            raise ValueError("pinned Python must expose exactly one CUDA device")
        if "5070 Ti" not in str(payload.get("cuda_name", "")):
            raise ValueError("remote transport requires the 5070 Ti GPU")
    return payload


def _validate_data_dir(data_dir: Path) -> dict[str, Any]:
    directory = data_dir.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("RT data directory is missing")
    sessions = sorted(directory.glob("sub-C_ses-RT-*_behavior+ecephys.nwb"))
    if len(sessions) != EXPECTED_SESSION_COUNT:
        raise ValueError(f"RT dataset requires exactly {EXPECTED_SESSION_COUNT} sessions")
    if any(not path.is_file() or path.stat().st_size <= 0 for path in sessions):
        raise ValueError("RT dataset contains an empty or non-regular session")
    resolved = [path.resolve() for path in sessions]
    if len(set(resolved)) != EXPECTED_SESSION_COUNT:
        raise ValueError("RT dataset contains duplicate resolved sessions")
    return {
        "path": str(directory),
        "session_count": len(sessions),
        "session_names": [path.name for path in sessions],
        "total_bytes": sum(path.stat().st_size for path in sessions),
    }


def _validate_teacher(teacher: Path) -> dict[str, Any]:
    checkpoint = teacher.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("teacher checkpoint is missing or empty")
    digest = _sha(checkpoint)
    if digest != EXPECTED_TEACHER_SHA256:
        raise ValueError("teacher checkpoint SHA does not match the sealed FP32 teacher")
    return {"path": str(checkpoint), "sha256": digest, "bytes": checkpoint.stat().st_size}


def _validate_run_root(stage_root: Path, run_root: Path) -> tuple[Path, Path]:
    root = run_root.expanduser().resolve()
    base = (stage_root / RESULT_BASE_RELATIVE / "runs").resolve()
    if root.parent != base or run_root.is_symlink():
        raise ValueError(f"run root must be one fresh direct child of {base}")
    if root.exists():
        raise FileExistsError("append-only run root already exists")
    if not root.name or root.name in {".", ".."}:
        raise ValueError("nonempty run identifier required")
    return root, base


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise ValueError("no existing ancestor for disk probe")
        current = current.parent
    return current


def _validate_disk(path: Path, minimum_gib: float) -> dict[str, Any]:
    if minimum_gib <= 0:
        raise ValueError("disk floor must be positive")
    usage = shutil.disk_usage(_nearest_existing(path))
    minimum = int(minimum_gib * (1024 ** 3))
    if usage.free < minimum:
        raise ValueError(f"disk floor failed: need {minimum_gib:g} GiB free")
    return {"free_bytes": usage.free, "minimum_free_bytes": minimum}


def _parse_gpu_inventory(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise ValueError("unexpected nvidia-smi inventory row")
        rows.append({"index": parts[0], "uuid": parts[1], "name": parts[2]})
    return rows


def _gpu_inventory() -> list[dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi inventory failed: {result.stderr[-1000:]}")
    rows = _parse_gpu_inventory(result.stdout)
    if len(rows) != 1 or rows[0]["index"] != "0" or "5070 Ti" not in rows[0]["name"]:
        raise ValueError("remote stage requires exactly one physical 5070 Ti at index 0")
    return rows


def _parse_compute_pids(stdout: str) -> list[int]:
    rows: list[int] = []
    for line in stdout.splitlines():
        value = line.strip()
        if not value or value == "No running processes found":
            continue
        try:
            rows.append(int(value))
        except ValueError as error:
            raise ValueError("unexpected nvidia-smi compute PID") from error
    return rows


def _compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi compute probe failed: {result.stderr[-1000:]}")
    return _parse_compute_pids(result.stdout)


def _ancestor_pids(pid: int | None = None) -> set[int]:
    current = os.getpid() if pid is None else pid
    result: set[int] = set()
    while current > 1 and current not in result:
        result.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
    return result


def _ld_processes_from_rows(rows: Iterable[tuple[int, str]], ignored: set[int]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for pid, command in rows:
        if pid in ignored:
            continue
        is_fit = "src/train.py" in command and "experiment=rt_ld_" in command
        is_eval = "rt_clean_nested_loso_eval.py" in command
        is_transport = "rt_ld_remote_5070_v1.py" in command and "--mode run" in command
        if is_fit or is_eval or is_transport:
            found.append({"pid": pid, "command": command})
    return found


def _existing_ld_processes() -> list[dict[str, Any]]:
    rows: list[tuple[int, str]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            rows.append((int(proc.name), (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")))
        except (OSError, ValueError):
            continue
    return _ld_processes_from_rows(rows, _ancestor_pids())


def _runtime_release(stability_seconds: float) -> dict[str, Any]:
    if stability_seconds < 0:
        raise ValueError("GPU stability interval cannot be negative")
    inventory = _gpu_inventory()
    first = _compute_pids()
    conflicts_first = _existing_ld_processes()
    time.sleep(stability_seconds)
    second = _compute_pids()
    conflicts_second = _existing_ld_processes()
    if first or second:
        raise RuntimeError("5070 Ti is not empty in both release probes")
    if conflicts_first or conflicts_second:
        raise RuntimeError("an RT L-D fit/evaluator/transport process already exists")
    return {"inventory": inventory, "compute_pids_first": first, "compute_pids_second": second}


def _transport_overrides(data_dir: Path, teacher: Path) -> list[str]:
    return [f"data.data_dir={data_dir.resolve()}", f"paths.teacher_ckpt_path={teacher.resolve()}"]


def _train_command(python: Path, experiment: str, arm_dir: Path, data_dir: Path, teacher: Path) -> list[str]:
    if experiment not in ARM_SPECS:
        raise ValueError("unknown RT L-D arm")
    return [
        str(python), "src/train.py", f"experiment={experiment}", "seed=42",
        f"hydra.run.dir={arm_dir.resolve()}", *_transport_overrides(data_dir, teacher),
    ]


def _eval_command(python: Path, arm_dir: Path, checkpoint: Path) -> list[str]:
    arm = arm_dir.resolve()
    return [
        str(python), "src/rt_clean_nested_loso_eval.py",
        "--config", str(arm / ".hydra/config.yaml"),
        "--checkpoint", str(checkpoint.resolve()),
        "--split-manifest", str(arm / "split_manifest.json"),
        "--selection-receipt", str(arm / "rt_nested_selection_receipt.json"),
        "--output", str(arm / "rt_ld_outer_eval.json"),
        "--device", "cuda:0",
    ]


def _validate_transport_composition(python: Path, data_dir: Path, teacher: Path) -> None:
    for experiment in ARM_ORDER:
        result = subprocess.run(
            [str(python), "src/train.py", f"experiment={experiment}", "seed=42",
             *_transport_overrides(data_dir, teacher), "--cfg", "job", "--resolve"],
            cwd=STREAMING, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"remote Hydra compose failed for {experiment}: {result.stderr[-1000:]}")
        cfg = yaml.safe_load(result.stdout)
        if not isinstance(cfg, Mapping):
            raise ValueError("remote Hydra compose did not return a mapping")
        spec = ARM_SPECS[experiment]
        if cfg.get("run_id") != spec["run_id"] or cfg.get("model", {}).get("rt_ld_arm") != spec["model_rt_ld_arm"]:
            raise ValueError("remote composed arm identity drift")
        if cfg.get("data", {}).get("rt_ld_gain_source") != spec["data_rt_ld_gain_source"]:
            raise ValueError("remote composed gain-source drift")
        if int(cfg.get("seed", -1)) != 42 or int(cfg.get("data", {}).get("outer_loso_fold", -1)) != 0:
            raise ValueError("remote composed seed/fold drift")
        if Path(str(cfg.get("data", {}).get("data_dir", ""))).resolve() != data_dir.resolve():
            raise ValueError("remote composed dataset path drift")
        if Path(str(cfg.get("model", {}).get("teacher_ckpt_path", ""))).resolve() != teacher.resolve():
            raise ValueError("remote composed teacher path drift")
        V8.V4._callback_contract(cfg)


def _static_preflight(stage_root: Path, python: Path, data_dir: Path, teacher: Path, run_root: Path) -> dict[str, Any]:
    stage = _require_stage_root(stage_root)
    bindings = _validate_bindings(stage)
    _validate_scientific_contract()
    pinned = _validate_python_path(python)
    python_info = _python_probe(pinned, require_cuda=False)
    data_info = _validate_data_dir(data_dir)
    teacher_info = _validate_teacher(teacher)
    resolved_run_root, _ = _validate_run_root(stage, run_root)
    _validate_transport_composition(pinned, Path(data_info["path"]), Path(teacher_info["path"]))
    return {
        "schema": "rt_ld_remote_5070_static_preflight_v1",
        "status": "STATIC_PASS_NO_GPU_LAUNCH",
        "stage_root": str(stage),
        "run_root": str(resolved_run_root),
        "python": python_info,
        "dataset": data_info,
        "teacher": teacher_info,
        "sealed_bindings": bindings,
        "arm_order": list(ARM_ORDER),
        "seed": 42,
        "fold": 0,
        "frozen_gates": {"g_full_minus_a0": 0.03, "g_full_minus_g_xls": 0.03},
        "formal_heldout_opened": False,
        "gpu_launched": False,
    }


def preflight(stage_root: Path, python: Path, data_dir: Path, teacher: Path, run_root: Path,
              *, minimum_free_gib: float, stability_seconds: float) -> dict[str, Any]:
    result = _static_preflight(stage_root, python, data_dir, teacher, run_root)
    result["python"] = _python_probe(python.resolve(), require_cuda=True)
    result["disk"] = _validate_disk(Path(result["run_root"]), minimum_free_gib)
    result["gpu_release"] = _runtime_release(stability_seconds)
    result["schema"] = "rt_ld_remote_5070_runtime_preflight_v1"
    result["status"] = "RUNTIME_PASS_READY_NO_GPU_LAUNCH"
    return result


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"append-only receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.chmod(path, 0o444)


def execute(stage_root: Path, python: Path, data_dir: Path, teacher: Path, run_root: Path,
            *, minimum_free_gib: float, stability_seconds: float) -> Path:
    ready = preflight(
        stage_root, python, data_dir, teacher, run_root,
        minimum_free_gib=minimum_free_gib, stability_seconds=stability_seconds,
    )
    stage = Path(ready["stage_root"])
    root = Path(ready["run_root"])
    result_base = stage / RESULT_BASE_RELATIVE
    readiness = result_base / "readiness" / f"{root.name}_READINESS_v1.json"
    terminal = result_base / "terminal" / f"{root.name}_RT_LD_FOLD0_TERMINAL_v4.json"
    if readiness.exists() or terminal.exists() or readiness.is_symlink() or terminal.is_symlink():
        raise FileExistsError("append-only readiness/terminal destination already exists")

    # O_EXCL-equivalent directory reservation.  A failed run is not resumed or
    # overwritten by this transport; the incomplete root remains evidence.
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    ready = dict(ready)
    ready.update({
        "schema": "rt_ld_remote_5070_readiness_receipt_v1",
        "status": "APPEND_ONLY_RUN_ROOT_RESERVED",
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": _sha(Path(__file__).resolve()),
        "readiness_path": str(readiness.resolve()),
        "terminal_output": str(terminal.resolve()),
        "accuracy_values_present": False,
        "local_ci_latch_used": False,
        "resource_release_replacement_only": True,
    })
    _write_immutable_json(readiness, ready)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    pinned = python.resolve()
    resolved_data = data_dir.resolve()
    resolved_teacher = teacher.resolve()
    for experiment in ARM_ORDER:
        # Exact source hashes and runtime exclusion are rechecked immediately
        # before every arm.  No result file is opened by this predicate.
        _validate_bindings(stage)
        _runtime_release(stability_seconds)
        spec = ARM_SPECS[experiment]
        arm_dir = (root / spec["directory"]).resolve()
        if arm_dir.parent != root:
            raise ValueError("arm directory escapes append-only run root")
        train = subprocess.run(
            _train_command(pinned, experiment, arm_dir, resolved_data, resolved_teacher),
            cwd=STREAMING, env=env, check=False,
        )
        if train.returncode:
            raise RuntimeError(f"remote training failed: {experiment}")
        checkpoint = V8.V4._validate_fit_artifacts(arm_dir, experiment)
        evaluated = subprocess.run(
            _eval_command(pinned, arm_dir, Path(checkpoint)),
            cwd=STREAMING, env=env, check=False,
        )
        outer = arm_dir / "rt_ld_outer_eval.json"
        if evaluated.returncode or not outer.is_file():
            raise RuntimeError(f"remote outer evaluation failed: {experiment}")
        V8._outer_scope_v8(arm_dir, experiment)

    # Existing v4 owns all score reads, pairing, inclusive +0.03 gates, and
    # the immutable PASS/STOP receipt.  The transport does not inspect it.
    terminal_result = subprocess.run(
        [str(pinned), str(V4_TERMINALIZER), "--run-root", str(root), "--write", str(terminal)],
        cwd=ROOT, env=env, check=False,
    )
    if terminal_result.returncode or not terminal.is_file() or terminal.stat().st_mode & 0o777 != 0o444:
        raise RuntimeError("existing v4 terminalization failed")
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "preflight", "run"), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--stability-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    common = (args.stage_root, args.python, args.data_dir, args.teacher_checkpoint, args.run_root)
    if args.mode == "dry-run":
        result = _static_preflight(*common)
        print(json.dumps(result, sort_keys=True))
        return
    if args.mode == "preflight":
        result = preflight(
            *common, minimum_free_gib=args.min_free_gib,
            stability_seconds=args.stability_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return
    terminal = execute(
        *common, minimum_free_gib=args.min_free_gib,
        stability_seconds=args.stability_seconds,
    )
    print(json.dumps({"status": "TERMINALIZED_BY_EXISTING_V4", "terminal_receipt": str(terminal)}, sort_keys=True))


if __name__ == "__main__":
    main()
