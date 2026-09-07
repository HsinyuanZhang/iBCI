"""Isolated, no-GPU launcher and postflight for RT annex v2.

The shared ``run_rt_clean_nested_loso.py`` predates the annex and does not
expose ``afc4_mb4`` in its CLI.  This module therefore uses the existing
Hydra ``src/train.py`` entry point directly, with an explicit side-feature
override, and binds the shared runner/evaluator/config source hashes in every
readiness receipt.  It never edits or monkeypatches the shared entry points.
If a required shared API is absent, all builders fail closed.

All commands are returned as argument lists.  The CLI prints them by default;
execution requires both ``--execute`` and ``RT_ANNEX_V2_ENABLE_EXECUTION=1``.
The tests and readiness path never read NWB files, instantiate a Trainer, or
touch CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from . import spec


ANNEX_ID = "rt_seed_robustness_annex_v2"
ROOT = spec.ROOT
V2_ROOT = Path(__file__).resolve().parent
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
DEFAULT_PYTHON = Path("/home/xinyuan/miniconda3/envs/spint/bin/python")
DEFAULT_ARTIFACT_ROOT = Path(
    "/home/xinyuan/rt_seed_robustness_annex_v2_1_artifacts_20260810"
)
EXECUTION_ENABLE_ENV = "RT_ANNEX_V2_ENABLE_EXECUTION"
EXECUTION_AUTHORIZATION_ENV = "RT_ANNEX_V2_GPU_AUTHORIZATION"
EXECUTION_AUTHORIZATION_VALUE = "I_AUTHORIZE_RT_SEED_43_44_GPU_LAUNCH"
ALLOWED_ARMS = (spec.FULL_ARM, spec.MB4_ARM)
EXPECTED_FOLDS = tuple(spec.SELECTED_FOLDS)
EXPECTED_SEEDS = tuple(spec.EXPECTED_SEEDS)
EXPECTED_STATUS = spec.EXPECTED_STATUS

TRAIN_ENTRY_REL = Path("streaming_calibration_exp/src/train.py")
RUNNER_REL = spec.FULL_SOURCE_REL
EVAL_REL = spec.EVAL_SOURCE_REL
EXPERIMENT_REL = spec.EXPERIMENT_CONFIG_REL
DATA_REL = spec.DATA_CONFIG_REL
MODEL_CONFIG_REL = spec.MODEL_CONFIG_REL
MODEL_SOURCE_REL = spec.MODEL_REL
DATA_MODULE_REL = spec.DATA_MODULE_REL
SELECTION_CALLBACK_REL = spec.SELECTION_CALLBACK_REL
TRAINER_REL = spec.TRAINER_CONFIG_REL
INSTANTIATOR_REL = Path("streaming_calibration_exp/src/utils/instantiators.py")
STREAMING_MODULE_REL = Path("streaming_calibration_exp/src/models/streaming_calibration_module.py")
RUNTIME_REL = Path("sua_exploration/rt_seed_robustness_annex_v2/runtime.py")
LAUNCHER_REL = Path("sua_exploration/rt_seed_robustness_annex_v2/launcher.py")
SUPERVISOR_REL = Path("sua_exploration/rt_seed_robustness_annex_v2/supervisor.py")
SPEC_REL = Path("sua_exploration/rt_seed_robustness_annex_v2/spec.py")

PROVENANCE_HASH_FIELDS = (
    "implementation_snapshot_sha256",
    "train_entry_sha256",
    "runner_source_sha256",
    "evaluator_source_sha256",
    "experiment_config_sha256",
    "data_config_sha256",
    "model_config_sha256",
    "selection_callback_sha256",
    "runtime_hook_sha256",
)


class LaunchReadinessError(RuntimeError):
    """Raised when a static launch or receipt contract is not satisfied."""


def _require_pinned_python(executable: str | Path) -> Path:
    """Return the sole authorized interpreter path, without accepting aliases."""

    requested = Path(executable).expanduser()
    if not requested.is_absolute():
        raise LaunchReadinessError("python_executable must be the absolute pinned path")
    if requested != DEFAULT_PYTHON:
        raise LaunchReadinessError(
            f"python_executable must be exactly {DEFAULT_PYTHON}; got {requested}"
        )
    if not requested.is_file() or not os.access(requested, os.X_OK):
        raise LaunchReadinessError(f"pinned interpreter is not executable: {requested}")
    return requested


def validate_artifact_root(
    artifact_root: str | Path, *, require_fresh: bool = False
) -> Path:
    """Require an absolute, external, narrowly scoped artifact root."""

    requested = Path(artifact_root).expanduser()
    if not requested.is_absolute():
        raise LaunchReadinessError("artifact_root must be absolute")
    root = requested.resolve(strict=False)
    repository = ROOT.resolve()
    if root == repository or repository in root.parents:
        raise LaunchReadinessError("artifact_root must be outside the repository source tree")
    broad_roots = {Path("/"), Path.home().resolve(), Path("/home").resolve()}
    if root in broad_roots:
        raise LaunchReadinessError("artifact_root is too broad")
    if require_fresh and root.exists():
        raise LaunchReadinessError(f"artifact_root must be fresh and absent: {root}")
    return root


def require_execution_gate() -> None:
    """Fail before any subprocess unless both explicit GPU gates are present."""

    if os.environ.get(EXECUTION_ENABLE_ENV) != "1":
        raise LaunchReadinessError(
            f"execution disabled; set {EXECUTION_ENABLE_ENV}=1 and pass --execute"
        )
    if os.environ.get(EXECUTION_AUTHORIZATION_ENV) != EXECUTION_AUTHORIZATION_VALUE:
        raise LaunchReadinessError(
            f"GPU authorization missing; set {EXECUTION_AUTHORIZATION_ENV}="
            f"{EXECUTION_AUTHORIZATION_VALUE} only after authorization"
        )


def execution_environment(*, cuda_visible_devices: str | None = None) -> dict[str, str]:
    """Return the explicit import environment used by every launch."""

    env = dict(os.environ)
    roots = [str(ROOT.resolve()), str(STREAMING_ROOT.resolve())]
    # Do not inherit an ambient PYTHONPATH: the source/import graph is part of
    # the sealed snapshot and must resolve only through these two exact roots.
    env["PYTHONPATH"] = os.pathsep.join(roots)
    # The pinned interpreter has its own torch build.  Never permit Python's
    # per-user site directory to shadow that conda environment.
    env["PYTHONNOUSERSITE"] = "1"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return env


def render_execution_wrapper(
    command: Sequence[str], *, cuda_visible_devices: str | None = None
) -> str:
    """Render a reproducible shell wrapper with cwd and PYTHONPATH."""

    env = execution_environment(cuda_visible_devices=cuda_visible_devices)
    cuda = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
        if cuda_visible_devices is not None
        else ""
    )
    return (
        f"cd {shlex.quote(str(STREAMING_ROOT.resolve()))} && "
        f"PYTHONNOUSERSITE=1 PYTHONPATH={shlex.quote(env['PYTHONPATH'])} {cuda}"
        + " ".join(shlex.quote(str(part)) for part in command)
    )


@dataclass(frozen=True)
class Cell:
    """One paired source-fit/outer-evaluation cell."""

    seed: int
    fold: int
    arm: str

    def __post_init__(self) -> None:
        if int(self.seed) not in EXPECTED_SEEDS:
            raise LaunchReadinessError(f"unexpected annex seed: {self.seed!r}")
        if int(self.fold) not in EXPECTED_FOLDS:
            raise LaunchReadinessError(f"fold is not in selected annex subset: {self.fold!r}")
        if self.arm not in ALLOWED_ARMS:
            raise LaunchReadinessError(
                f"arm must be exactly one of {ALLOWED_ARMS}, got {self.arm!r}"
            )

    @property
    def run_id(self) -> str:
        return f"rt_seed_robustness_v2_{self.arm}_f{int(self.fold)}_s{int(self.seed)}"

    @property
    def key(self) -> str:
        return f"s{int(self.seed)}_f{int(self.fold)}_{self.arm}"


def _read_text(rel: Path) -> str:
    path = ROOT / rel
    if not path.is_file():
        raise LaunchReadinessError(f"missing static source/config binding: {path}")
    return path.read_text(encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise LaunchReadinessError(f"cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchReadinessError(f"cannot read JSON receipt {path}: {error}") from error
    if not isinstance(value, dict):
        raise LaunchReadinessError(f"receipt is not a JSON object: {path}")
    return value


def _write_exclusive(
    path: Path, payload: Mapping[str, Any], *, readonly: bool = False
) -> None:
    """Write a receipt once; never overwrite a completed cell."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LaunchReadinessError(f"refusing to overwrite annex receipt: {path}")
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if readonly:
        path.chmod(0o444)


@lru_cache(maxsize=8)
def interpreter_metadata(executable: str = str(DEFAULT_PYTHON)) -> dict[str, Any]:
    """Verify the pinned interpreter and collect import/CUDA metadata."""

    requested = _require_pinned_python(executable)
    path = requested.resolve()
    # The readiness probe is deliberately CPU-only.  It imports torch to bind
    # the installed package but never calls a CUDA runtime query or allocator.
    env = execution_environment(cuda_visible_devices="")
    probe = r'''
import json, platform, os, site, sys
import torch, lightning, hydra, omegaconf
from sua_exploration.rt_seed_robustness_annex_v2.runtime import AnnexInitialStateCallback
import src
print(json.dumps({
  "python_version": sys.version,
  "python_implementation": platform.python_implementation(),
  "sys_executable": sys.executable,
  "sys_prefix": sys.prefix,
  "base_prefix": sys.base_prefix,
  "user_site": site.getusersitepackages(),
  "user_site_enabled": bool(site.ENABLE_USER_SITE),
  "sys_path": list(sys.path),
  "python_no_user_site_env": os.environ.get("PYTHONNOUSERSITE"),
  "torch_version": getattr(torch, "__version__", None),
  "torch_file": getattr(torch, "__file__", None),
  "lightning_version": getattr(lightning, "__version__", None),
  "lightning_file": getattr(lightning, "__file__", None),
  "hydra_version": getattr(hydra, "__version__", None),
  "omegaconf_version": getattr(omegaconf, "__version__", None),
  "callback_import": AnnexInitialStateCallback.__name__,
  "src_import": getattr(src, "__file__", None),
  "cuda": {
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "runtime_probe_performed": False,
    "allocation_performed": False,
  },
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(requested), "-c", probe],
        cwd=STREAMING_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise LaunchReadinessError(
            "pinned interpreter import probe failed: " + completed.stderr[-2000:]
        )
    try:
        runtime = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise LaunchReadinessError("pinned interpreter probe returned invalid JSON") from error
    required = (
        "torch_version",
        "lightning_version",
        "hydra_version",
        "omegaconf_version",
        "callback_import",
        "src_import",
    )
    missing = [field for field in required if not runtime.get(field)]
    if missing:
        raise LaunchReadinessError(
            f"pinned interpreter probe missing required imports: {missing}"
        )
    expected_prefix = DEFAULT_PYTHON.parent.parent.resolve()
    runtime_prefix = Path(str(runtime.get("sys_prefix", ""))).resolve()
    if runtime_prefix != expected_prefix:
        raise LaunchReadinessError(
            f"pinned interpreter prefix mismatch: {runtime_prefix} != {expected_prefix}"
        )
    if runtime.get("python_no_user_site_env") != "1" or runtime.get("user_site_enabled") is not False:
        raise LaunchReadinessError("pinned interpreter did not disable the user site")
    user_site = Path(str(runtime.get("user_site", ""))).resolve()
    sys_paths = [Path(str(item)).resolve() for item in runtime.get("sys_path", []) if item]
    if any(item == user_site or user_site in item.parents for item in sys_paths):
        raise LaunchReadinessError("pinned interpreter sys.path contains the user site")
    torch_file = Path(str(runtime.get("torch_file", ""))).resolve()
    if expected_prefix not in torch_file.parents:
        raise LaunchReadinessError(
            f"torch resolved outside the pinned conda environment: {torch_file}"
        )
    if torch_file == user_site or user_site in torch_file.parents:
        raise LaunchReadinessError(f"torch resolved from the user site: {torch_file}")
    runtime["torch_within_pinned_prefix"] = True
    runtime["torch_outside_user_site"] = True
    return {
        "path": str(requested),
        "realpath": str(path.resolve()),
        "sha256": sha256_file(path),
        "runtime": runtime,
        "execution_cwd": str(STREAMING_ROOT.resolve()),
        "pythonpath": env["PYTHONPATH"],
        "python_no_user_site": env["PYTHONNOUSERSITE"],
        "cpu_only_probe": True,
    }


def implementation_snapshot() -> dict[str, dict[str, str]]:
    """Hash all shared inputs and this isolated implementation.

    The snapshot is emitted into the readiness receipt and passed to the
    source callback.  A later launcher can refuse a run if any shared source
    hash no longer matches the receipt.
    """

    paths = (
        TRAIN_ENTRY_REL,
        RUNNER_REL,
        EVAL_REL,
        EXPERIMENT_REL,
        DATA_REL,
        MODEL_CONFIG_REL,
        MODEL_SOURCE_REL,
        DATA_MODULE_REL,
        SELECTION_CALLBACK_REL,
        TRAINER_REL,
        INSTANTIATOR_REL,
        STREAMING_MODULE_REL,
        RUNTIME_REL,
        LAUNCHER_REL,
        SUPERVISOR_REL,
        SPEC_REL,
    )
    result: dict[str, dict[str, str]] = {}
    for rel in paths:
        path = ROOT / rel
        result[rel.as_posix()] = {
            "sha256": sha256_file(path),
            "mode": f"{path.stat().st_mode & 0o777:03o}",
        }
    return result


def implementation_snapshot_sha256(
    snapshot: Mapping[str, Mapping[str, str]],
) -> str:
    return hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()


def assert_implementation_snapshot(
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    """Re-hash every bound input immediately before an authorized launch."""

    current = implementation_snapshot()
    if dict(expected) != current:
        changed = sorted(
            key
            for key in set(expected) | set(current)
            if expected.get(key) != current.get(key)
        )
        raise LaunchReadinessError(
            "implementation snapshot drift before execution: " + ", ".join(changed)
        )


def provenance_bindings(
    snapshot: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    snapshot = dict(snapshot or implementation_snapshot())
    return {
        "implementation_snapshot_sha256": implementation_snapshot_sha256(snapshot),
        "train_entry_sha256": _snapshot_sha(snapshot, TRAIN_ENTRY_REL),
        "runner_source_sha256": _snapshot_sha(snapshot, RUNNER_REL),
        "evaluator_source_sha256": _snapshot_sha(snapshot, EVAL_REL),
        "experiment_config_sha256": _snapshot_sha(snapshot, EXPERIMENT_REL),
        "data_config_sha256": _snapshot_sha(snapshot, DATA_REL),
        "model_config_sha256": _snapshot_sha(snapshot, MODEL_CONFIG_REL),
        "selection_callback_sha256": _snapshot_sha(snapshot, SELECTION_CALLBACK_REL),
        "runtime_hook_sha256": _snapshot_sha(snapshot, RUNTIME_REL),
    }


def validate_shared_api() -> dict[str, Any]:
    """Validate only APIs used by the isolated route; fail closed otherwise.

    The historical shared CLI is recorded as a reference but is not called:
    it currently omits ``afc4_mb4`` from its argument choices.  Calling it for
    the MB4 arm would be an implicit workaround.  The isolated route instead
    calls the existing Hydra train entry directly, with an explicit arm
    override, and uses the existing one-shot evaluator for target scoring.
    """

    runner = _read_text(RUNNER_REL)
    train = _read_text(TRAIN_ENTRY_REL)
    evaluator = _read_text(EVAL_REL)
    experiment = _read_text(EXPERIMENT_REL)
    data = _read_text(DATA_REL)
    model = _read_text(MODEL_SOURCE_REL)
    instantiator = _read_text(INSTANTIATOR_REL)

    required_checks = {
        "train_entry_instantiates_callbacks": "instantiate_callbacks" in train
        and "trainer.fit" in train,
        "train_entry_has_test_false_path": "test=false" not in train or "cfg.get(\"test\")" in train,
        "experiment_is_nested_loso": "rt_clean_nested_loso_m24" in experiment
        and "nested_loso" in experiment,
        "data_has_explicit_side_feature_group": "side_feature_group: null" in data,
        "data_supports_mb4": "afc4_mb4" in _read_text(DATA_MODULE_REL),
        "model_is_b3s_width4": "variant: B3S" in _read_text(MODEL_CONFIG_REL)
        and "side_dim: 4" in _read_text(MODEL_CONFIG_REL),
        "callback_instantiation_is_leaf_based": "hydra.utils.instantiate" in instantiator,
        "evaluator_records_state_before": "model_state_sha256_before" in evaluator
        and "_state_digest" in evaluator,
        "evaluator_records_state_after": "model_state_sha256_after" in evaluator,
        "evaluator_forbids_optimizer": '"optimizer_present": False' in evaluator,
        "evaluator_forbids_target_backprop": '"target_backpropagation": False' in evaluator,
        "decoder_accounting_api": "decoder_cost_comparison_receipt" in model
        and "persistent_state_bytes_fp32" in model,
    }
    failed = [name for name, passed in required_checks.items() if not passed]
    if failed:
        raise LaunchReadinessError(
            "shared API preflight failed closed: " + ", ".join(failed)
        )

    # This is deliberately informational.  The isolated route does not call
    # the historical runner because its CLI does not expose afc4_mb4; hiding
    # that fact would make the receipt misleading.
    runner_cli_supports_mb4 = '"afc4_mb4"' in runner or "'afc4_mb4'" in runner
    return {
        "required_checks": required_checks,
        "runner_cli_supports_afc4_vel": "afc4_vel" in runner,
        "runner_cli_supports_afc4_mb4": runner_cli_supports_mb4,
        "runner_cli_used_by_isolated_route": False,
        "runner_cli_policy": (
            "not_called__historical_cli_missing_afc4_mb4__direct_hydra_route_only"
            if not runner_cli_supports_mb4
            else "reference_only__direct_hydra_route_still_used_for_pairing"
        ),
        "fail_closed_if_required_direct_api_missing": True,
    }


def _snapshot_sha(snapshot: Mapping[str, Mapping[str, str]], rel: Path) -> str:
    value = snapshot.get(rel.as_posix())
    if not isinstance(value, Mapping) or not value.get("sha256"):
        raise LaunchReadinessError(f"snapshot lacks hash for {rel}")
    return str(value["sha256"])


def _validate_int(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise LaunchReadinessError(f"{label} must be integer") from error
    return parsed


def _cell_from_args(*, seed: int, fold: int, arm: str) -> Cell:
    return Cell(seed=_validate_int(seed, label="seed"), fold=_validate_int(fold, label="fold"), arm=str(arm))


def _artifact_path(artifact_root: str | Path, cell: Cell, filename: str) -> Path:
    root = validate_artifact_root(artifact_root)
    return root / cell.key / filename


def _cell_dir(artifact_root: str | Path, cell: Cell) -> Path:
    root = validate_artifact_root(artifact_root)
    return root / cell.key


def _fit_dir(artifact_root: str | Path, cell: Cell) -> Path:
    return _cell_dir(artifact_root, cell) / "fit"


def build_train_command(
    cell: Cell,
    *,
    artifact_root: str | Path,
    accelerator: str = "gpu",
    devices: int = 1,
    python_executable: str = str(DEFAULT_PYTHON),
    interpreter: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Mapping[str, str]] | None = None,
) -> list[str]:
    """Build a strict direct-Hydra source-fit command; never execute it."""

    if accelerator not in {"cpu", "gpu"}:
        raise LaunchReadinessError("accelerator must be cpu or gpu")
    if _validate_int(devices, label="devices") <= 0:
        raise LaunchReadinessError("devices must be positive")
    validate_artifact_root(artifact_root, require_fresh=True)
    api = validate_shared_api()
    if api["runner_cli_used_by_isolated_route"]:
        raise LaunchReadinessError("isolated route unexpectedly selected the shared CLI")
    pinned_python = _require_pinned_python(python_executable)
    snapshot = dict(snapshot or implementation_snapshot())
    interpreter = dict(interpreter or interpreter_metadata(str(python_executable)))
    if str(pinned_python.resolve()) != interpreter.get("realpath"):
        raise LaunchReadinessError("train command interpreter does not match bound metadata")
    initial_path = _artifact_path(artifact_root, cell, "source_initial_state.json")
    fit_dir = _fit_dir(artifact_root, cell)
    provenance = provenance_bindings(snapshot)
    return [
        str(DEFAULT_PYTHON),
        str((ROOT / TRAIN_ENTRY_REL).resolve()),
        "experiment=rt_clean_nested_loso_m24",
        f"run_id={cell.run_id}",
        "task=rt",
        f"data.loso_fold={cell.fold}",
        f"data.outer_loso_fold={cell.fold}",
        f"data.side_feature_group={cell.arm}",
        f"data.side_feature_shuffle_seed={cell.seed}",
        f"data.sampler_seed={cell.seed}",
        f"seed={cell.seed}",
        "train=true",
        "test=false",
        "ckpt_path=null",
        "model.encoder_warmstart_path=null",
        "no_early_stopping=true",
        "trainer.max_epochs=35",
        f"trainer.accelerator={accelerator}",
        f"trainer.devices={_validate_int(devices, label='devices')}",
        f"hydra.run.dir={fit_dir}",
        "hydra.job.chdir=false",
        "+callbacks.annex_initial_state._target_=sua_exploration.rt_seed_robustness_annex_v2.runtime.AnnexInitialStateCallback",
        f"+callbacks.annex_initial_state.output_path={initial_path}",
        f"+callbacks.annex_initial_state.arm={cell.arm}",
        f"+callbacks.annex_initial_state.fold={cell.fold}",
        f"+callbacks.annex_initial_state.seed={cell.seed}",
        f"+callbacks.annex_initial_state.run_id={cell.run_id}",
        "+callbacks.annex_initial_state.initial_state_phase=before_first_optimizer_step",
        *(
            f"+callbacks.annex_initial_state.{field}={value}"
            for field, value in provenance.items()
        ),
    ]


def _config_arm(config_path: str | Path, expected_arm: str) -> None:
    text = Path(config_path).read_text(encoding="utf-8")
    needle = f"side_feature_group: {expected_arm}"
    if needle not in text:
        raise LaunchReadinessError(
            f"resolved config does not bind side_feature_group={expected_arm!r}"
        )
    other = [arm for arm in ALLOWED_ARMS if arm != expected_arm and f"side_feature_group: {arm}" in text]
    if other:
        raise LaunchReadinessError(
            f"resolved config contains conflicting side_feature_group entries: {other}"
        )


def build_eval_command(
    cell: Cell,
    *,
    config: str | Path,
    checkpoint: str | Path,
    split_manifest: str | Path,
    selection_receipt: str | Path,
    output: str | Path,
    device: str = "cpu",
    python_executable: str = str(DEFAULT_PYTHON),
    interpreter: Mapping[str, Any] | None = None,
) -> list[str]:
    """Build the existing one-shot target evaluator command."""

    validate_shared_api()
    _config_arm(config, cell.arm)
    pinned_python = _require_pinned_python(python_executable)
    interpreter = dict(interpreter or interpreter_metadata(str(python_executable)))
    if str(pinned_python.resolve()) != interpreter.get("realpath"):
        raise LaunchReadinessError("eval command interpreter does not match bound metadata")
    return [
        str(DEFAULT_PYTHON),
        str((ROOT / RUNNER_REL).resolve()),
        "eval",
        "--config",
        str(Path(config).resolve()),
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--split-manifest",
        str(Path(split_manifest).resolve()),
        "--selection-receipt",
        str(Path(selection_receipt).resolve()),
        "--output",
        str(Path(output).resolve()),
        "--outer-fold",
        str(cell.fold),
        "--device",
        str(device),
    ]


def _required_hash(value: Any, *, label: str, strict_sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise LaunchReadinessError(f"{label} must be a non-empty hash string")
    if strict_sha256 and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LaunchReadinessError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_source_initial_receipt(payload: Mapping[str, Any], cell: Cell) -> None:
    if payload.get("schema") != "rt_seed_robustness_annex_v2_source_initial_state_v1":
        raise LaunchReadinessError("source initial receipt schema mismatch")
    if payload.get("status") != "PASS_SOURCE_INITIAL_STATE_RECORDED":
        raise LaunchReadinessError("source initial receipt did not pass")
    if int(payload.get("fold", -1)) != cell.fold or int(payload.get("seed", -1)) != cell.seed:
        raise LaunchReadinessError("source initial receipt fold/seed mismatch")
    if payload.get("arm") != cell.arm:
        raise LaunchReadinessError("source initial receipt arm mismatch")
    _required_hash(
        payload.get("initial_state_hash"), label="initial_state_hash", strict_sha256=True
    )
    if payload.get("initial_state_phase") != "before_first_optimizer_step":
        raise LaunchReadinessError("initial state was not recorded before the first optimizer step")
    for field in PROVENANCE_HASH_FIELDS:
        _required_hash(payload.get(field), label=field, strict_sha256=True)
    accounting = payload.get("accounting")
    if not isinstance(accounting, Mapping):
        raise LaunchReadinessError("source initial receipt lacks accounting")
    for field in ("parameter_count", "macs_per_decode_call", "cached_state_bytes"):
        value = _validate_int(accounting.get(field), label=field)
        if value <= 0 and field != "cached_state_bytes":
            raise LaunchReadinessError(f"source accounting {field} is not positive")
    if int(accounting.get("cached_state_bytes", -1)) < 0:
        raise LaunchReadinessError("source accounting cached_state_bytes is negative")


def _validate_outer_eval(payload: Mapping[str, Any], cell: Cell) -> None:
    if payload.get("schema") != "rt_clean_nested_loso_outer_eval_v1":
        raise LaunchReadinessError("outer evaluator schema mismatch")
    if payload.get("status") != EXPECTED_STATUS:
        raise LaunchReadinessError("outer evaluator did not produce a clean one-shot pass")
    if payload.get("arm") != cell.arm or int(payload.get("seed", -1)) != cell.seed:
        raise LaunchReadinessError("outer evaluator arm/seed mismatch")
    if int(payload.get("outer_loso_fold", -1)) != cell.fold:
        raise LaunchReadinessError("outer evaluator fold mismatch")
    if payload.get("target_backpropagation") is not False:
        raise LaunchReadinessError("target backpropagation is not false")
    if payload.get("optimizer_present") is not False:
        raise LaunchReadinessError("target optimizer is present or unproven absent")
    if payload.get("model_state_unchanged") is not True:
        raise LaunchReadinessError("target model state was not proven unchanged")
    before = _required_hash(
        payload.get("model_state_sha256_before"),
        label="state before",
        strict_sha256=True,
    )
    after = _required_hash(
        payload.get("model_state_sha256_after"),
        label="state after",
        strict_sha256=True,
    )
    if before != after:
        raise LaunchReadinessError("target state hashes differ")
    for flag in (
        "target_query_labels_used_for_calibration",
        "target_query_labels_used_for_normalization",
        "target_query_labels_used_for_checkpoint_selection",
    ):
        if payload.get(flag) is not False:
            raise LaunchReadinessError(f"target query-label guard failed: {flag}")
    if payload.get("target_query_labels_used_for_scoring_only") is not True:
        raise LaunchReadinessError("target query labels are not marked scoring-only")
    r2 = payload.get("r2_variance_weighted")
    if not isinstance(r2, (float, int)) or not float(r2) == float(r2):
        raise LaunchReadinessError("outer evaluator R2 is not finite")
    if int(payload.get("query_windows_evaluated", 0)) <= 0:
        raise LaunchReadinessError("outer evaluator has no query windows")


def _source_projections(
    selection: Mapping[str, Any], *, split_manifest: Path
) -> dict[str, str]:
    recorded = selection.get("split_manifest_sha256")
    if not isinstance(recorded, str) or not recorded:
        raise LaunchReadinessError("selection receipt lacks split_manifest_sha256")
    if sha256_file(split_manifest) != recorded:
        raise LaunchReadinessError("split manifest does not match selection receipt")
    manifest = _json_load(split_manifest)
    normalizer = manifest.get("source_only_normalizer")
    if not isinstance(normalizer, Mapping):
        raise LaunchReadinessError("fit split manifest lacks source_only_normalizer")
    try:
        scope_hash = spec.source_split_scope_sha256(manifest)
        numeric_hash = spec.source_normalizer_numeric_sha256(normalizer)
    except spec.SpecError as error:
        raise LaunchReadinessError(str(error)) from error
    return {
        "source_normalizer_sha256": hashlib.sha256(
            canonical_json_bytes(normalizer)
        ).hexdigest(),
        "source_split_scope_sha256": scope_hash,
        "source_normalizer_numeric_sha256": numeric_hash,
    }


def _validate_current_source_provenance(source: Mapping[str, Any]) -> None:
    current = provenance_bindings()
    mismatched = [
        field for field, expected in current.items() if source.get(field) != expected
    ]
    if mismatched:
        raise LaunchReadinessError(
            "source receipt provenance does not match current implementation: "
            + ", ".join(mismatched)
        )


def _validate_paired_initial_receipt(
    payload: Mapping[str, Any],
    *,
    cell: Cell,
    source: Mapping[str, Any],
    source_initial_receipt: Path,
) -> None:
    if payload.get("schema") != "rt_seed_robustness_annex_v2_paired_initial_state_v1":
        raise LaunchReadinessError("paired initial-state receipt schema mismatch")
    if payload.get("status") != "PASS_PAIRED_INITIAL_STATE_EQUAL_NOT_AUTHORIZED":
        raise LaunchReadinessError("paired initial-state receipt did not pass")
    if int(payload.get("seed", -1)) != cell.seed or int(payload.get("fold", -1)) != cell.fold:
        raise LaunchReadinessError("paired initial-state receipt fold/seed mismatch")
    if payload.get("arms") != [spec.FULL_ARM, spec.MB4_ARM]:
        raise LaunchReadinessError("paired initial-state receipt arm order mismatch")
    arm_path_field = "full_receipt" if cell.arm == spec.FULL_ARM else "mb4_receipt"
    if Path(str(payload.get(arm_path_field, ""))).resolve() != source_initial_receipt.resolve():
        raise LaunchReadinessError("paired receipt does not bind this source receipt")
    if payload.get("initial_state_hash") != source.get("initial_state_hash"):
        raise LaunchReadinessError("paired receipt initial-state hash mismatch")
    accounting = payload.get("accounting")
    if not isinstance(accounting, Mapping) or dict(accounting) != dict(source["accounting"]):
        raise LaunchReadinessError("paired receipt accounting mismatch")
    required_equal = set(PROVENANCE_HASH_FIELDS) | {
        "initial_state_hash",
        "parameter_count",
        "macs_per_decode_call",
        "cached_state_bytes",
    }
    if not required_equal.issubset(set(payload.get("paired_equality_fields", []))):
        raise LaunchReadinessError("paired receipt lacks required equality fields")


def build_cell_receipt(
    cell: Cell,
    *,
    source_initial_receipt: str | Path,
    outer_eval_receipt: str | Path,
    selection_receipt: str | Path,
    split_manifest: str | Path,
    paired_initial_receipt: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Normalize the shared evaluator output into the v2 target-field schema."""

    source = _json_load(source_initial_receipt)
    outer = _json_load(outer_eval_receipt)
    selection = _json_load(selection_receipt)
    paired = _json_load(paired_initial_receipt)
    _validate_source_initial_receipt(source, cell)
    _validate_current_source_provenance(source)
    _validate_paired_initial_receipt(
        paired,
        cell=cell,
        source=source,
        source_initial_receipt=Path(source_initial_receipt),
    )
    _validate_outer_eval(outer, cell)
    if selection.get("schema") != "rt_clean_nested_loso_selection_receipt_v1":
        raise LaunchReadinessError("selection receipt schema mismatch")
    if selection.get("status") != "PASS_FIT_INNER_SELECTION_ONLY":
        raise LaunchReadinessError("selection receipt is not a clean inner-fit pass")
    if int(selection.get("outer_loso_fold", -1)) != cell.fold or int(selection.get("seed", -1)) != cell.seed:
        raise LaunchReadinessError("selection receipt fold/seed mismatch")
    if selection.get("arm") != cell.arm:
        raise LaunchReadinessError("selection receipt arm mismatch")
    projections = _source_projections(selection, split_manifest=Path(split_manifest))
    accounting = source["accounting"]
    checkpoint_sha = _required_hash(
        outer.get("checkpoint_sha256"),
        label="checkpoint_sha256",
        strict_sha256=True,
    )
    row: dict[str, Any] = {
        "schema": "rt_seed_robustness_annex_v2_cell_v1",
        "status": EXPECTED_STATUS,
        "development_only": True,
        "formal_heldout_opened": False,
        "seed": cell.seed,
        "fold": cell.fold,
        "arm": cell.arm,
        "target_session": outer.get("outer_target_session"),
        "inner_validation_session": outer.get("inner_validation_session"),
        "r2_variance_weighted": float(outer["r2_variance_weighted"]),
        "query_windows_evaluated": int(outer["query_windows_evaluated"]),
        "selected_epoch": int(selection.get("selected_epoch", -1)),
        "selected_global_step": int(selection.get("selected_global_step", -1)),
        "target_model_state_unchanged": True,
        "target_model_state_before_sha256": outer["model_state_sha256_before"],
        "target_model_state_after_sha256": outer["model_state_sha256_after"],
        "target_backpropagation": False,
        "target_optimizer_present": False,
        "target_forward_only_accounting": {
            "parameter_count": int(accounting["parameter_count"]),
            "macs_per_decode_call": int(accounting["macs_per_decode_call"]),
            "cached_state_bytes": int(accounting["cached_state_bytes"]),
            "paired_full_mb4_equal_before_target_eval": True,
            "scope": "loaded_student_architecture_used_for_forward_only_outer_target_eval",
        },
        "source_split_manifest_sha256": selection["split_manifest_sha256"],
        "source_normalizer_sha256": projections["source_normalizer_sha256"],
        "source_split_scope_sha256": projections["source_split_scope_sha256"],
        "source_normalizer_numeric_sha256": projections["source_normalizer_numeric_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "parameter_count": int(accounting["parameter_count"]),
        "macs_per_decode_call": int(accounting["macs_per_decode_call"]),
        "cached_state_bytes": int(accounting["cached_state_bytes"]),
        "initial_state_hash": source["initial_state_hash"],
        "implementation_snapshot_sha256": source["implementation_snapshot_sha256"],
        "paired_initial_state_equal_before_target_eval": True,
        "source_initial_state_receipt": str(Path(source_initial_receipt).resolve()),
        "paired_initial_state_receipt": str(Path(paired_initial_receipt).resolve()),
        "outer_eval_receipt": str(Path(outer_eval_receipt).resolve()),
        "selection_receipt": str(Path(selection_receipt).resolve()),
    }
    _write_exclusive(Path(output), row)
    return row


def pair_initial_state_receipts(
    *,
    cell_seed: int,
    cell_fold: int,
    full_receipt: str | Path,
    mb4_receipt: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Require identical initialization/accounting for Full and MB4."""

    full = _json_load(full_receipt)
    mb4 = _json_load(mb4_receipt)
    full_cell = Cell(seed=cell_seed, fold=cell_fold, arm=spec.FULL_ARM)
    mb4_cell = Cell(seed=cell_seed, fold=cell_fold, arm=spec.MB4_ARM)
    _validate_source_initial_receipt(full, full_cell)
    _validate_source_initial_receipt(mb4, mb4_cell)
    equality_fields = (
        "initial_state_hash",
        *PROVENANCE_HASH_FIELDS,
    )
    for field in equality_fields:
        if full.get(field) != mb4.get(field):
            raise LaunchReadinessError(f"paired initial-state mismatch: {field}")
    for field in ("parameter_count", "macs_per_decode_call", "cached_state_bytes"):
        if full["accounting"].get(field) != mb4["accounting"].get(field):
            raise LaunchReadinessError(f"paired accounting mismatch: {field}")
    result = {
        "schema": "rt_seed_robustness_annex_v2_paired_initial_state_v1",
        "status": "PASS_PAIRED_INITIAL_STATE_EQUAL_NOT_AUTHORIZED",
        "development_only": True,
        "formal_heldout_opened": False,
        "seed": int(cell_seed),
        "fold": int(cell_fold),
        "arms": [spec.FULL_ARM, spec.MB4_ARM],
        "initial_state_hash": full["initial_state_hash"],
        "accounting": dict(full["accounting"]),
        "paired_equality_fields": list(equality_fields)
        + ["parameter_count", "macs_per_decode_call", "cached_state_bytes"],
        "full_receipt": str(Path(full_receipt).resolve()),
        "mb4_receipt": str(Path(mb4_receipt).resolve()),
    }
    _write_exclusive(Path(output), result)
    return result


def _build_readiness_payload_v2_1(
    *,
    python_executable: str = str(DEFAULT_PYTHON),
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Build the v2.1 static launch/e2e-wave payload without writing it.

    Building the payload never creates artifact directories, reads NWB data,
    starts a Trainer, or launches a subprocess other than the CPU-only
    import/version probe in :func:`interpreter_metadata`.
    """

    pinned_python = _require_pinned_python(python_executable)
    campaign_root = validate_artifact_root(artifact_root, require_fresh=True)
    api = validate_shared_api()
    snapshot = implementation_snapshot()
    interpreter = interpreter_metadata(str(pinned_python))
    from . import supervisor  # local import avoids launcher/supervisor cycle

    waves: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for fold in spec.SELECTED_FOLDS:
            wave_root = campaign_root / f"s{seed}_f{fold}_paired_wave"
            plan = supervisor.build_wave_plan(
                    seed=seed,
                    fold=fold,
                    artifact_root=wave_root,
                    python_executable=str(pinned_python),
                    snapshot=snapshot,
                )
            # Keep the receipt self-contained and avoid duplicating a second
            # mutable plan file.  Each wave still carries the exact command,
            # deterministic output paths, and the provenance policy.
            waves.append(plan)
            cells.extend(plan["cells"])
    receipt = {
        "schema": "rt_seed_robustness_annex_v2_1_launch_readiness_v1",
        "status": "PASS_STATIC_E2E_WAVE_READINESS_NOT_AUTHORIZED",
        "supersedes": "rt_seed_robustness_annex_v2_launch_readiness_v1",
        "development_only": True,
        "gpu_authorized": False,
        "gpu_launched": False,
        "training_started": False,
        "nwb_read": False,
        "formal_heldout_opened": False,
        "execution_policy": {
            "default": "print_static_plan_only",
            "execute_requires_cli_flag": "--execute",
            "execute_requires_environment": f"{EXECUTION_ENABLE_ENV}=1",
            "execute_requires_gpu_authorization": (
                f"{EXECUTION_AUTHORIZATION_ENV}={EXECUTION_AUTHORIZATION_VALUE}"
            ),
            "shared_runner_monkeypatch": False,
            "shared_runner_cli_called": False,
            "fail_closed_on_required_shared_api_missing": True,
            "no_mtime_or_score_based_path_selection": True,
            "source_fit_ckpt_path": "null",
            "encoder_warmstart_path": "null",
            "artifact_root_must_be_fresh_at_authorized_wave_start": True,
        },
        "interpreter": interpreter,
        "execution_environment": {
            "cwd": str(STREAMING_ROOT.resolve()),
            "pythonpath": execution_environment()["PYTHONPATH"],
            "python_no_user_site": execution_environment()["PYTHONNOUSERSITE"],
        },
        "artifact_policy": {
            "campaign_root": str(campaign_root),
            "absolute": True,
            "external_to_repository": True,
            "absent_at_readiness_build": True,
            "one_fresh_root_per_paired_wave": True,
        },
        "selection": {
            "method": "performance_independent_fold_id_modulo_v2",
            "folds": list(EXPECTED_FOLDS),
            "seeds": list(EXPECTED_SEEDS),
            "arms": list(ALLOWED_ARMS),
            "cells": len(cells),
            "paired_waves": len(waves),
            "selection_used_scores": False,
            "score_bearing_anchor_hashes_used": False,
            "selection_preimage_sha256": spec.compute_selection()["selection_preimage_sha256"],
        },
        "shared_api": api,
        "implementation_snapshot": snapshot,
        "implementation_snapshot_sha256": implementation_snapshot_sha256(snapshot),
        "source_config_evaluator_bindings": provenance_bindings(snapshot),
        "runtime_hook": {
            "path": RUNTIME_REL.as_posix(),
            "callback": "AnnexInitialStateCallback",
            "initial_state_scope": "student.state_dict at on_fit_start before optimizer step",
            "target_fields_normalized_by": LAUNCHER_REL.as_posix(),
        },
        "projection_contract": {
            "raw_hashes_retained_per_arm": [
                "source_split_manifest_sha256", "source_normalizer_sha256",
            ],
            "paired_compare_only": [
                "source_split_scope_sha256", "source_normalizer_numeric_sha256",
            ],
            "normalizer_numeric_projection_excludes": ["feature_group"],
        },
        "failure_policy": {
            "source_failure": "stop paired wave before target evaluation",
            "pair_hash_or_accounting_mismatch": "stop wave and do not aggregate",
            "selection_or_checkpoint_provenance_failure": "stop wave before target evaluation",
            "target_eval_or_finalize_failure": "stop wave and do not aggregate",
        },
        "seed42_endpoint_policy": {
            "seed42_checkpoint_reuse": False,
            "seed42_endpoint_read_by_readiness": False,
            "seed42_endpoint_rewritten": False,
            "seed42_primary_preserved_unchanged": True,
        },
        "gpu_lane_policy": {
            "independent_lanes": True,
            "full_cuda_visible_devices": "0",
            "mb4_cuda_visible_devices": "1",
            "trainer_devices_per_lane": 1,
            "current_receipt_authorizes_or_launches_gpu": False,
        },
        "waves": waves,
        "cells": cells,
    }
    return receipt


def _validate_readiness_output(output: str | Path, campaign_root: str | Path) -> Path:
    readiness_output = Path(output).resolve(strict=False)
    root = Path(campaign_root).resolve(strict=False)
    if readiness_output == root or root in readiness_output.parents:
        raise LaunchReadinessError(
            "readiness receipt must remain outside the fresh artifact campaign root"
        )
    return readiness_output


def build_readiness_receipt_v2_1(
    *,
    output: str | Path,
    python_executable: str = str(DEFAULT_PYTHON),
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Write a new v2.1 static launch/e2e-wave receipt exclusively."""

    receipt = _build_readiness_payload_v2_1(
        python_executable=python_executable,
        artifact_root=artifact_root,
    )
    path = _validate_readiness_output(
        output, receipt["artifact_policy"]["campaign_root"]
    )
    _write_exclusive(path, receipt, readonly=True)
    return receipt


def build_readiness_supplemental_v2_1(
    *,
    output: str | Path,
    python_executable: str = str(DEFAULT_PYTHON),
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    prior_receipt: str | Path = V2_ROOT / "launch_readiness_receipt.json",
) -> dict[str, Any]:
    """Append a user-site-isolated readiness receipt without rewriting the prior seal."""

    prior_path = Path(prior_receipt).resolve()
    prior = _json_load(prior_path)
    if prior.get("schema") != "rt_seed_robustness_annex_v2_1_launch_readiness_v1":
        raise LaunchReadinessError("prior readiness receipt schema mismatch")
    if prior.get("status") != "PASS_STATIC_E2E_WAVE_READINESS_NOT_AUTHORIZED":
        raise LaunchReadinessError("prior readiness receipt status mismatch")
    if prior_path.stat().st_mode & 0o777 != 0o444:
        raise LaunchReadinessError("prior readiness receipt is not sealed mode 0444")

    receipt = _build_readiness_payload_v2_1(
        python_executable=python_executable,
        artifact_root=artifact_root,
    )
    interpreter = receipt["interpreter"]
    runtime = interpreter["runtime"]
    receipt.update(
        {
            "schema": "rt_seed_robustness_annex_v2_1_launch_readiness_supplemental_v1",
            "status": "PASS_STATIC_E2E_WAVE_READINESS_USER_SITE_ISOLATED_SUPPLEMENTAL_NOT_AUTHORIZED",
            "supersedes": None,
            "append_only_supplement": {
                "path": str(prior_path),
                "sha256": sha256_file(prior_path),
                "schema": prior["schema"],
                "status": prior["status"],
                "mode": "0444",
                "preserved_unmodified": True,
                "reason": "prior seal resolved torch from the Python user site because PYTHONNOUSERSITE was unset",
                "prior_torch_file": prior.get("interpreter", {})
                .get("runtime", {})
                .get("torch_file"),
            },
            "interpreter_isolation_summary": {
                "exact_python_path": interpreter["path"],
                "python_realpath": interpreter["realpath"],
                "python_sha256": interpreter["sha256"],
                "sys_prefix": runtime["sys_prefix"],
                "python_no_user_site": interpreter["python_no_user_site"],
                "user_site": runtime["user_site"],
                "user_site_enabled": runtime["user_site_enabled"],
                "sys_path_excludes_user_site": True,
                "torch_file": runtime["torch_file"],
                "torch_version": runtime["torch_version"],
                "torch_within_pinned_prefix": runtime["torch_within_pinned_prefix"],
                "torch_outside_user_site": runtime["torch_outside_user_site"],
            },
            "target_forward_only_accounting_summary": {
                "outer_target_device": "cpu",
                "target_model_state_before_field": "target_model_state_before_sha256",
                "target_model_state_after_field": "target_model_state_after_sha256",
                "target_model_state_equal_required": True,
                "target_optimizer_present": False,
                "target_backpropagation": False,
                "target_query_labels_scoring_only": True,
                "accounting_fields": [
                    "parameter_count",
                    "macs_per_decode_call",
                    "cached_state_bytes",
                ],
                "full_mb4_accounting_equal_before_target_eval_required": True,
                "cell_receipt_field": "target_forward_only_accounting",
            },
            "paired_initial_state_summary": {
                "pairing_unit": "seed_and_fold",
                "arms": [spec.FULL_ARM, spec.MB4_ARM],
                "receipt_schema": "rt_seed_robustness_annex_v2_paired_initial_state_v1",
                "validated_before_any_outer_target_eval": True,
                "initial_state_hash_equal_required": True,
                "provenance_hashes_equal_required": list(PROVENANCE_HASH_FIELDS),
                "accounting_equal_required": [
                    "parameter_count",
                    "macs_per_decode_call",
                    "cached_state_bytes",
                ],
                "mismatch_policy": "stop_wave_before_target_eval_no_aggregate",
            },
        }
    )
    path = _validate_readiness_output(
        output, receipt["artifact_policy"]["campaign_root"]
    )
    _write_exclusive(path, receipt, readonly=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ready = sub.add_parser("readiness", help="write static launch-readiness receipt")
    ready.add_argument("--output", type=Path, required=True)
    ready.add_argument("--python", dest="python_executable", default=str(DEFAULT_PYTHON))
    ready.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)

    supplemental = sub.add_parser(
        "supplemental-readiness",
        help="append a user-site-isolated v2.1 readiness receipt",
    )
    supplemental.add_argument("--output", type=Path, required=True)
    supplemental.add_argument(
        "--prior-receipt",
        type=Path,
        default=V2_ROOT / "launch_readiness_receipt.json",
    )
    supplemental.add_argument(
        "--python", dest="python_executable", default=str(DEFAULT_PYTHON)
    )
    supplemental.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT
    )

    train = sub.add_parser("train", help="print one source-fit command")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    train.add_argument("--artifact-root", type=Path, required=True)
    train.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    train.add_argument("--devices", type=int, default=1)
    train.add_argument("--python", dest="python_executable", default=str(DEFAULT_PYTHON))
    train.add_argument("--execute", action="store_true")

    evaluate = sub.add_parser("eval", help="print one outer-evaluation command")
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--fold", type=int, required=True)
    evaluate.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--selection-receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--python", dest="python_executable", default=str(DEFAULT_PYTHON))
    evaluate.add_argument("--execute", action="store_true")

    pair = sub.add_parser("pair", help="validate paired source initialization/accounting")
    pair.add_argument("--seed", type=int, required=True)
    pair.add_argument("--fold", type=int, required=True)
    pair.add_argument("--full", type=Path, required=True)
    pair.add_argument("--mb4", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)

    finalize = sub.add_parser("finalize", help="normalize one target eval into a v2 cell row")
    finalize.add_argument("--seed", type=int, required=True)
    finalize.add_argument("--fold", type=int, required=True)
    finalize.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    finalize.add_argument("--source-initial", type=Path, required=True)
    finalize.add_argument("--outer-eval", type=Path, required=True)
    finalize.add_argument("--selection", type=Path, required=True)
    finalize.add_argument("--split-manifest", type=Path, required=True)
    finalize.add_argument("--paired-initial", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "readiness":
        receipt = build_readiness_receipt_v2_1(
            output=args.output,
            python_executable=args.python_executable,
            artifact_root=args.artifact_root,
        )
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))
        return 0
    if args.mode == "supplemental-readiness":
        receipt = build_readiness_supplemental_v2_1(
            output=args.output,
            prior_receipt=args.prior_receipt,
            python_executable=args.python_executable,
            artifact_root=args.artifact_root,
        )
        print(
            json.dumps(
                {"status": receipt["status"], "output": str(args.output)},
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "train":
        cell = _cell_from_args(seed=args.seed, fold=args.fold, arm=args.arm)
        command = build_train_command(
            cell,
            artifact_root=args.artifact_root,
            accelerator=args.accelerator,
            devices=args.devices,
            python_executable=args.python_executable,
        )
        print(" ".join(shlex.quote(part) for part in command))
        if args.execute:
            raise LaunchReadinessError(
                "direct single-arm execution is forbidden; use the paired supervisor"
            )
        return 0
    if args.mode == "eval":
        cell = _cell_from_args(seed=args.seed, fold=args.fold, arm=args.arm)
        command = build_eval_command(
            cell,
            config=args.config,
            checkpoint=args.checkpoint,
            split_manifest=args.split_manifest,
            selection_receipt=args.selection_receipt,
            output=args.output,
            device=args.device,
            python_executable=args.python_executable,
        )
        print(" ".join(shlex.quote(part) for part in command))
        if args.execute:
            raise LaunchReadinessError(
                "direct target evaluation is forbidden; the paired supervisor must "
                "validate Full/MB4 initialization and accounting first"
            )
        return 0
    if args.mode == "pair":
        result = pair_initial_state_receipts(
            cell_seed=args.seed,
            cell_fold=args.fold,
            full_receipt=args.full,
            mb4_receipt=args.mb4,
            output=args.output,
        )
        print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
        return 0
    if args.mode == "finalize":
        cell = _cell_from_args(seed=args.seed, fold=args.fold, arm=args.arm)
        result = build_cell_receipt(
            cell,
            source_initial_receipt=args.source_initial,
            outer_eval_receipt=args.outer_eval,
            selection_receipt=args.selection,
            split_manifest=args.split_manifest,
            paired_initial_receipt=args.paired_initial,
            output=args.output,
        )
        print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
