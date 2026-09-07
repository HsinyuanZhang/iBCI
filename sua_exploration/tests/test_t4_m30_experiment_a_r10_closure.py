from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


SUA = Path(__file__).resolve().parents[1]
ROOT = SUA.parent
SCRIPTS = SUA / "scripts"
IMPORT_ROOTS = (SCRIPTS, SUA, ROOT / "streaming_calibration_exp")


def source(name: str) -> str:
    return (SCRIPTS / name).read_text()


def load_writer():
    path = SCRIPTS / "write_t4_m30_experiment_a_prelaunch_r10.py"
    spec = importlib.util.spec_from_file_location("r10_writer_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_package_initializers(path: Path) -> set[Path]:
    found = {path.resolve()}
    for import_root in IMPORT_ROOTS:
        try:
            relative = path.resolve().relative_to(import_root.resolve())
        except ValueError:
            continue
        parent = import_root.resolve()
        for part in relative.parts[:-1]:
            parent /= part
            initializer = parent / "__init__.py"
            if initializer.is_file():
                found.add(initializer.resolve())
        break
    return found


def _resolve_absolute_module(module: str) -> Path | None:
    parts = module.split(".") if module else []
    for import_root in IMPORT_ROOTS:
        candidate = import_root.joinpath(*parts)
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file():
                return path.resolve()
    return None


def _resolve_relative_module(current: Path, level: int, module: str | None) -> Path | None:
    base = current.parent
    for _ in range(level - 1):
        base = base.parent
    candidate = base.joinpath(*(module or "").split("."))
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path.resolve()
    return None


def _recursive_local_imports(starts: set[Path]) -> set[Path]:
    pending = [path.resolve() for path in starts]
    discovered: set[Path] = set()
    while pending:
        path = pending.pop()
        for package_path in _with_package_initializers(path):
            if package_path not in discovered:
                discovered.add(package_path)
                if package_path != path:
                    pending.append(package_path)
        tree = ast.parse(path.read_text(), filename=str(path))
        resolved: list[Path] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                resolved.extend(
                    candidate
                    for alias in node.names
                    if (candidate := _resolve_absolute_module(alias.name)) is not None
                )
            elif isinstance(node, ast.ImportFrom):
                candidate = (
                    _resolve_relative_module(path, node.level, node.module)
                    if node.level
                    else _resolve_absolute_module(node.module or "")
                )
                if candidate is not None:
                    resolved.append(candidate)
        pending.extend(candidate for candidate in resolved if candidate not in discovered)
    return discovered


def test_r10_runtime_chain_is_version_pinned_and_foreground_only() -> None:
    scheduler = source("schedule_t4_m30_experiment_a_r10_2gpu.sh")
    runner = source("run_t4_m30_experiment_a_r10_one_cell.sh")
    authorization = source("t4_m30_experiment_a_r10_authorization.py")
    aggregate = source("aggregate_t4_m30_experiment_a_r10.py")

    assert "managed foreground exec session" in scheduler
    assert "never use nohup/disown" in scheduler
    assert 'bash "$RUNNER"' in scheduler
    assert "verify_t4_m30_experiment_a_r10_authorization.py" in scheduler
    assert '[[ "${#cells[@]}" -eq 18 ]]' in scheduler
    assert all(
        field in scheduler
        for field in ("result_sha256", "metadata_sha256", "cost_sha256")
    )
    assert "aggregate_t4_m30_experiment_a_r10.py" in scheduler
    assert "aggregate_r10.json" in scheduler

    assert "sua_t4_m30_component_attribution_v10" in runner
    assert "t4_m30_experiment_a_v10" in runner
    assert "verify_t4_m30_experiment_a_r10_authorization.py" in runner
    assert "--side_feature_pool_size 30" in runner
    assert "--calibration_n_trials 30" in runner
    assert "--max_epochs 12" in runner
    assert "--require_gpu" in runner

    assert "v3_r10_20260803" in authorization
    assert "attribution_v10" in authorization
    assert "aggregate_t4_m30_experiment_a_r5" in aggregate
    assert "t4_m30_experiment_a_r10_authorization" in aggregate
    assert "aggregate_t4_m30_experiment_a_v3" in source(
        "aggregate_t4_m30_experiment_a_r5.py"
    )


def test_r10_writer_seals_complete_declared_dependency_closure() -> None:
    writer = load_writer()
    required = {
        "streaming_calibration_exp/src/__init__.py",
        "streaming_calibration_exp/src/metrics/__init__.py",
        "streaming_calibration_exp/src/metrics/gate2_matrix.py",
        "streaming_calibration_exp/src/metrics/run_artifacts.py",
        "streaming_calibration_exp/src/models/__init__.py",
        "streaming_calibration_exp/src/models/components/__init__.py",
        "streaming_calibration_exp/src/models/components/neuron_dropout.py",
        "streaming_calibration_exp/src/models/components/spint.py",
        "streaming_calibration_exp/src/models/components/streaming_encoders.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_calibration_exp/src/models/components/streaming_spint_t4_logit_residual_adapter.py",
        "streaming_calibration_exp/src/models/falcon_module.py",
        "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "streaming_calibration_exp/src/models/t4_logit_residual_module.py",
        "streaming_calibration_exp/src/utils/__init__.py",
        "streaming_calibration_exp/src/utils/clean_teacher_validation.py",
        "streaming_calibration_exp/src/utils/instantiators.py",
        "streaming_calibration_exp/src/utils/logging_utils.py",
        "streaming_calibration_exp/src/utils/pylogger.py",
        "streaming_calibration_exp/src/utils/rich_utils.py",
        "streaming_calibration_exp/src/utils/utils.py",
        "sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem",
        "sua_exploration/mc_maze/__init__.py",
        "sua_exploration/mc_maze/datamodule.py",
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "sua_exploration/mc_maze/sua_auxiliary_stage0.py",
        "sua_exploration/mc_maze/unit_side_features.py",
        "sua_exploration/scripts/aggregate_t4_m30_experiment_a_r10.py",
        "sua_exploration/scripts/aggregate_t4_m30_experiment_a_r5.py",
        "sua_exploration/scripts/aggregate_t4_m30_experiment_a_v3.py",
        "sua_exploration/scripts/dandi688_gradient_free_protocol.py",
        "sua_exploration/scripts/eval_adaptation_dandi688.py",
        "sua_exploration/scripts/eval_epoch_window_generic_dandi688.py",
        "sua_exploration/scripts/eval_t4_m30_experiment_a.py",
        "sua_exploration/scripts/run_t4_m30_experiment_a_r10_one_cell.sh",
        "sua_exploration/scripts/schedule_t4_m30_experiment_a_r10_2gpu.sh",
        "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
        "sua_exploration/scripts/t4_m30_experiment_a_r10_authorization.py",
        "sua_exploration/scripts/t4_m30_experiment_a_r4_authorization.py",
        "sua_exploration/scripts/t4_m30_experiment_a_r5_authorization.py",
        "sua_exploration/scripts/train_variant_dandi688.py",
        "sua_exploration/scripts/verify_t4_m30_experiment_a_r10_authorization.py",
        "sua_exploration/scripts/write_t4_m30_experiment_a_prelaunch_r10.py",
        "sua_exploration/tests/test_t4_m30_experiment_a_r5_disk_fixture.py",
        "sua_exploration/tests/test_t4_m30_experiment_a_r10_closure.py",
    }
    declared = set(writer.RUNTIME_SOURCES)
    assert required <= declared
    assert len(declared) == len(writer.RUNTIME_SOURCES)
    assert all((ROOT / relative).is_file() for relative in declared)


def test_r10_declared_sources_cover_recursive_local_runtime_imports() -> None:
    writer = load_writer()
    entrypoints = {
        SCRIPTS / "train_variant_dandi688.py",
        SCRIPTS / "eval_t4_m30_experiment_a.py",
        SCRIPTS / "aggregate_t4_m30_experiment_a_r10.py",
        SCRIPTS / "verify_t4_m30_experiment_a_r10_authorization.py",
    }
    discovered = {
        str(path.relative_to(ROOT)) for path in _recursive_local_imports(entrypoints)
    }
    missing = discovered - set(writer.RUNTIME_SOURCES)
    assert not missing, f"unsealed recursive local imports: {sorted(missing)}"


def test_r10_writer_is_write_once_and_receipt_is_last_stage(tmp_path: Path) -> None:
    writer = load_writer()
    assert writer.EXPECTED_PUBLIC_KEY_SHA256 == (
        "ff9d1b2b985c9cd8c2697cfb0e1e5353b8c19d1f3ff094d2987aa1537c79a375"
    )
    text = source("write_t4_m30_experiment_a_prelaunch_r10.py")
    assert "if output_dir.exists()" in text
    assert "raise FileExistsError" in text
    assert "gpu_launch_authorized\": False" in text
    assert "output_dir.mkdir" in text
    assert text.index("source_hashes =") < text.index("output_dir.mkdir")
