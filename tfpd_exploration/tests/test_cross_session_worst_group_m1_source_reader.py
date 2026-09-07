"""No-data/no-CUDA adversarial gates for the CS-WG route-owned M1 reader."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.cross_session_worst_group_v1 import core  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import plan  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as lifecycle  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as physical  # noqa: E402
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader as reader  # noqa: E402


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write_synthetic_sources(tmp_path: Path) -> tuple[Path, physical.FrozenM1SourceManifest]:
    source_root = tmp_path / "synthetic_m1_root"
    directory = source_root / "m1_heldin"
    directory.mkdir(parents=True)
    rows: dict[str, physical.SourceFileDescriptor] = {}
    for position, session in enumerate(plan.HELD_IN_SOURCE_SESSIONS):
        body = f"synthetic-no-nwb-{session}".encode("ascii")
        relative = f"m1_heldin/{session}.bin"
        path = source_root / relative
        path.write_bytes(body)
        rows[session] = physical.SourceFileDescriptor(
            session_id=session,
            relative_path=relative,
            sha256=_sha(body),
            byte_count=len(body),
        )
    return source_root, physical.FrozenM1SourceManifest(rows)


class _FakeFalconDataModule:
    constructor_calls: list[Mapping[str, object]] = []
    prepare_calls: list[tuple[str, object, Mapping[str, object]]] = []
    setup_calls = 0

    def __init__(self, **kwargs: object) -> None:
        self.__class__.constructor_calls.append(dict(kwargs))

    def setup(self, *args: object, **kwargs: object) -> None:
        self.__class__.setup_calls += 1
        raise AssertionError("ordinary FalconDataModule.setup must never be called")

    def prepare_session_data(self, source_path: Path, task: object, **kwargs: object) -> dict[str, object]:
        self.__class__.prepare_calls.append((str(source_path), task, dict(kwargs)))
        named = Path(os.readlink(source_path)).name
        session = named.removesuffix(".bin")
        return {
            "session_id": session,
            "neural": np.zeros((250, 64), dtype=np.float32),
            "covariates": np.zeros((250, 16), dtype=np.float32),
            "trial_change": np.zeros((250,), dtype=np.bool_),
            "eval_mask": np.ones((250,), dtype=np.bool_),
        }


class _FakeFalconDataset:
    constructor_calls: list[Mapping[str, object]] = []
    row_count = 3

    def __init__(self, *, sessions_dict: Mapping[str, Mapping[str, object]],
                 calib_sessions_dict: Mapping[str, Mapping[str, object]], **kwargs: object) -> None:
        self.__class__.constructor_calls.append({
            "sessions_dict": sessions_dict, "calib_sessions_dict": calib_sessions_dict, **kwargs,
        })
        assert tuple(sessions_dict) == tuple(calib_sessions_dict)
        self.session = next(iter(sessions_dict))
        marker = float(plan.HELD_IN_SOURCE_SESSIONS.index(self.session) + 1)
        self.calibration = np.full(plan.M1_CALIBRATION_SHAPE_PER_ROW, marker, dtype=np.float32)
        self.window_indices = [(self.session, index) for index in range(self.__class__.row_count)]
        self.eval_mask = {
            self.session: np.ones((plan.M1_WINDOW_SIZE + self.__class__.row_count,), dtype=np.bool_),
        }

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        session, start = self.window_indices[index]
        x = np.full((plan.M1_WINDOW_SIZE, 64), float(index + 1), dtype=np.float32)
        covariates = np.zeros((plan.M1_WINDOW_SIZE, 16), dtype=np.float32)
        covariates[-1, index % plan.M1_RAW_BEHAVIOR_OUTPUTS] = float(index + 1)
        return x, covariates, self.calibration.copy(), session


def _fake_runtime(_root: Path) -> reader.NativeM1Runtime:
    return reader.NativeM1Runtime(
        falcon_datamodule_type=_FakeFalconDataModule,
        falcon_dataset_type=_FakeFalconDataset,
        task="m1-task",
        external_versions={
            "falcon_challenge": "fake", "lightning": "fake", "numpy": "fake",
            "scipy": "fake", "torch": "fake",
        },
        parser_module_path="synthetic://falcon_datamodule.py",
    )


def _reset_fake_runtime() -> None:
    _FakeFalconDataModule.constructor_calls.clear()
    _FakeFalconDataModule.prepare_calls.clear()
    _FakeFalconDataModule.setup_calls = 0
    _FakeFalconDataset.constructor_calls.clear()
    _FakeFalconDataset.row_count = 3


def test_repair_workorder_and_explicit_reader_closure_bindings() -> None:
    assert lifecycle.READER_REPAIR_WORKORDER_SHA256 == "e535af20ba42faf7fee2053d709aeb546a01d592e434f2086f7781e3e24fa129"
    assert lifecycle.ACCEPTED_SOURCE_LIFECYCLE_CLOSURE_SHA256 == "2f2078bcd89476b84c42d78abedd6b2430d6114bd6550e1ef5e938352e429bdf"
    closure = lifecycle.implementation_closure(ROOT)
    paths = [row["path"] for row in closure["paths"]]
    assert lifecycle.READER_REPAIR_WORKORDER_RELATIVE in paths
    assert "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py" in paths
    assert "streaming_calibration_exp/src/data/falcon_datamodule.py" in paths
    assert "streaming_calibration_exp/src/data/falcon_t4_features.py" in paths
    assert lifecycle.M1_METADATA_MANIFEST_RELATIVE in paths
    metadata_row = next(row for row in closure["paths"] if row["path"] == lifecycle.M1_METADATA_MANIFEST_RELATIVE)
    assert metadata_row["sha256"] == lifecycle.M1_METADATA_MANIFEST_SHA256
    assert closure["reader_repair_workorder_sha256"] == lifecycle.READER_REPAIR_WORKORDER_SHA256


def test_clean_subprocess_keeps_route_package_and_historical_src_namespace_distinct() -> None:
    code = (
        "import pathlib, torch; "
        "import tfpd_exploration.src.cross_session_worst_group_v1.source_physical as route; "
        "import src.data.falcon_datamodule as native; "
        "print(pathlib.Path(route.__file__).resolve()); print(pathlib.Path(native.__file__).resolve()); "
        "print(torch.cuda.is_initialized())"
    )
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{ROOT / 'streaming_calibration_exp'}:{ROOT}",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    route_path, native_path, cuda = completed.stdout.splitlines()
    assert "/tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py" in route_path
    assert "/streaming_calibration_exp/src/data/falcon_datamodule.py" in native_path
    assert cuda == "False"


def test_actual_closure_bound_native_runtime_imports_without_source_or_cuda() -> None:
    code = (
        "from pathlib import Path; "
        "import tfpd_exploration.src.cross_session_worst_group_v1.source_reader as reader; "
        f"runtime=reader.load_native_m1_runtime(Path({str(ROOT)!r})); "
        "import src.data.falcon_datamodule as native; import torch; "
        "print(runtime.parser_module_path); print(native.__file__); print(torch.cuda.is_initialized())"
    )
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{ROOT / 'streaming_calibration_exp'}:{ROOT}",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    parser_path, native_path, cuda = completed.stdout.splitlines()
    assert parser_path == native_path
    assert parser_path.endswith("/streaming_calibration_exp/src/data/falcon_datamodule.py")
    assert cuda == "False"


def test_reader_directly_uses_native_primitives_with_exact_recipe_and_repeated_session_calibration(
    tmp_path: Path,
) -> None:
    _reset_fake_runtime()
    source_root, manifest = _write_synthetic_sources(tmp_path)
    physical_reader = reader.RouteOwnedM1SourceReader(ROOT, source_root, runtime_loader=_fake_runtime)
    descriptor = manifest.files_by_session["20120926"]
    material = physical_reader.read_source_session(descriptor)
    assert _FakeFalconDataModule.setup_calls == 0
    assert len(_FakeFalconDataModule.prepare_calls) == 1
    parser_path, task, parser_kwargs = _FakeFalconDataModule.prepare_calls[0]
    assert parser_path.startswith("/proc/self/fd/") and task == "m1-task"
    assert parser_kwargs == {
        "standardize_covariates": False, "covariates_mean": None, "covariates_std": None,
        "use_intertrials": True, "include_trial_targets": False, "include_trial_obj_ids": False,
    }
    native_kwargs = _FakeFalconDataset.constructor_calls[0]
    for key, value in reader.NATIVE_M1_READER_RECIPE.dataset_kwargs().items():
        assert native_kwargs[key] == value
    assert material.calibration_session == descriptor.session_id
    assert material.payload()["valid_final_bin_count"] == 3
    row_digests = {row.calibration_sha256 for row in material.rows_by_sample_index.values()}
    assert row_digests == {material.calibration_sha256}
    assert all(row.calibration_session == descriptor.session_id for row in material.rows_by_sample_index.values())
    assert material.native_evidence["held_source_identity_before"] == material.native_evidence["held_source_identity_after"]
    assert material.native_evidence["parser"] == "FalconDataModule.prepare_session_data"
    assert material.native_evidence["dataset"] == "FalconDataset"


def test_reader_retains_one_session_calibration_backing_and_core_stacks_only_the_selected_b32_rows(
    tmp_path: Path,
) -> None:
    _reset_fake_runtime()
    # This would require roughly 2.5 GiB of calibration copies under the
    # rejected per-query implementation.  The reader instead retains one
    # 2.5 MiB backing allocation and every typed row owns a read-only header.
    _FakeFalconDataset.row_count = 1024
    try:
        source_root, manifest = _write_synthetic_sources(tmp_path)
        physical_reader = reader.RouteOwnedM1SourceReader(ROOT, source_root, runtime_loader=_fake_runtime)
        material = physical_reader.read_source_session(manifest.files_by_session["20120926"])
        backing = material.calibration_backing
        assert backing.flags.c_contiguous and not backing.flags.writeable
        assert backing.nbytes == 10 * 1024 * 64 * 4
        assert material.payload()["calibration_backing_single_allocation"] is True
        assert material.payload()["calibration_backing_nbytes"] == backing.nbytes
        row_calibrations = [
            row.model_inputs["calib_trialized_neural_features"]
            for row in material.rows_by_sample_index.values()
        ]
        assert len(row_calibrations) == 1024
        assert all(array.flags.c_contiguous and not array.flags.writeable for array in row_calibrations)
        assert all(np.shares_memory(array, backing) for array in row_calibrations)
        assert all(array.__array_interface__["data"][0] == backing.__array_interface__["data"][0]
                   for array in row_calibrations)
        # The only batch-sized materialization is the actual selected-row
        # stack.  It is independent memory, never a batch-wide broadcast view
        # over the session backing.
        selected_b32 = np.stack(row_calibrations[:32], axis=0)
        assert selected_b32.shape == (32, *plan.M1_CALIBRATION_SHAPE_PER_ROW)
        assert not np.shares_memory(selected_b32, backing)
        assert selected_b32.strides[0] == backing.nbytes
    finally:
        _reset_fake_runtime()


def test_reader_all_four_fold_selection_avoids_outer_target_and_does_not_glob(tmp_path: Path) -> None:
    source_root, manifest = _write_synthetic_sources(tmp_path)
    source_root.joinpath("m1_heldin", "unexpected_heldout.nwb").write_bytes(b"not selected")
    physical_reader = reader.RouteOwnedM1SourceReader(ROOT, source_root, runtime_loader=_fake_runtime)
    for target in plan.HELD_IN_SOURCE_SESSIONS:
        spec, _erm = lifecycle.build_fold_route_specs(target)
        selected = manifest.select_exact_sources(spec)
        assert tuple(item.session_id for item in selected) == spec.stage0_spec.source_sessions
        assert target not in tuple(item.session_id for item in selected)
    assert physical_reader.read_events == []


def test_concrete_sealed_metadata_descriptor_loader_is_inert_until_prepare_and_never_resolves_target(
    tmp_path: Path,
) -> None:
    """The only live construction path gets byte counts after attempt only.

    This fixture contains synthetic regular files at the exact *metadata*
    relative paths.  They deliberately do not match the sealed NWB digests:
    descriptor resolution may inspect only the held descriptor/size, while the
    native reader is responsible for the subsequent held body-SHA check.
    """
    authority = lifecycle.load_m1_metadata_manifest_authority(ROOT)
    source_root = tmp_path / "deferred_source_root"
    provider = reader.build_sealed_metadata_descriptor_provider(
        source_root=source_root,
        metadata_authority=authority,
    )
    assert provider.resolution_events == []
    assert isinstance(provider.loader, reader.DeferredHeldM1DescriptorLoader)
    # Factory/provider construction must not resolve or stat the lexical root.
    assert not source_root.exists()

    spec = lifecycle.source_smoke_spec()
    rows = {str(row["session_id"]): row for row in authority["source_rows"]}
    for position, session in enumerate(spec.stage0_spec.source_sessions, start=1):
        path = source_root / str(rows[session]["relative_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-deferred-descriptor-" + bytes([position]))
    target = spec.stage0_spec.outer_target_session
    assert not (source_root / str(rows[target]["relative_path"])).exists()

    descriptors = provider.resolve_exact_sources(spec)
    assert provider.resolution_events == [spec.stage0_spec.source_sessions]
    assert tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions
    assert tuple(item.relative_path for item in descriptors) == tuple(
        str(rows[session]["relative_path"]) for session in spec.stage0_spec.source_sessions
    )
    assert all(item.byte_count == len(b"synthetic-deferred-descriptor-\x01") for item in descriptors)
    assert target not in tuple(item.session_id for item in descriptors)


def test_cross_session_calibration_substitution_dtype_shape_digest_and_broadcast_fail_closed(tmp_path: Path) -> None:
    source_root, manifest = _write_synthetic_sources(tmp_path)
    physical_reader = reader.RouteOwnedM1SourceReader(ROOT, source_root, runtime_loader=_fake_runtime)
    first = physical_reader.read_source_session(manifest.files_by_session["20120924"])
    second = physical_reader.read_source_session(manifest.files_by_session["20120926"])
    row = next(iter(second.rows_by_sample_index.values()))
    foreign = physical.UnassignedSourceM1Row(
        session_id=second.descriptor.session_id,
        sample_index=row.sample_index,
        sample_id="cross-session",
        calibration_session=second.descriptor.session_id,
        calibration_sha256=first.calibration_sha256,
        model_inputs={
            "x": row.model_inputs["x"],
            "calib_trialized_neural_features": next(iter(first.rows_by_sample_index.values())).model_inputs[
                "calib_trialized_neural_features"
            ],
        },
        raw_final_target=row.raw_final_target,
    )
    wrong_rows = dict(second.rows_by_sample_index)
    wrong_rows[row.sample_index] = foreign
    with pytest.raises(physical.SourcePhysicalError, match="physical source row target/session/calibration-backing identity drift"):
        physical.SourceSessionMaterial(
            descriptor=second.descriptor,
            labels=second.labels,
            rows_by_sample_index=wrong_rows,
            valid_source_windows=second.valid_source_windows,
            calibration_session=second.calibration_session,
            calibration_sha256=second.calibration_sha256,
            calibration_backing=second.calibration_backing,
            native_evidence=second.native_evidence,
        )
    with pytest.raises(physical.SourcePhysicalError, match="physical-view"):
        physical.UnassignedSourceM1Row(
            session_id=second.descriptor.session_id,
            sample_index=row.sample_index,
            sample_id="wrong-dtype",
            calibration_session=second.descriptor.session_id,
            calibration_sha256=row.calibration_sha256,
            model_inputs={
                "x": row.model_inputs["x"].astype(np.float64),
                "calib_trialized_neural_features": row.model_inputs["calib_trialized_neural_features"],
            },
            raw_final_target=row.raw_final_target,
        )
    broadcast = np.broadcast_to(
        row.model_inputs["calib_trialized_neural_features"],
        (2, *plan.M1_CALIBRATION_SHAPE_PER_ROW),
    )
    assert broadcast.shape[0] == 2
    with pytest.raises(physical.SourcePhysicalError, match="physical-view"):
        physical.UnassignedSourceM1Row(
            session_id=second.descriptor.session_id,
            sample_index=row.sample_index,
            sample_id="batch-broadcast",
            calibration_session=second.descriptor.session_id,
            calibration_sha256=row.calibration_sha256,
            model_inputs={"x": row.model_inputs["x"], "calib_trialized_neural_features": broadcast},
            raw_final_target=row.raw_final_target,
        )


def test_held_source_descriptor_rejects_symlink_hardlink_size_body_parent_and_named_inode_swaps(tmp_path: Path) -> None:
    source_root, manifest = _write_synthetic_sources(tmp_path)
    descriptor = manifest.files_by_session["20120924"]
    body_path = source_root / descriptor.relative_path
    with reader.HeldSourceFile.open(source_root, descriptor) as held:
        replacement = body_path.with_suffix(".replacement")
        replacement.write_bytes(body_path.read_bytes())
        os.replace(replacement, body_path)
        with pytest.raises(reader.SourceReaderError, match="identity|named"):
            held.revalidate_after_parse()
    body_path.write_bytes(b"too-short")
    with pytest.raises(reader.SourceReaderError, match="size"):
        reader.HeldSourceFile.open(source_root, descriptor)
    body_path.write_bytes(f"synthetic-no-nwb-{descriptor.session_id}".encode("ascii"))
    mutated = bytearray(body_path.read_bytes())
    mutated[0] ^= 1
    body_path.write_bytes(bytes(mutated))
    with pytest.raises(reader.SourceReaderError, match="SHA"):
        reader.HeldSourceFile.open(source_root, descriptor)
    body_path.write_bytes(f"synthetic-no-nwb-{descriptor.session_id}".encode("ascii"))
    alias = body_path.with_name("20120924_alias.bin")
    os.link(body_path, alias)
    alias_descriptor = physical.SourceFileDescriptor(
        descriptor.session_id, "m1_heldin/20120924_alias.bin", descriptor.sha256, descriptor.byte_count,
    )
    with pytest.raises(reader.SourceReaderError, match="hardlink"):
        reader.HeldSourceFile.open(source_root, alias_descriptor)
    alias.unlink()
    symlink = body_path.with_name("20120924_symlink.bin")
    symlink.symlink_to(body_path.name)
    symlink_descriptor = physical.SourceFileDescriptor(
        descriptor.session_id, "m1_heldin/20120924_symlink.bin", descriptor.sha256, descriptor.byte_count,
    )
    with pytest.raises(reader.SourceReaderError, match="no-follow|descriptor"):
        reader.HeldSourceFile.open(source_root, symlink_descriptor)


def test_held_source_parent_replacement_rejects_a_matching_named_body(tmp_path: Path) -> None:
    source_root, manifest = _write_synthetic_sources(tmp_path)
    descriptor = manifest.files_by_session["20120924"]
    body_path = source_root / descriptor.relative_path
    parent = body_path.parent
    with reader.HeldSourceFile.open(source_root, descriptor) as held:
        displaced = source_root / "m1_heldin_displaced"
        os.rename(parent, displaced)
        parent.mkdir()
        (parent / body_path.name).write_bytes((displaced / body_path.name).read_bytes())
        with pytest.raises(reader.SourceReaderError, match="named source|parent"):
            held.revalidate_after_parse()


def test_held_source_open_fstat_binds_the_actual_root_and_child_directory_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Interpose precisely between named ``lstat/stat`` and ``open``.  The
    # production code must reject the FD it actually opened, rather than only
    # relying on the stale named identity that was observed before open.
    source_root, manifest = _write_synthetic_sources(tmp_path)
    descriptor = manifest.files_by_session["20120924"]
    original_open_dir = reader.HeldSourceFile._open_dir
    root_swapped = False

    def swap_root_before_open(path: Path) -> int:
        nonlocal root_swapped
        if Path(path) == source_root and not root_swapped:
            root_swapped = True
            displaced = source_root.with_name("synthetic_m1_root_displaced")
            os.rename(source_root, displaced)
            source_root.mkdir()
        return original_open_dir(path)

    monkeypatch.setattr(reader.HeldSourceFile, "_open_dir", staticmethod(swap_root_before_open))
    with pytest.raises(reader.SourceReaderError, match="root changed between lstat/open"):
        reader.HeldSourceFile.open(source_root, descriptor)

    # Use a fresh tree for the child-directory window.  The source body need
    # not be parsed: changing the child identity after ``stat`` but before its
    # O_NOFOLLOW directory open must itself fail closed.
    monkeypatch.undo()
    source_root, manifest = _write_synthetic_sources(tmp_path / "child")
    descriptor = manifest.files_by_session["20120924"]
    original_os_open = reader.os.open
    child_swapped = False

    def swap_child_before_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal child_swapped
        if path == "m1_heldin" and dir_fd is not None and not child_swapped:
            child_swapped = True
            directory = source_root / "m1_heldin"
            displaced = source_root / "m1_heldin_displaced"
            os.rename(directory, displaced)
            directory.mkdir()
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(reader.os, "open", swap_child_before_open)
    with pytest.raises(reader.SourceReaderError, match="parent changed between stat/open"):
        reader.HeldSourceFile.open(source_root, descriptor)


def test_source_reader_audit_cli_is_static_and_cannot_authorize() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_reader_audit.py"
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    dry = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=ROOT, env=env,
                         check=True, text=True, capture_output=True)
    payload = __import__("json").loads(dry.stdout)
    assert payload["imports_torch"] is False and payload["opens_source_or_target"] is False
    assert payload["current_gpu_smoke_capability_issuable"] is False
    assert payload["future_gpu_smoke_requires_successor_identity"] is True
    assert "source-audit terminal" in payload["future_gpu_smoke_required_predecessor"]
    assert payload["source_descriptor_resolution"]["current_closure_bound_metadata_only_manifest_available"] is True
    assert payload["source_descriptor_resolution"]["metadata_manifest"]["body_sha256"] == lifecycle.M1_METADATA_MANIFEST_SHA256
    denied = subprocess.run([sys.executable, str(script), "--audit"], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    assert denied.returncode != 0 and "root-reviewed" in denied.stderr
