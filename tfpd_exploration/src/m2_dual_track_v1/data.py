"""Source-only + ext-4 compact session banks for m2_dual_track_v1.

Training entry constructs the pinned M2 datamodule with ``setup("fit")``,
``include_heldout_in_fit=False``, and an explicit record-file allowlist.
The external evaluator opens visible ext-4 held-out-calib files only.

Calibration arrays are mask-filtered 100-bin interpolated views.
Query arrays are a different timeline with 49-bin leading padding.
Boundaries are mapped by original trial IDs / raw intervals — never by
comparing the two array offsets directly.
"""

from __future__ import annotations

import gc
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch

from . import champion, contracts, plan

SESSION_RE = re.compile(r"(ses-\d{4}-\d{2}-\d{2}-Run\d+)")
QUERY_PAD_BINS = plan.WINDOW - 1  # 49
HIDDEN_TOKENS = ("hidden", "held-out-test", "heldout-test", "evalai-test", "/test/")
M2_DATA_RELATIVE = "SPINT-main/data/000953"

Role = Literal["source", "ext4", "any"]


def m2_data_dir() -> Path:
    return plan.repo_root() / M2_DATA_RELATIVE


def parse_session_name(path: Path | str) -> str:
    match = SESSION_RE.search(Path(path).name)
    plan.require(match is not None, f"cannot parse session from {path}")
    return match.group(1)


def classify_nwb_role(path: Path | str) -> str:
    text = str(path).replace("\\", "/")
    name = Path(path).name.lower()
    lowered = text.lower()
    if any(token in lowered or token in name for token in HIDDEN_TOKENS):
        return "hidden_or_test"
    if "held-out-calib" in name or "held-out-calib" in lowered:
        session = parse_session_name(path)
        if session in plan.EXCLUDED_EXTERNAL_SESSIONS:
            return "ext_excluded_nov24"
        if session in plan.EXT4_SESSIONS:
            return "ext4"
        return "ext_other"
    if "held-in-calib" in name or "held-in-calib" in lowered:
        return "source_calib"
    if "held-in-minival" in name or "held-in-minival" in lowered:
        return "source_minival"
    if "held-out" in name or "held-out" in lowered:
        return "heldout_other"
    return "unknown"


def source_nwb_allowlist() -> tuple[Path, ...]:
    root = m2_data_dir()
    paths: list[Path] = []
    for session in plan.HELDIN_SESSIONS:
        calib = sorted(root.rglob(f"*held-in-calib*{session}*.nwb"))
        mini = sorted(root.rglob(f"*held-in-minival*{session}*.nwb"))
        plan.require(len(calib) == 1, f"source calib count for {session}: {calib}")
        plan.require(len(mini) == 1, f"source minival count for {session}: {mini}")
        paths.extend(calib)
        paths.extend(mini)
    return tuple(paths)


def ext4_nwb_allowlist() -> tuple[Path, ...]:
    root = m2_data_dir()
    paths: list[Path] = []
    for session in plan.EXT4_SESSIONS:
        found = sorted(root.rglob(f"*held-out-calib*{session}*.nwb"))
        plan.require(len(found) == 1, f"ext4 count for {session}: {found}")
        paths.append(found[0])
    return tuple(paths)


def forbidden_nwb_paths() -> tuple[Path, ...]:
    root = m2_data_dir()
    extra: list[Path] = []
    for session in plan.EXCLUDED_EXTERNAL_SESSIONS:
        extra.extend(root.rglob(f"*held-out-calib*{session}*.nwb"))
    extra.extend(root.rglob("*hidden*"))
    extra.extend(root.rglob("*held-out-test*"))
    return tuple(sorted(set(extra)))


@dataclass
class FileAccessLog:
    """Records every NWB path opened by this process for the data-access audit."""

    role: Role = "any"
    opened: list[str] = field(default_factory=list)

    def record(self, path: Path | str, *, role: Role | None = None) -> None:
        resolved = str(Path(path).resolve()) if Path(path).exists() else str(Path(path))
        kind = classify_nwb_role(resolved)
        effective = role or self.role
        if kind == "hidden_or_test":
            raise plan.DualTrackError(f"refusing hidden/test NWB: {resolved}")
        if effective == "source" and kind not in {"source_calib", "source_minival"}:
            raise plan.DualTrackError(f"train path opened non-source NWB ({kind}): {resolved}")
        if effective == "ext4" and kind != "ext4":
            raise plan.DualTrackError(f"evaluator opened non-ext4 NWB ({kind}): {resolved}")
        self.opened.append(resolved)

    def as_receipt(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "opened": list(self.opened),
            "kinds": [classify_nwb_role(path) for path in self.opened],
            "hidden_or_test_opened": False,
        }


ACCESS = FileAccessLog(role="any")


class _LoadTracker:
    def __init__(self, log: FileAccessLog, role: Role) -> None:
        self.log = log
        self.role = role
        self._original = None

    def __enter__(self) -> FileAccessLog:
        ensure_streaming_on_path()
        from src.data.falcon_datamodule import FalconDataModule

        self._original = FalconDataModule.load_data

        def tracked(module, file, task, use_intertrials=True):
            self.log.record(file, role=self.role)
            return self._original(module, file, task, use_intertrials=use_intertrials)

        FalconDataModule.load_data = tracked  # type: ignore[method-assign]
        return self.log

    def __exit__(self, exc_type, exc, tb) -> None:
        from src.data.falcon_datamodule import FalconDataModule

        if self._original is not None:
            FalconDataModule.load_data = self._original  # type: ignore[method-assign]


def ensure_streaming_on_path() -> Path:
    return champion.ensure_streaming_on_path()


def cache_root() -> Path:
    root = plan.active_run_root() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage0_dir() -> Path:
    dest = plan.active_run_root() / "stage0"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_memmap(path: Path, array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.lib.format.open_memmap(
        path, mode="w+", dtype=array.dtype, shape=array.shape
    )
    mapped[...] = array
    mapped.flush()
    return mapped


def read_memmap(path: Path, mode: str = "r") -> np.ndarray:
    return np.lib.format.open_memmap(path, mode=mode)


def query_support_boundary_padded(
    trial_start_indices_padded: np.ndarray,
    support_horizon: int = plan.SUPPORT_HORIZON,
) -> int:
    starts = np.asarray(trial_start_indices_padded, dtype=np.int64)
    plan.require(starts.size > support_horizon, "need a trial after the support prefix")
    return int(starts[support_horizon])


def raw_bin_from_padded(padded_bin: int, pad: int = QUERY_PAD_BINS) -> int:
    return int(padded_bin - pad)


def filter_disjoint_window_starts(
    window_starts: np.ndarray,
    boundary_padded: int,
) -> np.ndarray:
    """Keep windows whose *entire* 50-bin span starts at/after the support boundary."""
    starts = np.asarray(window_starts, dtype=np.int64)
    return np.ascontiguousarray(starts[starts >= int(boundary_padded)], dtype=np.int64)


def refuse_calib_vs_query_offset_compare(*_args: Any, **_kwargs: Any) -> None:
    raise plan.DualTrackError(
        "never compare mask-filtered interpolated calib offsets to padded query offsets; "
        "map via original trial IDs / raw intervals"
    )


def construct_source_datamodule(*, access: FileAccessLog | None = None):
    """Pinned M2 datamodule, setup("fit"), held-out never opened."""
    ensure_streaming_on_path()
    from src.data.falcon_datamodule import FalconDataModule

    log = access or ACCESS
    log.role = "source"
    data_module = FalconDataModule(
        task="m2",
        data_dir=str(m2_data_dir()),
        heldin_session_names=list(plan.HELDIN_SESSIONS),
        batch_size=plan.EFFECTIVE_BATCH,
        window_size=plan.WINDOW,
        calibration_n_trials=plan.SUPPORT_HORIZON,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=plan.CALIB_TRIAL_LENGTH,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        num_workers=0,
        pin_memory=False,
        validation_protocol="minival",
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=0,
        sampler_seed=plan.SEED_PRIMARY,
        side_feature_group="t4",
        side_feature_shuffle_seed=plan.SEED_PRIMARY,
    )
    plan.require(bool(data_module.hparams.include_heldout_in_fit) is False, "include_heldout_in_fit must be False")
    plan.require(bool(data_module.hparams.include_heldout_in_test) is False, "include_heldout_in_test must stay False")
    with _LoadTracker(log, "source"):
        data_module.setup("fit")
    plan.require(getattr(data_module, "val_heldout_dataset", None) is None, "held-out leaked into fit")
    opened = {classify_nwb_role(path) for path in log.opened}
    plan.require(opened <= {"source_calib", "source_minival"}, f"unexpected source opens: {opened}")
    return data_module


def _window_starts_for_session(dataset: Any, session: str) -> np.ndarray:
    return np.asarray(
        [start for name, start in dataset.window_indices if name == session],
        dtype=np.int64,
    )


def _mapping_receipt(
    *,
    session: str,
    surface: str,
    dataset: Any,
    eligible: np.ndarray,
    support_horizon: int = plan.SUPPORT_HORIZON,
    apply_disjoint: bool,
) -> dict[str, Any]:
    padded_starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
    n_trials = int(padded_starts.size)
    support_ids = tuple(range(min(support_horizon, n_trials)))
    raw_ids = tuple(range(n_trials))
    boundary = None
    raw_boundary = None
    if apply_disjoint and n_trials > support_horizon:
        boundary = query_support_boundary_padded(padded_starts, support_horizon)
        raw_boundary = raw_bin_from_padded(boundary)
    return {
        "session_id": session,
        "surface": surface,
        "query_pad_bins": QUERY_PAD_BINS,
        "calib_is_mask_filtered_100bin_interpolated": True,
        "query_is_padded_timeline": True,
        "do_not_compare_calib_offset_to_query_offset": True,
        "support_horizon": support_horizon,
        "support_trial_ids": list(support_ids),
        "raw_trial_ids": list(raw_ids),
        "n_raw_trials": n_trials,
        "support_boundary_padded_bin": boundary,
        "support_boundary_raw_bin": raw_boundary,
        "eligible_window_count": int(eligible.size),
        "eligible_starts_padded": eligible.astype(np.int64).tolist(),
        "apply_disjoint_on_this_query_file": apply_disjoint,
        "mapping_rule": (
            "window start is on the padded query timeline; a window is legal iff "
            "start >= padded start of trial_id == support_horizon (first post-support trial). "
            "Minival query files are a different recording slice; their trial 0 is not calib trial 0."
        ),
    }


def _session_dir(surface: str, session: str) -> Path:
    dest = cache_root() / surface / session
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _extract_query_arrays(
    dataset: Any,
    session: str,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    covariates = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    targets = np.stack(
        [covariates[int(start) + plan.WINDOW - 1] for start in eligible],
        axis=0,
    ).astype(np.float32, copy=False)
    return neural, targets


def _calib_bundle(dataset: Any, session: str) -> dict[str, np.ndarray]:
    activity = np.asarray(
        dataset.calib_trialized_neural_features[session][: plan.SUPPORT_HORIZON],
        dtype=np.float32,
    )
    plan.require(
        activity.shape == (plan.SUPPORT_HORIZON, plan.CALIB_TRIAL_LENGTH, plan.CHANNELS),
        f"{session} calib activity shape {activity.shape}",
    )
    return {
        "activity": np.ascontiguousarray(activity),
        "calib_neural": np.ascontiguousarray(dataset.calib_neural[session], dtype=np.float32),
        "calib_trial_change": np.ascontiguousarray(dataset.calib_trial_change[session], dtype=bool),
        "angles": np.ascontiguousarray(dataset.calib_trial_target_angles[session], dtype=np.float32),
    }


def _persist_surface_session(
    *,
    surface: str,
    session: str,
    X: np.ndarray,
    targets_native: np.ndarray,
    eligible: np.ndarray,
    mapping: dict[str, Any],
    activity: np.ndarray,
    move_t4: np.ndarray,
    extra: dict[str, Any],
) -> Path:
    dest = _session_dir(surface, session)
    write_memmap(dest / "X_store.npy", np.ascontiguousarray(X, dtype=np.float32))
    write_memmap(dest / "target_store.npy", np.ascontiguousarray(targets_native, dtype=np.float32))
    np.save(dest / "eligible_starts.npy", np.ascontiguousarray(eligible, dtype=np.int64))
    write_memmap(dest / "calib_activity.npy", np.ascontiguousarray(activity, dtype=np.float32))
    np.save(dest / "T.npy", np.ascontiguousarray(move_t4, dtype=np.float32))
    _write_json(dest / "mapping.json", mapping)
    _write_json(dest / "extra.json", extra)
    return dest


def _attach_e0_u(dest: Path, e0: torch.Tensor, u: torch.Tensor, provenance: dict[str, Any]) -> None:
    torch.save({"E0": e0.cpu(), "frozen_u": u.cpu()}, dest / "e0_u.pt")
    _write_json(dest / "provenance.json", provenance)


def load_session_bank(surface: str, session: str, *, device: str | torch.device = "cpu") -> contracts.SessionBank:
    dest = _session_dir(surface, session)
    device = torch.device(device)
    X = read_memmap(dest / "X_store.npy")
    targets = read_memmap(dest / "target_store.npy")
    starts = np.load(dest / "eligible_starts.npy")
    t4 = torch.from_numpy(np.load(dest / "T.npy")).to(device=device)
    payload = torch.load(dest / "e0_u.pt", map_location="cpu", weights_only=False)
    e0 = payload["E0"].to(device=device)
    u = payload["frozen_u"].to(device=device)
    mapping = json.loads((dest / "mapping.json").read_text(encoding="utf-8"))
    provenance = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    mask = torch.ones(plan.CHANNELS, dtype=torch.bool, device=device)
    return contracts.SessionBank(
        session_id=session,
        support_trial_ids=tuple(mapping["support_trial_ids"]),
        raw_trial_ids=tuple(mapping["raw_trial_ids"]),
        X_store=X,
        target_store=targets,
        eligible_starts=starts,
        E0=e0,
        T=t4,
        unit_mask=mask,
        provenance=provenance,
        frozen_u=u,
    )


def iter_session_batches(
    bank: contracts.SessionBank,
    *,
    batch_size: int = plan.EFFECTIVE_BATCH,
    device: str | torch.device = "cpu",
    target_space: str = plan.TRAINING_TARGET_SPACE,
    drop_last: bool = False,
) -> Iterator[contracts.Batch]:
    """Group-by-session batches. Tail batch is kept unless drop_last is requested."""
    plan.require(drop_last is False, "workorder forbids dropping the tail batch")
    device = torch.device(device)
    starts = np.asarray(bank.eligible_starts, dtype=np.int64)
    n_win = int(starts.size)
    store = np.asarray(bank.X_store)
    windowed = store.ndim == 3
    for offset in range(0, n_win, batch_size):
        chunk = starts[offset : offset + batch_size]
        if windowed:
            windows = np.ascontiguousarray(store[offset : offset + chunk.size], dtype=np.float32)
        else:
            windows = np.stack(
                [np.asarray(store[int(start) : int(start) + plan.WINDOW], dtype=np.float32) for start in chunk],
                axis=0,
            )
        native = np.asarray(bank.target_store[offset : offset + chunk.size], dtype=np.float32)
        if target_space == plan.TRAINING_TARGET_SPACE:
            last = native * np.float32(plan.BEHAVIOR_SCALE)
        elif target_space == plan.SCORING_TARGET_SPACE:
            last = native
        else:
            raise plan.DualTrackError(f"unknown target space {target_space}")
        yield contracts.Batch(
            session_id=bank.session_id,
            X=torch.from_numpy(np.ascontiguousarray(windows)).to(device),
            last_target=torch.from_numpy(np.array(last, dtype=np.float32, copy=True)).to(device),
            bank=bank,
            window_ids=tuple(int(start) for start in chunk),
            unit_mask=bank.unit_mask.to(device),
        )


def _digest_starts(starts: np.ndarray) -> str:
    array = np.ascontiguousarray(starts, dtype=np.int64)
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _expected_ext4_reference() -> dict[str, Any]:
    split_path = plan.repo_root() / plan.M33_SPLIT_MANIFEST_RELATIVE
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        "path": str(split_path),
        "used_as": "window_boundary_reference_only",
        "inherited_fold_split": False,
        "inherited_fold_normalizer": False,
        "audit": {
            session: manifest["heldout_query_window_audit"][session]
            for session in plan.EXT4_SESSIONS
        },
    }


def _build_ext4_dataset(access: FileAccessLog):
    ensure_streaming_on_path()
    from falcon_challenge.config import FalconTask
    from src.data.falcon_datamodule import FalconDataModule, FalconDataset

    access.role = "ext4"
    task = FalconTask.m2
    sessions = OrderedDict()
    with _LoadTracker(access, "ext4"):
        helper = FalconDataModule(
            task="m2",
            data_dir=str(m2_data_dir()),
            include_heldout_in_fit=False,
            include_heldout_in_test=False,
            num_workers=0,
            pin_memory=False,
            interpolate_trials=True,
            interpolate_trials_kind="cubic",
            use_intertrials=True,
            side_feature_group="t4",
        )
        for path in ext4_nwb_allowlist():
            session = parse_session_name(path)
            sessions[session] = helper.prepare_session_data(
                path,
                task,
                standardize_covariates=False,
                use_intertrials=True,
                include_trial_targets=True,
            )
    dataset = FalconDataset(
        sessions_dict=sessions,
        calib_sessions_dict=sessions,
        window_size=plan.WINDOW,
        split="val_heldout",
        calibration_n_trials=plan.SUPPORT_HORIZON,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=plan.CALIB_TRIAL_LENGTH,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        side_feature_group="t4",
        query_start_trial=plan.SUPPORT_HORIZON,
        allow_empty_query_sessions=False,
    )
    return dataset


def _compute_identity_for_dest(
    dest: Path,
    encoder: Any,
    move_t4: np.ndarray,
    session: str,
    surface: str,
    mapping: dict[str, Any],
    extra: dict[str, Any],
) -> None:
    activity = read_memmap(dest / "calib_activity.npy")
    trials = torch.from_numpy(np.array(activity, dtype=np.float32, copy=True))
    side = champion.empty_contrast_side(move_t4)
    e0, u = champion.native_e0_and_u(encoder, trials, side)
    provenance = {
        **champion.cache_key_parts(),
        "session_id": session,
        "surface": surface,
        "unit_roster": "m2_96_contiguous",
        "support_ids": mapping["support_trial_ids"],
        "e0_path": "push_trial/finalize_identity",
        "e0_not_batched_mean_gemm": True,
        "e0_sha256": champion.array_sha256(e0.numpy()),
        "u_sha256": champion.array_sha256(u.numpy()),
        "t4_sha256": champion.array_sha256(np.asarray(move_t4, dtype=np.float32)),
        "eligible_start_sha256": _digest_starts(np.load(dest / "eligible_starts.npy")),
        **extra,
    }
    _attach_e0_u(dest, e0, u, provenance)


def stage0_data() -> dict[str, Any]:
    """Build compact memmaps + SessionBanks once. Coordinator hook."""
    import os

    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    torch.set_num_threads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4") or 4)))

    source_log = FileAccessLog(role="source")
    ext_log = FileAccessLog(role="ext4")
    allowlist = [str(path) for path in source_nwb_allowlist()]
    ext_allow = [str(path) for path in ext4_nwb_allowlist()]
    expected_ref = _expected_ext4_reference()

    data_module = construct_source_datamodule(access=source_log)
    train_ds = data_module.train_dataset
    minival_ds = data_module.val_heldin_dataset
    plan.require(set(train_ds.calib_trialized_neural_features) == set(plan.HELDIN_SESSIONS), "held-in drift")

    move_inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    source_calib: dict[str, dict[str, np.ndarray]] = {}
    for session in plan.HELDIN_SESSIONS:
        bundle = _calib_bundle(train_ds, session)
        source_calib[session] = bundle
        move_inputs[session] = (
            bundle["calib_neural"],
            bundle["calib_trial_change"],
            bundle["angles"],
        )
    mean, std, normalizer = champion.fit_source_move_normalizer(move_inputs)
    _write_json(cache_root() / "move_t4_normalizer.json", normalizer)
    plan.require(normalizer["inherited_m33_fold_normalizer"] is False, "fold normalizer leaked")

    source_counts: dict[str, int] = {}
    minival_counts: dict[str, int] = {}
    source_mappings: dict[str, Any] = {}
    for session in plan.HELDIN_SESSIONS:
        padded = np.asarray(train_ds.trial_start_indices[session], dtype=np.int64)
        boundary = query_support_boundary_padded(padded)
        raw_starts = _window_starts_for_session(train_ds, session)
        eligible = filter_disjoint_window_starts(raw_starts, boundary)
        plan.require(eligible.size > 0, f"{session} has no post-support source windows")
        X, targets = _extract_query_arrays(train_ds, session, eligible)
        mapping = _mapping_receipt(
            session=session,
            surface="source_train",
            dataset=train_ds,
            eligible=eligible,
            apply_disjoint=True,
        )
        t4 = champion.fit_move_t4(
            source_calib[session]["calib_neural"],
            source_calib[session]["calib_trial_change"],
            source_calib[session]["angles"],
            session=session,
            mean=mean,
            std=std,
        )
        _persist_surface_session(
            surface="source_train",
            session=session,
            X=X,
            targets_native=targets,
            eligible=eligible,
            mapping=mapping,
            activity=source_calib[session]["activity"],
            move_t4=t4,
            extra={"query_file": "held-in-calib", "support_file": "held-in-calib"},
        )
        source_counts[session] = int(eligible.size)
        source_mappings[session] = mapping

        mini_starts = _window_starts_for_session(minival_ds, session)
        plan.require(mini_starts.size > 0, f"{session} minival empty")
        X_m, y_m = _extract_query_arrays(minival_ds, session, mini_starts)
        mini_map = _mapping_receipt(
            session=session,
            surface="source_minival",
            dataset=minival_ds,
            eligible=mini_starts,
            apply_disjoint=False,
        )
        mini_map["support_from"] = "held-in-calib"
        mini_map["query_from"] = "held-in-minival"
        mini_map["pool_rebuilt_from_query_file"] = False
        mini_map["support_trial_ids"] = list(range(plan.SUPPORT_HORIZON))
        mini_map["minival_local_trial_ids_are_not_calib_ids"] = True
        _persist_surface_session(
            surface="source_minival",
            session=session,
            X=X_m,
            targets_native=y_m,
            eligible=mini_starts,
            mapping=mini_map,
            activity=source_calib[session]["activity"],
            move_t4=t4,
            extra={"query_file": "held-in-minival", "support_file": "held-in-calib"},
        )
        minival_counts[session] = int(mini_starts.size)

    del data_module, train_ds, minival_ds, source_calib, move_inputs
    gc.collect()

    frozen = champion.load_frozen_champion(device="cpu")
    encoder = frozen.student.id_encoder
    for surface, sessions, mappings in (
        ("source_train", plan.HELDIN_SESSIONS, source_mappings),
        ("source_minival", plan.HELDIN_SESSIONS, None),
    ):
        for session in sessions:
            dest = _session_dir(surface, session)
            mapping = json.loads((dest / "mapping.json").read_text(encoding="utf-8"))
            t4 = np.load(dest / "T.npy")
            _compute_identity_for_dest(dest, encoder, t4, session, surface, mapping, {"champion": "REF"})

    ext_dataset = _build_ext4_dataset(ext_log)
    ext_counts: dict[str, int] = {}
    ext_explain: dict[str, Any] = {}
    frozen_window_ids: dict[str, list[int]] = {}
    for session in plan.EXT4_SESSIONS:
        expected = plan.EXT4_EXPECTED_WINDOWS[session]
        starts = _window_starts_for_session(ext_dataset, session)
        audit = ext_dataset.query_window_audit[session]
        if int(starts.size) != expected:
            ext_explain[session] = {
                "expected": expected,
                "actual": int(starts.size),
                "audit": audit,
                "action": "not_fudged",
                "note": (
                    "Window count differs from M33-disjoint reference. "
                    "IDs below are the actual frozen set; do not pad/trim to match 519/490/425/635."
                ),
            }
        else:
            ext_explain[session] = {
                "expected": expected,
                "actual": int(starts.size),
                "match": True,
                "audit_eligible_windows": audit.get("eligible_windows"),
            }
        X, targets = _extract_query_arrays(ext_dataset, session, starts)
        mapping = _mapping_receipt(
            session=session,
            surface="ext4",
            dataset=ext_dataset,
            eligible=starts,
            apply_disjoint=True,
        )
        mapping["m33_reference_eligible"] = expected
        mapping["query_window_audit"] = {
            key: audit[key]
            for key in (
                "total_trials",
                "support_trials",
                "query_start_trial",
                "query_trials",
                "raw_query_start_bin",
                "minimum_window_start_padded_bin",
                "eligible_windows",
                "full_window_disjoint",
            )
            if key in audit
        }
        bundle = _calib_bundle(ext_dataset, session)
        t4 = champion.fit_move_t4(
            bundle["calib_neural"],
            bundle["calib_trial_change"],
            bundle["angles"],
            session=session,
            mean=mean,
            std=std,
        )
        dest = _persist_surface_session(
            surface="ext4",
            session=session,
            X=X,
            targets_native=targets,
            eligible=starts,
            mapping=mapping,
            activity=bundle["activity"],
            move_t4=t4,
            extra={"query_file": "held-out-calib", "support_file": "held-out-calib"},
        )
        _compute_identity_for_dest(dest, encoder, t4, session, "ext4", mapping, {"champion": "REF"})
        ext_counts[session] = int(starts.size)
        frozen_window_ids[session] = [int(start) for start in starts]

    del ext_dataset, frozen
    gc.collect()

    counts_match = ext_counts == dict(plan.EXT4_EXPECTED_WINDOWS)
    receipt = {
        "schema": plan.SCHEMA,
        "contract_version": plan.CONTRACT_VERSION,
        "pass": True,
        "source_allowlist": allowlist,
        "ext4_allowlist": ext_allow,
        "forbidden_not_opened": [str(path) for path in forbidden_nwb_paths()],
        "source_access": source_log.as_receipt(),
        "ext4_access": ext_log.as_receipt(),
        "source_train_windows": source_counts,
        "source_minival_windows": minival_counts,
        "ext4_windows": ext_counts,
        "ext4_expected_windows": dict(plan.EXT4_EXPECTED_WINDOWS),
        "ext4_counts_match_expected": counts_match,
        "ext4_count_explanation": ext_explain,
        "frozen_ext4_window_ids": frozen_window_ids,
        "m33_split_manifest": expected_ref,
        "move_t4_normalizer": {
            "mean": normalizer["mean"],
            "std": normalizer["std"],
            "inherited_m33_fold_normalizer": False,
        },
        "cache_root": str(cache_root()),
        "workers": 0,
        "persistent_workers": False,
        "dense_window_copies": False,
        "query_hidden_cache": False,
        "output_space": champion.OUTPUT_SPACE_CONTRACT,
    }
    hidden_opened = any(
        classify_nwb_role(path) == "hidden_or_test"
        for path in source_log.opened + ext_log.opened
    )
    receipt["pass"] = (not hidden_opened) and all(
        classify_nwb_role(path) in {"source_calib", "source_minival"} for path in source_log.opened
    ) and all(classify_nwb_role(path) == "ext4" for path in ext_log.opened)
    _write_json(cache_root() / "meta.json", receipt)
    _write_json(stage0_dir() / "data_receipt.json", receipt)
    _write_json(stage0_dir() / "source_allowlist.json", {"files": allowlist})
    _write_json(
        stage0_dir() / "ext4_windows.json",
        {
            "counts": ext_counts,
            "expected": dict(plan.EXT4_EXPECTED_WINDOWS),
            "match": counts_match,
            "explanation": ext_explain,
            "frozen_window_ids": frozen_window_ids,
        },
    )
    return receipt


def scoring_manifest(*, surface: str = "ext4") -> dict[str, Any]:
    sessions = list(plan.EXT4_SESSIONS if surface == "ext4" else plan.HELDIN_SESSIONS)
    return {
        "surface": surface,
        "sessions": sessions,
        "cache_root": str(cache_root() / surface),
        "behavior_scale": plan.BEHAVIOR_SCALE,
        "prediction_space": plan.TRAINING_TARGET_SPACE,
        "scoring_space": plan.SCORING_TARGET_SPACE,
        "divide_by_behavior_scale": True,
        "window_ids_are_padded_starts": True,
    }


def load_surface_banks(surface: str, *, device: str | torch.device = "cpu") -> dict[str, contracts.SessionBank]:
    sessions = plan.EXT4_SESSIONS if surface == "ext4" else plan.HELDIN_SESSIONS
    return {session: load_session_bank(surface, session, device=device) for session in sessions}
