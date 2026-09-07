"""Source-only live adapter and authority builders for Track-B v2.

This module is the first code path allowed to open *development source* NWB
files for the H1-excluded CEBRA comparator.  It has deliberately no target
session argument, no arbitrary data path, no CEBRA import, no checkpoint, no
GPU operation, and no score function.  A target/fold successor must consume
the immutable source authorities produced here and separately establish the
stricter target support/query, exact-query, and control gates.

The source route serves two purposes:

* bind and execute the canonical subject-M SUA/pMUA and RT loaders after their
  source SHA/symbol gate has passed; and
* materialise independent source-only authorities for neural input handling,
  dense behavior scaling, and source-readout embedding identity handling.

The neural-input authority is intentionally a parameter-free, dimension-
agnostic float32 count identity.  A source per-channel mean/std is not
deployable to an unseen session with a different number of units, so this code
explicitly refuses that tempting but invalid transform.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_live_contract as live
import track_b_v2_metric_pointer_authority as metric_pointer


SOURCE_ONLY_ADAPTER_SCHEMA = "track_b_v2_source_only_live_adapter_v1"
SOURCE_ROSTER_SCHEMA = "track_b_v2_source_roster_receipt_v1"
SOURCE_COVERAGE_SCHEMA = "track_b_v2_source_coverage_receipt_v1"
SOURCE_NEURAL_AUTHORITY_SCHEMA = "track_b_v2_source_neural_input_authority_v1"
SOURCE_BEHAVIOR_AUTHORITY_SCHEMA = "track_b_v2_source_behavior_auxiliary_scaler_authority_v1"
SOURCE_EMBEDDING_AUTHORITY_SCHEMA = "track_b_v2_source_readout_embedding_identity_authority_v1"
SOURCE_SELECTOR_PLAN_SCHEMA = "track_b_v2_source_only_dual_geometry_execution_plan_v1"
REFERENCE_POINTER_PROPOSAL_SCHEMA = "track_b_v2_metric_pointer_mint_proposal_v1"
RT_15FOLD_SOURCE_AUTHORITY_PLAN_SCHEMA = "track_b_v2_rt_15fold_source_authority_plan_v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = {
    # subject-M is an *external target* domain.  The matched V9 T4 reference
    # trains only on the strict 27 sub-C CO source sessions, never by LODO
    # fitting the other sub-M sessions.
    "subject_m": REPO_ROOT / "sua_exploration" / "data" / "dandi_000688" / "sub-C",
    "rt": REPO_ROOT / "sua_exploration" / "data" / "dandi_000688" / "sub-C",
}
_SUBJECT_M_SOURCE_SESSION = re.compile(r"sub-C_ses-CO-\d{8}")
_RT_SESSION = re.compile(r"ses-RT-\d{8}")
_STRICT27_MANIFEST = REPO_ROOT / "sua_exploration" / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
_STRICT27_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
_V9_SOURCE_MANIFEST = (
    REPO_ROOT / "sua_exploration" / "results" / "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist"
    / "c1_train_val_33_manifest.json"
)
_V9_SOURCE_MANIFEST_SHA256 = "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb"


class TrackBV2SourceAdapterError(live.TrackBV2LiveContractError):
    """Raised before a source-only adapter may create a usable authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2SourceAdapterError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    """Hash dtype, shape, and contiguous bytes; never coerce a semantic array."""
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _strict27_train_ids() -> tuple[str, ...]:
    """Read and cross-bind both source-manifest lineage layers, source only."""
    strict_raw = _read_regular_file(_STRICT27_MANIFEST, label="strict27 source manifest")
    require(hashlib.sha256(strict_raw).hexdigest() == _STRICT27_MANIFEST_SHA256, "strict27 source manifest SHA drift")
    v9_raw = _read_regular_file(_V9_SOURCE_MANIFEST, label="V9 source manifest")
    require(hashlib.sha256(v9_raw).hexdigest() == _V9_SOURCE_MANIFEST_SHA256, "V9 source manifest SHA drift")
    try:
        strict = json.loads(strict_raw.decode("utf-8")); v9 = json.loads(v9_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2SourceAdapterError("strict source manifests are not valid JSON") from exc
    require(isinstance(strict, Mapping) and isinstance(v9, Mapping), "strict source manifests are malformed")
    splits = strict.get("session_splits")
    require(isinstance(splits, Mapping), "strict27 source split map missing")
    train = tuple(splits.get("train", ()))
    require(len(train) == 27 and all(_SUBJECT_M_SOURCE_SESSION.fullmatch(str(item)) for item in train),
            "strict27 train roster drift")
    require(v9.get("source_manifest_sha256") == _STRICT27_MANIFEST_SHA256,
            "V9 source manifest no longer binds strict27 manifest")
    v9_splits = v9.get("session_splits")
    require(isinstance(v9_splits, Mapping) and tuple(v9_splits.get("train", ())) == train,
            "V9 source manifest train roster differs from strict27")
    return tuple(train)


def _read_regular_file(path: Path, *, label: str) -> bytes:
    """Read a mutable manifest through one non-following descriptor.

    The strict27 and V9 manifest bodies are intentionally not required to be
    mode ``0444``: they are versioned source inputs whose exact bytes are
    pinned by the caller's SHA.  They still cannot be read through the old
    ``lstat(); read_bytes()`` pattern, because a pathname replacement could
    make the checked inode differ from the decoded/hashed bytes.  Require a
    regular non-symlink file, read/hashable bytes from one ``O_NOFOLLOW`` fd,
    verify that fd did not change, then reject an open-then-rename pathname
    swap before returning those bytes.
    """
    path = Path(path)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required for source manifest reads")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise TrackBV2SourceAdapterError(f"cannot open {label} without following links: {path}") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            f"{label} changed while being read",
        )
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
    except OSError as exc:
        raise TrackBV2SourceAdapterError(f"{label} disappeared or was replaced after read: {path}") from exc
    require(stat.S_ISREG(named.st_mode) and not stat.S_ISLNK(named.st_mode),
            f"{label} pathname must remain a regular non-symlink")
    require(
        (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        == (named.st_dev, named.st_ino, named.st_mode, named.st_size),
        f"{label} pathname inode changed after read",
    )
    return raw


def _json_pointer_values_from_verified_bytes(raw: bytes, pointer: str, *, label: str) -> tuple[float, ...]:
    """Resolve a metric pointer from the exact fd-verified JSON bytes.

    This is intentionally a tiny JSON-Pointer reader, rather than a generic
    configuration walker.  It has just enough surface for the legacy baseline
    bodies: mapping keys, array indices, and ``*`` for the one explicitly
    declared per-fold list.  A proposal may never carry an unparsed path plus
    an independently typed metric literal -- its numeric values must be
    obtained from the same immutable bytes whose SHA was just checked.
    """
    require(isinstance(pointer, str) and pointer.startswith("/"), f"{label} metric JSON pointer invalid")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2SourceAdapterError(f"{label} is not valid UTF-8 JSON") from exc
    nodes: list[Any] = [decoded]
    for encoded_part in pointer.split("/")[1:]:
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        next_nodes: list[Any] = []
        for node in nodes:
            if part == "*":
                require(isinstance(node, list), f"{label} wildcard pointer component is not an array")
                next_nodes.extend(node)
            elif isinstance(node, Mapping):
                require(part in node, f"{label} metric JSON pointer missing component: {part}")
                next_nodes.append(node[part])
            elif isinstance(node, list) and part.isdecimal():
                index = int(part)
                require(index < len(node), f"{label} metric JSON pointer array index out of range: {part}")
                next_nodes.append(node[index])
            else:
                raise TrackBV2SourceAdapterError(f"{label} metric JSON pointer cannot traverse component: {part}")
        nodes = next_nodes
    require(nodes, f"{label} metric JSON pointer selected no values")
    values: list[float] = []
    for value in nodes:
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                f"{label} metric JSON pointer did not resolve finite numeric values")
        values.append(float(value))
    return tuple(values)


def _canonical_source_ids(dataset: str, values: Sequence[str], *, allow_strict27_smoke_subset: bool = False) -> tuple[str, ...]:
    ids = tuple(str(value) for value in values)
    require(len(ids) >= 2, "source-only geometry selection requires at least two source sessions")
    require(all(value and value == value.strip() for value in ids), "source session ID is empty or padded")
    require(len(set(ids)) == len(ids), "source session IDs contain duplicates")
    pattern = _SUBJECT_M_SOURCE_SESSION if dataset == "subject_m" else _RT_SESSION
    require(all(pattern.fullmatch(value) is not None for value in ids), "source session ID violates canonical dataset grammar")
    canonical = tuple(sorted(ids))
    if dataset == "subject_m":
        strict27 = _strict27_train_ids()
        if allow_strict27_smoke_subset:
            require(set(canonical).issubset(strict27), "source-only smoke may use only strict27 sub-C train sessions")
        else:
            require(canonical == strict27, "subject-M authority must use exactly the strict27 sub-C train roster")
    return canonical


def canonical_source_only_adapter_spec(
    dataset: str,
    view: str | None,
    *,
    source_session_ids: Sequence[str],
    proposed_data_path: object | None = None,
    target_session_id: object | None = None,
    source_only_smoke: bool = False,
) -> dict[str, Any]:
    """Validate source scope before inspecting any source or target-like object.

    A target argument intentionally exists only as a poisonable guard: this
    source-only path rejects it without coercion.  It exposes no raw path
    argument; source filenames are derived from the frozen root plus canonical
    session IDs after H1/M2 rejection.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(proposed_data_path is None, "source-only adapter accepts no arbitrary data path")
    require(target_session_id is None, "source-only adapter has no target session or query access")
    source_ids = _canonical_source_ids(dataset, source_session_ids, allow_strict27_smoke_subset=source_only_smoke)
    adapter = live.canonical_adapter_spec(dataset, view)
    semantics = live.build_sealed_loader_semantics_contract(dataset, view)
    live.validate_sealed_loader_semantics_contract(adapter, semantics)
    root = _SOURCE_ROOTS[dataset]
    info = root.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), "canonical source root is not a regular directory")
    return {
        "schema": SOURCE_ONLY_ADAPTER_SCHEMA,
        "status": (
            "SOURCE_ONLY_LOADER_SMOKE_GATE_PASSED__NO_TARGET_NO_CEBRA_NO_SCORE"
            if source_only_smoke else "SOURCE_ONLY_GATE_PASSED__STRICT_SOURCE_AUTHORITY__NO_TARGET_NO_CEBRA_NO_SCORE"
        ),
        "dataset": dataset,
        "view": view,
        "adapter": adapter,
        "loader_semantics": semantics,
        "loader_semantics_sha256": _sha_json(semantics),
        "source_session_ids": list(source_ids),
        "strict27_source_authority_required": dataset == "subject_m",
        "strict27_source_roster_exact": dataset != "subject_m" or not source_only_smoke,
        "source_only_smoke": source_only_smoke,
        "strict27_source_manifest": (
            {"path": str(_STRICT27_MANIFEST.resolve()), "sha256": _STRICT27_MANIFEST_SHA256}
            if dataset == "subject_m" else None
        ),
        "v9_source_manifest": (
            {"path": str(_V9_SOURCE_MANIFEST.resolve()), "sha256": _V9_SOURCE_MANIFEST_SHA256}
            if dataset == "subject_m" else None
        ),
        "canonical_source_root": str(root.resolve()),
        "source_root_discovered": False,
        "target_session_id": None,
        "target_data_opened": False,
        "target_query_opened": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "checkpoint_loaded": False,
        "gpu_used": False,
        "score_emitted": False,
    }


def _canonical_source_path(dataset: str, source_session_id: str) -> Path:
    root = _SOURCE_ROOTS[dataset]
    if dataset == "subject_m":
        name = f"{source_session_id}_behavior+ecephys.nwb"
    else:
        name = f"sub-C_{source_session_id}_behavior+ecephys.nwb"
    candidate = root / name
    # The filename originates only from the strict session grammar above.  The
    # resolved-parent check still prevents a malformed future grammar escaping
    # the canonical source root.
    require(candidate.parent.resolve() == root.resolve(), "derived source path escaped canonical root")
    info = candidate.lstat()
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"canonical source session is missing or symlinked: {source_session_id}")
    return candidate


@dataclass(frozen=True)
class SourceSessionMaterialization:
    """One source-only canonical loader result, with no target fields."""

    dataset: str
    view: str | None
    session_id: str
    source_path: str
    source_nwb_sha256: str
    neural: Any
    dense_behavior: Any
    source_trial_count: int
    actual_pooling_provenance: Mapping[str, Any] | None = None
    rt_t4d_label_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        import numpy as np

        dataset, view = base.validate_scope(self.dataset, self.view)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "view", view)
        pattern = _SUBJECT_M_SOURCE_SESSION if dataset == "subject_m" else _RT_SESSION
        require(pattern.fullmatch(self.session_id) is not None, "materialized source session ID invalid")
        require(isinstance(self.source_path, str) and self.source_path, "materialized source path invalid")
        require(len(self.source_nwb_sha256) == 64 and all(c in "0123456789abcdef" for c in self.source_nwb_sha256),
                "materialized source NWB SHA invalid")
        neural = np.asarray(self.neural)
        behavior = np.asarray(self.dense_behavior)
        require(neural.ndim == 2 and neural.shape[0] > 0 and neural.shape[1] > 0, "source neural must be [rows, units]")
        require(behavior.ndim == 2 and behavior.shape[0] == neural.shape[0] and behavior.shape[1] == 2,
                "source dense behavior must be [same rows, 2]")
        require(neural.dtype == np.float32 and behavior.dtype == np.float32, "source arrays must be float32")
        require(np.isfinite(neural).all() and np.isfinite(behavior).all(), "source arrays contain non-finite values")
        require(isinstance(self.source_trial_count, int) and self.source_trial_count >= base.SUPPORT_BUDGET_TRIALS[dataset],
                "source session lacks canonical support budget")
        if dataset == "subject_m" and view == "pseudo_mua":
            require(isinstance(self.actual_pooling_provenance, Mapping), "pMUA source lacks actual pooling provenance")
            require(self.actual_pooling_provenance.get("replay_equal") is True, "pMUA replay did not equal canonical loader output")
        else:
            require(self.actual_pooling_provenance is None, "non-pMUA source may not carry pMUA provenance")

    def as_coverage_dict(self) -> dict[str, Any]:
        import numpy as np

        neural = np.asarray(self.neural)
        behavior = np.asarray(self.dense_behavior)
        unique_rows = int(np.unique(behavior, axis=0).shape[0])
        payload = {
            "session_id": self.session_id,
            "source_path": self.source_path,
            "source_nwb_sha256": self.source_nwb_sha256,
            "raw_observation_coverage": {
                "namespace": "raw_bin_index",
                "start_inclusive": 0,
                "stop_exclusive": int(neural.shape[0]),
                "row_count": int(neural.shape[0]),
                "complete_contiguous_source_fit_coverage": True,
            },
            "source_trial_count": self.source_trial_count,
            "neural_feature": {
                "dtype": "float32",
                "shape": [int(value) for value in neural.shape],
                "sha256": _array_sha256(neural),
            },
            "dense_behavior": {
                "dtype": "float32",
                "shape": [int(value) for value in behavior.shape],
                "sha256": _array_sha256(behavior),
                "row_count": int(behavior.shape[0]),
                "scalar_count": int(behavior.size),
                "unique_row_count": unique_rows,
                "semantics": "continuous_velocity_dense_bin_level__two_velocity_coordinates_per_row",
            },
            # These are *source-model training* inputs.  The source model sees
            # every raw source row/trial declared here.  Do not insert subject-M
            # target M30/M50 or RT target M24 comparison accounting into this
            # row: that would misleadingly recast full-source exposure as a
            # target-support budget.  The later target/query adapter owns those
            # non-interchangeable counts in a separate receipt.
            "source_training_exposure": {
                "source_model_uses_full_source_session_raw_rows": True,
                "source_model_neural_raw_row_count": int(neural.shape[0]),
                "source_model_dense_behavior_row_count": int(behavior.shape[0]),
                "source_model_dense_behavior_scalar_count": int(behavior.size),
                "source_model_dense_behavior_unique_row_count": unique_rows,
                "source_model_uses_full_source_trial_set": True,
                "source_model_trial_count": self.source_trial_count,
                "source_observation_exposure_vs_a2_t4": {
                    "cebra_source_fit_observation_semantics": "full_continuous_source_raw_bin_range",
                    "a2_t4_source_fit_observation_semantics": "rewarded_trial_result_filter_R_intervals_only",
                    "source_observation_exposure_matched": False,
                    "bias_direction": "favors_CEBRA_accuracy",
                    "disclosure": (
                        "full continuous CEBRA source bins can include non-rewarded and intertrial "
                        "observations that A2/T4 source fitting excludes"
                    ),
                    "rewarded_trial_only_cebra_source_sensitivity": {
                        "status": "PREDECLARED_NOT_IMPLEMENTED__NO_SILENT_MATCHED_CLAIM",
                        "requires_preserved_trial_boundaries_and_cross_boundary_positive_audit": True,
                        "may_not_concatenate_rewarded_trials_and_claim_matched_source_exposure": True,
                    },
                },
                "target_support_budget_trials_present_in_this_source_row": False,
                "t4_target_label_accounting_present_in_this_source_row": False,
            },
            "future_target_support_comparison": {
                "status": "NOT_MATERIALIZED__OWNED_BY_TARGET_QUERY_ADAPTER",
                "must_not_infer_target_M30_M50_or_M24_from_source_training_exposure": True,
                "required_future_receipt_fields": [
                    "target_support_trial_indices",
                    "target_query_trial_indices",
                    "t4_sparse_reference_label_event_count",
                    "t4_sparse_reference_label_row_count",
                    "t4_sparse_reference_label_scalar_count",
                    "t4_neural_support_trial_count",
                    "cebra_neural_support_trial_count",
                    "cebra_dense_label_support_trial_count",
                ],
            },
            "actual_pooling_provenance": dict(self.actual_pooling_provenance or {}),
            "rt_t4d_label_provenance": dict(self.rt_t4d_label_provenance or {}),
            "target_data_opened": False,
            "target_query_opened": False,
        }
        return payload | {"coverage_sha256": _sha_json(payload)}


def materialize_canonical_source_sessions(
    *, dataset: str, view: str | None, source_session_ids: Sequence[str], source_only_smoke: bool = False
) -> tuple[dict[str, Any], tuple[SourceSessionMaterialization, ...]]:
    """Open only canonical source sessions after static source-gate validation.

    This is intentionally the sole NWB-opening function in the Track-B v2
    route.  It accepts session IDs rather than paths and does not define an
    outer target, a target support/query block, a model, or a scorer.
    """
    request = canonical_source_only_adapter_spec(
        dataset, view, source_session_ids=source_session_ids, source_only_smoke=source_only_smoke,
    )
    dataset = request["dataset"]
    view = request["view"]
    ids = tuple(request["source_session_ids"])
    paths = tuple(_canonical_source_path(dataset, session_id) for session_id in ids)

    import numpy as np

    if dataset == "subject_m":
        sua_root = str(REPO_ROOT / "sua_exploration")
        # The canonical evaluator imports its sibling protocol as a historical
        # top-level module when launched as a script.  Supply exactly its own
        # package root and scripts directory; neither is a data-discovery path.
        evaluator_scripts = str(REPO_ROOT / "sua_exploration" / "scripts")
        for entry in (sua_root, evaluator_scripts):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        evaluator = importlib.import_module("scripts.eval_adaptation_dandi688")
        multi = importlib.import_module("mc_maze.multisession_datamodule")
        behavior_mean, behavior_std = multi.fit_behavior_stats(paths, bin_size_ms=20, cache_dir=None)
        rows: list[SourceSessionMaterialization] = []
        for session_id, path in zip(ids, paths, strict=True):
            record = evaluator.load_session_with_trials(
                path, 20, 50, 50, 100, -1.0, behavior_mean, behavior_std,
                signal_view=view,
            )
            require(record.get("name") == session_id and record.get("signal_view") == view,
                    "canonical subject-M loader returned a scope/session drift")
            pooling: dict[str, Any] | None = None
            if view == "pseudo_mua":
                sua_record = evaluator.load_session_with_trials(
                    path, 20, 50, 50, 100, -1.0, behavior_mean, behavior_std,
                    signal_view="sua",
                )
                with evaluator.NWBHDF5IO(str(path), "r") as io:
                    nwb = io.read()
                    electrode_ids = multi.electrode_ids_from_units(nwb.units.to_dataframe())
                replay, channel_ids = multi.pool_spikes_by_electrode(sua_record["neural"], electrode_ids)
                require(np.array_equal(replay, record["neural"]), "canonical pMUA loader/replay feature mismatch")
                require(np.array_equal(sua_record["behavior"], record["behavior"]), "pMUA changed dense behavior")
                pooling = {
                    "method": "electrode_ids_from_units_then_pool_spikes_by_electrode",
                    "input_sua_feature_sha256": _array_sha256(sua_record["neural"]),
                    "electrode_ids_sha256": _array_sha256(electrode_ids),
                    "channel_ids_sha256": _array_sha256(channel_ids),
                    "output_pseudo_mua_feature_sha256": _array_sha256(record["neural"]),
                    "input_dtype": str(sua_record["neural"].dtype),
                    "output_dtype": str(record["neural"].dtype),
                    "channel_order": "np_unique_ascending_electrode_id",
                    "exactly_one_electrode_per_sorted_unit": True,
                    "replay_equal": True,
                }
            rows.append(SourceSessionMaterialization(
                dataset=dataset, view=view, session_id=session_id, source_path=str(path.resolve()),
                source_nwb_sha256=_sha_file(path), neural=np.asarray(record["neural"], dtype=np.float32),
                dense_behavior=np.asarray(record["behavior"], dtype=np.float32),
                source_trial_count=len(record["trials"]), actual_pooling_provenance=pooling,
            ))
        # ``load_session_with_trials`` has already applied this exact transform
        # to ``record['behavior']``.  Bind its values (not only a SHA) so a
        # later target adapter can reproduce the *one* canonical stage from raw
        # target velocity.  A second source z-score on these final arrays would
        # be a hidden composed transform and is therefore forbidden below.
        stats_state = {
            "method": "canonical_fit_behavior_stats_componentwise_zscore",
            "input": "raw_binned_cursor_velocity_float32",
            "output": "load_session_with_trials_record_behavior__final_cebra_auxiliary",
            "fit_scope": (
                "strict27_subc_co_train_only" if not source_only_smoke
                else "declared_strict27_source_smoke_subset_only"
            ),
            "fit_session_ids": list(ids),
            "feature_dimension": 2,
            "mean_float32": np.asarray(behavior_mean, dtype=np.float32).tolist(),
            "std_float32": np.asarray(behavior_std, dtype=np.float32).tolist(),
            "mean_array_sha256": _array_sha256(behavior_mean),
            "std_array_sha256": _array_sha256(behavior_std),
            "parameter_count": 4,
        }
        request["canonical_loader_behavior_scaler"] = stats_state | {
            "state_sha256": _sha_json(stats_state),
        }
        return request, tuple(rows)

    sce_root = str(REPO_ROOT / "streaming_calibration_exp")
    sua_root = str(REPO_ROOT / "sua_exploration")
    for entry in (str(REPO_ROOT), sce_root, sua_root):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    rt_loader = importlib.import_module("src.data.rt_k4_loader")
    comparator = importlib.import_module("mc_maze.rt_classical_comparators")
    t4d_loader = importlib.import_module("src.data.rt_sparse_endpoint_loader")
    rows = []
    for session_id, path in zip(ids, paths, strict=True):
        record = rt_loader.load_rt_session(path)
        require(record.get("session_name") == session_id, "canonical RT loader returned a session drift")
        layout = comparator.rt_outer_window_layout(
            session_id, record["neural"], record["covariates"], record["trial_change"], record["eval_mask"],
        )
        t4d = t4d_loader.load_rt_sparse_endpoint_t4d_session(path)
        audit = t4d.get("t4d_audit")
        require(isinstance(audit, Mapping) and audit.get("m24_trials") == 24,
                "canonical RT T4d sparse-label provenance drift")
        rows.append(SourceSessionMaterialization(
            dataset=dataset, view=view, session_id=session_id, source_path=str(path.resolve()),
            source_nwb_sha256=_sha_file(path), neural=np.asarray(record["neural"], dtype=np.float32),
            dense_behavior=np.asarray(record["covariates"], dtype=np.float32),
            source_trial_count=int(layout.query_window_audit["total_trials"]),
            rt_t4d_label_provenance={
                "m24_trial_event_count": 24,
                "eligible_endpoint_reach_row_count": int(audit["eligible_reach_rows"]),
                "unique_endpoint_coordinate_scalar_count": int(audit["unique_endpoint_coordinate_scalars"]),
                "label_semantics": "endpoint_displacement_coordinates__derived_direction_for_T4d",
                "carrier_sha256_before_dense_target": audit["carrier_sha256_before_dense_target"],
                "carrier_unchanged_after_dense_target": audit["carrier_unchanged_after_dense_target"],
            },
        ))
    return request, tuple(rows)


def _zscore_authority(values_by_session: Mapping[str, Any], *, kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a dimension-stable pooled two-coordinate scaler on source data only."""
    import numpy as np

    ordered = tuple(sorted(values_by_session))
    require(ordered and all(np.asarray(values_by_session[item]).ndim == 2 for item in ordered),
            "source behavior scaler needs nonempty 2-D source arrays")
    arrays = tuple(np.asarray(values_by_session[item], dtype=np.float32) for item in ordered)
    feature_dim = arrays[0].shape[1]
    require(feature_dim == 2 and all(array.shape[1] == feature_dim for array in arrays),
            "behavior scaler requires exactly two dense velocity coordinates")
    stacked = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0.0),
            "source behavior scaler is non-finite or degenerate")
    transformed = {
        session: ((np.asarray(values_by_session[session], dtype=np.float64) - mean) / scale).astype(np.float32)
        for session in ordered
    }
    state = {
        "kind": kind,
        "method": "outer_source_pooled_componentwise_zscore",
        "fit_scope": "declared_source_sessions_only",
        "feature_dimension": 2,
        "mean_float64": mean.tolist(),
        "scale_float64": scale.tolist(),
        "parameter_count": 4,
        "source_session_ids": list(ordered),
    }
    return state | {"state_sha256": _sha_json(state)}, {
        session: {"sha256": _array_sha256(transformed[session]), "shape": list(transformed[session].shape)}
        for session in ordered
    }


def build_source_only_authority_bundle(
    *, request: Mapping[str, Any], sessions: Sequence[SourceSessionMaterialization],
    selector: base.SourceOnlySelectorSpec | None = None,
    outer_fold_lineage: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build unminted source-only authorities from actual source arrays.

    The returned payloads are ready for immutable O_EXCL publication, but their
    status deliberately says development source authority rather than official
    CEBRA execution.  They contain no target ID, target array, model, score,
    checkpoint, or CEBRA import.
    """
    import numpy as np

    require(request.get("status") == "SOURCE_ONLY_GATE_PASSED__STRICT_SOURCE_AUTHORITY__NO_TARGET_NO_CEBRA_NO_SCORE",
            "source authority requires canonical source-only gate")
    dataset, view = base.validate_scope(str(request.get("dataset")), request.get("view"))
    source_ids = _canonical_source_ids(dataset, request.get("source_session_ids", ()))
    require(tuple(row.session_id for row in sessions) == source_ids, "source materialization/session roster drift")
    require(all(row.dataset == dataset and row.view == view for row in sessions), "source materialization scope drift")
    require(request.get("target_session_id") is None and request.get("target_data_opened") is False,
            "source authority must not contain target access")
    selector = selector or base.SourceOnlySelectorSpec()

    fold_lineage: dict[str, Any] | None = None
    if outer_fold_lineage is not None:
        # The only supported fold-specific source authority is RT LOO.  Its
        # held-out session stays an opaque *lineage* name: it is never an
        # argument to canonical_source_only_adapter_spec or a source loader.
        require(dataset == "rt" and view is None,
                "outer-fold source lineage is currently defined only for canonical RT")
        candidate = dict(outer_fold_lineage)
        require(set(candidate) == {
            "outer_fold_id", "outer_fold_index", "opaque_held_out_target_session_id",
            "reference_per_fold_lineage_body_sha256", "reference_per_fold_lineage_ordered_digest",
        }, "RT outer-fold source lineage field set drift")
        require(isinstance(candidate["outer_fold_id"], str) and candidate["outer_fold_id"],
                "RT outer-fold source lineage ID invalid")
        require(isinstance(candidate["outer_fold_index"], int) and candidate["outer_fold_index"] in range(15),
                "RT outer-fold source lineage index invalid")
        require(_RT_SESSION.fullmatch(candidate["opaque_held_out_target_session_id"]) is not None,
                "RT held-out target identifier violates grammar")
        require(candidate["opaque_held_out_target_session_id"] not in source_ids,
                "RT source authority may not include its held-out target")
        require(all(isinstance(candidate[key], str) and len(candidate[key]) == 64
                    and all(char in "0123456789abcdef" for char in candidate[key])
                    for key in ("reference_per_fold_lineage_body_sha256", "reference_per_fold_lineage_ordered_digest")),
                "RT outer-fold reference lineage SHA malformed")
        fold_lineage = candidate | {
            "held_out_target_data_discovered": False,
            "held_out_target_data_opened": False,
            "held_out_target_passed_to_source_loader": False,
            "metric_authority": False,
        }

    coverage_rows = [row.as_coverage_dict() for row in sessions]
    roster_body = {
        "schema": SOURCE_ROSTER_SCHEMA,
        "status": "DEVELOPMENT_SOURCE_ONLY__NOT_CEBRA_EXECUTION__NOT_CITABLE",
        "dataset": dataset,
        "view": view,
        "adapter_id": request["adapter"]["adapter_id"],
        "source_session_ids": list(source_ids),
        "source_nwb_sha256_by_session": {row.session_id: row.source_nwb_sha256 for row in sessions},
        "loader_semantics_sha256": request["loader_semantics_sha256"],
        "outer_fold_lineage": fold_lineage,
        "target_session_id": None,
        "target_data_opened": False,
        "target_query_opened": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    roster_sha = _sha_json(roster_body)
    roster_body["receipt_payload_sha256"] = roster_sha
    coverage_body = {
        "schema": SOURCE_COVERAGE_SCHEMA,
        "status": "DEVELOPMENT_SOURCE_ONLY__FULL_SOURCE_RAW_COVERAGE__NOT_CEBRA_EXECUTION",
        "dataset": dataset,
        "view": view,
        "adapter_id": request["adapter"]["adapter_id"],
        "source_roster_receipt_payload_sha256": roster_sha,
        "source_sessions": coverage_rows,
        "all_source_arrays_float32": True,
        "target_support_or_query_present": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "auxiliary": "continuous_velocity_dense_bin_level",
        "information_matched": False,
        "label_information_bias_direction": "favors_CEBRA_accuracy",
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    coverage_sha = _sha_json(coverage_body)
    coverage_body["receipt_payload_sha256"] = coverage_sha

    # A source-channel vector cannot transfer to an unseen target whose unit
    # width differs.  Freeze the only valid headline preprocessing form now.
    neural_body = {
        "schema": SOURCE_NEURAL_AUTHORITY_SCHEMA,
        "status": "DEVELOPMENT_SOURCE_ONLY__DIMENSION_AGNOSTIC_IDENTITY__NOT_CEBRA_EXECUTION",
        "dataset": dataset,
        "view": view,
        "source_roster_receipt_payload_sha256": roster_sha,
        "source_coverage_receipt_payload_sha256": coverage_sha,
        "method": "dimension_agnostic_identity_float32_binned_counts",
        "parameter_count": 0,
        "source_channel_vector_or_prefix_mapping_permitted": False,
        "target_support_neural_standardization_permitted_in_headline": False,
        "target_support_neural_standardization_requires_separate_sensitivity_receipt": True,
        "source_feature_sha256_by_session": {
            row["session_id"]: row["neural_feature"]["sha256"] for row in coverage_rows
        },
        "target_data_used": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "target_query_used": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    neural_sha = _sha_json(neural_body)
    neural_body["receipt_payload_sha256"] = neural_sha

    behavior_values = {row.session_id: row.dense_behavior for row in sessions}
    if dataset == "subject_m":
        behavior_state = request.get("canonical_loader_behavior_scaler")
        require(isinstance(behavior_state, Mapping), "subject-M source authority lacks canonical loader behavior scaler")
        behavior_state = dict(behavior_state)
        require(
            behavior_state.get("method") == "canonical_fit_behavior_stats_componentwise_zscore"
            and behavior_state.get("output") == "load_session_with_trials_record_behavior__final_cebra_auxiliary"
            and behavior_state.get("fit_scope") == "strict27_subc_co_train_only"
            and behavior_state.get("fit_session_ids") == list(source_ids)
            and behavior_state.get("feature_dimension") == 2
            and behavior_state.get("parameter_count") == 4,
            "subject-M canonical behavior scaler scope or semantics drift",
        )
        mean = behavior_state.get("mean_float32")
        std = behavior_state.get("std_float32")
        require(isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 2,
                "subject-M canonical behavior scaler values malformed")
        require(
            behavior_state.get("mean_array_sha256") == _array_sha256(np.asarray(mean, dtype=np.float32))
            and behavior_state.get("std_array_sha256") == _array_sha256(np.asarray(std, dtype=np.float32)),
            "subject-M canonical behavior scaler value SHA drift",
        )
        state_without_sha = {key: value for key, value in behavior_state.items() if key != "state_sha256"}
        require(behavior_state.get("state_sha256") == _sha_json(state_without_sha),
                "subject-M canonical behavior scaler state SHA drift")
        behavior_outputs = {
            session: {"sha256": _array_sha256(values), "shape": list(values.shape)}
            for session, values in sorted(behavior_values.items())
        }
        behavior_method = "canonical_loader_final_auxiliary__no_second_stage_refit"
    else:
        behavior_state, behavior_outputs = _zscore_authority(
            behavior_values, kind="rt_source_fitted_behavior_auxiliary_scaler",
        )
        behavior_method = "rt_source_pooled_componentwise_zscore_from_native_loader_velocity"
    behavior_body = {
        "schema": SOURCE_BEHAVIOR_AUTHORITY_SCHEMA,
        "status": "DEVELOPMENT_SOURCE_ONLY__DENSE_BEHAVIOR_SCALER__NOT_CEBRA_EXECUTION",
        "dataset": dataset,
        "view": view,
        "source_roster_receipt_payload_sha256": roster_sha,
        "source_coverage_receipt_payload_sha256": coverage_sha,
        "source_neural_input_authority_payload_sha256": neural_sha,
        "behavior_auxiliary_scaler": behavior_state,
        "behavior_auxiliary_method": behavior_method,
        "final_source_behavior_sha256_by_session": behavior_outputs,
        "second_behavior_refit_on_canonical_final_auxiliary_permitted": False,
        "target_support_behavior_auxiliary_in_scaler_fit": False,
        "target_query_behavior_auxiliary_in_scaler_fit": False,
        "target_data_used": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "target_query_used": False,
        "auxiliary": "continuous_velocity_dense_bin_level",
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    behavior_sha = _sha_json(behavior_body)
    behavior_body["receipt_payload_sha256"] = behavior_sha

    embedding_identity = {
        "method": "explicit_identity_embedding_standardizer",
        "fit_scope": "outer_source_embeddings_only",
        "parameter_count": 0,
        "embedding_dimension": "selected_geometry_dimension__not_yet_executed",
        "target_support_embedding_in_standardizer_fit": False,
        "target_query_embedding_in_standardizer_fit": False,
    }
    embedding_body = {
        "schema": SOURCE_EMBEDDING_AUTHORITY_SCHEMA,
        "status": "DEVELOPMENT_SOURCE_ONLY__EXPLICIT_IDENTITY__NOT_CEBRA_EXECUTION",
        "dataset": dataset,
        "view": view,
        "readout_route": "source_only_consumer_mechanism_alignment",
        "source_roster_receipt_payload_sha256": roster_sha,
        "source_neural_input_authority_payload_sha256": neural_sha,
        "source_behavior_auxiliary_scaler_authority_payload_sha256": behavior_sha,
        "embedding_standardizer": embedding_identity | {"state_sha256": _sha_json(embedding_identity)},
        "target_data_used": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "target_query_used": False,
        "target_support_only_standard_cebra_accuracy_authority": "LIVE_TARGET_BLOCKER__MUST_BE_SEPARATE",
        "source_plus_target_support_hybrid_sensitivity_authority": "LIVE_TARGET_BLOCKER__MUST_BE_SEPARATE",
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    embedding_sha = _sha_json(embedding_body)
    embedding_body["receipt_payload_sha256"] = embedding_sha

    # MultiSessionSolver has one fitted encoder per joint-fit session.  A
    # classic leave-one-session-out fit cannot transform the held session at
    # all, so it is not a valid deployment simulation.  Each inner fold instead
    # turns one *source* session into a pseudo-target: fit the other source
    # sessions plus exactly that held session's chronological M support block,
    # then evaluate only its after-M query.  This is source-only selection; it
    # never reads an outer subject-M/RT target session.
    inner_folds = [
        {
            "inner_fold_id": f"source_pseudo_target_support_query__{held}",
            "pseudo_target_source_session_id": held,
            "peer_source_session_ids_in_joint_fit": [item for item in source_ids if item != held],
            "joint_fit_session_ids": list(source_ids),
            "joint_fit_session_input_blocks": {
                "peer_source_sessions": "FULL_CONTINUOUS_SOURCE_SESSION_ROWS",
                "held_source_pseudo_target_session": "CHRONOLOGICAL_M_SUPPORT_PREFIX_ONLY",
            },
            "pseudo_target_support": {
                "used_in_joint_fit": True,
                "support_budget_trials": base.SUPPORT_BUDGET_TRIALS[dataset],
                "trial_prefix_semantics": (
                    "chronological_first_50_rewarded_trials" if dataset == "subject_m"
                    else "chronological_first_24_trials__M24"
                ),
                "cebra_input_sequence": {
                    "sequence_semantics": "one_continuous_chronological_prefix",
                    "start": "CANONICAL_SOURCE_RECORD_START_RAW_BIN",
                    "stop": (
                        "STOP_OF_REWARDED_TRIAL_50" if dataset == "subject_m"
                        else "STOP_OF_CHRONOLOGICAL_TRIAL_24"
                    ),
                    "all_intervening_raw_rows_included": True,
                    "rewarded_segments_concatenated": False,
                    "boundary_matched": True,
                    "trial_bin_or_neural_exposure_matched": False,
                    "bias_direction": "favors_CEBRA_accuracy",
                    "required_authorities_before_selector_execution": [
                        "pseudo_target_support_prefix_start_stop_authority_sha256",
                        "pseudo_target_support_prefix_raw_coverage_expansion_sha256",
                    ],
                },
                "neural_in_joint_fit": True,
                "dense_velocity_labels_in_joint_fit": True,
                "required_authorities_before_selector_execution": [
                    "pseudo_target_support_raw_coverage_index_sha256",
                    "pseudo_target_support_trial_boundary_authority_sha256",
                    "pseudo_target_support_feature_sha256",
                ],
            },
            "pseudo_target_query": {
                "trial_semantics": (
                    "rewarded_trials_strictly_after_M50" if dataset == "subject_m"
                    else "sealed_RT_outer_q24_eligible_full_windows_only"
                ),
                "neural_in_joint_fit": False,
                "dense_velocity_labels_in_joint_fit": False,
                "is_only_inner_score_block": True,
                "required_authorities_before_selector_execution": [
                    "pseudo_target_query_raw_coverage_index_sha256",
                    "pseudo_target_query_receptive_field_alignment_authority_sha256",
                    "pseudo_target_ordered_prediction_target_raw_bin_indices_sha256",
                ],
            },
            "never_transform_unfitted_held_session": True,
            "adapt_true_permitted": False,
            "outer_target_data_used": False,
        }
        for held in source_ids
    ]
    selector_body = {
        "schema": SOURCE_SELECTOR_PLAN_SCHEMA,
        "status": "SOURCE_ONLY_GEOMETRY_SELECTION_REQUIRED__NOT_EXECUTED__NO_CEBRA",
        "dataset": dataset,
        "view": view,
        "source_roster_receipt_payload_sha256": roster_sha,
        "source_coverage_receipt_payload_sha256": coverage_sha,
        "source_neural_input_authority_payload_sha256": neural_sha,
        "source_behavior_auxiliary_scaler_authority_payload_sha256": behavior_sha,
        "source_readout_embedding_identity_authority_payload_sha256": embedding_sha,
        "inner_source_folds": inner_folds,
        "inner_selector_deployment_simulation": "joint_fit_peers_plus_held_source_pseudo_target_support__score_held_post_M_query_only",
        "selector": selector.as_dict(),
        "linear_ridge_selection": {
            "candidate_count": len(selector.linear_candidates()),
            "metric": "source_inner_query_linear_ridge_pooled_r2",
            "target_data_used": False,
        },
        "knn_cosine_k3_selection": {
            "candidate_count": len(selector.knn_candidates()),
            "metric": "source_inner_query_cosine_knn_k3_pooled_r2",
            "normalized_lambda": "NOT_APPLICABLE__MUST_NOT_AFFECT_KNN_SELECTION",
            "target_data_used": False,
        },
        "separate_geometry_selection_required": True,
        "target_data_opened": False,
        "target_query_opened": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    selector_sha = _sha_json(selector_body)
    selector_body["receipt_payload_sha256"] = selector_sha
    return {
        "source_roster": roster_body,
        "source_coverage": coverage_body,
        "source_neural_input_authority": neural_body,
        "source_behavior_auxiliary_scaler_authority": behavior_body,
        "source_readout_embedding_identity_authority": embedding_body,
        "source_only_dual_geometry_selection_plan": selector_body,
    }


def build_rt_15fold_source_authority_plan() -> dict[str, Any]:
    """Bind RT's real 15-fold LOO source rosters without opening any NWB file.

    This is intentionally stronger than the historic two-source smoke: all
    15 outer folds are derived from the sealed RT reference lineage, and each
    fold has exactly 14 allowed source sessions.  The held session remains an
    opaque field in the plan; the canonical source-loader request has no
    target argument and is built solely from the 14 source IDs.

    The function is plan-only.  A later explicitly authorised source-only
    materialization may call :func:`build_rt_outer_fold_source_authority_bundle`
    with real *source* arrays, but this plan never opens them itself.
    """
    topology = metric_pointer.build_rt_outer_fold_lineage_plan()
    metric_pointer.validate_rt_outer_fold_lineage_plan(topology)
    semantics = live.build_sealed_loader_semantics_contract("rt", None)
    fold_plans: list[dict[str, Any]] = []
    for item in topology["outer_folds"]:
        source_ids = tuple(item["source_session_ids"])
        request = canonical_source_only_adapter_spec("rt", None, source_session_ids=source_ids)
        require(request.get("target_session_id") is None and request.get("target_data_opened") is False,
                "RT full source plan attempted target routing")
        require(request["loader_semantics_sha256"] == _sha_json(semantics),
                "RT source request loader semantics drift")
        fold_plans.append({
            "outer_fold_id": item["outer_fold_id"],
            "outer_fold_index": item["outer_fold_index"],
            "opaque_held_out_target_session_id": item["opaque_held_out_target_session_id"],
            "source_session_ids": list(source_ids),
            "source_session_count": len(source_ids),
            "source_session_roster_sha256": _sha_json(list(source_ids)),
            "source_loader_request_sha256": _sha_json(request),
            "target_support_budget_trials": item["target_support_budget_trials"],
            "target_support_semantics": item["target_support_semantics"],
            "target_query_semantics": item["target_query_semantics"],
            "standard_cebra_support_sequence": dict(item["standard_cebra_support_sequence"]),
            "held_out_target_data_discovered": False,
            "held_out_target_data_opened": False,
            "held_out_target_passed_to_source_loader": False,
            "source_materialization_status": "NOT_RUN__EXPLICIT_SOURCE_ONLY_AUTHORIZATION_REQUIRED",
            "source_authority_bundle_status": "NOT_BUILT__ONE_DEVELOPMENT_BUNDLE_REQUIRED_PER_OUTER_FOLD",
        })
    payload = {
        "schema": RT_15FOLD_SOURCE_AUTHORITY_PLAN_SCHEMA,
        "status": "RT_FULL_15FOLD_SOURCE_AUTHORITY_PLAN_ONLY__NO_TARGET_OPEN_NO_CEBRA_NO_SCORE",
        "dataset": "rt",
        "view": None,
        "rt_outer_fold_lineage_plan_sha256": topology["rt_outer_fold_lineage_plan_sha256"],
        "reference_per_fold_lineage_body_sha256": topology["reference_per_fold_lineage_body_sha256"],
        "reference_per_fold_lineage_ordered_digest": topology["reference_per_fold_lineage_ordered_digest"],
        "loader_semantics": semantics,
        "loader_semantics_sha256": _sha_json(semantics),
        "outer_fold_count": 15,
        "outer_folds": fold_plans,
        "all_outer_folds_require_distinct_per_fold_source_authority_bundles": True,
        "target_data_discovered": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_receipt_minted": False,
    }
    return payload | {"rt_15fold_source_authority_plan_sha256": _sha_json(payload)}


def validate_rt_15fold_source_authority_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on a tampered or stale RT 15-fold source topology."""
    require(isinstance(payload, Mapping), "RT 15-fold source authority plan must be a mapping")
    expected = build_rt_15fold_source_authority_plan()
    require(dict(payload) == expected, "RT 15-fold source authority plan differs from sealed-lineage plan")
    return expected


def build_rt_outer_fold_source_authority_bundle(
    *,
    rt_15fold_plan: Mapping[str, Any],
    outer_fold_id: str,
    request: Mapping[str, Any],
    sessions: Sequence[SourceSessionMaterialization],
    selector: base.SourceOnlySelectorSpec | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one RT fold's source-only bundle while keeping its target opaque.

    This does not discover or open a target.  Every supplied materialization
    must be one of the exact 14 source sessions selected by the independently
    verified plan; the held-out target ID is only copied into a lineage field
    of the source roster receipt.
    """
    plan = validate_rt_15fold_source_authority_plan(rt_15fold_plan)
    require(isinstance(outer_fold_id, str) and outer_fold_id, "RT outer fold ID invalid")
    matches = [item for item in plan["outer_folds"] if item["outer_fold_id"] == outer_fold_id]
    require(len(matches) == 1, "RT outer fold ID missing or ambiguous in 15-fold plan")
    fold = matches[0]
    source_ids = tuple(fold["source_session_ids"])
    require(tuple(request.get("source_session_ids", ())) == source_ids,
            "RT outer-fold request source roster differs from sealed plan")
    require(request.get("dataset") == "rt" and request.get("view") is None,
            "RT outer-fold request scope drift")
    require(request.get("target_session_id") is None and request.get("target_data_opened") is False,
            "RT outer-fold bundle request contains target access")
    require(tuple(row.session_id for row in sessions) == source_ids,
            "RT outer-fold materialization roster differs from sealed plan")
    require(fold["opaque_held_out_target_session_id"] not in {row.session_id for row in sessions},
            "RT held-out target may not be materialized by its source-only fold")
    return build_source_only_authority_bundle(
        request=request,
        sessions=sessions,
        selector=selector,
        outer_fold_lineage={
            "outer_fold_id": fold["outer_fold_id"],
            "outer_fold_index": fold["outer_fold_index"],
            "opaque_held_out_target_session_id": fold["opaque_held_out_target_session_id"],
            "reference_per_fold_lineage_body_sha256": plan["reference_per_fold_lineage_body_sha256"],
            "reference_per_fold_lineage_ordered_digest": plan["reference_per_fold_lineage_ordered_digest"],
        },
    )


def write_development_source_only_authority_bundle(
    *, output_dir: Path, bundle: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    """Publish independent source-only authority pairs without minting an execution receipt."""
    names = {
        "source_roster": "source_roster.json",
        "source_coverage": "source_coverage.json",
        "source_neural_input_authority": "source_neural_input_authority.json",
        "source_behavior_auxiliary_scaler_authority": "source_behavior_auxiliary_scaler_authority.json",
        "source_readout_embedding_identity_authority": "source_readout_embedding_identity_authority.json",
        "source_only_dual_geometry_selection_plan": "source_only_dual_geometry_selection_plan.json",
    }
    require(set(bundle) == set(names), "source authority bundle member set drift")
    output_dir = Path(output_dir)
    require(not output_dir.exists() and not output_dir.is_symlink(), "refusing to reuse source authority output directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    receipts: dict[str, dict[str, str]] = {}
    try:
        for key, filename in names.items():
            receipts[key] = base.write_immutable_receipt(output_dir / filename, dict(bundle[key]))
    except BaseException:
        # Preserve every successfully sealed record for inspection rather than
        # deleting an immutable authority after a partial publication failure.
        raise
    return receipts


def canonical_metric_pointer_mint_proposal(dataset: str, view: str | None = None) -> dict[str, Any]:
    """Read known sidecarless baseline bytes and propose, but never mint, a pointer.

    A root-audited immutable pointer remains mandatory before a result can use
    these values as authority.  This source-only function deliberately does
    not create that pointer, does not add sidecars to legacy files, and makes
    clear which bodies carry aggregate summaries versus per-session/seed or
    RT query lineage.
    """
    dataset, view = base.validate_scope(dataset, view)
    if dataset == "subject_m":
        summary = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json"
        cell = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805/aggregate/endpoint_aggregate_torchmetrics151.json"
        metric_pointer = f"/summary/{'sua' if view == 'sua' else 'pseudo_mua'}/50/mean_r2"
        exact = {
            "summary_cross_check_body": {
                "path": str(summary.resolve()), "sha256": "12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2",
                "schema": "dandi_000688_subm_v9_t4_label_budget_torchmetrics151_v1",
                "metric_json_pointer": metric_pointer,
                "authority_scope": "aggregate_M50_summary_cross_check_only__cells_do_not_contain_M50",
            },
            "per_session_seed_body": {
                "path": str(cell.resolve()), "sha256": "8a5ba373169cc28237666917f21fc893003afcc9ca486aa8f4b78d3f65aca7f4",
                "schema": "dandi_000688_subm_co_v9_torchmetrics151_authoritative_v1",
                "required_filter": {"arm": "shared_t4", "view": view},
                "authority_scope": "M50_15_sessions_x_3_seeds__paired_distribution_lineage",
            },
        }
    else:
        aggregate = REPO_ROOT / "sua_exploration/comparators/receipts/rt_classical_comparators/rt_classical_comparators_receipt.json"
        per_fold = (
            REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical"
            / "rt_sparse_t4d_b2_forward_reeval_v2_20260811"
            / "RT_T4D_VS_B2_D1024_FORWARD_ONLY_15FOLD_FINAL_v1.json"
        )
        stage2 = REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/STAGE2_MATRIX_AGGREGATE_v1.json"
        exact = {
            "aggregate_metric_query_identity_body": {
                "path": str(aggregate.resolve()), "sha256": "c51cb0ff7dadd3c40ca7861dea92d80f3457c709801ea8ec7ba3e91b9e52b042",
                "schema": "rt_classical_comparators_v1", "metric_json_pointer": "/results/arms/t4d_reference/mean",
                "authority_scope": "absolute_T4d_mean_and_ordered_query_identity",
            },
            "per_fold_t4d_body": {
                "path": str(per_fold.resolve()), "sha256": "c36ec0e31ed913ed4e8077f9a4d9d634d53529ce037ad06af1f48d279b16820e",
                "required_json_pointer": "/rows/*/t4d_r2",
                "authority_scope": "15_fold_T4d_per_session_lineage",
            },
            "stage2_delta_companion": {
                "path": str(stage2.resolve()), "sha256": "bb2806953e979180c408fb55744534be6fa470d4144f210cc50917a9b1006b7d",
                "schema": "rt_sparse_endpoint_stage2_matrix_aggregate_v1",
                "authority_scope": "terminal_T4d_minus_full_and_T4d_minus_zero4_diagnostics_only__not_absolute_T4d_metric",
            },
        }
    verified_raw: dict[str, bytes] = {}
    for name, item in exact.items():
        path = Path(item["path"])
        raw = live._read_immutable_regular_same_fd(path, label=f"known {name}")
        require(hashlib.sha256(raw).hexdigest() == item["sha256"],
                f"known baseline body SHA drift: {name}")
        require(not os.path.lexists(path.with_name(f"{path.name}.sha256")),
                f"legacy sidecarless baseline unexpectedly has a sidecar: {name}")
        verified_raw[name] = raw

    # Resolve every declared metric pointer *after* its body has passed the
    # same-fd SHA/mode/topology checks.  The values in the return payload are
    # derived from those bytes, never accepted as rounded configuration input.
    for name, item in exact.items():
        pointer_key = "metric_json_pointer" if "metric_json_pointer" in item else "required_json_pointer"
        if pointer_key in item:
            values = _json_pointer_values_from_verified_bytes(
                verified_raw[name], str(item[pointer_key]), label=f"known {name}",
            )
            item["resolved_metric_values_from_verified_bytes"] = list(values)
            item["metric_pointer_verified_against_same_fd_bytes"] = True

    if dataset == "rt":
        aggregate_values = exact["aggregate_metric_query_identity_body"]["resolved_metric_values_from_verified_bytes"]
        fold_values = exact["per_fold_t4d_body"]["resolved_metric_values_from_verified_bytes"]
        require(len(aggregate_values) == 1 and len(fold_values) == 15,
                "RT metric pointer cardinality drift")
        per_fold_mean = sum(float(value) for value in fold_values) / len(fold_values)
        require(aggregate_values[0] == per_fold_mean,
                "RT aggregate T4d metric no longer equals exact per-fold T4d mean")
        exact["aggregate_metric_query_identity_body"]["exact_per_fold_t4d_mean_from_verified_bytes"] = per_fold_mean
        exact["aggregate_metric_query_identity_body"]["equals_per_fold_t4d_mean"] = True
    return {
        "schema": REFERENCE_POINTER_PROPOSAL_SCHEMA,
        "status": "ROOT_AUDITED_IMMUTABLE_POINTER_REQUIRED__NOT_MINTED__NOT_AUTHORITY",
        "dataset": dataset,
        "view": view,
        "bodies": exact,
        "rounded_literals_accepted": False,
        "legacy_bodies_modified": False,
        "pointer_mint_owner": "root_independent_audit",
        "target_data_opened": False,
        "gpu_used": False,
        "score_emitted": False,
    }
