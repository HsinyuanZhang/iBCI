"""Fail-closed live-integration contracts for the H1-excluded Track-B v2 route.

This additive module is deliberately an interface layer, not a data runner.  It
does not import an NWB loader, CEBRA, the historical comparator, NumPy, Torch,
or a checkpoint.  Later authorised code must pass these contracts *before*
discovering subject-M/RT data and before an actual CEBRA CPU control or fold
can be executed.

The module binds the established loader/query semantics by module path, symbol,
and source-file SHA.  It never accepts a raw data path, and validates scope
before inspecting a proposed discovery object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence

import track_b_v2_contract as base


LIVE_CONTRACT_SCHEMA = "track_b_v2_live_contract_v1"
LOADER_SEMANTICS_SCHEMA = "track_b_v2_sealed_loader_semantics_v1"
INDEX_MANIFEST_SCHEMA = "track_b_v2_source_support_query_index_manifest_v1"
SOURCE_NEURAL_INPUT_PREPROCESSOR_SCHEMA = "track_b_v2_source_only_neural_input_preprocessor_receipt_v1"
INNER_SELECTOR_SCHEMA = "track_b_v2_inner_fold_selector_receipt_v1"
MULTISESSION_PROVENANCE_SCHEMA = "track_b_v2_multisession_provenance_plan_v1"
CPU_CONTROL_RUNNER_SCHEMA = "track_b_v2_actual_cebra_cpu_control_runner_v1"
SEALED_REFERENCE_AUTHORITY_SCHEMA = "track_b_v2_reference_authority_v2"
SEALED_BODY_POINTER_SCHEMA = "track_b_v2_sealed_body_pointer_authority_v1"
LIVE_FOLD_REFERENCE_BINDING_SCHEMA = "track_b_v2_live_fold_reference_binding_v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
MULTISESSION_SOLVER = "cebra.solver.MultiSessionSolver"
LIVE_ARMS = ("cebra_joint_behavior", "cebra_frozen_source_adapt", "cebra_adapt_unaligned")
MODEL_ARM_ROLES: dict[str, dict[str, Any]] = {
    "cebra_joint_behavior": {
        "role": "standard_supported_cebra_multisession_accuracy_model_arm",
        "accuracy_table_model_arm": True,
        "mandatory_report": True,
        "may_be_substituted_by_outcome": False,
    },
    "cebra_frozen_source_adapt": {
        "role": "vendored_deployment_oriented_sensitivity_mechanism_model_arm",
        "accuracy_table_model_arm": False,
        "mandatory_report": True,
        "may_be_substituted_by_outcome": False,
    },
    "cebra_adapt_unaligned": {
        "role": "diagnostic_distribution_only_negative_control_model_arm",
        "accuracy_table_model_arm": False,
        "mandatory_report": True,
        "may_be_substituted_by_outcome": False,
    },
}
SUBJECT_M_T4_LABEL_SEMANTICS = "one_trial_direction_angle_annotation__cos_sin_is_derived_not_two_annotations"
RT_T4D_LABEL_SEMANTICS = "chronological_M24_trial_events__endpoint_displacement_coordinates_and_derived_direction_rows_are_distinct"


class TrackBV2LiveContractError(base.TrackBV2ContractError):
    """Raised before a live integration could discover data or execute CEBRA."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2LiveContractError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _read_immutable_regular_same_fd(path: Path, *, label: str) -> bytes:
    """Read a sealed file through one non-following fd and detect path swaps.

    The old implementation did ``lstat(); read_bytes()``.  An attacker (or a
    concurrent writer) could replace the pathname between those operations,
    making the checked inode differ from the hashed/parsed bytes.  The digest,
    JSON parse, and mode check below all refer to the *same* opened inode.  A
    post-read pathname identity check then rejects an open-then-rename attack
    before the caller may consume the bytes.
    """
    path = Path(path)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required for sealed authority reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrackBV2LiveContractError(f"cannot open {label} without following links: {path}") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        require(stat.S_IMODE(before.st_mode) == 0o444, f"{label} must be mode 0444")
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
        raise TrackBV2LiveContractError(f"{label} disappeared or was replaced after read: {path}") from exc
    require(not stat.S_ISLNK(named.st_mode) and stat.S_ISREG(named.st_mode),
            f"{label} pathname must remain a regular non-symlink")
    require(
        (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        == (named.st_dev, named.st_ino, named.st_mode, named.st_size),
        f"{label} pathname inode changed after read",
    )
    return raw


def _canonical_ids(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    ids = tuple(str(value) for value in values)
    require(ids, f"{label} must not be empty")
    require(all(value and value == value.strip() for value in ids), f"{label} has an empty or padded ID")
    require(len(set(ids)) == len(ids), f"{label} has duplicate IDs")
    return tuple(sorted(ids))


# These are literal bindings to the existing, sealed loader/query semantics.
# They are data-agnostic source paths; no module is imported merely to build a
# contract.  The later source gate must SHA-check each binding before discovery.
_SEMANTICS: dict[tuple[str, str | None], dict[str, Any]] = {
    ("subject_m", "sua"): {
        "adapter_id": "subject_m_dandi688_sua_m50_v1",
        "signal_view": "sua",
        "auxiliary": "continuous_velocity_dense_bin_level",
        "support_semantics": "chronological_first_50_rewarded_trials",
        "query_semantics": "rewarded_trials_after_m50_only",
        "normalizer_semantics": "outer_source_sessions_only__no_target_refit",
        "loader_bindings": {
            "session_identity": {
                "path": "sua_exploration/mc_maze/multisession_datamodule.py",
                "symbol": "session_name_from_path",
            },
            "record_loader": {
                "path": "sua_exploration/scripts/eval_adaptation_dandi688.py",
                "symbol": "load_session_with_trials",
            },
            "calibration_extractor": {
                "path": "sua_exploration/scripts/eval_adaptation_dandi688.py",
                "symbol": "build_calib_trials_for_indices",
            },
            "source_behavior_stats": {
                "path": "sua_exploration/mc_maze/multisession_datamodule.py",
                "symbol": "fit_behavior_stats",
            },
        },
    },
    ("subject_m", "pseudo_mua"): {
        "adapter_id": "subject_m_dandi688_pseudo_mua_m50_v1",
        "signal_view": "pseudo_mua",
        "auxiliary": "continuous_velocity_dense_bin_level",
        "support_semantics": "chronological_first_50_rewarded_trials",
        "query_semantics": "rewarded_trials_after_m50_only",
        "normalizer_semantics": "outer_source_sessions_only__no_target_refit",
        # A pMUA live adapter is not allowed to obtain this view by merely
        # changing a string.  It must bind this exact construction symbol and
        # later record the resulting feature SHA for every source/target fold.
        "feature_construction": "pool_spikes_by_electrode__canonical_subject_m_pseudo_mua",
        "feature_construction_parameters": {
            "input": "float32_binned_sorted_unit_spike_counts",
            "electrode_assignment": "electrode_ids_from_units__exactly_one_electrode_per_sorted_unit",
            "aggregation": "sum_sorted_unit_counts_per_electrode",
            "channel_order": "np_unique_ascending_electrode_id",
            "output_dtype": "float32",
        },
        "loader_bindings": {
            "session_identity": {
                "path": "sua_exploration/mc_maze/multisession_datamodule.py",
                "symbol": "session_name_from_path",
            },
            "record_loader": {
                "path": "sua_exploration/scripts/eval_adaptation_dandi688.py",
                "symbol": "load_session_with_trials",
            },
            "calibration_extractor": {
                "path": "sua_exploration/scripts/eval_adaptation_dandi688.py",
                "symbol": "build_calib_trials_for_indices",
            },
            "source_behavior_stats": {
                "path": "sua_exploration/mc_maze/multisession_datamodule.py",
                "symbol": "fit_behavior_stats",
            },
            "pseudo_mua_constructor": {
                "path": "sua_exploration/mc_maze/multisession_datamodule.py",
                "symbol": "pool_spikes_by_electrode",
            },
            "pseudo_mua_electrode_assignment": {
                "path": "sua_exploration/mc_maze/multisession_datamodule.py",
                "symbol": "electrode_ids_from_units",
            },
        },
    },
    ("rt", None): {
        "adapter_id": "rt_dandi688_subc_sua_m24_q24_v1",
        "signal_view": "sua",
        "auxiliary": "continuous_velocity_dense_bin_level",
        "support_semantics": "chronological_first_24_trials__M24",
        "query_semantics": "sealed_rt_outer_q24_eligible_full_windows_only",
        "normalizer_semantics": "outer_source_sessions_only__no_target_refit",
        "loader_bindings": {
            "session_discovery": {
                "path": "streaming_calibration_exp/src/data/rt_k4_loader.py",
                "symbol": "find_rt_sessions",
            },
            "record_loader": {
                "path": "streaming_calibration_exp/src/data/rt_k4_loader.py",
                "symbol": "load_rt_session",
            },
            "session_identity": {
                "path": "sua_exploration/mc_maze/rt_classical_comparators.py",
                "symbol": "session_name_from_nwb_path",
            },
            "query_layout": {
                "path": "sua_exploration/mc_maze/rt_classical_comparators.py",
                "symbol": "rt_outer_window_layout",
            },
            "t4d_sparse_label_loader": {
                "path": "streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py",
                "symbol": "load_rt_sparse_endpoint_t4d_session",
            },
        },
    },
}


def canonical_adapter_spec(
    dataset: str,
    view: str | None = None,
    *,
    proposed_data_path: object | None = None,
    proposed_discovery: object | None = None,
) -> dict[str, Any]:
    """Return a legal adapter only after scope rejection, never discovering data.

    The two proposed-object arguments deliberately have type ``object``.  The
    function validates the H1/M2/undeclared scope first and never coerces an
    illegal path through ``Path``, ``str`` or ``__fspath__``.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(proposed_data_path is None, "Track-B v2 adapter accepts no arbitrary data path before source gate")
    require(proposed_discovery is None, "Track-B v2 adapter accepts no arbitrary discovery callable")
    spec = _SEMANTICS[(dataset, view)]
    return {
        "schema": LIVE_CONTRACT_SCHEMA,
        "dataset": dataset,
        "view": view,
        "adapter_id": spec["adapter_id"],
        "signal_view": spec["signal_view"],
        "carrier_equivalent_support_budget_trials": base.SUPPORT_BUDGET_TRIALS[dataset],
        "support_semantics": spec["support_semantics"],
        "query_semantics": spec["query_semantics"],
        "normalizer_semantics": spec["normalizer_semantics"],
        "auxiliary": spec["auxiliary"],
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
        "loader_bindings_required": spec["loader_bindings"],
        "feature_construction": spec.get("feature_construction"),
        "feature_construction_parameters": spec.get("feature_construction_parameters"),
        "source_root_discovery_permitted": False,
        "target_query_discovery_permitted": False,
        "H1_permitted": False,
        "M2_permitted": False,
    }


def current_loader_bindings(adapter: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """SHA-bind the named canonical loader source files without importing them."""
    bindings = adapter.get("loader_bindings_required")
    require(isinstance(bindings, Mapping) and bindings, "adapter lacks canonical loader bindings")
    result: dict[str, dict[str, str]] = {}
    for name, expected in bindings.items():
        require(isinstance(name, str) and isinstance(expected, Mapping), "loader binding schema malformed")
        relative = expected.get("path")
        symbol = expected.get("symbol")
        require(isinstance(relative, str) and isinstance(symbol, str) and symbol, "loader binding value malformed")
        path = (REPO_ROOT / relative).resolve()
        require(path.is_file() and not path.is_symlink(), f"canonical loader binding missing or symlinked: {relative}")
        # Do not import the module (which may import a real-data dependency),
        # but reject a stale binding that only preserves a file-level SHA label
        # after the named loader/construction function disappeared.
        try:
            source_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise TrackBV2LiveContractError(f"canonical loader source is not UTF-8: {relative}") from exc
        require(f"def {symbol}(" in source_text, f"canonical loader symbol missing: {relative}::{symbol}")
        result[name] = {"path": relative, "symbol": symbol, "sha256": _sha256_file(path)}
    return result


def build_sealed_loader_semantics_contract(dataset: str, view: str | None = None) -> dict[str, Any]:
    """Build in-memory source-code semantics bindings; never create a receipt."""
    adapter = canonical_adapter_spec(dataset, view)
    return {
        "schema": LOADER_SEMANTICS_SCHEMA,
        "dataset": adapter["dataset"],
        "view": adapter["view"],
        "adapter_id": adapter["adapter_id"],
        "support_semantics": adapter["support_semantics"],
        "query_semantics": adapter["query_semantics"],
        "normalizer_semantics": adapter["normalizer_semantics"],
        "auxiliary": adapter["auxiliary"],
        "feature_construction": adapter.get("feature_construction"),
        "feature_construction_parameters": adapter.get("feature_construction_parameters"),
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
        "loader_bindings": current_loader_bindings(adapter),
        "data_discovered": False,
        "nwb_opened": False,
        "target_query_opened": False,
        "gpu_used": False,
    }


def validate_sealed_loader_semantics_contract(
    adapter: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Require an exact, current canonical-loader binding before discovery."""
    require(receipt.get("schema") == LOADER_SEMANTICS_SCHEMA, "loader semantics receipt schema drift")
    expected_scope = (adapter.get("dataset"), adapter.get("view"), adapter.get("adapter_id"))
    observed_scope = (receipt.get("dataset"), receipt.get("view"), receipt.get("adapter_id"))
    require(observed_scope == expected_scope, "loader semantics receipt scope drift")
    for key in (
        "support_semantics", "query_semantics", "normalizer_semantics", "auxiliary",
        "feature_construction", "feature_construction_parameters",
    ):
        require(receipt.get(key) == adapter.get(key), f"loader semantics drift: {key}")
    require(receipt.get("information_matched") is False, "loader semantics may not claim label-information matching")
    require(receipt.get("bias_direction") == "favors_CEBRA_accuracy", "loader semantics bias direction drift")
    require(receipt.get("loader_bindings") == current_loader_bindings(adapter), "loader source SHA/symbol binding drift")
    require(receipt.get("data_discovered") is False, "loader semantics gate may not claim data discovery")
    require(receipt.get("nwb_opened") is False, "loader semantics gate may not open NWB")
    require(receipt.get("target_query_opened") is False, "loader semantics gate may not open target query")
    require(receipt.get("gpu_used") is False, "loader semantics gate may not use GPU")
    return dict(receipt)


@dataclass(frozen=True)
class IndexBlock:
    """Canonical raw observation coverage exposed to one live block.

    ``indices`` must describe all raw observations exposed to the model/readout,
    not merely each window's endpoint.  This makes support/query disjointness
    meaningful for RT's overlapping causal windows.
    """

    session_id: str
    block: str
    index_namespace: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        require(self.session_id and self.session_id == self.session_id.strip(), "index block session ID invalid")
        require(self.block in {"source_fit", "target_support", "target_query"}, "index block role invalid")
        require(self.index_namespace in {"trial_index", "raw_bin_index", "padded_bin_index"}, "index namespace invalid")
        require(self.indices, "index block cannot be empty")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in self.indices),
                "index block indices must be non-negative integers")
        require(tuple(sorted(self.indices)) == self.indices and len(set(self.indices)) == len(self.indices),
                "index block indices must be sorted and unique")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "block": self.block,
            "index_namespace": self.index_namespace,
            "indices": list(self.indices),
        }
        return payload | {"index_sha256": _sha256_json(payload)}

    @property
    def sha256(self) -> str:
        return self.as_dict()["index_sha256"]


@dataclass(frozen=True)
class SourceSupportQueryIndexManifest:
    """Full source/target index provenance for one outer target session."""

    dataset: str
    view: str | None
    outer_fold_id: str
    target_session_id: str
    source_session_ids: tuple[str, ...]
    source_fit_blocks: tuple[IndexBlock, ...]
    target_support_block: IndexBlock
    target_query_block: IndexBlock
    target_support_trial_indices: tuple[int, ...]
    target_query_trial_indices: tuple[int, ...]
    target_support_label_scalar_count: int
    target_support_label_unique_rows: int
    t4_sparse_reference_label_event_count: int
    t4_sparse_reference_label_row_count: int
    t4_sparse_reference_label_scalar_count: int
    t4_sparse_reference_label_semantics: str
    t4_neural_support_trial_count: int
    cebra_neural_support_trial_count: int
    cebra_dense_label_support_trial_count: int
    neural_exposure_matched: bool
    neural_exposure_bias_direction: str
    # pMUA consumes an explicitly constructed feature matrix.  Per-session
    # SHA bindings make it impossible to substitute an arbitrary feature view
    # after a source-only gate has passed.
    view_feature_sha256_by_session: Mapping[str, str] = field(default_factory=dict)
    auxiliary: str = "continuous_velocity_dense_bin_level"
    information_matched: bool = False
    bias_direction: str = "favors_CEBRA_accuracy"

    def __post_init__(self) -> None:
        dataset, view = base.validate_scope(self.dataset, self.view)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "view", view)
        require(self.outer_fold_id and self.outer_fold_id == self.outer_fold_id.strip(), "outer fold ID invalid")
        require(self.target_session_id and self.target_session_id == self.target_session_id.strip(), "target session ID invalid")
        source_ids = _canonical_ids(self.source_session_ids, label="source_session_ids")
        object.__setattr__(self, "source_session_ids", source_ids)
        require(self.target_session_id not in source_ids, "target session is present in source roster")
        require(len(self.source_fit_blocks) == len(source_ids), "source fit block/session count mismatch")
        require(tuple(block.session_id for block in self.source_fit_blocks) == source_ids,
                "source fit blocks must be canonically ordered by session ID")
        require(all(block.block == "source_fit" for block in self.source_fit_blocks), "source block role drift")
        require(all(block.session_id != self.target_session_id for block in self.source_fit_blocks),
                "target session entered source fit block")
        require(self.target_support_block.block == "target_support", "target support block role drift")
        require(self.target_query_block.block == "target_query", "target query block role drift")
        require(self.target_support_block.session_id == self.target_session_id and
                self.target_query_block.session_id == self.target_session_id,
                "target support/query session drift")
        require(self.target_support_block.index_namespace == self.target_query_block.index_namespace,
                "support/query must disclose one common raw-coverage namespace")
        budget = base.SUPPORT_BUDGET_TRIALS[dataset]
        support_trials = tuple(self.target_support_trial_indices)
        query_trials = tuple(self.target_query_trial_indices)
        require(support_trials == tuple(range(budget)), "support trial indices must be exact chronological M budget")
        require(query_trials and tuple(sorted(query_trials)) == query_trials and len(set(query_trials)) == len(query_trials),
                "query trial indices must be sorted, unique and non-empty")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= budget for value in query_trials),
                "query trial indices must begin strictly after the support budget")
        require(set(self.target_support_block.indices).isdisjoint(self.target_query_block.indices),
                "target support/query raw observation coverage overlaps")
        for name, value in (
            ("target_support_label_scalar_count", self.target_support_label_scalar_count),
            ("target_support_label_unique_rows", self.target_support_label_unique_rows),
            ("t4_sparse_reference_label_event_count", self.t4_sparse_reference_label_event_count),
            ("t4_sparse_reference_label_row_count", self.t4_sparse_reference_label_row_count),
            ("t4_sparse_reference_label_scalar_count", self.t4_sparse_reference_label_scalar_count),
            ("t4_neural_support_trial_count", self.t4_neural_support_trial_count),
            ("cebra_neural_support_trial_count", self.cebra_neural_support_trial_count),
            ("cebra_dense_label_support_trial_count", self.cebra_dense_label_support_trial_count),
        ):
            require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                    f"{name} must be a positive integer")
        require(self.target_support_label_unique_rows <= self.target_support_label_scalar_count,
                "unique continuous-label rows cannot exceed scalar count")
        require(self.t4_sparse_reference_label_event_count <= self.t4_sparse_reference_label_row_count,
                "T4 sparse label events cannot exceed rows")
        require(self.t4_sparse_reference_label_row_count <= self.t4_sparse_reference_label_scalar_count,
                "T4 sparse label rows cannot exceed scalar count")
        require(isinstance(self.t4_sparse_reference_label_semantics, str) and self.t4_sparse_reference_label_semantics,
                "T4 sparse label semantics are required")
        if dataset == "subject_m":
            require(
                (self.t4_neural_support_trial_count, self.t4_sparse_reference_label_event_count,
                 self.t4_sparse_reference_label_row_count, self.t4_sparse_reference_label_scalar_count,
                 self.cebra_neural_support_trial_count, self.cebra_dense_label_support_trial_count,
                 self.neural_exposure_matched, self.neural_exposure_bias_direction,
                 self.t4_sparse_reference_label_semantics)
                == (30, 50, 50, 50, 50, 50, False, "favors_CEBRA_accuracy", SUBJECT_M_T4_LABEL_SEMANTICS),
                "subject-M M50 neural-exposure or T4 label-accounting drift",
            )
        else:
            require(
                (self.t4_neural_support_trial_count, self.t4_sparse_reference_label_event_count,
                 self.cebra_neural_support_trial_count, self.cebra_dense_label_support_trial_count,
                 self.neural_exposure_matched, self.neural_exposure_bias_direction,
                 self.t4_sparse_reference_label_semantics)
                == (24, 24, 24, 24, True, "no_declared_trial_count_advantage", RT_T4D_LABEL_SEMANTICS),
                "RT M24 neural-exposure or T4 label-accounting drift",
            )
        require(isinstance(self.view_feature_sha256_by_session, Mapping),
                "feature SHA map must be a mapping")
        feature_shas = dict(self.view_feature_sha256_by_session)
        require(all(isinstance(session_id, str) for session_id in feature_shas),
                "feature SHA map session IDs must be strings")
        if dataset == "subject_m" and view == "pseudo_mua":
            expected_feature_sessions = set(source_ids) | {self.target_session_id}
            require(set(feature_shas) == expected_feature_sessions,
                    "pseudo_mua feature SHA map must bind every source and target session")
            require(all(_valid_sha(value) for value in feature_shas.values()),
                    "pseudo_mua feature SHA map contains a malformed SHA")
            object.__setattr__(self, "view_feature_sha256_by_session", dict(sorted(feature_shas.items())))
        else:
            require(not feature_shas,
                    "only subject_m pseudo_mua may declare feature-construction SHAs")
        require(self.auxiliary == "continuous_velocity_dense_bin_level",
                "only dense bin-level continuous velocity is legal for Track-B v2")
        require(self.information_matched is False,
                "prefix-budget matching may not be claimed to match label information")
        require(self.bias_direction == "favors_CEBRA_accuracy",
                "label-density bias direction must favor CEBRA accuracy")

    @property
    def source_session_roster_sha256(self) -> str:
        return _sha256_json(list(self.source_session_ids))

    @property
    def sha256(self) -> str:
        return _sha256_json(self.as_dict(include_manifest_sha=False))

    def as_dict(self, *, include_manifest_sha: bool = True) -> dict[str, Any]:
        payload = {
            "schema": INDEX_MANIFEST_SCHEMA,
            "dataset": self.dataset,
            "view": self.view,
            "outer_fold_id": self.outer_fold_id,
            "target_session_id": self.target_session_id,
            "source_session_ids": list(self.source_session_ids),
            "source_fit_blocks": [block.as_dict() for block in self.source_fit_blocks],
            "target_support_block": self.target_support_block.as_dict(),
            "target_query_block": self.target_query_block.as_dict(),
            "target_support_trial_indices": list(self.target_support_trial_indices),
            "target_query_trial_indices": list(self.target_query_trial_indices),
            "target_query_neural_in_fit": False,
            "target_query_labels_in_fit": False,
            "target_query_is_score_only": True,
            "target_support_is_not_scored": True,
            "source_selector_may_read_target": False,
            "support_query_disjoint": True,
            "source_session_roster_sha256": self.source_session_roster_sha256,
            "target_support_index_sha256": self.target_support_block.sha256,
            "target_query_index_sha256": self.target_query_block.sha256,
            "auxiliary": self.auxiliary,
            "target_support_label_scalar_count": self.target_support_label_scalar_count,
            "target_support_label_unique_rows": self.target_support_label_unique_rows,
            "t4_sparse_reference_label_event_count": self.t4_sparse_reference_label_event_count,
            "t4_sparse_reference_label_row_count": self.t4_sparse_reference_label_row_count,
            "t4_sparse_reference_label_scalar_count": self.t4_sparse_reference_label_scalar_count,
            "t4_sparse_reference_label_semantics": self.t4_sparse_reference_label_semantics,
            "t4_neural_support_trial_count": self.t4_neural_support_trial_count,
            "cebra_neural_support_trial_count": self.cebra_neural_support_trial_count,
            "cebra_dense_label_support_trial_count": self.cebra_dense_label_support_trial_count,
            "neural_exposure_matched": self.neural_exposure_matched,
            "neural_exposure_bias_direction": self.neural_exposure_bias_direction,
            "target_support_dense_labels_used_for_cebra_encoder_fit": True,
            "view_feature_sha256_by_session": dict(self.view_feature_sha256_by_session),
            "information_matched": self.information_matched,
            "bias_direction": self.bias_direction,
        }
        if include_manifest_sha:
            payload["index_manifest_sha256"] = _sha256_json(payload)
        return payload


def index_manifest_from_dict(payload: Mapping[str, Any]) -> SourceSupportQueryIndexManifest:
    """Reconstruct and revalidate a receipt payload; no arrays or paths are read."""
    require(payload.get("schema") == INDEX_MANIFEST_SCHEMA, "index manifest schema drift")

    def block(value: object) -> IndexBlock:
        require(isinstance(value, Mapping), "index block payload malformed")
        candidate = IndexBlock(
            session_id=str(value.get("session_id", "")),
            block=str(value.get("block", "")),
            index_namespace=str(value.get("index_namespace", "")),
            indices=tuple(value.get("indices", ())),
        )
        require(value.get("index_sha256") == candidate.sha256, "index block SHA drift")
        return candidate

    source = tuple(block(value) for value in payload.get("source_fit_blocks", ()))
    manifest = SourceSupportQueryIndexManifest(
        dataset=str(payload.get("dataset", "")),
        view=payload.get("view"),
        outer_fold_id=str(payload.get("outer_fold_id", "")),
        target_session_id=str(payload.get("target_session_id", "")),
        source_session_ids=tuple(payload.get("source_session_ids", ())),
        source_fit_blocks=source,
        target_support_block=block(payload.get("target_support_block")),
        target_query_block=block(payload.get("target_query_block")),
        target_support_trial_indices=tuple(payload.get("target_support_trial_indices", ())),
        target_query_trial_indices=tuple(payload.get("target_query_trial_indices", ())),
        target_support_label_scalar_count=payload.get("target_support_label_scalar_count"),
        target_support_label_unique_rows=payload.get("target_support_label_unique_rows"),
        t4_sparse_reference_label_event_count=payload.get("t4_sparse_reference_label_event_count"),
        t4_sparse_reference_label_row_count=payload.get("t4_sparse_reference_label_row_count"),
        t4_sparse_reference_label_scalar_count=payload.get("t4_sparse_reference_label_scalar_count"),
        t4_sparse_reference_label_semantics=str(payload.get("t4_sparse_reference_label_semantics", "")),
        t4_neural_support_trial_count=payload.get("t4_neural_support_trial_count"),
        cebra_neural_support_trial_count=payload.get("cebra_neural_support_trial_count"),
        cebra_dense_label_support_trial_count=payload.get("cebra_dense_label_support_trial_count"),
        neural_exposure_matched=payload.get("neural_exposure_matched"),
        neural_exposure_bias_direction=str(payload.get("neural_exposure_bias_direction", "")),
        view_feature_sha256_by_session=payload.get("view_feature_sha256_by_session", {}),
        auxiliary=str(payload.get("auxiliary", "")),
        information_matched=payload.get("information_matched"),
        bias_direction=str(payload.get("bias_direction", "")),
    )
    expected = manifest.as_dict()
    require(dict(payload) == expected, "index manifest canonical field/SHA drift")
    return manifest


def verify_support_query_disjointness(manifest: SourceSupportQueryIndexManifest) -> dict[str, Any]:
    """Emit a proof summary after constructor-level complete-coverage checks."""
    return {
        "schema": "track_b_v2_support_query_disjointness_v1",
        "dataset": manifest.dataset,
        "view": manifest.view,
        "outer_fold_id": manifest.outer_fold_id,
        "target_session_id": manifest.target_session_id,
        "index_manifest_sha256": manifest.sha256,
        "target_support_index_sha256": manifest.target_support_block.sha256,
        "target_query_index_sha256": manifest.target_query_block.sha256,
        "raw_coverage_overlap_count": len(set(manifest.target_support_block.indices) & set(manifest.target_query_block.indices)),
        "support_query_disjoint": True,
        "target_query_neural_in_fit": False,
        "target_query_labels_in_fit": False,
        "target_query_is_only_score_block": True,
        "auxiliary": manifest.auxiliary,
        "target_support_label_scalar_count": manifest.target_support_label_scalar_count,
        "target_support_label_unique_rows": manifest.target_support_label_unique_rows,
        "t4_sparse_reference_label_event_count": manifest.t4_sparse_reference_label_event_count,
        "t4_sparse_reference_label_row_count": manifest.t4_sparse_reference_label_row_count,
        "t4_sparse_reference_label_scalar_count": manifest.t4_sparse_reference_label_scalar_count,
        "t4_sparse_reference_label_semantics": manifest.t4_sparse_reference_label_semantics,
        "t4_neural_support_trial_count": manifest.t4_neural_support_trial_count,
        "cebra_neural_support_trial_count": manifest.cebra_neural_support_trial_count,
        "cebra_dense_label_support_trial_count": manifest.cebra_dense_label_support_trial_count,
        "neural_exposure_matched": manifest.neural_exposure_matched,
        "neural_exposure_bias_direction": manifest.neural_exposure_bias_direction,
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "view_feature_sha256_by_session": dict(manifest.view_feature_sha256_by_session),
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
    }


def build_source_only_neural_input_preprocessor_receipt(
    *,
    adapter: Mapping[str, Any],
    loader_semantics: Mapping[str, Any],
    manifest: SourceSupportQueryIndexManifest,
    source_fitted_neural_input_preprocessor_sha256: str,
) -> dict[str, Any]:
    """Describe only source-fitted neural-input preprocessing, not other scalers.

    This receipt is intentionally not a catch-all ``normalizer``.  It covers
    the neural inputs supplied to CEBRA and forbids a target-support/query
    refit.  Behavior-auxiliary scaling and readout-embedding standardization
    are distinct future receipts because their legal fit scopes differ.
    """
    require(manifest.dataset == adapter.get("dataset") and manifest.view == adapter.get("view"),
            "normalizer adapter/index scope drift")
    validate_sealed_loader_semantics_contract(adapter, loader_semantics)
    require(_valid_sha(source_fitted_neural_input_preprocessor_sha256),
            "source-fitted neural-input preprocessor SHA malformed")
    return {
        "schema": SOURCE_NEURAL_INPUT_PREPROCESSOR_SCHEMA,
        "dataset": manifest.dataset,
        "view": manifest.view,
        "outer_fold_id": manifest.outer_fold_id,
        "adapter_id": adapter["adapter_id"],
        "index_manifest_sha256": manifest.sha256,
        "source_session_roster_sha256": manifest.source_session_roster_sha256,
        "source_fit_index_sha256_by_session": {block.session_id: block.sha256 for block in manifest.source_fit_blocks},
        "source_fitted_neural_input_preprocessor_sha256": source_fitted_neural_input_preprocessor_sha256,
        "neural_input_preprocessor_fit_scope": "outer_source_sessions_only",
        "neural_input_preprocessor_method": "dimension_agnostic_identity_float32_binned_counts",
        "neural_input_preprocessor_parameter_count": 0,
        "source_channel_vector_or_prefix_mapping_permitted": False,
        "target_support_neural_standardization_permitted_in_headline": False,
        "target_support_neural_standardization_requires_separate_sensitivity_receipt": True,
        "target_support_neural_in_neural_input_preprocessor_fit": False,
        "target_support_dense_labels_in_neural_input_preprocessor_fit": False,
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "target_query_neural_in_neural_input_preprocessor_fit": False,
        "target_query_neural_in_fit": False,
        "target_query_labels_in_fit": False,
        "target_data_used": False,
        "auxiliary": manifest.auxiliary,
        "target_support_label_scalar_count": manifest.target_support_label_scalar_count,
        "target_support_label_unique_rows": manifest.target_support_label_unique_rows,
        "t4_sparse_reference_label_event_count": manifest.t4_sparse_reference_label_event_count,
        "t4_sparse_reference_label_row_count": manifest.t4_sparse_reference_label_row_count,
        "t4_sparse_reference_label_scalar_count": manifest.t4_sparse_reference_label_scalar_count,
        "t4_sparse_reference_label_semantics": manifest.t4_sparse_reference_label_semantics,
        "t4_neural_support_trial_count": manifest.t4_neural_support_trial_count,
        "cebra_neural_support_trial_count": manifest.cebra_neural_support_trial_count,
        "cebra_dense_label_support_trial_count": manifest.cebra_dense_label_support_trial_count,
        "neural_exposure_matched": manifest.neural_exposure_matched,
        "neural_exposure_bias_direction": manifest.neural_exposure_bias_direction,
        "behavior_auxiliary_scaler_receipt": "LIVE_BLOCKER__SEPARATE_NAMED_RECEIPT_REQUIRED",
        "behavior_auxiliary_scaler_fit_scope": "UNSET__MUST_BE_DECLARED_BEFORE_LIVE",
        "readout_embedding_standardizer_receipt": "LIVE_BLOCKER__SEPARATE_NAMED_RECEIPT_REQUIRED",
        "readout_embedding_standardizer_fit_scope": "UNSET__MUST_BE_DECLARED_PER_NAMED_READOUT",
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
    }


def build_inner_fold_selector_receipt(
    *,
    manifest: SourceSupportQueryIndexManifest,
    neural_input_preprocessor_receipt_sha256: str,
    selector: base.SourceOnlySelectorSpec,
    linear_results: Sequence[base.SourceOnlySelectionResult],
    knn_results: Sequence[base.KnnSourceOnlySelectionResult],
) -> dict[str, Any]:
    """Freeze separate source-only linear and kNN geometry choices per fold."""
    require(_valid_sha(neural_input_preprocessor_receipt_sha256),
            "neural-input preprocessor receipt SHA malformed")
    selection = base.select_source_only_geometries(
        selector=selector, linear_results=linear_results, knn_results=knn_results,
    )
    require(selection.get("target_data_used") is False, "source selector accessed target data")
    all_inner = sorted({
        fold for result in tuple(linear_results) + tuple(knn_results) for fold in result.inner_source_fold_ids
    })
    require(all_inner, "selector lacks source inner folds")
    return {
        "schema": INNER_SELECTOR_SCHEMA,
        "dataset": manifest.dataset,
        "view": manifest.view,
        "outer_fold_id": manifest.outer_fold_id,
        "target_session_id": manifest.target_session_id,
        "index_manifest_sha256": manifest.sha256,
        "source_session_roster_sha256": manifest.source_session_roster_sha256,
        "source_neural_input_preprocessor_receipt_sha256": neural_input_preprocessor_receipt_sha256,
        "source_inner_fold_ids": all_inner,
        "selector": selector.as_dict(),
        "selection": selection,
        "linear_ridge_geometry_selected_source_only": selection["linear_ridge"]["selected"]["candidate"],
        "knn_geometry_selected_source_only": selection["knn_cosine_k3"]["selected"]["candidate"],
        "target_support_read": False,
        "target_query_read": False,
        "target_data_used": False,
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "target_support_dense_labels_used_for_selector_fit": False,
        "auxiliary": manifest.auxiliary,
        "target_support_label_scalar_count": manifest.target_support_label_scalar_count,
        "target_support_label_unique_rows": manifest.target_support_label_unique_rows,
        "t4_sparse_reference_label_event_count": manifest.t4_sparse_reference_label_event_count,
        "t4_sparse_reference_label_row_count": manifest.t4_sparse_reference_label_row_count,
        "t4_sparse_reference_label_scalar_count": manifest.t4_sparse_reference_label_scalar_count,
        "t4_sparse_reference_label_semantics": manifest.t4_sparse_reference_label_semantics,
        "t4_neural_support_trial_count": manifest.t4_neural_support_trial_count,
        "cebra_neural_support_trial_count": manifest.cebra_neural_support_trial_count,
        "cebra_dense_label_support_trial_count": manifest.cebra_dense_label_support_trial_count,
        "neural_exposure_matched": manifest.neural_exposure_matched,
        "neural_exposure_bias_direction": manifest.neural_exposure_bias_direction,
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
    }


def build_multisession_provenance_plan(
    *,
    manifest: SourceSupportQueryIndexManifest,
    source_neural_input_preprocessor_receipt_sha256: str,
    selector_receipt_sha256: str,
    arm: str,
    execution_requested: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Specify model/embedding/readout/prediction receipt lineage without executing it."""
    require(arm in LIVE_ARMS, f"unsupported Track-B v2 MultiSession arm {arm!r}")
    require(device == "cpu", "Track-B v2 live plan permits CPU only")
    require(execution_requested is False, "actual CEBRA execution remains separately unauthorised")
    require(_valid_sha(source_neural_input_preprocessor_receipt_sha256),
            "neural-input preprocessor receipt SHA malformed")
    require(_valid_sha(selector_receipt_sha256), "selector receipt SHA malformed")
    behavior_scaler_requirement = {
        "fit_scope": "outer_source_behavior_auxiliary_only",
        "target_support_behavior_auxiliary_in_scaler_fit": False,
        "target_query_behavior_auxiliary_in_scaler_fit": False,
        "target_support_dense_labels_used_for_cebra_encoder_fit_after_source_scaling": True,
        "separate_from_neural_input_preprocessor": True,
    }
    if manifest.dataset == "subject_m":
        behavior_scaler_requirement |= {
            "required_method": "canonical_fit_behavior_stats_componentwise_zscore",
            "target_support_application": "apply_pointer_bound_strict27_mean_std_once_to_raw_binned_velocity",
            "record_behavior_is_final_auxiliary_after_that_single_stage": True,
            "second_refit_or_zscore_permitted": False,
        }
    else:
        behavior_scaler_requirement |= {
            "required_method": "rt_source_pooled_componentwise_zscore_from_native_loader_velocity",
            "target_support_application": "apply_pointer_bound_source_pooled_velocity_scaler_once",
            "second_refit_or_zscore_permitted": False,
        }
    source_authority_binding_required = {
        "required_immutable_source_only_authorities": [
            "source_roster", "source_neural_input_authority",
            "source_behavior_auxiliary_scaler_authority",
            "source_readout_embedding_identity_authority",
            "source_only_dual_geometry_selection_plan",
        ],
        "all_authorities_must_be_bound_by_body_sha256_and_standard_sidecar": True,
        "source_only_authority_may_not_contain_target_data": True,
    }
    if manifest.dataset == "subject_m":
        source_authority_binding_required |= {
            "source_domain": "strict27_subc_co_train_only",
            "target_domain": "subm_external_target_only",
            "source_manifest_sha256": "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9",
            "v9_cross_binding_manifest_sha256": "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb",
            "subm_lodo_source_permitted": False,
        }
    else:
        source_authority_binding_required |= {
            "source_domain": "outer_fold_declared_rt_source_sessions_only",
            "target_domain": "one_explicit_rt_outer_target_only",
            "target_session_in_source_authority_permitted": False,
        }
    return {
        "schema": MULTISESSION_PROVENANCE_SCHEMA,
        "status": "PLAN_ONLY__NO_DATA_NO_CEBRA_NO_SCORE",
        "dataset": manifest.dataset,
        "view": manifest.view,
        "outer_fold_id": manifest.outer_fold_id,
        "solver": MULTISESSION_SOLVER,
        "arm": arm,
        "model_arm_role": MODEL_ARM_ROLES[arm],
        "accuracy_table_model_arm": "cebra_joint_behavior",
        "model_arm_selection_permitted_at_runtime": False,
        "device": "cpu",
        "execution_requested": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "checkpoint_loaded": False,
        "score_emitted": False,
        "index_manifest_sha256": manifest.sha256,
        "source_neural_input_preprocessor_receipt_sha256": source_neural_input_preprocessor_receipt_sha256,
        "source_only_selector_receipt_sha256": selector_receipt_sha256,
        "source_authority_binding_required": source_authority_binding_required,
        "model_receipt_required": {
            "kind": "track_b_v2_multisession_model",
            "must_bind": [
                "index_manifest_sha256", "source_neural_input_preprocessor_receipt_sha256",
                "source_fitted_behavior_auxiliary_scaler_sha256",
                "source_only_selector_receipt_sha256",
            ],
            "behavior_auxiliary_scaler": behavior_scaler_requirement,
            "neural_input_preprocessor": {
                "required_method": "dimension_agnostic_identity_float32_binned_counts",
                "parameter_count": 0,
                "source_channel_vector_or_prefix_mapping_permitted": False,
                "target_support_neural_standardization_permitted_in_headline": False,
                "target_support_neural_standardization_requires_separate_sensitivity_receipt": True,
            },
        },
        "embedding_receipts_required": {
            "source": "one SHA-bound embedding receipt per source session",
            "target_support": "one SHA-bound support embedding receipt",
            "target_query": "one SHA-bound prediction-only embedding receipt",
        },
        # These are three *required, separately named* interpretations.  A
        # successor may not choose one post hoc by leaving a may-fit option.
        # The source-only route is the alignment/consumer comparator, the
        # support-only route is standard CEBRA decoding, and the pooled route
        # is a sensitivity analysis rather than an accuracy upper bound.
        "readout_receipts_required": {
            "source_only_consumer_mechanism_alignment": {
                "fit_on": ["source_fit"],
                "target_support_dense_labels_used_for_cebra_encoder_fit": True,
                "target_support_dense_labels_used_for_readout_fit": False,
                "target_query_in_readout_fit": False,
                "estimand": "source_readout_transferred_to_target_query_embedding__mechanism_alignment_decomposition",
                "accuracy_table_headline": False,
                "embedding_standardizer": {
                    "required_receipt_sha_field": "source_fitted_embedding_standardizer_sha256_or_explicit_identity_sha256",
                    "fit_scope": "outer_source_embeddings_only",
                    "target_support_embedding_in_standardizer_fit": False,
                    "target_query_embedding_in_standardizer_fit": False,
                },
                "required_reports": list(base.SUPPORTED_DECODERS),
            },
            "target_support_only_standard_cebra_accuracy": {
                "fit_on": ["target_support"],
                "target_support_dense_labels_used_for_cebra_encoder_fit": True,
                "target_support_dense_labels_used_for_readout_fit": True,
                "target_query_in_readout_fit": False,
                "estimand": "standard_target_support_fitted_cebra_readout_accuracy",
                "additional_target_estimator_disclosed": True,
                "accuracy_table_headline": True,
                "embedding_standardizer": {
                    "required_receipt_sha_field": "target_support_fitted_embedding_standardizer_sha256_or_explicit_identity_sha256",
                    "fit_scope": "target_support_embeddings_only",
                    "source_embedding_in_standardizer_fit": False,
                    "target_query_embedding_in_standardizer_fit": False,
                },
                "required_reports": list(base.SUPPORTED_DECODERS),
            },
            "source_plus_target_support_hybrid_sensitivity": {
                "fit_on": ["source_fit", "target_support"],
                "target_support_dense_labels_used_for_cebra_encoder_fit": True,
                "target_support_dense_labels_used_for_readout_fit": True,
                "target_query_in_readout_fit": False,
                "estimand": "pooled_source_plus_target_support_readout_hybrid_sensitivity",
                "additional_target_estimator_disclosed": True,
                "not_an_accuracy_upper_bound": True,
                "accuracy_table_headline": False,
                "embedding_standardizer": {
                    "required_receipt_sha_field": "source_plus_target_support_embedding_standardizer_sha256_or_explicit_identity_sha256",
                    "fit_scope": "source_plus_target_support_embeddings_only",
                    "target_query_embedding_in_standardizer_fit": False,
                },
                "required_reports": list(base.SUPPORTED_DECODERS),
            },
        },
        "accuracy_table_headline_estimand": "target_support_only_standard_cebra_accuracy",
        "headline_estimand_selection_permitted_at_runtime": False,
        "accuracy_table_headline": {
            "model_arm": "cebra_joint_behavior",
            "readout_estimand": "target_support_only_standard_cebra_accuracy",
            "frozen_source_adapt_is_mandatory_sensitivity_not_standard_model_substitute": True,
            "model_or_readout_outcome_selection_permitted": False,
        },
        "all_named_readout_routes_required": True,
        "cebra_score_seed_policy_required": {
            "engineering_smoke": {
                "cebra_seeds": [42], "noncitable": True, "may_select": False,
            },
            "terminal_development": {
                "cebra_seeds": [42, 43, 44],
                "aggregate": "paired_session_by_seed__session_then_seed_not_bin_pseudoreplication",
                "same_model_bundle_per_seed_scores_all_named_readout_routes_and_both_decoders": True,
                "best_seed_choice_permitted": False,
            },
            "subject_m_reference": {
                "reference": "sealed_T4_three_seed_aggregate",
                "comparison": "CEBRA_session_by_seed_distribution_vs_T4_aggregate__no_one_to_one_stochastic_pairing_claim",
            },
            "rt_reference": {
                "reference": "sealed_fixed_reference_lineage",
                "cebra_sensitivity_seeds": [42, 43, 44],
                "best_seed_choice_permitted": False,
            },
            "target_score_selection_permitted": False,
        },
        "prediction_receipt_required": {
            "input": "target_query_embedding_only",
            "must_bind": [
                "target_query_index_sha256", "model_receipt_sha256", "readout_receipt_sha256",
                "ordered_target_velocity_float32_bytes_sha256", "t4_reference_query_identity_sha256",
                "cebra_query_receptive_field_raw_coverage_sha256", "metric_implementation_authority_sha256",
            ],
            "ordered_target_rows_must_equal_t4_reference_bytes_for_paired_claim": True,
            "cebra_embedding_receptive_field_must_lie_wholly_in_target_query_and_disjoint_from_support": True,
            "unmatched_query_route": "explicit_unmatched_query_sensitivity__not_a_paired_accuracy_claim",
            "metric": {
                "outputs": 2,
                "no_pooled_window_pseudoreplication": True,
                "aggregate_order": "session_then_seed",
                "subject_m_parent": "torchmetrics_1_5_1_exact_multioutput_reduction_authority_required",
                "linear_and_knn_use_same_ordered_target_rows": True,
            },
            "query_score_may_be_emitted_only_by_successor": True,
        },
        "target_query_neural_in_cebra_fit": False,
        "target_query_labels_in_cebra_fit": False,
        "target_query_in_readout_fit": False,
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "target_support_label_scalar_count": manifest.target_support_label_scalar_count,
        "target_support_label_unique_rows": manifest.target_support_label_unique_rows,
        "t4_sparse_reference_label_event_count": manifest.t4_sparse_reference_label_event_count,
        "t4_sparse_reference_label_row_count": manifest.t4_sparse_reference_label_row_count,
        "t4_sparse_reference_label_scalar_count": manifest.t4_sparse_reference_label_scalar_count,
        "t4_sparse_reference_label_semantics": manifest.t4_sparse_reference_label_semantics,
        "t4_neural_support_trial_count": manifest.t4_neural_support_trial_count,
        "cebra_neural_support_trial_count": manifest.cebra_neural_support_trial_count,
        "cebra_dense_label_support_trial_count": manifest.cebra_dense_label_support_trial_count,
        "neural_exposure_matched": manifest.neural_exposure_matched,
        "neural_exposure_bias_direction": manifest.neural_exposure_bias_direction,
        "auxiliary": manifest.auxiliary,
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
    }


def build_actual_cebra_cpu_control_runner_contract(
    *,
    dataset: str,
    view: str | None,
    linear_ridge_scoring_geometry: base.CandidateGeometry,
    knn_scoring_geometry: base.KnnCandidateGeometry,
    execution_requested: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Predeclare a real-CEBRA CPU control gate while retaining a closed execution flag."""
    adapter = canonical_adapter_spec(dataset, view)
    require(device == "cpu", "actual CEBRA control contract permits CPU only")
    require(execution_requested is False, "actual CEBRA control execution is not authorised by this scaffold")
    return {
        "schema": CPU_CONTROL_RUNNER_SCHEMA,
        "status": "CONTRACT_ONLY__CEBRA_NOT_IMPORTED_OR_EXECUTED",
        "dataset": adapter["dataset"],
        "view": adapter["view"],
        "adapter_id": adapter["adapter_id"],
        "solver": MULTISESSION_SOLVER,
        "device": "cpu",
        "execution_requested": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "linear_ridge_scoring_geometry": linear_ridge_scoring_geometry.as_dict(),
        "knn_cosine_k3_scoring_geometry": knn_scoring_geometry.as_dict(),
        "required_seeds": list(base.DEFAULT_CONTROL_SEEDS),
        "positive_hard_gate_arms": list(base.POSITIVE_ARMS),
        "diagnostic_unaligned_arm": base.NEGATIVE_ARM,
        "expected_measurement_count": len(base.DEFAULT_CONTROL_SEEDS) * (len(base.POSITIVE_ARMS) + 1),
        "required_decoders": list(base.SUPPORTED_DECODERS),
        "expected_decoder_score_count": len(base.DEFAULT_CONTROL_SEEDS) * (len(base.POSITIVE_ARMS) + 1) * len(base.SUPPORTED_DECODERS),
        "target_query_neural_in_fit": False,
        "target_query_labels_in_fit": False,
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "requires_actual_cebra_cpu_measurements": True,
        "required_live_label_density_bindings": [
            "target_support_label_scalar_count",
            "target_support_label_unique_rows",
            "t4_sparse_reference_label_event_count",
            "t4_sparse_reference_label_row_count",
            "t4_sparse_reference_label_scalar_count",
            "t4_sparse_reference_label_semantics",
            "t4_neural_support_trial_count",
            "cebra_neural_support_trial_count",
            "cebra_dense_label_support_trial_count",
            "neural_exposure_matched",
            "neural_exposure_bias_direction",
            "information_matched",
            "bias_direction",
        ],
        "auxiliary": adapter["auxiliary"],
        "information_matched": False,
        "bias_direction": "favors_CEBRA_accuracy",
        "positive_control_hard_gate": {
            "required_seeds": list(base.DEFAULT_CONTROL_SEEDS),
            "per_seed_target_query_r2_min": 0.70,
            "required_decoders": list(base.SUPPORTED_DECODERS),
        },
        "unaligned_control": {
            "role": "diagnostic_distribution_only__all_8_seeds_and_both_decoders_reported",
            "historical_context": "2_of_8_unaligned_measurements_exceeded_0p20__not_a_failure",
            "hard_threshold": None,
            "may_select_or_rescue": False,
        },
        "synthetic_deranged_support_auxiliary_hard_null": "SEPARATE_PENDING_SUCCESSOR_CONTRACT_REQUIRED",
    }


def build_synthetic_deranged_support_auxiliary_hard_null_successor_contract(
    *,
    dataset: str,
    view: str | None,
    linear_ridge_scoring_geometry: base.CandidateGeometry,
    knn_scoring_geometry: base.KnnCandidateGeometry,
) -> dict[str, Any]:
    """Predeclare a real synthetic hard-null without manufacturing a target receipt.

    This replaces the scientifically inappropriate assumption that unaligned
    adaptation must be below a fixed R² in every seed.  A future authorised CPU
    control must deterministically derange only target-support auxiliary rows;
    neural rows and the label multiset remain exactly fixed, the permutation is
    SHA-bound independently of CEBRA seed, and true query labels remain score
    only.  Its threshold remains explicitly pending until a real synthetic
    smoke establishes a defensible frozen criterion.
    """
    adapter = canonical_adapter_spec(dataset, view)
    return {
        "schema": "track_b_v2_synthetic_deranged_support_auxiliary_hard_null_successor_v1",
        "status": "PLAN_ONLY__PENDING_REAL_SYNTHETIC_CPU_SMOKE__NOT_AUTHORISED",
        "dataset": adapter["dataset"],
        "view": adapter["view"],
        "solver": MULTISESSION_SOLVER,
        "arm": "cebra_joint_behavior__target_support_auxiliary_rows_deranged",
        "device": "cpu",
        "cebra_imported": False,
        "cebra_solver_called": False,
        "linear_ridge_scoring_geometry": linear_ridge_scoring_geometry.as_dict(),
        "knn_cosine_k3_scoring_geometry": knn_scoring_geometry.as_dict(),
        "required_seeds": list(base.DEFAULT_CONTROL_SEEDS),
        "required_decoders": list(base.SUPPORTED_DECODERS),
        "permutation_authority_required": {
            "must_bind": [
                "target_support_index_manifest_sha256",
                "target_support_neural_rows_sha256_before_after_equal",
                "target_support_auxiliary_label_multiset_sha256_before_after_equal",
                "fixed_derangement_permutation_sha256",
            ],
            "permutation_seed_independent_of_cebra_seed": True,
            "permutation_is_nonidentity_derangement": True,
            "neural_multiset_unchanged": True,
            "auxiliary_label_multiset_unchanged": True,
        },
        "true_target_query_labels_used_only_for_scoring": True,
        "target_query_neural_in_fit": False,
        "target_query_labels_in_fit": False,
        "hard_null_threshold_status": "PENDING__FREEZE_ONLY_AFTER_REAL_SYNTHETIC_CPU_SMOKE_AND_BEFORE_REAL_TARGET",
        "threshold_may_not_be_relaxed_or_backfilled": True,
        "not_a_replacement_for_unaligned_diagnostic_distribution": True,
        "gpu_used": False,
        "score_emitted": False,
    }


def assert_eight_seed_actual_cebra_cpu_controls(
    *,
    linear_ridge_scoring_geometry: base.CandidateGeometry,
    knn_scoring_geometry: base.KnnCandidateGeometry,
    measurements: Sequence[base.ControlMeasurement],
) -> dict[str, Any]:
    """Accept real control records only with exact geometry and all eight seeds."""
    spec = base.ExactGeometryControlSpec(
        scoring_geometry=linear_ridge_scoring_geometry,
        knn_scoring_geometry=knn_scoring_geometry,
        seeds=base.DEFAULT_CONTROL_SEEDS,
    )
    result = base.assert_exact_geometry_controls(spec=spec, measurements=measurements)
    require(result["measurement_count"] == 24, "actual CPU control coverage must be 3 arms x 8 seeds")
    return result | {
        "schema": "track_b_v2_eight_seed_actual_cebra_cpu_control_gate_v1",
        "required_seeds": list(base.DEFAULT_CONTROL_SEEDS),
        "actual_cebra_cpu_required": True,
    }


@dataclass(frozen=True)
class ExplicitSealedReceiptPair:
    """An explicitly named immutable source, never a discovered/globbed input."""

    role: str
    body_path: Path
    sidecar_path: Path


def _strict_readonly_pair(pair: ExplicitSealedReceiptPair) -> tuple[dict[str, Any], str]:
    body = Path(pair.body_path)
    sidecar = Path(pair.sidecar_path)
    require(body.parent == sidecar.parent, "sealed body/sidecar must share a parent directory")
    require(sidecar == body.with_name(f"{body.name}.sha256"), "sealed sidecar path must be body basename plus .sha256")
    raw = _read_immutable_regular_same_fd(body, label="sealed receipt body")
    sidecar_raw = _read_immutable_regular_same_fd(sidecar, label="sealed receipt sidecar")
    digest = hashlib.sha256(raw).hexdigest()
    require(sidecar_raw == f"{digest}  {body.name}\n".encode("ascii"), "sealed receipt sidecar/body drift")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2LiveContractError(f"{pair.role}: sealed receipt is not a JSON object") from exc
    require(isinstance(payload, dict), f"{pair.role}: sealed receipt root must be an object")
    return payload, digest


def _require_receipt_scope(payload: Mapping[str, Any], *, adapter: Mapping[str, Any], role: str) -> None:
    require(payload.get("dataset") == adapter.get("dataset") and payload.get("view") == adapter.get("view"),
            f"{role}: sealed receipt scope drift")


def _load_pointer_bound_sealed_body(
    *, pointer: ExplicitSealedReceiptPair, adapter: Mapping[str, Any]
) -> tuple[dict[str, Any], str, dict[str, float]]:
    """Resolve a sealed body that deliberately has no adjacent SHA sidecar.

    Historical canonical aggregates are sealed 0444 JSON bodies but some predate
    the body+sidecar convention.  They must remain untouched.  The only legal
    bridge is a new independently immutable pointer receipt, itself verified as
    a normal pair, which binds the old bytes, mode and schema before the body is
    read.  Metric values are extracted by JSON pointer from those verified bytes
    rather than accepted as caller-provided/rounded literals.
    """
    require(pointer.role == "canonical_reference_body_pointer", "sealed body pointer role drift")
    payload, _pointer_sha = _strict_readonly_pair(pointer)
    _require_receipt_scope(payload, adapter=adapter, role="sealed body pointer")
    require(payload.get("schema") == SEALED_BODY_POINTER_SCHEMA, "sealed body pointer schema drift")
    require(payload.get("status") == "ROOT_AUDITED_POINTER__NOT_A_RESULT", "sealed body pointer audit status drift")
    require(payload.get("pointer_role") == "canonical_reference_terminal", "sealed body pointer target role drift")
    raw_path = payload.get("sealed_body_path")
    require(isinstance(raw_path, str) and raw_path, "sealed body pointer path missing")
    body = Path(raw_path)
    raw = _read_immutable_regular_same_fd(body, label="pointed sealed body")
    info = body.lstat()
    require(stat.S_IMODE(info.st_mode) == 0o444 and payload.get("sealed_body_mode") == "0444",
            "pointed sealed body mode drift")
    require(payload.get("sealed_body_has_adjacent_sidecar") is False,
            "sidecarless sealed body pointer must declare no adjacent sidecar")
    require(not os.path.lexists(body.with_name(f"{body.name}.sha256")),
            "sidecarless pointer refuses an unexpected adjacent sidecar")
    body_sha = hashlib.sha256(raw).hexdigest()
    require(body_sha == payload.get("sealed_body_sha256"), "pointed sealed body SHA drift")
    try:
        source = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2LiveContractError("pointed sealed body is not valid UTF-8 JSON") from exc
    require(isinstance(source, dict), "pointed sealed body root must be an object")
    schema_binding = payload.get("sealed_body_schema")
    require(isinstance(schema_binding, Mapping), "sealed body schema binding missing")
    schema_field = schema_binding.get("field")
    schema_value = schema_binding.get("value")
    require(isinstance(schema_field, str) and schema_field and isinstance(schema_value, str) and schema_value,
            "sealed body schema binding malformed")
    require(source.get(schema_field) == schema_value, "pointed sealed body schema drift")
    pointers = payload.get("reference_metric_json_pointers")
    require(isinstance(pointers, Mapping) and pointers, "sealed body pointer lacks metric JSON pointers")
    metrics: dict[str, float] = {}
    for name, pointer_text in pointers.items():
        require(isinstance(name, str) and name and isinstance(pointer_text, str) and pointer_text.startswith("/"),
                "sealed body metric JSON pointer malformed")
        current: Any = source
        for token in pointer_text.removeprefix("/").split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, Mapping):
                require(token in current, f"sealed body metric pointer missing mapping key {token!r}")
                current = current[token]
            elif isinstance(current, list):
                require(token.isdecimal(), "sealed body list metric pointer must be decimal")
                index = int(token)
                require(0 <= index < len(current), "sealed body metric list pointer out of range")
                current = current[index]
            else:
                raise TrackBV2LiveContractError("sealed body metric pointer traverses scalar")
        require(isinstance(current, (int, float)) and math.isfinite(float(current)),
                f"sealed body metric {name!r} is not finite numeric")
        metrics[name] = float(current)
    return source, body_sha, metrics


def build_canonical_metric_reference_authority_from_sealed_pointer(
    *,
    adapter: Mapping[str, Any],
    canonical_reference_body_pointer: ExplicitSealedReceiptPair,
) -> dict[str, Any]:
    """Build aggregate-metric authority only; never smuggle in one fold's lineage.

    Subject-M T4 and RT canonical references are multi-fold aggregates.  A
    pointer-bound terminal can therefore authority the exact reported metric,
    but cannot truthfully authority a particular live source roster, target
    support/query split, or normalizer.  Those are bound separately for every
    live fold by :func:`build_live_fold_reference_binding`.
    """
    _terminal, terminal_sha, metrics = _load_pointer_bound_sealed_body(
        pointer=canonical_reference_body_pointer, adapter=adapter
    )
    _pointer_payload, pointer_sha = _strict_readonly_pair(canonical_reference_body_pointer)
    return {
        "schema": SEALED_REFERENCE_AUTHORITY_SCHEMA,
        "status": "BUILT_IN_MEMORY_FROM_EXPLICIT_SEALED_POINTER__NOT_MINTED",
        "authority_scope": "aggregate_metric_reference_only__not_live_fold_lineage",
        "dataset": adapter["dataset"],
        "view": adapter["view"],
        "adapter_id": adapter["adapter_id"],
        "canonical_reference_receipt_sha256": terminal_sha,
        "reference_metrics": metrics,
        "sealed_sources": {
            "canonical_reference_body_pointer": {
                "path": str(canonical_reference_body_pointer.body_path),
                "sha256": pointer_sha,
                "pointed_body_sha256": terminal_sha,
            },
        },
        "reference_metrics_from_pointer_bound_sealed_body_only": True,
        "rounded_literal_reference_metrics_accepted": False,
        "live_fold_index_binding_required": True,
        "target_data_opened_by_builder": False,
        "gpu_used_by_builder": False,
    }


def build_live_fold_reference_binding(
    *,
    adapter: Mapping[str, Any],
    canonical_reference_body_pointer: ExplicitSealedReceiptPair,
    source_roster: ExplicitSealedReceiptPair,
    source_neural_input_preprocessor: ExplicitSealedReceiptPair,
    source_support_query_index: ExplicitSealedReceiptPair,
) -> dict[str, Any]:
    """Bind one live fold's lineage without claiming it is aggregate authority.

    All three receipt pairs must be explicit, immutable, and scope/fold exact.
    The resulting object contains no result metric.  A later sealed receipt can
    bind this object's SHA to the separately root-audited aggregate metric
    pointer, avoiding the false claim that one fold describes every aggregate
    source/query configuration.
    """
    # Bind the same independently sealed, root-audited pointer used by the
    # metric-only authority builder.  A caller-supplied bare SHA would not
    # establish that the supposed aggregate authority actually exists.
    _terminal, terminal_sha, _metrics = _load_pointer_bound_sealed_body(
        pointer=canonical_reference_body_pointer, adapter=adapter
    )
    _pointer_payload, pointer_sha = _strict_readonly_pair(canonical_reference_body_pointer)
    require(source_roster.role == "source_roster", "source roster receipt role drift")
    require(source_neural_input_preprocessor.role == "source_neural_input_preprocessor",
            "neural-input preprocessor receipt role drift")
    require(source_support_query_index.role == "source_support_query_index", "index receipt role drift")
    roster, roster_sha = _strict_readonly_pair(source_roster)
    neural_preprocessor, neural_preprocessor_sha = _strict_readonly_pair(source_neural_input_preprocessor)
    index_payload, index_receipt_sha = _strict_readonly_pair(source_support_query_index)
    for payload, role in ((roster, "source roster"), (neural_preprocessor, "source neural-input preprocessor"),
                          (index_payload, "index manifest")):
        _require_receipt_scope(payload, adapter=adapter, role=role)
    require(roster.get("schema") == "track_b_v2_source_roster_receipt_v1", "source roster receipt schema drift")
    source_ids = _canonical_ids(roster.get("source_session_ids", ()), label="sealed source roster")
    manifest = index_manifest_from_dict(index_payload)
    require(manifest.dataset == adapter.get("dataset") and manifest.view == adapter.get("view"), "index manifest scope drift")
    require(roster.get("outer_fold_id") == manifest.outer_fold_id, "source roster outer-fold drift")
    require(roster.get("target_session_id") == manifest.target_session_id, "source roster target-session drift")
    require(manifest.source_session_ids == source_ids, "sealed source roster/index manifest drift")
    require(neural_preprocessor.get("schema") == SOURCE_NEURAL_INPUT_PREPROCESSOR_SCHEMA,
            "source neural-input preprocessor receipt schema drift")
    require(neural_preprocessor.get("outer_fold_id") == manifest.outer_fold_id,
            "neural-input preprocessor outer-fold drift")
    require(neural_preprocessor.get("index_manifest_sha256") == manifest.sha256,
            "neural-input preprocessor/index manifest SHA drift")
    require(neural_preprocessor.get("source_session_roster_sha256") == manifest.source_session_roster_sha256,
            "neural-input preprocessor/source roster SHA drift")
    neural_preprocessor_value_sha = neural_preprocessor.get("source_fitted_neural_input_preprocessor_sha256")
    require(_valid_sha(neural_preprocessor_value_sha), "sealed neural-input preprocessor value SHA malformed")
    require(
        neural_preprocessor.get("target_data_used") is False
        and neural_preprocessor.get("target_support_neural_in_neural_input_preprocessor_fit") is False,
        "neural-input preprocessor must remain source fitted",
    )
    return {
        "schema": LIVE_FOLD_REFERENCE_BINDING_SCHEMA,
        "status": "BUILT_IN_MEMORY_FROM_EXPLICIT_SEALED_FOLD_RECEIPTS__NOT_MINTED",
        "dataset": adapter["dataset"],
        "view": adapter["view"],
        "adapter_id": adapter["adapter_id"],
        "canonical_reference_body_pointer_sha256": pointer_sha,
        "canonical_reference_receipt_sha256": terminal_sha,
        "authority_scope": "one_live_fold_lineage__not_aggregate_metric_authority",
        "outer_fold_id": manifest.outer_fold_id,
        "target_session_id": manifest.target_session_id,
        "source_session_roster_sha256": manifest.source_session_roster_sha256,
        "target_support_index_sha256": manifest.target_support_block.sha256,
        "target_query_index_sha256": manifest.target_query_block.sha256,
        "source_fitted_neural_input_preprocessor_sha256": neural_preprocessor_value_sha,
        "target_support_dense_labels_used_for_cebra_encoder_fit": True,
        "target_support_dense_labels_used_for_readout_fit": "declared_by_named_readout_receipt_only",
        "target_query_neural_in_fit": False,
        "target_query_labels_in_fit": False,
        "sealed_sources": {
            "source_roster": {"path": str(source_roster.body_path), "sha256": roster_sha},
            "source_neural_input_preprocessor": {
                "path": str(source_neural_input_preprocessor.body_path),
                "sha256": neural_preprocessor_sha,
            },
            "source_support_query_index": {"path": str(source_support_query_index.body_path), "sha256": index_receipt_sha},
        },
        "reference_metrics_present": False,
        "target_data_opened_by_builder": False,
        "gpu_used_by_builder": False,
    }
