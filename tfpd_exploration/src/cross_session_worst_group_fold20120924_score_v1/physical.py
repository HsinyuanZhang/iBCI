"""Deferred physical evaluator for the one-session CS-WG held-in score.

The only parser/model operations here are composed from reviewed M1 helpers.
Importing this file does not import Torch, open a source body, load a
checkpoint tensor, or initialize CUDA.  ``prepare_inputs`` is called only
after the new score attempt and ``score`` only after its input authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.cross_session_worst_group_v1 import core
from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v2
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader

from . import plan, score


class HeldInScorePhysicalError(score.HeldInScoreError):
    """Fail closed for the route-owned metric-only target evaluator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldInScorePhysicalError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _target_metadata_row(metadata_authority: Mapping[str, object]) -> Mapping[str, object]:
    rows = metadata_authority.get("source_rows")
    _require(isinstance(rows, list) and len(rows) == 4,
             "CS-WG held-in score sealed metadata source-row topology drift")
    candidates = [row for row in rows if isinstance(row, Mapping) and row.get("session_id") == plan.TARGET_SESSION]
    _require(len(candidates) == 1, "CS-WG held-in score target metadata row drift")
    row = candidates[0]
    _require(isinstance(row.get("relative_path"), str) and isinstance(row.get("sha256"), str),
             "CS-WG held-in score target metadata descriptor drift")
    return row


@dataclass
class HeldInTargetDescriptorResolver:
    """Resolve only the sealed target descriptor after durable attempt.

    The metadata authority supplies exact relative path and SHA literals.  It
    deliberately omits byte counts, so the body size is derived from one
    no-follow opened descriptor at ``prepare_inputs``; the inherited reader
    then independently holds, hashes, parses, and revalidates that same body.
    """

    source_root: Path
    metadata_authority: Mapping[str, object]
    resolution_events: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        authority = dict(self.metadata_authority) if isinstance(self.metadata_authority, Mapping) else {}
        expected = v1.m1_metadata_manifest_binding_payload()
        _require(isinstance(self.source_root, Path) and self.source_root.is_absolute()
                 and all(authority.get(key) == value for key, value in expected.items())
                 and authority.get("metadata_only") is True,
                 "CS-WG held-in score target descriptor authority drift")
        _target_metadata_row(authority)
        object.__setattr__(self, "metadata_authority", MappingProxyType(authority))

    def resolve(self) -> base_physical.SourceFileDescriptor:
        row = _target_metadata_row(self.metadata_authority)
        relative = plan.safe_relative(row["relative_path"])
        parts = Path(relative).parts
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        _require(isinstance(no_follow, int) and no_follow != 0
                 and isinstance(directory, int) and directory != 0,
                 "CS-WG held-in score target resolver no-follow support absent")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        root_fd: int | None = None
        directories: list[int] = []
        leaf_fd: int | None = None
        try:
            named_root = os.lstat(self.source_root)
            _require(stat.S_ISDIR(named_root.st_mode) and not stat.S_ISLNK(named_root.st_mode),
                     "CS-WG held-in score target source root type/symlink drift")
            root_fd = os.open(self.source_root, flags)
            opened_root = os.fstat(root_fd)
            _require((int(opened_root.st_dev), int(opened_root.st_ino))
                     == (int(named_root.st_dev), int(named_root.st_ino)),
                     "CS-WG held-in score target source root changed during open")
            current_fd = root_fd
            for component in parts[:-1]:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                         "CS-WG held-in score target parent type/symlink drift")
                child = os.open(component, flags, dir_fd=current_fd)
                directories.append(child)
                opened = os.fstat(child)
                _require((int(opened.st_dev), int(opened.st_ino)) == (int(before.st_dev), int(before.st_ino)),
                         "CS-WG held-in score target parent changed during open")
                current_fd = child
            before_leaf = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            _require(stat.S_ISREG(before_leaf.st_mode) and not stat.S_ISLNK(before_leaf.st_mode),
                     "CS-WG held-in score target source body type/symlink drift")
            leaf_fd = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=current_fd)
            opened_leaf = os.fstat(leaf_fd)
            _require(stat.S_ISREG(opened_leaf.st_mode) and int(opened_leaf.st_nlink) == 1
                     and (int(opened_leaf.st_dev), int(opened_leaf.st_ino), int(opened_leaf.st_size))
                     == (int(before_leaf.st_dev), int(before_leaf.st_ino), int(before_leaf.st_size))
                     and int(opened_leaf.st_size) > 0,
                     "CS-WG held-in score target source body identity/link/size drift")
            descriptor = base_physical.SourceFileDescriptor(
                session_id=plan.TARGET_SESSION,
                relative_path=relative,
                sha256=str(row["sha256"]),
                byte_count=int(opened_leaf.st_size),
            )
            self.resolution_events.append(plan.TARGET_SESSION)
            return descriptor
        except OSError as error:
            raise HeldInScorePhysicalError("CS-WG held-in score target descriptor no-follow resolution failed") from error
        finally:
            if leaf_fd is not None:
                os.close(leaf_fd)
            for descriptor in reversed(directories):
                os.close(descriptor)
            if root_fd is not None:
                os.close(root_fd)


@dataclass(frozen=True)
class MetricOnlyTargetMaterial:
    """Rebind raw parser rows as metric-only target evidence, never source labels."""

    descriptor: base_physical.SourceFileDescriptor
    rows: tuple[base_physical.UnassignedSourceM1Row, ...] = field(repr=False, compare=False)
    targets: np.ndarray = field(repr=False, compare=False)
    input_record_sha256: str = ""
    ordered_window_start_sha256: str = ""
    calibration_sha256: str = ""
    target_sha256: str = ""
    native_evidence: Mapping[str, object] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        target = np.ascontiguousarray(self.targets, dtype=np.float32)
        evidence = dict(self.native_evidence)
        _require(isinstance(self.descriptor, base_physical.SourceFileDescriptor)
                 and self.descriptor.session_id == plan.TARGET_SESSION
                 and self.rows and all(isinstance(row, base_physical.UnassignedSourceM1Row)
                                       and row.session_id == plan.TARGET_SESSION for row in self.rows)
                 and target.shape == (len(self.rows), plan.MODEL_SHAPE["outputs"])
                 and target.flags.c_contiguous and np.isfinite(target).all()
                 and self.target_sha256 == core.array_digest(target)
                 and self.calibration_sha256 == self.rows[0].calibration_sha256
                 and all(row.calibration_sha256 == self.calibration_sha256 for row in self.rows)
                 and all(row.final_bin_valid is True for row in self.rows)
                 and isinstance(evidence.get("ordered_window_start_sha256"), str)
                 and evidence.get("ordered_window_start_sha256") == self.ordered_window_start_sha256
                 and isinstance(evidence.get("ordered_query_identity_sha256"), str)
                 and isinstance(evidence.get("held_source_identity_before"), Mapping)
                 and isinstance(evidence.get("held_source_identity_after"), Mapping),
                 "CS-WG held-in score metric-only target material drift")
        target.setflags(write=False)
        object.__setattr__(self, "targets", target)
        object.__setattr__(self, "native_evidence", MappingProxyType(evidence))

    @classmethod
    def from_source_material(cls, material: base_physical.SourceSessionMaterial) -> "MetricOnlyTargetMaterial":
        _require(isinstance(material, base_physical.SourceSessionMaterial)
                 and material.descriptor.session_id == plan.TARGET_SESSION,
                 "CS-WG held-in score parser material target identity drift")
        indices = tuple(sorted(material.rows_by_sample_index))
        rows = tuple(material.rows_by_sample_index[index] for index in indices)
        targets = np.ascontiguousarray(np.stack([np.asarray(row.raw_final_target) for row in rows], axis=0), dtype=np.float32)
        native = dict(material.native_evidence)
        target_sha = core.array_digest(targets)
        record = {
            "schema": "cross_session_worst_group_m1_metric_only_target_input_record_v1",
            "descriptor": material.descriptor.payload(),
            "ordered_sample_indices": list(indices),
            "native_ordered_query_identity_sha256": native.get("ordered_query_identity_sha256"),
            "native_ordered_window_start_sha256": native.get("ordered_window_start_sha256"),
            "native_target_evalmask_sha256": native.get("ordered_target_evalmask_sha256"),
            "calibration_sha256": material.calibration_sha256,
            "target_sha256": target_sha,
            "metric_only_target_labels": True,
            "source_stratum_fit_or_refit_forbidden": True,
        }
        return cls(
            descriptor=material.descriptor,
            rows=rows,
            targets=targets,
            input_record_sha256=_sha(_json_bytes(record)),
            ordered_window_start_sha256=str(native["ordered_window_start_sha256"]),
            calibration_sha256=material.calibration_sha256,
            target_sha256=target_sha,
            native_evidence=native,
        )

    def authority_fragment(self) -> dict[str, object]:
        return {
            "n_windows": len(self.rows),
            "model_input_shape": [plan.MODEL_SHAPE["window"], plan.MODEL_SHAPE["units"]],
            "calibration_shape_per_row": list(plan.MODEL_SHAPE["calibration_shape_per_row"]),
            "input_record_sha256": self.input_record_sha256,
            "ordered_window_start_sha256": self.ordered_window_start_sha256,
            "target_sha256": self.target_sha256,
            "calibration_sha256": self.calibration_sha256,
            "target_descriptor": self.descriptor.payload(),
            "target_reader_native_evidence": dict(self.native_evidence),
            "metric_only_target_labels": True,
            "source_stratum_fit_or_refit_forbidden": True,
            "target_training_forbidden": True,
        }


def variance_weighted_last_bin_r2(predictions: Any, targets: Any) -> float:
    """Exact governing M1 metric, imported lazily and usable in CPU tests."""
    import torch
    from torchmetrics.regression import R2Score

    prediction = torch.as_tensor(predictions, dtype=torch.float32)
    target = torch.as_tensor(targets, dtype=torch.float32)
    _require(prediction.ndim == target.ndim == 2
             and tuple(prediction.shape) == tuple(target.shape)
             and prediction.shape[1] == plan.MODEL_SHAPE["outputs"]
             and prediction.shape[0] > 0
             and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all()),
             "CS-WG held-in score R2 tensor shape/finite drift")
    metric = R2Score(multioutput="variance_weighted").to(prediction.device)
    value = metric(prediction, target)
    _require(bool(torch.isfinite(value)), "CS-WG held-in score R2 is nonfinite")
    return float(value.detach().cpu())


@dataclass
class PhysicalHeldInScoreBackend:
    """One metric-only target reader + exact selected-checkpoint evaluator."""

    code_root: Path
    source_root: Path
    device: str
    descriptor_resolver: HeldInTargetDescriptorResolver
    reader: Any = field(repr=False, compare=False)
    route_profile: Any | None = field(default=None, repr=False, compare=False)
    _prepared: MetricOnlyTargetMaterial | None = field(default=None, init=False, repr=False)
    _progress: score.ScoreProgress = field(default_factory=score.ScoreProgress, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.code_root, Path) and isinstance(self.source_root, Path)
                 and isinstance(self.device, str) and self.device
                 and isinstance(self.descriptor_resolver, HeldInTargetDescriptorResolver)
                 and callable(getattr(self.reader, "read_source_session", None)),
                 "CS-WG held-in score physical backend construction drift")

    def _validate_route_identity_graph(self, identity: object, graph: object) -> None:
        """Validate the default V1 route or a closure-bound successor codec."""
        if self.route_profile is None:
            _require(isinstance(identity, plan.ScoreIdentity) and isinstance(graph, score.CompletedFullGraph),
                     "CS-WG held-in score physical lifecycle drift")
            return
        validator = getattr(self.route_profile, "validate_identity_graph", None)
        _require(callable(validator), "CS-WG held-in score successor physical profile seam drift")
        validator(identity, graph)

    def _profile_input_authority(
        self, identity: object, graph: object, prepared: Mapping[str, object],
    ) -> Mapping[str, object]:
        if self.route_profile is None:
            return score._input_authority_payload(identity, graph, prepared)
        builder = getattr(self.route_profile, "input_authority_payload", None)
        _require(callable(builder), "CS-WG held-in score successor input profile seam drift")
        value = builder(identity, graph, prepared)
        _require(isinstance(value, Mapping), "CS-WG held-in score successor input profile type drift")
        return value

    def _profile_score_payload(
        self, identity: object, graph: object, input_authority: Mapping[str, object], *,
        forward_count: int, prediction_sha256: str, target_sha256: str,
        governing_r2: float, model_state_before_sha256: str, model_state_after_sha256: str,
    ) -> Mapping[str, object]:
        if self.route_profile is None:
            return {
                "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_payload_v1",
                "identity_sha256": identity.sha256,
                "completed_full_graph_sha256": graph.sha256,
                "input_authority_sha256": _sha(_json_bytes(dict(input_authority))),
                "target_session": plan.TARGET_SESSION,
                "selected_checkpoint_role": "best_source_train_loss",
                "metric": plan.METRIC_LABEL,
                "last_bin_only": True,
                "eval_mode": True,
                "no_grad": True,
                "dynamic_dropout_disabled": True,
                "target_metric_only": True,
                "target_labels_used_only_for_metric": True,
                "target_optimizer_steps": 0,
                "target_backward_calls": 0,
                "target_update_calls": 0,
                "full_system_forward_count": forward_count,
                "n_windows": len(self._prepared.rows) if self._prepared is not None else 0,
                "governing_r2": governing_r2,
                "prediction_sha256": prediction_sha256,
                "target_sha256": target_sha256,
                "model_state_before_sha256": model_state_before_sha256,
                "model_state_after_sha256": model_state_after_sha256,
            }
        builder = getattr(self.route_profile, "score_payload", None)
        _require(callable(builder), "CS-WG held-in score successor score profile seam drift")
        value = builder(
            identity, graph, input_authority, forward_count=forward_count,
            n_windows=len(self._prepared.rows) if self._prepared is not None else 0,
            prediction_sha256=prediction_sha256, target_sha256=target_sha256,
            governing_r2=governing_r2, model_state_before_sha256=model_state_before_sha256,
            model_state_after_sha256=model_state_after_sha256,
        )
        _require(isinstance(value, Mapping), "CS-WG held-in score successor score profile type drift")
        return value

    def launch_payload(self, identity: object, graph: object) -> Mapping[str, object]:
        self._validate_route_identity_graph(identity, graph)
        _require(not self._closed, "CS-WG held-in score physical launch lifecycle drift")
        if self.route_profile is not None:
            builder = getattr(self.route_profile, "launch_payload", None)
            _require(callable(builder), "CS-WG held-in score successor launch profile seam drift")
            value = builder(identity, graph, device=self.device)
            _require(isinstance(value, Mapping), "CS-WG held-in score successor launch profile type drift")
            return value
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_physical_launch_v1",
            "parser": "FalconDataModule.prepare_session_data",
            "dataset": "FalconDataset",
            "model": "SpintModel",
            "device": self.device,
            "target_opened": False,
            "checkpoint_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "target_metric_only": True,
        }

    def prepare_inputs(self, identity: object, graph: object) -> Mapping[str, object]:
        self._validate_route_identity_graph(identity, graph)
        _require(not self._closed and self._prepared is None,
                 "CS-WG held-in score physical input preparation lifecycle drift")
        descriptor = self.descriptor_resolver.resolve()
        self._progress = score.ScoreProgress(target_resolved_or_opened=True)
        material = self.reader.read_source_session(descriptor)
        prepared = MetricOnlyTargetMaterial.from_source_material(material)
        self._prepared = prepared
        return prepared.authority_fragment()

    def _checkpoint_graph(self, graph: object) -> score.CompletedFullGraph:
        if self.route_profile is None:
            _require(isinstance(graph, score.CompletedFullGraph),
                     "CS-WG held-in score selected graph type drift")
            return graph
        getter = getattr(self.route_profile, "shared_checkpoint_graph", None)
        _require(callable(getter), "CS-WG held-in score successor selected graph seam drift")
        value = getter(graph)
        _require(isinstance(value, score.CompletedFullGraph),
                 "CS-WG held-in score successor selected graph type drift")
        return value

    def _load_selected_model(self, graph: object) -> tuple[Any, str]:
        import torch

        checkpoint_graph = self._checkpoint_graph(graph)
        body = score.read_selected_checkpoint_bytes(self.code_root, checkpoint_graph)
        self._progress = score.ScoreProgress(
            target_resolved_or_opened=True, checkpoint_opened=True,
            cuda_initialized=bool(torch.cuda.is_initialized()),
        )
        captured: list[Any] = []

        def factory() -> Any:
            model = base_physical.load_exact_m1_spint_model(self.code_root).to(self.device)
            base_physical.materialize_exact_m1_model(model, device=self.device)
            captured.append(model)
            return model

        observed = base_physical.strict_reload_checkpoint_bytes(
            body,
            expected_state_sha256=checkpoint_graph.expectation.best_checkpoint_state_sha256,
            model_factory=factory,
            device=self.device,
        )
        _require(len(captured) == 1, "CS-WG held-in score strict model factory cardinality drift")
        model = captured[0]
        model.eval()
        _require(model.training is False and observed == checkpoint_graph.expectation.best_checkpoint_state_sha256,
                 "CS-WG held-in score strict selected checkpoint/eval mode drift")
        self._progress = score.ScoreProgress(
            target_resolved_or_opened=True, checkpoint_opened=True, model_constructed=True,
            cuda_initialized=bool(torch.cuda.is_initialized()),
        )
        return model, observed

    def score(
        self, identity: object, graph: object, input_authority: Mapping[str, object],
    ) -> Mapping[str, object]:
        self._validate_route_identity_graph(identity, graph)
        _require(not self._closed and self._prepared is not None,
                 "CS-WG held-in score physical evaluation lifecycle drift")
        prepared = self._prepared
        _require(dict(input_authority) == self._profile_input_authority(identity, graph, prepared.authority_fragment()),
                 "CS-WG held-in score physical input authority replacement drift")
        import torch

        model, state_before = self._load_selected_model(graph)
        predictions: list[np.ndarray] = []
        forward_count = 0
        with torch.no_grad():
            for start in range(0, len(prepared.rows), plan.EVAL_BATCH_SIZE):
                rows = prepared.rows[start:start + plan.EVAL_BATCH_SIZE]
                x = np.ascontiguousarray(np.stack([np.asarray(row.model_inputs["x"]) for row in rows], axis=0), dtype=np.float32)
                calibration = np.ascontiguousarray(
                    np.stack([np.asarray(row.model_inputs["calib_trialized_neural_features"]) for row in rows], axis=0),
                    dtype=np.float32,
                )
                x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
                calibration_t = torch.as_tensor(calibration, dtype=torch.float32, device=self.device)
                output = model(x_t, calib_trialized_neural_features=calibration_t)
                _require(tuple(output.shape) == (len(rows), plan.MODEL_SHAPE["window"], plan.MODEL_SHAPE["outputs"])
                         and bool(torch.isfinite(output).all()),
                         "CS-WG held-in score exact M1 forward output drift")
                predictions.append(np.ascontiguousarray(output[:, -1, :].detach().cpu().numpy(), dtype=np.float32))
                forward_count += 1
        prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
        _require(prediction.shape == prepared.targets.shape and forward_count > 0,
                 "CS-WG held-in score final-bin output topology drift")
        r2 = variance_weighted_last_bin_r2(prediction, prepared.targets)
        state_after = base_physical._state_digest(model)
        _require(state_before == state_after, "CS-WG held-in score model state changed during no-grad evaluation")
        self._progress = score.ScoreProgress(
            target_resolved_or_opened=True, checkpoint_opened=True, model_constructed=True,
            cuda_initialized=bool(torch.cuda.is_initialized()), full_system_forward_count=forward_count,
            input_authority_published=True,
        )
        return self._profile_score_payload(
            identity, graph, input_authority, forward_count=forward_count,
            prediction_sha256=core.array_digest(prediction), target_sha256=prepared.target_sha256,
            governing_r2=r2, model_state_before_sha256=state_before,
            model_state_after_sha256=state_after,
        )

    def progress(self) -> score.ScoreProgress:
        return self._progress

    def close(self) -> None:
        self._closed = True


def build_reviewed_heldin_score_backend(
    *, root: Path, source_root: Path, device: str, route_profile: Any | None = None,
) -> PhysicalHeldInScoreBackend:
    """Construct a deferred evaluator without resolving target data or Torch.

    The reviewed namespace bootstrap preserves historical top-level ``src``
    ownership for ``streaming_calibration_exp``.  It does not open target
    bytes; descriptor resolution and native parsing remain inside
    ``prepare_inputs`` after the durable score attempt.
    """
    module = source_audit_v2.bootstrap_reviewed_v1_route(Path(root))
    _require(getattr(module, "__name__", None)
             == "tfpd_exploration.src.cross_session_worst_group_v1.source_physical",
             "CS-WG held-in score reviewed physical module seam drift")
    metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    resolver = HeldInTargetDescriptorResolver(Path(source_root), metadata)
    reader = source_reader.route_owned_reader_factory(code_root=Path(root), source_root=Path(source_root))
    return PhysicalHeldInScoreBackend(Path(root), Path(source_root), device, resolver, reader, route_profile)


__all__ = (
    "HeldInScorePhysicalError", "HeldInTargetDescriptorResolver", "MetricOnlyTargetMaterial",
    "PhysicalHeldInScoreBackend", "variance_weighted_last_bin_r2", "build_reviewed_heldin_score_backend",
)
