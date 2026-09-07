"""Deferred physical evaluator for the four-session M1 paired gap.

Importing this module does not import Torch, open an NWB, load a checkpoint
tensor, or initialize CUDA.  Session preparation and scoring run only after
the durable attempt; the frozen selected model is loaded once and its state
digest is proven unchanged across every session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from tfpd_exploration.src.cross_session_worst_group_v1 import core
from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v2
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader

from . import plan, score


class M1GapPhysicalError(score.M1GapScoreError):
    """Fail closed for the route-owned paired-gap evaluator."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M1GapPhysicalError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metadata_row(metadata_authority: Mapping[str, object], session_id: str) -> Mapping[str, object]:
    authority = dict(metadata_authority) if isinstance(metadata_authority, Mapping) else {}
    expected = v1.m1_metadata_manifest_binding_payload()
    _require(all(authority.get(key) == value for key, value in expected.items())
             and authority.get("metadata_only") is True
             and session_id in plan.SCORE_ORDER,
             "M1 gap sealed metadata source-row authority drift")
    rows = authority.get("source_rows")
    _require(isinstance(rows, list) and len(rows) == 4,
             "M1 gap sealed metadata source-row topology drift")
    candidates = [row for row in rows if isinstance(row, Mapping) and row.get("session_id") == session_id]
    _require(len(candidates) == 1, f"M1 gap sealed metadata row drift: {session_id}")
    row = candidates[0]
    _require(isinstance(row.get("relative_path"), str) and isinstance(row.get("sha256"), str),
             f"M1 gap sealed metadata descriptor drift: {session_id}")
    return row


@dataclass
class SessionDescriptorResolver:
    """Resolve one sealed session descriptor after the durable attempt."""

    source_root: Path
    metadata_authority: Mapping[str, object]
    resolution_events: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.source_root, Path) and self.source_root.is_absolute(),
                 "M1 gap session resolver source-root drift")
        _metadata_row(self.metadata_authority, plan.SCORE_ORDER[0])
        object.__setattr__(self, "metadata_authority", MappingProxyType(dict(self.metadata_authority)))

    def resolve(self, session_id: str) -> base_physical.SourceFileDescriptor:
        row = _metadata_row(self.metadata_authority, session_id)
        relative = plan.safe_relative(row["relative_path"])
        parts = Path(relative).parts
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        _require(isinstance(no_follow, int) and no_follow != 0
                 and isinstance(directory, int) and directory != 0,
                 "M1 gap session resolver no-follow support absent")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        root_fd: int | None = None
        directories: list[int] = []
        leaf_fd: int | None = None
        try:
            named_root = os.lstat(self.source_root)
            _require(stat.S_ISDIR(named_root.st_mode) and not stat.S_ISLNK(named_root.st_mode),
                     "M1 gap session source root type/symlink drift")
            root_fd = os.open(self.source_root, flags)
            opened_root = os.fstat(root_fd)
            _require((int(opened_root.st_dev), int(opened_root.st_ino))
                     == (int(named_root.st_dev), int(named_root.st_ino)),
                     "M1 gap session source root changed during open")
            current_fd = root_fd
            for component in parts[:-1]:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                         "M1 gap session parent type/symlink drift")
                child = os.open(component, flags, dir_fd=current_fd)
                directories.append(child)
                opened = os.fstat(child)
                _require((int(opened.st_dev), int(opened.st_ino))
                         == (int(before.st_dev), int(before.st_ino)),
                         "M1 gap session parent changed during open")
                current_fd = child
            before_leaf = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
            _require(stat.S_ISREG(before_leaf.st_mode) and not stat.S_ISLNK(before_leaf.st_mode),
                     "M1 gap session source body type/symlink drift")
            leaf_fd = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=current_fd)
            opened_leaf = os.fstat(leaf_fd)
            _require(stat.S_ISREG(opened_leaf.st_mode) and int(opened_leaf.st_nlink) == 1
                     and (int(opened_leaf.st_dev), int(opened_leaf.st_ino), int(opened_leaf.st_size))
                     == (int(before_leaf.st_dev), int(before_leaf.st_ino), int(before_leaf.st_size))
                     and int(opened_leaf.st_size) > 0,
                     "M1 gap session source body identity/link/size drift")
            descriptor = base_physical.SourceFileDescriptor(
                session_id=session_id,
                relative_path=relative,
                sha256=str(row["sha256"]),
                byte_count=int(opened_leaf.st_size),
            )
            self.resolution_events.append(session_id)
            return descriptor
        except OSError as error:
            raise M1GapPhysicalError(
                f"M1 gap session descriptor no-follow resolution failed: {session_id}",
            ) from error
        finally:
            if leaf_fd is not None:
                os.close(leaf_fd)
            for descriptor_fd in reversed(directories):
                os.close(descriptor_fd)
            if root_fd is not None:
                os.close(root_fd)


@dataclass(frozen=True)
class MetricOnlySessionMaterial:
    """Rebind raw parser rows as metric-only session evidence, never training labels."""

    descriptor: base_physical.SourceFileDescriptor
    rows: tuple[base_physical.UnassignedSourceM1Row, ...] = field(repr=False, compare=False)
    targets: np.ndarray = field(repr=False, compare=False)
    input_record_sha256: str = ""
    ordered_window_start_sha256: str = ""
    calibration_sha256: str = ""
    target_sha256: str = ""
    native_evidence: Mapping[str, object] = field(repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        session_id = self.descriptor.session_id
        target = np.ascontiguousarray(self.targets, dtype=np.float32)
        evidence = dict(self.native_evidence)
        _require(self.rows and all(isinstance(row, base_physical.UnassignedSourceM1Row)
                                   and row.session_id == session_id for row in self.rows)
                 and target.shape == (len(self.rows), plan.MODEL_SHAPE["outputs"])
                 and target.flags.c_contiguous and bool(np.isfinite(target).all())
                 and self.target_sha256 == core.array_digest(target)
                 and self.calibration_sha256 == self.rows[0].calibration_sha256
                 and all(row.calibration_sha256 == self.calibration_sha256 for row in self.rows)
                 and all(row.final_bin_valid is True for row in self.rows)
                 and isinstance(evidence.get("ordered_window_start_sha256"), str)
                 and evidence.get("ordered_window_start_sha256") == self.ordered_window_start_sha256
                 and isinstance(evidence.get("ordered_query_identity_sha256"), str)
                 and isinstance(evidence.get("held_source_identity_before"), Mapping)
                 and isinstance(evidence.get("held_source_identity_after"), Mapping),
                 f"M1 gap metric-only session material drift: {session_id}")
        target.setflags(write=False)
        object.__setattr__(self, "targets", target)
        object.__setattr__(self, "native_evidence", MappingProxyType(evidence))

    @classmethod
    def from_source_material(
        cls, material: base_physical.SourceSessionMaterial,
    ) -> "MetricOnlySessionMaterial":
        session_id = material.descriptor.session_id
        _require(isinstance(material, base_physical.SourceSessionMaterial)
                 and session_id in plan.SCORE_ORDER,
                 f"M1 gap parser material session identity drift: {session_id}")
        indices = tuple(sorted(material.rows_by_sample_index))
        rows = tuple(material.rows_by_sample_index[index] for index in indices)
        targets = np.ascontiguousarray(
            np.stack([np.asarray(row.raw_final_target) for row in rows], axis=0), dtype=np.float32,
        )
        native = dict(material.native_evidence)
        target_sha = core.array_digest(targets)
        record = {
            "schema": "m1_heldin_heldout_gap_metric_only_session_input_record_v1",
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
             "M1 gap R2 tensor shape/finite drift")
    metric = R2Score(multioutput="variance_weighted").to(prediction.device)
    value = metric(prediction, target)
    _require(bool(torch.isfinite(value)), "M1 gap R2 is nonfinite")
    return float(value.detach().cpu())


@dataclass
class PhysicalGapScoreBackend:
    """Sealed per-session reader plus one exact frozen-checkpoint evaluator."""

    code_root: Path
    source_root: Path
    device: str
    descriptor_resolver: SessionDescriptorResolver
    reader: Any = field(repr=False, compare=False)
    _prepared: dict[str, MetricOnlySessionMaterial] = field(default_factory=dict, init=False, repr=False)
    _anchor: Any = field(default=None, init=False, repr=False)
    _graph: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _model_state_sha256: str | None = field(default=None, init=False, repr=False)
    _progress: score.GapProgress = field(default_factory=score.GapProgress, init=False, repr=False)
    _forward_batches: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.code_root, Path) and isinstance(self.source_root, Path)
                 and self.source_root.is_absolute()
                 and isinstance(self.device, str) and self.device
                 and isinstance(self.descriptor_resolver, SessionDescriptorResolver)
                 and callable(getattr(self.reader, "read_source_session", None)),
                 "M1 gap physical backend construction drift")

    def launch_payload(
        self, identity: plan.GapIdentity, graph: object, anchor: object,
    ) -> Mapping[str, object]:
        _require(isinstance(identity, plan.GapIdentity) and not self._closed
                 and isinstance(anchor, score.AnchorEvidence)
                 and isinstance(graph, score.shared.CompletedFullGraph),
                 "M1 gap physical launch lifecycle drift")
        self._anchor = anchor
        self._graph = graph
        return {
            "schema": "m1_heldin_heldout_gap_physical_launch_v1",
            "parser": "FalconDataModule.prepare_session_data",
            "dataset": "FalconDataset",
            "model": "SpintModel",
            "device": self.device,
            "operator_gpu_constraint": "CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only)",
            "producer_root_relative": plan.PRODUCER_ROOT_RELATIVE,
            "selected_checkpoint_role": "best_source_train_loss",
            "score_order": list(plan.SCORE_ORDER),
            "sessions_opened": False,
            "checkpoint_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "target_metric_only": True,
        }

    def prepare_session(
        self, identity: plan.GapIdentity, graph: object, session_id: str,
    ) -> Mapping[str, object]:
        _require(isinstance(identity, plan.GapIdentity) and not self._closed
                 and session_id in plan.SCORE_ORDER and session_id not in self._prepared,
                 f"M1 gap physical input preparation lifecycle drift: {session_id}")
        descriptor = self.descriptor_resolver.resolve(session_id)
        opened = tuple(sorted({*self._progress.sessions_resolved_or_opened, session_id}))
        self._progress = score.GapProgress(
            sessions_resolved_or_opened=opened,
            checkpoint_opened=self._progress.checkpoint_opened,
            model_constructed=self._progress.model_constructed,
            cuda_initialized=self._progress.cuda_initialized,
            full_system_forward_batches=self._progress.full_system_forward_batches,
            input_authorities_published=self._progress.input_authorities_published,
            session_scores_published=self._progress.session_scores_published,
        )
        material = self.reader.read_source_session(descriptor)
        prepared = MetricOnlySessionMaterial.from_source_material(material)
        self._prepared[session_id] = prepared
        return prepared.authority_fragment()

    def _load_selected_model(self, code_root: Path) -> tuple[Any, str]:
        import torch

        if self._model is not None:
            return self._model, str(self._model_state_sha256)
        graph = self._graph
        _require(isinstance(graph, score.shared.CompletedFullGraph),
                 "M1 gap selected producer graph binding drift")
        body = score.shared.read_selected_checkpoint_bytes(Path(code_root), graph)
        self._progress = score.GapProgress(
            sessions_resolved_or_opened=self._progress.sessions_resolved_or_opened,
            checkpoint_opened=True,
            cuda_initialized=bool(torch.cuda.is_initialized()),
        )
        captured: list[Any] = []

        def factory() -> Any:
            model = base_physical.load_exact_m1_spint_model(Path(code_root)).to(self.device)
            base_physical.materialize_exact_m1_model(model, device=self.device)
            captured.append(model)
            return model

        observed = base_physical.strict_reload_checkpoint_bytes(
            body,
            expected_state_sha256=plan.PRODUCER_BEST_CHECKPOINT_STATE_SHA256,
            model_factory=factory,
            device=self.device,
        )
        _require(len(captured) == 1, "M1 gap strict model factory cardinality drift")
        model = captured[0]
        model.eval()
        _require(model.training is False
                 and observed == plan.PRODUCER_BEST_CHECKPOINT_STATE_SHA256,
                 "M1 gap strict selected checkpoint/eval mode drift")
        state_digest = base_physical._state_digest(model)
        _require(state_digest == plan.ANCHOR_MODEL_STATE_SHA256,
                 "M1 gap loaded model state digest drift")
        self._model = model
        self._model_state_sha256 = state_digest
        self._progress = score.GapProgress(
            sessions_resolved_or_opened=self._progress.sessions_resolved_or_opened,
            checkpoint_opened=True, model_constructed=True,
            cuda_initialized=bool(torch.cuda.is_initialized()),
            full_system_forward_batches=self._progress.full_system_forward_batches,
            input_authorities_published=self._progress.input_authorities_published,
            session_scores_published=self._progress.session_scores_published,
        )
        return model, state_digest

    def score_session(
        self, identity: plan.GapIdentity, graph: object, session_id: str,
        input_authority: Mapping[str, object],
    ) -> Mapping[str, object]:
        _require(isinstance(identity, plan.GapIdentity) and not self._closed
                 and session_id in self._prepared
                 and isinstance(self._anchor, score.AnchorEvidence)
                 and graph is self._graph,
                 f"M1 gap physical evaluation lifecycle drift: {session_id}")
        prepared = self._prepared[session_id]
        model, state_before = self._load_selected_model(self.code_root)
        import torch

        predictions: list[np.ndarray] = []
        forward_count = 0
        with torch.no_grad():
            for start in range(0, len(prepared.rows), plan.EVAL_BATCH_SIZE):
                rows = prepared.rows[start:start + plan.EVAL_BATCH_SIZE]
                x = np.ascontiguousarray(
                    np.stack([np.asarray(row.model_inputs["x"]) for row in rows], axis=0),
                    dtype=np.float32,
                )
                calibration = np.ascontiguousarray(
                    np.stack([np.asarray(row.model_inputs["calib_trialized_neural_features"])
                              for row in rows], axis=0),
                    dtype=np.float32,
                )
                x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
                calibration_t = torch.as_tensor(calibration, dtype=torch.float32, device=self.device)
                output = model(x_t, calib_trialized_neural_features=calibration_t)
                _require(tuple(output.shape) == (len(rows), plan.MODEL_SHAPE["window"],
                                                 plan.MODEL_SHAPE["outputs"])
                         and bool(torch.isfinite(output).all()),
                         f"M1 gap exact M1 forward output drift: {session_id}")
                predictions.append(
                    np.ascontiguousarray(output[:, -1, :].detach().cpu().numpy(), dtype=np.float32),
                )
                forward_count += 1
        prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
        _require(prediction.shape == prepared.targets.shape and forward_count > 0,
                 f"M1 gap final-bin output topology drift: {session_id}")
        r2 = variance_weighted_last_bin_r2(prediction, prepared.targets)
        state_after = base_physical._state_digest(model)
        _require(state_before == state_after,
                 f"M1 gap model state changed during no-grad evaluation: {session_id}")
        self._forward_batches += forward_count
        self._progress = score.GapProgress(
            sessions_resolved_or_opened=self._progress.sessions_resolved_or_opened,
            checkpoint_opened=True, model_constructed=True,
            cuda_initialized=bool(torch.cuda.is_initialized()),
            full_system_forward_batches=self._forward_batches,
            input_authorities_published=self._progress.input_authorities_published,
            session_scores_published=self._progress.session_scores_published + 1,
        )
        return score._session_score_payload(
            identity, graph, self._anchor, session_id, input_authority,
            forward_count=forward_count,
            prediction_sha256=core.array_digest(prediction),
            target_sha256=prepared.target_sha256,
            governing_r2=r2,
            model_state_before_sha256=state_before,
            model_state_after_sha256=state_after,
        )

    def progress(self) -> score.GapProgress:
        return self._progress

    def close(self) -> None:
        self._closed = True


def build_reviewed_gap_backend(
    *, root: Path, source_root: Path, device: str,
) -> PhysicalGapScoreBackend:
    """Construct a deferred evaluator without resolving session data or Torch."""
    module = source_audit_v2.bootstrap_reviewed_v1_route(Path(root))
    _require(getattr(module, "__name__", None)
             == "tfpd_exploration.src.cross_session_worst_group_v1.source_physical",
             "M1 gap reviewed physical module seam drift")
    metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    resolver = SessionDescriptorResolver(Path(source_root), metadata)
    reader = source_reader.route_owned_reader_factory(code_root=Path(root), source_root=Path(source_root))
    return PhysicalGapScoreBackend(Path(root), Path(source_root), device, resolver, reader)


__all__ = (
    "M1GapPhysicalError", "SessionDescriptorResolver", "MetricOnlySessionMaterial",
    "PhysicalGapScoreBackend", "variance_weighted_last_bin_r2", "build_reviewed_gap_backend",
)
