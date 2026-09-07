"""One-PIT-prepare STATIC30 source-only AOF coordinator.

The activity identity is deliberately the official PIT-cubic first-30 stack;
only the selected-support four-trial ridge is used for its frozen T4 side.
No APFG optimizer or alpha-training helper is reachable here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import stat

import numpy as np
import torch

from . import core, plan


class RunnerError(RuntimeError):
    pass


def _need(ok, msg):
    if not ok:
        raise RunnerError(msg)


def _digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True,
                   separators=(",", ":")).encode() + array.tobytes()
    ).hexdigest()


def _official_array_digest(value):
    """Official payload's dtype-string + shape-list + bytes framing."""
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode() + json.dumps(list(array.shape), separators=(",", ":")).encode()
        + array.tobytes(order="C")
    ).hexdigest()


def _official_receipt(root):
    """Read the pinned public export receipt without inventing sidecar semantics."""
    path = os.path.abspath(os.path.join(os.fspath(root), plan.OFFICIAL_ACT30_RECEIPT_RELATIVE))
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _need(isinstance(nofollow, int) and nofollow != 0, "AOF official receipt O_NOFOLLOW unavailable")
    fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode), "AOF official act30 receipt type drift")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    body = b"".join(chunks)
    _need(hashlib.sha256(body).hexdigest() == plan.OFFICIAL_ACT30_RECEIPT_SHA256,
          "AOF official act30 receipt body drift")
    payload = json.loads(body.decode("utf-8"))
    _need(payload.get("schema_version") == "m2_dopt4_act30_official_payload_receipt_v1"
          and payload.get("status") == "EXPORTED_NOT_SUBMITTED" and payload.get("session_count") == 13
          and payload.get("checkpoint_sha256") == plan.CHECKPOINT_SHA256
          and payload.get("activity_budget") == 30 and payload.get("label_budget") == 4
          and isinstance(payload.get("session_records"), dict), "AOF official act30 receipt semantics drift")
    return payload


def _official_native_score(root, official_receipt):
    """Held-FD read of the sealed screen's actual native source predictions."""
    relative = plan.OFFICIAL_ACT30_SCORE_RELATIVE
    directory, leaf = os.path.split(relative)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    _need(isinstance(nofollow, int) and nofollow != 0 and isinstance(directory_flag, int),
          "AOF official score FD flags unavailable")
    root_fd = os.open(os.path.join(os.fspath(root), directory), os.O_RDONLY | directory_flag | nofollow)
    try:
        root_info = os.fstat(root_fd)
        _need(stat.S_ISDIR(root_info.st_mode), "AOF official score parent type drift")
        def read_leaf(name):
            fd = os.open(name, os.O_RDONLY | nofollow, dir_fd=root_fd)
            try:
                info = os.fstat(fd)
                _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                      f"AOF official score {name} mode/link/type drift")
                chunks = []
                while True:
                    block = os.read(fd, 1 << 20)
                    if not block:
                        break
                    chunks.append(block)
                return b"".join(chunks)
            finally:
                os.close(fd)
        body = read_leaf(leaf)
        sidecar = read_leaf(leaf + ".sha256")
    finally:
        os.close(root_fd)
    digest = hashlib.sha256(body).hexdigest()
    _need(digest == plan.OFFICIAL_ACT30_SCORE_SHA256
          and sidecar == f"{digest}  score.json\n".encode("ascii"), "AOF official score body/sidecar drift")
    payload = json.loads(body.decode("utf-8"))
    _need(payload.get("schema") == "m2_t4_activity_budget_screen_v1" and payload.get("status") == "TERMINAL"
          and payload.get("checkpoint_sha256") == plan.CHECKPOINT_SHA256
          and payload.get("normalization_sha256") == official_receipt.get("normalization_sha256")
          and len(payload.get("rows", ())) == 65
          and official_receipt.get("sealed_screen_sha256") == digest,
          "AOF official native score semantics/link drift")
    wanted = [row for row in payload["rows"]
              if row.get("surface") == "within_post30" and row.get("cell") == "ridge_activity30_m4"]
    _need(len(wanted) == 7, "AOF official native score source-row count drift")
    by_session = {str(row["session"]): dict(row) for row in wanted}
    _need(len(by_session) == 7, "AOF official native score duplicate session")
    return by_session


def _official_cpu_identities(root, official_receipt):
    """Read the trusted official cached-identity payload after attempt."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _need(isinstance(nofollow, int) and nofollow != 0, "AOF official payload O_NOFOLLOW unavailable")
    path = os.path.join(os.fspath(root), plan.OFFICIAL_ACT30_PAYLOAD_RELATIVE)
    fd = os.open(path, os.O_RDONLY | nofollow)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode), "AOF official payload type drift")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    body = b"".join(chunks)
    _need(hashlib.sha256(body).hexdigest() == plan.OFFICIAL_ACT30_PAYLOAD_SHA256
          and official_receipt.get("payload_sha256") == plan.OFFICIAL_ACT30_PAYLOAD_SHA256,
          "AOF official CPU payload body/link drift")
    # The sealed pickle contains the CPU decoder class.  The qualified AOF
    # package never occupies top-level ``src``; establish streaming's legacy
    # import root only after the body SHA has been verified.
    from tfpd_exploration.src.pit_m2_v1.trainer import ensure_streaming_paths
    ensure_streaming_paths(root)
    payload = pickle.loads(body)
    _need(isinstance(payload, dict) and payload.get("schema_version") == "e8_t4_m2_cached_identity_v1"
          and payload.get("behavior_scaling_factor") == 5.0
          and isinstance(payload.get("identity_by_dataset_tag"), dict),
          "AOF official CPU payload semantics drift")
    return payload["identity_by_dataset_tag"]


def _post_identity(native_encoder, calib_trials, side_features):
    """Frozen probe's sequential post-fusion statistic, with no wrapper state."""
    values = native_encoder.pre_pool(calib_trials.permute(0, 1, 3, 2))
    side = side_features.unsqueeze(1).expand(-1, int(values.shape[1]), -1, -1)
    values = native_encoder.post_pool(torch.cat((values, side), dim=-1))
    total = values[:, 0]
    for index in range(1, int(values.shape[1])):
        total = total + values[:, index]
    return total / int(values.shape[1])


def gpu0():
    _need(torch.cuda.is_available(), "AOF GPU0 unavailable")
    torch.cuda.init()
    _need(torch.cuda.current_device() == 0 and torch.cuda.device_count() == 1,
          "AOF GPU topology")
    raw = str(torch.cuda.get_device_properties(0).uuid)
    uuid = raw if raw.startswith("GPU-") else "GPU-" + raw
    _need(uuid == plan.GPU_UUID, "AOF GPU UUID")
    return {"cuda_visible_devices": "0", "logical_device": 0, "physical_device": 0,
            "uuid": uuid, "uuid_raw": raw, "cuda_initialized": True}


class StaticAOFRunner:
    """Source-only, lexical 5/2 STATIC30 evaluator with no optimizer path."""

    FIT = (
        "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
        "ses-2020-10-20-Run2", "ses-2020-10-27-Run1",
    )
    VALIDATION = ("ses-2020-10-27-Run2", "ses-2020-10-28-Run1")

    def __init__(self, root, launch):
        self.root = root
        self.launch = launch
        self.prepared = None
        self.dataset = None
        self.static = {}

    def prepare(self):
        """Construct one source PIT stack and strict selected-T4 clone, no Adam."""
        from tfpd_exploration.src.pit_m2_v1 import trainer as pit_trainer
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as pooled_physical
        from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as official_activity

        pit = pit_trainer.PitM2ArmedRunner(self.root, "t0m", device="cuda:0")
        pit_authority = pit.prepare(attach_operator=False)
        base = pit._litmodule
        datamodule = pit._datamodule
        _need(base is not None and datamodule is not None, "AOF PIT source stack absent")
        module, strict = pooled_physical._strict_sealed_pooled_clone(repo_root=self.root, base_module=base)
        _need(strict["selected_t4_checkpoint_sha256"] == plan.CHECKPOINT_SHA256
              and strict["student_state_after_load_sha256"] == plan.STUDENT_STATE_SHA256,
              "AOF selected checkpoint/state strict-load drift")
        hparams = getattr(module, "hparams", None)
        _need(bool(getattr(hparams, "predict_scaled_behavior", False))
              and float(getattr(hparams, "behavior_scaling_factor", float("nan"))) == 5.0,
              "AOF inherited behavior scale law drift")
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        _need(not module.training and not module.student.id_encoder.training
              and all(not item.requires_grad for item in module.parameters()),
              "AOF frozen/eval state drift")
        self.prepared = type("Prepared", (), {"module": module,
                                                "datamodule": datamodule, "pit": pit})()
        self.dataset = datamodule.train_dataset
        official_receipt = _official_receipt(self.root)
        official_native_score = _official_native_score(self.root, official_receipt)
        official_cpu_ids = _official_cpu_identities(self.root, official_receipt)
        names = tuple(sorted(self.dataset.calib_trialized_neural_features))
        _need(names == self.FIT + self.VALIDATION, "AOF fixed lexical source roster drift")
        before = pooled_physical._student_state_sha(module)
        _need(before == plan.STUDENT_STATE_SHA256, "AOF strict selected student state drift")
        # Calibration/side authority is target-free.  Do not materialize
        # behaviors or query windows here; fit and validation targets open in
        # separate later phases of ``run``.
        for session in names:
            activity = np.ascontiguousarray(
                np.asarray(self.dataset.calib_trialized_neural_features[session][:30], dtype=np.float32)
            )
            _need(activity.shape == (30, 100, 96) and np.isfinite(activity).all(),
                  f"{session}: official act30 PIT-cubic activity drift")
            selected = np.ascontiguousarray(official_activity._support_indices(self.dataset, session, 4), dtype=np.int64)
            side, side_evidence = official_activity._ridge_side(self.dataset, session, selected)
            _need(np.asarray(side).shape == (96, 4) and np.isfinite(side).all(),
                  f"{session}: official act30_dopt4 side drift")
            record = official_receipt["session_records"].get(session)
            native_row = official_native_score.get(session)
            _need(isinstance(record, dict), f"{session}: official act30 session authority missing")
            _need(isinstance(native_row, dict), f"{session}: official native prediction authority missing")
            _need(_official_array_digest(activity) == record.get("activity_sha256")
                  and _official_array_digest(side) == record.get("side_sha256")
                  and selected.tolist() == record.get("selected_indices"),
                  f"{session}: official act30 activity/side/support drift")
            cpu_identity = np.ascontiguousarray(official_cpu_ids.get(record["dataset_tag"]), dtype=np.float32)
            _need(cpu_identity.shape == (96, 50)
                  and _official_array_digest(cpu_identity) == record.get("identity_sha256"),
                  f"{session}: official CPU payload identity drift")
            with torch.inference_mode():
                computed_cpu_identity = module.student.id_encoder.forward_batch(
                    torch.from_numpy(activity).unsqueeze(0), side_features=torch.from_numpy(side).unsqueeze(0)
                )
            # The public payload is the exact CPU identity authority.  The
            # independently reconstructed strict clone is deliberately a
            # *numeric* bridge only: the exporter and this loader may differ
            # in harmless CPU kernel accumulation order, while the governing
            # screen anchor is established below by the GPU direct-native
            # prediction SHA.  Do not replace the official CPU payload with
            # this recomputation or mislabel the two SHA framings as equal.
            computed_cpu = np.ascontiguousarray(computed_cpu_identity.squeeze(0).detach().numpy(), dtype=np.float32)
            cpu_reconstruction_maxabs = float(np.max(np.abs(computed_cpu - cpu_identity)))
            _need(cpu_reconstruction_maxabs <= 2e-6,
                  f"{session}: CPU reconstructed-to-official identity bridge drift")
            self.static[session] = {"activity": activity, "side": np.ascontiguousarray(side, dtype=np.float32),
                                    "official_cpu_identity": cpu_identity, "selected": selected,
                                    "official_record": dict(record), "official_native_row": native_row,
                                    "side_evidence": dict(side_evidence),
                                    "cpu_reconstruction_identity_sha256": _official_array_digest(computed_cpu),
                                    "cpu_reconstruction_maxabs": cpu_reconstruction_maxabs}
        module.to(torch.device("cuda:0")).eval()
        after = pooled_physical._student_state_sha(module)
        _need(before == after == plan.STUDENT_STATE_SHA256, "AOF source authority mutated model")
        return {
            "pit_materializations": 1,
            "sessions": list(names),
            "checkpoint_sha256": plan.CHECKPOINT_SHA256,
            "student_state_sha256": plan.STUDENT_STATE_SHA256,
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "model_state_unchanged": True,
            "activity_authority": "official_act30_dopt4_pit_cubic",
            "static_pool_trials": 30,
            "optimizer_constructed": False,
            "source_targets_loader_preloaded": True,
            "source_target_access": True,
            "fit_target_indices_opened_at_prepare": False,
            "validation_target_indices_opened_at_prepare": False,
            "parameter_updates": 0,
            "target_parameter_updates": 0,
            "pit_authority": dict(pit_authority),
            "strict_load": dict(strict),
            "official_act30_receipt": {"relative_path": plan.OFFICIAL_ACT30_RECEIPT_RELATIVE,
                                         "body_sha256": plan.OFFICIAL_ACT30_RECEIPT_SHA256,
                                         "mode_or_sidecar_claimed": False,
                                         "session_count": 13},
            "official_cpu_payload": {"relative_path": plan.OFFICIAL_ACT30_PAYLOAD_RELATIVE,
                                        "body_sha256": plan.OFFICIAL_ACT30_PAYLOAD_SHA256,
                                        "identity_execution_device": "cpu", "payload_used_for_fit": False},
            "official_native_score": {"relative_path": plan.OFFICIAL_ACT30_SCORE_RELATIVE,
                                        "body_sha256": plan.OFFICIAL_ACT30_SCORE_SHA256,
                                        "sidecar_required": True, "mode": "0444", "nlink": 1,
                                        "source_rows": 7},
            "per_session_static_authority": {
                session: {
                    "ordered_first30_activity_sha256": _digest(row["activity"]),
                    "support_indices": row["selected"].tolist(),
                    "ordered_support_indices_sha256": _digest(row["selected"]),
                    "normalized_t4_sha256": _digest(row["side"]),
                    "official_activity_sha256": row["official_record"]["activity_sha256"],
                    "official_side_sha256": row["official_record"]["side_sha256"],
                    "official_cpu_identity_sha256": row["official_record"]["identity_sha256"],
                    "cpu_reconstruction_identity_sha256": row["cpu_reconstruction_identity_sha256"],
                    "cpu_reconstruction_to_official_maxabs": row["cpu_reconstruction_maxabs"],
                    "cpu_reconstruction_to_official_maxabs_limit": 2e-6,
                    "raw_t4_sha256": str(row["side_evidence"]["raw_t4_sha256"]),
                    "side_evidence": dict(row["side_evidence"]),
                }
                for session, row in self.static.items()
            },
        }

    def _starts(self, session):
        from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as activity_core
        starts = np.asarray([start for name, start in self.dataset.window_indices if name == session], dtype=np.int64)
        return np.ascontiguousarray(
            activity_core.select_common_post30_window_starts(starts, self.dataset.trial_start_indices[session])
        )

    def pairs(self, session):
        """Open this source session's behavior targets only at this call.

        Zero is the independent direct-native decode.  The 2B concatenation
        is a pre-registered numerical parity diagnostic, never the zero row.
        """
        _need(session in self.static and self.prepared is not None, "AOF session/material lifecycle")
        static = self.static[session]
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as pooled_physical
        before = pooled_physical._student_state_sha(self.prepared.module)
        _need(before == plan.STUDENT_STATE_SHA256, "AOF pre-prediction state drift")
        activity = static["activity"]
        side = static["side"]
        starts = self._starts(session)
        neural = np.asarray(self.dataset.neural_data[session], dtype=np.float32)
        behavior = np.asarray(self.dataset.covariate_data[session], dtype=np.float32)
        _need(starts.size > 0 and np.isfinite(neural).all() and np.isfinite(behavior[starts + 49]).all(),
              f"{session}: source query/target finite authority")
        cal = torch.from_numpy(activity).unsqueeze(0).to("cuda:0")
        side_tensor = torch.from_numpy(side).unsqueeze(0).to("cuda:0")
        with torch.inference_mode():
            native_identity = self.prepared.module.student.id_encoder.forward_batch(cal, side_features=side_tensor)
            post_identity = _post_identity(self.prepared.module.student.id_encoder, cal, side_tensor)
        indexes = starts[:, None] + np.arange(plan.WINDOW_BINS)[None, :]
        windows = np.ascontiguousarray(neural[indexes], dtype=np.float32)
        direct_native, post_prediction, cpu_bridge_prediction = [], [], []
        native_parity_maxabs = post_parity_maxabs = 0.0
        native_parity_r2_abs_difference = post_parity_r2_abs_difference = 0.0
        with torch.inference_mode():
            for offset in range(0, len(windows), plan.DECODE_BATCH):
                x = torch.from_numpy(windows[offset:offset + plan.DECODE_BATCH]).to("cuda:0")
                width = int(x.shape[0])
                # Production zero anchor: exactly one native identity / B decode.
                native_out = self.prepared.module.student.decode_with_identity(
                    x, native_identity.expand(width, -1, -1)
                )[:, -1, :].detach().cpu().numpy().astype(np.float32) / np.float32(5.0)
                post_out = self.prepared.module.student.decode_with_identity(
                    x, post_identity.expand(width, -1, -1)
                )[:, -1, :].detach().cpu().numpy().astype(np.float32) / np.float32(5.0)
                direct_native.append(native_out)
                post_prediction.append(post_out)
                cpu_bridge_out = self.prepared.module.student.decode_with_identity(
                    x, torch.from_numpy(static["official_cpu_identity"]).unsqueeze(0).to("cuda:0").expand(width, -1, -1)
                )[:, -1, :].detach().cpu().numpy().astype(np.float32) / np.float32(5.0)
                cpu_bridge_prediction.append(cpu_bridge_out)
                if offset == 0:
                    paired = self.prepared.module.student.decode_with_identity(
                        torch.cat((x, x), dim=0),
                        torch.cat((native_identity.expand(width, -1, -1), post_identity.expand(width, -1, -1)), dim=0),
                    )[:, -1, :].detach().cpu().numpy().astype(np.float32) / np.float32(5.0)
                    native_delta = paired[:width] - native_out
                    post_delta = paired[width:] - post_out
                    native_parity_maxabs = float(np.max(np.abs(native_delta)))
                    post_parity_maxabs = float(np.max(np.abs(post_delta)))
                    first_target = np.ascontiguousarray(behavior[starts[offset:offset + width] + 49], dtype=np.float32)
                    native_parity_r2_abs_difference = abs(core.r2(first_target, paired[:width]) - core.r2(first_target, native_out))
                    post_parity_r2_abs_difference = abs(core.r2(first_target, paired[width:]) - core.r2(first_target, post_out))
        native = np.ascontiguousarray(np.concatenate(direct_native), dtype=np.float32)
        post = np.ascontiguousarray(np.concatenate(post_prediction), dtype=np.float32)
        cpu_bridge = np.ascontiguousarray(np.concatenate(cpu_bridge_prediction), dtype=np.float32)
        target = np.ascontiguousarray(behavior[starts + plan.WINDOW_BINS - 1], dtype=np.float32)
        zero = core.fuse(native, post, 0.0)
        _need(np.array_equal(zero, native), f"{session}: actual direct +0 native prediction drift")
        _need(native_parity_maxabs <= 2e-6 and post_parity_maxabs <= 2e-6
              and native_parity_r2_abs_difference <= 2e-7 and post_parity_r2_abs_difference <= 2e-7,
              f"{session}: 2B B-half diagnostic parity drift")
        after = pooled_physical._student_state_sha(self.prepared.module)
        _need(before == after == plan.STUDENT_STATE_SHA256, f"{session}: frozen model mutated")
        identity_maxabs = float(np.max(np.abs(native_identity.squeeze(0).detach().cpu().numpy()
                                             - static["official_cpu_identity"])))
        bridge_prediction_maxabs = float(np.max(np.abs(native - cpu_bridge)))
        bridge_r2_abs_difference = abs(core.r2(target, native) - core.r2(target, cpu_bridge))
        _need(identity_maxabs <= 2e-6 and bridge_prediction_maxabs <= 2e-6 and bridge_r2_abs_difference <= 2e-7,
              f"{session}: official CPU-to-GPU numeric bridge drift")
        witness = static["official_native_row"]
        _need(_digest(starts) == witness["ordered_window_starts_sha256"]
              and _digest(target) == witness["target_sha256"]
              and _digest(native) == witness["prediction_sha256"]
              and int(starts.size) == int(witness["window_count"])
              and abs(core.r2(target, native) - float(witness["r2"])) <= 1e-12,
              f"{session}: official native prediction witness drift")
        return native, post, target, {
            "windows": int(starts.size), "starts_sha256": _digest(starts),
            "native_prediction_sha256": _digest(native), "post_prediction_sha256": _digest(post),
            "zero_prediction_sha256": _digest(zero), "target_sha256": _digest(target),
            "native_identity_sha256": _digest(native_identity.detach().cpu().numpy()),
            "post_identity_sha256": _digest(post_identity.detach().cpu().numpy()),
            "ordered_first30_activity_sha256": _digest(activity),
            "ordered_support_indices_sha256": _digest(static["selected"]),
            "raw_t4_sha256": str(static["side_evidence"]["raw_t4_sha256"]),
            "normalized_t4_sha256": _digest(side), "support_indices": static["selected"].tolist(),
            "model_state_before_sha256": before, "model_state_after_sha256": after,
            "model_state_unchanged": True, "zero_prediction_exact": True,
            "zero_branch": "direct_native_single_identity_single_decoder",
            "two_b_diagnostic": {"native_half_maxabs": native_parity_maxabs,
                "post_half_maxabs": post_parity_maxabs,
                "native_half_r2_abs_difference": native_parity_r2_abs_difference,
                "post_half_r2_abs_difference": post_parity_r2_abs_difference,
                "not_used_for_fit_or_zero_anchor": True},
            "behavior_scaling_factor": 5.0, "parameter_updates": 0, "target_parameter_updates": 0,
            "official_native_prediction_witness": {"prediction_sha256": witness["prediction_sha256"],
                "target_sha256": witness["target_sha256"], "ordered_window_starts_sha256": witness["ordered_window_starts_sha256"],
                "window_count": int(witness["window_count"]), "r2": float(witness["r2"]), "exact": True},
            "official_cpu_to_gpu_bridge": {"official_cpu_identity_sha256": static["official_record"]["identity_sha256"],
                "gpu_identity_sha256": _official_array_digest(native_identity.squeeze(0).detach().cpu().numpy()),
                "identity_maxabs": identity_maxabs, "prediction_maxabs": bridge_prediction_maxabs,
                "r2_abs_difference": bridge_r2_abs_difference, "identity_maxabs_limit": 2e-6,
                "prediction_maxabs_limit": 2e-6, "r2_abs_difference_limit": 2e-7,
                "cpu_identity_used_for_fit": False},
        }

    def run(self):
        _need(tuple(self.static) == self.FIT + self.VALIDATION, "AOF static authority roster/order drift")
        # First open exactly the five fit source targets, then only after beta
        # materialize the two validation targets.
        raw, evidence = {}, {}
        for session in self.FIT:
            native, post, target, receipt = self.pairs(session)
            raw[session] = (native, post, target)
            evidence[session] = receipt
        fit = core.beta_equal_session(raw, expected_sessions=self.FIT)
        fit.update({"objective": "equal_session_mse_surrogate_not_direct_r2", "behavior_scaling_factor": 5.0,
                    "fit_sessions": list(self.FIT)})
        _need(len(fit["per_session"]) == plan.FIT_COUNT, "AOF fit cardinality")
        validation_raw = {}
        for session in self.VALIDATION:
            native, post, target, receipt = self.pairs(session)
            validation_raw[session] = (native, post, target)
            evidence[session] = receipt
        validation = core.gate(validation_raw, fit["beta"], expected_sessions=self.VALIDATION)
        _need(len(validation["rows"]) == plan.VALIDATION_COUNT, "AOF validation cardinality")
        result = {"fit": fit, "validation": validation, "session_evidence": evidence,
                  "source_split": {"fit": list(self.FIT), "validation": list(self.VALIDATION)},
                  "source_target_access": True, "hidden_external_evalai_target_access": False}
        if validation["passed"]:
            result["all7_refit"] = core.beta_equal_session({**raw, **validation_raw}, expected_sessions=self.FIT + self.VALIDATION)
        else:
            result["all7_refit"] = None
        return result
