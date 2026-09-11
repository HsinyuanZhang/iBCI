#!/usr/bin/env python3
"""Build a sealed static EvalAI payload from one formal M1/NONE R100 run.

``build`` seals only a completed local training and scoring run. ``host`` then
performs local public-store parity checks. Neither stage contacts EvalAI or
Docker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = Path(__file__).resolve().parent
V1 = ROOT / "btransform_unified_v1"
V2 = ROOT / "btransform_unified_v2"
NONE_RUNNER = V2 / "scripts" / "m1_muscle_r100_none_ablation_v1"
V1_SRC = V1 / "src"
V2_SRC = V2 / "src"
V1_SCRIPTS = V1 / "scripts"
SOURCE = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
ALL_SESSIONS = SOURCE + HO
TAGS = {session.removeprefix("ses-") for session in SOURCE} | set(HO)
EPOCHS = tuple(range(1, 25))
UPDATES = 159_960
CELL = "M1-RIFT-R100-D4-JOINT-B3S-CONCAT-P16-NONE-ABLATION-V1"
PAYLOAD = DEST / "artifacts" / "m1_rift_none_r100_selected.pkl"
RECEIPT = DEST / "artifacts" / "build_receipt.json"
PKG = DEST / "artifacts" / "pkg"
SKIP_PKG = {"joint_m2_model.py", "m2_mechanism_model.py", "joint_m1_model.py"}
SAFE_INIT = '''"""Container-safe RIFT decoder package."""\nfrom .config import RiftTemporalConfig\nfrom .model import RiftDecoder\nfrom .streaming import RiftStreamDecoder\nfrom .temporal import RiftTemporal, RiftTemporalState\nfrom .cpu_temporal import CpuRiftTemporalRuntime\n__all__ = ["CpuRiftTemporalRuntime", "RiftDecoder", "RiftStreamDecoder", "RiftTemporal", "RiftTemporalConfig", "RiftTemporalState"]\n'''


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    need(path.is_file(), f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    need(not path.exists(), f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    need(not temporary.exists(), f"stale temporary artifact exists: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_none_runner() -> Any:
    """Load the NONE facade only at build execution time, never at import time."""
    for entry in (V2_SRC, V1_SRC, V1_SCRIPTS, NONE_RUNNER, ROOT):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import train as none_runner
    audited = none_runner._load_audited()
    # This only installs function bindings; it does not create a decoder or
    # access training data.  The formal run records the baseline path itself.
    return none_runner, audited


def expected_none_contract(runner: Any) -> dict[str, Any]:
    return runner.none_contract()


def _validate_score_rows(score: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
    rows = score.get("ema_by_epoch")
    need(isinstance(rows, dict) and set(rows) == {str(epoch) for epoch in EPOCHS}, "score receipt must contain exactly 24 EMA rows")
    values: dict[int, float] = {}
    expected_windows = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
    for epoch in EPOCHS:
        row = rows[str(epoch)]
        need(isinstance(row, dict) and row.get("partial") is False and int(row.get("n_windows", -1)) == 3881, f"epoch {epoch} score row is incomplete")
        sessions = row.get("per_session")
        need(isinstance(sessions, dict) and set(sessions) == set(HO), f"epoch {epoch} held-out roster drift")
        per_session: list[float] = []
        for session in HO:
            item = sessions[session]
            need(isinstance(item, dict) and int(item.get("window_count", -1)) == expected_windows[session], f"epoch {epoch}/{session} score inventory drift")
            value = float(item.get("channel_variance_weighted_r2", np.nan))
            need(np.isfinite(value), f"epoch {epoch}/{session} score is not finite")
            for binding_key in ("prediction_sha256", "target_sha256", "starts_sha256"):
                digest = item.get(binding_key)
                need(isinstance(digest, str) and len(digest) == 64, f"epoch {epoch}/{session} {binding_key} binding drift")
            per_session.append(value)
        mean = float(row.get("equal_session_mean_channel_variance_weighted_r2", np.nan))
        need(np.isfinite(mean) and np.isclose(mean, np.mean(per_session), rtol=0.0, atol=1e-12), f"epoch {epoch} equal-session channel score drift")
        need(np.isfinite(float(row.get("equal_session_mean", np.nan))), f"epoch {epoch} legacy score is not finite")
        values[epoch] = mean
    selected = min(values, key=lambda epoch: (-values[epoch], epoch))
    selection = score.get("selection")
    need(isinstance(selection, dict) and selection.get("metric") == "channel_variance_weighted_r2" and int(selection.get("epoch", -1)) == selected, "score selection is not earliest maximum")
    return selected, rows[str(selected)]


def validate_run(run_dir: Path, runner: Any, audited: Any) -> tuple[dict[str, Any], dict[str, Any], int, Mapping[str, Any], dict[str, Any]]:
    """Verify source, noncarrier route, receipts, every score, and all checkpoints."""
    meta = read_json(run_dir / "run_meta.json")
    baseline = meta.get("baseline_contract", {}).get("baseline_run") if isinstance(meta.get("baseline_contract"), dict) else None
    need(isinstance(baseline, str) and baseline, "NONE run metadata lacks its baseline binding")
    runner._configure(audited, Path(baseline))
    contract = expected_none_contract(runner)
    expected_meta = {
        "schema": "m1_carrier_v4_train_v1", "status": "FORMAL", "cell": CELL,
        "task": "m1", "information_arm": "none", "fusion": "concat", "proj_dim": 16,
        "seed": 42, "sampler_seed": 42, "context_bins": 100, "query_pad_bins": 99,
        "layer_windows": [25, 25, 25, 24], "depth": 4, "epochs": 24,
        "updates_per_epoch": 6665, "total_updates": UPDATES, "official_test_used": False,
        "information_arm_contract": contract,
    }
    need(all(meta.get(key) == value for key, value in expected_meta.items()), "formal NONE run metadata identity drift")
    need(meta.get("identity_interface") == "live_b3s_fusion", "training decoder identity interface drift")
    need(meta.get("source_train_only_for_gradients") is True, "training-gradient scope attestation drift")
    source_contract = meta.get("source_contract")
    need(isinstance(source_contract, dict) and source_contract.get("total_windows") == 213336 and source_contract.get("sessions") == list(SOURCE), "source noncarrier window contract drift")
    need(isinstance(source_contract.get("query_hashes"), dict) and set(source_contract["query_hashes"]) == set(SOURCE), "source target hash roster drift")
    for session in SOURCE:
        target_sha = source_contract["query_hashes"][session].get("query_target_sha256")
        need(isinstance(target_sha, str) and len(target_sha) == 64, f"{session}: source target SHA drift")
    need(meta.get("source_hashes") == audited._source_hashes(), "training source hash inventory drift")
    binding = meta.get("carrier_binding")
    need(isinstance(binding, dict) and binding.get("schema") == "m1_none_literal_zero_carrier_binding_v1", "NONE zero-route binding drift")
    need(binding.get("carrier_input") == "none; no pack accepted or read" and binding.get("shape") == [64, 4] and binding.get("dtype") == "float32", "NONE carrier-input contract drift")
    need(binding.get("all_nonzero_counts") == {session: 0 for session in ALL_SESSIONS}, "NONE carrier binding is nonzero")
    binding_body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    expected_binding_sha = hashlib.sha256(json.dumps(binding_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    need(binding.get("binding_sha256") == expected_binding_sha, "NONE carrier binding SHA drift")
    need(meta.get("fit_sha256") == "0" * 64 == binding.get("fit_sha256"), "NONE sentinel fit binding drift")
    train = read_json(run_dir / "train_receipt.json")
    train_expected = {"schema": "m1_carrier_v4_train_receipt_v1", "status": "COMPLETED", "cell": CELL,
                      "information_arm": "none", "fusion": "concat", "proj_dim": 16, "seed": 42,
                      "sampler_seed": 42, "epochs": 24, "steps": UPDATES, "post_training_scoring_required": True}
    need(all(train.get(key) == value for key, value in train_expected.items()), "NONE train receipt identity/completion drift")
    for key in ("source_hashes", "source_contract", "b3s", "carrier_binding", "fit_sha256", "information_arm_contract", "baseline_contract"):
        need(train.get(key) == meta.get(key), f"train receipt {key} does not bind run metadata")
    score = read_json(run_dir / "score_receipt.json")
    score_expected = {"schema": "m1_carrier_v4_ho_calib_epoch_scan_v1", "status": "COMPLETED", "cell": CELL,
                      "information_arm": "none", "fusion": "concat", "proj_dim": 16, "seed": 42,
                      "sampler_seed": 42, "official_test_used": False}
    need(all(score.get(key) == value for key, value in score_expected.items()), "NONE score receipt identity/completion drift")
    for key in ("source_hashes", "source_contract", "b3s", "carrier_binding", "fit_sha256", "information_arm_contract", "baseline_contract"):
        need(score.get(key) == meta.get(key), f"score receipt {key} does not bind run metadata")
    selected, selected_row = _validate_score_rows(score)
    inventory = score.get("checkpoint_sha256_by_epoch")
    need(isinstance(inventory, dict) and set(inventory) == {str(epoch) for epoch in EPOCHS}, "score receipt lacks complete checkpoint SHA inventory")
    selected_state: dict[str, Any] | None = None
    for epoch in EPOCHS:
        checkpoint = run_dir / f"epoch_{epoch:03d}.pt"
        need(checkpoint.is_file() and inventory[str(epoch)] == sha256(checkpoint), f"checkpoint SHA binding drift at epoch {epoch}")
        state = audited._validate_checkpoint(checkpoint, meta, expected_epoch=epoch)
        if epoch == selected:
            selected_state = state
    need(selected_state is not None, "selected checkpoint was not loaded")
    return meta, score, selected, selected_row, selected_state


def copy_minimal_runtime_package() -> None:
    need(not PKG.exists(), f"refusing to overwrite package tree: {PKG}")
    for source, name in ((V1_SRC / "btransform_unified_v1", "btransform_unified_v1"), (V2_SRC / "btransform_unified_v2", "btransform_unified_v2")):
        target_root = PKG / name
        for item in source.rglob("*.py"):
            if item.name in SKIP_PKG:
                continue
            target = target_root / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    (PKG / "btransform_unified_v2" / "__init__.py").write_text(SAFE_INIT, encoding="utf-8")


def build(run_dir: Path) -> dict[str, Any]:
    need(not PAYLOAD.exists() and not RECEIPT.exists() and not PKG.exists(), "build requires fresh payload, receipt, and package targets")
    need(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    runner, audited = load_none_runner()
    meta, score, epoch, score_row, checkpoint_state = validate_run(run_dir, runner, audited)
    from btransform_unified_v1.ema import DecoderEMA
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from m1_rift_none_falcon_decoder import PAYLOAD_SCHEMA

    # NONE session installation receives only names; it retains neither E0 nor T
    # and _identity is deliberately the sole call used to seal literal zeros.
    joint = runner.M1InformationArmDecoder("none", fusion="concat", proj_dim=16, seed=42)
    class Session:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
    sessions = {session: Session(session) for session in ALL_SESSIONS}
    joint.install_session_memory(sessions, {})
    joint.load_state_dict(checkpoint_state["raw_state_dict"], strict=True)
    ema = DecoderEMA(joint, decay=float(meta["ema_decay"]))
    ema.load_state_dict(checkpoint_state["ema"])
    # DecoderEMA intentionally shadows only its trainable parameter set.  The
    # NONE arm retains frozen encoder parameters in the raw checkpoint, so
    # iterating ``joint.named_parameters()`` would incorrectly demand encoder
    # keys absent from the EMA.  apply_to preserves those frozen raw values and
    # all state-dict aliases/buffers while replacing every trained key by EMA.
    ema.apply_to(joint)
    joint.eval()

    banks: dict[str, dict[str, Any]] = {}
    for session in ALL_SESSIONS:
        e0, direct_t = joint._identity([session], torch.device("cpu"))
        e0_array = np.ascontiguousarray(e0[0].detach().cpu().numpy(), dtype=np.float32)
        t_array = np.ascontiguousarray(direct_t[0].detach().cpu().numpy(), dtype=np.float32)
        need(e0_array.shape == (64, 100) and t_array.shape == (64, 4), f"{session}: NONE identity geometry drift")
        need(np.isfinite(e0_array).all() and np.isfinite(t_array).all() and not e0_array.any() and not t_array.any(), f"{session}: NONE identity is not literal zero")
        tag = session.removeprefix("ses-")
        banks[tag] = {"E0": e0_array, "T": t_array, "unit_mask": np.ones(64, dtype=np.bool_), "session": session,
                      "e0_sha256": array_sha256(e0_array), "t_sha256": array_sha256(t_array)}
    need(set(banks) == TAGS, "payload tag roster drift")

    concat = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42)
    concat_keys = set(concat.state_dict())
    ema_state = {name: value.detach().cpu().float().clone() for name, value in joint.state_dict().items() if name in concat_keys}
    missing, unexpected = concat.load_state_dict(ema_state, strict=True)
    need(not missing and not unexpected and set(ema_state) == concat_keys, "selected NONE EMA cannot strictly populate static concat decoder")
    concat.eval()
    checkpoint = run_dir / f"epoch_{epoch:03d}.pt"
    route = {"e0": "literal_zero_64x100", "direct_t": "literal_zero_64x4", "encoder": "retained_state_keys_frozen_never_called"}
    payload = {
        "schema": PAYLOAD_SCHEMA, "task": "m1", "arm": "NONE", "context_bins": 100, "bias_mode": "recency",
        "identity_interface": "literal_zero_e0_t_none_concat", "information_route": route,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "run_dir": str(run_dir.resolve()), "run_meta_sha256": sha256(run_dir / "run_meta.json"),
        "train_receipt_sha256": sha256(run_dir / "train_receipt.json"), "score_receipt_sha256": sha256(run_dir / "score_receipt.json"),
        "source_hashes": meta["source_hashes"], "source_hashes_sha256": hashlib.sha256(json.dumps(meta["source_hashes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "information_arm_contract": meta["information_arm_contract"], "carrier_binding": meta["carrier_binding"],
        "bank_by_dataset_tag": banks, "ema_state_dict": {name: value.detach().cpu().float().clone() for name, value in concat.state_dict().items()},
        "selection": {"epoch": epoch, "metric": "channel_variance_weighted_r2", "rule": "earliest maximum equal-session mean EMA on visible HO3 calibration",
                      "equal_session_mean_channel_variance_weighted_r2": float(score_row["equal_session_mean_channel_variance_weighted_r2"]),
                      "legacy_equal_session_mean": float(score_row["equal_session_mean"]), "official_test_used": False},
    }
    PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    with PAYLOAD.open("xb") as handle:
        pickle.dump(payload, handle, protocol=4)
    copy_minimal_runtime_package()
    receipt = {"schema": "m1_rift_none_r100_static_build_receipt_v1", "status": "BUILT_NOT_HOST_VERIFIED",
               "payload": str(PAYLOAD), "payload_sha256": sha256(PAYLOAD), "run_dir": str(run_dir.resolve()),
               "run_meta_sha256": payload["run_meta_sha256"], "train_receipt_sha256": payload["train_receipt_sha256"],
               "score_receipt_sha256": payload["score_receipt_sha256"], "checkpoint": str(checkpoint.resolve()),
               "checkpoint_sha256": payload["checkpoint_sha256"], "selected_epoch": epoch,
               "selection_metric": payload["selection"]["metric"], "tags": sorted(TAGS), "information_route": route,
               "runtime_package": str(PKG)}
    atomic_json_new(RECEIPT, receipt)
    return receipt


HOST_STEPS = 128
HOST_GATE = 2.0e-5
HOST_RECEIPT = DEST / "artifacts" / "host_verify.json"
SMOKE_FIXTURE = DEST / "artifacts" / "smoke_window.npz"


def _full_context(raw: np.ndarray, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a left-padded W100 reference without inspecting amplitudes."""
    need(raw.ndim == 2 and raw.shape[1] == 64 and 0 <= end < len(raw), "invalid raw M1 context")
    width = min(end + 1, 100)
    window = np.zeros((1, 100, 64), dtype=np.float32)
    window[0, 100 - width:] = raw[end - width + 1:end + 1]
    valid = torch.zeros((1, 100), dtype=torch.bool)
    valid[:, 100 - width:] = True
    return torch.from_numpy(window), valid


def _cached_row_state(runtime: Any, row: int) -> tuple[torch.Tensor, ...]:
    cached = runtime.cached_temporal
    need(cached is not None and runtime.last is not None, "cached runtime state is unavailable")
    return (runtime.raw4[row].clone(), runtime.last[row].clone(),
            *(value[row].clone() for value in cached.keys),
            *(value[row].clone() for value in cached.values),
            *(value[row].clone() for value in cached.lengths))


def _same_cached_row_state(before: tuple[torch.Tensor, ...], after: tuple[torch.Tensor, ...]) -> bool:
    return len(before) == len(after) and all(torch.equal(left, right) for left, right in zip(before, after))


def _dataset_stem(tag: str) -> str:
    kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
    return f"sub-MonkeyL-{kind}_ses-{tag}_behavior+ecephys"


def _host_raw_inputs(run_dir: Path) -> dict[str, np.ndarray]:
    """Read only public padded neural stores, then remove their 99 query bins."""
    meta = read_json(run_dir / "run_meta.json")
    baseline = meta.get("baseline_contract", {}).get("baseline_run") if isinstance(meta.get("baseline_contract"), dict) else None
    need(isinstance(baseline, str) and baseline, "NONE run metadata lacks its baseline binding")
    runner, audited = load_none_runner()
    runner._configure(audited, Path(baseline))
    zeros, _binding = runner._zero_binding(audited)
    dataset, _sampler = audited.legacy.build_fullsession_face()
    heldout = audited._ho_material(zeros)
    raw: dict[str, np.ndarray] = {}
    for session in SOURCE:
        padded = np.ascontiguousarray(dataset.neural_data[session], dtype=np.float32)
        need(padded.ndim == 2 and padded.shape[1] == 64 and len(padded) >= 99 + HOST_STEPS, f"{session}: public padded store geometry drift")
        raw[session.removeprefix("ses-")] = np.ascontiguousarray(padded[99:])
    for session in HO:
        padded = np.ascontiguousarray(heldout[session]["dataset"].neural_data[session], dtype=np.float32)
        need(padded.ndim == 2 and padded.shape[1] == 64 and len(padded) >= 99 + HOST_STEPS, f"{session}: public padded store geometry drift")
        raw[session] = np.ascontiguousarray(padded[99:])
    need(set(raw) == TAGS, "public raw tag roster drift")
    return raw


def host(run_dir: Path) -> dict[str, Any]:
    """Verify static full-R100 references against cached runtime state machines."""
    need(not HOST_RECEIPT.exists() and not SMOKE_FIXTURE.exists(), "host verification requires fresh receipt and fixture targets")
    build_receipt = read_json(RECEIPT)
    need(build_receipt.get("status") == "BUILT_NOT_HOST_VERIFIED", "host requires a completed unverified build receipt")
    need(Path(build_receipt.get("run_dir", "")).resolve() == run_dir.resolve(), "host run directory differs from build receipt")
    payload_path = Path(build_receipt.get("payload", ""))
    need(payload_path.resolve() == PAYLOAD.resolve() and payload_path.is_file(), "build receipt payload binding drift")
    need(build_receipt.get("payload_sha256") == sha256(payload_path), "build receipt payload SHA drift")
    from falcon_challenge.config import FalconConfig, FalconTask
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from btransform_unified_v2.streaming import RiftStreamDecoder
    from m1_rift_none_falcon_decoder import M1RiftNoneCachedFalconDecoder, _task_bank, load_payload, validate_payload

    payload = load_payload(payload_path)
    validate_payload(payload)
    need(payload.get("checkpoint_sha256") == build_receipt.get("checkpoint_sha256"), "payload/checkpoint receipt binding drift")
    need(int(payload.get("selection", {}).get("epoch", -1)) == int(build_receipt.get("selected_epoch", -1)), "payload selection receipt binding drift")
    raw = _host_raw_inputs(run_dir)
    # Fixture input is captured before the separate all-zero-valid probe is
    # created. It is therefore exactly the first 40 true post-padding bins.
    fixture_tag = "20121004"
    fixture_raw = np.ascontiguousarray(raw[fixture_tag][:40].copy())
    traces = {tag: np.ascontiguousarray(value[:HOST_STEPS].copy()) for tag, value in raw.items()}
    for value in traces.values():
        value[40, :] = 0.0
    need(np.array_equal(fixture_raw, raw[fixture_tag][:40]), "fixture raw bins changed before sealing")

    static = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42)
    missing, unexpected = static.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32) for key, value in payload["ema_state_dict"].items()}, strict=True)
    need(not missing and not unexpected, "payload static decoder state mismatch")
    static.eval()
    config = FalconConfig(task=FalconTask.m1)
    per_tag: dict[str, Any] = {}
    max_packed_full = 0.0
    for tag in sorted(TAGS):
        packed = M1RiftNoneCachedFalconDecoder(config, str(payload_path), 1)
        packed.reset([_dataset_stem(tag)])
        bank = _task_bank(tag, payload["bank_by_dataset_tag"][tag])
        maximum = 0.0
        for end in range(HOST_STEPS):
            full_input, valid = _full_context(traces[tag], end)
            with torch.inference_mode():
                full = static(full_input, bank, input_valid_mask=valid).cpu().numpy()
            cached = packed.predict(traces[tag][end:end + 1])
            maximum = max(maximum, float(np.max(np.abs(cached - full))))
        need(maximum <= HOST_GATE, f"{tag}: packed cached/full-R100 parity failed: {maximum}")
        per_tag[tag] = {"steps": HOST_STEPS, "packed_cached_vs_full_r100_max_abs": maximum,
                        "startup_endpoints_checked": [0, 1], "full_context_endpoint_checked": 99,
                        "all_zero_valid_endpoint": 40}
        max_packed_full = max(max_packed_full, maximum)

    tags = sorted(TAGS)
    b8_tags = tags + [tags[0]]
    b8_banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag], session_id=f"host-b8-{index}") for index, tag in enumerate(b8_tags)]
    b8_ids = [f"host-b8-{index}" for index in range(8)]
    runtime_b8 = CpuRiftRuntime(static, b8_banks, b8_ids, temporal_backend="cached")
    packed_b8 = M1RiftNoneCachedFalconDecoder(config, str(payload_path), 8)
    packed_b8.reset([_dataset_stem(tag) for tag in b8_tags])
    histories: list[list[np.ndarray]] = [[] for _ in b8_tags]
    max_b8_full = 0.0
    max_packed_b8_cached = 0.0
    max_packed_b8_full = 0.0
    inactive_checked = False
    packed_resume_checked = False
    zero_valid_checked = False
    for end in range(HOST_STEPS):
        observed = np.stack([traces[tag][end] for tag in b8_tags]).astype(np.float32, copy=False)
        valid_rows = torch.ones(8, dtype=torch.bool)
        before = None
        if end == 41:
            valid_rows[7] = False
            before = _cached_row_state(runtime_b8, 7)
        if end == 40:
            zero_valid_checked = bool(not np.any(observed))
        got = runtime_b8.advance(torch.from_numpy(observed), valid_mask=valid_rows)
        active_count = int(valid_rows.sum())
        packed = packed_b8.predict(observed[:active_count])
        max_packed_b8_cached = max(max_packed_b8_cached, float(np.max(np.abs(packed - got[:active_count].cpu().numpy()))))
        if end == 42:
            packed_resume_checked = bool(np.allclose(packed[7], got[7].cpu().numpy(), atol=HOST_GATE, rtol=0.0))
        for row, bank in enumerate(b8_banks):
            if not bool(valid_rows[row]):
                need(before is not None and _same_cached_row_state(before, _cached_row_state(runtime_b8, row)), "inactive B8 row changed cached state")
                inactive_checked = True
                continue
            histories[row].append(np.ascontiguousarray(observed[row].copy()))
            consumed = np.stack(histories[row])
            full_input, valid = _full_context(consumed, len(consumed) - 1)
            with torch.inference_mode():
                full = static(full_input, bank, input_valid_mask=valid)[0]
            max_b8_full = max(max_b8_full, float((got[row] - full).abs().max()))
            if row < active_count:
                max_packed_b8_full = max(max_packed_b8_full, float(np.max(np.abs(packed[row] - full.cpu().numpy()))))
    runtime_b8.reset_rows([b8_ids[0]])
    reset_observed = torch.from_numpy(np.stack([traces[tag][HOST_STEPS - 1] for tag in b8_tags]).astype(np.float32, copy=False))
    reset_got = runtime_b8.advance(reset_observed)
    reset_input, reset_valid = _full_context(traces[b8_tags[0]][HOST_STEPS - 1:HOST_STEPS], 0)
    with torch.inference_mode():
        reset_full = static(reset_input, b8_banks[0], input_valid_mask=reset_valid)[0]
    reset_delta = float((reset_got[0] - reset_full).abs().max())
    need(max_b8_full <= HOST_GATE and max_packed_b8_cached <= HOST_GATE and max_packed_b8_full <= HOST_GATE and inactive_checked and packed_resume_checked and zero_valid_checked and reset_delta <= HOST_GATE, "B8 cached/full, packed runtime, inactive, all-zero-valid, or reset check failed")

    mixed = M1RiftNoneCachedFalconDecoder(config, str(payload_path), len(tags))
    mixed.reset([_dataset_stem(tag) for tag in tags])
    mixed_max = 0.0
    for end in range(HOST_STEPS):
        observed = np.stack([traces[tag][end] for tag in tags]).astype(np.float32, copy=False)
        got = mixed.predict(observed)
        wants = []
        for tag in tags:
            full_input, valid = _full_context(traces[tag], end)
            with torch.inference_mode():
                wants.append(static(full_input, _task_bank(tag, payload["bank_by_dataset_tag"][tag]), input_valid_mask=valid)[0].cpu().numpy())
        mixed_max = max(mixed_max, float(np.max(np.abs(got - np.stack(wants)))))
    need(mixed_max <= HOST_GATE, f"mixed-seven cached/full-R100 parity failed: {mixed_max}")

    bank = _task_bank(fixture_tag, payload["bank_by_dataset_tag"][fixture_tag])
    stream = RiftStreamDecoder(static)
    padded = None
    for index in range(100):
        observed = torch.zeros((1, 64), dtype=torch.float32) if index < 99 else torch.from_numpy(traces[fixture_tag][0:1])
        padded = stream.stream_step(observed, bank, ["query-pad"], valid_mask=torch.tensor([index >= 99]))
    first_input, first_valid = _full_context(traces[fixture_tag], 0)
    with torch.inference_mode():
        first_full = static(first_input, bank, input_valid_mask=first_valid)
    query_pad_delta = float((first_full - padded).abs().max()) if padded is not None else float("inf")
    need(query_pad_delta <= HOST_GATE, "99-bin query-padding/full-R100 parity failed")

    fixture_decoder = M1RiftNoneCachedFalconDecoder(config, str(payload_path), 1)
    fixture_decoder.reset([_dataset_stem(fixture_tag)])
    expected_trace = np.concatenate([fixture_decoder.predict(row[None, :]) for row in fixture_raw], axis=0)
    need(expected_trace.shape == (40, 16) and np.isfinite(expected_trace).all(), "fixture expected trace geometry/nonfinite drift")
    SMOKE_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with SMOKE_FIXTURE.open("xb") as handle:
        np.savez(handle, tag_stem=np.asarray(_dataset_stem(fixture_tag)), window=fixture_raw,
                 expected_trace=expected_trace, expected=expected_trace[-1])
    route = {"e0_literal_zero": all(not bool(np.any(row["E0"])) for row in payload["bank_by_dataset_tag"].values()),
             "direct_t_literal_zero": all(not bool(np.any(row["T"])) for row in payload["bank_by_dataset_tag"].values()),
             "encoder_never_called": True,
             "exact_legal_tags": set(payload["bank_by_dataset_tag"]) == TAGS}
    need(all(route.values()), "NONE route/tag validation failed")
    report = {"schema": "m1_rift_none_r100_host_verify_v1", "status": "HOST_VERIFIED",
              "payload": str(payload_path), "payload_sha256": sha256(payload_path), "build_receipt_sha256": sha256(RECEIPT),
              "selected_epoch": int(payload["selection"]["epoch"]), "per_tag": per_tag,
              "max_packed_cached_vs_full_r100": max_packed_full, "native_b8_cached_vs_full_r100_max_abs": max_b8_full,
              "native_b8_packed_vs_cached_max_abs": max_packed_b8_cached, "native_b8_packed_vs_full_r100_max_abs": max_packed_b8_full,
              "native_b8_inactive_row_state_unchanged": inactive_checked, "native_b8_packed_resume_after_inactive_checked": packed_resume_checked,
              "native_b8_reset_vs_consumed_history_max_abs": reset_delta,
              "native_b8_all_zero_valid_checked": zero_valid_checked, "mixed_seven_cached_vs_full_r100_max_abs": mixed_max,
              "query_pad_99_full_stream_max_abs": query_pad_delta, "fixture": {"path": str(SMOKE_FIXTURE), "sha256": sha256(SMOKE_FIXTURE), "raw_bins": 40, "tag": fixture_tag},
              "raw_input_contract": {"query_pad_bins_stripped": 99, "validity": "derived from time coordinate; never neural amplitude"}, "route": route}
    atomic_json_new(HOST_RECEIPT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("build", "host"))
    args = parser.parse_args()
    result = build(args.run_dir.resolve()) if args.stage == "build" else host(args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
