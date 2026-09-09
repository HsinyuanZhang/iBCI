#!/usr/bin/env python3
"""Pack the selected M1 muscle-response R100 EMA as an EvalAI image.

The selected epoch is read from a completed 24-epoch score receipt.  Epoch
selection uses the official channel-centered variance-weighted R² score, with
three held-out calibration sessions weighted equally and earliest-max ties.
This program only builds/verifies a local payload and image; it never submits.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = Path(__file__).resolve().parent
V1 = ROOT / "btransform_unified_v1"
V2 = ROOT / "btransform_unified_v2"
RUNNER = V2 / "scripts" / "m1_muscle_r100_v1"
V1_SRC, V2_SRC, V1_SCRIPTS = V1 / "src", V2 / "src", V1 / "scripts"
SOURCE = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
ALL = SOURCE + HO
N_STEPS, GATE = 120, 1.0e-5
M1_CELL = "M1-MUSCLE-R100-D4-JOINT-B3S-CONCAT-V1"
HO_WINDOWS = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
IMAGE_TAG = "m1-rift-muscle-r100-cached:v1"
METHOD_LABEL = "M1 RIFT muscle-response R100 concat selected EMA cached CPU; architecture=RIFT backend=cached identity=muscle_response16_svd4/global_rms_sealed"
METHOD_NAME = "M1 RIFT muscle-response R100 selected cached"
SKIP_PKG = {"joint_m2_model.py", "m2_mechanism_model.py", "joint_m1_model.py"}
SAFE_INIT = '''"""Container-safe RIFT decoder package."""
from .config import RiftTemporalConfig
from .model import RiftDecoder
from .streaming import RiftStreamDecoder
from .temporal import RiftTemporal, RiftTemporalState
from .cpu_temporal import CpuRiftTemporalRuntime
__all__ = ["CpuRiftTemporalRuntime", "RiftDecoder", "RiftStreamDecoder", "RiftTemporal", "RiftTemporalConfig", "RiftTemporalState"]
'''
for path in (V2_SRC, V1_SRC, V1_SCRIPTS, RUNNER, DEST, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON is missing: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return body


def selected_epoch(run_root: Path) -> tuple[int, dict[str, Any], dict[str, Any], Path]:
    """Validate all 24 official selection rows before returning earliest max."""
    score_path = run_root / "score_receipt.json"
    score = read_json(score_path)
    if score.get("schema") != "m1_muscle_r100_ho_calib_epoch_scan_v1" or score.get("status") != "COMPLETED" or score.get("cell") != M1_CELL:
        raise RuntimeError("score receipt identity/status/cell drift")
    if score.get("official_test_used") is not False:
        raise RuntimeError("score receipt must attest official_test_used=false")
    selection = score.get("selection")
    rows = score.get("ema_by_epoch")
    if not isinstance(selection, dict) or selection.get("metric") != "channel_variance_weighted_r2":
        raise RuntimeError("score receipt selection.metric must be channel_variance_weighted_r2")
    if not isinstance(rows, dict) or set(rows) != {str(i) for i in range(1, 25)}:
        raise RuntimeError("score receipt must contain exactly the 24 scored epochs")
    metric = "equal_session_mean_channel_variance_weighted_r2"
    values: dict[int, float] = {}
    for epoch in range(1, 25):
        row = rows[str(epoch)]
        if not isinstance(row, dict) or row.get("partial") is not False or int(row.get("n_windows", -1)) != sum(HO_WINDOWS.values()):
            raise RuntimeError(f"epoch {epoch} score row is malformed or partial")
        sessions = row.get("per_session")
        if not isinstance(sessions, dict) or set(sessions) != set(HO):
            raise RuntimeError(f"epoch {epoch} does not contain the three held-out sessions")
        per_session = []
        for session in HO:
            if int(sessions[session].get("window_count", -1)) != HO_WINDOWS[session]:
                raise RuntimeError(f"epoch {epoch}/{session} held-out window inventory drift")
            value = float(sessions[session].get("channel_variance_weighted_r2", np.nan))
            if not np.isfinite(value):
                raise RuntimeError(f"epoch {epoch}/{session} channel score is not finite")
            per_session.append(value)
        reported = float(row.get(metric, np.nan))
        if not np.isfinite(reported) or not np.isclose(reported, np.mean(per_session), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"epoch {epoch} channel equal-session mean drift")
        values[epoch] = reported
    expected = min(values, key=lambda epoch: (-values[epoch], epoch))
    if int(selection.get("epoch", -1)) != expected:
        raise RuntimeError(f"selection epoch {selection.get('epoch')} is not earliest channel-score maximum {expected}")
    if "equal_session_mean" not in rows[str(expected)]:
        raise RuntimeError("legacy flattened equal_session_mean must remain recorded")
    return expected, score, rows[str(expected)], score_path


def copy_pkg() -> None:
    pkg = DEST / "artifacts" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    for src, name in ((V1_SRC / "btransform_unified_v1", "btransform_unified_v1"), (V2_SRC / "btransform_unified_v2", "btransform_unified_v2")):
        dst = pkg / name
        for item in src.rglob("*.py"):
            if item.name in SKIP_PKG:
                continue
            target = dst / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    (pkg / "btransform_unified_v2" / "__init__.py").write_text(SAFE_INIT, encoding="utf-8")


def validate_training_bindings(runner: Any, run_root: Path, carrier_pack: Path, meta: Mapping[str, Any]) -> None:
    """Bind this pack to the exact training code, carrier fit, and 24-epoch run."""
    if meta.get("schema") != "m1_muscle_r100_train_v1" or meta.get("status") != "FORMAL" or meta.get("cell") != M1_CELL:
        raise RuntimeError("run root is not a formal M1 muscle R100 run")
    if meta.get("official_test_used") is not False:
        raise RuntimeError("formal run metadata must attest official_test_used=false")
    _mapping, current_binding = runner._carrier_binding(carrier_pack)
    if current_binding != meta.get("carrier_binding"):
        raise RuntimeError("current carrier binding differs from the binding used for training")
    if runner._source_hashes() != meta.get("source_hashes"):
        raise RuntimeError("training source-code/data hash inventory drift")
    receipt = read_json(run_root / "train_receipt.json")
    required = {
        "schema": "m1_muscle_r100_train_receipt_v1", "status": "COMPLETED", "cell": M1_CELL,
        "arm": meta.get("arm"), "seed": meta.get("seed"), "sampler_seed": meta.get("sampler_seed"),
        "epochs": 24, "steps": 159960, "source_hashes": meta.get("source_hashes"),
        "source_contract": meta.get("source_contract"), "b3s": meta.get("b3s"),
        "carrier_binding": meta.get("carrier_binding"), "fit_sha256": meta.get("carrier_binding", {}).get("fit_sha256"),
        "post_training_scoring_required": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("completed training receipt does not match run metadata")


def build_payload(run_root: Path, carrier_pack: Path) -> tuple[Path, str, dict[str, Any], dict[str, np.ndarray]]:
    import train as runner
    from carrier import load_carrier_pack
    from btransform_unified_v1.ema import DecoderEMA
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.joint_m1_model import ARM_D, JointM1ConcatDecoder
    from m1_rift_falcon_decoder import PAYLOAD_SCHEMA

    run_root, carrier_pack = run_root.resolve(), carrier_pack.resolve()
    meta = read_json(run_root / "run_meta.json")
    if meta.get("arm") != ARM_D:
        raise RuntimeError("run root is not the D4 joint arm")
    validate_training_bindings(runner, run_root, carrier_pack, meta)
    carriers, carrier_receipt = load_carrier_pack(carrier_pack)
    binding = meta.get("carrier_binding")
    if not isinstance(binding, dict) or binding.get("carrier_pack_npz_sha256") != sha256(carrier_pack):
        raise RuntimeError("run metadata does not bind the supplied carrier pack")
    if binding.get("carrier_pack_receipt_sha256") != sha256(carrier_pack.with_suffix(".json")):
        raise RuntimeError("run metadata carrier receipt binding drift")
    if binding.get("carrier_pack_receipt_body") != carrier_receipt:
        raise RuntimeError("run metadata carrier receipt body drift")
    epoch, score, score_row, score_path = selected_epoch(run_root)
    for key in ("cell", "arm", "seed", "sampler_seed", "source_hashes", "source_contract", "b3s", "carrier_binding", "fit_sha256"):
        if score.get(key) != meta.get(key):
            raise RuntimeError(f"score receipt {key} does not match formal run metadata")
    inventory = score.get("checkpoint_sha256_by_epoch")
    if not isinstance(inventory, dict) or set(inventory) != {str(i) for i in range(1, 25)}:
        raise RuntimeError("score receipt checkpoint SHA inventory must contain exactly 24 epochs")
    for candidate_epoch in range(1, 25):
        candidate = run_root / f"epoch_{candidate_epoch:03d}.pt"
        if not candidate.is_file() or inventory[str(candidate_epoch)] != sha256(candidate):
            raise RuntimeError(f"checkpoint SHA inventory drift at epoch {candidate_epoch}")
    checkpoint = run_root / f"epoch_{epoch:03d}.pt"
    expected_sha = score.get("checkpoint_sha256_by_epoch", {}).get(str(epoch))
    if not checkpoint.is_file() or expected_sha != sha256(checkpoint):
        raise RuntimeError("selected checkpoint does not match the score receipt")
    state = runner._validate_checkpoint(checkpoint, meta, expected_epoch=epoch)

    dataset, _sampler = runner.legacy.build_fullsession_face()
    legacy_banks, _report = runner.legacy.build_fullsession_banks(dataset)
    source_banks = runner._replace_carriers(legacy_banks, carriers, SOURCE)
    source_calib = runner.m1_plan.calib_trials_from_dataset(dataset)
    heldout = runner._ho_material(carriers)
    all_banks = {**source_banks, **{session: heldout[session]["bank"] for session in HO}}
    all_calib = {**source_calib, **{session: heldout[session]["calib10"] for session in HO}}

    joint = JointM1ConcatDecoder(ARM_D, seed=42)
    joint.install_session_memory(all_banks, all_calib)
    joint.load_state_dict(state["raw_state_dict"], strict=True)
    ema = DecoderEMA(joint, decay=float(meta["ema_decay"]))
    ema.load_state_dict(state["ema"])
    with torch.no_grad():
        for name, parameter in joint.named_parameters():
            parameter.copy_(ema.shadow[name].to(dtype=parameter.dtype, device=parameter.device))
    joint.eval()

    sealed: dict[str, dict[str, Any]] = {}
    for session in ALL:
        bank = all_banks[session]
        with torch.inference_mode():
            e0, carrier = joint._identity([bank.session_id], torch.device("cpu"))
        tag = session.removeprefix("ses-")
        e0_np = np.ascontiguousarray(e0[0].cpu().numpy(), dtype=np.float32)
        t_np = np.ascontiguousarray(carrier[0].cpu().numpy(), dtype=np.float32)
        expected_t = np.ascontiguousarray(carriers[session], dtype=np.float32)
        if not np.array_equal(t_np, expected_t):
            raise RuntimeError(f"sealed muscle carrier drift for {session}")
        sealed[tag] = {"E0": e0_np, "T": t_np, "unit_mask": np.ascontiguousarray(bank.unit_mask, dtype=np.bool_),
                       "session": bank.session_id, "e0_sha256": array_sha(e0_np), "t_sha256": array_sha(t_np)}

    concat = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42)
    allowed = set(concat.state_dict())
    ema_state = {name: value.detach().cpu().float().clone() for name, value in joint.state_dict().items() if name in allowed}
    missing, unexpected = concat.load_state_dict(ema_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"selected EMA state is not a strict concat state: missing={missing}, unexpected={unexpected}")
    concat.eval()
    host_neural = {session.removeprefix("ses-"): np.ascontiguousarray(
        (dataset.neural_data[session] if session in SOURCE else heldout[session]["dataset"].neural_data[session])[99:99 + N_STEPS], dtype=np.float32)
        for session in ALL}
    if any(value.shape != (N_STEPS, 64) for value in host_neural.values()):
        raise RuntimeError("host verification neural geometry drift")
    seal_parity: dict[str, Any] = {}
    for session in ALL:
        tag = session.removeprefix("ses-")
        source_index = next(index for index, row in enumerate(dataset.window_indices) if row[0] == session) if session in SOURCE else 0
        source_window = dataset[source_index][0] if session in SOURCE else heldout[session]["dataset"][0][0]
        window = torch.from_numpy(np.ascontiguousarray(np.asarray(source_window), dtype=np.float32)).unsqueeze(0)
        if tuple(window.shape) != (1, 100, 64):
            raise RuntimeError(f"{session}: actual W100 input geometry drift")
        valid = torch.ones((1, 100), dtype=torch.bool) if session in SOURCE else runner.valid_mask_from_padded_starts(
            (heldout[session]["starts"][0],), device=torch.device("cpu"))
        sealed_bank = dataclasses.replace(all_banks[session], E0=sealed[tag]["E0"], carrier=sealed[tag]["T"])
        with torch.inference_mode():
            live = joint(window, all_banks[session], input_valid_mask=valid)
            frozen = concat(window, sealed_bank, input_valid_mask=valid)
        maximum = float((live.float() - frozen.float()).abs().max())
        if maximum > 2e-5:
            raise RuntimeError(f"{session}: live-joint/sealed-concat parity failed: {maximum}")
        seal_parity[tag] = {"e0_sha256": sealed[tag]["e0_sha256"], "t_sha256": sealed[tag]["t_sha256"],
                            "w100_max_abs": maximum, "valid_bins": int(valid.sum())}
    selection = {"surface": "visible HO3 M10", "metric": "channel_variance_weighted_r2", "epoch": epoch,
                 "equal_session_mean_channel_variance_weighted_r2": float(score_row["equal_session_mean_channel_variance_weighted_r2"]),
                 "legacy_equal_session_mean_flattened_r2": float(score_row["equal_session_mean"]), "official_test_used": False}
    payload = {"schema": PAYLOAD_SCHEMA, "task": "m1", "context_bins": 100, "bias_mode": "recency",
               "identity_interface": "muscle_response16_svd4/global_rms_sealed", "behavior_scaling_factor": 1.0,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "run_meta_sha256": sha256(run_root / "run_meta.json"),
               "score_receipt_sha256": sha256(score_path), "carrier_pack_sha256": sha256(carrier_pack),
               "carrier_fit_sha256": binding["fit_sha256"], "bank_by_dataset_tag": sealed, "seal_validation": {
                   "status": "PASSED", "comparison": "live_joint_new_muscle_T_raw_M10_vs_sealed_concat", "per_session": seal_parity},
               "ema_state_dict": {key: value.detach().cpu().float().clone() for key, value in concat.state_dict().items()}, "selection": selection}
    dest = DEST / "artifacts" / "m1_rift_muscle_r100.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite an existing payload: {dest}")
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    return dest, sha256(dest), selection, host_neural


def host_verify(payload_path: Path, host_neural: Mapping[str, np.ndarray]) -> dict[str, Any]:
    from falcon_challenge.config import FalconConfig, FalconTask
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from btransform_unified_v2.streaming import RiftStreamDecoder
    from m1_rift_falcon_decoder import M1RiftCachedFalconDecoder, _task_bank, load_payload
    payload = load_payload(payload_path)
    config = FalconConfig(task=FalconTask.m1)
    report: dict[str, Any] = {}
    def stem(tag: str) -> str:
        kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
        return f"sub-MonkeyL-{kind}_ses-{tag}_behavior+ecephys"
    for name, tags in (("B1", [HO[0]]), ("B3", list(HO)), ("B4", [s.removeprefix("ses-") for s in SOURCE])):
        packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=[stem(tag) for tag in tags])
        ref_model = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42).eval()
        ref_model.load_state_dict({key: torch.as_tensor(value) for key, value in payload["ema_state_dict"].items()}, strict=True)
        ref = CpuRiftRuntime(ref_model, [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags], tags, temporal_backend="cached")
        maximum = 0.0
        for step in range(N_STEPS):
            x = np.stack([host_neural[tag][step] for tag in tags])
            got = packed.predict(x)
            want = ref.advance(torch.from_numpy(x)).numpy()
            maximum = max(maximum, float(np.max(np.abs(got - want))))
        if maximum > GATE:
            raise RuntimeError(f"{name} container/reference cached parity failed: {maximum}")
        report[name] = {"sessions": tags, "steps": N_STEPS, "max_abs": maximum}
    tag = HO[0]
    bank = _task_bank(tag, payload["bank_by_dataset_tag"][tag])
    model = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42).eval()
    model.load_state_dict({key: torch.as_tensor(value) for key, value in payload["ema_state_dict"].items()}, strict=True)
    window = torch.from_numpy(host_neural[tag][:100]).unsqueeze(0)
    with torch.inference_mode():
        full = model(window, bank, input_valid_mask=torch.ones((1, 100), dtype=torch.bool))
    stream = RiftStreamDecoder(model)
    for step in range(100):
        streamed = stream.stream_step(window[:, step], bank, [tag], valid_mask=torch.ones(1, dtype=torch.bool))
    delta = float((full - streamed).abs().max())
    if delta > 2e-5:
        raise RuntimeError(f"full-window/streaming parity failed: {delta}")
    smoke = DEST / "artifacts" / "smoke_window.npz"
    one = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    one.reset(dataset_tags=[stem(tag)])
    pred = None
    for row in host_neural[tag][:40]: pred = one.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(stem(tag)), window=host_neural[tag][:40], expected=pred)
    report.update({"full_window_stream_max_abs": delta, "smoke_window": str(smoke), "status": "HOST_PACK_VERIFY_PASS"})
    return report


def docker_build(payload_sha: str, method_label: str) -> dict[str, Any]:
    subprocess.run(["docker", "build", "--build-arg", f"PAYLOAD_SHA256={payload_sha}", "--build-arg", f"METHOD_LABEL={method_label}", "-t", IMAGE_TAG, str(DEST)], check=True)
    image_id, size = subprocess.check_output(["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}} {{.Size}}"], text=True).strip().split()
    return {"image_tag": IMAGE_TAG, "image_id": image_id, "image_size": int(size)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--carrier-pack", type=Path, required=True)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    copy_pkg()
    payload, payload_sha, selection, neural = build_payload(args.run_root, args.carrier_pack)
    host = host_verify(payload, neural)
    method_label = f"{METHOD_LABEL}; selected_epoch={selection['epoch']}"
    method_name = f"{METHOD_NAME} e{selection['epoch']}"
    method_description = ("M1 RIFT R100/D4 with sealed muscle-response carrier and B3S EMA E0. "
                          f"HO3 channel-centered variance-weighted R2 pick e{selection['epoch']}: "
                          f"{selection['equal_session_mean_channel_variance_weighted_r2']:.6f}.")
    budget_disclosure = "Official seven-tag M10 banks; visible held-out calibration used only for post-training EMA selection; no official test, no TTA."
    docker = None if args.skip_docker else docker_build(payload_sha, method_label)
    candidate = {"arm": "m1_rift_muscle_r100_cached", "evalai_opened": False, "register": False, "payload_sha256": payload_sha,
                 "selection_metric": selection["metric"], "selection_epoch": selection["epoch"],
                 "selection_mean": selection["equal_session_mean_channel_variance_weighted_r2"],
                 "legacy_selection_mean_flattened_r2": selection["legacy_equal_session_mean_flattened_r2"], "host_verify": host,
                 "method_label": method_label, "method_name": method_name, "method_description": method_description,
                 "budget_disclosure": budget_disclosure, "image_tag": None, "image_id": None, "image_size": None,
                 "state_path": str(DEST / "artifacts" / "evalai_push_state.json"), "window": 100}
    if docker:
        candidate.update(docker)
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PACKED_NOT_REGISTERED", "payload": str(payload), "selection": selection, "docker": docker, "host_verify": {k: v for k, v in host.items() if k != "smoke_window"}}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
