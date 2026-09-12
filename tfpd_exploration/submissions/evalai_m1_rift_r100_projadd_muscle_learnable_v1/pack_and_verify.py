#!/usr/bin/env python3
"""Pack M1 proj_add + muscle learned_slope after a completed HO3 all24 scan.

Banks keep legacy fullsession E0 and replace only the 64x4 T carrier from
carrier_official4 (SHA 68dc80c8734f4310c0ec378f599fd041a439906e90973db4351d582c9a30887e).
Does not EvalAI-submit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = Path(__file__).resolve().parent
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
LEARNABLE_SRC = ROOT / "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1"
FROZEN_DIR = ROOT / "btransform_unified_v2/scripts/m1_muscle_r100_v1"
CARRIER_PACK = ROOT / "btransform_unified_v2/results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz"
CARRIER_SHA = "68dc80c8734f4310c0ec378f599fd041a439906e90973db4351d582c9a30887e"
SOURCE = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
ALL = SOURCE + HO
HO_WINDOWS = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
SCORE_SCHEMA = "m1_projadd_muscle_learnable_ho_calib_epoch_scan_v1"
CKPT_SCHEMA = "m1_projadd_muscle_learnable_epoch_checkpoint_v1"
CELL = "M1-MUSCLE-R100-D4-P16-PROJADD-LEARNABLE-LEARNED_SLOPE-V1"
N_STEPS = 80
GATE = 1.0e-5
SKIP_PKG = {"joint_m2_model.py", "m2_mechanism_model.py", "joint_m1_model.py"}
SAFE_INIT = '''"""Container-safe RIFT decoder package. Joint/mechanism modules are omitted."""

from .config import RiftTemporalConfig
from .model import RiftDecoder
from .streaming import RiftStreamDecoder
from .temporal import RiftTemporal, RiftTemporalState
from .cpu_temporal import CpuRiftTemporalRuntime

__all__ = [
    "CpuRiftTemporalRuntime",
    "RiftDecoder",
    "RiftStreamDecoder",
    "RiftTemporal",
    "RiftTemporalConfig",
    "RiftTemporalState",
]
'''
LEARNABLE_SAFE_INIT = '''"""Container-safe learnable recency package. wrap.py / JointM1 are omitted."""

from .config import LearnableRecencyConfig, config_from_run_meta
from .cpu_temporal import CpuLearnableRecencyRuntime
from .temporal import LearnableRecencyTemporal

__all__ = [
    "CpuLearnableRecencyRuntime",
    "LearnableRecencyConfig",
    "LearnableRecencyTemporal",
    "config_from_run_meta",
]
'''


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


def _copy_pkg() -> Path:
    pkg = DEST / "artifacts" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    for src, name in ((V1_SRC / "btransform_unified_v1", "btransform_unified_v1"), (V2_SRC / "btransform_unified_v2", "btransform_unified_v2")):
        dst = pkg / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*.py"):
            if item.name in SKIP_PKG:
                continue
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    (pkg / "btransform_unified_v2" / "__init__.py").write_text(SAFE_INIT)
    lr = pkg / "learnable_recency_v1"
    lr.mkdir(parents=True, exist_ok=True)
    for name in ("config.py", "temporal.py", "cpu_temporal.py"):
        shutil.copy2(LEARNABLE_SRC / name, lr / name)
    (lr / "__init__.py").write_text(LEARNABLE_SAFE_INIT)
    return pkg


def _use_live_tree() -> None:
    for path in (
        ROOT / "streaming_calibration_exp",
        ROOT,
        V1_SRC,
        ROOT / "btransform_unified_v1/scripts",
        V2_SRC,
        LEARNABLE_SRC.parent,
        FROZEN_DIR,
        DEST,
    ):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _use_packed_pkg(pkg: Path) -> None:
    for path in (str(ROOT / "streaming_calibration_exp"), str(ROOT), str(pkg), str(DEST), str(FROZEN_DIR)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_frozen():
    spec = importlib.util.spec_from_file_location("_m1_muscle_frozen_train", FROZEN_DIR / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen muscle trainer")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stem(tag: str) -> str:
    kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
    return f"sub-MonkeyL-{kind}_ses-{tag}_behavior+ecephys"


def selected_epoch(run_root: Path) -> tuple[int, dict[str, Any], dict[str, Any]]:
    score = read_json(run_root / "score_receipt.json")
    if score.get("schema") != SCORE_SCHEMA or score.get("status") != "COMPLETED" or score.get("cell") != CELL:
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
    return expected, score, rows[str(expected)]


def _build_payload(run_root: Path) -> tuple[Path, str, dict[str, Any], dict[str, np.ndarray]]:
    from btransform_unified_v1.ema import DecoderEMA
    from learnable_recency_v1.config import config_from_run_meta
    from m1_rift_falcon_decoder import PAYLOAD_SCHEMA, build_decoder

    run_root = run_root.resolve()
    if sha256(CARRIER_PACK) != CARRIER_SHA:
        raise RuntimeError("official4 carrier pack SHA drift")
    meta = read_json(run_root / "run_meta.json")
    if meta.get("cell") != CELL or meta.get("identity_interface") != "proj_add":
        raise RuntimeError("run_meta is not muscle proj_add learned P16")
    if meta.get("carrier_variant") != "muscle_response16_svd4/global_rms":
        raise RuntimeError("run_meta carrier_variant drift")
    binding = meta.get("carrier_binding")
    if not isinstance(binding, dict) or binding.get("carrier_pack_npz_sha256") != CARRIER_SHA:
        raise RuntimeError("run_meta is not bound to carrier_official4")
    recency_cfg = config_from_run_meta(meta, "m1")
    if recency_cfg.ladder != "default" or recency_cfg.tier != "learned_slope" or not recency_cfg.per_layer:
        raise RuntimeError(f"run_meta learnable_config drifted: {recency_cfg}")
    seed = int(meta["seed"])
    epoch, score, score_row = selected_epoch(run_root)
    inventory = score.get("checkpoint_sha256_by_epoch")
    if not isinstance(inventory, dict) or set(inventory) != {str(i) for i in range(1, 25)}:
        raise RuntimeError("score receipt checkpoint SHA inventory must contain exactly 24 epochs")
    for candidate_epoch in range(1, 25):
        candidate = run_root / f"epoch_{candidate_epoch:03d}.pt"
        if not candidate.is_file() or inventory[str(candidate_epoch)] != sha256(candidate):
            raise RuntimeError(f"checkpoint SHA inventory drift at epoch {candidate_epoch}")
    ckpt_path = run_root / f"epoch_{epoch:03d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != CKPT_SCHEMA or int(ckpt.get("epoch", -1)) != epoch or ckpt.get("tier") != "learned_slope":
        raise RuntimeError("selected checkpoint schema/epoch/tier drift")

    frozen = _load_frozen()
    from carrier import load_carrier_pack

    carriers, _receipt = load_carrier_pack(CARRIER_PACK)
    dataset, _sampler = frozen.legacy.build_fullsession_face()
    legacy_banks, _report = frozen.legacy.build_fullsession_banks(dataset)
    source_banks = frozen._replace_carriers(legacy_banks, carriers, SOURCE)
    heldout = frozen._ho_material(carriers)
    all_banks = {**source_banks, **{session: heldout[session]["bank"] for session in HO}}
    sealed: dict[str, dict[str, Any]] = {}
    for session in ALL:
        bank = all_banks[session]
        tag = session.removeprefix("ses-")
        expected_t = np.ascontiguousarray(carriers[session], dtype=np.float32)
        t_np = np.ascontiguousarray(bank.carrier, dtype=np.float32)
        if not np.array_equal(t_np, expected_t):
            raise RuntimeError(f"muscle carrier drift for {session}")
        e0_np = np.ascontiguousarray(bank.E0, dtype=np.float32)
        sealed[tag] = {
            "E0": e0_np,
            "T": t_np,
            "unit_mask": np.ascontiguousarray(bank.unit_mask, dtype=np.bool_),
            "session": bank.session_id,
            "e0_sha256": array_sha(e0_np),
            "t_sha256": array_sha(t_np),
        }

    model = build_decoder(recency_cfg, seed)
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(model)
    model.eval()
    ema_state = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}
    selection = {
        "surface": "visible HO3 M10",
        "metric": "channel_variance_weighted_r2",
        "epoch": epoch,
        "seed": seed,
        "equal_session_mean_channel_variance_weighted_r2": float(score_row["equal_session_mean_channel_variance_weighted_r2"]),
        "legacy_equal_session_mean_flattened_r2": float(score_row["equal_session_mean"]),
        "n_windows": 3881,
        "official_test_used": False,
    }
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "m1",
        "context_bins": 100,
        "bias_mode": "learnable_learned_slope",
        "identity_interface": "proj_add",
        "proj_dim": 16,
        "tier": "learned_slope",
        "ladder": "default",
        "per_layer": True,
        "seed": seed,
        "carrier_variant": "muscle_response16_svd4/global_rms",
        "carrier_pack_sha256": CARRIER_SHA,
        "behavior_scaling_factor": 1.0,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "learnable_config": dict(meta["learnable_config"]),
        "bank_by_dataset_tag": sealed,
        "ema_state_dict": ema_state,
        "selection": selection,
    }
    dest = DEST / "artifacts" / "m1_rift_r100_projadd_muscle_learnable.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    traces = {
        session.removeprefix("ses-"): np.ascontiguousarray(
            (dataset.neural_data[session] if session in SOURCE else heldout[session]["dataset"].neural_data[session])[99:99 + N_STEPS],
            dtype=np.float32,
        )
        for session in ALL
    }
    return dest, sha256(dest), selection, traces


def _host_verify(payload_path: Path, traces: dict[str, np.ndarray]) -> dict[str, Any]:
    from falcon_challenge.config import FalconConfig, FalconTask
    from m1_rift_falcon_decoder import (
        CpuLearnableRiftRuntime,
        M1RiftCachedFalconDecoder,
        _task_bank,
        build_decoder,
        load_payload,
        recency_config_from_payload,
    )

    payload = load_payload(payload_path)
    config = FalconConfig(task=FalconTask.m1)
    report: dict[str, Any] = {}
    recency_cfg = recency_config_from_payload(payload)
    seed = int(payload["seed"])
    plans = (
        ("B1", ["20121004"]),
        ("B3", list(HO)),
        ("B4", [session.removeprefix("ses-") for session in SOURCE]),
    )
    for name, tags in plans:
        packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=[_stem(tag) for tag in tags])
        model = build_decoder(recency_cfg, seed)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuLearnableRiftRuntime(model, banks, [str(tag) for tag in tags])
        max_abs = 0.0
        last = None
        for step in range(N_STEPS):
            batch_x = np.stack([traces[tag][step] for tag in tags], axis=0)
            got = packed.predict(batch_x)
            want = ref.advance(torch.from_numpy(batch_x)).numpy() / 1.0
            last = got
            max_abs = max(max_abs, float(np.max(np.abs(got - want))))
            if max_abs > GATE:
                raise RuntimeError(f"{name} host gate fail t={step} max_abs={max_abs}")
        report[name] = {"steps": N_STEPS, "max_abs": max_abs, "last_shape": list(last.shape), "sessions": tags}

    smoke = DEST / "artifacts" / "smoke_window.npz"
    tag = "20121004"
    neural = np.ascontiguousarray(traces[tag][:40], dtype=np.float32)
    packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[_stem(tag)])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(_stem(tag)), window=neural, expected=pred)
    report["smoke_window"] = str(smoke)
    report["status"] = "HOST_PACK_VERIFY_PASS"
    return report


def _docker_build(payload_sha: str, image_tag: str, method_label: str) -> dict[str, Any]:
    cmd = [
        "docker",
        "build",
        "--build-arg",
        f"PAYLOAD_SHA256={payload_sha}",
        "--build-arg",
        f"METHOD_LABEL={method_label}",
        "-t",
        image_tag,
        str(DEST),
    ]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(["docker", "image", "inspect", image_tag, "--format", "{{.Id}} {{.Size}}"], text=True).strip()
    image_id, size = inspect.split()
    return {"image_tag": image_tag, "image_id": image_id, "image_size": int(size)}


def _container_smoke(smoke_path: Path, image_tag: str) -> dict[str, Any]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{smoke_path.resolve()}:/tmp/smoke.npz:ro",
        image_tag,
        "python",
        "/m1_rift_falcon_decoder.py",
        "--smoke-payload",
        "/data/decoder.pkl",
        "--smoke-window",
        "/tmp/smoke.npz",
    ]
    raw = subprocess.check_output(cmd, text=True).strip()
    last = raw.splitlines()[-1]
    parsed = json.loads(last)
    if parsed.get("status") != "CONTAINER_SMOKE_PASS":
        raise RuntimeError(f"container smoke unexpected: {raw}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("[pack] 1/5 copying package modules...", flush=True)
    pkg = _copy_pkg()
    _use_live_tree()

    print("[pack] 2/5 building payload pkl...", flush=True)
    payload_path, payload_sha, selection, traces = _build_payload(args.run_root)
    _use_packed_pkg(pkg)
    print(f"[pack] payload {payload_path.name} sha256={payload_sha} seed={selection['seed']} e{selection['epoch']}", flush=True)

    print("[pack] 3/5 running host-verify (B1/B3/B7)...", flush=True)
    host_report = _host_verify(payload_path, traces)
    (DEST / "artifacts" / "host_verify.json").write_text(json.dumps(host_report, indent=2, sort_keys=True) + "\n")
    print(f"[pack] host verify PASS: {host_report}", flush=True)

    image_tag = f"m1-rift-r100-projadd-muscle-learnable-s{selection['seed']}-e{selection['epoch']}:v1"
    method_label = (
        f"M1 RIFT R100 proj_add P16 learned_slope muscle official4 s{selection['seed']} e{selection['epoch']} "
        f"cached CPU; architecture=RIFT backend=cached identity=proj_add "
        f"carrier=muscle_response16_svd4/global_rms; HO3 ch-var "
        f"{selection['equal_session_mean_channel_variance_weighted_r2']:.6f}"
    )
    method_name = f"M1 RIFT muscle proj_add learned s{selection['seed']} e{selection['epoch']}"
    method_description = (
        "M1 RIFT R100 D4/width256 proj_add P16, per-layer learned recency slopes on the default ladder, "
        f"muscle_response16_svd4/global_rms official4 carrier. HO3 channel-variance pick e{selection['epoch']} "
        f"seed {selection['seed']}: {selection['equal_session_mean_channel_variance_weighted_r2']:.6f}. "
        "Not BT-EORT, not ORT, not SPINT."
    )

    if args.skip_docker:
        print("[pack] --skip-docker set, exiting after host verify", flush=True)
        return

    print("[pack] 4/5 building Docker image...", flush=True)
    build_info = _docker_build(payload_sha, image_tag, method_label)
    print(f"[pack] docker build ok: {build_info}", flush=True)

    print("[pack] 5/5 running container smoke test...", flush=True)
    smoke_info = _container_smoke(Path(host_report["smoke_window"]), image_tag)
    print(f"[pack] container smoke PASS: {smoke_info}", flush=True)

    candidate = {
        "schema_version": "m1_rift_evalai_candidate_v1",
        "arm": "m1_rift_r100_projadd_muscle_learnable_cached",
        "image_tag": image_tag,
        "image_id": build_info["image_id"],
        "image_size": build_info["image_size"],
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "method_name": method_name,
        "method_description": method_description,
        "method_label": method_label,
        "budget_disclosure": (
            "Official seven-tag M10 banks; visible held-out calibration used only for post-training "
            "EMA selection; no official test, no TTA."
        ),
        "selection": selection,
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "receipt_path": str(DEST / "artifacts" / "evalai_submit_receipt.json"),
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print("[pack] candidate manifest written to artifacts/evalai_candidate.json", flush=True)


if __name__ == "__main__":
    main()
