#!/usr/bin/env python3
"""Build the M1 RIFT joint-D e3 cached payload, host-verify, and docker-pack.

Seals live B3S E0 computed from the EMA encoder + official M10/rSyn3.
Does not EvalAI-submit. Does not overwrite 582045 or H1 582073.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = Path(__file__).resolve().parent
CKPT = ROOT / "btransform_unified_v2/results/rift_v1/m1_r100_joint_d_s42_formal_v1/epoch_003.pt"
CKPT_SHA = "d7eba8ba9cea88b1290a68ed86fa3a75d51a11256747c6232fcb1c17b341efba"
SCORE = ROOT / "btransform_unified_v2/results/rift_v1/m1_r100_joint_d_s42_formal_v1/score_receipt.json"
BANK_PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_m1_projadd_depth2_ort_v1/artifacts/m1_projadd_p16_d2_s42_ema_e21.pkl"
SFIX = ROOT / "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt"
SFIX_SHA = "7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a"
V1_SRC = ROOT / "btransform_unified_v1/src"
V2_SRC = ROOT / "btransform_unified_v2/src"
V1_SCRIPTS = ROOT / "btransform_unified_v1/scripts"
V2_SCRIPTS = ROOT / "btransform_unified_v2/scripts"
IMAGE_TAG = "m1-rift-joint-d-cached-e3:v1"
METHOD_LABEL = (
    "M1 RIFT joint D R100 concat e3 EMA cached CPU; architecture=RIFT backend=cached "
    "identity=live_b3s_concat_sealed; not BT-EORT; not ORT; not SPINT"
)
METHOD_NAME = "M1 RIFT joint D R100 concat e3 cached"
SOURCE = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
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

for path in (str(V2_SRC), str(V1_SRC), str(V1_SCRIPTS), str(V2_SCRIPTS / "rift_v1"), str(DEST), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _copy_pkg() -> None:
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


def _official_banks() -> dict:
    from m1_rift_falcon_decoder import CPUUnpickler

    with open(BANK_PAYLOAD, "rb") as handle:
        payload = CPUUnpickler(handle).load()
    banks = payload["bank_by_dataset_tag"]
    if set(banks) != {s.removeprefix("ses-") for s in SOURCE} | set(HO):
        raise RuntimeError(f"official M1 tag set drifted: {sorted(banks)}")
    return banks


def _build_payload() -> tuple[Path, str, dict]:
    from btransform_unified_v1.ema import DecoderEMA
    from btransform_unified_v1.m1_b3s_joint import encode_b3s
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.joint_m1_model import ARM_D, JointM1ConcatDecoder
    from m1_rift_falcon_decoder import PAYLOAD_SCHEMA
    import m1_joint_train as joint
    import m1_projadd_depth2_series as legacy
    from btransform_unified_v1 import m1_projadd as mp

    if sha256(CKPT) != CKPT_SHA:
        raise RuntimeError("epoch_003.pt SHA drift")
    if sha256(SFIX) != SFIX_SHA:
        raise RuntimeError("B3 Sfix e11 SHA drift")
    score = json.loads(SCORE.read_text())
    if int(score["selection"]["epoch"]) != 3:
        raise RuntimeError(f"score selection drifted: {score['selection']}")
    official = _official_banks()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "m1_rift_joint_epoch_checkpoint_v1" or int(ckpt.get("epoch", -1)) != 3:
        raise RuntimeError("checkpoint is not formal M1 joint D e3")
    if ckpt.get("config", {}).get("arm") != ARM_D:
        raise RuntimeError("checkpoint arm is not D_JOINT")

    joint_model = JointM1ConcatDecoder(ARM_D, seed=42)
    joint_model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    ema = DecoderEMA(joint_model, decay=0.9995)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_to(joint_model)
    joint_model.eval()

    dataset, _sampler = legacy.build_fullsession_face()
    source_banks, _report = legacy.build_fullsession_banks(dataset)
    source_calib = mp.calib_trials_from_dataset(dataset)
    ho_material = joint._ho_material()
    install_banks = {name: source_banks[name] for name in SOURCE}
    install_calib = {name: source_calib[name] for name in SOURCE}
    for session in HO:
        install_banks[session] = ho_material[session]["bank"]
        install_calib[session] = ho_material[session]["calib10"]
    joint_model.install_session_memory(install_banks, install_calib)
    joint_model.eval()

    sealed = {}
    for tag, row in official.items():
        session = tag if tag in install_banks else f"ses-{tag}"
        bank = install_banks[session]
        official_t = np.ascontiguousarray(row["T"], dtype=np.float32)
        if not np.array_equal(official_t, np.ascontiguousarray(bank.carrier, dtype=np.float32)):
            raise RuntimeError(f"rSyn3 T drift vs 582045 at {tag}")
        with torch.inference_mode():
            e0, carrier = joint_model._identity([bank.session_id], torch.device("cpu"))
        e0_np = np.ascontiguousarray(e0[0].cpu().numpy(), dtype=np.float32)
        t_np = np.ascontiguousarray(carrier[0].cpu().numpy(), dtype=np.float32)
        if not np.array_equal(t_np, official_t):
            raise RuntimeError(f"live carrier T drift vs 582045 at {tag}")
        sealed[tag] = {
            "E0": e0_np,
            "T": official_t,
            "unit_mask": np.ascontiguousarray(row["unit_mask"], dtype=np.bool_),
            "session": tag,
            "e0_sha256": _array_sha(e0_np),
            "t_sha256": _array_sha(official_t),
        }

    concat = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42)
    allowed = set(concat.state_dict())
    ema_state = {k: v.detach().cpu().float().clone() for k, v in joint_model.state_dict().items() if k in allowed}
    missing, unexpected = concat.load_state_dict(ema_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"concat state mismatch missing={missing} unexpected={unexpected}")
    concat.eval()

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "task": "m1",
        "context_bins": 100,
        "bias_mode": "recency",
        "identity_interface": "live_b3s_concat_sealed",
        "behavior_scaling_factor": 1.0,
        "checkpoint": str(CKPT),
        "checkpoint_sha256": CKPT_SHA,
        "sfix_sha256": SFIX_SHA,
        "bank_source_payload_sha256": sha256(BANK_PAYLOAD),
        "bank_by_dataset_tag": sealed,
        "ema_state_dict": {k: v.detach().cpu().float().clone() for k, v in concat.state_dict().items()},
        "selection": {
            "surface": "visible HO3 M10",
            "epoch": 3,
            "equal_session_mean": float(score["ema_by_epoch"]["3"]["equal_session_mean"]),
            "official_test_used": False,
        },
        "ho_neural": {
            session: np.ascontiguousarray(ho_material[session]["dataset"].neural_data[session][:N_STEPS], dtype=np.float32)
            for session in HO
        },
    }
    dest = DEST / "artifacts" / "m1_rift_joint_d_e3.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # do not pickle host-only neural into the submission payload
    host_neural = payload.pop("ho_neural")
    with dest.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    payload["ho_neural"] = host_neural
    return dest, sha256(dest), payload["selection"], host_neural


def _host_verify(payload_path: Path, host_neural: dict[str, np.ndarray]) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
    from m1_rift_falcon_decoder import M1RiftCachedFalconDecoder, _task_bank, load_payload

    payload = load_payload(payload_path)
    config = FalconConfig(task=FalconTask.m1)
    report = {}
    def _stem(tag: str) -> str:
        kind = "held-in-calib" if tag.startswith("201209") else "held-out-calib"
        return f"sub-MonkeyL-{kind}_ses-{tag}_behavior+ecephys"

    plans = (("B1", ["20121004"]), ("B3", ["20121004", "20121017", "20121024"]))
    for name, tags in plans:
        streams = [host_neural[tag] for tag in tags]
        packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=len(tags))
        packed.reset(dataset_tags=[_stem(tag) for tag in tags])
        model = RiftConcatDecoder("m1", context_bins=100, bias_mode="recency", seed=42)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in payload["ema_state_dict"].items()}, strict=True)
        model.eval()
        banks = [_task_bank(tag, payload["bank_by_dataset_tag"][tag]) for tag in tags]
        ref = CpuRiftRuntime(model, banks, [str(tag) for tag in tags], temporal_backend="cached")
        max_abs = 0.0
        last = None
        for step in range(N_STEPS):
            batch_x = np.stack([row[step] for row in streams], axis=0)
            got = packed.predict(batch_x)
            want = ref.advance(torch.from_numpy(batch_x)).numpy()
            last = got
            max_abs = max(max_abs, float(np.max(np.abs(got - want))))
            if max_abs > GATE:
                raise RuntimeError(f"{name} host gate fail t={step} max_abs={max_abs}")
        report[name] = {
            "steps": N_STEPS,
            "max_abs": max_abs,
            "last_shape": list(last.shape),
            "sessions": tags,
        }
    smoke = DEST / "artifacts" / "smoke_window.npz"
    tag = "20121004"
    neural = host_neural[tag][:40]
    packed = M1RiftCachedFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[_stem(tag)])
    pred = None
    for row in neural:
        pred = packed.predict(row.reshape(1, -1))
    np.savez(smoke, tag_stem=np.asarray(tag), window=neural, expected=pred)
    report["smoke_window"] = str(smoke)
    report["status"] = "HOST_PACK_VERIFY_PASS"
    return report


def _docker_build(payload_sha: str) -> dict:
    cmd = ["docker", "build", "--build-arg", f"PAYLOAD_SHA256={payload_sha}", "-t", IMAGE_TAG, str(DEST)]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(["docker", "image", "inspect", IMAGE_TAG, "--format", "{{.Id}} {{.Size}}"], text=True).strip()
    image_id, size = inspect.split()
    return {"image_tag": IMAGE_TAG, "image_id": image_id, "image_size": int(size)}


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _copy_pkg()
    payload_path, payload_sha, selection, host_neural = _build_payload()
    host = _host_verify(payload_path, host_neural)
    docker = _docker_build(payload_sha)
    candidate = {
        "arm": "m1_rift_joint_d_e3_cached",
        "budget_disclosure": (
            "Official 7-tag M1 banks (M10 raw + sealed rSyn3 T from 582045). "
            "Live B3S E0 sealed from joint-D EMA e3. Visible HO3 equal-session "
            f"{selection['equal_session_mean']:.6f} is development selection, not official HO. "
            "Cached CPU runtime, no TTA, no ORT."
        ),
        "evalai_opened": False,
        "image_id": docker["image_id"],
        "image_size": docker["image_size"],
        "image_tag": docker["image_tag"],
        "method_description": (
            "M1 RIFT joint D (live B3S + concat R100/D4), recency, cached CPU KV. "
            f"Same rSyn3 T as 582045; E0 is trained B3S on M10. HO3 pick e3 mean {selection['equal_session_mean']:.6f}. "
            "Not BT-EORT, not ORT, not SPINT, not 582045."
        ),
        "method_label": METHOD_LABEL,
        "method_name": METHOD_NAME,
        "payload_sha256": payload_sha,
        "register": False,
        "selection_mean": selection["equal_session_mean"],
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "window": 100,
        "host_verify": host,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PACKED_NOT_REGISTERED",
                **{k: candidate[k] for k in ("image_tag", "image_id", "payload_sha256", "image_size")},
                "host_verify": {k: host[k] for k in host if k != "smoke_window"},
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
