#!/usr/bin/env python3
"""Pack M1 proj_add P16 depth-2 e21 EMA into an ORT exact-E image. Does not register."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT), str(WORKSPACE_ROOT / "SPINT-main"), str(PACKAGE_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import m1_projadd as mp
from btransform_unified_v1 import plan
from btransform_unified_v1.bank import array_sha256
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import export_state_numpy
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.payload import sha256_file

from pack_m1_projadd_exacte_v1 import HO_CALIB, _file_sha256, _ho_calib_trials, _official_ho_carriers, _row

TRAIN_DEST = PACKAGE_ROOT / "results/m1_projadd_depth2/fullsession_20260907_gpu0"
PICK_PATH = TRAIN_DEST / "depth2/epoch_pick.json"
DEST = Path(
    os.environ.get(
        "M1_D2_ORT_DEST",
        str(WORKSPACE_ROOT / "tfpd_exploration/submissions/evalai_m1_projadd_depth2_ort_v1"),
    )
)
WHEEL_SRC = Path(
    os.environ.get(
        "ORT_WHEEL",
        str(
            PACKAGE_ROOT
            / "results/m1_projadd_runtime_v1/20260907T020150Z/scripts/vendor"
            / "onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        ),
    )
)
PAYLOAD_SCHEMA = "m1_projadd_exacte_falcon_payload_v1"
BASE_IMAGE = "spint-original-m1:e9-epoch019-052e9ea"
SEALED_CKPT_SHA = "1a6e26f30a79c19a4ad7672db86a7c2eb73a3e0634925ff1b3278afdb293a3fe"
EXPECTED_PARAMS = 2479408
PROJ_DIM = 16
TOKEN_IN = 20
DEPTH = 2
EPOCH = 21
GATE = 1.0e-5


def _apply_ema(model, ckpt: dict) -> None:
    model.load_state_dict(ckpt["raw_state_dict"], strict=True)
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def _build_depth2() -> BTransformerUnifiedDecoderIdentity:
    model = BTransformerUnifiedDecoderIdentity(
        mp.m1_projadd_geometry(PROJ_DIM),
        seed=42,
        identity_mode="proj_add",
        proj_dim=PROJ_DIM,
        temporal_layers=DEPTH,
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    plan.require(n_params == EXPECTED_PARAMS, f"param count {n_params} != {EXPECTED_PARAMS}")
    plan.require(len(model.temporal.blocks) == DEPTH, "temporal depth drift")
    return model


def _load_deploy_banks() -> dict:
    inventory = json.loads((TRAIN_DEST / "session_inventory.json").read_text(encoding="utf-8"))
    plan.require(set(inventory["sessions"]) == set(mp.M1_SESSIONS), f"session drift {inventory['sessions']}")
    plan.require(
        "ALL 4 held-in" in str(inventory.get("train_protocol", "")),
        "refusing a LOSO/source-only dest: submission candidate is stage-2 all-4 held-in",
    )
    plan.require(int(inventory["total_train_windows"]) == 213336, "train window count drift")
    dm = mp.build_loso_datamodule()
    dataset, _ = mp.assemble_training_universe(dm)
    calib = mp.calib_trials_from_dataset(dataset)
    carriers = mp.load_source_carriers()
    outer, _ = mp.encode_outer_carrier()
    carriers = {**carriers, mp.M1_OUTER_SESSION: outer}
    train_banks, _ = mp.build_banks(calib, carriers)
    for session in mp.M1_SESSIONS:
        plan.require(
            array_sha256(train_banks[session].E0) == inventory["banks"][session]["e0_sha256"],
            f"train E0 SHA drift vs inventory: {session}",
        )
        plan.require(
            array_sha256(train_banks[session].carrier) == inventory["banks"][session]["carrier_sha256"],
            f"train T SHA drift vs inventory: {session}",
        )
    banks = {
        session.removeprefix("ses-"): _row(
            train_banks[session].E0, train_banks[session].carrier, train_banks[session].unit_mask
        )
        for session in mp.M1_SESSIONS
    }
    encoder = mp.load_b3_id_encoder()
    ho_t = _official_ho_carriers()
    pick = json.loads(PICK_PATH.read_text(encoding="utf-8"))
    for session, spec in HO_CALIB.items():
        path = spec["path"]
        plan.require(path.is_file(), f"missing official calib {path}")
        plan.require(_file_sha256(path) == spec["sha256"], f"official calib SHA drift: {session}")
        e0 = mp.b3_identity(encoder, _ho_calib_trials(dm, path, session))
        tag = spec["tag"]
        banks[tag] = _row(e0, ho_t[tag], np.ones(mp.M1_UNITS, dtype=np.bool_))
        sealed = pick["sessions"][tag]["bank"]
        plan.require(array_sha256(e0) == sealed["e0_sha256"], f"HO E0 SHA drift vs pick: {tag}")
        plan.require(array_sha256(ho_t[tag]) == sealed["carrier_sha256"], f"HO T SHA drift vs pick: {tag}")
    plan.require(len(banks) == 7, f"deploy banks {len(banks)} != 7")
    return banks


def _run_stream(decoder, stream: np.ndarray) -> np.ndarray:
    pred = None
    for row in stream:
        pred = decoder.predict(row.reshape(1, -1))
    plan.require(pred is not None, "empty stream")
    return np.asarray(pred, dtype=np.float32)


def main() -> dict:
    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    pick = json.loads(PICK_PATH.read_text(encoding="utf-8"))
    selected = pick["selected"]
    plan.require(int(selected["epoch"]) == EPOCH, f"expected e{EPOCH}")
    plan.require(selected["ckpt_sha256"] == SEALED_CKPT_SHA, "pick SHA drift vs sealed")
    ckpt_path = Path(selected["ckpt"])
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")
    plan.require(_file_sha256(ckpt_path) == SEALED_CKPT_SHA, "checkpoint file SHA drift")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts").mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts" / "ort_graphs").mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts" / "vendor").mkdir(parents=True, exist_ok=True)
    payload_name = "m1_projadd_p16_d2_s42_ema_e21.pkl"
    payload_path = DEST / "artifacts" / payload_name
    plan.require(not payload_path.exists(), f"refusing to overwrite {payload_path}")
    wheel_dest = DEST / "artifacts" / "vendor" / WHEEL_SRC.name
    if not wheel_dest.is_file():
        shutil.copy2(WHEEL_SRC, wheel_dest)
    os.environ.setdefault("RT_SUBMITTED_DECODER", str(DEST / "trf_falcon_decoder.py"))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _build_depth2()
    _apply_ema(model, ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight_sha = tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()})
    banks = _load_deploy_banks()

    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank, data as src_data

    loaded = src_bank.load()
    dm = src_data.build_source_only_datamodule(loaded)
    session = "ses-20120926"
    neural = np.asarray(dm.train_dataset.base.neural_data[session], dtype=np.float32)
    pad = 99
    window = np.ascontiguousarray(neural[pad : pad + mp.M1_WINDOW], dtype=np.float32)
    long_stream = np.ascontiguousarray(neural[pad : pad + mp.M1_WINDOW + 100], dtype=np.float32)
    plan.require(window.shape == (mp.M1_WINDOW, mp.M1_UNITS), f"smoke window {window.shape}")
    official = banks["20120926"]
    task_bank = mp.make_m1_bank(session, official["E0"], official["T"], unit_mask=official["unit_mask"])
    x = torch.from_numpy(window[None])
    with torch.inference_mode():
        train_pred = model(x, task_bank).detach().cpu().numpy()

    config = FalconConfig(task=FalconTask.m1)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": config.task,
        "kind": "proj_add",
        "state_dict": export_state_numpy(model),
        "bank_by_dataset_tag": banks,
        "window_size": mp.M1_WINDOW,
        "behavior_scaling_factor": mp.M1_DIVISOR,
        "smooth_observations": False,
        "metadata": {
            "kind": "proj_add",
            "cell": "M1-PROJADD-P16-D2",
            "seed": 42,
            "epoch": EPOCH,
            "view": "EMA",
            "proj_dim": PROJ_DIM,
            "token_in": TOKEN_IN,
            "temporal_layers": DEPTH,
            "decoder_params": EXPECTED_PARAMS,
            "ckpt": str(ckpt_path),
            "ckpt_sha256": SEALED_CKPT_SHA,
            "weight_sha256": weight_sha,
            "label_budget": 10,
            "exact_e": True,
            "ort": True,
            "ort_intra_op": 2,
            "dataloader_workers": 0,
            "torch_threads": 2,
            "evalai_opened": False,
            "train_sessions": list(mp.M1_SESSIONS),
            "train_windows": 213336,
            "selection_surface": "dev-on-official-selected visible HO-calib trio; hidden/test unread",
            "selection_mean": selected["equal_session_mean"],
            "worst_session_r2": selected["worst_session_r2"],
            "pick_rule": pick["selection_rule"],
        },
    }
    with payload_path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    payload_sha = sha256_file(payload_path)

    sys.path.insert(0, str(DEST))
    from m1_exacte_ort import OrtTrfFalconDecoder, export_ort_graphs
    from trf_falcon_decoder import TrfFalconDecoder

    stem = "L_20120926_held_in_eval"
    packed = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed.reset(dataset_tags=[stem])
    packed_pred = _run_stream(packed, window)
    delta_train = float(np.max(np.abs(packed_pred - train_pred)))
    plan.require(np.isfinite(packed_pred).all(), "non-finite packed pred")
    plan.require(delta_train <= GATE, f"packed vs train max|Δ|={delta_train}")

    graph_dir = DEST / "artifacts" / "ort_graphs"
    packed_for_export = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    export_rows = export_ort_graphs(packed_for_export.decoder, graph_dir, batches=(1, 2, 3, 4), n_units=64)
    plan.require(all((graph_dir / f"ort_adv_b{b}.onnx").is_file() for b in range(1, 5)), "missing ORT graphs")

    ort = OrtTrfFalconDecoder(
        task_config=config, model_path=str(payload_path), batch_size=1, graph_dir=graph_dir, intra_op=2, inter_op=1
    )
    ort.reset(dataset_tags=[stem])
    ort_pred = _run_stream(ort, window)
    delta_ort = float(np.max(np.abs(ort_pred - packed_pred)))
    plan.require(delta_ort <= GATE, f"ORT vs packed smoke max|Δ|={delta_ort}")

    packed_long = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
    packed_long.reset(dataset_tags=[stem])
    packed_long_pred = _run_stream(packed_long, long_stream)
    ort_long = OrtTrfFalconDecoder(
        task_config=config, model_path=str(payload_path), batch_size=1, graph_dir=graph_dir, intra_op=2, inter_op=1
    )
    ort_long.reset(dataset_tags=[stem])
    ort_long_pred = _run_stream(ort_long, long_stream)
    delta_long = float(np.max(np.abs(ort_long_pred - packed_long_pred)))
    plan.require(delta_long <= GATE, f"ORT vs packed advance max|Δ|={delta_long}")

    b4_tags = [stem, "L_20120927_held_in_eval", "L_20120928_held_in_eval", "L_20120924_held_in_eval"]
    streams4 = []
    for sid in ("ses-20120926", "ses-20120927", "ses-20120928", "ses-20120926"):
        neural_s = np.asarray(dm.train_dataset.base.neural_data[sid], dtype=np.float32)
        streams4.append(np.ascontiguousarray(neural_s[pad : pad + 200], dtype=np.float32))
    packed_b4 = TrfFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=4)
    packed_b4.reset(dataset_tags=b4_tags)
    ort_b4 = OrtTrfFalconDecoder(
        task_config=config, model_path=str(payload_path), batch_size=4, graph_dir=graph_dir, intra_op=2, inter_op=1
    )
    ort_b4.reset(dataset_tags=b4_tags)
    delta_b4 = 0.0
    for t in range(200):
        batch = np.stack([s[t] for s in streams4], axis=0)
        p = packed_b4.predict(batch)
        o = ort_b4.predict(batch)
        delta_b4 = max(delta_b4, float(np.max(np.abs(o - p))))
    plan.require(delta_b4 <= GATE, f"ORT vs packed B4 max|Δ|={delta_b4}")

    smoke_window = DEST / "artifacts" / "smoke_window.npz"
    np.savez(smoke_window, tag_stem=np.asarray(stem), window=window, expected=np.asarray(packed_pred))

    (DEST / "artifacts" / "payload.receipt.json").write_text(
        json.dumps(
            {
                "schema": "m1_projadd_depth2_ort_payload_v1",
                "payload_path": str(payload_path),
                "payload_sha256": payload_sha,
                "ckpt_sha256": SEALED_CKPT_SHA,
                "weight_sha256": weight_sha,
                "selection": selected,
                "parity_packed_vs_train_max_abs": delta_train,
                "parity_ort_vs_packed_smoke_max_abs": delta_ort,
                "parity_ort_vs_packed_advance_max_abs": delta_long,
                "parity_ort_vs_packed_b4_max_abs": delta_b4,
                "ort_graphs": export_rows,
                "evalai_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    method_name = "M1 proj_add P16 CausalPE depth-2 EMA e21 ORT exact-E w0"
    method_label = (
        "M1 proj_add P16 CausalPE depth-2 (seed42 EMA e21 L=100) ORT exact-E intra2; "
        "not SPINT; not 582019 depth-4"
    )
    method_description = (
        "M1 proj_add P16 + CausalPE depth-2 (2 temporal layers, 2,479,408 params), "
        "seed 42 EMA e21 after 24-epoch all-4 held-in train (213,336 windows). "
        "ext6 pick on visible official HO-calib trio (eq 0.6131; worst 20121024 0.556). "
        "M10 + sealed rSyn3 banks. Same M1 ORT exact-E recipe (intra_op=2, graphs B=1..4). "
        "dev-on-official-selected; hidden/test unread. Exact-E, no TTA."
    )
    budget_disclosure = (
        "10 public calibration trials for B3 E0 and rSyn3; static banks; no TTA. "
        "Checkpoint is sealed depth-2 HO-calib ext6 pick e21 EMA. Local minival/LOSO are diagnostic only."
    )
    image_tag = f"spint-m1:projadd-p16-d2-ort-e21-w0-{payload_sha[:8]}"
    (DEST / ".dockerignore").write_text(
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!trf_falcon_decoder.py\n"
        "!m1_exacte_fast.py\n!m1_exacte_ort.py\n!artifacts/\n"
        f"!artifacts/{payload_name}\n!artifacts/ort_graphs/\n!artifacts/ort_graphs/**\n"
        f"!artifacts/vendor/\n!artifacts/vendor/{wheel_dest.name}\n",
        encoding="utf-8",
    )
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.task="m1"\n'
        'LABEL ai.eval.label_budget="10"\n'
        'LABEL ai.eval.exact_e="true"\n'
        'LABEL ai.eval.proj_dim="16"\n'
        'LABEL ai.eval.dataloader_workers="0"\n'
        'LABEL ai.eval.torch_threads="2"\n'
        'LABEL ai.eval.ort="true"\n'
        'LABEL ai.eval.temporal_layers="2"\n'
        f"COPY artifacts/vendor/{wheel_dest.name} /tmp/{wheel_dest.name}\n"
        f"RUN python -m pip install --no-index --no-deps /tmp/{wheel_dest.name} "
        f"&& rm -f /tmp/{wheel_dest.name}\n"
        f"COPY artifacts/{payload_name} /data/decoder.pkl\n"
        "COPY artifacts/ort_graphs /graphs\n"
        "COPY trf_falcon_decoder.py /trf_falcon_decoder.py\n"
        "COPY m1_exacte_fast.py /m1_exacte_fast.py\n"
        "COPY m1_exacte_ort.py /m1_exacte_ort.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=m1 PHASE=test BATCH_SIZE=4 "
        "PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 "
        "ORT_GRAPH_DIR=/graphs ORT_INTRA_OP=2 "
        "RT_SUBMITTED_DECODER=/trf_falcon_decoder.py\n"
        'CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC '
        '--model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]\n',
        encoding="utf-8",
    )

    subprocess.run(
        ["docker", "build", "-t", image_tag, "--build-arg", f"PAYLOAD_SHA256={payload_sha}", str(DEST)],
        check=True,
    )
    inspect = subprocess.check_output(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Id}} {{.Size}}"],
        text=True,
    ).strip()
    image_id, size = inspect.split()
    smoke_out = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "RT_SUBMITTED_DECODER=/trf_falcon_decoder.py",
            "-v",
            f"{payload_path.resolve()}:/data/decoder.pkl:ro",
            "-v",
            f"{smoke_window.resolve()}:/data/smoke_window.npz:ro",
            image_tag,
            "python",
            "/m1_exacte_ort.py",
            "--smoke-payload",
            "/data/decoder.pkl",
            "--smoke-window",
            "/data/smoke_window.npz",
            "--graph-dir",
            "/graphs",
        ],
        text=True,
    )
    last = [line for line in smoke_out.splitlines() if line.strip()][-1]
    smoke_report = json.loads(last)
    plan.require(smoke_report.get("status") == "CONTAINER_SMOKE_PASS", f"smoke failed: {smoke_out}")

    candidate = {
        "arm": "m1_projadd_p16_d2_s42_ema_e21_ort",
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_name": method_name,
        "method_label": method_label,
        "method_description": method_description,
        "budget_disclosure": budget_disclosure,
        "payload_sha256": payload_sha,
        "parity_packed_vs_train_max_abs": delta_train,
        "parity_ort_vs_packed_smoke_max_abs": delta_ort,
        "parity_ort_vs_packed_advance_max_abs": delta_long,
        "parity_ort_vs_packed_b4_max_abs": delta_b4,
        "selection_mean": selected["equal_session_mean"],
        "proj_dim": PROJ_DIM,
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "register": False,
        "evalai_opened": False,
    }
    (DEST / "artifacts" / "evalai_candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (DEST / "artifacts" / "evalai_push_state.json").write_text(
        json.dumps(
            {
                "arm": candidate["arm"],
                "image_tag": image_tag,
                "payload_sha256": payload_sha,
                "schema_version": "m1_projadd_evalai_push_state",
                "register": False,
                "created": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PACKED_NOT_REGISTERED", "candidate": candidate}, indent=2), flush=True)
    return candidate


if __name__ == "__main__":
    main()
