#!/usr/bin/env python3
"""Pack H1 C2-CAL-1 B2 e18 into an ORT exact-E Docker image. Does not register."""
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
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import adapters, h1_config, plan
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import HELDIN_SESSIONS
from tfpd_exploration.src.m2_dual_track_v1.champion import tensor_state_sha256
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime.banks import bank_receipt, collect_official_banks
from tfpd_exploration.src.two_mainlines_long_v1.h1_runtime.load_weights import export_state_numpy
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.payload import sha256_file

DEST = Path(
    os.environ.get(
        "H1_B2_ORT_DEST",
        str(WORKSPACE_ROOT / "tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1"),
    )
)
B2_ROOT = Path(
    os.environ.get(
        "H1_B2_TRAIN_DEST",
        str(PACKAGE_ROOT / "results/h1_c2_cal1_b2_l200_p16/20260907T034938Z"),
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
PAYLOAD_SCHEMA = "h1_projadd_exacte_falcon_payload_v1"
SCALE = h1_config.TARGET_MULTIPLIER
WINDOW = 200
EPOCH = 18
GATE = 1.0e-5


def _apply_ema(model: BTransformerUnifiedDecoderIdentity, ckpt: dict) -> None:
    model.load_state_dict(ckpt["raw_state_dict"])
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    plan.require(set(named) == set(shadow), "EMA/RAW key mismatch")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def _make_geometry(window: int) -> dict:
    geometry = dict(plan.TASK_GEOMETRY["h1"])
    geometry["e0_dim"] = h1_config.matrix_e0_dim("proj_add")
    geometry["task"] = f"h1-c2-cal1-b2-L{window}-proj_add"
    return geometry


def _run_stream(decoder, stream: np.ndarray) -> np.ndarray:
    pred = None
    for row in stream:
        pred = decoder.predict(row.reshape(1, -1))
    plan.require(pred is not None, "empty stream")
    return np.asarray(pred, dtype=np.float32)


def main() -> dict:
    plan.require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    selection = json.loads((B2_ROOT / "ho_m3_selection.json").read_text(encoding="utf-8"))
    plan.require(selection.get("status") == "SEALED_HO_M3_SELECTION", "B2 HO-M3 not sealed")
    selected = selection["selected"]
    plan.require(int(selected["epoch"]) == EPOCH, f"expected e{EPOCH}, got {selected['epoch']}")
    ckpt_path = Path(selected["path"])
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")
    plan.require(ckpt_path.resolve().is_relative_to(B2_ROOT.resolve()), "ckpt escaped B2 dest")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts").mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts" / "ort_graphs").mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts" / "vendor").mkdir(parents=True, exist_ok=True)
    payload_name = f"h1_c2_cal1_b2_s42_ema_e{EPOCH}_L{WINDOW}.pkl"
    payload_path = DEST / "artifacts" / payload_name
    receipt_path = DEST / "artifacts" / "payload.receipt.json"
    wheel_dest = DEST / "artifacts" / "vendor" / WHEEL_SRC.name
    if not wheel_dest.is_file():
        shutil.copy2(WHEEL_SRC, wheel_dest)
    resume = os.environ.get("H1_B2_ORT_RESUME") == "1" and payload_path.is_file() and receipt_path.is_file()
    if resume:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload_sha = receipt["payload_sha256"]
        plan.require(sha256_file(payload_path) == payload_sha, "resume payload SHA drift")
        plan.require(all((DEST / "artifacts" / "ort_graphs" / f"ort_adv_b{b}.onnx").is_file() for b in range(1, 9)), "resume missing graphs")
        smoke_window = DEST / "artifacts" / "smoke_window.npz"
        plan.require(smoke_window.is_file(), "resume missing smoke window")
        delta_train = float(receipt["parity_packed_vs_train_max_abs"])
        delta_ort = float(receipt["parity_ort_vs_packed_smoke_max_abs"])
        delta_long = float(receipt["parity_ort_vs_packed_advance_max_abs"])
        plan.require(max(delta_train, delta_ort, delta_long) <= GATE, "resume host gate failed")
    else:
        plan.require(not payload_path.exists(), f"refusing to overwrite {payload_path}")

    os.environ.setdefault("RT_PACKED_DECODER", str(DEST / "h1_trf_falcon_decoder.py"))
    if resume:
        pass
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = BTransformerUnifiedDecoderIdentity(
            _make_geometry(WINDOW), seed=42, override_prefix=0, override_window=WINDOW, identity_mode="proj_add"
        )
        _apply_ema(model, ckpt)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        weight_sha = tensor_state_sha256({k: v.detach() for k, v in model.state_dict().items()})
        banks = collect_official_banks()
        config = FalconConfig(task=FalconTask.h1)
        payload = {
            "schema_version": PAYLOAD_SCHEMA,
            "task": config.task,
            "kind": "proj_add",
            "state_dict": export_state_numpy(model),
            "bank_by_dataset_tag": {
                tag: {
                    "E0": np.ascontiguousarray(row["E0"], dtype=np.float32),
                    "T": np.ascontiguousarray(row["T"], dtype=np.float32),
                    "unit_mask": np.ascontiguousarray(row["unit_mask"], dtype=np.bool_),
                }
                for tag, row in banks.items()
            },
            "window_size": WINDOW,
            "behavior_scaling_factor": SCALE,
            "smooth_observations": False,
            "metadata": {
                "kind": "proj_add",
                "cell": f"H1-C2-CAL1-B2-L{WINDOW}-PROJ-ADD-ORT",
                "seed": 42,
                "epoch": EPOCH,
                "view": "EMA",
                "ckpt": str(ckpt_path),
                "weight_sha256": weight_sha,
                "selection_surface": "C2-CAL-1 B2 HO-M3 pick after 32 epochs (e18)",
                "selection_mean": selected["val_ho_m3_grouped/r2_mean"],
                "worst_session_r2": selected["worst_session_r2"],
                "online_kv_cache": False,
                "old_spint_decoder": False,
                "exact_e": True,
                "ort": True,
                "ort_intra_op": 2,
                "dataloader_workers": 0,
                "official_tags": list(C.OFFICIAL_TAGS),
                "evalai_opened": False,
            },
        }
        with payload_path.open("xb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        payload_sha = sha256_file(payload_path)

        session = HELDIN_SESSIONS[0]
        stem = f"sub-HumanPitt-held-in-minival_{session}"
        tag = config.hash_dataset(stem)
        plan.require(tag in banks, f"hashed tag {tag} missing from official banks")
        cache = adapters._h1_source_cache()
        neural = np.ascontiguousarray(cache["minival"][session]["neural"], dtype=np.float32)
        stream = neural[:WINDOW]
        long_stream = neural[: WINDOW + 200]
        plan.require(stream.shape == (WINDOW, 176), f"smoke stream shape {stream.shape}")
        official = banks[tag]
        task_bank = TaskBank(
            session_id=session,
            E0=official["E0"],
            carrier=official["T"],
            unit_mask=official["unit_mask"],
            X_store=stream[None],
            target_store=np.zeros((1, 7), dtype=np.float32),
            window_ids=np.array([0], dtype=np.int64),
            calibration_meta={
                "shape": list(official["E0"].shape),
                "trial_count": 3,
                "estimator": "official_c2_m3_for_parity",
                "array_sha256": array_sha256(official["E0"]),
                "budget": 3,
            },
        )
        x = torch.from_numpy(stream[None])
        with torch.inference_mode():
            train_pred = (model(x, task_bank) / SCALE).detach().cpu().numpy()

        sys.path.insert(0, str(DEST))
        from h1_trf_falcon_decoder import H1ProjAddFalconDecoder
        from h1_exacte_ort import OrtH1ProjAddFalconDecoder, export_ort_graphs

        packed = H1ProjAddFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
        packed.reset(dataset_tags=[stem])
        packed_pred = _run_stream(packed, stream)
        delta_train = float(np.max(np.abs(packed_pred - train_pred)))
        plan.require(np.isfinite(packed_pred).all(), "non-finite packed pred")
        plan.require(delta_train <= GATE, f"packed vs train max|Δ|={delta_train}")

        graph_dir = DEST / "artifacts" / "ort_graphs"
        packed_for_export = H1ProjAddFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
        export_rows = export_ort_graphs(
            packed_for_export.decoder, graph_dir, batches=tuple(range(1, 9)), n_units=176, window=WINDOW
        )
        plan.require(all((graph_dir / f"ort_adv_b{b}.onnx").is_file() for b in range(1, 9)), "missing ORT graphs")

        ort = OrtH1ProjAddFalconDecoder(
            task_config=config,
            model_path=str(payload_path),
            batch_size=1,
            graph_dir=graph_dir,
            intra_op=2,
            inter_op=1,
        )
        ort.reset(dataset_tags=[stem])
        ort_pred = _run_stream(ort, stream)
        delta_ort = float(np.max(np.abs(ort_pred - packed_pred)))
        plan.require(np.isfinite(ort_pred).all(), "non-finite ORT pred")
        plan.require(delta_ort <= GATE, f"ORT vs packed smoke max|Δ|={delta_ort}")

        packed_long = H1ProjAddFalconDecoder(task_config=config, model_path=str(payload_path), batch_size=1)
        packed_long.reset(dataset_tags=[stem])
        packed_long_pred = _run_stream(packed_long, long_stream)
        ort_long = OrtH1ProjAddFalconDecoder(
            task_config=config,
            model_path=str(payload_path),
            batch_size=1,
            graph_dir=graph_dir,
            intra_op=2,
            inter_op=1,
        )
        ort_long.reset(dataset_tags=[stem])
        ort_long_pred = _run_stream(ort_long, long_stream)
        delta_long = float(np.max(np.abs(ort_long_pred - packed_long_pred)))
        plan.require(delta_long <= GATE, f"ORT vs packed advance max|Δ|={delta_long}")

        smoke_window = DEST / "artifacts" / "smoke_window.npz"
        np.savez(smoke_window, tag_stem=np.asarray(stem), window=stream, expected=np.asarray(packed_pred))

        (DEST / "artifacts" / "payload.receipt.json").write_text(
            json.dumps(
                {
                    "schema": "h1_c2_cal1_b2_ort_payload_v1",
                    "payload_path": str(payload_path),
                    "payload_sha256": payload_sha,
                    "bytes": payload_path.stat().st_size,
                    "weight_sha256": weight_sha,
                    "selection": selected,
                    "parity_packed_vs_train_max_abs": delta_train,
                    "parity_ort_vs_packed_smoke_max_abs": delta_ort,
                    "parity_ort_vs_packed_advance_max_abs": delta_long,
                    "ort_graphs": export_rows,
                    "banks": bank_receipt(banks),
                    "evalai_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (DEST / ".dockerignore").write_text(
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!h1_trf_falcon_decoder.py\n"
        "!h1_exacte_fast.py\n!h1_exacte_ort.py\n!artifacts/\n"
        f"!artifacts/{payload_name}\n!artifacts/ort_graphs/\n!artifacts/ort_graphs/**\n"
        f"!artifacts/vendor/\n!artifacts/vendor/{wheel_dest.name}\n",
        encoding="utf-8",
    )
    method_name = f"H1 C2-CAL-1 B2 P16 L200 e18 ORT exact-E w0"
    method_label = (
        "H1 C2-CAL-1 B2 proj_add CausalPE4 (seed42 EMA e18 L=200 P16) "
        "ORT exact-E intra2; not SPINT; not C2-identical; not 582025 e24"
    )
    method_description = (
        "H1 C2-CAL-1 B2: scheduled M7 starts + M in {7,5,4}->V1 M4 carrier, "
        "M=3->fit_deployment_carrier. Seed 42, AdamW 1e-4, EMA 0.9995, 13 held-in, "
        "HO-M3 pick e18 (local 0.378). Official HO banks remain C2 earliest-M3. "
        "Same M1 ORT exact-E recipe (intra_op=2, graphs B=1..8). Exact-E, no TTA."
    )
    budget_disclosure = (
        "H1 CAL-1 deploy M3 (budget=3); C2 e15 frozen materializer; static official banks; "
        "no TTA. Checkpoint is sealed B2 HO-M3 e18 EMA after 32-epoch 13-session train. "
        "Local HO-M3 is development selection, not official HO."
    )
    image_tag = f"spint-t4-h1:c2-cal1-b2-ort-e18-L200-w0-{payload_sha[:8]}"
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={C.BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.task="h1"\n'
        'LABEL ai.eval.old_spint_decoder="false"\n'
        'LABEL ai.eval.exact_e="true"\n'
        'LABEL ai.eval.dataloader_workers="0"\n'
        'LABEL ai.eval.ort="true"\n'
        'LABEL ai.eval.candidate="h1_c2_cal1_b2_s42_ema_e18_L200_ort"\n'
        f"COPY artifacts/vendor/{wheel_dest.name} /tmp/{wheel_dest.name}\n"
        f"RUN python -m pip install --no-index --no-deps /tmp/{wheel_dest.name} "
        f"&& rm -f /tmp/{wheel_dest.name}\n"
        f"COPY artifacts/{payload_name} /data/decoder.pkl\n"
        "COPY artifacts/ort_graphs /graphs\n"
        "COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py\n"
        "COPY h1_exacte_fast.py /h1_exacte_fast.py\n"
        "COPY h1_exacte_ort.py /h1_exacte_ort.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8 "
        "PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 "
        "ORT_GRAPH_DIR=/graphs ORT_INTRA_OP=2 "
        "RT_PACKED_DECODER=/h1_trf_falcon_decoder.py\n"
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
            "RT_PACKED_DECODER=/h1_trf_falcon_decoder.py",
            "-v",
            f"{payload_path.resolve()}:/data/decoder.pkl:ro",
            "-v",
            f"{smoke_window.resolve()}:/data/smoke_window.npz:ro",
            image_tag,
            "python",
            "/h1_exacte_ort.py",
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
        "arm": "h1_c2_cal1_b2_s42_ema_e18_L200_ort",
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
        "selection_mean": selected["val_ho_m3_grouped/r2_mean"],
        "window": WINDOW,
        "register": False,
        "evalai_opened": False,
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "hold_reason": "packed; submit only after this script prints PACKED_NOT_REGISTERED and user-authorized submit runs",
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
                "schema_version": "h1_trf_evalai_push_state",
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
