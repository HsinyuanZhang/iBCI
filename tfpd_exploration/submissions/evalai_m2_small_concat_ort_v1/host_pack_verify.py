#!/usr/bin/env python3
"""Host pack+verify for M2 SMALL concat ORT exact-E. Does not EvalAI-submit."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from falcon_challenge.config import FalconConfig, FalconTask

DEST = Path(__file__).resolve().parent
SRC = Path(
    "/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1"
)
WHEEL_SRC = Path(
    "/home/xinyuan/Work_host/SPINT/btransform_unified_v1/results/m1_projadd_runtime_v1/"
    "20260907T020150Z/scripts/vendor/"
    "onnxruntime-1.19.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
CACHE = Path(
    "/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/m2_dual_track_v1/"
    "20260905_101500/cache/source_train"
)
SEALED_SHA = "4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4"
PAYLOAD_NAME = "m2_small_trf_s42_ema_e08_ext6.pkl"
GATE = 1.0e-5
WINDOW = 50
ADVANCE_EXTRA = 120
SMOKE_BINS = 50
BASE_IMAGE = "spint-m2:e8-epoch027-76f0fb2"
METHOD_LABEL = (
    "M2 SMALL concat CausalPE4 (S1-SMALL-COS seed42 EMA e8) ORT exact-E intra2; "
    "concat SMALL e8 ORT, not proj_add, not 581971"
)
METHOD_NAME = "M2 SMALL concat S1-SMALL-COS EMA e8 s42 ORT exact-E w0"
METHOD_DESCRIPTION = (
    "B-transformer unified runtime wrap of sealed M2 SMALL concat cell EvalAI 581973 "
    "(official HO 0.390305). kind=small, tokens=concat(local16, E0 50, T4), W=50, N=96, "
    "8-slot, 4-layer CausalPE, out=2, scale=/5, official batch 7, 13 tags. ORT exact-E "
    "intra_op=2 inter_op=1 graphs B=1..7. Not SPINT. Not proj_add P32 (582009). Not 581971 e19."
)
BUDGET = (
    "33 public calibration trials for native E0 and MOVE-T4; static banks; no TTA. "
    "Checkpoint is sealed S1-SMALL-COS seed42 EMA e8 (581973). Runtime wrap only."
)
HOLD = (
    "prepared; submit only when parent/user authorizes after H1 582044 / M1 582045 are healthy"
)

os.environ.setdefault("RT_PACKED_DECODER", str(DEST / "trf_falcon_decoder.py"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(DEST))
from m2_concat_exacte_fast import FastTrfFalconDecoder  # noqa: E402
from m2_concat_exacte_ort import OrtTrfFalconDecoder, export_ort_graphs  # noqa: E402
from trf_falcon_decoder import TrfFalconDecoder, load_payload  # noqa: E402

TAGS1 = ["Run1_20201019"]
TAGS7 = [
    "Run1_20201019",
    "Run2_20201019",
    "Run1_20201020",
    "Run2_20201020",
    "Run1_20201027",
    "Run2_20201027",
    "Run1_20201028",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag_to_session_dir(tag: str) -> str:
    run, ymd = tag.split("_")
    return f"ses-{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}-{run}"


def load_stream(tag: str) -> np.ndarray:
    path = CACHE / tag_to_session_dir(tag) / "X_store.npy"
    return np.ascontiguousarray(np.load(path), dtype=np.float32)


def stacked_stream(tags: list[str], n_bins: int, offset: int = 0) -> np.ndarray:
    arrs = [load_stream(t)[offset : offset + n_bins] for t in tags]
    return np.stack(arrs, axis=1)


def run_stream(decoder, data: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(data.shape[0]):
        preds.append(np.asarray(decoder.predict(data[i]), dtype=np.float32))
    return np.stack(preds)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def main() -> dict:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    src_pkl = SRC / "artifacts" / PAYLOAD_NAME
    require(src_pkl.is_file(), f"missing sealed payload {src_pkl}")
    payload_sha = sha256_file(src_pkl)
    require(payload_sha == SEALED_SHA, f"payload SHA drift {payload_sha}")
    dest_pkl = DEST / "artifacts" / PAYLOAD_NAME
    dest_pkl.parent.mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts" / "ort_graphs").mkdir(parents=True, exist_ok=True)
    (DEST / "artifacts" / "vendor").mkdir(parents=True, exist_ok=True)
    if dest_pkl.exists():
        require(sha256_file(dest_pkl) == SEALED_SHA, "dest payload SHA drift")
    else:
        os.link(src_pkl, dest_pkl)
        require(sha256_file(dest_pkl) == SEALED_SHA, "hardlink SHA drift")
    src_dec = SRC / "trf_falcon_decoder.py"
    dest_dec = DEST / "trf_falcon_decoder.py"
    if not dest_dec.is_file():
        shutil.copy2(src_dec, dest_dec)
    wheel_dest = DEST / "artifacts" / "vendor" / WHEEL_SRC.name
    if not wheel_dest.is_file():
        shutil.copy2(WHEEL_SRC, wheel_dest)
    require(wheel_dest.name == WHEEL_SRC.name, "wheel filename must stay original")

    payload = load_payload(dest_pkl)
    require(payload.get("kind") == "small", f"kind={payload.get('kind')}")
    require(int(payload.get("window_size")) == WINDOW, "window")
    require(len(payload["bank_by_dataset_tag"]) == 13, "need 13 tags")
    masks = [np.asarray(row["unit_mask"]) for row in payload["bank_by_dataset_tag"].values()]
    require(all(m.all() for m in masks), "unit masks must be all-true for unmasked ORT")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = FalconConfig(task=FalconTask.m2)
    smoke_data = stacked_stream(TAGS1, SMOKE_BINS, offset=0)
    long_data = stacked_stream(TAGS1, SMOKE_BINS + ADVANCE_EXTRA, offset=0)
    b7_data = stacked_stream(TAGS7, SMOKE_BINS + ADVANCE_EXTRA, offset=0)

    packed = TrfFalconDecoder(task_config=config, model_path=str(dest_pkl), batch_size=1)
    packed.reset(dataset_tags=TAGS1)
    packed_smoke = run_stream(packed, smoke_data)
    fast = FastTrfFalconDecoder(task_config=config, model_path=str(dest_pkl), batch_size=1)
    fast.reset(dataset_tags=TAGS1)
    fast_smoke = run_stream(fast, smoke_data)
    d_fast_smoke = max_abs(fast_smoke, packed_smoke)
    require(np.isfinite(fast_smoke).all(), "non-finite fast smoke")
    require(d_fast_smoke <= GATE, f"fast vs packed smoke max_abs={d_fast_smoke}")

    packed_long = TrfFalconDecoder(task_config=config, model_path=str(dest_pkl), batch_size=1)
    packed_long.reset(dataset_tags=TAGS1)
    packed_long_pred = run_stream(packed_long, long_data)
    fast_long = FastTrfFalconDecoder(task_config=config, model_path=str(dest_pkl), batch_size=1)
    fast_long.reset(dataset_tags=TAGS1)
    fast_long_pred = run_stream(fast_long, long_data)
    d_fast_long = max_abs(fast_long_pred, packed_long_pred)
    require(d_fast_long <= GATE, f"fast vs packed advance max_abs={d_fast_long}")

    graph_dir = DEST / "artifacts" / "ort_graphs"
    export_model = TrfFalconDecoder(task_config=config, model_path=str(dest_pkl), batch_size=1)
    export_rows = export_ort_graphs(export_model.decoder, graph_dir, batches=tuple(range(1, 8)))
    for b in range(1, 8):
        require((graph_dir / f"ort_adv_b{b}.onnx").is_file(), f"missing adv B={b}")
        require((graph_dir / f"ort_rebuild_b{b}.onnx").is_file(), f"missing rebuild B={b}")

    ort = OrtTrfFalconDecoder(
        task_config=config, model_path=str(dest_pkl), batch_size=1, graph_dir=graph_dir, intra_op=2, inter_op=1
    )
    ort.reset(dataset_tags=TAGS1)
    ort_smoke = run_stream(ort, smoke_data)
    d_ort_smoke = max_abs(ort_smoke, packed_smoke)
    require(np.isfinite(ort_smoke).all(), "non-finite ORT smoke")
    require(d_ort_smoke <= GATE, f"ORT vs packed smoke max_abs={d_ort_smoke}")

    ort_long = OrtTrfFalconDecoder(
        task_config=config, model_path=str(dest_pkl), batch_size=1, graph_dir=graph_dir, intra_op=2, inter_op=1
    )
    ort_long.reset(dataset_tags=TAGS1)
    ort_long_pred = run_stream(ort_long, long_data)
    d_ort_long = max_abs(ort_long_pred, packed_long_pred)
    require(d_ort_long <= GATE, f"ORT vs packed advance max_abs={d_ort_long}")

    packed_b7 = TrfFalconDecoder(task_config=config, model_path=str(dest_pkl), batch_size=7)
    packed_b7.reset(dataset_tags=TAGS7)
    packed_b7_pred = run_stream(packed_b7, b7_data)
    ort_b7 = OrtTrfFalconDecoder(
        task_config=config, model_path=str(dest_pkl), batch_size=7, graph_dir=graph_dir, intra_op=2, inter_op=1
    )
    ort_b7.reset(dataset_tags=TAGS7)
    ort_b7_pred = run_stream(ort_b7, b7_data)
    d_ort_b7 = max_abs(ort_b7_pred, packed_b7_pred)
    require(np.isfinite(ort_b7_pred).all(), "non-finite ORT B7")
    require(d_ort_b7 <= GATE, f"ORT vs packed B7 max_abs={d_ort_b7}")

    smoke_window = DEST / "artifacts" / "smoke_window.npz"
    np.savez(
        smoke_window,
        tag_stem=np.asarray(TAGS1[0]),
        window=smoke_data[:, 0, :],
        expected=np.asarray(packed_smoke[-1]),
    )

    (DEST / ".dockerignore").write_text(
        "**\n!.dockerignore\n!Dockerfile\n!decode.py\n!trf_falcon_decoder.py\n"
        "!m2_concat_exacte_fast.py\n!m2_concat_exacte_ort.py\n!artifacts/\n"
        f"!artifacts/{PAYLOAD_NAME}\n!artifacts/ort_graphs/\n!artifacts/ort_graphs/**\n"
        f"!artifacts/vendor/\n!artifacts/vendor/{wheel_dest.name}\n",
        encoding="utf-8",
    )
    image_tag = f"spint-t4-m2:small-concat-ort-e8-ext6-w0-{payload_sha[:8]}"
    (DEST / "Dockerfile").write_text(
        f"ARG BASE_IMAGE={BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f'LABEL ai.eval.method="{METHOD_LABEL}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.task="m2"\n'
        'LABEL ai.eval.old_spint_decoder="false"\n'
        'LABEL ai.eval.exact_e="true"\n'
        'LABEL ai.eval.dataloader_workers="0"\n'
        'LABEL ai.eval.ort="true"\n'
        'LABEL ai.eval.identity="concat"\n'
        'LABEL ai.eval.candidate="m2_small_concat_s42_ema_e8_ort"\n'
        'LABEL ai.eval.label_budget="33"\n'
        'LABEL ai.eval.activity_budget="33"\n'
        f"COPY artifacts/vendor/{wheel_dest.name} /tmp/{wheel_dest.name}\n"
        f"RUN python -m pip install --no-index --no-deps /tmp/{wheel_dest.name} "
        f"&& rm -f /tmp/{wheel_dest.name}\n"
        f"COPY artifacts/{PAYLOAD_NAME} /data/decoder.pkl\n"
        "COPY artifacts/ort_graphs /graphs\n"
        "COPY trf_falcon_decoder.py /trf_falcon_decoder.py\n"
        "COPY m2_concat_exacte_fast.py /m2_concat_exacte_fast.py\n"
        "COPY m2_concat_exacte_ort.py /m2_concat_exacte_ort.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7 "
        "PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 "
        "ORT_GRAPH_DIR=/graphs ORT_INTRA_OP=2 "
        "RT_PACKED_DECODER=/trf_falcon_decoder.py\n"
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
    labels = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", image_tag, "--format", "{{json .Config.Labels}}"],
            text=True,
        )
    )
    require(labels.get("ai.eval.task") == "m2", "label task")
    require(labels.get("ai.eval.exact_e") == "true", "label exact_e")
    require(labels.get("ai.eval.payload.sha256") == payload_sha, "label payload sha")
    require("concat SMALL e8 ORT" in labels.get("ai.eval.method", ""), "method label")
    require("not proj_add" in labels.get("ai.eval.method", ""), "method not proj_add")
    require("not 581971" in labels.get("ai.eval.method", ""), "method not 581971")

    smoke_out = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "RT_PACKED_DECODER=/trf_falcon_decoder.py",
            "-v",
            f"{dest_pkl.resolve()}:/data/decoder.pkl:ro",
            "-v",
            f"{smoke_window.resolve()}:/data/smoke_window.npz:ro",
            image_tag,
            "python",
            "/m2_concat_exacte_ort.py",
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
    require(smoke_report.get("status") == "CONTAINER_SMOKE_PASS", f"smoke failed: {smoke_out}")

    src_receipt = json.loads((SRC / "artifacts" / "payload.receipt.json").read_text(encoding="utf-8"))
    receipt = {
        "schema": "m2_small_concat_ort_payload_v1",
        "payload_path": str(dest_pkl),
        "payload_sha256": payload_sha,
        "bytes": dest_pkl.stat().st_size,
        "source_submission_id": 581973,
        "source_official_ho": 0.3903054960458745,
        "source_official_latency": 0.7360138729810096,
        "kind": "small",
        "identity": "concat",
        "not_proj_add": True,
        "not_581971": True,
        "not_582009": True,
        "parity_fast_vs_packed_smoke_max_abs": d_fast_smoke,
        "parity_fast_vs_packed_advance_max_abs": d_fast_long,
        "parity_ort_vs_packed_smoke_max_abs": d_ort_smoke,
        "parity_ort_vs_packed_advance_max_abs": d_ort_long,
        "parity_ort_vs_packed_b7_max_abs": d_ort_b7,
        "ort_graphs": export_rows,
        "banks": src_receipt.get("banks"),
        "selected": src_receipt.get("selected"),
        "weight_sha256": src_receipt.get("weight_sha256"),
        "evalai_opened": False,
        "register": False,
        "container_smoke": smoke_report.get("status"),
        "image_tag": image_tag,
        "image_id": image_id,
    }
    (DEST / "artifacts" / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    candidate = {
        "arm": "m2_small_concat_s42_ema_e8_ort",
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_name": METHOD_NAME,
        "method_label": METHOD_LABEL,
        "method_description": METHOD_DESCRIPTION,
        "budget_disclosure": BUDGET,
        "payload_sha256": payload_sha,
        "parity_fast_vs_packed_smoke_max_abs": d_fast_smoke,
        "parity_fast_vs_packed_advance_max_abs": d_fast_long,
        "parity_ort_vs_packed_smoke_max_abs": d_ort_smoke,
        "parity_ort_vs_packed_advance_max_abs": d_ort_long,
        "parity_ort_vs_packed_b7_max_abs": d_ort_b7,
        "container_smoke": smoke_report.get("status"),
        "register": False,
        "evalai_opened": False,
        "state_path": str(DEST / "artifacts" / "evalai_push_state.json"),
        "hold_reason": HOLD,
        "source_submission_id": 581973,
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
                "schema_version": "m2_small_concat_ort_evalai_push_state",
                "register": False,
                "evalai_opened": False,
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
