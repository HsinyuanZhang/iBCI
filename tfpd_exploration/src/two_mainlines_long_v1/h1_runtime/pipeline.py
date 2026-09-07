"""S3/S4 host export, container proof, and quota-aware EvalAI register."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tfpd_exploration.src.two_mainlines_long_v1.registrar import quota, write_slot

from . import constants as C
from .parity import estimate_cpu_hours, host_stream_parity
from .payload import build_payload, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _merge_slot(slot: str, **fields: Any) -> dict[str, Any]:
    path = C.SLOT_ROOT / f"slot_{slot}.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(fields)
    current["slot"] = slot
    current["owner"] = "H1"
    current["updated"] = datetime.now(timezone.utc).isoformat()
    write_slot(slot, current)
    return current


def _sub(kind: str) -> Path:
    return C.FLAT_SUB if kind == "flat" else C.ROUTE_SUB


def _payload_name(kind: str) -> str:
    return "h1_temporal_flat_ema_e5.pkl" if kind == "flat" else "h1_temporal_route_ema_e5.pkl"


def _copy_submission_sources(kind: str) -> None:
    dest = _sub(kind)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "artifacts").mkdir(parents=True, exist_ok=True)
    runtime = Path(__file__).resolve().parent / "container_decoder.py"
    shutil.copy2(runtime, dest / "h1_trf_falcon_decoder.py")
    decode = dest / "decode.py"
    decode.write_text(
        '"""EvalAI entry for the H1 temporal Transformer runtime."""\n'
        "import argparse\n"
        "from falcon_challenge.config import FalconConfig, FalconTask\n"
        "from falcon_challenge.evaluator import FalconEvaluator\n"
        "from h1_trf_falcon_decoder import H1TemporalFalconDecoder\n"
        "\n"
        "def main() -> None:\n"
        "    parser = argparse.ArgumentParser()\n"
        '    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)\n'
        '    parser.add_argument("--model-path", default="/data/decoder.pkl")\n'
        '    parser.add_argument("--split", choices=("h1",), default="h1")\n'
        '    parser.add_argument("--phase", choices=("minival", "test"), default="test")\n'
        '    parser.add_argument("--batch-size", type=int, default=8)\n'
        "    args = parser.parse_args()\n"
        "    config = FalconConfig(task=FalconTask.h1)\n"
        "    decoder = H1TemporalFalconDecoder(\n"
        "        task_config=config, model_path=args.model_path, batch_size=args.batch_size\n"
        "    )\n"
        "    evaluator = FalconEvaluator(eval_remote=args.evaluation == \"remote\", split=\"h1\")\n"
        "    evaluator.evaluate(decoder, phase=args.phase)\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n",
        encoding="utf-8",
    )
    dest.joinpath(".dockerignore").write_text(
        "**\n"
        "!.dockerignore\n"
        "!Dockerfile\n"
        "!decode.py\n"
        "!h1_trf_falcon_decoder.py\n"
        "!artifacts/\n"
        f"!artifacts/{_payload_name(kind)}\n",
        encoding="utf-8",
    )


def _write_dockerfile(kind: str, payload_sha: str, method_label: str, candidate: str) -> None:
    dest = _sub(kind)
    payload = _payload_name(kind)
    dest.joinpath("Dockerfile").write_text(
        f"ARG BASE_IMAGE={C.BASE_IMAGE}\n"
        "FROM ${BASE_IMAGE}\n"
        f"ARG PAYLOAD_SHA256={payload_sha}\n"
        f"ARG CALIBRATION_SHA256={C.C2_CKPT_SHA256}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.calibration.sha256="${CALIBRATION_SHA256}"\n'
        'LABEL ai.eval.old_c2_decoder="false"\n'
        'LABEL ai.eval.c2_identity_swap="false"\n'
        'LABEL ai.eval.epoch="5"\n'
        'LABEL ai.eval.epochs_target="12"\n'
        'LABEL ai.eval.view="EMA"\n'
        f'LABEL ai.eval.candidate="{candidate}"\n'
        'LABEL ai.eval.decoder="h1-temporal-trf"\n'
        f"COPY artifacts/{payload} /data/decoder.pkl\n"
        "COPY h1_trf_falcon_decoder.py /h1_trf_falcon_decoder.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=h1 PHASE=test BATCH_SIZE=8\n"
        'CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]\n',
        encoding="utf-8",
    )


def _docker_build(kind: str, payload_sha: str) -> dict[str, Any]:
    dest = _sub(kind)
    tag = (
        f"h1-temporal-trf:flat-ema-e5-{payload_sha[:8]}"
        if kind == "flat"
        else f"h1-temporal-trf:route-ema-e5-{payload_sha[:8]}"
    )
    cmd = ["docker", "build", "-t", tag, "--build-arg", f"PAYLOAD_SHA256={payload_sha}", str(dest)]
    subprocess.run(cmd, check=True)
    inspect = subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}} {{.Size}}"],
        text=True,
    ).strip()
    image_id, size = inspect.split()
    return {"image_tag": tag, "image_id": image_id, "image_size": int(size)}


def _container_smoke(kind: str, image_tag: str, payload_path: Path, smoke_window: Path) -> dict[str, Any]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-v",
        f"{payload_path.resolve()}:/data/decoder.pkl:ro",
        "-v",
        f"{smoke_window.resolve()}:/data/smoke_window.npz:ro",
        "-v",
        f"{_sub(kind) / 'h1_trf_falcon_decoder.py'}:/h1_trf_falcon_decoder.py:ro",
        image_tag,
        "python",
        "/h1_trf_falcon_decoder.py",
        "--smoke-payload",
        "/data/decoder.pkl",
        "--smoke-window",
        "/data/smoke_window.npz",
    ]
    out = subprocess.check_output(cmd, text=True)
    last = [line for line in out.splitlines() if line.strip()][-1]
    report = json.loads(last)
    require(report.get("status") == "CONTAINER_SMOKE_PASS", f"container smoke failed: {out}")
    require(report.get("on_done_noop") is True, "container on_done must be no-op")
    require(int(report.get("pe_max_len", 0)) >= 700, "container PE must cover 700")
    return {"status": "CONTAINER_PASS", "stdout": last, "cpu_only": True}


def _try_register(kind: str, candidate: dict[str, Any]) -> dict[str, Any]:
    limits = quota()
    if int(limits["active"]) >= 3 or int(limits["today"]) >= 6 or not limits["can_register"]:
        return {
            "status": "NOT_REGISTERED",
            "reason": (
                f"quota blocked: active={limits['active']} today={limits['today']} "
                f"can_register={limits['can_register']}; S1 581937 / S2 581938 count if still queued/running"
            ),
            "quota": limits,
        }
    manifest = _sub(kind) / "artifacts" / "evalai_candidate.json"
    manifest.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cmd = [
        "/usr/bin/python3",
        str(Path(__file__).resolve().parent / "submit_evalai.py"),
        "--manifest",
        str(manifest),
        "--execute",
        "--confirm-image-id",
        candidate["image_id"],
        "--confirm-payload-sha256",
        candidate["payload_sha256"],
    ]
    out = subprocess.check_output(cmd, text=True)
    lines = [line for line in out.splitlines() if line.strip()]
    parsed = json.loads(lines[-1])
    parsed["status"] = "REGISTERED"
    parsed["quota_after"] = quota()
    return parsed


def run_slot(kind: str, *, stage: str = "all", skip_register: bool = False) -> dict[str, Any]:
    require(kind in {"flat", "route"}, f"unknown kind {kind}")
    slot = "S3" if kind == "flat" else "S4"
    name = "H1-TEMPORAL-TRF-FLAT" if kind == "flat" else "H1-TEMPORAL-TRF-ROUTE"
    if stage != "register":
        _merge_slot(
            slot,
            name=name,
            epoch=C.EPOCH,
            disclosure="known-source development; snapshot epoch 5 of intended 12; not clean LODO",
            status="PACKAGING",
            view=C.VIEW,
        )
    report: dict[str, Any] = {"kind": kind, "slot": slot, "epoch": C.EPOCH, "view": C.VIEW}
    dest = _sub(kind)
    _copy_submission_sources(kind)
    payload_path = dest / "artifacts" / _payload_name(kind)

    if stage in {"export", "all"}:
        if not payload_path.exists():
            report["payload"] = build_payload(kind, payload_path)
        else:
            report["payload"] = json.loads((payload_path.parent / "payload.receipt.json").read_text(encoding="utf-8"))
        _merge_slot(slot, payload_sha256=report["payload"]["payload_sha256"], status="EXPORTED")

    if stage in {"parity", "all"}:
        report["parity"] = host_stream_parity(kind, payload_path)
        hours = estimate_cpu_hours(float(report["parity"]["ms_per_window"]))
        report["cpu_hours_estimate"] = hours
        _merge_slot(
            slot,
            status="PARITY_PASS",
            parity_max_abs=report["parity"]["max_abs_host_vs_stream"],
            ms_per_window=report["parity"]["ms_per_window"],
            cpu_hours_estimate=hours["cpu_hours_heldin_plus_heldout_proxy"],
            six_hour_risk=bool(hours["six_hour_risk"]),
        )

    if stage in {"image", "all"}:
        method_label = C.FLAT_METHOD_LABEL if kind == "flat" else C.ROUTE_METHOD_LABEL
        candidate_name = "h1_temporal_flat_ema_e5" if kind == "flat" else "h1_temporal_route_ema_e5"
        payload_sha = report.get("payload", {}).get("payload_sha256") or sha256_file(payload_path)
        _write_dockerfile(kind, payload_sha, method_label, candidate_name)
        image = _docker_build(kind, payload_sha)
        report["image"] = image
        smoke_window = Path(
            (report.get("parity") or {}).get("smoke_window")
            or (payload_path.parent / f"{kind}_smoke_window.npz")
        )
        require(smoke_window.is_file(), f"missing smoke window {smoke_window}")
        smoke = _container_smoke(
            kind,
            image["image_tag"],
            payload_path,
            smoke_window,
        )
        report["container"] = smoke
        candidate = {
            "arm": candidate_name,
            "image_tag": image["image_tag"],
            "image_id": image["image_id"],
            "payload_sha256": payload_sha,
            "method_label": method_label,
            "method_name": C.FLAT_METHOD_NAME if kind == "flat" else C.ROUTE_METHOD_NAME,
            "method_description": C.FLAT_METHOD_DESCRIPTION if kind == "flat" else C.ROUTE_METHOD_DESCRIPTION,
            "budget_disclosure": (
                "first-3 public calibration trials; C2 fused identity + H-C banks; "
                "no TTA; snapshot epoch 5 of intended 12; known-source development"
            ),
            "state_path": str(dest / "artifacts" / "evalai_push_state.json"),
        }
        report["candidate"] = candidate
        (dest / "artifacts" / "evalai_candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _merge_slot(
            slot,
            status="CONTAINER_PASS",
            image_id=image["image_id"],
            image_tag=image["image_tag"],
            payload_sha256=payload_sha,
            submission_id=None,
        )

    if stage in {"register", "all"} and not skip_register:
        if "candidate" not in report:
            manifest = dest / "artifacts" / "evalai_candidate.json"
            require(manifest.is_file(), f"missing candidate manifest {manifest}")
            report["candidate"] = json.loads(manifest.read_text(encoding="utf-8"))
        state_path = Path(report["candidate"]["state_path"])
        if state_path.is_file():
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            if prior.get("registered") and prior.get("submission_id"):
                report["register"] = {
                    "status": "REGISTERED",
                    "submission_id": prior["submission_id"],
                    "image_id": prior.get("image_id"),
                    "payload_sha256": prior.get("payload_sha256"),
                    "recovered_from_state": True,
                }
                _merge_slot(
                    slot,
                    status="REGISTERED",
                    submission_id=prior["submission_id"],
                    image_id=prior.get("image_id") or report["candidate"]["image_id"],
                    payload_sha256=prior.get("payload_sha256") or report["candidate"]["payload_sha256"],
                )
                _write_pack_md()
                return report
        registered = _try_register(kind, report["candidate"])
        report["register"] = registered
        if registered.get("status") == "REGISTERED":
            _merge_slot(
                slot,
                status="REGISTERED",
                submission_id=registered.get("submission_id"),
                image_id=registered.get("image_id"),
                payload_sha256=registered.get("payload_sha256"),
            )
        else:
            _merge_slot(
                slot,
                status="CONTAINER_PASS",
                submission_id=None,
                not_registered_reason=registered.get("reason"),
            )
    _write_pack_md()
    return report


def _write_pack_md() -> None:
    s3 = json.loads((C.SLOT_ROOT / "slot_S3.json").read_text(encoding="utf-8")) if (C.SLOT_ROOT / "slot_S3.json").exists() else {}
    s4 = json.loads((C.SLOT_ROOT / "slot_S4.json").read_text(encoding="utf-8")) if (C.SLOT_ROOT / "slot_S4.json").exists() else {}

    def _id(row: dict[str, Any]) -> str:
        if row.get("status") == "REGISTERED" and row.get("submission_id"):
            return str(row["submission_id"])
        reason = row.get("not_registered_reason") or row.get("status") or "unknown"
        return f"NOT_REGISTERED + {reason}"

    lines = [
        "# H1 pack — temporal Transformer FLAT / ROUTE (E_H=5, EMA)",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "New Falcon runtime (Conv1→16 k5, concat local+E0+H-C, 8-slot set attn d=256, "
        "4-layer causal Transformer, 256→128→7 last-bin, output /20). "
        "Not a C2 identity swap. Snapshot epoch 5 of intended 12. Known-source development, not clean LODO.",
        "",
        "## S3 FLAT",
        "",
        f"- status: `{s3.get('status')}`",
        f"- submission: {_id(s3)}",
        f"- epoch/view: {s3.get('epoch')} / {s3.get('view', 'EMA')}",
        f"- payload: `{s3.get('payload_sha256')}`",
        f"- image: `{s3.get('image_id')}`",
        f"- ms/window: {s3.get('ms_per_window')}",
        f"- CPU hours estimate: {s3.get('cpu_hours_estimate')} (6h risk={s3.get('six_hour_risk')})",
        "",
        "## S4 ROUTE",
        "",
        f"- status: `{s4.get('status')}`",
        f"- submission: {_id(s4)}",
        f"- epoch/view: {s4.get('epoch')} / {s4.get('view', 'EMA')}",
        f"- payload: `{s4.get('payload_sha256')}`",
        f"- image: `{s4.get('image_id')}`",
        f"- ms/window: {s4.get('ms_per_window')}",
        f"- CPU hours estimate: {s4.get('cpu_hours_estimate')} (6h risk={s4.get('six_hour_risk')})",
        "",
        "C2 SHA ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215 is calibration materialization only.",
        "on_done is a no-op. No cross-window KV cache. PE covers 700. All 27 official H1 tags.",
        "Training PIDs were not killed. GPU0/GPU1 were not used for packaging.",
        "",
    ]
    (C.SLOT_ROOT / "h1_pack.md").write_text("\n".join(lines), encoding="utf-8")
