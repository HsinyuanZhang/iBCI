"""S1/S2 host export, container proof, and EvalAI register."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import constants as C
from .parity import estimate_cpu_hours, host_stream_parity
from .payload import build_payload, sha256_file
from .replay import replay_ext4

REPO = C.REPO_ROOT
RUNTIME_DECODER = Path(__file__).resolve().parent / "container_decoder.py"
SMALL_SUB = REPO / "tfpd_exploration/submissions/evalai_m2_small_trf_pick_v1"
LARGE_SUB = REPO / "tfpd_exploration/submissions/evalai_m2_large_trf_pick_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _slot_path(kind: str) -> Path:
    return C.SLOT_ROOT / ("slot_S1.json" if kind == "small" else "slot_S2.json")


def write_slot(kind: str, **fields: Any) -> dict[str, Any]:
    path = _slot_path(kind)
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(fields)
    current["kind"] = kind
    current["slot"] = "S1" if kind == "small" else "S2"
    current["updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return current


def _sub(kind: str) -> Path:
    return SMALL_SUB if kind == "small" else LARGE_SUB


def _payload_name(kind: str) -> str:
    return "m2_small_trf_s1_ema_e19.pkl" if kind == "small" else "m2_large_trf_s2_raw_e20.pkl"


def _copy_submission_sources(kind: str) -> None:
    dest = _sub(kind)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNTIME_DECODER, dest / "trf_falcon_decoder.py")
    decode = dest / "decode.py"
    if not decode.exists():
        decode.write_text(
            '"""EvalAI entry for the M2 causal Transformer runtime."""\n'
            "import argparse\n"
            "from falcon_challenge.config import FalconConfig, FalconTask\n"
            "from falcon_challenge.evaluator import FalconEvaluator\n"
            "from trf_falcon_decoder import TrfFalconDecoder\n"
            "\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser()\n"
            '    parser.add_argument("--evaluation", choices=("local", "remote"), required=True)\n'
            '    parser.add_argument("--model-path", default="/data/decoder.pkl")\n'
            '    parser.add_argument("--split", choices=("m2",), default="m2")\n'
            '    parser.add_argument("--phase", choices=("minival", "test"), default="test")\n'
            '    parser.add_argument("--batch-size", type=int, default=7)\n'
            "    args = parser.parse_args()\n"
            "    config = FalconConfig(task=getattr(FalconTask, args.split))\n"
            "    decoder = TrfFalconDecoder(\n"
            "        task_config=config, model_path=args.model_path, batch_size=args.batch_size\n"
            "    )\n"
            "    evaluator = FalconEvaluator(eval_remote=args.evaluation == \"remote\", split=args.split)\n"
            "    evaluator.evaluate(decoder, phase=args.phase)\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    main()\n",
            encoding="utf-8",
        )
    dockerignore = dest / ".dockerignore"
    dockerignore.write_text(
        "**\n"
        "!.dockerignore\n"
        "!Dockerfile\n"
        "!decode.py\n"
        "!trf_falcon_decoder.py\n"
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
        f"ARG CHECKPOINT_SHA256={C.CHAMPION_CKPT_SHA256}\n"
        f'LABEL ai.eval.method="{method_label}"\n'
        'LABEL ai.eval.payload.sha256="${PAYLOAD_SHA256}"\n'
        'LABEL ai.eval.checkpoint.sha256="${CHECKPOINT_SHA256}"\n'
        'LABEL ai.eval.label_budget="33"\n'
        'LABEL ai.eval.activity_budget="33"\n'
        f'LABEL ai.eval.candidate="{candidate}"\n'
        "LABEL ai.eval.old_spint_decoder=\"false\"\n"
        f"COPY artifacts/{payload} /data/decoder.pkl\n"
        "COPY trf_falcon_decoder.py /trf_falcon_decoder.py\n"
        "COPY decode.py /decode.py\n"
        "ENV EVALUATION_LOC=remote TASK=m2 PHASE=test BATCH_SIZE=7\n"
        'CMD ["/bin/bash", "-c", "python /decode.py --evaluation $EVALUATION_LOC --model-path /data/decoder.pkl --split $TASK --phase $PHASE --batch-size $BATCH_SIZE"]\n',
        encoding="utf-8",
    )


def _docker_build(kind: str, payload_sha: str) -> dict[str, Any]:
    dest = _sub(kind)
    tag = (
        f"spint-t4-m2:small-trf-s1-ema-e19-{payload_sha[:8]}"
        if kind == "small"
        else f"spint-t4-m2:large-trf-s2-raw-e20-{payload_sha[:8]}"
    )
    cmd = [
        "docker",
        "build",
        "-t",
        tag,
        "--build-arg",
        f"PAYLOAD_SHA256={payload_sha}",
        str(dest),
    ]
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
        f"{_sub(kind) / 'trf_falcon_decoder.py'}:/trf_falcon_decoder.py:ro",
        image_tag,
        "python",
        "/trf_falcon_decoder.py",
        "--smoke-payload",
        "/data/decoder.pkl",
        "--smoke-window",
        "/data/smoke_window.npz",
    ]
    out = subprocess.check_output(cmd, text=True)
    last = [line for line in out.splitlines() if line.strip()][-1]
    report = json.loads(last)
    require(report.get("status") == "CONTAINER_SMOKE_PASS", f"container smoke failed: {out}")
    return {"status": "CONTAINER_PASS", "stdout": last, "cpu_only": True}


def _register(kind: str, candidate: dict[str, Any]) -> dict[str, Any]:
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
    return parsed


def run_slot(kind: str, *, stage: str = "all", skip_register: bool = False) -> dict[str, Any]:
    require(kind in {"small", "large"}, f"unknown kind {kind}")
    slot = "S1" if kind == "small" else "S2"
    write_slot(kind, status="FROZEN", owner="D", name="M2-SMALL-TRF-PICK" if kind == "small" else "M2-LARGE-TRF-PICK")
    report: dict[str, Any] = {"kind": kind, "slot": slot}
    dest = _sub(kind)
    _copy_submission_sources(kind)
    payload_path = dest / "artifacts" / _payload_name(kind)

    if stage in {"replay", "all"}:
        report["replay"] = replay_ext4(kind, device="cpu")
        write_slot(kind, status="FROZEN", replay_sha256=report["replay"]["replay_sha256"], replay_match=True)

    if stage in {"export", "all"}:
        if not payload_path.exists():
            report["payload"] = build_payload(kind, payload_path)
        else:
            report["payload"] = json.loads((payload_path.parent / "payload.receipt.json").read_text(encoding="utf-8"))
        write_slot(kind, payload_sha256=report["payload"]["payload_sha256"])

    if stage in {"parity", "all"}:
        report["parity"] = host_stream_parity(kind, payload_path)
        hours = estimate_cpu_hours(float(report["parity"]["ms_per_bin"]))
        report["cpu_hours_estimate"] = hours
        if kind == "large" and hours > 6.0:
            write_slot(
                kind,
                status="FROZEN",
                blocker=f"S2 CPU 6h risk: estimated {hours:.2f}h from {report['parity']['ms_per_bin']:.1f} ms/bin",
            )
            report["blocker"] = write_slot(kind)["blocker"]
            _write_worker_md(report)
            return report

    if stage in {"image", "all"}:
        method_label = C.S1_METHOD_LABEL if kind == "small" else C.S2_METHOD_LABEL
        candidate_name = "small_trf_s1_ema_e19" if kind == "small" else "large_trf_s2_raw_e20"
        payload_sha = report.get("payload", {}).get("payload_sha256") or sha256_file(payload_path)
        _write_dockerfile(kind, payload_sha, method_label, candidate_name)
        image = _docker_build(kind, payload_sha)
        report["image"] = image
        smoke = _container_smoke(
            kind,
            image["image_tag"],
            payload_path,
            Path(report["parity"]["smoke_window"]),
        )
        report["container"] = smoke
        write_slot(
            kind,
            status="CONTAINER_PASS",
            image_id=image["image_id"],
            image_tag=image["image_tag"],
        )

        candidate = {
            "arm": candidate_name,
            "image_tag": image["image_tag"],
            "image_id": image["image_id"],
            "payload_sha256": payload_sha,
            "method_label": method_label,
            "method_name": C.S1_METHOD_NAME if kind == "small" else C.S2_METHOD_NAME,
            "method_description": C.S1_METHOD_DESCRIPTION if kind == "small" else C.S2_METHOD_DESCRIPTION,
            "budget_disclosure": (
                "33 public calibration trials for native E0 and MOVE-T4; static banks; no TTA"
            ),
            "state_path": str(dest / "artifacts" / "evalai_push_state.json"),
        }
        report["candidate"] = candidate
        (dest / "artifacts" / "evalai_candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if stage in {"register", "all"} and not skip_register:
        registered = _register(kind, report["candidate"])
        report["register"] = registered
        write_slot(
            kind,
            status="REGISTERED",
            submission_id=registered.get("submission_id"),
            image_id=registered.get("image_id"),
            payload_sha256=registered.get("payload_sha256"),
        )
    _write_worker_md(report)
    return report


def _write_worker_md(report: dict[str, Any]) -> None:
    dest = C.SLOT_ROOT / "m2_D_worker.md"
    s1 = json.loads((C.SLOT_ROOT / "slot_S1.json").read_text(encoding="utf-8")) if (C.SLOT_ROOT / "slot_S1.json").exists() else {}
    s2 = json.loads((C.SLOT_ROOT / "slot_S2.json").read_text(encoding="utf-8")) if (C.SLOT_ROOT / "slot_S2.json").exists() else {}
    lines = [
        "# D worker report (M2 S1/S2)",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## S1 small Transformer",
        "",
        f"- status: `{s1.get('status')}`",
        f"- replay SHA: `{s1.get('replay_sha256')}` match={s1.get('replay_match')}",
        f"- payload: `{s1.get('payload_sha256')}`",
        f"- image: `{s1.get('image_id')}`",
        f"- submission_id: `{s1.get('submission_id')}`",
        "",
        "## S2 large Transformer",
        "",
        f"- status: `{s2.get('status')}`",
        f"- replay SHA: `{s2.get('replay_sha256')}` match={s2.get('replay_match')}",
        f"- payload: `{s2.get('payload_sha256')}`",
        f"- image: `{s2.get('image_id')}`",
        f"- submission_id: `{s2.get('submission_id')}`",
        f"- blocker: {s2.get('blocker', 'none')}",
        "",
        "S1 is a visible-product epoch-pick on local ext-4. S2 is an architecture probe; worst-session 0.2466 on 11-19 is disclosed.",
        "",
        "```json",
        json.dumps({"latest": {k: report.get(k) for k in ("kind", "slot", "cpu_hours_estimate", "blocker")}, "s1": s1, "s2": s2}, indent=2, sort_keys=True),
        "```",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
