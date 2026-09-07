#!/usr/bin/env python3
"""Build the M1 V3 selected-T image. Does not register."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/xinyuan/Work_host/SPINT")
DEST = ROOT / "tfpd_exploration/submissions/evalai_m1_runtime_v3_selected_t_v1"
PAYLOAD = DEST / "artifacts/m1_optimized_v2_t_ema_e6.pkl"
PICK = ROOT / "tfpd_exploration/results/m1_optimized_v2_source_dev_epoch_pick_v1/selection.json"
PAYLOAD_SHA256 = "5e44fc37965640138bf977da42a988e7636daa996d1eb9b50a75fa5628f9d7b3"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> dict:
    payload_sha = sha256_file(PAYLOAD)
    if payload_sha != PAYLOAD_SHA256:
        raise RuntimeError(f"payload drift {payload_sha}")
    pick = json.loads(PICK.read_text(encoding="utf-8"))
    if pick.get("submit_operator") != "current_query" or int(pick["image_state"]["epoch_one_based"]) != 6:
        raise RuntimeError("selection is not T e6")
    method_name = "M1 optimized-v2 current-query T EMA e6 runtime-v3 t2 w0 cast-f32"
    method_label = (
        "M1 current-query Transformer V3 (formal12_chron80_v2 seed42 EMA e6) t2 w0; "
        "official uint8/float coerce; source-dev epoch-pick; not outer-24"
    )
    image_tag = f"spint-m1:v3-t-ema-e6-t2-{payload_sha[:8]}"
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            image_tag,
            "--build-arg",
            f"PAYLOAD_SHA256={payload_sha}",
            str(DEST),
        ],
        check=True,
    )
    inspect = subprocess.check_output(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Id}} {{.Size}}"],
        text=True,
    ).strip()
    image_id, size = inspect.split()
    fixture = DEST / "fixtures/source_b4_public_parity.npz"
    parity_out = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--cpuset-cpus=8-11",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "MKL_NUM_THREADS=2",
            "-e",
            "OPENBLAS_NUM_THREADS=2",
            "-e",
            "NUMEXPR_NUM_THREADS=2",
            "-v",
            f"{DEST.resolve()}:/work:ro",
            image_tag,
            "python",
            "/work/container_parity.py",
        ],
        text=True,
    )
    last = [line for line in parity_out.splitlines() if line.strip()][-1]
    if "PASS" not in last:
        raise RuntimeError(f"container parity failed: {parity_out}")
    candidate = {
        "arm": "m1_optv2_t_ema_e6_runtime_v3_t2_w0",
        "budget_disclosure": (
            "10 public calibration trials for B3 E0 and rSyn3-refit-v1 T; "
            "source-dev epoch-pick on ses-20120926/27/28 chronological 80/20 tails; "
            "ses-20120924 query unread. V3 stream, torch threads=2, dataloader_workers=0."
        ),
        "image_id": image_id,
        "image_tag": image_tag,
        "image_size": int(size),
        "method_description": (
            "M1 optimized-v2 current-query Transformer, seed 42 EMA epoch 6, "
            "V3 heterogeneous stream (W=100/k=5), torch threads=2, workers=0. "
            "Adapter casts official uint8/float64/noncontiguous bins to C-contiguous float32. "
            f"Source-dev equal-session mean {pick['selected']['current_query']['external_equal_session_mean']:.6f}. "
            "Not an outer-24 result. Replaces failed 581980 dtype reject."
        ),
        "method_label": method_label,
        "method_name": method_name,
        "payload_sha256": payload_sha,
        "register": False,
        "replaces": "581980 failed on official uint8 dtype reject",
        "selection_path": str(PICK),
        "state_path": str(DEST / "artifacts/evalai_push_state.json"),
        "container_parity": last,
    }
    (DEST / "artifacts/evalai_candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (DEST / "artifacts/evalai_push_state.json").write_text(
        json.dumps(
            {
                "arm": candidate["arm"],
                "image_tag": image_tag,
                "payload_sha256": payload_sha,
                "schema_version": "m1_v3_evalai_push_state",
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
