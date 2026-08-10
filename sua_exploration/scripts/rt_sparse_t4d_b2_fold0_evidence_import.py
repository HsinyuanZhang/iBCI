#!/usr/bin/env python3
"""SHA-verified, non-destructive import of the sealed RT B2-D1024 fold-0 evidence.

This imports immutable legacy bytes only; it neither invokes Torch nor opens
NWB data, starts no GPU work, and never rewrites the historical remote run.
The imported checkpoint is a local preservation copy.  The forward-only
re-evaluation still uses the source-compatible remote layout for fold 0 unless
a dedicated path-virtualization adapter is separately audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HOST = "xinyuan@100.103.97.12"
REMOTE_FIT = (
    "/home/xinyuan/Work_host/SPINT/rt_clean_nested_5070ti_stage/streaming_calibration_exp/"
    "outputs/rt_stage_r_b2_remote/gpu_runs_zero4_v2/b2_d1024_zero4/fold_00/seed_42/fit"
)
REMOTE_TEACHER = (
    "/home/xinyuan/Work_host/SPINT/rt_clean_nested_5070ti_stage/streaming_calibration_exp/"
    "outputs/rt_stage_r_b2_remote/gpu_runs_zero4_v2/_artifacts/"
    "rt_clean_nested_loso_m24_b2_d1024_zero4_f0_s42_20260808_004655/teacher_metadata.json"
)
REMOTE_FILES = {
    "checkpoint": REMOTE_FIT + "/checkpoints/best_ckpt/epoch_008.ckpt",
    "selection": REMOTE_FIT + "/rt_nested_selection_receipt.json",
    "split": REMOTE_FIT + "/split_manifest.json",
    "config": REMOTE_FIT + "/.hydra/config.yaml",
    "teacher_metadata": REMOTE_TEACHER,
}
DESTINATIONS = {
    "checkpoint": "fit/checkpoints/best_ckpt/epoch_008.ckpt",
    "selection": "fit/rt_nested_selection_receipt.json",
    "split": "fit/split_manifest.json",
    "config": "fit/.hydra/config.yaml",
    "teacher_metadata": "teacher_metadata.json",
}
DEFAULT_ROOT = ROOT / "sua_exploration/results/rt_sparse_t4d_b2_forward_reeval_v1/imported_fold0_legacy_v1"


class ImportErrorClosed(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ImportErrorClosed(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_sha(host: str) -> dict[str, str]:
    program = "sha256sum " + " ".join(shlex.quote(path) for path in REMOTE_FILES.values())
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host, program],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    _need(completed.returncode == 0, f"remote sha256sum failed: {completed.stderr.strip()}")
    rows = [line.split(maxsplit=1) for line in completed.stdout.splitlines() if line.strip()]
    found = {path.strip(): digest for digest, path in rows if len(digest) == 64}
    result = {name: found.get(path, "") for name, path in REMOTE_FILES.items()}
    _need(all(len(value) == 64 for value in result.values()), "remote SHA response omitted an evidence file")
    return result


def _copy_one(host: str, remote: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _need(not destination.exists(), f"refusing to overwrite imported evidence: {destination}")
    with tempfile.NamedTemporaryFile(prefix=".fold0-import-", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        completed = subprocess.run(
            ["scp", "-p", f"{host}:{remote}", str(temporary)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        _need(completed.returncode == 0, f"scp failed for {remote}: {completed.stderr.strip()}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def import_fold0(*, destination_root: Path = DEFAULT_ROOT, host: str = HOST) -> dict[str, Any]:
    _need(not destination_root.exists(), f"refusing to overwrite existing evidence root: {destination_root}")
    remote_sha = _remote_sha(host)
    destination_root.mkdir(parents=True, exist_ok=False)
    try:
        local: dict[str, dict[str, str]] = {}
        for name, remote in REMOTE_FILES.items():
            destination = destination_root / DESTINATIONS[name]
            _copy_one(host, remote, destination)
            local_sha = _sha(destination)
            _need(local_sha == remote_sha[name], f"SHA mismatch after transfer: {name}")
            os.chmod(destination, 0o444)
            local[name] = {"path": str(destination), "sha256": local_sha, "remote_path": remote}
        receipt = {
            "schema": "rt_sparse_t4d_b2_fold0_evidence_import_v1",
            "status": "PASS_SHA_VERIFIED_NONDESTRUCTIVE_IMPORT",
            "host": host,
            "fold": 0,
            "seed": 42,
            "files": local,
            "non_interference": {
                "remote_write": False, "torch_imported": False, "nwb_opened": False,
                "gpu_opened": False, "training_started": False, "optimizer_constructed": False,
            },
            "note": "Imported evidence is preservation-only. The historical selection receipt retains remote absolute paths; fold-0 forward-only scoring must use the original remote layout or a separately audited path adapter.",
        }
        output = destination_root / "FOLD0_IMPORT_RECEIPT_v1.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(output, 0o444)
        return {"status": receipt["status"], "receipt": str(output), "receipt_sha256": _sha(output), "files": local}
    except Exception:
        # The root stays as evidence of an interrupted import rather than being
        # deleted.  Every copied file is read-only only after its SHA passes.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args()
    print(json.dumps(import_fold0(destination_root=args.destination_root, host=args.host), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ImportErrorClosed as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2))
        raise SystemExit(2) from error
