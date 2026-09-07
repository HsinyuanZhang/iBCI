"""Stage-0 authoritative receipt runner.

Transactional, fail-closed receipt minting:

- CPU/env hard gate: refuses to run if user-site is enabled, if CUDA is visible,
  or if CUDA_VISIBLE_DEVICES is not the empty string;
- launch/final closure equality: the SHA-256 closure of every bound source file
  is hashed before the gates run and re-hashed before the receipt is written;
  any drift aborts with a failure receipt and nonzero exit;
- transactional write: the receipt is written to a private temporary file,
  fsynced, then atomically linked into place with O_EXCL so an existing
  receipt root can never be overwritten or partially written;
- failure semantics: a gate failure or an execution error still writes a
  failure receipt (transactionally) and exits nonzero; there is no partial pass.

Usage (from the package root; the runner is self-contained and does not rely on
pytest state):
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_stage0_synth.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # package root, so `src.tfpd` resolves

BOUND_PATTERNS = ("src/tfpd/*.py", "scripts/run_stage0_synth.py", "tests/test_stage0_gates.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_closure() -> dict:
    files = {}
    for pattern in BOUND_PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            files[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    closure = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"files": files, "closure_sha256": closure}


def enforce_environment() -> dict:
    problems = []
    if not sys.flags.no_user_site:
        problems.append("PYTHONNOUSERSITE is not set (user-site shadowing risk)")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        problems.append("CUDA_VISIBLE_DEVICES is not the empty string")
    if problems:
        print("environment hard gate failed: " + "; ".join(problems), file=sys.stderr)
        raise SystemExit(3)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "no_user_site": bool(sys.flags.no_user_site),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def write_receipt_transactionally(path: Path, payload: dict) -> None:
    """fsync temp file, then link into place with O_EXCL; no overwrite path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".receipt-", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        # O_EXCL link: fails if the receipt already exists — immutable roots.
        os.link(tmp_path, path)
        os.unlink(tmp_path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar_body = (sha256_file(path) + "  " + path.name + "\n").encode("utf-8")
        sfd, s_tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".sidecar-", suffix=".tmp")
        s_path = Path(s_tmp)
        with os.fdopen(sfd, "wb") as handle:
            handle.write(sidecar_body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(s_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.link(s_path, sidecar)
        os.unlink(s_path)
    except FileExistsError:
        tmp_path.unlink(missing_ok=True)
        print(f"refusing to overwrite existing receipt {path}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage0_synthetic_gates_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="bilinear,population_vector,bilinear_lowrank,bilinear_state_gain",
        help="comma-separated subset of gate models to run",
    )
    args = parser.parse_args()

    environment = enforce_environment()

    import torch

    from src.tfpd.gates import run_all_gates

    if torch.cuda.is_available():
        print("environment hard gate failed: CUDA is available to torch", file=sys.stderr)
        return 3

    receipt_path = args.output_root / "receipt.json"
    if receipt_path.exists():
        print(f"receipt root already exists: {args.output_root}", file=sys.stderr)
        return 2

    contract = ROOT / "docs/TFPD_STAGE0_SYNTHETIC_GATE_CONTRACT_20260815.md"
    launch_closure = source_closure()

    failure_payload_base = {
        "schema": "tfpd_stage0_synthetic_gates_v1",
        "contract": {
            "path": str(contract.relative_to(ROOT)),
            "sha256": sha256_file(contract),
        },
        "seed": args.seed,
        "environment": environment,
        "launch_closure": launch_closure,
    }

    try:
        results = run_all_gates(seed=args.seed, models=tuple(m.strip() for m in args.models.split(",") if m.strip()))
    except Exception as error:  # noqa: BLE001 — any gate crash is a fail-closed event
        failure_payload_base.update(
            {"status": "FAIL_EXECUTION_ERROR", "error": f"{type(error).__name__}: {error}"}
        )
        write_receipt_transactionally(receipt_path, failure_payload_base)
        print(f"execution error; failure receipt: {receipt_path}", file=sys.stderr)
        return 1

    final_closure = source_closure()
    if final_closure["closure_sha256"] != launch_closure["closure_sha256"]:
        failure_payload_base.update(
            {"status": "FAIL_SOURCE_CLOSURE_DRIFT", "final_closure": final_closure}
        )
        write_receipt_transactionally(receipt_path, failure_payload_base)
        print("source closure drifted between launch and final", file=sys.stderr)
        return 1

    all_passed = all(passed for model in results.values() for passed, _ in model.values())
    receipt = {
        **failure_payload_base,
        "final_closure": final_closure,
        "launch_final_closure_equal": True,
        "gates": {
            model: {gate: {"passed": passed, "detail": detail} for gate, (passed, detail) in gates.items()}
            for model, gates in results.items()
        },
        "status": "PASS_ALL_GATES" if all_passed else "FAIL",
        "authorizes": (
            "drafting of a Stage-1 seed-42 matched-pair contract for separate review"
            if all_passed
            else "nothing; fix implementation and rerun from a fresh receipt root"
        ),
    }
    write_receipt_transactionally(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}, indent=1))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
