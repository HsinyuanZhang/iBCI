#!/usr/bin/env python3
"""Seal metadata-only scorer-adapter parity-v2 prelaunch evidence.

This writer is deliberately source/JSON-only: it never imports the runtime
trace helper, torch, PyNWB, a model, or the score runners.  It does not open a
checkpoint, normalizer NPZ, or NWB.  Its only purpose is to bind the final
score-only-v2 source/prelaunch closure before a later parity execution may even
be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v2"
DOC = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V2.md"
V1_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v1"
V2_ROOT = "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v2"

FINAL_V2_SOURCES = {
    "sua_exploration/mc_maze/subm_co_score_only_v2.py": "213e5495b4fa8776967ea07bf57743db18619b74aea593bac1d17d5944308e4d",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v2.py": "dc271dbe0b7ede31865f5c35890444b969296096f9107f960183079cc63a76c7",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v2.py": "d8f4b37a091233c421bd052238c165c194bb31318ed2c8172e0049a444d2f58c",
    "sua_exploration/tests/test_dandi688_subm_co_score_only_v2.py": "944e5a35ca7c61d18655a4aa4e3c661252cd8cc9e97e687034a3c28f4d3ff339",
}
FINAL_V2_PRELAUNCH = {
    "prelaunch_authorization_draft.json": "2a8455fff85e8a0644e5dbb1f9932f126db55b856842cc3207d3f59bf7c59eae",
    "receipt.json": "44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868",
    "prelaunch_artifact_seal.json": "2b8a3591eb7b3d18b2587f38526cbe139c7b7d40686d8a041b992f07f13635ea",
}
V1_PARITY_ARTIFACTS = {
    "parity_protocol_draft.json": "410e48d76beff0e63cf89d67fe5467d68952a155134878d8d50904924fe28204",
    "receipt.json": "90c57871c8f60135973a29f73eb945f36e23f5de20ed6d65c3c35ae0870486a4",
    "seal.json": "093ab7be963caa2f04c4d7e55ddae31ee3f186ee1c2a68cdd03e33e681e06a54",
}
PARITY_V2_SOURCES = (
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v2.py",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v2.py",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v2.py",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v2.py",
)
MANIFEST = "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/c1_train_val_33_manifest.json"
METADATA = "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s44/run_metadata.json"


class StaticParityV2Error(RuntimeError):
    """A static source/prelaunch mismatch that prohibits parity execution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticParityV2Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticParityV2Error(f"cannot parse {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _pinned_files(root: Path, relative_root: str, pins: Mapping[str, str], *, label: str) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, expected in pins.items():
        path = (root / relative_root / name).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise StaticParityV2Error(f"unsafe {label} path: {name}") from exc
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {name}")
        observed = sha256_file(path)
        require(observed == expected, f"{label} SHA drift: {name}")
        verified[name] = {"path": str(path.relative_to(root)), "sha256": observed, "bytes": path.stat().st_size}
    return verified


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    """Verify source and JSON pins without importing runtime/data code."""
    root = root.resolve()
    final_sources = _pinned_files(root, "", FINAL_V2_SOURCES, label="final score-only-v2 source")
    final_prelaunch = _pinned_files(root, V2_ROOT, FINAL_V2_PRELAUNCH, label="final score-only-v2 prelaunch")
    old = _pinned_files(root, V1_ROOT, V1_PARITY_ARTIFACTS, label="preserved parity-v1 artifact")
    v1_draft = read_json(root / V1_ROOT / "parity_protocol_draft.json", "preserved parity-v1 draft")
    old_v2_sha = v1_draft.get("static_authority", {}).get("source_hashes", {}).get("sua_exploration/mc_maze/subm_co_score_only_v2.py")
    require(isinstance(old_v2_sha, str) and old_v2_sha != FINAL_V2_SOURCES["sua_exploration/mc_maze/subm_co_score_only_v2.py"], "parity-v1 is not demonstrably bound to a prior source revision")
    v2_draft = read_json(root / V2_ROOT / "prelaunch_authorization_draft.json", "final score-only-v2 draft")
    v2_receipt = read_json(root / V2_ROOT / "receipt.json", "final score-only-v2 receipt")
    v2_seal = read_json(root / V2_ROOT / "prelaunch_artifact_seal.json", "final score-only-v2 seal")
    expected_snapshot = dict(FINAL_V2_SOURCES)
    require(v2_draft.get("runner_source_snapshot") == expected_snapshot, "final score-only-v2 draft source snapshot drift")
    require(v2_receipt.get("runner_source_snapshot") == expected_snapshot, "final score-only-v2 receipt source snapshot drift")
    require(v2_seal.get("runner_source_snapshot") == expected_snapshot, "final score-only-v2 seal source snapshot drift")
    require(v2_draft.get("status") == v2_receipt.get("status") == v2_seal.get("status") == "NOT_AUTHORIZED_FOR_SCORING", "final score-only-v2 prelaunch is not blocked")
    parity_sources = _pinned_files(root, "", {path: sha256_file(root / path) for path in PARITY_V2_SOURCES}, label="parity-v2 source")
    helper_source = (root / PARITY_V2_SOURCES[0]).read_text(encoding="utf-8")
    for symbol in ("def prepare_fixed_c1_batch(", "def capture_c1_reference_trace(", "def capture_future_adapter_trace(", "def assert_trace_parity(", "def execute_parity_once_not_authorized("):
        require(symbol in helper_source, f"shared parity-v2 helper is missing {symbol}")
    runner_source = (root / PARITY_V2_SOURCES[1]).read_text(encoding="utf-8")
    require('choices=("dry-run", "execute")' in runner_source and "PARITY_EXECUTION_NOT_AUTHORIZED" in runner_source, "parity-v2 runner hard fence drift")
    document_path = root / DOC
    require(document_path.is_file() and not document_path.is_symlink(), "parity-v2 protocol document missing")
    doc_sha = sha256_file(document_path)
    text = document_path.read_text(encoding="utf-8")
    require("STALE_NON_AUTHORIZING_SUPERSEDED" in text and "R2_RECOMPUTE_ATOL = 1e-6" in text, "parity-v2 document disposition/tolerance drift")
    manifest_path = root / MANIFEST
    metadata_path = root / METADATA
    manifest = read_json(manifest_path, "C1 data manifest")
    metadata = read_json(metadata_path, "C1 shared-T4 seed-44 metadata")
    require(sha256_file(manifest_path) == "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb", "C1 manifest SHA drift")
    require(sha256_file(metadata_path) == "d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7", "C1 metadata SHA drift")
    val = manifest.get("file_inventory", {}).get("val")
    require(isinstance(val, list) and val and val[0].get("session") == "sub-C_ses-CO-20151103", "fixed consumed development session drift")
    require(metadata.get("seed") == 44 and metadata.get("training_kind") == "shared_paired_view", "fixed shared-T4 seed-44 metadata drift")
    require(metadata.get("formal_sua_files_opened") is False and metadata.get("held_out_test_evaluated") is False, "C1 metadata formal access drift")
    return {
        "final_score_only_v2_sources": final_sources,
        "final_score_only_v2_prelaunch": final_prelaunch,
        "parity_v1_disposition": {
            "status": "STALE_NON_AUTHORIZING_SUPERSEDED",
            "preserved_artifacts": old,
            "old_core_sha256": old_v2_sha,
            "final_core_sha256": FINAL_V2_SOURCES["sua_exploration/mc_maze/subm_co_score_only_v2.py"],
            "superseded_scope": "only parity prelaunch source/prelaunch binding; v1 bytes and non-authorizing status are retained",
        },
        "parity_v2_sources": parity_sources,
        "protocol_document": {"path": DOC, "sha256": doc_sha},
        "fixture_metadata": {
            "manifest": {"path": MANIFEST, "sha256": sha256_file(manifest_path)},
            "run_metadata": {"path": METADATA, "sha256": sha256_file(metadata_path)},
            "session": "sub-C_ses-CO-20151103",
            "view": "sua",
            "checkpoint": "shared_t4 seed44 epoch_011",
        },
    }


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    authority = static_authority(root)
    return {
        "schema_version": 2,
        "kind": "dandi_000688_subc_scorer_adapter_parity_prelaunch_draft_v2",
        "status": "NOT_AUTHORIZED_FOR_PARITY_EXECUTION",
        "append_only": True,
        "authority": authority,
        "fixed_fixture": {
            "view": "sua",
            "checkpoint": {"arm": "shared_t4", "seed": 44, "epoch": "epoch_011.ckpt", "sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6"},
            "consumed_development_session": {"session": "sub-C_ses-CO-20151103", "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7"},
            "support_query_batch": {"support": "trials[0:50]", "identity": "first_n30", "query": "trials[50:]", "loader": {"batch_size": 128, "shuffle": False, "num_workers": 0}, "rows": "0:16"},
            "device": {"torch_device": "cpu", "cuda_visible_devices": "", "cuda_initialization": "FORBIDDEN", "deterministic_algorithms": True, "tf32": False, "autocast": False, "threads": 1},
        },
        "shared_trace_contract": {
            "one_prepared_c1_batch_passed_to_reference_and_adapter": True,
            "reference_and_adapter_share_trace_scaling_target_metric_path": True,
            "independent_loader_or_metric_loop": "FORBIDDEN",
            "prediction_exact": True,
            "r2_recompute_atol": 1.0e-6,
        },
        "future_execution_artifacts": ["input_trace.json", "c1_reference_trace.json", "adapter_trace.json", "reference_prediction_target.npz", "adapter_prediction_target.npz", "receipt.json", "seal.json"],
        "blocked_until": [
            "root provides a valid one-time authorization bound to this v2 draft/receipt/seal",
            "a future adapter implements PreparedBatchAdapter without a second data/metric path",
            "root Ed25519 key and a later score-runner parity receipt pin are appended separately",
        ],
        "operations_by_this_prelaunch": {
            "checkpoint_files_opened": 0,
            "normalizer_files_opened": 0,
            "nwb_files_opened": 0,
            "model_forward_calls": 0,
            "torchmetrics_calls": 0,
            "gpu_used": False,
            "subm_nwb_paths_constructed": False,
            "subm_nwb_files_accessed": False,
        },
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": 2,
        "receipt_kind": "dandi_000688_subc_scorer_adapter_parity_prelaunch_receipt_v2",
        "status": "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED",
        "append_only": True,
        "draft": {"filename": "parity_protocol_draft.json", "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest()},
        "operations": dict(draft["operations_by_this_prelaunch"]),
        "blockers": list(draft["blocked_until"]),
        "v1_disposition": draft["authority"]["parity_v1_disposition"],
    }
    return draft, receipt


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"immutable mode failed: {path}")
    return hashlib.sha256(payload).hexdigest()


def write_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"parity-v2 prelaunch root already exists: {output_dir}")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path, receipt_path, seal_path = output_dir / "parity_protocol_draft.json", output_dir / "receipt.json", output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(draft_sha == receipt["draft"]["sha256"], "internal parity-v2 draft binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": 2,
        "kind": "dandi_000688_subc_scorer_adapter_parity_prelaunch_seal_v2",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {"path": draft_path.name, "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ],
        "parity_execution_still_not_authorized": True,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {"output_dir": str(output_dir), "draft": {"path": str(draft_path), "sha256": draft_sha}, "receipt": {"path": str(receipt_path), "sha256": receipt_sha}, "seal": {"path": str(seal_path), "sha256": seal_sha}, "status": receipt["status"]}


def verify_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    files = (prelaunch_dir / "parity_protocol_draft.json", prelaunch_dir / "receipt.json", prelaunch_dir / "seal.json")
    for path in files:
        require(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"stored parity-v2 prelaunch file missing/unsafe/mutable: {path.name}")
    draft, receipt, seal = (read_json(files[0], "stored parity-v2 draft"), read_json(files[1], "stored parity-v2 receipt"), read_json(files[2], "stored parity-v2 seal"))
    draft_sha, receipt_sha = sha256_file(files[0]), sha256_file(files[1])
    require(draft.get("status") == "NOT_AUTHORIZED_FOR_PARITY_EXECUTION", "stored parity-v2 draft status drift")
    require(receipt.get("status") == seal.get("status") == "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED", "stored parity-v2 receipt/seal status drift")
    require(receipt.get("draft") == {"filename": files[0].name, "sha256": draft_sha}, "stored parity-v2 draft receipt binding drift")
    require(seal.get("artifacts") == [{"path": files[0].name, "sha256": draft_sha, "bytes": files[0].stat().st_size, "mode": "0444"}, {"path": files[1].name, "sha256": receipt_sha, "bytes": files[1].stat().st_size, "mode": "0444"}], "stored parity-v2 seal binding drift")
    live = static_authority(root)
    require(draft.get("authority") == live, "stored parity-v2 authority no longer matches live final pins")
    return {"draft_sha256": draft_sha, "receipt_sha256": receipt_sha, "seal_sha256": sha256_file(files[2]), "status": receipt["status"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = static_authority(args.repo_root) if args.verify_only else write_prelaunch(args.output_dir, args.repo_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except StaticParityV2Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
