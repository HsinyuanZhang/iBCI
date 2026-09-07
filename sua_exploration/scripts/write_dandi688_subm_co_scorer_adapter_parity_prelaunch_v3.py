#!/usr/bin/env python3
"""Write immutable metadata-only prelaunch evidence for data-adapter parity v3.

This writer is source/JSON only. It never imports Torch, PyNWB, the v3 runtime
helper, score runners, model/data owners, or checkpoint/NWB/normalizer data.
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
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v3"
DOC = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V3.md"
SCORE_V2_ROOT = "sua_exploration/results/dandi_000688_subm_co_score_only_prelaunch_v2"
PARITY_V1_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v1"
PARITY_V2_ROOT = "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v2"
MANIFEST = "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/c1_train_val_33_manifest.json"
METADATA = "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s44/run_metadata.json"

FINAL_SCORE_ONLY_V2_SOURCES = {
    "sua_exploration/mc_maze/subm_co_score_only_v2.py": "213e5495b4fa8776967ea07bf57743db18619b74aea593bac1d17d5944308e4d",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v2.py": "dc271dbe0b7ede31865f5c35890444b969296096f9107f960183079cc63a76c7",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v2.py": "d8f4b37a091233c421bd052238c165c194bb31318ed2c8172e0049a444d2f58c",
    "sua_exploration/tests/test_dandi688_subm_co_score_only_v2.py": "944e5a35ca7c61d18655a4aa4e3c661252cd8cc9e97e687034a3c28f4d3ff339",
}
FINAL_SCORE_ONLY_V2_PRELAUNCH = {
    "prelaunch_authorization_draft.json": "2a8455fff85e8a0644e5dbb1f9932f126db55b856842cc3207d3f59bf7c59eae",
    "receipt.json": "44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868",
    "prelaunch_artifact_seal.json": "2b8a3591eb7b3d18b2587f38526cbe139c7b7d40686d8a041b992f07f13635ea",
}
PARITY_V1_ARTIFACTS = {
    "parity_protocol_draft.json": "410e48d76beff0e63cf89d67fe5467d68952a155134878d8d50904924fe28204",
    "receipt.json": "90c57871c8f60135973a29f73eb945f36e23f5de20ed6d65c3c35ae0870486a4",
    "seal.json": "093ab7be963caa2f04c4d7e55ddae31ee3f186ee1c2a68cdd03e33e681e06a54",
}
PARITY_V2_ARTIFACTS = {
    "parity_protocol_draft.json": "0ad38e787666e4ef7a9e73c8ed0c3cddcb1c59a264f700499cc845be7b2dae28",
    "receipt.json": "33876977d1a6ee2aed5b0f46d51651a15d76c24060f9f32fce9fc79875a68602",
    "seal.json": "b978108608e79fcf0b1497eb045ba668df1717ae2ed3e88b88e8e0b542e75bb9",
}
PARITY_V2_SOURCES = {
    "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V2.md": "a9ad636a7485be80a8b7608eceb5bc4e8d9e6de1f364e1752ad304f9e095f9ed",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v2.py": "6574ff63b056efbd56092123408559e8f7e3ded1696eb9854ca5df50d4268e8f",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v2.py": "a81c2de5d666c2b6e2ba42ac5b6a22640bdbd1199296340cecd0e45d8c7cbc7c",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v2.py": "ac9bad9b4f3e62fe8c55b9c45a33e835fbcbbee0a5d7e5f9534e8c8fa2c70d77",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v2.py": "10ff360d1721dfeb2e893a28fd53904379bf6bff9470a0954220beed02661d37",
}
V3_SOURCES = (
    DOC,
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v3.py",
    "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v3.py",
    "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v3.py",
    "sua_exploration/tests/test_dandi688_subm_co_scorer_adapter_parity_v3.py",
)
V1_OWNER_SOURCES = {
    "sua_exploration/mc_maze/subm_co_score_only.py": "8e6aa7b9efdeb894dd0f04a06aae0b01226aafb9ea3a6c493b9cb571648a9be4",
    "sua_exploration/mc_maze/multisession_datamodule.py": "674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d",
    "sua_exploration/mc_maze/datamodule.py": "0c93359991c32e81b552e00169fa1f31a5b71d345782c9bf81b136bd5c708506",
    "sua_exploration/mc_maze/unit_side_features.py": "059faefcd766dfc8e25253d9ded2b619a46dea408e6f00a30cfa5b2ecd185ab6",
    "sua_exploration/scripts/eval_adaptation_dandi688.py": "e452d19d738316a2ff54074585b22e526bb1cc9275bfcfe5d33aa1becc5ccc30",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py": "1fee6f482b1cd93e867b8f1b64bd13bdb967881b56f5a5563feaa635091daeb9",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py": "a0d1b331c0548a967bbd08804231c2045a360e8947f869dea2d8c251e4d70e68",
}


class StaticParityV3Error(RuntimeError):
    """A static source/prelaunch mismatch that prohibits execution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticParityV3Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticParityV3Error(f"cannot parse {label}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _pinned_files(
    root: Path, relative_root: str, pins: Mapping[str, str], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, expected in pins.items():
        path = (root / relative_root / name).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise StaticParityV3Error(f"unsafe {label} path: {name}") from exc
        require(path.is_file() and not path.is_symlink(), f"missing {label}: {name}")
        observed = sha256_file(path)
        require(observed == expected, f"{label} SHA drift: {name}")
        result[name] = {
            "path": str(path.relative_to(root)),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return result


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    """Check source/JSON pins only; runtime/data/model imports are prohibited."""
    root = root.resolve()
    final_sources = _pinned_files(
        root, "", FINAL_SCORE_ONLY_V2_SOURCES, label="final score-only-v2 source"
    )
    final_prelaunch = _pinned_files(
        root, SCORE_V2_ROOT, FINAL_SCORE_ONLY_V2_PRELAUNCH,
        label="final score-only-v2 prelaunch",
    )
    parity_v1 = _pinned_files(
        root, PARITY_V1_ROOT, PARITY_V1_ARTIFACTS, label="preserved parity-v1 artifact"
    )
    parity_v2 = _pinned_files(
        root, PARITY_V2_ROOT, PARITY_V2_ARTIFACTS, label="preserved parity-v2 artifact"
    )
    parity_v2_sources = _pinned_files(
        root, "", PARITY_V2_SOURCES, label="preserved parity-v2 source"
    )
    owner_sources = _pinned_files(
        root, "", V1_OWNER_SOURCES, label="score-only-v1 owner source"
    )
    v3_sources = _pinned_files(
        root, "", {source: sha256_file(root / source) for source in V3_SOURCES},
        label="v3 source",
    )

    v2_draft = read_json(root / PARITY_V2_ROOT / "parity_protocol_draft.json", "preserved parity-v2 draft")
    v2_receipt = read_json(root / PARITY_V2_ROOT / "receipt.json", "preserved parity-v2 receipt")
    v2_seal = read_json(root / PARITY_V2_ROOT / "seal.json", "preserved parity-v2 seal")
    require(v2_draft.get("status") == "NOT_AUTHORIZED_FOR_PARITY_EXECUTION", "parity-v2 draft unexpectedly authorizes")
    require(
        v2_receipt.get("status") == v2_seal.get("status")
        == "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED",
        "parity-v2 receipt/seal status drift",
    )
    require(
        v2_draft.get("shared_trace_contract", {}).get(
            "one_prepared_c1_batch_passed_to_reference_and_adapter"
        )
        is True,
        "parity-v2 insufficiency evidence drift",
    )

    score_v2_draft = read_json(root / SCORE_V2_ROOT / "prelaunch_authorization_draft.json", "final score-only-v2 draft")
    score_v2_receipt = read_json(root / SCORE_V2_ROOT / "receipt.json", "final score-only-v2 receipt")
    score_v2_seal = read_json(root / SCORE_V2_ROOT / "prelaunch_artifact_seal.json", "final score-only-v2 seal")
    expected_snapshot = dict(FINAL_SCORE_ONLY_V2_SOURCES)
    require(score_v2_draft.get("runner_source_snapshot") == expected_snapshot, "final score-only-v2 draft source snapshot drift")
    require(score_v2_receipt.get("runner_source_snapshot") == expected_snapshot, "final score-only-v2 receipt source snapshot drift")
    require(score_v2_seal.get("runner_source_snapshot") == expected_snapshot, "final score-only-v2 seal source snapshot drift")
    require(
        score_v2_draft.get("status") == score_v2_receipt.get("status")
        == score_v2_seal.get("status") == "NOT_AUTHORIZED_FOR_SCORING",
        "final score-only-v2 prelaunch is not blocked",
    )

    helper = (root / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v3.py").read_text(encoding="utf-8")
    for text in (
        "def prepare_c1_reference_fixture(",
        "def prepare_score_only_adapter_fixture(",
        "def _forward_and_observe(",
        "def compare_prediction_target_exact(",
        "def concrete_parity_after_future_authorization(",
        "c1.load_session_with_trials(",
        "owners[\"load_dandi688_session\"](",
        "owners[\"load_unit_side_features\"](",
        "owners[\"MCMazeSessionDataset\"](",
        "owners[\"list_datamodule_rewarded_trials\"](",
        "calibration_n_trials=SUPPORT_TRIALS",
        "exclude_calibration_trials_from_windows=True",
        "rebuild_record, indices, IDENTITY_TRIALS",
        "LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30",
    ):
        require(text in helper, f"v3 concrete adapter helper missing: {text}")
    require("Protocol" not in helper, "v3 may not use a placeholder adapter protocol")
    require(helper.count("def _forward_and_observe(") == 1, "v3 has more than one fixed forward helper")

    document = root / DOC
    doc_text = document.read_text(encoding="utf-8")
    require(
        "INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING" in doc_text
        and "LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30" in doc_text,
        "v3 document correction gate/disposition drift",
    )
    manifest = read_json(root / MANIFEST, "C1 manifest")
    metadata = read_json(root / METADATA, "C1 seed-44 metadata")
    require(
        sha256_file(root / MANIFEST)
        == "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb",
        "C1 manifest SHA drift",
    )
    require(
        sha256_file(root / METADATA)
        == "d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7",
        "C1 metadata SHA drift",
    )
    val = manifest.get("file_inventory", {}).get("val")
    require(
        isinstance(val, list) and val and val[0].get("session") == "sub-C_ses-CO-20151103",
        "fixed consumed session drift",
    )
    require(
        metadata.get("seed") == 44 and metadata.get("training_kind") == "shared_paired_view",
        "fixed shared-T4 seed-44 metadata drift",
    )
    return {
        "final_score_only_v2_sources": final_sources,
        "final_score_only_v2_prelaunch": final_prelaunch,
        "parity_v1_preserved": parity_v1,
        "parity_v2_disposition": {
            "status": "INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING",
            "preserved_artifacts": parity_v2,
            "preserved_sources": parity_v2_sources,
            "reason": (
                "v2 passed one C1 prepared batch to both labels and did not "
                "independently exercise load_dandi688_session/"
                "load_unit_side_features/MCMazeSessionDataset"
            ),
        },
        "score_only_v1_owner_sources": owner_sources,
        "v3_sources": v3_sources,
        "fixed_fixture": {
            "manifest": {"path": MANIFEST, "sha256": sha256_file(root / MANIFEST)},
            "run_metadata": {"path": METADATA, "sha256": sha256_file(root / METADATA)},
            "session": "sub-C_ses-CO-20151103",
            "view": "sua",
            "checkpoint": "shared_t4 seed44 epoch_011",
        },
    }


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    authority = static_authority(root)
    return {
        "schema_version": 3,
        "kind": "dandi_000688_subc_scorer_adapter_data_parity_prelaunch_draft_v3",
        "status": "NOT_AUTHORIZED_FOR_PARITY_EXECUTION",
        "append_only": True,
        "authority": authority,
        "data_adapter_protocol": {
            "fixture": {
                "session": "sub-C_ses-CO-20151103",
                "view": "sua",
                "arm": "shared_t4",
                "seed": 44,
            },
            "reference_owner_chain": [
                "load_session_with_trials",
                "select_calibration_trial_indices(first_n30)",
                "build_calib_trials_for_indices",
                "attach_side_features",
                "make_subset_dataset",
            ],
            "adapter_owner_chain": [
                "load_dandi688_session",
                "list_datamodule_rewarded_trials",
                "load_unit_side_features",
                "MCMazeSessionDataset",
            ],
            "calibration_gate": "LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30",
            "loader_query_boundary": "adapter loader receives 50 and owns post-50 valid_starts",
            "identity_activity": "C1 first_n30 rebuilt from independent adapter neural data and adapter owner chronology by C1 existing builder",
            "t4_pool_size": 50,
            "reference_batch_reused_by_adapter": False,
            "input_gate_before_forward": [
                "full chronology",
                "valid_starts",
                "neural",
                "behavior",
                "calibration",
                "side_features",
                "electrode_ids",
                "whole first batch",
                "first16 dtype/shape/value/hash",
            ],
            "shared_forward_and_metric": {
                "helper": "_forward_and_observe",
                "prediction_target_exact": True,
                "r2_atol": 1.0e-6,
            },
        },
        "operations_by_this_prelaunch": {
            "checkpoint_files_opened": 0,
            "normalizer_files_opened": 0,
            "nwb_files_opened": 0,
            "runtime_helper_imported": False,
            "model_forward_calls": 0,
            "torchmetrics_calls": 0,
            "gpu_used": False,
            "subm_nwb_paths_constructed": False,
            "subm_nwb_files_accessed": False,
        },
        "blocked_until": [
            "root reviews this v3 prelaunch and provides a valid one-time append-only parity authorization",
            "the independently loaded C1/adapter inputs pass the v3 exact observer gate",
            "a later sealed runtime receipt and independent root review are appended; no external endpoint score is authorized by v3",
        ],
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": 3,
        "receipt_kind": "dandi_000688_subc_scorer_adapter_data_parity_prelaunch_receipt_v3",
        "status": "STATIC_PROTOCOL_AUDITED_V3_DATA_ADAPTER_PARITY_EXECUTION_NOT_AUTHORIZED",
        "append_only": True,
        "draft": {
            "filename": "parity_protocol_draft.json",
            "sha256": hashlib.sha256(canonical_bytes(draft)).hexdigest(),
        },
        "operations": dict(draft["operations_by_this_prelaunch"]),
        "blockers": list(draft["blocked_until"]),
        "parity_v2_disposition": dict(draft["authority"]["parity_v2_disposition"]),
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
    require(not output_dir.exists(), f"parity-v3 prelaunch root already exists: {output_dir}")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "parity_protocol_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "seal.json"
    draft_sha = _write_immutable(draft_path, draft)
    require(draft_sha == receipt["draft"]["sha256"], "v3 draft binding drift")
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema_version": 3,
        "kind": "dandi_000688_subc_scorer_adapter_data_parity_prelaunch_seal_v3",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {
                "path": draft_path.name,
                "sha256": draft_sha,
                "bytes": draft_path.stat().st_size,
                "mode": "0444",
            },
            {
                "path": receipt_path.name,
                "sha256": receipt_sha,
                "bytes": receipt_path.stat().st_size,
                "mode": "0444",
            },
        ],
        "parity_execution_still_not_authorized": True,
        "parity_v2_disposition": "INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING",
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {
        "output_dir": str(output_dir),
        "draft": {"path": str(draft_path), "sha256": draft_sha},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "seal": {"path": str(seal_path), "sha256": seal_sha},
        "status": receipt["status"],
    }


def verify_stored_prelaunch(prelaunch_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    prelaunch_dir = prelaunch_dir.resolve()
    paths = tuple(
        prelaunch_dir / name
        for name in ("parity_protocol_draft.json", "receipt.json", "seal.json")
    )
    for path in paths:
        require(
            path.is_file()
            and not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode) == 0o444,
            f"stored parity-v3 file missing/unsafe/mutable: {path.name}",
        )
    draft, receipt, seal = (
        read_json(paths[0], "stored v3 draft"),
        read_json(paths[1], "stored v3 receipt"),
        read_json(paths[2], "stored v3 seal"),
    )
    draft_sha, receipt_sha = sha256_file(paths[0]), sha256_file(paths[1])
    require(draft.get("status") == "NOT_AUTHORIZED_FOR_PARITY_EXECUTION", "stored v3 draft status drift")
    require(
        receipt.get("status") == seal.get("status")
        == "STATIC_PROTOCOL_AUDITED_V3_DATA_ADAPTER_PARITY_EXECUTION_NOT_AUTHORIZED",
        "stored v3 receipt/seal status drift",
    )
    require(
        receipt.get("draft") == {"filename": paths[0].name, "sha256": draft_sha},
        "stored v3 draft/receipt binding drift",
    )
    require(
        seal.get("artifacts")
        == [
            {
                "path": paths[0].name,
                "sha256": draft_sha,
                "bytes": paths[0].stat().st_size,
                "mode": "0444",
            },
            {
                "path": paths[1].name,
                "sha256": receipt_sha,
                "bytes": paths[1].stat().st_size,
                "mode": "0444",
            },
        ],
        "stored v3 seal binding drift",
    )
    live = static_authority(root)
    require(draft.get("authority") == live, "stored v3 authority differs from live static pins")
    return {
        "draft_sha256": draft_sha,
        "receipt_sha256": receipt_sha,
        "seal_sha256": sha256_file(paths[2]),
        "status": receipt["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = (
            static_authority(args.repo_root)
            if args.verify_only
            else write_prelaunch(args.output_dir, args.repo_root)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except StaticParityV3Error as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

