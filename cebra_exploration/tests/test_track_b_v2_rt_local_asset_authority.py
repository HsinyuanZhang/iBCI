"""No-data tests for the additive RT 15-session local-asset authority bridge."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_rt_local_asset_authority as authority  # noqa: E402


PUBLISHER = REPO_ROOT / "cebra_exploration/scripts/publish_track_b_v2_rt_local_asset_authority.py"


@pytest.fixture(scope="module")
def graph() -> dict:
    return authority.audit_existing_asset_identity_graph()


def _rows(graph: dict) -> list[dict]:
    return [
        {
            "session_id": row["session_id"],
            "canonical_path": row["canonical_path"],
            "byte_size": 10_000_000 + index,
            "sha256": row["sha256"],
        }
        for index, row in enumerate(graph["asset_identities"])
    ]


def test_existing_sealed_graph_is_path_sha_complete_but_has_no_byte_size(graph: dict) -> None:
    assert graph["status"] == authority.AUDIT_STATUS
    assert graph["metric_pointer"]["body_sha256"] == authority.POINTER_SHA256
    assert graph["full15_source_manifest"]["body_sha256"] == authority.MANIFEST_SHA256
    assert graph["full15_source_aggregate"]["body_sha256"] == authority.AGGREGATE_SHA256
    assert graph["asset_count"] == graph["fold_count"] == 15
    assert len(graph["asset_identities"]) == len(graph["fold_bindings"]) == 15
    assert {row["identity_source_fold_occurrence_count"] for row in graph["asset_identities"]} == {14}
    assert {row["identity_variant_count"] for row in graph["asset_identities"]} == {1}
    assert all(row["byte_size"] is None for row in graph["asset_identities"])
    assert graph["path_and_sha_authority_complete"] is True
    assert graph["byte_size_authority_complete"] is False
    assert graph["complete_path_byte_size_sha_authority_exists"] is False
    assert graph["missing_authority_field"] == "byte_size"
    assert graph["data_file_opened"] is graph["data_file_statted"] is graph["data_file_rehashed"] is False


def test_every_fold_held_target_exactly_matches_one_consensus_asset(graph: dict) -> None:
    assets = graph["asset_identities"]
    folds = graph["fold_bindings"]
    assert [row["session_id"] for row in assets] == [row["held_target_session_id"] for row in folds]
    for index, (asset, fold) in enumerate(zip(assets, folds, strict=True)):
        assert asset["held_target_outer_fold_id"] == fold["outer_fold_id"] == f"rt_outer_fold_{index:02d}"
        assert asset["held_target_outer_fold_index"] == fold["outer_fold_index"] == index
        assert asset["session_id"] not in fold["ordered_source_session_ids"]
        assert len(fold["ordered_source_session_ids"]) == 14


def test_explicit_root_rows_build_unminted_authority_without_data_access(graph: dict) -> None:
    payload = authority.build_unminted_authority(
        root_asset_rows=_rows(graph), identity_graph=graph, entrypoint=PUBLISHER,
    )
    assert payload["schema"] == authority.SCHEMA
    assert payload["status"] == authority.STATUS
    assert payload["asset_count"] == 15
    assert payload["parent_bindings"]["rt_metric_pointer_body_sha256"] == authority.POINTER_SHA256
    assert payload["parent_bindings"]["rt_full15_source_manifest_body_sha256"] == authority.MANIFEST_SHA256
    assert payload["parent_bindings"]["rt_full15_source_aggregate_body_sha256"] == authority.AGGREGATE_SHA256
    assert payload["root_input_contract"]["all_15_rows_explicit"] is True
    assert payload["root_input_contract"]["directory_glob_scan_or_available_tree_inference_used"] is False
    assert payload["publisher_data_file_opened"] is payload["publisher_data_file_statted"] is False
    assert payload["target_data_opened"] is payload["formal_data_opened"] is False
    assert set(payload["implementation_closure"]) == {
        "rt_local_asset_authority_core",
        "rt_local_asset_authority_publisher",
        "rt_local_asset_authority_focused_test",
        "track_b_v2_contract",
        "track_b_v2_live_contract",
        "track_b_v2_metric_pointer_authority",
        "track_b_v2_source_adapter",
    }
    assert payload["implementation_closure_sha256"] == hashlib.sha256(
        base.canonical_json_bytes(payload["implementation_closure"])
    ).hexdigest()


def test_unminted_authority_rejects_noncanonical_publisher(graph: dict, tmp_path: Path) -> None:
    substitute = tmp_path / PUBLISHER.name
    substitute.write_bytes(PUBLISHER.read_bytes())
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError, match="canonical publisher"):
        authority.build_unminted_authority(
            root_asset_rows=_rows(graph), identity_graph=graph, entrypoint=substitute,
        )


def test_published_payload_rejects_closure_sha_or_dependency_tamper(graph: dict) -> None:
    payload = authority.build_unminted_authority(
        root_asset_rows=_rows(graph), identity_graph=graph, entrypoint=PUBLISHER,
    )
    authority._validate_published_payload(payload, graph)
    poison = json.loads(json.dumps(payload))
    poison["implementation_closure_sha256"] = "0" * 64
    bare = dict(poison)
    bare.pop("authority_payload_sha256")
    poison["authority_payload_sha256"] = hashlib.sha256(base.canonical_json_bytes(bare)).hexdigest()
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError, match="closure SHA"):
        authority._validate_published_payload(poison, graph)


@pytest.mark.parametrize("poison", ("missing", "path", "sha", "bytes", "order"))
def test_root_input_tamper_fails_closed(graph: dict, poison: str) -> None:
    rows = _rows(graph)
    if poison == "missing":
        rows.pop()
    elif poison == "path":
        rows[0]["canonical_path"] += ".copy"
    elif poison == "sha":
        rows[0]["sha256"] = "0" * 64
    elif poison == "bytes":
        rows[0]["byte_size"] = 0
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError):
        authority.build_unminted_authority(
            root_asset_rows=rows, identity_graph=graph, entrypoint=PUBLISHER,
        )


def test_held_target_fold_or_parent_graph_tamper_fails_closed(graph: dict) -> None:
    for field in ("held_target_session_id", "outer_fold_index"):
        poison = json.loads(json.dumps(graph))
        poison["fold_bindings"][0][field] = "bad" if field == "held_target_session_id" else 1
        bare = dict(poison)
        bare.pop("identity_graph_sha256")
        poison["identity_graph_sha256"] = hashlib.sha256(base.canonical_json_bytes(bare)).hexdigest()
        with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError):
            authority.build_unminted_authority(
                root_asset_rows=_rows(graph), identity_graph=poison, entrypoint=PUBLISHER,
            )


def test_same_fd_asset_handle_hashes_rewinds_and_reuses_one_descriptor(tmp_path: Path) -> None:
    data = b"synthetic-not-RT-data\n" * 17
    asset = tmp_path / "synthetic.bin"
    asset.write_bytes(data)
    authority_payload = {"asset_rows": [{
        "session_id": "ses-RT-synthetic",
        "canonical_path": str(asset),
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }]}
    with authority._verified_same_fd_asset_handle(
            authority=authority_payload, session_id="ses-RT-synthetic") as handle:
        descriptor = handle.fileno()
        assert handle.tell() == 0
        assert handle.read() == data
        assert handle.fileno() == descriptor
    poison = json.loads(json.dumps(authority_payload))
    poison["asset_rows"][0]["sha256"] = "0" * 64
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError, match="SHA drift"):
        with authority._verified_same_fd_asset_handle(
                authority=poison, session_id="ses-RT-synthetic"):
            pass


def test_same_fd_asset_handle_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"abc")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(real)
    payload = {"asset_rows": [{
        "session_id": "ses-RT-synthetic", "canonical_path": str(alias),
        "byte_size": 3, "sha256": hashlib.sha256(b"abc").hexdigest(),
    }]}
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError, match="without following"):
        with authority._verified_same_fd_asset_handle(
                authority=payload, session_id="ses-RT-synthetic"):
            pass


def test_same_fd_asset_handle_rejects_path_replacement_during_parser_use(tmp_path: Path) -> None:
    data = b"verified-original-bytes"
    asset = tmp_path / "asset.bin"
    displaced = tmp_path / "displaced.bin"
    asset.write_bytes(data)
    payload = {"asset_rows": [{
        "session_id": "ses-RT-synthetic", "canonical_path": str(asset),
        "byte_size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }]}
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError, match="pathname changed"):
        with authority._verified_same_fd_asset_handle(
                authority=payload, session_id="ses-RT-synthetic") as handle:
            asset.rename(displaced)
            asset.write_bytes(data)
            assert handle.read() == data


def test_pair_writer_rolls_back_owned_body_if_sidecar_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = tmp_path / "authority.json"
    original = authority._write_exclusive_regular
    calls = 0

    def fail_second(path: Path, raw: bytes, *, mode: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError("synthetic sidecar race")
        return original(path, raw, mode=mode)

    monkeypatch.setattr(authority, "_write_exclusive_regular", fail_second)
    with pytest.raises(authority.TrackBV2RTLocalAssetAuthorityError, match="rolled back"):
        authority._write_immutable_pair(body, {"schema": "synthetic"})
    assert not os.path.lexists(body)
    assert not os.path.lexists(body.with_name(f"{body.name}.sha256"))


def test_implementation_contains_no_directory_discovery_surface() -> None:
    text = Path(authority.__file__).read_text()
    for forbidden in (".glob(", ".rglob(", ".iterdir(", "os.listdir(", "os.scandir("):
        assert forbidden not in text
    audit_source = inspect.getsource(authority.audit_existing_asset_identity_graph)
    assert "SOURCE_ROOT / \"folds\" / fold_id" in audit_source
    assert "source_coverage.json" in audit_source
    publish_source = inspect.getsource(authority.build_unminted_authority)
    assert "os.stat" not in publish_source and ".resolve(" not in publish_source
    canonical_consumer = inspect.getsource(authority.open_canonical_verified_asset_after_authority)
    assert canonical_consumer.index("load_published_authority_before_target_access") < canonical_consumer.index(
        "_verified_same_fd_asset_handle"
    )


def test_publisher_default_is_no_receipt_read_no_data_no_write() -> None:
    assert not os.path.lexists(authority.CANONICAL_OUTPUT)
    assert not os.path.lexists(authority.CANONICAL_OUTPUT.with_name(
        f"{authority.CANONICAL_OUTPUT.name}.sha256"))
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, str(PUBLISHER)], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_PLAN_ONLY__NO_RECEIPT_READ__NO_DATA__NO_WRITE"
    assert payload["required_explicit_asset_row_count"] == 15
    assert payload["data_file_opened"] is payload["data_file_statted"] is False
    assert payload["receipt_minted"] is False
    assert not os.path.lexists(authority.CANONICAL_OUTPUT)


@pytest.mark.parametrize("flag", ("--execute", "--i-am-root-publisher"))
def test_publisher_requires_both_root_execution_flags(flag: str) -> None:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, str(PUBLISHER), flag], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    assert "requires both" in result.stderr
    assert not os.path.lexists(authority.CANONICAL_OUTPUT)


def test_publisher_rejects_alternate_output_without_minting(tmp_path: Path) -> None:
    alternate = tmp_path / "substitute.json"
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, str(PUBLISHER), "--output", str(alternate)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    assert "alternate RT local asset authority output is forbidden" in result.stderr
    assert not alternate.exists()
