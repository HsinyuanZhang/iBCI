"""No-target tests for Track-B v2 root metric-only pointer authority."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_metric_pointer_authority as pointer  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redirect_canonical_pairs_to_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise mint paths without ever touching the real canonical locations."""
    pointer_root = tmp_path / "canonical_root"
    pointer_paths = {
        ("subject_m", "sua"): pointer_root / "subject_m_sua_metric_pointer.json",
        ("subject_m", "pseudo_mua"): pointer_root / "subject_m_pseudo_mua_metric_pointer.json",
        ("rt", None): pointer_root / "rt_metric_pointer.json",
    }
    attestation_paths = {
        ("subject_m", "sua"): pointer_root / "subject_m_sua_root_audit_attestation.json",
        ("subject_m", "pseudo_mua"): pointer_root / "subject_m_pseudo_mua_root_audit_attestation.json",
        ("rt", None): pointer_root / "rt_root_audit_attestation.json",
    }
    monkeypatch.setattr(pointer, "_CANONICAL_POINTER_BODY_PATHS", pointer_paths)
    monkeypatch.setattr(pointer, "_CANONICAL_AUDIT_ATTESTATION_BODY_PATHS", attestation_paths)


def _publish_test_attestation(
    *, dry: dict[str, object], root_authorized_publish: bool = True,
) -> pointer.live.ExplicitSealedReceiptPair:
    return pointer.publish_root_metric_pointer_audit_attestation_pair(
        dry_plan=dry, root_authorized_publish=root_authorized_publish,
    )


@pytest.mark.parametrize(
    ("dataset", "view", "metric_name", "metric_value", "lineage_names"),
    (
        ("subject_m", "sua", "t4_m50_sua_mean_r2", 0.35682823575205275, {"per_session_seed_body"}),
        ("subject_m", "pseudo_mua", "t4_m50_pseudo_mua_mean_r2", 0.3060729397667779, {"per_session_seed_body"}),
        ("rt", None, "t4d_mean_r2", 0.44817638439717034, {"per_fold_t4d_body", "stage2_delta_companion"}),
    ),
)
def test_exact_three_scope_dry_plans_resolve_metrics_from_same_fd_legacy_bytes(
    dataset: str, view: str | None, metric_name: str, metric_value: float, lineage_names: set[str],
) -> None:
    dry = pointer.build_metric_pointer_dry_plan(dataset, view)
    assert dry["status"].startswith("DRY_PLAN_ONLY")
    assert dry["official_receipt_minted"] is False
    template = dry["pointer_payload_template"]
    assert template["metric_values_resolved_from_same_fd_bytes"] == {metric_name: metric_value}
    assert set(template["lineage_companions"]) == lineage_names
    assert all(item["metric_authority"] is False for item in template["lineage_companions"].values())
    assert all(item["verified_from_same_fd_bytes"] is True for item in template["lineage_companions"].values())
    if dataset == "subject_m":
        lineage = template["lineage_companions"]["per_session_seed_body"]
        assert lineage["matching_cell_count_from_same_fd_bytes"] == 45
        assert lineage["unique_session_count"] == 15
        assert lineage["seed_set"] == [42, 43, 44]
        assert lineage["arithmetic_mean_r2_from_same_fd_bytes"] == metric_value
    else:
        lineage = template["lineage_companions"]["per_fold_t4d_body"]
        assert len(lineage["resolved_scalars_from_same_fd_bytes"]) == 15
        assert lineage["folds"] == list(range(15))
        assert lineage["unique_session_count"] == 15
        assert lineage["arithmetic_mean_t4d_r2_from_same_fd_bytes"] == metric_value
        assert "not_absolute_metric" in template["lineage_companions"]["stage2_delta_companion"]["authority_scope"]


def test_subject_m_and_rt_lineage_validators_reject_duplicate_missing_or_wrong_fold_rows() -> None:
    subject_rows = [
        {
            "arm": "shared_t4", "view": "sua", "session_id": f"sub-M_ses-CO-{session:08d}",
            "seed": seed, "r2": 0.5, "query_window_count": 10, "asset_id": f"asset-{session}",
        }
        for session in range(15) for seed in (42, 43, 44)
    ]
    subject_rows[-1] = dict(subject_rows[-1], seed=42)
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="duplicate session×seed"):
        pointer._validate_subject_m_lineage_rows(subject_rows, view="sua")
    rt_rows = [{"fold": fold, "session": f"ses-RT-{fold:08d}", "t4d_r2": 0.4} for fold in range(15)]
    rt_rows[-1] = dict(rt_rows[-1], fold=13)
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="exactly 0..14"):
        pointer._validate_rt_per_fold_rows(rt_rows)


def test_rt_outer_fold_topology_is_derived_from_verified_lineage_without_exposing_fold_scores() -> None:
    topology = pointer.build_rt_outer_fold_lineage_plan()
    assert topology["status"].startswith("RT_15_OUTER_FOLD_TOPOLOGY_ONLY")
    assert topology["outer_fold_count"] == 15
    assert topology["reference_per_fold_lineage_metric_authority"] is False
    assert topology["reference_per_fold_lineage_scalar_values_exposed"] is False
    held_ids = set()
    for fold_index, fold in enumerate(topology["outer_folds"]):
        assert fold["outer_fold_index"] == fold_index
        assert fold["outer_fold_id"] == f"rt_outer_fold_{fold_index:02d}"
        assert len(fold["source_session_ids"]) == 14
        assert fold["opaque_held_out_target_session_id"] not in fold["source_session_ids"]
        assert fold["held_out_target_data_discovered"] is False
        assert fold["held_out_target_data_opened"] is False
        assert fold["held_out_target_passed_to_source_loader"] is False
        held_ids.add(fold["opaque_held_out_target_session_id"])
    assert len(held_ids) == 15
    assert topology["target_data_opened"] is False
    assert topology["cebra_imported"] is False
    assert topology["score_emitted"] is False

    tampered = dict(topology)
    tampered["outer_folds"] = list(topology["outer_folds"])
    tampered["outer_folds"][0] = dict(tampered["outer_folds"][0])
    tampered["outer_folds"][0]["opaque_held_out_target_session_id"] = "ses-RT-19990101"
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="differs from same-FD verified topology"):
        pointer.validate_rt_outer_fold_lineage_plan(tampered)


def test_root_authorized_canonical_attestation_then_pointer_publish_validation_and_oexcl_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_canonical_pairs_to_tmp(monkeypatch, tmp_path)
    dry = pointer.build_metric_pointer_dry_plan("subject_m", "sua")
    attestation_body = pointer.canonical_root_metric_pointer_audit_attestation_body_path("subject_m", "sua")
    output = pointer.canonical_metric_pointer_body_path("subject_m", "sua")
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="root_authorized_publish"):
        _publish_test_attestation(dry=dry, root_authorized_publish=False)
    attestation = _publish_test_attestation(dry=dry)
    assert stat.S_IMODE(attestation_body.stat().st_mode) == 0o444
    assert stat.S_IMODE(attestation.sidecar_path.stat().st_mode) == 0o444
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="root_authorized_publish"):
        pointer.publish_root_audited_metric_pointer_pair(
            dry_plan=dry, root_audit_attestation_pair=attestation,
        )
    pair = pointer.publish_root_audited_metric_pointer_pair(
        dry_plan=dry, root_audit_attestation_pair=attestation,
        root_authorized_publish=True,
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert stat.S_IMODE(pair.sidecar_path.stat().st_mode) == 0o444
    validation = pointer.validate_root_audited_metric_pointer_pair(
        dataset="subject_m", view="sua", pointer_pair=pair,
    )
    assert validation["reference_metrics"] == {"t4_m50_sua_mean_r2": 0.35682823575205275}
    assert validation["lineage_companions_are_not_metric_authority"] is True
    assert validation["root_metric_pointer_audit_attestation"]["body_sha256"] == _sha(
        attestation_body.read_text(encoding="utf-8")
    )
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="already exists"):
        pointer.publish_root_audited_metric_pointer_pair(
            dry_plan=dry, root_audit_attestation_pair=attestation,
            root_authorized_publish=True,
        )


def test_validator_rejects_missing_attestation_arbitrary_sha_alias_and_sidecar_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_canonical_pairs_to_tmp(monkeypatch, tmp_path)
    dry = pointer.build_metric_pointer_dry_plan("rt")
    missing = pointer.live.ExplicitSealedReceiptPair(
        "root_metric_pointer_audit_attestation",
        pointer.canonical_root_metric_pointer_audit_attestation_body_path("rt"),
        pointer.canonical_root_metric_pointer_audit_attestation_body_path("rt").with_name(
            "rt_root_audit_attestation.json.sha256"
        ),
    )
    with pytest.raises(pointer.live.TrackBV2LiveContractError):
        pointer.render_root_audited_metric_pointer_payload(dry_plan=dry, root_audit_attestation_pair=missing)
    with pytest.raises(TypeError):
        pointer.render_root_audited_metric_pointer_payload(  # type: ignore[call-arg]
            dry_plan=dry, root_audit_attestation_sha256=_sha("arbitrary-hex-is-not-an-attestation"),
        )
    attestation = _publish_test_attestation(dry=dry)
    payload = pointer.render_root_audited_metric_pointer_payload(
        dry_plan=dry, root_audit_attestation_pair=attestation,
    )
    # A self-consistent immutable pair with a wrong metric pointer still must
    # fail against freshly verified legacy bytes/dry template.
    drift = dict(payload)
    drift["reference_metric_json_pointers"] = {"t4d_mean_r2": "/results/arms/t4d_reference/median"}
    drift_path = pointer.canonical_metric_pointer_body_path("rt")
    import track_b_v2_contract as base
    base.write_immutable_receipt(drift_path, drift)
    drift_pair = pointer.live.ExplicitSealedReceiptPair(
        "canonical_reference_body_pointer", drift_path, drift_path.with_name(f"{drift_path.name}.sha256"),
    )
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="dry plan/attestation"):
        pointer.validate_root_audited_metric_pointer_pair(dataset="rt", view=None, pointer_pair=drift_pair)

    alias_pair = pointer.live.ExplicitSealedReceiptPair(
        "canonical_reference_body_pointer", tmp_path / "alias-copy.json", tmp_path / "alias-copy.json.sha256",
    )
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="one canonical output path"):
        pointer.validate_root_audited_metric_pointer_pair(dataset="rt", view=None, pointer_pair=alias_pair)
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="canonical path"):
        pointer.publish_root_audited_metric_pointer_pair(
            dry_plan=dry, root_audit_attestation_pair=attestation,
            output_body_path=tmp_path / "alias-publish.json", root_authorized_publish=True,
        )

    # Re-route a second scope to check strict sidecar parity without changing
    # or deleting the immutable drift pair.
    clean_dry = pointer.build_metric_pointer_dry_plan("subject_m", "pseudo_mua")
    clean_attestation = _publish_test_attestation(dry=clean_dry)
    clean_pair = pointer.publish_root_audited_metric_pointer_pair(
        dry_plan=clean_dry, root_audit_attestation_pair=clean_attestation, root_authorized_publish=True,
    )
    clean_pair.sidecar_path.chmod(0o644)
    clean_pair.sidecar_path.write_text("bad sidecar\n", encoding="ascii")
    clean_pair.sidecar_path.chmod(0o444)
    with pytest.raises(pointer.live.TrackBV2LiveContractError, match="sidecar/body drift"):
        pointer.validate_root_audited_metric_pointer_pair(
            dataset="subject_m", view="pseudo_mua", pointer_pair=clean_pair,
        )


def test_canonical_pointer_publisher_rejects_a_symlink_at_its_only_allowed_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_canonical_pairs_to_tmp(monkeypatch, tmp_path)
    dry = pointer.build_metric_pointer_dry_plan("subject_m", "sua")
    attestation = _publish_test_attestation(dry=dry)
    body = pointer.canonical_metric_pointer_body_path("subject_m", "sua")
    body.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.json"
    outside.write_text("not a pointer", encoding="utf-8")
    body.symlink_to(outside)
    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="already exists or is symlinked"):
        pointer.publish_root_audited_metric_pointer_pair(
            dry_plan=dry, root_audit_attestation_pair=attestation, root_authorized_publish=True,
        )


def test_dry_plan_cli_has_no_publish_target_data_or_gpu_switches() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_metric_pointer_dry_plan.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--publish" not in result.stdout
    assert "--target" not in result.stdout
    assert "--gpu" not in result.stdout
    rejected = subprocess.run(
        [sys.executable, str(script), "--dataset", "falcon_h1"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True,
    )
    assert rejected.returncode != 0
    assert "invalid choice" in rejected.stderr


def test_root_attestation_dry_plan_cli_has_no_publish_target_data_or_gpu_switches() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_root_metric_pointer_audit_attestation_dry_plan.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--publish" not in result.stdout
    assert "--target" not in result.stdout
    assert "--gpu" not in result.stdout
    rendered = subprocess.run(
        [sys.executable, str(script), "--dataset", "rt"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert '"schema":"track_b_v2_root_metric_pointer_audit_attestation_dry_plan_v2"' in rendered.stdout
    assert '"official_receipt_minted":false' in rendered.stdout
