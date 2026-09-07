"""Receipt-only tests for the H-C0 date evaluator and aggregate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pytest

from scripts import h1_carrierid_hc0_fivedate_evaluate as hc0


DATES = ("20260101", "20260102")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_immutable(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _metric(
    date: str,
    arm: str,
    pooled_r2: float,
    per_session_r2: tuple[float, float],
    samples: tuple[int, int],
) -> dict[str, Any]:
    state_sha = _sha(f"state-{date}-{arm}")
    return {
        "pooled_r2": pooled_r2,
        "samples": sum(samples),
        "batches": 2,
        "last_batch_size": samples[-1],
        "r2_accumulator_dtype": "float64",
        "state_immutable": True,
        "state_sha256_before": state_sha,
        "state_sha256_after": state_sha,
        "query_window_indices_sha256": _sha(f"windows-{date}"),
        "per_session": {
            f"ses-{date}-a": {"r2": per_session_r2[0], "samples": samples[0]},
            f"ses-{date}-b": {"r2": per_session_r2[1], "samples": samples[1]},
        },
    }


def _publish_date(
    date: str,
    *,
    h_c: float,
    h_c0: float,
    h_s: float,
    per_session: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    samples: tuple[int, int],
) -> dict[str, Any]:
    h_c_metric = _metric(date, "hc", h_c, per_session[0], samples)
    h_c0_metric = _metric(date, "hc0", h_c0, per_session[1], samples)
    h_s_metric = _metric(date, "hs", h_s, per_session[2], samples)
    reference = {
        "schema": hc0.REFERENCE_SCHEMA,
        "status": f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED",
        "outer_date": date,
        "metrics": {"h_c": h_c_metric, "h_s": h_s_metric},
    }
    reference_path = _write_immutable(hc0._reference_path(date), reference)
    pair = {
        "schema": hc0.PAIR_PREFLIGHT_SCHEMA,
        "status": hc0.PAIR_PREFLIGHT_STATUS,
        "outer_date": date,
    }
    pair_path = _write_immutable(hc0.ARTIFACT_ROOT / "pair_receipts" / date / "pair.json", pair)
    run_dir = hc0._run_dir(date)
    receipt = {
        "schema": hc0.DATE_RECEIPT_SCHEMA,
        "status": hc0.DATE_RECEIPT_STATUS,
        "outer_date": date,
        "checkpoint": {
            "path": str((run_dir / "checkpoints/fixed_epoch50/epoch_049.ckpt").resolve()),
            "sha256": _sha(f"checkpoint-{date}"),
        },
        "config": {
            "path": str((run_dir / ".hydra/config.yaml").resolve()),
            "sha256": _sha(f"config-{date}"),
        },
        "pair_preflight": {"path": str(pair_path.resolve()), "sha256": hc0.sha256_file(pair_path)},
        "reference": {"path": str(reference_path.resolve()), "sha256": hc0.sha256_file(reference_path)},
        "metrics": {
            "h_c0": h_c0_metric,
            "h_c_reference": h_c_metric,
            "h_s_reference": h_s_metric,
            "h_c_minus_h_c0": h_c - h_c0,
            "h_c0_minus_h_s": h_c0 - h_s,
        },
        "scope": {
            "development_outer_date_replay": True,
            "formal_heldout_opened": False,
            "evalai_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
    }
    receipt["canonical_content_sha256"] = hc0._canonical_json_sha(receipt)
    _write_immutable(hc0._output_path(date), receipt)
    return receipt


@pytest.fixture
def receipt_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    monkeypatch.setattr(hc0, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(hc0, "REFERENCE_ROOT", tmp_path / "references")
    monkeypatch.setattr(hc0, "CONFIRMATORY_DATES", DATES)
    return {
        DATES[0]: _publish_date(
            DATES[0], h_c=0.70, h_c0=0.50, h_s=0.40,
            per_session=((0.80, 0.60), (0.50, 0.50), (0.35, 0.45)), samples=(6, 4),
        ),
        DATES[1]: _publish_date(
            DATES[1], h_c=0.35, h_c0=0.40, h_s=0.20,
            per_session=((0.45, 0.25), (0.40, 0.40), (0.15, 0.25)), samples=(18, 12),
        ),
    }


def _rewrite_receipt(date: str, value: dict[str, Any], *, rehash: bool) -> None:
    if rehash:
        value.pop("canonical_content_sha256", None)
        value["canonical_content_sha256"] = hc0._canonical_json_sha(value)
    _write_immutable(hc0._output_path(date), value)


def test_existing_date_receipt_is_validated_without_model_or_data_access(
    receipt_tree: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("receipt reuse must not open a model or dataset")

    monkeypatch.setattr(hc0.torch, "load", forbidden)
    monkeypatch.setattr(hc0.hydra.utils, "instantiate", forbidden)
    monkeypatch.setattr(hc0.target_module, "load_target_dependencies", forbidden)
    monkeypatch.setattr(hc0.target_module, "load_outer_date_target_records", forbidden)
    result = hc0.evaluate_date(DATES[0], device_name="cuda")
    assert result["outer_date"] == DATES[0]


@pytest.mark.parametrize("failure", ("status", "scope", "nonfinite", "reference_path", "sample_grid"))
def test_semantically_invalid_rehashed_receipts_fail_closed(
    receipt_tree: dict[str, dict[str, Any]], failure: str,
) -> None:
    date = DATES[0]
    receipt = json.loads(json.dumps(receipt_tree[date]))
    if failure == "status":
        receipt["status"] = "WRONG"
    elif failure == "scope":
        receipt["scope"]["target_backward_steps"] = 1
    elif failure == "nonfinite":
        receipt["metrics"]["h_c0"]["pooled_r2"] = float("nan")
    elif failure == "reference_path":
        receipt["reference"]["path"] = str(hc0._reference_path(DATES[1]).resolve())
    elif failure == "sample_grid":
        receipt["metrics"]["h_c0"]["per_session"][f"ses-{date}-a"]["samples"] += 1
        receipt["metrics"]["h_c0"]["samples"] += 1
    _rewrite_receipt(date, receipt, rehash=True)
    with pytest.raises(ValueError):
        hc0.evaluate_date(date, device_name="cpu")


def test_canonical_content_tampering_fails_closed(receipt_tree: dict[str, dict[str, Any]]) -> None:
    date = DATES[0]
    receipt = json.loads(json.dumps(receipt_tree[date]))
    receipt["metrics"]["h_c_minus_h_c0"] = 999.0
    _rewrite_receipt(date, receipt, rehash=False)
    with pytest.raises(ValueError, match="canonical SHA drift"):
        hc0.evaluate_date(date, device_name="cpu")


def test_aggregate_validates_rows_and_reports_three_descriptive_weightings(
    receipt_tree: dict[str, dict[str, Any]],
) -> None:
    result = hc0.aggregate()
    output = hc0.ARTIFACT_ROOT / "H1_CARRIERID_HC0_FIVEDATE_AGGREGATE_v1.json"
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert result["mean_date_h_c_minus_h_c0"] == pytest.approx(0.075)
    assert result["sample_count_weighted_mean_date_h_c_minus_h_c0"] == pytest.approx(0.0125)
    assert result["mean_date_h_c0_minus_h_s"] == pytest.approx(0.15)
    assert result["median_date_h_c0_minus_h_s"] == pytest.approx(0.15)
    assert result["positive_dates_h_c0_minus_h_s"] == 2
    assert result["bootstrap_date_95ci_h_c0_minus_h_s"] == pytest.approx([0.1, 0.2])
    assert result["sample_count_weighted_mean_date_h_c0_minus_h_s"] == pytest.approx(0.175)
    recording_deltas = [row["h_c_minus_h_c0"] for row in result["per_recording"]]
    assert result["equal_recording"]["mean_h_c_minus_h_c0"] == pytest.approx(sum(recording_deltas) / 4)
    assert result["equal_recording"]["positive_h_c_minus_h_c0"] == 3
    assert result["equal_recording"]["negative_h_c_minus_h_c0"] == 1
    assert "primary inference unit" in result["statistics_limit"]
    assert "not pooled-prediction or global SSE/TSS R2 contrasts" in result["statistics_limit"]
    assert "sample_weighted_date_pooled_h_c_minus_h_c0" not in result
    assert "sample_weighted_date_pooled_h_c0_minus_h_s" not in result


@pytest.mark.parametrize("dates", ((DATES[1], DATES[0]), (DATES[0], DATES[0])))
def test_aggregate_rejects_wrong_order_or_duplicate_contract_dates(
    receipt_tree: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch, dates: tuple[str, str],
) -> None:
    monkeypatch.setattr(hc0, "CONFIRMATORY_DATES", dates)
    with pytest.raises(ValueError, match="duplicates|chronological order"):
        hc0.aggregate()


def test_aggregate_rejects_tampered_date_receipt(receipt_tree: dict[str, dict[str, Any]]) -> None:
    date = DATES[1]
    receipt = json.loads(json.dumps(receipt_tree[date]))
    receipt["outer_date"] = DATES[0]
    _rewrite_receipt(date, receipt, rehash=True)
    with pytest.raises(ValueError, match="outer_date drift"):
        hc0.aggregate()
