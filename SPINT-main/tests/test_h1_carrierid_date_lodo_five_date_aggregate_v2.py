"""Focused no-NWB contracts for the receipt-only H1 five-date v2 closure."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat

import pytest

from scripts import h1_carrierid_date_lodo_five_date_aggregate as v1
from scripts import h1_carrierid_date_lodo_five_date_aggregate_v2 as v2


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_immutable(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _receipt(date: str, *, hs: float, hc: float) -> dict[str, object]:
    sessions = list(v2.EXPECTED_RECORDINGS_BY_DATE[date])
    samples = [7 + index for index in range(len(sessions))]
    total = sum(samples)
    window_hash = _sha(f"windows-{date}")
    files = {session: _sha(f"file-{session}") for session in sessions}

    def checkpoint(arm: str) -> dict[str, object]:
        config = _sha(f"config-{date}-{arm}")
        return {
            "path": f"/receipt-only/{date}-{arm}.ckpt", "sha256": _sha(f"checkpoint-{date}-{arm}"),
            "config_path": f"/receipt-only/{date}-{arm}.yaml", "config_sha256": config,
            "metadata": {
                "arm": arm, "outer_date": date, "config_sha256": config,
                "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
                "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
                "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
                "initial_state_sha256": _sha(f"initial-{date}-{arm}"),
                "phase2_source_binding_sha256": _sha(f"binding-{date}"),
                "phase1_source_manifest_sha256": _sha(f"source-{date}"),
                "phase1_preflight_sha256": _sha(f"preflight-{date}"),
            },
        }

    def metric(value: float) -> dict[str, object]:
        return {
            "pooled_r2": value, "samples": total, "batches": 1, "last_batch_size": total,
            "r2_accumulator_dtype": "float64", "state_immutable": True,
            "state_sha256_before": _sha(f"state-{date}"), "state_sha256_after": _sha(f"state-{date}"),
            "query_window_indices_sha256": window_hash,
            "per_session": {
                session: {"samples": samples[index], "r2": value + (index - 1) * 0.01}
                for index, session in enumerate(sessions)
            },
        }

    return {
        "schema": v1.EVALUATION_SCHEMA, "status": v1._expected_status(date), "outer_date": date,
        "checkpoint_binding_completed_before_target_open": True, "source_manifest_sha256": _sha(f"source-{date}"),
        "checkpoints": {"H-S": checkpoint("H-S"), "H-C": checkpoint("H-C")},
        "target": {
            "sessions": sessions, "files": files,
            "strict_dataset": {"outer_date": date, "sessions": sessions, "window_indices_sha256": window_hash,
                               "samples": total, "all_query_histories_start_at_or_after_fifth_trial": True},
            "all_query_histories_start_at_or_after_fifth_trial": True,
        },
        "metrics": {"h_s": metric(hs), "h_c": metric(hc), "h_c_minus_h_s": hc - hs},
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": "__REPLACED_BY_WRITER__", "same_date_prior_terminal_evaluation_receipts": 0},
        "scope": {"formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }


def _directory(tmp_path: Path, *, values: dict[str, tuple[float, float]] | None = None) -> Path:
    directory = tmp_path / "terminal_evaluations"
    for date in v2.DATES:
        hs, hc = (values or {}).get(date, (0.30, 0.35))
        body = _receipt(date, hs=hs, hc=hc)
        output = v1.canonical_evaluation_path(date, evaluation_dir=directory)
        body["one_shot"]["canonical_output_path"] = str(output)
        _write_immutable(output, body)
    return directory


def test_v2_reports_full_recording_date_and_paired_date_level_statistics(tmp_path: Path) -> None:
    values = {
        "19250108": (0.30, 0.40), "19250113": (0.45, 0.42), "19250115": (0.20, 0.20),
        "19250119": (0.10, 0.17), "19250120": (0.50, 0.55),
    }
    directory = _directory(tmp_path, values=values)
    output = tmp_path / "aggregate-v2.json"
    result = v2.aggregate(evaluation_dir=directory, output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == v2.STATUS and stat.S_IMODE(output.stat().st_mode) == 0o444
    assert body["all_five_date_receipts_present_and_validated"] is True
    assert body["all_eleven_recording_pairs_present_and_validated"] is True
    assert len(body["per_recording"]) == 11 and len(body["per_date"]) == 5
    paired = body["overall_paired_h_c_minus_h_s"]
    assert paired["mean"] == pytest.approx(0.038)
    assert paired["median"] == pytest.approx(0.05)
    assert paired["sign_counts"] == {"positive": 3, "negative": 1, "zero": 1, "total": 5}
    assert paired["paired_standard_error"] > 0.0
    assert paired["paired_outer_date_bootstrap_95"]["resampling_unit"].startswith("outer_date")
    assert paired["paired_outer_date_bootstrap_95"]["rng_seed"] == v2.BOOTSTRAP_SEED

    output_two = tmp_path / "aggregate-v2-two.json"
    v2.aggregate(evaluation_dir=directory, output=output_two)
    assert json.loads(output_two.read_text())["overall_paired_h_c_minus_h_s"]["paired_outer_date_bootstrap_95"] == paired["paired_outer_date_bootstrap_95"]


def test_v2_fails_closed_on_missing_date_or_jointly_omitted_recording(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    missing = v1.canonical_evaluation_path("19250119", evaluation_dir=directory)
    missing.unlink()
    output = tmp_path / "missing-date.json"
    with pytest.raises(v2.FiveDateDescriptiveAggregateError, match="19250119: immutable held-out receipt is missing"):
        v2.aggregate(evaluation_dir=directory, output=output)
    assert not output.exists()

    directory = _directory(tmp_path / "recording")
    date, session = "19250108", v2.EXPECTED_RECORDINGS_BY_DATE["19250108"][-1]
    path = v1.canonical_evaluation_path(date, evaluation_dir=directory)
    path.chmod(0o644)
    body = json.loads(path.read_text())
    body["target"]["sessions"].remove(session)
    body["target"]["strict_dataset"]["sessions"].remove(session)
    body["target"]["files"].pop(session)
    remaining = sum(item["samples"] for item in body["metrics"]["h_s"]["per_session"].values() if item is not body["metrics"]["h_s"]["per_session"][session])
    for arm in ("h_s", "h_c"):
        body["metrics"][arm]["per_session"].pop(session)
        body["metrics"][arm]["samples"] = remaining
    body["target"]["strict_dataset"]["samples"] = remaining
    path.write_text(json.dumps(body), encoding="utf-8"); path.chmod(0o444)
    with pytest.raises(v2.FiveDateDescriptiveAggregateError, match="target recording grid is incomplete"):
        v2.aggregate(evaluation_dir=directory, output=tmp_path / "missing-recording.json")


def test_v2_accepts_only_a_complete_explicit_five_receipt_grid_for_offline_imports(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    paths = {date: v1.canonical_evaluation_path(date, evaluation_dir=directory) for date in v2.DATES}
    # An imported local receipt may retain its remote canonical output path.
    first = paths["19250108"]
    first.chmod(0o644)
    body = json.loads(first.read_text())
    body["one_shot"]["canonical_output_path"] = "/remote/stage/terminal_evaluations/" + first.name
    first.write_text(json.dumps(body), encoding="utf-8"); first.chmod(0o444)
    output = tmp_path / "explicit-grid.json"
    v2.aggregate(output=output, receipt_paths=paths)
    rendered = json.loads(output.read_text())
    row = next(item for item in rendered["per_date"] if item["outer_date"] == "19250108")
    assert row["input_receipt"]["path"] == str(first.resolve())
    assert row["input_receipt"]["declared_canonical_output_path"].startswith("/remote/stage/")

    with pytest.raises(v2.FiveDateDescriptiveAggregateError, match="exactly the five frozen dates"):
        v2.aggregate(output=tmp_path / "partial-explicit-grid.json",
                     receipt_paths={date: path for date, path in paths.items() if date != "19250120"})


def test_v2_fails_closed_when_hs_and_hc_do_not_pair_the_same_recording_samples(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    date = "19250113"
    path = v1.canonical_evaluation_path(date, evaluation_dir=directory)
    path.chmod(0o644)
    body = json.loads(path.read_text())
    first, second = v2.EXPECTED_RECORDINGS_BY_DATE[date]
    # Preserve every v1 total/date contract while breaking the recording-level pairing.
    body["metrics"]["h_c"]["per_session"][first]["samples"] = 6
    body["metrics"]["h_c"]["per_session"][second]["samples"] = 9
    path.write_text(json.dumps(body), encoding="utf-8"); path.chmod(0o444)
    with pytest.raises(v2.FiveDateDescriptiveAggregateError, match="per-recording sample count mismatch"):
        v2.aggregate(evaluation_dir=directory, output=tmp_path / "sample-mismatch.json")


def test_v2_remains_offline_and_cannot_overwrite_an_aggregate(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    output = tmp_path / "aggregate.json"
    v2.aggregate(evaluation_dir=directory, output=output)
    with pytest.raises(v2.FiveDateDescriptiveAggregateError, match="refusing to overwrite"):
        v2.aggregate(evaluation_dir=directory, output=output)
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "src.data", "torch.load", "NWBHDF5IO", "Trainer", "subprocess"):
        assert forbidden not in source
    assert "bootstrap and paired SE use outer date" in source
