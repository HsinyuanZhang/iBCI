"""Synthetic/no-NWB contracts for five-date receipt-only H1 LODO aggregation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from scripts import h1_carrierid_date_lodo_five_date_aggregate as aggregate


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_immutable(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _receipt(date: str, *, hs: float = 0.35, hc: float = 0.40) -> dict[str, object]:
    sessions = [f"ses-{date}-a", f"ses-{date}-b"]
    window_hash = _sha(f"windows-{date}")
    files = {name: _sha(f"file-{name}") for name in sessions}
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
            "pooled_r2": value, "samples": 10, "batches": 1, "last_batch_size": 10,
            "r2_accumulator_dtype": "float64", "state_immutable": True,
            "state_sha256_before": _sha(f"state-{date}"), "state_sha256_after": _sha(f"state-{date}"),
            "query_window_indices_sha256": window_hash,
            "per_session": {
                sessions[0]: {"samples": 6, "r2": value - 0.02},
                sessions[1]: {"samples": 4, "r2": value + 0.02},
            },
        }
    return {
        "schema": aggregate.EVALUATION_SCHEMA, "status": aggregate._expected_status(date), "outer_date": date,
        "checkpoint_binding_completed_before_target_open": True,
        "source_manifest_sha256": _sha(f"source-{date}"),
        "checkpoints": {"H-S": checkpoint("H-S"), "H-C": checkpoint("H-C")},
        "target": {
            "sessions": sessions, "files": files,
            "strict_dataset": {"outer_date": date, "sessions": sessions, "window_indices_sha256": window_hash,
                               "samples": 10, "all_query_histories_start_at_or_after_fifth_trial": True},
            "all_query_histories_start_at_or_after_fifth_trial": True,
        },
        "metrics": {"h_s": metric(hs), "h_c": metric(hc), "h_c_minus_h_s": hc - hs},
        "deployment_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "one_shot": {"canonical_output_path": "__REPLACED_BY_WRITER__", "same_date_prior_terminal_evaluation_receipts": 0},
        "scope": {"formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }


def _evaluation_dir(tmp_path: Path, *, values: dict[str, tuple[float, float]] | None = None, missing: str | None = None) -> Path:
    directory = tmp_path / "terminal_evaluations"
    for date in aggregate.DATES:
        if date == missing:
            continue
        hs, hc = (values or {}).get(date, (0.35, 0.40))
        body = _receipt(date, hs=hs, hc=hc)
        path = aggregate.canonical_evaluation_path(date, evaluation_dir=directory)
        body["one_shot"]["canonical_output_path"] = str(path)
        _write_immutable(path, body)
    return directory


def test_missing_one_of_five_immutable_receipts_fails_closed_without_output(tmp_path: Path):
    directory = _evaluation_dir(tmp_path, missing="19250119")
    output = tmp_path / "aggregate.json"
    with pytest.raises(aggregate.FiveDateAggregateError, match="19250119: canonical terminal receipt is missing"):
        aggregate.aggregate(evaluation_dir=directory, output=output)
    assert not output.exists()


def test_aggregate_requires_all_five_but_reports_negative_date_without_stopping_the_chain(tmp_path: Path):
    values = {"19250108": (0.30, 0.40), "19250113": (0.45, 0.42), "19250115": (0.20, 0.20),
              "19250119": (0.10, 0.17), "19250120": (0.50, 0.55)}
    directory = _evaluation_dir(tmp_path, values=values)
    output = tmp_path / "aggregate.json"
    result = aggregate.aggregate(evaluation_dir=directory, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == aggregate.AGGREGATE_STATUS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert payload["all_five_date_receipts_present_and_validated"] is True
    assert payload["summary"]["h_c_minus_h_s_sign_pattern"]["by_date"] == {
        "19250108": "positive", "19250113": "negative", "19250115": "zero",
        "19250119": "positive", "19250120": "positive",
    }
    assert payload["summary"]["h_c_minus_h_s_sign_pattern"]["positive_date_count"] == 3
    assert payload["route_prerequisite"]["prior_status"] == "MISSING_IMPLEMENTATION_FAIL_CLOSED"
    assert payload["route_prerequisite"]["status"] == "source/date screen complete"
    assert payload["route_prerequisite"]["automatic_route_selection"] == "FORBIDDEN"
    assert payload["route_prerequisite"]["EST4"].startswith("NOT_SELECTED")
    assert payload["route_prerequisite"]["CI64"].startswith("NOT_SELECTED")


@pytest.mark.parametrize("mutate, needle", [
    (lambda value: value["deployment_updates"].update({"optimizer_steps": 1}), "nonzero deployment target updates"),
    (lambda value: value["checkpoints"]["H-C"]["metadata"].update({"target_backward_steps": 1}), "nonzero target optimizer/backward"),
    (lambda value: value["metrics"]["h_s"].update({"state_sha256_after": "0" * 64}), "model state changed"),
    (lambda value: value.update({"source_manifest_sha256": _sha("wrong-source")}), "source-manifest binding mismatch"),
    (lambda value: value.update({"schema": "wrong"}), "schema/status drift"),
])
def test_receipt_contract_failures_prevent_publication(tmp_path: Path, mutate, needle: str):
    directory = _evaluation_dir(tmp_path)
    date = "19250115"
    path = aggregate.canonical_evaluation_path(date, evaluation_dir=directory)
    path.chmod(0o644)
    value = json.loads(path.read_text(encoding="utf-8")); mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8"); path.chmod(0o444)
    output = tmp_path / "aggregate.json"
    with pytest.raises(aggregate.FiveDateAggregateError, match=needle):
        aggregate.aggregate(evaluation_dir=directory, output=output)
    assert not output.exists()


def test_receipts_are_canonical_and_aggregate_output_cannot_be_overwritten(tmp_path: Path):
    directory = _evaluation_dir(tmp_path)
    output = tmp_path / "aggregate.json"
    aggregate.aggregate(evaluation_dir=directory, output=output)
    with pytest.raises(aggregate.FiveDateAggregateError, match="refusing to overwrite"):
        aggregate.aggregate(evaluation_dir=directory, output=output)


def test_aggregator_is_receipt_only_without_recording_checkpoint_trainer_or_gpu_imports():
    source = Path(aggregate.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "src.data", "torch.load", "load_nwb", "NWBHDF5IO", "Trainer", "subprocess"):
        assert forbidden not in source
    assert "rglob(" not in source
    assert '"nwb_opened_by_aggregator": False' in source
    assert '"gpu_constructed_or_launched": False' in source
