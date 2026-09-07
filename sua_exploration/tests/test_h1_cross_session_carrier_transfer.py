from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
SCRIPT = SPINT / "scripts/h1_cross_session_carrier_transfer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("h1_cross_session_carrier_transfer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_record(session_name: str, num_neurons: int) -> object:
    from src.data.h1_m4_eb_pilot import H1PilotRecord

    neural = np.ones((100, num_neurons), dtype=np.float32)
    velocity = np.zeros((100, 7), dtype=np.float32)
    return H1PilotRecord(
        session_name=session_name,
        date="19250101",
        path=Path(f"/tmp/{session_name}.nwb"),
        input_sha256="deadbeef",
        neural=neural,
        velocity=velocity,
        trial_change=np.zeros(100, dtype=bool),
        eval_mask=np.ones(100, dtype=bool),
        trial_num=np.ones(100, dtype=np.float64),
        trial_values=(1.0, 2.0, 3.0, 4.0, 5.0),
        trials=(),
    )


@pytest.fixture(scope="module")
def module():
    if str(SPINT) not in sys.path:
        sys.path.insert(0, str(SPINT))
    return _load_module()


def test_transfer_permutation_has_no_fixed_points(module) -> None:
    recordings = ("ses-19250101T111740", "ses-19250101T112404")
    mapping = module.cross_session_transfer_map(recordings)
    assert set(mapping.keys()) == set(recordings)
    assert set(mapping.values()) == set(recordings)
    assert all(name != donor for name, donor in mapping.items())


def test_swapped_carrier_matches_donor_full_carrier(module) -> None:
    if str(SPINT) not in sys.path:
        sys.path.insert(0, str(SPINT))
    from src.data.h1_context_event_carrier import H1ContextEventDataModule, build_context_target_dataset

    snapshot_receipt = SPINT / "pilot_artifacts/h1_context_event_carrier/source_snapshot/H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json"
    data_dir = SPINT / "data/000954"
    if not snapshot_receipt.exists() or not data_dir.exists():
        pytest.skip("context snapshot or data unavailable")
    source = H1ContextEventDataModule(
        task="h1",
        data_dir=str(data_dir),
        cache_dir=str(SPINT / "pilot_artifacts/h1_context_event_carrier/shared_source_cache"),
        source_snapshot_receipt=str(snapshot_receipt),
    )
    source.setup("fit")
    target = build_context_target_dataset(data_dir=data_dir, source_module=source)
    transfer_map = module.cross_session_transfer_map(target.records.keys())
    swapped = module.build_cross_session_carriers(target, transfer_map)
    for name, donor in transfer_map.items():
        donor_full = np.asarray(target.support[donor].carriers["full"], np.float64)
        assert np.array_equal(swapped[name], donor_full)


def test_channel_count_mismatch_fails_closed(module) -> None:
    records = {
        "ses-a": _synthetic_record("ses-a", 176),
        "ses-b": _synthetic_record("ses-b", 175),
    }
    with pytest.raises(ValueError, match="got 175"):
        module.verify_same_array_channel_contract(records)


def test_pooled_r2_aggregation_matches_hand_computation(module) -> None:
    truth = {
        "ses-a": np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
        "ses-b": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
    }
    estimate = {
        "ses-a": np.array([[1.1, 1.9], [2.2, 2.8]], dtype=np.float64),
        "ses-b": np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float64),
    }
    pooled = module.aggregate_pooled_r2(truth, estimate)
    direct_truth = np.concatenate([truth["ses-a"], truth["ses-b"]])
    direct_estimate = np.concatenate([estimate["ses-a"], estimate["ses-b"]])
    assert pooled == module.pooled_r2(direct_truth, direct_estimate)


def test_receipt_refuses_overwrite(module, tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    payload = {"schema": module.SCHEMA, "status": "test"}
    path, digest = module.write_immutable_json(output, payload)
    assert path.exists()
    assert digest
    with pytest.raises(FileExistsError):
        module.write_immutable_json(output, payload)
