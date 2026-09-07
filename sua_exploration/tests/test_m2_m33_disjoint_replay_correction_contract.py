"""Static fail-closed contract checks for the M2/M33 replay correction scripts."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]


def test_runner_is_cpu_only_attempt_scoped_and_never_reads_legacy_scores() -> None:
    text = (ROOT / "sua_exploration/scripts/run_m2_m33_disjoint_replay_correction_one_arm.sh").read_text()
    assert 'trainer.accelerator=cpu' in text and 'trainer.devices=1' in text
    assert '--attempt' in text and '${GROUP}_${CELL}_${LOG_SUFFIX}_cpu.log' in text
    assert 'native_mua_heldout_t4_v1/aggregate_heldout.json' not in text
    assert 'data.query_start_trial=33' in text and 'data.allow_empty_heldout_query=true' in text


def test_aggregate_requires_an_explicit_immutable_seal_list() -> None:
    text = (ROOT / "sua_exploration/scripts/aggregate_m2_m33_disjoint_replay_correction.py").read_text()
    assert 'parser.add_argument("--seal-list", type=Path, required=True' in text
    assert 'explicit seal SHA' in text
    assert 'glob(' not in text


def test_t4_ts4_normalization_addendum_is_score_free_and_validator_bound() -> None:
    addendum = ROOT / "sua_exploration/results/m2_m33_disjoint_replay_correction_v1/normalization_parity_addendum_v1.json"
    record = json.loads(addendum.read_text())
    assert record["protocol"]["sha256"] == "da4eb05db43401f2e314351e8882c6606cc4d020aed3ed45e860d52db9e59df1"
    assert "does not open metrics CSVs" in record["data_access_disclosure"]
    assert set(record["norms"]) == {"t4", "ts4"}
    validator = (ROOT / "sua_exploration/scripts/write_m2_m33_disjoint_replay_correction_provenance.py").read_text()
    assert 'normalization-parity addendum SHA drift' in validator
    assert 'addendum-bound train-only T4/TS4 normalization' in validator
    assert 'T4/TS4 same-cell parity' in validator


def test_invalid_first_profile_is_explicitly_forbidden() -> None:
    receipt = ROOT / (
        "streaming_calibration_exp/outputs/streaming_calibration/"
        "m2_m33_disjoint_replay_correction_v1_f0_m2_f1_s42_20260801_115446/"
        "INVALID_PROFILE_RECEIPT.json"
    )
    text = receipt.read_text()
    assert 'invalid_unsealed_scores_forbidden' in text
    assert 'log was overwritten by the retry' in text
    assert 'forbidden from every aggregate' in text
