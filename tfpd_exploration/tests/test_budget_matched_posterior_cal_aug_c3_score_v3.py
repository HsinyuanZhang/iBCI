from __future__ import annotations

import os
from pathlib import Path

from budget_matched_posterior_cal_aug_c3_v1 import score as v1
from budget_matched_posterior_cal_aug_c3_v1 import score_v3 as v3


def _canonical_env(repository: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "SUBC_DATA_ROOT": str(repository / v3.SUBC_RELATIVE),
        "SUBM_DATA_ROOT": str(repository / v3.SUBM_RELATIVE),
    }


def test_v3_validates_both_exact_failed_predecessors():
    repository = Path(__file__).resolve().parents[2]
    assert v3.validate_failed_v2(repository)["attempt_sha256"] == v3.FAILED_V2_ATTEMPT_SHA256
    assert v3.validate_failed_v2(repository)["failure_sha256"] == v3.FAILED_V2_FAILURE_SHA256


def test_pre_attempt_runtime_contract_requires_exact_env(monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    for key, value in _canonical_env(repository).items():
        monkeypatch.setenv(key, value)
    result = v3.pre_attempt_runtime_contract(repository)
    assert result["src_origin"] == str((repository / "tfpd_exploration/src/__init__.py").resolve())
    assert result["streaming_encoder_origin"] == str((
        repository / "streaming_calibration_exp/src/models/components/streaming_encoders.py"
    ).resolve())
    monkeypatch.setenv("SUBC_DATA_ROOT", result["subc_data_root"] + "-wrong")
    try:
        v3.pre_attempt_runtime_contract(repository)
    except v1.C3ScoreError:
        pass
    else:
        raise AssertionError("wrong SUBC_DATA_ROOT was accepted")


def test_v3_closure_separates_review_files():
    repository = Path(__file__).resolve().parents[2]
    execution = v3.execution_closure(repository)
    review = v3.review_closure(repository)
    assert set(execution["files"]) == set(v3.EXECUTION_PATHS)
    assert set(review["files"]) == set(v3.REVIEW_PATHS)
    assert not set(execution["files"]) & set(review["files"])


def test_v3_result_root_is_fresh():
    repository = Path(__file__).resolve().parents[2]
    path = repository / v3.RESULT_ROOT_RELATIVE
    assert not path.exists() and not path.is_symlink()

