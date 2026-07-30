from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import aggregate_t4_encoder_int8 as aggregate


SESSIONS = [f"val-{index}" for index in range(6)]


def _source(delta_b0: float = 0.06, delta_ts4: float = 0.05) -> dict:
    return {
        "formal_test_files_opened": False,
        "protocol": {
            "same_trial_count_and_prefix_for_all_arms": True,
            "evaluation_backward_gradients": False,
            "scored_epoch_window": list(range(5, 13)),
        },
        "contrasts": {
            "t4_vs_original_spint_b0": {
                "mean_paired_delta_r2": delta_b0,
            },
            "t4_vs_shuffled_label_ts4": {
                "mean_paired_delta_r2": delta_ts4,
            },
        },
    }


def _report(*, method: str, delta: float) -> dict:
    quant_key = "int8_encoder" if method == "ptq" else "qat_int8_encoder"
    payload = {
        "decoder_quantized_in_this_run": False,
        "protocol": {
            "formal_test_files_opened": False,
            "validation_sessions": SESSIONS,
        },
        "r2": {
            "fp32_encoder": {
                "mean": 0.5,
                "per_session": {name: 0.5 for name in SESSIONS},
            },
            quant_key: {
                "mean": 0.5 + delta,
                "per_session": {name: 0.5 + delta for name in SESSIONS},
            },
            "delta_int8_minus_fp32": {
                "mean": delta,
                "per_session": {name: delta for name in SESSIONS},
            },
        },
        "max_edge_saturation": 0.001,
        "integer_alignment": {
            "int32_overflow_count": 0,
            "max_abs_E": 0.0,
        },
        "integer_package": {
            "path": "package.npz",
            "sha256": "0" * 64,
        },
    }
    if method == "ptq":
        payload.update({"ptq_pass": True, "next_step": "accept_encoder_ptq"})
    else:
        payload.update({"qat_pass": True})
    return payload


def test_mixed_ptq_qat_seed_aggregate(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "fp32.json"
    source_path.write_text(json.dumps(_source()))
    result_dir = tmp_path / "int8"
    for seed, method, delta in (
        (42, "ptq", -0.002),
        (43, "qat", 0.001),
        (44, "qat", -0.004),
    ):
        seed_dir = result_dir / f"seed{seed}"
        (seed_dir / "ptq").mkdir(parents=True)
        if method == "ptq":
            (seed_dir / "ptq" / "ptq_report.json").write_text(
                json.dumps(_report(method="ptq", delta=delta))
            )
        else:
            failed_ptq = _report(method="ptq", delta=-0.02)
            failed_ptq.update(
                {"ptq_pass": False, "next_step": "run_encoder_qat"}
            )
            (seed_dir / "ptq" / "ptq_report.json").write_text(
                json.dumps(failed_ptq)
            )
            (seed_dir / "qat").mkdir()
            (seed_dir / "qat" / "qat_report.json").write_text(
                json.dumps(_report(method="qat", delta=delta))
            )

    out = result_dir / "aggregate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_t4_encoder_int8.py",
            "--source_fp32_aggregate",
            str(source_path),
            "--result_dir",
            str(result_dir),
            "--out",
            str(out),
        ],
    )
    aggregate.main()
    payload = json.loads(out.read_text())
    assert payload["methods"] == {"42": "ptq", "43": "qat", "44": "qat"}
    assert payload["decoder_quantized_in_this_run"] is False
    assert payload["formal_test_files_opened"] is False
    assert payload["integer_max_abs_E"] == 0.0
    assert payload["int32_overflow_count"] == 0
    assert payload["mean_delta_int8_minus_fp32_r2"] == pytest.approx(
        (-0.002 + 0.001 - 0.004) / 3
    )


def test_positive_trigger_is_strict() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        aggregate._validate_positive_source(_source(delta_b0=0.0))
