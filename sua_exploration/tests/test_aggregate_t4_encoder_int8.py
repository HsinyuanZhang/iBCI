from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
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
            "seeds": [42, 43, 44],
            "sessions": SESSIONS,
        },
        "contrasts": {
            "t4_vs_original_spint_b0": {
                "mean_paired_delta_r2": delta_b0,
                "passes_all_gates": True,
                "gates": {"frozen_gate": True},
            },
            "t4_vs_shuffled_label_ts4": {
                "mean_paired_delta_r2": delta_ts4,
                "passes_all_gates": True,
                "gates": {"frozen_gate": True},
            },
        },
    }


def _report(
    *, method: str, delta: float, seed: int, root: Path, selection_path: Path
) -> dict:
    quant_key = "int8_encoder" if method == "ptq" else "qat_int8_encoder"
    payload = {
        "schema_version": 1,
        "seed": seed,
        "scope": (
            "T4/B3S identity encoder W8A8 + FP32 decoder"
            if method == "ptq"
            else "T4/B3S identity encoder QAT W8A8 + frozen FP32 decoder"
        ),
        "decoder_quantized_in_this_run": False,
        "final_architecture_selection_sha256": hashlib.sha256(
            selection_path.read_bytes()
        ).hexdigest(),
        "protocol": {
            "formal_test_files_opened": False,
            "validation_sessions": SESSIONS,
            "sealed_formal_test_receipts": [f"test-{index}" for index in range(6)],
            "activity_calibration_n": 30,
            "t4_label_feature_pool_n": 50,
            "evaluation_start_trial": 50,
            "scale_fit_sessions": [f"train-{index}" for index in range(27)],
            "training_sessions": [f"train-{index}" for index in range(27)],
            "validation_labels_used_for_scale_selection": False,
            "fixed_epoch_budget": 8,
            "validation_used_for_epoch_selection": False,
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
        "gates": {"frozen_gate": True},
        "integer_alignment": {
            "int32_overflow_count": 0,
            "max_abs_E": 0.0,
        },
    }
    checkpoint = root / f"seed{seed}.ckpt"
    checkpoint.write_bytes(f"checkpoint-{seed}".encode())
    package = root / f"seed{seed}.npz"
    shapes = {
        "pre_pool": (64, 100),
        "post0": (64, 68),
        "post1": (64, 64),
        "post2": (50, 64),
    }
    arrays = {}
    for name, shape in shapes.items():
        arrays[f"{name}.weight_int8"] = np.zeros(shape, dtype=np.int8)
        arrays[f"{name}.weight_scale_fp32"] = np.ones(shape[0], dtype=np.float32)
        arrays[f"{name}.bias_int32"] = np.zeros(shape[0], dtype=np.int32)
        arrays[f"{name}.requant_mult_int64"] = np.ones(shape[0], dtype=np.int64)
        arrays[f"{name}.requant_shift_int32"] = np.zeros(1, dtype=np.int32)
    np.savez_compressed(package, **arrays)
    payload.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "integer_package": {
                "path": str(package),
                "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                "layers": [
                    {
                        "name": name,
                        "weight_bits": 8,
                        "activation_bits": 8,
                        "accumulator_bits": 32,
                        "integer_requant": True,
                        "weight_shape": list(shapes[name]),
                    }
                    for name in ("pre_pool", "post0", "post1", "post2")
                ],
                "side_concat": {
                    "side_dim": 4,
                    "post0_input_dim": 68,
                    "integer_domain_concat": True,
                },
                "decoder_quantized": False,
            },
        }
    )
    if method == "ptq":
        payload.update({"ptq_pass": True, "next_step": "accept_encoder_ptq"})
    else:
        payload.update({"qat_pass": True})
    return payload


def test_mixed_ptq_qat_seed_aggregate(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "fp32.json"
    source_path.write_text(json.dumps(_source()))
    result_dir = tmp_path / "int8"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_splits": {
                    "train": [f"train-{index}" for index in range(27)],
                    "val": SESSIONS,
                    "test": [f"test-{index}" for index in range(6)],
                }
            }
        )
    )
    selection = {
        "selected_architecture": "coupled_ordinary_B3S_T4",
        "strict_manifest_path": str(manifest_path),
        "selected_seed_artifacts": {},
    }
    for seed, method, delta in (
        (42, "ptq", -0.002),
        (43, "qat", 0.001),
        (44, "qat", -0.004),
    ):
        seed_dir = result_dir / f"seed{seed}"
        (seed_dir / "ptq").mkdir(parents=True)
        checkpoint = tmp_path / f"seed{seed}.ckpt"
        selection["selected_seed_artifacts"][str(seed)] = {
            "checkpoint": {"path": str(checkpoint)}
        }
        if method == "ptq":
            (seed_dir / "ptq" / "ptq_report.json").write_text(
                json.dumps(
                    _report(
                        method="ptq",
                        delta=delta,
                        seed=seed,
                        root=tmp_path,
                        selection_path=selection_path,
                    )
                )
            )
        else:
            failed_ptq = _report(
                method="ptq",
                delta=-0.02,
                seed=seed,
                root=tmp_path,
                selection_path=selection_path,
            )
            failed_ptq.update(
                {"ptq_pass": False, "next_step": "run_encoder_qat"}
            )
            (seed_dir / "ptq" / "ptq_report.json").write_text(
                json.dumps(failed_ptq)
            )
            (seed_dir / "qat").mkdir()
            (seed_dir / "qat" / "qat_report.json").write_text(
                json.dumps(
                    _report(
                        method="qat",
                        delta=delta,
                        seed=seed,
                        root=tmp_path,
                        selection_path=selection_path,
                    )
                )
            )

    out = result_dir / "aggregate.json"
    monkeypatch.setattr(
        aggregate,
        "validate_selection",
        lambda _path, source_fp32_path=None: (
            selection,
            _source(),
            {"t4_minus_b0": 0.06, "t4_minus_ts4": 0.05},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_t4_encoder_int8.py",
            "--source_fp32_aggregate",
            str(source_path),
            "--selection",
            str(selection_path),
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
    with pytest.raises(ValueError, match=">=0.03"):
        aggregate._validate_positive_source(_source(delta_b0=0.0))


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("parity", "STE/integer output is not exact"),
        ("overflow", "INT32 overflow is nonzero"),
        ("saturation", "saturation missed"),
        ("delta", "R2 delta missed"),
        ("gate", "incomplete/failed gate"),
        ("budget", "30/50/50"),
        ("scope", "scope drifted"),
        ("package", "not W8A8/INT32"),
    ],
)
def test_seed_aggregate_rejects_self_reported_pass_with_broken_contract(
    tmp_path: Path, fault: str, message: str
) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")
    seed_dir = tmp_path / "seed42"
    (seed_dir / "ptq").mkdir(parents=True)
    report = _report(
        method="ptq",
        delta=-0.002,
        seed=42,
        root=tmp_path,
        selection_path=selection_path,
    )
    if fault == "parity":
        report["integer_alignment"]["max_abs_E"] = 1e-4
    elif fault == "overflow":
        report["integer_alignment"]["int32_overflow_count"] = 1
    elif fault == "saturation":
        report["max_edge_saturation"] = 0.006
    elif fault == "delta":
        report["r2"]["int8_encoder"]["mean"] = 0.48
        report["r2"]["delta_int8_minus_fp32"]["mean"] = -0.02
        report["r2"]["int8_encoder"]["per_session"] = {
            name: 0.48 for name in SESSIONS
        }
        report["r2"]["delta_int8_minus_fp32"]["per_session"] = {
            name: -0.02 for name in SESSIONS
        }
    elif fault == "gate":
        report["gates"]["frozen_gate"] = False
    elif fault == "budget":
        report["protocol"]["t4_label_feature_pool_n"] = 30
    elif fault == "scope":
        report["scope"] = "full-model INT8"
    elif fault == "package":
        report["integer_package"]["layers"][0]["weight_bits"] = 16
    (seed_dir / "ptq" / "ptq_report.json").write_text(json.dumps(report))
    selection = {
        "selected_seed_artifacts": {
            "42": {"checkpoint": {"path": report["checkpoint"]}}
        }
    }
    with pytest.raises(ValueError, match=message):
        aggregate._load_seed(
            seed_dir,
            42,
            selection_path=selection_path,
            selection=selection,
            expected_train_sessions=[f"train-{index}" for index in range(27)],
            expected_sessions=sorted(SESSIONS),
            expected_formal_sessions=[f"test-{index}" for index in range(6)],
        )


def test_seed_aggregate_opens_npz_and_rejects_array_shape_drift(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")
    seed_dir = tmp_path / "seed42"
    (seed_dir / "ptq").mkdir(parents=True)
    report = _report(
        method="ptq",
        delta=-0.002,
        seed=42,
        root=tmp_path,
        selection_path=selection_path,
    )
    package = Path(report["integer_package"]["path"])
    with np.load(package, allow_pickle=False) as original:
        arrays = {name: original[name] for name in original.files}
    arrays["post0.weight_int8"] = np.zeros((64, 67), dtype=np.int8)
    np.savez_compressed(package, **arrays)
    report["integer_package"]["sha256"] = hashlib.sha256(
        package.read_bytes()
    ).hexdigest()
    (seed_dir / "ptq" / "ptq_report.json").write_text(json.dumps(report))
    selection = {
        "selected_seed_artifacts": {
            "42": {"checkpoint": {"path": report["checkpoint"]}}
        }
    }
    with pytest.raises(ValueError, match="post0.weight_int8 dtype/shape drifted"):
        aggregate._load_seed(
            seed_dir,
            42,
            selection_path=selection_path,
            selection=selection,
            expected_train_sessions=[f"train-{index}" for index in range(27)],
            expected_sessions=sorted(SESSIONS),
            expected_formal_sessions=[f"test-{index}" for index in range(6)],
        )
