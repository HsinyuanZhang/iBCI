from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_pseudomua_t4_bridge as agg  # noqa: E402


TRAIN = [f"train-{index:02d}" for index in range(27)]
VAL = [f"val-{index:02d}" for index in range(6)]
TEST = [f"test-{index:02d}" for index in range(6)]
SPLITS = {"train": TRAIN, "val": VAL, "test": TEST}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_screen(
    root: Path,
    *,
    view: str,
    gains: dict[str, float],
    seed_offsets: dict[int, float] | None = None,
) -> Path:
    results_dir = root / f"{view}_results"
    results_dir.mkdir(parents=True)
    seed_offsets = seed_offsets or {42: -0.002, 43: 0.0, 44: 0.002}
    base_score = 0.30
    for group in agg.GROUPS:
        for seed in agg.SEEDS:
            run_dir = root / "runs" / view / group.lower() / str(seed)
            run_dir.mkdir(parents=True)
            variant = agg.GROUP_CONTRACT[group]["variant"]
            side_group = agg.GROUP_CONTRACT[group]["side_features_group"]
            metadata = {
                "schema_version": 1,
                "status": "completed",
                "variant": variant,
                "seed": seed,
                "signal_view": view,
                "task": "CO",
                "split_counts": [27, 6, 6],
                "max_units_exclusive": 100,
                "held_out_test_evaluated": False,
                "output_dir": str(run_dir.resolve()),
                "session_splits": SPLITS,
                "side_features": {"group": side_group, "pool_size": 50},
                "training": {
                    "max_epochs": 12,
                    "no_early_stopping": True,
                    "checkpoint_every_epoch": True,
                },
            }
            metadata_path = run_dir / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            session_values = {
                session: base_score
                + gains[group]
                + seed_offsets[seed]
                + session_index * 0.001
                for session_index, session in enumerate(VAL)
            }
            per_epoch = {
                str(epoch): {
                    "mean_r2": sum(session_values.values()) / len(session_values),
                    "per_session_r2": session_values,
                }
                for epoch in agg.EXPECTED_EPOCHS
            }
            artifact = {
                "schema_version": 1,
                "variant": variant,
                "seed": seed,
                "signal_view": view,
                "task": "CO",
                "split_counts": [27, 6, 6],
                "max_units_exclusive": 100,
                "no_test_files_evaluated": True,
                "calibration_trial_selection_uses_behavior_labels": False,
                "uses_behavior_labels_for_weight_updates": False,
                "uses_backward_gradients": False,
                "session_splits": SPLITS,
                "protocol": {
                    "total_epochs": 12,
                    "burn_in_epochs": 4,
                    "epoch_window": agg.EXPECTED_EPOCHS,
                    "selection_mode": "first",
                    "calibration_n": 30,
                    "pool_size": 50,
                },
                "epoch_list": agg.EXPECTED_EPOCHS,
                "per_epoch": per_epoch,
                "run_dir": str(run_dir.resolve()),
                "run_metadata_path": str(metadata_path.resolve()),
                "run_metadata_sha256": _sha256(metadata_path),
            }
            (results_dir / f"{group.lower()}_s{seed}.json").write_text(
                json.dumps(artifact, sort_keys=True), encoding="utf-8"
            )
    return results_dir


def _table(delta: float):
    sessions = [f"s{index}" for index in range(6)]
    table = {}
    for group, score in (
        ("T4", 0.5),
        ("F0", 0.5 - delta),
        ("TS4", 0.5 - delta),
    ):
        for seed in agg.SEEDS:
            table[group, seed] = {
                session: score + 0.0001 * seed for session in sessions
            }
    return table, sessions


def test_pair_effective_uses_real_paired_and_unpaired_uncertainty():
    table, sessions = _table(0.05)
    result = agg.compute_pair(table, "T4", "F0", sessions)
    assert result["verdict"] == "effective"
    assert result["mean_delta"] == pytest.approx(0.05)
    assert result["paired_se"] == pytest.approx(0.0, abs=1e-15)
    assert result["unpaired_quadrature_se"] > 0.0
    assert result["implied_seed_correlation"] is not None
    assert result["n_sessions_positive"] == 6
    assert result["n_seeds_positive"] == 3


def test_end_to_end_equal_views_are_view_invariant(tmp_path):
    sua = _write_screen(
        tmp_path / "sua",
        view="sua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    pseudo = _write_screen(
        tmp_path / "pseudo",
        view="pseudo_mua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    payload = agg.aggregate(pseudo, sua)
    assert payload["views"]["sua"]["T4_group_verdict"] == "effective"
    assert payload["views"]["pseudo_mua"]["T4_group_verdict"] == "effective"
    assert payload["Gamma"]["mean"] == pytest.approx(0.0, abs=1e-15)
    assert payload["Gamma"]["paired_se"] == pytest.approx(0.0, abs=1e-15)
    assert payload["Gamma"]["verdict"] == "view_invariant_within_tolerance"


@pytest.mark.parametrize(
    ("sua_gain", "pseudo_gain", "expected"),
    [
        (0.10, 0.04, "sua_specific_amplification"),
        (0.04, 0.10, "pseudo_mua_amplification"),
    ],
)
def test_gamma_direction(tmp_path, sua_gain, pseudo_gain, expected):
    sua = _write_screen(
        tmp_path / "sua",
        view="sua",
        gains={"F0": 0.0, "T4": sua_gain, "TS4": 0.0},
    )
    pseudo = _write_screen(
        tmp_path / "pseudo",
        view="pseudo_mua",
        gains={"F0": 0.0, "T4": pseudo_gain, "TS4": 0.0},
    )
    assert agg.aggregate(pseudo, sua)["Gamma"]["verdict"] == expected


def test_missing_artifact_is_rejected(tmp_path):
    sua = _write_screen(
        tmp_path / "sua",
        view="sua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    pseudo = _write_screen(
        tmp_path / "pseudo",
        view="pseudo_mua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    (pseudo / "t4_s43.json").unlink()
    with pytest.raises(FileNotFoundError, match="t4_s43"):
        agg.aggregate(pseudo, sua)


def test_signal_view_mismatch_is_rejected(tmp_path):
    results = _write_screen(
        tmp_path,
        view="pseudo_mua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    path = results / "t4_s42.json"
    payload = json.loads(path.read_text())
    payload["signal_view"] = "sua"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signal_view"):
        agg.load_artifact(path, group="T4", seed=42, view="pseudo_mua")


def test_cross_protocol_mismatch_is_rejected(tmp_path):
    sua = _write_screen(
        tmp_path / "sua",
        view="sua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    pseudo = _write_screen(
        tmp_path / "pseudo",
        view="pseudo_mua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    path = pseudo / "ts4_s44.json"
    payload = json.loads(path.read_text())
    payload["protocol"]["calibration_n"] = 29
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol.calibration_n"):
        agg.aggregate(pseudo, sua)


def test_run_metadata_hash_mutation_is_rejected(tmp_path):
    results = _write_screen(
        tmp_path,
        view="pseudo_mua",
        gains={"F0": 0.0, "T4": 0.08, "TS4": 0.0},
    )
    artifact_path = results / "f0_s42.json"
    artifact = json.loads(artifact_path.read_text())
    metadata_path = Path(artifact["run_metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="run_metadata_sha256"):
        agg.load_artifact(
            artifact_path, group="F0", seed=42, view="pseudo_mua"
        )
