from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from sua_exploration.evalai_t4_m2_activity_budget import export_budget_payload as export


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chronological_budget_indices() -> None:
    class Dataset:
        calib_trial_target_angles = {"s": np.linspace(-np.pi, np.pi, 30, dtype=np.float32)}

    assert export.support_indices(Dataset(), "s", 10).tolist() == list(range(10))
    assert export.support_indices(Dataset(), "s", 30).tolist() == list(range(30))


def test_m4_is_doptimal_inside_first30_and_not_chronological_alias() -> None:
    class Dataset:
        pass

    dataset = Dataset()
    angles = np.full(35, np.nan, dtype=np.float32)
    angles[[1, 5, 11, 18, 24, 29]] = np.asarray(
        [0.0, np.pi / 3, 2 * np.pi / 3, np.pi, -2 * np.pi / 3, -np.pi / 3],
        dtype=np.float32,
    )
    dataset.calib_trial_target_angles = {"s": angles}
    selected = export.support_indices(dataset, "s", 4)
    assert selected.shape == (4,)
    assert np.all(selected < 30)
    assert np.isfinite(angles[selected]).all()
    assert selected.tolist() != [0, 1, 2, 3]


def test_identity_inputs_use_activity30_but_only_budgeted_side_rows(monkeypatch) -> None:
    class Dataset:
        pass

    dataset = Dataset()
    session = "s"
    dataset.calib_trialized_neural_features = {
        session: np.arange(35 * 100 * 96, dtype=np.float32).reshape(35, 100, 96)
    }
    dataset.calib_trial_target_angles = {session: np.linspace(-np.pi, np.pi, 35, dtype=np.float32)}
    selected_seen = []

    def fake_ridge(_dataset, _session, selected):
        selected_seen.append(np.asarray(selected).copy())
        return np.zeros((96, 4), dtype=np.float32), {"budget": int(len(selected))}

    import tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical as parent

    monkeypatch.setattr(parent, "_ridge_side", fake_ridge)
    activity, side, evidence = export.build_identity_inputs(dataset, session, 10)
    assert activity.shape == (30, 100, 96)
    assert np.array_equal(activity, dataset.calib_trialized_neural_features[session][:30])
    assert side.shape == (96, 4)
    assert selected_seen[0].tolist() == list(range(10))
    assert evidence["activity_budget"] == 30


def test_dry_cli_is_torch_free_and_writes_nothing(tmp_path: Path) -> None:
    script = REPO_ROOT / "sua_exploration/evalai_t4_m2_activity_budget/export_budget_payload.py"
    output = tmp_path / "candidate.pkl"
    probe = (
        "import runpy,sys; "
        f"sys.argv=['export','--budget','4','--output',{str(output)!r}]; "
        f"runpy.run_path({str(script)!r},run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE_NO_SUBMISSION"
    assert payload["budget"] == 4 and payload["activity_budget"] == 30
    assert not output.exists()


def test_docker_candidate_reuses_validated_runtime() -> None:
    dockerfile = (
        REPO_ROOT / "sua_exploration/evalai_t4_m2_activity_budget/Dockerfile"
    ).read_text(encoding="utf-8")
    assert "COPY evalai_t4_m2/t4_spint_decoder.py" in dockerfile
    assert "COPY evalai_t4_m2/decode.py" in dockerfile
    assert "COPY ${PAYLOAD_FILE} /data/decoder.pkl" in dockerfile
