from __future__ import annotations

from pathlib import Path

import pytest

from scripts import execute_cebra_pseudosession_sua_cell as subject


def test_train_argv_is_exactly_accepted() -> None:
    from scripts.train_cebra_pseudosession_sua import validate_argv

    assert validate_argv(subject.train_argv("t4", 42)) == ("t4", 42)
    assert validate_argv(subject.train_argv("z4", 42)) == ("z4", 42)


def test_paths_are_arm_and_seed_isolated() -> None:
    t4 = subject.paths_for("t4", 42)
    z4 = subject.paths_for("z4", 42)
    assert set(t4) == set(z4)
    assert all(t4[key] != z4[key] for key in t4)


def test_freshness_rejects_body_or_sidecar(tmp_path: Path) -> None:
    targets = {"one": tmp_path / "one"}
    subject.validate_fresh(targets)
    Path(str(targets["one"]) + ".sha256").write_text("stale\n", encoding="ascii")
    with pytest.raises(subject.ExecuteError):
        subject.validate_fresh(targets)


def test_live_plan_is_no_write() -> None:
    targets = subject.paths_for("t4", 42)
    if any(path.exists() or Path(str(path) + ".sha256").exists() for path in targets.values()):
        pytest.skip("stage-P T4 has already been launched")
    payload = subject.plan("t4", 42, 0)
    assert payload["training_started"] is False
    assert payload["formal_subc_test_nwb_opened"] is False
    assert all(not path.exists() for path in targets.values())
