from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

from budget_matched_posterior_cal_aug_c3_v1 import score as v1
from budget_matched_posterior_cal_aug_c3_v1 import score_v2 as v2


def _publish(root: Path, name: str, source: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, root / name)
    shutil.copyfile(Path(str(source) + ".sha256"), root / f"{name}.sha256")
    (root / name).chmod(0o444)
    (root / f"{name}.sha256").chmod(0o444)


def test_failed_v1_exact_graph_and_semantics(tmp_path: Path):
    repository = Path(__file__).resolve().parents[2]
    source = repository / v2.FAILED_V1_ROOT_RELATIVE
    target = tmp_path / v2.FAILED_V1_ROOT_RELATIVE
    _publish(target, "attempt.json", source / "attempt.json")
    _publish(target, "failure.json", source / "failure.json")
    target.chmod(0o555)
    result = v2.validate_failed_v1(tmp_path)
    assert result["attempt_sha256"] == v2.FAILED_V1_ATTEMPT_SHA256
    assert result["failure_sha256"] == v2.FAILED_V1_FAILURE_SHA256
    assert result["exact_leaf_count"] == 4


def test_failed_v1_rejects_valid_rehashed_semantic_drift(tmp_path: Path):
    repository = Path(__file__).resolve().parents[2]
    source = repository / v2.FAILED_V1_ROOT_RELATIVE
    target = tmp_path / v2.FAILED_V1_ROOT_RELATIVE
    _publish(target, "attempt.json", source / "attempt.json")
    failure = json.loads((source / "failure.json").read_text())
    failure["stage"] = "post_attempt"
    target.mkdir(parents=True, exist_ok=True)
    body = target / "failure.json"
    body.write_text(json.dumps(failure, sort_keys=True, separators=(",", ":")) + "\n")
    digest = v1._file_sha(body)
    (target / "failure.json.sha256").write_text(f"{digest}  failure.json\n")
    body.chmod(0o444)
    (target / "failure.json.sha256").chmod(0o444)
    target.chmod(0o555)
    try:
        v2.validate_failed_v1(tmp_path)
    except v1.C3ScoreError:
        pass
    else:
        raise AssertionError("semantic/predecessor digest drift was accepted")


def test_streaming_builder_uses_independent_namespace():
    from models.components.streaming_encoders import build_encoder

    assert callable(build_encoder)


def test_v2_closure_keeps_review_non_numeric():
    repository = Path(__file__).resolve().parents[2]
    execution = v2.execution_closure(repository)
    review = v2.review_closure(repository)
    assert set(execution["files"]) == set(v2.EXECUTION_PATHS)
    assert set(review["files"]) == set(v2.REVIEW_PATHS)
    assert not set(execution["files"]) & set(review["files"])
    changed = {**review, "files": dict(review["files"]), "closure_sha256": "f" * 64}
    first = next(iter(changed["files"]))
    changed["files"][first] = {"sha256": "0" * 64, "bytes": 0}
    assert v1.review_drift(review, changed)["status"] == "ACCEPTED_NON_NUMERIC_DRIFT"


def test_v2_result_root_is_preserved_as_exact_failed_predecessor():
    repository = Path(__file__).resolve().parents[2]
    path = repository / v2.RESULT_ROOT_RELATIVE
    assert path.is_dir() and not path.is_symlink()
    assert stat.S_IMODE(path.lstat().st_mode) == 0o555
    assert sorted(entry.name for entry in path.iterdir()) == [
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"
    ]
