"""Prove that v5 evaluators are v4 copies except for the device repair block."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _main_body(path: Path) -> list[ast.stmt]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            # The evaluator's production statements are deliberately enclosed
            # by one release-snapshot ``try/finally``.  The parser/ownership
            # preamble remains outside and is unchanged by the v5 delta; the
            # only allowed replacement is in this protected try body.
            tries = [statement for statement in node.body if isinstance(statement, ast.Try)]
            if len(tries) != 1:
                raise AssertionError(f"{path} main() must retain one evaluator try/finally")
            return tries[0].body
    raise AssertionError(f"{path} has no main()")


def _dump(statements: list[ast.stmt]) -> str:
    return ast.dump(ast.Module(body=statements, type_ignores=[]), include_attributes=False)


def _call_path(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    node: ast.AST = statement.value.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _assignment_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _first_index(body: list[ast.stmt], predicate) -> int:
    for index, statement in enumerate(body):
        if predicate(statement):
            return index
    raise AssertionError("expected statement not found")


@pytest.mark.parametrize(
    "v4_relative,v5_relative,arm,resume_kind",
    [
        (
            "SPINT-main/src/evaluate_post33_phase_c_v4.py",
            "SPINT-main/src/evaluate_post33_phase_c_v5.py",
            "spint",
            "split_assignment",
        ),
        (
            "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
            "streaming_calibration_exp/src/evaluate_post33_phase_c_v5.py",
            "t4",
            "cost_write",
        ),
    ],
)
def test_v5_main_is_exact_v4_copy_except_post_test_device_block(
    v4_relative: str, v5_relative: str, arm: str, resume_kind: str,
) -> None:
    v4_path = ROOT / v4_relative
    v5_path = ROOT / v5_relative
    v4 = _main_body(v4_path)
    v5 = _main_body(v5_path)

    old_start = _first_index(v4, lambda statement: _call_path(statement) == "model.eval")
    new_start = _first_index(v5, lambda statement: _assignment_target(statement) == "neural_cpu")
    assert _dump(v4[:old_start]) == _dump(v5[:new_start])

    if resume_kind == "split_assignment":
        old_resume = _first_index(v4, lambda statement: _assignment_target(statement) == "split")
        new_resume = _first_index(v5, lambda statement: _assignment_target(statement) == "split")
    else:
        old_resume = _first_index(v4, lambda statement: _call_path(statement) == "write_json_exclusive")
        new_resume = _first_index(v5, lambda statement: _call_path(statement) == "write_json_exclusive")
    assert _dump(v4[old_resume:]) == _dump(v5[new_resume:])

    # The historical two-statement block is the only executable code removed.
    assert len(v4[old_start:old_resume]) == 2
    assert _call_path(v4[old_start]) == "model.eval"
    assert _call_path(v4[old_start + 1]) == "profiler.benchmark_online_b1"

    # The exact replacement first validates the same CPU B=1 window, then
    # prepares the named arm, binds checks to the profiler's actual CUDA input,
    # and finally re-checks canonical cache bytes/hash after all profiling calls.
    replacement = v5[new_start:new_resume]
    assert len(replacement) == 4
    assert _assignment_target(replacement[0]) == "neural_cpu"
    assert _assignment_target(replacement[1]) == "prepared"
    assert _call_path(replacement[2]) == "profiler.benchmark_online_b1"
    assert _call_path(replacement[3]) == "prepared.assert_ready"
    prepare_call = replacement[1].value  # type: ignore[union-attr]
    assert isinstance(prepare_call, ast.Call)
    assert _call_path(ast.Expr(value=prepare_call)) == "prepare_cached_online_b1_after_trainer_test_v5"
    keywords = {keyword.arg: keyword.value for keyword in prepare_call.keywords}
    assert isinstance(keywords.get("arm"), ast.Constant)
    assert keywords["arm"].value == arm
    assert isinstance(keywords.get("neural_cpu"), ast.Name)
    assert keywords["neural_cpu"].id == "neural_cpu"

    text = v5_path.read_text(encoding="utf-8")
    assert "prepare_cached_online_b1_after_trainer_test_v5" in text
    assert "prepared.bind_checked_forward(model.cached_online_forward)" in text
    assert "prepared.assert_ready()" in text
