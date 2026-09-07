"""Adversarial, score-free tests for the r6c live-signing coordinator.

These tests deliberately operate only on a separately imported production
module plus ``tmp_path`` fixtures.  They never call a real r6c bootstrap,
touch a real Phase-C root/anchor/nonce, start CUDA, or open an endpoint.

The suite is intentionally stricter than a happy-path unit test.  r6b proved
that an apparently valid Stage-A launch is worthless if a live signer cannot
issue a valid decision signature and the conditional continuation capabilities
after the score-sealed Stage-A boundary.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from sua_exploration.mc_maze import m2_native_post33_program_v4 as program


ROOT = Path(__file__).resolve().parents[2]
SIGNER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py"
GATE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py"


def _load_module(path: Path, label: str):
    """Import source without executing its CLI/bootstrap entrypoint."""
    module_name = f"_r6c_adversarial_{label}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def signer_module():
    return _load_module(SIGNER, "signer")


@pytest.fixture
def gate_module():
    return _load_module(GATE, "gate")


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, f"expected exactly one function named {name!r}"
    return matches[0]


def test_r6c_controller_and_gate_are_both_sealed_program_runtime_roots() -> None:
    """A hidden controller/gate is a source-closure escape hatch."""
    roots = program.PROGRAM_RUNTIME_ROOTS["open_and_finalize_clis"]
    expected = {
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py",
    }
    assert expected <= set(roots)
    assert expected <= set(program.PROGRAM_EXPECTED_CLOSURE)
    assert expected <= set(program.audit_phase_c_program_closure())


def test_source_has_no_private_key_serialization_or_fork_copy_path() -> None:
    """The only private-key lifetime must be one controller address space.

    ``posix_spawn`` is allowed because it avoids a fork-copy of Python memory.
    ``subprocess``/``Popen``/``fork`` are deliberately rejected: even an
    immediate exec gives a child a transient copy of the signing key.
    """
    source = SIGNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [_dotted_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "subprocess.run" not in calls
    assert "subprocess.Popen" not in calls
    assert "os.fork" not in calls
    assert "os.posix_spawn" in calls or "os.posix_spawnp" in calls
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"private_bytes", "private_bytes_raw"}
        for node in ast.walk(tree)
    )

    # The key must not appear in a CLI option, environment construction, or
    # spawn call.  This is source-level because the real child must never run
    # in an adversarial test.
    argument_literals = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "parser.add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert not any("private" in value.lower() or "secret" in value.lower() for value in argument_literals)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) not in {
            "os.posix_spawn", "os.posix_spawnp"
        }:
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "private_key" not in rendered
        assert "PRIVATE_KEY" not in rendered

    # A helper process is real, so receipts/docs must not make the stronger,
    # false claim that there is *no* child.  The proven property is narrower:
    # no private key is serialized or supplied through the helper boundary.
    forbidden_absolute_claims = (
        "private_key_passed_to_child",
        "never passes one to a child",
        "fresh process rather than a conventional fork child",
        "copy-on-write view of this process",
    )
    assert not [claim for claim in forbidden_absolute_claims if claim in source]
    # Accept either a direct positive statement or its exact logical inverse;
    # both are auditable as long as the literal Boolean is emitted in receipts.
    assert (
        "private_key_not_serialized" in source
        or "private_key_serialized_or_disk_persisted" in source
    )
    assert (
        "private_key_not_supplied_via_argv_environment_file_or_ipc" in source
        or "private_key_not_supplied_via_argv_environment_file_stdin_or_ipc" in source
    )
    assert "helper_spawn_api" in source and "os.posix_spawn" in source


def test_same_controller_process_reloads_r6c_authorization_after_anchor_patch(
    signer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A top-level r6b import must never silently sign an r6c plan.

    The production builder patches the fixed authorization anchor after the
    long-lived controller bootstraps.  Therefore the controller needs an
    explicit reload at plan validation/signing time, not just a top-level
    import cached when its process started.
    """
    assert hasattr(signer_module, "_reload_live_authorization"), (
        "r6c signer must expose a deliberate delayed/reload authorization boundary"
    )
    assert hasattr(signer_module, "importlib"), "reload helper must use importlib, not a stale alias"

    stale = SimpleNamespace(PUBLIC_KEY=Path("/synthetic/r6b.pem"), PUBLIC_KEY_SHA256="old")
    stale_program = SimpleNamespace()
    fresh = SimpleNamespace(PUBLIC_KEY=signer_module.PUBLIC_KEY, PUBLIC_KEY_SHA256="new")
    fresh_program = SimpleNamespace()
    monkeypatch.setattr(signer_module, "authorization", stale, raising=False)
    monkeypatch.setattr(signer_module, "program_module", stale_program, raising=False)
    monkeypatch.setattr(
        signer_module.importlib,
        "reload",
        lambda target: fresh if target is stale else fresh_program,
    )
    signer_module._reload_live_authorization()
    assert signer_module.authorization is fresh
    assert signer_module.authorization is not stale
    assert signer_module.program_module is fresh_program

    tree = ast.parse(SIGNER.read_text(encoding="utf-8"))
    bootstrap = _function(tree, "bootstrap")
    bootstrap_source = ast.get_source_segment(SIGNER.read_text(encoding="utf-8"), bootstrap) or ""
    plan_read = bootstrap_source.index("plan = ")
    reload_at = bootstrap_source.index("_reload_live_authorization()")
    expected_paths_at = bootstrap_source.index("paths = _expected_paths(plan)")
    stage_a_at = bootstrap_source.index("_sign_stage_a(plan")
    assert plan_read < reload_at < expected_paths_at < stage_a_at, (
        "the r6c module reload must occur after the plan is read but before any plan validation/signature"
    )


def _ready_for_binding_test(signer_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Create a fully synthetic ready mapping bound to this test process."""
    anchor = tmp_path / "r6c_public.pem"
    anchor.write_bytes(b"synthetic public anchor only\n")
    monkeypatch.setattr(signer_module, "PUBLIC_KEY", anchor)
    monkeypatch.setattr(signer_module.os, "getpid", lambda: 77123)
    monkeypatch.setattr(signer_module, "_proc_starttime", lambda _pid: 991)
    monkeypatch.setattr(signer_module, "_proc_cmdline_sha256", lambda _pid: "cmd-hash")
    # Force the explicit canonical-host API used by the ready receipt.  The
    # module is patched even before production imports socket, so absence of
    # host binding cannot hide behind an implementation detail.
    monkeypatch.setattr(
        signer_module,
        "socket",
        SimpleNamespace(gethostname=lambda: "synthetic-host"),
        raising=False,
    )
    return {
        "schema": "m2_post33_phase_c_v4_r6c_live_signer_ready_v1",
        "pid": 77123,
        "proc_starttime_ticks": 991,
        "proc_cmdline_sha256": "cmd-hash",
        "host_id": "synthetic-host",
        "controller_source": signer_module.file_metadata(Path(signer_module.__file__).resolve()),
        "public_key": signer_module.file_metadata(anchor),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 77124),
        ("proc_starttime_ticks", 992),
        ("proc_cmdline_sha256", "other-command"),
        ("host_id", "other-host"),
        ("controller_source", {"canonical_path": "/wrong/source.py", "size_bytes": 1, "sha256": "0" * 64}),
        ("public_key", {"canonical_path": "/wrong/anchor.pem", "size_bytes": 1, "sha256": "1" * 64}),
    ],
)
def test_ready_liveness_binding_rejects_every_identity_drift(
    signer_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """PID, starttime, command, host, source, and public anchor are one binding."""
    ready = _ready_for_binding_test(signer_module, tmp_path, monkeypatch)
    signer_module._assert_live_binding(ready)
    tampered = deepcopy(ready)
    tampered[field] = value
    with pytest.raises(PermissionError):
        signer_module._assert_live_binding(tampered)


def test_bootstrap_refuses_an_existing_runtime_directory_before_key_generation(
    signer_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale runtime event/failure must burn the attempt instead of coexisting.

    This invokes only the initial fresh-root guard with every production path
    replaced by a disposable directory.  It cannot generate a usable anchor,
    read a plan, or spawn a helper.
    """
    sandbox = tmp_path / "sandbox"
    receipts = sandbox / "receipts"
    cells = sandbox / "cells"
    runtime = sandbox / "runtime"
    anchor = sandbox / "anchor.pem"
    ready = runtime / "live_signer_ready.json"
    runtime.mkdir(parents=True)
    monkeypatch.setattr(signer_module, "R6C_RECEIPTS", receipts)
    monkeypatch.setattr(signer_module, "R6C_CELL_ROOT", cells)
    monkeypatch.setattr(signer_module, "RUNTIME", runtime)
    monkeypatch.setattr(signer_module, "PUBLIC_KEY", anchor)
    monkeypatch.setattr(signer_module, "READY", ready)
    with pytest.raises(FileExistsError):
        signer_module.bootstrap()
    assert not anchor.exists(), "freshness must be checked before the key/public anchor is generated"


def test_partial_or_unstable_plan_cannot_satisfy_the_wait_boundary(
    signer_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existence alone is not publication: a truncated JSON plan cannot return.

    A bounded fake sleep avoids an indefinite wait while accepting either safe
    behavior: reject the malformed file immediately or continue waiting for a
    complete canonical publication.  Returning normally is unsafe.
    """
    partial = tmp_path / "r6c_live_signer_plan.json"
    partial.write_text('{"schema":', encoding="utf-8")

    def exhausted(_seconds: float) -> None:
        raise TimeoutError("test bounded an intentional stable-publication wait")

    monkeypatch.setattr(signer_module.time, "sleep", exhausted)
    with pytest.raises((PermissionError, ValueError, TimeoutError)):
        signer_module._wait_for(partial, label="synthetic truncated plan")


def test_gate_has_three_distinct_no_output_terminal_states(
    signer_module, gate_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid STOP is not an invalid/crashed validator result."""
    assert {
        gate_module.EXIT_CONTINUE,
        gate_module.EXIT_VALIDATED_STOP,
        gate_module.EXIT_INVALID,
    } == {0, 1, 10}
    assert {
        signer_module.EXIT_CONTINUE,
        signer_module.EXIT_VALIDATED_STOP,
        signer_module.EXIT_INVALID,
    } == {0, 1, 10}

    def invoke(value: object | BaseException) -> int:
        def validate(_root: Path, *, require_continue: bool):
            assert require_continue is False
            if isinstance(value, BaseException):
                raise value
            return value

        monkeypatch.setattr(gate_module, "validate_stage_a_decision", validate)
        monkeypatch.setattr(sys, "argv", [str(GATE), "--root", "/synthetic/root", "--classify-signed"])
        with pytest.raises(SystemExit) as stopped:
            gate_module.main()
        return int(stopped.value.code)

    assert invoke({"decision": "continue_without_positive_claim"}) == gate_module.EXIT_CONTINUE
    assert invoke({"decision": "seed42_severe_negative_futility_stop"}) == gate_module.EXIT_VALIDATED_STOP
    assert invoke(RuntimeError("validator crash")) == gate_module.EXIT_INVALID
    assert invoke({"decision": "unknown"}) == gate_module.EXIT_INVALID

    monkeypatch.setattr(signer_module, "_run_score_blind_child", lambda _argv: signer_module.EXIT_CONTINUE)
    assert signer_module._decision_gate() == "continue"
    monkeypatch.setattr(signer_module, "_run_score_blind_child", lambda _argv: signer_module.EXIT_VALIDATED_STOP)
    assert signer_module._decision_gate() == "validated_stop"
    monkeypatch.setattr(signer_module, "_run_score_blind_child", lambda _argv: signer_module.EXIT_INVALID)
    with pytest.raises(PermissionError, match="failed closed"):
        signer_module._decision_gate()


def test_posix_spawn_helper_receives_only_key_free_fixed_boundary(
    signer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspect the helper call without ever executing it."""
    observed: dict[str, object] = {}
    monkeypatch.setattr(signer_module.os, "open", lambda *_args: 71)
    monkeypatch.setattr(signer_module.os, "close", lambda _fd: None)

    def spawn(executable: str, argv: list[str], environment: dict[str, str], *, file_actions: object):
        observed.update({
            "executable": executable,
            "argv": list(argv),
            "environment": dict(environment),
            "file_actions": file_actions,
        })
        return 9191

    monkeypatch.setattr(signer_module.os, "posix_spawn", spawn)
    monkeypatch.setattr(signer_module.os, "waitpid", lambda _pid, _flags: (9191, 0))
    result = signer_module._run_score_blind_child(
        [sys.executable, str(GATE), "--root", "/synthetic/no-score-root", "--classify-signed"]
    )
    assert result == 0
    assert observed["executable"] == sys.executable
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert not any("private" in token.lower() or "secret" in token.lower() for token in argv)
    environment = observed["environment"]
    assert environment == {"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"}
    actions = observed["file_actions"]
    assert isinstance(actions, list) and len(actions) == 3
    assert {action[2] for action in actions} == {0, 1, 2}


def test_conditional_auth_expiry_is_checked_before_any_output_is_materialized() -> None:
    """An expired Stage-B/full authorization must not leave an unsigned file."""
    source = SIGNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = _function(tree, "_issue_one_conditional")
    function_source = ast.get_source_segment(source, function) or ""
    assert function_source.index("_assert_current_authorization_window(body)") < function_source.index(
        "write_json_exclusive(output"
    )
