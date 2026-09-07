"""Adversarial regression tests for the V8 formal-transition quarantine.

No fixture here creates a policy, authorization, grant, real checkpoint, or
external sub-M object.  The point is narrower: prove that no caller-owned
Python object/path/key/payload can enter any V8 privileged transition.
"""
from __future__ import annotations

import ast
import copy
import inspect
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sua_exploration.mc_maze import subm_co_three_arm_score_only_v8 as core
from sua_exploration.scripts import run_dandi688_subm_co_three_arm_score_only_v8 as runner
from sua_exploration.scripts import write_dandi688_subm_co_three_arm_score_only_prelaunch_v8 as writer


class AttackerObject:
    pass


class AttackerSubclass(dict):
    pass


class AttackerProxy:
    def __init__(self, target: object):
        self.target = target

    def __getattr__(self, name: str) -> object:
        return getattr(self.target, name)


def _spoofed_root_like_object() -> AttackerObject:
    attacker = AttackerObject()
    # This exactly mirrors the V7 P0 technique: a caller obtains/creates a
    # private-looking token and rewrites provenance fields with object.__setattr__.
    object.__setattr__(attacker, "_mint", object())
    object.__setattr__(attacker, "_origin", "pinned_v7_anchor")
    object.__setattr__(attacker, "_anchor_sha256", "a" * 64)
    object.__setattr__(attacker, "policy_root", "/attacker/policy")
    object.__setattr__(attacker, "checkpoint_root", "/attacker/checkpoints")
    object.__setattr__(attacker, "run_auth_root", "/attacker/auth")
    object.__setattr__(attacker, "public_key", b"attacker-owned-key")
    return attacker


def _privileged_functions():
    return (core.verify_complete_policy, core.construct_contract, core.verify_run_authorization, core.open_score_ledger)


def test_checked_in_v8_status_is_strictly_blocked() -> None:
    status = core.production_status()
    assert status["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"
    assert all(status[key] is False for key in (
        "formal_root_chain_active", "external_subm_nwb_allowed", "real_checkpoint_allowed",
        "torch_allowed", "gpu_allowed", "r2_allowed", "complete_policy_allowed",
        "verified_grant_allowed", "production_api_accepts_caller_roots_paths_keys_payloads",
    ))
    with pytest.raises(core.V8BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.refuse_blocked_execution()


def test_every_privileged_transition_has_zero_arguments() -> None:
    for function in _privileged_functions():
        assert list(inspect.signature(function).parameters) == [], function.__name__


def test_v7_root_mint_object_setattr_copy_pickle_subclass_and_proxy_cannot_enter() -> None:
    original = _spoofed_root_like_object()
    candidates = [
        original,
        copy.copy(original),
        copy.deepcopy(original),
        pickle.loads(pickle.dumps(original)),
        AttackerSubclass(root=original, public_key=b"self-signed"),
        AttackerProxy(original),
    ]
    for candidate in candidates:
        for function in _privileged_functions():
            with pytest.raises(TypeError):
                function(candidate)
    for function in _privileged_functions():
        with pytest.raises(core.V8BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
            function()


def test_root_mint_symbol_and_direct_capability_classes_do_not_exist_or_help(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not hasattr(core, "TrustedRoots")
    assert not hasattr(core, "VerifiedPolicy")
    assert not hasattr(core, "VerifiedGrant")
    assert not hasattr(core, "_ROOT_MINT")
    # Adding the familiar V7 private symbol after import cannot alter public
    # closures, because their source-pinned gate captured no module global.
    monkeypatch.setattr(core, "_ROOT_MINT", object(), raising=False)
    attacker = _spoofed_root_like_object()
    object.__setattr__(attacker, "_mint", core._ROOT_MINT)
    with pytest.raises(TypeError):
        core.construct_contract(attacker)
    with pytest.raises(core.V8BlockedError, match="BLOCKED_MISSING_ZERO4_TERMINALS"):
        core.construct_contract()


def test_monkeypatch_loader_constants_file_and_status_cannot_activate_captured_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_anchor = tmp_path / "active-anchor.json"
    fake_anchor.write_text('{"status":"ACTIVE"}\n')
    monkeypatch.setattr(core, "PINNED_PRODUCTION_ANCHOR_PATH", fake_anchor, raising=False)
    monkeypatch.setattr(core, "PINNED_PRODUCTION_ANCHOR_SHA256", "0" * 64, raising=False)
    monkeypatch.setattr(core, "_load_pinned_anchor", lambda *args, **kwargs: {"status": "ACTIVE"}, raising=False)
    monkeypatch.setattr(core, "__file__", str(fake_anchor), raising=False)
    monkeypatch.setattr(core, "BLOCKED_STATUS", "ATTACKER_ACTIVE", raising=True)
    for function in _privileged_functions():
        with pytest.raises(core.V8BlockedError, match="^BLOCKED_MISSING_ZERO4_TERMINALS$"):
            function()
    assert core.production_status()["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"


@pytest.mark.parametrize("function", _privileged_functions())
def test_caller_path_key_payload_and_contract_injection_are_type_errors(function) -> None:
    with pytest.raises(TypeError):
        function("/attacker/root")
    with pytest.raises(TypeError):
        function(root_path="/attacker/root", public_key="attacker-key", payload={"status": "ACTIVE"})
    with pytest.raises(TypeError):
        function(policy={"complete": True}, contract={"contract_sha256": "f" * 64})


def test_source_ast_has_no_torch_or_production_capability_object_definitions() -> None:
    tree = ast.parse(Path(core.__file__).read_text())
    imports_torch = any(
        (isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "torch" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
        for node in ast.walk(tree)
    )
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert not imports_torch
    assert {"TrustedRoots", "VerifiedPolicy", "VerifiedGrant"}.isdisjoint(classes)


def test_clean_process_cli_ignores_environment_module_injection_and_rejects_paths(tmp_path: Path) -> None:
    fake_module_root = tmp_path / "attacker_modules"
    fake_module_root.mkdir(); (fake_module_root / "sua_exploration.py").write_text("raise RuntimeError('attacker import')\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(fake_module_root)
    environment["SUBM_ROOT"] = "/attacker/root"
    environment["SUBM_PUBLIC_KEY"] = "attacker-key"
    production_cli = Path(__file__).resolve().parents[1] / "scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py"
    status = subprocess.run([sys.executable, "-I", str(production_cli), "--mode", "status"], cwd=tmp_path, env=environment, text=True, capture_output=True, check=False)
    assert status.returncode == 0
    assert '"status": "BLOCKED_MISSING_ZERO4_TERMINALS"' in status.stdout
    assert "attacker import" not in status.stderr
    formal = subprocess.run([sys.executable, "-I", str(production_cli), "--mode", "formal"], cwd=tmp_path, env=environment, text=True, capture_output=True, check=False)
    assert formal.returncode == 2
    assert "FAIL_CLOSED: BLOCKED_MISSING_ZERO4_TERMINALS" in formal.stderr
    injected = subprocess.run([sys.executable, "-I", str(production_cli), "--mode", "formal", "--root-path", "/attacker/root"], cwd=tmp_path, env=environment, text=True, capture_output=True, check=False)
    assert injected.returncode == 2
    assert "unrecognized arguments" in injected.stderr


def test_blocked_writer_and_wrapper_only_delegate_to_clean_process(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "v8-blocked"
    sealed = writer.write_blocked_prelaunch(output)
    assert sealed["status"] == "BLOCKED_MISSING_ZERO4_TERMINALS"
    assert sealed["v7_p0_not_passed"] is True
    assert sealed["not_an_independent_pass"] is True
    assert writer.load_stored_blocked_prelaunch(output)["verified_grant_created"] is False
    assert runner.main(["--mode", "dry-run", "--prelaunch-dir", str(output)]) == 0
    assert "BLOCKED_MISSING_ZERO4_TERMINALS" in capfd.readouterr().out
    assert runner.main(["--mode", "score", "--prelaunch-dir", str(output)]) == 2


def test_frozen_torchmetrics_variance_weighted_golden_parity_is_pure_synthetic() -> None:
    torch = pytest.importorskip("torch")
    metrics = pytest.importorskip("torchmetrics.regression")
    rng = np.random.default_rng(123)
    ordinary_target = rng.normal(size=(257, 2)).astype(np.float32)
    ordinary_prediction = (ordinary_target + rng.normal(scale=0.25, size=(257, 2))).astype(np.float32)
    constant_target = np.column_stack((np.linspace(-1, 1, 257), np.full(257, 2.0))).astype(np.float32)
    constant_prediction = constant_target.copy(); constant_prediction[:, 1] = -1.0
    near_target = np.column_stack((np.linspace(-1, 1, 257), np.linspace(0.0, 1.0e-5, 257))).astype(np.float32)
    near_prediction = near_target.copy(); near_prediction[:, 1] += np.float32(1.0)
    for prediction, target in ((ordinary_prediction, ordinary_target), (constant_prediction, constant_target), (near_prediction, near_target)):
        observed = core.frozen_torchmetrics_variance_weighted_r2(prediction, target)
        reference = float(metrics.R2Score(multioutput="variance_weighted")(torch.from_numpy(prediction), torch.from_numpy(target)))
        assert abs(observed - reference) <= 1.0e-4
