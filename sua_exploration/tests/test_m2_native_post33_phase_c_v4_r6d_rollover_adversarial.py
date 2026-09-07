"""No-bootstrap adversarial tests for the r6d CPU-only rollover builder.

All mutable paths are pytest-temporary.  The tests never invoke the real
builder entrypoint, create a real r6d receipt/cell root, start a signer, launch
a GPU worker, open data, or read a score.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
ROLLOVER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_rollover.py"

pytestmark = pytest.mark.skipif(not ROLLOVER.is_file(), reason="r6d rollover has not landed")


def _load(label: str):
    name = f"_r6d_rollover_adversarial_{label}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, ROLLOVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _regular(path: Path, data: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


class _SyntheticPreflight:
    """Enough score-free input to reach a selected preflight failure point."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.m = _load("preflight")
        m = self.m
        self.r6d = tmp_path / "r6d_receipts"
        self.cells = tmp_path / "r6d_cells"
        self.data = tmp_path / "data"
        self.data.mkdir()
        monkeypatch.setattr(m, "R6D", self.r6d)
        monkeypatch.setattr(m, "R6D_ROOT", self.cells)
        monkeypatch.setattr(m, "DATA", self.data)
        monkeypatch.setattr(m, "PUBLIC_KEY", tmp_path / "public.pem")
        _regular(m.PUBLIC_KEY)
        # Do not let a test reach production filesystem validation; each test
        # chooses which preflight gate throws first.
        monkeypatch.setattr(m, "require_canonical_regular_file", lambda path, **_kw: Path(path))
        monkeypatch.setattr(m, "_assert_r6c_retirement", lambda: {"synthetic": True})
        monkeypatch.setattr(m, "_assert_runtime_exact_ready", lambda: {"synthetic": True})
        monkeypatch.setattr(m, "_assert_checker_live", lambda: None)
        monkeypatch.setattr(m.authorization, "PUBLIC_KEY", m.PUBLIC_KEY)
        monkeypatch.setattr(m.authorization, "PUBLIC_KEY_SHA256", m.sha256_file(m.PUBLIC_KEY))
        monkeypatch.setattr(m, "_json", lambda _path: {})
        monkeypatch.setattr(m, "build_phase_c_program_receipt", lambda **_kw: {})
        monkeypatch.setattr(m, "_program_delta", lambda *_args: {})

    def assert_targets_absent(self) -> None:
        assert not self.r6d.exists(), "a preflight failure must not create r6d receipt root"
        assert not self.cells.exists(), "a preflight failure must not create r6d cell root"


@pytest.mark.parametrize("failure", ("required_source", "retirement", "runtime", "checker", "program_delta", "scratch"))
def test_r6d_each_preflight_failure_leaves_fresh_receipt_and_cell_roots_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    case = _SyntheticPreflight(monkeypatch, tmp_path)
    m = case.m

    if failure == "required_source":
        monkeypatch.setattr(
            m, "require_canonical_regular_file",
            lambda _path, **_kw: (_ for _ in ()).throw(PermissionError("synthetic source absence")),
        )
    elif failure == "retirement":
        monkeypatch.setattr(
            m, "_assert_r6c_retirement",
            lambda: (_ for _ in ()).throw(PermissionError("synthetic retirement mismatch")),
        )
    elif failure == "runtime":
        monkeypatch.setattr(
            m, "_assert_runtime_exact_ready",
            lambda: (_ for _ in ()).throw(PermissionError("synthetic runtime mismatch")),
        )
    elif failure == "checker":
        monkeypatch.setattr(
            m, "_assert_checker_live",
            lambda: (_ for _ in ()).throw(PermissionError("synthetic checker failure")),
        )
    elif failure == "program_delta":
        monkeypatch.setattr(
            m, "_program_delta",
            lambda *_args: (_ for _ in ()).throw(PermissionError("synthetic source-map delta failure")),
        )
    elif failure == "scratch":
        monkeypatch.setattr(
            m, "_scratch_validate",
            lambda **_kw: (_ for _ in ()).throw(PermissionError("synthetic scratch failure")),
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(failure)

    with pytest.raises(PermissionError):
        m.preflight_r6d()
    case.assert_targets_absent()


@pytest.mark.parametrize("target_name", ("r6d", "cells"))
def test_r6d_preflight_rejects_even_a_dangling_target_root_symlink_before_other_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target_name: str
) -> None:
    """``Path.exists`` alone would miss this fresh-root escape hatch.

    The test makes the *first* later source gate explode if it is reached, so
    acceptance is proved to happen at the root-absence boundary rather than by
    a subsequent incidental failure.  The symlink target is deliberately
    absent; no real r6d path, data, signer, or GPU is touched.
    """
    case = _SyntheticPreflight(monkeypatch, tmp_path)
    target = getattr(case, target_name)
    outside = tmp_path / "outside_missing"
    target.symlink_to(outside)
    monkeypatch.setattr(
        case.m,
        "require_canonical_regular_file",
        lambda _path, **_kw: (_ for _ in ()).throw(AssertionError("root check was bypassed")),
    )

    with pytest.raises(FileExistsError, match="receipt/cell roots must be absent"):
        case.m.preflight_r6d()

    assert target.is_symlink() and not target.exists()
    assert not outside.exists()
    other = case.cells if target_name == "r6d" else case.r6d
    assert not other.exists() and not other.is_symlink()


def test_r6d_scratch_validation_writes_only_disposable_scratch_not_fresh_r6d_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    m = _load("scratch")
    r6d, cells = tmp_path / "fresh_r6d", tmp_path / "fresh_cells"
    monkeypatch.setattr(m, "R6D", r6d)
    monkeypatch.setattr(m, "R6D_ROOT", cells)
    seen: list[Path] = []
    monkeypatch.setattr(m, "validate_phase_c_program_receipt", lambda path: seen.append(Path(path)) or {})
    monkeypatch.setattr(m, "validate_portable_transfer_manifest", lambda path, **_kw: seen.append(Path(path)) or {})
    monkeypatch.setattr(m, "validate_shard_manifest", lambda path, **_kw: seen.append(Path(path)) or {})
    m._scratch_validate(
        program={"synthetic": True},
        portable_template={"hash_closures": []},
        shard_templates={"gpu0": {}, "gpu1": {}, "opening": {}, "stage_b_gpu0": {}, "stage_b_gpu1": {}, "full_opening": {}},
    )
    assert len(seen) == 8  # program + portable + six shard shapes
    assert all("spint_r6d_preflight_" in str(path) for path in seen)
    assert not r6d.exists() and not cells.exists()


def _minimal_plan(m, tmp_path: Path):
    """A syntactically complete plan whose writes remain under temporary R6D."""
    program = {"program": "synthetic"}
    portable = {"portable": "synthetic"}
    shards = {name: {"shard": name} for name in (
        "gpu0", "gpu1", "opening", "stage_b_gpu0", "stage_b_gpu1", "full_opening"
    )}
    authorizations = {name: {"authorization": name} for name in ("gpu0", "gpu1", "opening")}
    launch = {
        "schema": "m2_post33_phase_c_v4_r6d_stage_a_launch_commands_v1",
        "commands": {
            "gpu0": {"supervisor_argv": ["fixed", "--gpu-id", "0"]},
            "gpu1": {"supervisor_argv": ["fixed", "--gpu-id", "1"]},
        },
    }
    core, proof, trigger = ({"core": "synthetic"}, {"proof": "synthetic"}, {"trigger": "synthetic"})
    return m.R6DStaticPlan(program, portable, shards, authorizations, launch, core, proof, trigger)


def test_r6d_build_writes_trigger_last_via_single_atomic_link_and_never_writes_afterward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    m = _load("trigger_last")
    r6d, cells = tmp_path / "r6d", tmp_path / "cells"
    ready = tmp_path / "runtime/live_signer_ready.json"
    _regular(ready)
    monkeypatch.setattr(m, "R6D", r6d)
    monkeypatch.setattr(m, "R6D_ROOT", cells)
    monkeypatch.setattr(m, "READY", ready)
    plan = _minimal_plan(m, tmp_path)
    monkeypatch.setattr(m, "preflight_r6d", lambda: plan)
    original_write = m.write_json_exclusive
    original_publish = m._atomic_publish_json
    events: list[tuple[str, Path]] = []

    def record_write(path: Path, payload):
        events.append(("write", Path(path)))
        return original_write(path, payload)

    def record_publish(path: Path, payload):
        events.append(("trigger", Path(path)))
        return original_publish(path, payload)

    monkeypatch.setattr(m, "write_json_exclusive", record_write)
    monkeypatch.setattr(m, "_atomic_publish_json", record_publish)
    receipt = m.build_r6d_rollover()
    trigger = r6d / "signer_requests/r6d_live_signer_trigger.json"
    assert Path(receipt["trigger"]["canonical_path"]) == trigger.resolve()
    assert events[-1] == ("trigger", trigger)
    assert all(kind == "write" for kind, _path in events[:-1])
    assert not cells.exists()
    with pytest.raises(FileExistsError):
        original_publish(trigger, plan.trigger)
    assert trigger.is_file() and not trigger.is_symlink() and trigger.stat().st_nlink == 1


def test_r6d_launch_has_no_raw_matrix_argv_and_builder_orders_core_proof_before_trigger() -> None:
    source = ROLLOVER.read_text(encoding="utf-8")
    preflight_start = source.index("def preflight_r6d(")
    preflight_body = source[preflight_start:source.index("\ndef build_r6d_rollover(", preflight_start)]
    build_start = source.index("def build_r6d_rollover(")
    build_body = source[build_start:source.index("\ndef main(", build_start)]
    assert '"matrix_argv"' not in preflight_body
    assert '"supervisor_argv"' in preflight_body
    trigger = build_body.index("_atomic_publish_json(R6D / \"signer_requests/r6d_live_signer_trigger.json\"")
    for token in (
        "write_json_exclusive(launch_path, plan.launch)",
        "write_json_exclusive(core_path, plan.core)",
        "write_json_exclusive(proof_path, plan.proof)",
    ):
        assert build_body.index(token) < trigger
    assert "write_json_exclusive" not in build_body[trigger:], "trigger must be the final builder write"


def test_r6d_source_delta_refuses_prebootstrap_authorization_anchor_and_accepts_only_postpivot_delta() -> None:
    """The builder cannot proceed until the verifier source has moved to r6d."""
    m = _load("anchor_delta")
    r6c_roots = {
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py",
    }
    r6d_roots = {path.replace("r6c", "r6d") for path in r6c_roots}
    authorization = "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py"
    program = "sua_exploration/mc_maze/m2_native_post33_program_v4.py"

    def row(path: str, version: str) -> dict[str, str]:
        return {"relative_path": path, "sha256": version}

    semantic = {
        "schema": "same", "protocol_id": "same", "phase_id": "same",
        "absolute_workspace_root": "same", "score_data_accessed": False,
        "formal_data_accessed": False, "gpu_used": False,
        "phase_a_b_eof_canonicalization": "same", "deep_source_audit_receipt": "same",
    }
    before = {
        **semantic,
        "source_map": [*(row(path, "r6c") for path in sorted(r6c_roots)), row(authorization, "r6c-anchor"), row(program, "r6c-program")],
    }
    prebootstrap_after = {
        **semantic,
        "source_map": [*(row(path, "r6d") for path in sorted(r6d_roots)), row(authorization, "r6c-anchor"), row(program, "r6d-program")],
    }
    with pytest.raises(ValueError, match="unexpected r6d source delta"):
        m._program_delta(before, prebootstrap_after)

    postpivot_after = {
        **semantic,
        "source_map": [*(row(path, "r6d") for path in sorted(r6d_roots)), row(authorization, "r6d-anchor"), row(program, "r6d-program")],
    }
    result = m._program_delta(before, postpivot_after)
    assert result["changed_governance_sources"] == [authorization, program]


def test_r6d_atomic_trigger_publish_is_exclusive_on_temp_path(tmp_path: Path) -> None:
    m = _load("atomic_publish")
    target = tmp_path / "trigger.json"
    first = m._atomic_publish_json(target, {"trigger": "first"})
    assert first == target.resolve()
    with pytest.raises(FileExistsError):
        m._atomic_publish_json(target, {"trigger": "second"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"trigger": "first"}


def test_r6d_synthetic_preflight_constructs_an_exact_core_proof_trigger_metadata_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise real plan construction with all dependencies made score-free/temp."""
    m = _load("exact_chain")
    r6d, cells = tmp_path / "r6d", tmp_path / "cells"
    data = tmp_path / "data"
    data.mkdir()
    public, ready, cost = tmp_path / "public.pem", tmp_path / "runtime/ready.json", tmp_path / "cost.json"
    for path in (public, ready, cost):
        _regular(path)
    monkeypatch.setattr(m, "R6D", r6d)
    monkeypatch.setattr(m, "R6D_ROOT", cells)
    monkeypatch.setattr(m, "DATA", data)
    monkeypatch.setattr(m, "PUBLIC_KEY", public)
    monkeypatch.setattr(m, "READY", ready)
    monkeypatch.setattr(m, "COST", cost)
    monkeypatch.setattr(m, "require_canonical_regular_file", lambda path, **_kw: Path(path))
    monkeypatch.setattr(m, "_assert_r6c_retirement", lambda: {"synthetic": True})
    monkeypatch.setattr(m, "_assert_runtime_exact_ready", lambda: {"synthetic": True})
    monkeypatch.setattr(m, "_assert_checker_live", lambda: None)
    monkeypatch.setattr(m.authorization, "PUBLIC_KEY", public)
    monkeypatch.setattr(m.authorization, "PUBLIC_KEY_SHA256", m.sha256_file(public))
    monkeypatch.setattr(m, "_program_delta", lambda *_args: {"synthetic": True})
    monkeypatch.setattr(m, "_scratch_validate", lambda **_kw: None)
    portable_template = {"hash_closures": []}
    shard_template = {"fold_allowlist": [0], "gpu_id": "GPU-synthetic"}
    auth_template = {"authorization": {"authorization_id": "synthetic-r6c-template"}}
    def fake_json(path: Path):
        name = Path(path).name
        if name.startswith("portable"):
            return portable_template
        if name.startswith("shard"):
            return shard_template
        if name.startswith("stage_a_execution") or name.startswith("stage_a_opening"):
            return auth_template
        return {"source_map": []}
    monkeypatch.setattr(m, "_json", fake_json)
    monkeypatch.setattr(m, "build_phase_c_program_receipt", lambda **_kw: {"source_map": []})

    plan = m.preflight_r6d()
    paths = {
        "program": r6d / "program/phase_c_program_r6d.json",
        "portable": r6d / "manifest/portable_r6d.json",
        "launch": r6d / "launch/stage_a_commands_r6d.json",
        "core": r6d / "signer_requests/r6d_live_signer_plan_core.json",
        "proof": r6d / "r6d_static_pretrigger_evidence.json",
    }
    program_meta = m._predicted_metadata(paths["program"], plan.program)
    portable_meta = m._predicted_metadata(paths["portable"], plan.portable)
    launch_meta = m._predicted_metadata(paths["launch"], plan.launch)
    core_meta = m._predicted_metadata(paths["core"], plan.core)
    proof_meta = m._predicted_metadata(paths["proof"], plan.proof)
    assert plan.core["program"] == program_meta
    assert plan.core["portable"] == portable_meta
    assert plan.core["stage_a_launch"] == launch_meta
    assert plan.proof["plan_core"] == core_meta and plan.proof["launch"] == launch_meta
    assert plan.proof["program"] == program_meta and plan.proof["portable"] == portable_meta
    assert plan.proof["builder_source"] == m.file_metadata(m.BUILDER)
    assert plan.trigger["plan_core"] == core_meta and plan.trigger["pretrigger_proof"] == proof_meta
    assert plan.trigger["stage_a_launch"] == launch_meta
    assert all("matrix_argv" not in row for row in plan.launch["commands"].values())
    assert not r6d.exists() and not cells.exists(), "preflight may construct but must not write target roots"
