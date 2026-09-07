"""Pure-temp adversarial tests for r6d trigger -> core -> proof sealing.

These tests import the controller but never call ``bootstrap``/``main``.  Every
receipt, authorization, public anchor, and attempted mutation lives beneath a
pytest temporary directory; no r6d root, data, score, endpoint, CUDA API, or
subprocess is opened.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py"
SUPERVISOR = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py"

requires_controller = pytest.mark.skipif(
    not CONTROLLER.is_file(), reason="r6d controller has not landed"
)
requires_supervisor = pytest.mark.skipif(
    not SUPERVISOR.is_file(), reason="r6d supervisor has not landed"
)


def _load(path: Path, label: str):
    name = f"_r6d_trigger_chain_adversarial_{label}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
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


def _source_row(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "relative_path": resolved.relative_to(ROOT.resolve(strict=True)).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


class _SyntheticChain:
    """A complete valid r6d control chain rooted only in ``tmp_path``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.m = _load(CONTROLLER, "chain")
        self.base = tmp_path / "r6d_temp"
        self.receipts = self.base / "receipts"
        self.cells = self.base / "cells"  # Intentionally never created.
        self.runtime = self.base / "runtime"
        self.receipts.mkdir(parents=True)
        self.runtime.mkdir()

        m = self.m
        monkeypatch.setattr(m, "R6D_RECEIPTS", self.receipts)
        monkeypatch.setattr(m, "R6D_CELL_ROOT", self.cells)
        monkeypatch.setattr(m, "RUNTIME", self.runtime)
        monkeypatch.setattr(m, "PUBLIC_KEY", self.base / "public.pem")
        monkeypatch.setattr(m, "READY", self.runtime / "live_signer_ready.json")
        monkeypatch.setattr(m, "TRIGGER", self.receipts / "signer_requests/r6d_live_signer_trigger.json")
        monkeypatch.setattr(m, "CORE", self.receipts / "signer_requests/r6d_live_signer_plan_core.json")
        monkeypatch.setattr(m, "PROOF", self.receipts / "r6d_static_pretrigger_evidence.json")
        monkeypatch.setattr(m, "LAUNCH", self.receipts / "launch/stage_a_commands_r6d.json")
        monkeypatch.setattr(m, "COST", self.base / "cost.json")
        monkeypatch.setattr(m, "R6C_RETIREMENT", self.base / "r6c_retirement.json")

        for path in (m.PUBLIC_KEY, m.READY, m.PROGRAM if hasattr(m, "PROGRAM") else self.receipts / "program/phase_c_program_r6d.json", m.COST, m.R6C_RETIREMENT):
            _regular(path)
        # The controller derives these two paths rather than exposing constants.
        self.program = self.receipts / "program/phase_c_program_r6d.json"
        self.portable = self.receipts / "manifest/portable_r6d.json"
        _regular(self.program)
        _regular(self.portable)
        _regular(m.LAUNCH)

        # Four runtime roots remain real, source-only files under the workspace;
        # their hashes are used exactly as the controller will use them.
        self.sources = [Path(m.__file__).resolve(), m.GATE.resolve(), m.SUPERVISOR.resolve(), m.ASSERT_LIVE.resolve()]
        self.program_payload = {"source_map": [_source_row(source) for source in self.sources]}
        monkeypatch.setattr(
            m.program_module, "validate_phase_c_program_receipt", lambda _path: self.program_payload
        )
        monkeypatch.setattr(m.authorization, "PUBLIC_KEY", m.PUBLIC_KEY)
        monkeypatch.setattr(m.authorization, "PUBLIC_KEY_SHA256", m.sha256_file(m.PUBLIC_KEY))

        # Unsigned Stage-A bodies provide the nonce inventory proof.  Their
        # matching signatures and all conditional authorization files stay absent.
        nonces = ["1" * 64, "2" * 64, "3" * 64]
        self.stage_auth: dict[str, Path] = {}
        for (name, (auth, _shard)), nonce in zip(m._expected_stage_a_paths().items(), nonces):
            _json(auth, {"authorization": {"single_use_nonce": nonce}})
            self.stage_auth[name] = auth

        issued = datetime.now(timezone.utc) - timedelta(minutes=1)
        expires = datetime.now(timezone.utc) + timedelta(minutes=1)
        self.core = {
            "schema": "m2_post33_phase_c_v4_r6d_signer_plan_core_v1",
            "receipt_root": str(self.receipts.resolve()),
            "cell_root": str(self.cells.resolve()),
            "ready": m.file_metadata(m.READY),
            "public_key": m.file_metadata(m.PUBLIC_KEY),
            "program": m.file_metadata(self.program),
            "portable": m.file_metadata(self.portable),
            "cost_supplement": m.file_metadata(m.COST),
            "controller_source": m.file_metadata(Path(m.__file__).resolve()),
            "continue_gate_source": m.file_metadata(m.GATE),
            "supervisor_source": m.file_metadata(m.SUPERVISOR),
            "assert_live_source": m.file_metadata(m.ASSERT_LIVE),
            "stage_a_launch": m.file_metadata(m.LAUNCH),
            "issued_at": issued.replace(microsecond=0).isoformat(),
            "expires_at": expires.replace(microsecond=0).isoformat(),
            "stage_a": {name: {} for name in ("gpu0", "gpu1", "opening")},
            "conditional_targets": {name: {} for name in ("gpu0", "gpu1", "full_opening")},
        }
        _json(m.CORE, self.core)

        self.proof = {
            "schema": "m2_post33_phase_c_v4_r6d_static_pretrigger_evidence_v1",
            "r6c_retirement": m.file_metadata(m.R6C_RETIREMENT),
            "r6d_fresh_receipt_root": str(self.receipts.resolve()),
            "r6d_fresh_cell_root": str(self.cells.resolve()),
            "plan_core": m.file_metadata(m.CORE),
            "launch": m.file_metadata(m.LAUNCH),
            "program": self.core["program"],
            "portable": self.core["portable"],
            "runtime_sources": {
                "controller": self.core["controller_source"],
                "continue_gate": self.core["continue_gate_source"],
                "supervisor": self.core["supervisor_source"],
                "checker": self.core["assert_live_source"],
            },
            # Builder provenance is bound by the proof, but intentionally is
            # not an execution-path source-map root.
            "builder_source": m.file_metadata(m.ROLLOVER),
            "stage_a_unsigned_authorizations": {
                name: m.file_metadata(path) for name, path in self.stage_auth.items()
            },
            "stage_a_nonce_inventory": {
                "nonces": nonces,
                "count": 3,
                "pairwise_distinct": True,
                "all_lowercase_256_bit_hex": True,
            },
            "stage_a_signature_absence": {
                name: {
                    "signature_path": str(path.with_suffix(".sig").resolve()),
                    "present_at_pretrigger": False,
                }
                for name, path in self.stage_auth.items()
            },
            "conditional_authorization_absence": {
                name: {
                    "authorization_path": str(auth.resolve()),
                    "authorization_present_at_pretrigger": False,
                    "signature_path": str(auth.with_suffix(".sig").resolve()),
                    "signature_present_at_pretrigger": False,
                }
                for name, (auth, _shard) in m._conditional_paths().items()
            },
            "gpu_used": False,
            "score_data_accessed": False,
            "formal_data_accessed": False,
        }
        _json(m.PROOF, self.proof)
        self.trigger = {
            "schema": "m2_post33_phase_c_v4_r6d_live_signer_trigger_v1",
            "receipt_root": str(self.receipts.resolve()),
            "cell_root": str(self.cells.resolve()),
            **{field: self.core[field] for field in (
                "ready", "public_key", "program", "portable", "cost_supplement",
                "controller_source", "continue_gate_source", "supervisor_source",
                "assert_live_source", "stage_a_launch",
            )},
            "plan_core": m.file_metadata(m.CORE),
            "pretrigger_proof": m.file_metadata(m.PROOF),
        }
        _json(m.TRIGGER, self.trigger)


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _SyntheticChain:
    if not CONTROLLER.is_file():
        pytest.skip("r6d controller has not landed")
    return _SyntheticChain(monkeypatch, tmp_path)


def _mutated_metadata(value: object) -> object:
    if not isinstance(value, dict):
        raise AssertionError("expected metadata mapping")
    result = copy.deepcopy(value)
    result["sha256"] = "0" * 64
    return result


def test_r6d_pristine_temp_trigger_core_proof_chain_is_accepted(chain: _SyntheticChain) -> None:
    paths, core = chain.m._validate_trigger(copy.deepcopy(chain.trigger))
    assert paths["program"] == chain.program.resolve()
    assert paths["portable"] == chain.portable.resolve()
    assert core["schema"] == chain.core["schema"]


@pytest.mark.parametrize(
    "field",
    (
        "ready", "public_key", "program", "portable", "cost_supplement",
        "controller_source", "continue_gate_source", "supervisor_source",
        "assert_live_source", "stage_a_launch",
    ),
)
def test_r6d_core_metadata_mutation_is_rejected_before_signing(chain: _SyntheticChain, field: str) -> None:
    bad_core = copy.deepcopy(chain.core)
    bad_core[field] = _mutated_metadata(bad_core[field])
    with pytest.raises(PermissionError):
        chain.m._validate_core(bad_core, chain.trigger)


@pytest.mark.parametrize(
    "field",
    (
        "ready", "public_key", "program", "portable", "cost_supplement",
        "controller_source", "continue_gate_source", "supervisor_source",
        "assert_live_source", "stage_a_launch", "plan_core", "pretrigger_proof",
    ),
)
def test_r6d_trigger_metadata_mutation_is_rejected_before_signing(chain: _SyntheticChain, field: str) -> None:
    bad_trigger = copy.deepcopy(chain.trigger)
    bad_trigger[field] = _mutated_metadata(bad_trigger[field])
    with pytest.raises(PermissionError):
        chain.m._validate_trigger(bad_trigger)


@pytest.mark.parametrize(
    "field",
    ("r6c_retirement", "plan_core", "launch", "program", "portable", "builder_source"),
)
def test_r6d_proof_root_metadata_mutation_is_rejected(chain: _SyntheticChain, field: str) -> None:
    bad_proof = copy.deepcopy(chain.proof)
    bad_proof[field] = _mutated_metadata(bad_proof[field])
    with pytest.raises(PermissionError):
        chain.m._validate_proof(bad_proof, trigger=chain.trigger, core=chain.core)


@pytest.mark.parametrize("source_key", ("controller", "continue_gate", "supervisor", "checker"))
def test_r6d_proof_runtime_source_metadata_mutation_is_rejected(
    chain: _SyntheticChain, source_key: str
) -> None:
    bad_proof = copy.deepcopy(chain.proof)
    bad_proof["runtime_sources"][source_key] = _mutated_metadata(
        bad_proof["runtime_sources"][source_key]
    )
    with pytest.raises(PermissionError):
        chain.m._validate_proof(bad_proof, trigger=chain.trigger, core=chain.core)


def test_r6d_program_source_map_rejects_omission_of_each_runtime_root(chain: _SyntheticChain, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trigger cannot bless a controller/gate/checker/supervisor source gap."""
    for missing in chain.sources:
        bad_map = {
            "source_map": [row for row in chain.program_payload["source_map"] if row != _source_row(missing)]
        }
        monkeypatch.setattr(
            chain.m.program_module, "validate_phase_c_program_receipt", lambda _path, payload=bad_map: payload
        )
        with pytest.raises(PermissionError, match="source map omits"):
            chain.m._validate_trigger(copy.deepcopy(chain.trigger))


def test_r6d_preexisting_stage_a_signature_rejects_trigger_chain(chain: _SyntheticChain) -> None:
    signature = chain.stage_auth["gpu0"].with_suffix(".sig")
    _regular(signature, "not-a-signature\n")
    with pytest.raises(PermissionError, match="signature is present"):
        chain.m._validate_proof(copy.deepcopy(chain.proof), trigger=chain.trigger, core=chain.core)


def test_r6d_preexisting_conditional_authorization_rejects_trigger_chain(chain: _SyntheticChain) -> None:
    conditional, _shard = chain.m._conditional_paths()["gpu0"]
    _json(conditional, {"authorization": {"single_use_nonce": "f" * 64}})
    with pytest.raises(PermissionError, match="conditional capability materialized"):
        chain.m._validate_proof(copy.deepcopy(chain.proof), trigger=chain.trigger, core=chain.core)


def test_r6d_dangling_cell_root_symlink_rejects_trigger_chain(chain: _SyntheticChain, tmp_path: Path) -> None:
    """A non-existent symlink target is still an occupied claim surface."""
    chain.cells.symlink_to(tmp_path / "outside_missing")
    with pytest.raises(PermissionError, match="cell root/claim surface"):
        chain.m._assert_proof_absence(copy.deepcopy(chain.proof))


@pytest.mark.parametrize("path_name", ("TRIGGER", "CORE", "PROOF", "LAUNCH"))
def test_r6d_final_pre_sign_reread_rejects_each_replaced_control_artifact(
    chain: _SyntheticChain, monkeypatch: pytest.MonkeyPatch, path_name: str
) -> None:
    """An artifact replaced after first validation cannot reach Stage-A signing."""
    chain.m._validate_trigger(copy.deepcopy(chain.trigger))
    target = getattr(chain.m, path_name)
    replacement = json.loads(target.read_text(encoding="utf-8"))
    replacement["adversarial_replacement_after_first_validation"] = True
    _json(target, replacement)
    monkeypatch.setattr(chain.m, "_assert_live_binding", lambda _ready: None)
    with pytest.raises(PermissionError):
        chain.m._revalidate_immediately_before_stage_a_sign({})


@requires_controller
def test_r6d_stage_a_ack_explicitly_binds_trigger_core_proof_launch_and_runtime_sources() -> None:
    """Transitive hashes are useful, but the live ACK must expose this closure."""
    source = CONTROLLER.read_text(encoding="utf-8")
    start = source.index("def _sign_stage_a(")
    end = source.index("\ndef _wait_for(", start)
    ack_writer = source[start:end]
    for token in (
        '"trigger": file_metadata(TRIGGER)',
        '"plan_core": file_metadata(CORE)',
        '"pretrigger_proof": file_metadata(PROOF)',
        '"launch": file_metadata(LAUNCH)',
        '"runtime_sources"',
    ):
        assert token in ack_writer, f"Stage-A ACK omits required direct binding: {token}"


@requires_supervisor
def test_r6d_supervisor_revalidates_all_stage_a_ack_preflight_bindings() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    start = source.index("def _verify_stage_a_ack(")
    end = source.index("\ndef _check_once(", start)
    verifier = source[start:end]
    for token in ("trigger", "plan_core", "pretrigger_proof", "launch", "runtime_sources"):
        assert token in verifier, f"supervisor ACK verifier omits required direct binding: {token}"


class _SyntheticAck:
    """Temporary-only Stage-A ACK and its full consumer-side closure."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.m = _load(SUPERVISOR, "ack_consumer")
        m = self.m
        base = tmp_path / "ack_temp"
        r6d = base / "receipts"
        monkeypatch.setattr(m, "R6D", r6d)
        monkeypatch.setattr(m, "READY", base / "ready.json")
        monkeypatch.setattr(m, "PROGRAM", base / "program.json")
        monkeypatch.setattr(m, "TRIGGER", base / "trigger.json")
        monkeypatch.setattr(m, "CORE", base / "core.json")
        monkeypatch.setattr(m, "PROOF", base / "proof.json")
        monkeypatch.setattr(m, "LAUNCH", base / "launch.json")
        for path in (m.READY, m.PROGRAM, m.TRIGGER, m.CORE, m.PROOF, m.LAUNCH):
            _regular(path)
        self.auth, self.signature = base / "auth.json", base / "auth.sig"
        _regular(self.auth)
        _regular(self.signature, "signature\n")
        self.paths = {"authorization": self.auth, "signature": self.signature}
        self.runtime_sources = {
            "controller": m.file_metadata(m.CONTROLLER),
            "continue_gate": m.file_metadata(m.GATE),
            "supervisor": m.file_metadata(Path(m.__file__).resolve()),
            "checker": m.file_metadata(m.ASSERT_LIVE),
        }
        self.payload = {
            "schema": "m2_post33_phase_c_v4_r6d_stage_a_capabilities_signed_v1",
            "ready": m.file_metadata(m.READY),
            "program": m.file_metadata(m.PROGRAM),
            "trigger": m.file_metadata(m.TRIGGER),
            "plan_core": m.file_metadata(m.CORE),
            "pretrigger_proof": m.file_metadata(m.PROOF),
            "launch": m.file_metadata(m.LAUNCH),
            "runtime_sources": self.runtime_sources,
            "stage_a": {
                "gpu0": {
                    "authorization": m.file_metadata(self.auth),
                    "signature": m.file_metadata(self.signature),
                }
            },
        }
        self.ack_path = r6d / "signer_acks/stage_a_capabilities_signed.json"
        _json(self.ack_path, self.payload)

    def write(self, payload: dict[str, object]) -> None:
        _json(self.ack_path, payload)


@pytest.fixture
def ack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _SyntheticAck:
    if not SUPERVISOR.is_file():
        pytest.skip("r6d supervisor has not landed")
    return _SyntheticAck(monkeypatch, tmp_path)


def test_r6d_supervisor_accepts_pristine_temp_ack_closure(ack: _SyntheticAck) -> None:
    ack.m._verify_stage_a_ack(0, ack.paths)


@pytest.mark.parametrize("field", ("trigger", "plan_core", "pretrigger_proof", "launch"))
def test_r6d_supervisor_rejects_each_mutated_ack_preflight_metadata(
    ack: _SyntheticAck, field: str
) -> None:
    bad = copy.deepcopy(ack.payload)
    bad[field] = _mutated_metadata(bad[field])
    ack.write(bad)
    with pytest.raises(PermissionError, match="control-chain binding mismatch"):
        ack.m._verify_stage_a_ack(0, ack.paths)


@pytest.mark.parametrize("source_key", ("controller", "continue_gate", "supervisor", "checker"))
def test_r6d_supervisor_rejects_each_mutated_ack_runtime_source_metadata(
    ack: _SyntheticAck, source_key: str
) -> None:
    bad = copy.deepcopy(ack.payload)
    bad["runtime_sources"][source_key] = _mutated_metadata(bad["runtime_sources"][source_key])
    ack.write(bad)
    with pytest.raises(PermissionError, match="control-chain binding mismatch"):
        ack.m._verify_stage_a_ack(0, ack.paths)
