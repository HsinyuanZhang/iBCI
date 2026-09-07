from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))
from src.tfsr_b3st4_ddrop_v1 import source_smoke as smoke
from src.tfsr_b3st4_ddrop_v1.contract import compute_live_closure


def _batch(n: int = 3, session: str = "source-a"):
    torch.manual_seed(4)
    side = torch.randn(n, 4)
    record = SimpleNamespace(side_features=side.numpy(), channel_ids=list(range(10, 10 + n)))
    batch = (torch.randn(32, 50, n), torch.randn(32, 50, 2), torch.randn(32, 30, 100, n), [session] * 32,
             side.unsqueeze(0).expand(32, -1, -1).clone())
    return batch, record


def _env(extra: str = ""):
    value = os.environ.copy()
    value.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                  "PYTHONPATH": ":".join(x for x in (extra, str(ROOT / "tfpd_exploration"), str(ROOT / "tfpd_exploration/src")) if x)})
    return value


def _exact_receipt() -> dict[str, object]:
    """A sealed-authority, synthetic-metrics receipt for validator/writer tests."""
    authorities = smoke.verify_canonical_source_authorities(ROOT)
    theta = smoke.load_verified_theta_artifact(ROOT, authorities)
    authorities = dict(authorities)
    authorities["theta_raw_t4_sha256"] = {name: theta["authority"][name]["raw_t4_sha256"] for name in authorities["roster"]}
    sessions = []
    for name in authorities["roster"]:
        row = authorities["source_lineage"]["rows_by_session"][name]
        sessions.append({"session": name, "source_path": row["path"], "source_size_bytes": row["size_bytes"],
                         "source_sha256": row["sha256"], "unit_count": row["unit_count"], "unit_order_digest": "a" * 64,
                         "raw_t4_sha256": authorities["theta_raw_t4_sha256"][name], "normalized_t4_sha256": authorities["normalized_t4_sha256"][name]})
    audit = {"roster": list(authorities["roster"]), "roster_digest": smoke._digest_json(tuple(authorities["roster"])), "sessions": sessions,
             "manifest_sha256": smoke.MANIFEST_SHA, "n_train_windows": 1086007, "steps_per_epoch": 33925, "batch_size": 32,
             "seed": 42, "side_normalizer_semantic_sha256": smoke.SIDE_SEMANTIC_SHA,
             "behavior_normalizer_semantic_sha256": smoke.BEHAVIOR_SEMANTIC_SHA,
             "m30_t4_support": "chronological_first_30_rewarded_trials"}
    closure = {**smoke.verify_stage0_and_phase_c_closures(ROOT), "authorities": authorities}
    first = sessions[0]
    return {"schema": "tfsr_b3st4_ddrop_source_smoke_v1", "status": "ENGINEERING_SOURCE_ONLY_SMOKE", "cell": smoke.CELL,
            "source_smoke": dict(smoke.FROZEN_SMOKE), "source_data_opened": True, "target_data_opened": False,
            "validation_data_opened": False, "formal_data_opened": False, "target_or_formal_opened": False,
            "target_optimizer_steps": 0, "scientific_result": False, "score": False, "authorizes_48_epoch": False,
            "environment": {"python_executable": "synthetic-python", "python": "synthetic-python-version", "torch": "synthetic-torch", "cuda": "synthetic-cuda", "cudnn": "synthetic-cudnn"},
            "device": dict(smoke.FROZEN_DEVICE), "primary_batch": {"session": first["session"], "units": first["unit_count"],
                "shapes": {"neural": [32, 50, first["unit_count"]], "behavior": [32, 50, 2], "calib": [32, 30, 100, first["unit_count"]], "side": [32, first["unit_count"], 4]},
                "unit_order_digest": first["unit_order_digest"], "raw_t4_sha256": first["raw_t4_sha256"], "normalized_t4_sha256": first["normalized_t4_sha256"]},
            "optimizer": dict(smoke.FROZEN_OPTIMIZER), "capture_diagnostics": False, "loss": 1.0,
            "critical_gradients": {name: True for name in smoke._CRITICAL_PARAMETER_PREFIXES}, "wall_seconds": 0.01,
            "rss_bytes": 1, "peak_allocated_bytes": 1, "peak_reserved_bytes": 1, "initial_state_digest": "b" * 64,
            "post_step_state_digest": "c" * 64, "dropout": {"p": 0.5, "gain": [[2.0] * first["unit_count"] for _ in range(32)], "survivors": [[True] * first["unit_count"] for _ in range(32)]},
            "launch_final_closure_equal": True, "launch_closure": closure, "final_closure": closure,
            "authorities": authorities, "live_source_audit": audit}


def test_default_plan_is_static_and_phase_c_closure_bound():
    plan = smoke.dry_plan(ROOT)
    assert plan["status"] == "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH"
    assert plan["source_smoke_only"]["epochs_authorized"] == 0
    assert plan["closures"]["phase_c"]["paths"] == list(smoke.PHASE_C_CLOSURE)
    assert not (ROOT / smoke.CANONICAL_RECEIPT_RELATIVE_PATH).exists()


def test_sealed_authority_join_and_normalized_hash_domain_are_explicit():
    authority = smoke.verify_canonical_source_authorities(ROOT)
    artifact = smoke.load_verified_theta_artifact(ROOT, authority)
    assert set(artifact["authority"]) == set(authority["roster"])
    signed_zero = torch.tensor([[-0.0, 1.0]], dtype=torch.float32)
    positive_zero = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    assert smoke.normalized_t4_authority_sha256(signed_zero) == smoke.normalized_t4_authority_sha256(positive_zero)
    assert smoke.raw_tensor_bytes_sha256(signed_zero) != smoke.raw_tensor_bytes_sha256(positive_zero)


def test_cli_flags_reject_before_torch_and_authorized_pair_stops_at_named_binding_blocker(tmp_path: Path):
    (tmp_path / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_source_smoke.py"
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=_env(str(tmp_path)), capture_output=True, text=True)
    assert dry.returncode == 0 and json.loads(dry.stdout)["authorization"] == "none"
    for args in (("--execute",), ("--i-have-source-smoke-authorization",), ("--unknown",), ("--execute", "--execute")):
        done = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=_env(str(tmp_path)), capture_output=True, text=True)
        assert done.returncode != 0 and "TORCH_IMPORTED" not in done.stderr + done.stdout


@pytest.mark.parametrize("mutator", ("mixed", "shape", "side", "session"))
def test_batch_validation_rejects_sampler_roster_shape_and_unit_order_drift(mutator: str):
    batch, record = _batch()
    values = list(batch)
    if mutator == "mixed": values[3] = ["source-a"] * 31 + ["source-b"]
    elif mutator == "shape": values[0] = values[0][:, :-1]
    elif mutator == "side": values[4][:, 0, 0] += 1
    else: values[3] = ["not-source"] * 32
    with pytest.raises(RuntimeError):
        smoke.validate_source_batch(tuple(values), record, {"source-a"})


def test_verified_side_capability_rejects_bare_or_mismatched_t4_and_never_renormalizes():
    batch, record = _batch()
    info = smoke.validate_source_batch(batch, record, {"source-a"})
    capability = smoke.capability_from_verified_side(info, batch[4], raw_authority_sha256="a" * 64,
        normalizer_authority_sha256="b" * 64, roster_digest="c" * 64, lineage=("m30",))
    assert torch.equal(capability.tensor, batch[4]) and info["side_was_renormalized"] is False
    with pytest.raises(RuntimeError):
        smoke.capability_from_verified_side({}, batch[4], raw_authority_sha256="a" * 64,
            normalizer_authority_sha256="b" * 64, roster_digest="c" * 64, lineage=("m30",))
    changed = batch[4].clone(); changed[:, 0, 0] += 1
    with pytest.raises(RuntimeError):
        smoke.capability_from_verified_side(info, changed, raw_authority_sha256="a" * 64,
            normalizer_authority_sha256="b" * 64, roster_digest="c" * 64, lineage=("m30",))


def test_phase_c_closure_drift_and_transactional_receipt_adversaries(tmp_path: Path, monkeypatch):
    for relative in smoke.PHASE_C_CLOSURE:
        target = tmp_path / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / relative, target)
    closure = compute_live_closure(tmp_path, smoke.PHASE_C_CLOSURE)
    (tmp_path / smoke.PHASE_C_CLOSURE[0]).write_bytes((tmp_path / smoke.PHASE_C_CLOSURE[0]).read_bytes() + b" ")
    assert compute_live_closure(tmp_path, smoke.PHASE_C_CLOSURE)["closure_sha256"] != closure["closure_sha256"]
    path = tmp_path / "receipt.json"
    payload = _exact_receipt()
    smoke.validate_smoke_receipt(payload); digest = smoke.write_smoke_receipt_transactionally(path, payload)
    assert len(digest) == 64 and oct(path.stat().st_mode & 0o777) == "0o444"
    with pytest.raises(RuntimeError): smoke.write_smoke_receipt_transactionally(path, payload)
    bad = dict(payload); bad["score"] = True
    with pytest.raises(RuntimeError): smoke.validate_smoke_receipt(bad)
    smoke.require_launch_final_closure({"a": 1}, {"a": 1})
    with pytest.raises(RuntimeError): smoke.require_launch_final_closure({"a": 1}, {"a": 2})


def test_preexecution_gate_prevents_any_authorized_import_path_on_output_conflict(tmp_path: Path):
    canonical = tmp_path / smoke.CANONICAL_RECEIPT_RELATIVE_PATH
    canonical.parent.mkdir(parents=True)
    canonical.write_text("old")
    with pytest.raises(RuntimeError):
        smoke.pre_execution_output_gate(tmp_path)
    canonical.unlink()
    canonical.symlink_to(tmp_path / "missing")
    with pytest.raises(RuntimeError):
        smoke.pre_execution_output_gate(tmp_path)
    canonical.unlink()
    canonical.parent.unlink() if canonical.parent.is_symlink() else None
    assert smoke.pre_execution_output_gate(tmp_path) == canonical


def test_transactional_writer_rejects_links_and_rolls_back_only_its_owned_inode(tmp_path: Path, monkeypatch):
    payload = _exact_receipt()
    link = tmp_path / "linked.json"; link.symlink_to(tmp_path / "other.json")
    with pytest.raises(RuntimeError): smoke.write_smoke_receipt_transactionally(link, payload)
    link.unlink()
    calls = {"n": 0}
    original = smoke._full_write
    def fail_second(fd, contents):
        calls["n"] += 1
        if calls["n"] == 2: raise OSError("synthetic second-link write failure")
        return original(fd, contents)
    monkeypatch.setattr(smoke, "_full_write", fail_second)
    path = tmp_path / "rollback.json"
    with pytest.raises(OSError): smoke.write_smoke_receipt_transactionally(path, payload)
    assert not path.exists() and not Path(str(path) + ".sha256").exists()


def test_authorized_cli_is_wired_through_static_pre_gate_executor_and_publisher(monkeypatch):
    """Exercise the public branch using only a fake static module; no data/GPU import occurs."""
    script = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_source_smoke.py"
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script)]
        spec = importlib.util.spec_from_file_location("mocked_tfsr_source_cli", script)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    calls: list[str] = []
    payload = _exact_receipt()
    fake = SimpleNamespace(
        pre_execution_output_gate=lambda root: calls.append("gate"),
        verify_canonical_source_authorities=lambda root: calls.append("authority") or {"roster": []},
        build_train_only_adapter=lambda root, authority: calls.append("adapter") or object(),
        execute_one_step=lambda adapter, root: calls.append("execute") or payload,
        validate_smoke_receipt=lambda value: calls.append("validate"),
        write_smoke_receipt_transactionally=lambda path, value: calls.append("publish") or "a" * 64,
        CANONICAL_RECEIPT_RELATIVE_PATH="tfpd_exploration/results/tfsr_b3st4_ddrop_v1/source_smoke_receipt.json",
    )
    monkeypatch.setattr(module, "_load_static_source_smoke", lambda: fake)
    monkeypatch.setattr(module, "_provided", set(module._AUTHORIZED))
    module.main()
    assert calls == ["gate", "authority", "gate", "adapter", "execute", "validate", "gate", "publish"]


def test_theta_artifact_authority_row_tampering_is_rejected_directly(monkeypatch):
    receipt = {"authority_sha256": "d" * 64,
               "alignment_proof": {"source-a": {"aligned": True, "theta_units": 2, "side_rows": 2,
                                                    "datamodule_channels": 2, "n_undefined": 0, "n_valid_directions": 8}}}
    buffer = io.BytesIO()
    torch.save({"kind": "tfpd_sparsification_theta_authority_v1", "authority_sha256": "d" * 64,
                "authority": {"source-a": {"theta": torch.ones(2), "valid": torch.ones(2, dtype=torch.bool),
                                             "n_units": 2, "raw_t4_sha256": "not-a-sha"}}}, buffer)
    monkeypatch.setattr(smoke, "_sealed", lambda root, relative, expected: (
        buffer.getvalue() if relative == smoke.THETA_ARTIFACT else json.dumps(receipt).encode()))
    with pytest.raises(RuntimeError, match="row type"):
        smoke.load_verified_theta_artifact(ROOT, {"roster": ["source-a"]})


def test_device_identity_and_each_critical_gradient_group_fail_closed(monkeypatch):
    class Cuda:
        def is_available(self): return True
        def device_count(self): return 1
    fake_torch = SimpleNamespace(cuda=Cuda())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CVD"):
        smoke.require_single_visible_cuda(fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "GPU-forged, 00000000:03:00.0, NVIDIA GeForce RTX 3090, 24576\n")
    with pytest.raises(RuntimeError, match="identity"):
        smoke.require_single_visible_cuda(fake_torch)
    parameters = [(prefix + "weight", torch.nn.Parameter(torch.ones(1))) for prefix in smoke._CRITICAL_PARAMETER_PREFIXES.values()]
    for _, parameter in parameters: parameter.grad = torch.ones_like(parameter)
    model = SimpleNamespace(named_parameters=lambda: iter(parameters))
    assert set(smoke.require_critical_gradients(model, torch)) == set(smoke._CRITICAL_PARAMETER_PREFIXES)
    parameters[0][1].grad.zero_()
    with pytest.raises(RuntimeError, match="critical gradient"):
        smoke.require_critical_gradients(model, torch)
    parameters[0][1].grad.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="critical gradient"):
        smoke.require_critical_gradients(model, torch)
    norm_index = list(smoke._CRITICAL_PARAMETER_PREFIXES).index("norm1")
    parameters[norm_index][1].grad = None
    with pytest.raises(RuntimeError, match="critical gradient"):
        smoke.require_critical_gradients(model, torch)


def test_same_fd_source_lineage_rejects_path_size_sha_and_link_tampering(tmp_path: Path):
    source = tmp_path / "source.nwb"; source.write_bytes(b"synthetic-source-bytes")
    row = {"path": str(source.absolute()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "size_bytes": source.stat().st_size, "unit_count": 2}
    smoke.verify_source_lineage_file(source, row)
    bad_size = dict(row); bad_size["size_bytes"] += 1
    with pytest.raises(RuntimeError, match="size"):
        smoke.verify_source_lineage_file(source, bad_size)
    bad_hash = dict(row); bad_hash["sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="SHA"):
        smoke.verify_source_lineage_file(source, bad_hash)
    alias = tmp_path / "alias.nwb"; alias.symlink_to(source)
    alias_row = dict(row); alias_row["path"] = str(alias.absolute())
    with pytest.raises(RuntimeError, match="alias"):
        smoke.verify_source_lineage_file(alias, alias_row)


def test_raw_t4_recompute_is_bracketed_by_pre_and_post_lineage_checks(tmp_path: Path):
    source = tmp_path / "source.nwb"; source.write_bytes(b"stable")
    row = {"path": str(source.absolute()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "size_bytes": source.stat().st_size, "unit_count": 2}
    calls: list[str] = []
    def stable_compute(path, **kwargs):
        calls.append("compute")
        assert kwargs == {"feature_group": "t4", "pool_size": 30, "bin_size_ms": 20, "window_size": 50,
                          "trial_result_filter": "R", "signal_view": "sua"}
        return torch.zeros(2, 4).numpy(), object()
    raw, _ = smoke.recompute_verified_raw_t4(source, row, stable_compute)
    assert raw.shape == (2, 4) and calls == ["compute"]
    def mutating_compute(path, **kwargs):
        path.write_bytes(b"replaced")
        return torch.zeros(2, 4).numpy(), object()
    with pytest.raises(RuntimeError, match="source (size|SHA|changed)"):
        smoke.recompute_verified_raw_t4(source, row, mutating_compute)


def test_receipt_exact_schema_rejects_metric_state_gradient_and_authority_tampering():
    payload = _exact_receipt()
    smoke.validate_smoke_receipt(payload)
    for key, replacement in (("loss", 0.0), ("wall_seconds", float("nan")), ("initial_state_digest", payload["post_step_state_digest"])):
        bad = dict(payload); bad[key] = replacement
        with pytest.raises(RuntimeError): smoke.validate_smoke_receipt(bad)
    extra = dict(payload); extra["unexpected"] = False
    with pytest.raises(RuntimeError, match="top-level"): smoke.validate_smoke_receipt(extra)
    gradients = dict(payload["critical_gradients"]); gradients["norm2"] = False
    bad = dict(payload); bad["critical_gradients"] = gradients
    with pytest.raises(RuntimeError, match="gradient"): smoke.validate_smoke_receipt(bad)
    bad_audit = dict(payload["live_source_audit"]); rows = list(bad_audit["sessions"]); rows[0] = dict(rows[0]); rows[0]["raw_t4_sha256"] = "0" * 64; bad_audit["sessions"] = rows
    bad = dict(payload); bad["live_source_audit"] = bad_audit
    with pytest.raises(RuntimeError, match="session authority"): smoke.validate_smoke_receipt(bad)


def test_created_parent_rename_symlink_race_is_rejected_without_deleting_replacement(tmp_path: Path, monkeypatch):
    parent = tmp_path / "new-output"; path = parent / "receipt.json"; moved = tmp_path / "moved-output"
    original = smoke._full_write; calls = {"n": 0}
    def rename_after_first_write(fd, contents):
        original(fd, contents); calls["n"] += 1
        if calls["n"] == 1:
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
    monkeypatch.setattr(smoke, "_full_write", rename_after_first_write)
    with pytest.raises(RuntimeError, match="renamed/replaced"):
        smoke.write_smoke_receipt_transactionally(path, _exact_receipt())
    assert parent.is_symlink() and moved.is_dir() and not (moved / "receipt.json").exists()
