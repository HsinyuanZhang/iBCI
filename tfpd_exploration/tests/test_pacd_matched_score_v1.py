"""No-data/no-CUDA tests for the deferred PACD matched scorer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
import stat
import numpy as np
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paired_anchored_calibration_dropout_score_v1 import binding, lifecycle, plan, score


class _Calib:
    def __getitem__(self, value):
        if isinstance(value, slice):
            return ("slice", value.start, value.stop)
        return ("selected", tuple(value))


class _Inputs:
    def __init__(self, surface, session):
        self.surface = surface
        self.session = session
        self.calib = _Calib()
        self.selected_by_budget = {30: tuple(range(30)), 10: tuple(range(10)), 4: (0, 1, 2, 3)}
        self.selected_sha_by_budget = {budget: f"selected-{session}-{budget}" for budget in plan.BUDGET_ORDER}
        self.side_sha_by_budget = {budget: f"side-{session}-{budget}" for budget in plan.BUDGET_ORDER}
        self.ridge_fit_by_budget = {budget: {"raw_t4_sha256": f"raw-{session}-{budget}"} for budget in plan.BUDGET_ORDER}
        self.target_sha256 = f"target-{session}"
        self.valid_mask_sha256 = f"valid-{session}"
        self.neural_sha256 = f"neural-{session}"
        self.query_window_starts_sha256 = f"starts-{session}"


class _Runtime:
    within_roster = tuple(f"within-{i}" for i in range(6))
    external_roster = tuple(f"external-{i}" for i in range(15))


def test_authority_is_session_outer_and_covers_both_regimes_exactly():
    calls = []

    def materialize(runtime, surface, session):
        calls.append((surface, session))
        return _Inputs(surface, session)

    authority, evidence = score.materialize_authority(
        _Runtime(), materialize_session=materialize, tensor_digest=lambda value: repr(value)
    )
    assert calls == [(surface, session) for surface in plan.SURFACE_ORDER
                     for session in (getattr(_Runtime, f"{surface}_roster"))]
    assert len(calls) == 21 and len(authority) == 126
    assert set(evidence["rosters"]) == set(plan.SURFACE_ORDER)
    honest = authority[("external", "external-0", 4, "honest_total")][1]
    isolated = authority[("external", "external-0", 4, "activity_isolation")][1]
    assert honest.selected_support_sha256 == isolated.selected_support_sha256
    assert honest.raw_t4_sha256 == isolated.raw_t4_sha256
    assert honest.normalized_t4_sha256 == isolated.normalized_t4_sha256
    assert honest.calibration_activity_sha256 != isolated.calibration_activity_sha256


def _rows_for_all_systems():
    rosters = {"within": _Runtime.within_roster, "external": _Runtime.external_roster}
    authority, _ = score.materialize_authority(_Runtime(), materialize_session=lambda r, s, n: _Inputs(s, n),
                                                tensor_digest=lambda value: repr(value))
    values = {"P0": 0.40, "P1": 0.45, "P2": 0.44, "T0": 0.39, "C1": 0.41, "SD": 0.38}

    def forward(system, inputs, record):
        def digest(value):
            return hashlib.sha256(value.encode()).hexdigest()
        return {
            "r2": values[system], "prediction_sha256": digest(f"prediction-{system}-{record.input_record_sha256}"),
            "identity_sha256": digest(f"identity-{system}"),
            "repeated_identity_sha256": digest(f"identity-{system}"),
            "repeated_prediction_sha256": digest(f"prediction-{system}-{record.input_record_sha256}"),
            "sentinel_coordinates": (0,),
            "sentinel_prediction_sha256": digest(f"sentinel-{system}"),
            "repeated_sentinel_prediction_sha256": digest(f"sentinel-{system}"),
            "n_windows": 2, "valid_last_bin_count": 1,
            "model_state_before_sha256": digest(f"state-{system}"), "model_state_after_sha256": digest(f"state-{system}"),
            "model_swa_sha256": digest(f"swa-{system}"), "strict_load": True,
            "eval_no_dropout_no_grad": True, "repeated_forward_equal": True, "identity_repeat_equal": True, "finite_prediction": True,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        }

    return score.score_session_outer(authority, rosters=rosters, system_forward=forward), rosters


def _valid_score_payload():
    rows, _rosters = _rows_for_all_systems()
    _authority, input_authority = score.materialize_authority(
        _Runtime(), materialize_session=lambda r, s, n: _Inputs(s, n), tensor_digest=lambda value: repr(value)
    )
    return {"rows": rows, "input_authority": input_authority}


def test_canonical_row_order_same_input_contrasts_and_gates():
    rows, rosters = _rows_for_all_systems()
    assert len(rows) == plan.EXPECTED_ROW_COUNT == 756
    assert [row["system"] for row in rows[:6]] == list(plan.SYSTEM_ORDER)
    assert [row["budget"] for row in rows[0:6] + rows[6 * 6 * 2:6 * 6 * 2 + 6]] == [30] * 6 + [10] * 6
    assert score.canonicalize_rows(rows, rosters) == rows
    contrasts = {
        f"{left}-{right}": score.paired_contrast(rows, left=left, right=right,
                                                   bootstrap=lambda deltas, seed: {"seed": seed, "interval": [min(deltas), max(deltas)]})
        for left, right in (("P1", "P0"), ("P2", "P0"), ("P1", "C1"), ("P2", "C1"),
                            ("P0", "T0"), ("P0", "SD"), ("C1", "T0"))
    }
    decisions = score.decision_gates(contrasts)
    assert decisions["p0_valid"] is True
    assert decisions["P1"]["primary_gate"] is True and decisions["P2"]["primary_gate"] is True
    assert decisions["P1"]["incremental_pairing_gate"] is True
    corrupted = [dict(row) for row in rows]
    corrupted[1]["input_record_sha256"] = "other"
    with pytest.raises(score.ScoreError, match="same-input drift"):
        score.assert_same_input(corrupted)


def test_deferred_producer_binding_is_typed_and_live_is_unmintable():
    synthetic = binding.PACDProducerBinding.synthetic()
    assert synthetic.mode == "synthetic" and [arm.arm for arm in synthetic.arms] == ["P0", "P1", "P2"]
    with pytest.raises(binding.BindingError, match="synthetic"):
        synthetic.require_live()
    with pytest.raises(binding.BindingError, match="synthetic"):
        lifecycle.prepare_live_authority(synthetic, repository_root=Path.cwd(), receipt=_Receipt())
    with pytest.raises(binding.BindingError, match="PACD V3 producer family"):
        binding.ProducerArm("P0", "tmp/p0_fullfull_seed42", "a" * 64, "a" * 64, "a" * 64,
                            "a" * 64, "a" * 64, "a" * 64, ("a" * 64,) * 4)


def test_held_artifact_descriptor_rejects_sidecar_drift(tmp_path):
    artifact = tmp_path / "artifact.json"
    body = b'{"terminal":true}\n'
    artifact.write_bytes(body)
    digest = __import__("hashlib").sha256(body).hexdigest()
    Path(str(artifact) + ".sha256").write_text(f"{digest}  artifact.json\n")
    os.chmod(artifact, 0o444)
    os.chmod(Path(str(artifact) + ".sha256"), 0o444)
    assert binding.descriptor_sha256(tmp_path, "artifact.json", digest)["sha256"] == digest
    os.chmod(Path(str(artifact) + ".sha256"), 0o644)
    Path(str(artifact) + ".sha256").write_text("wrong  artifact.json\n")
    os.chmod(Path(str(artifact) + ".sha256"), 0o444)
    with pytest.raises(binding.BindingError, match="sidecar"):
        binding.descriptor_sha256(tmp_path, "artifact.json", digest)


class _Receipt:
    def write_receipt_transactionally(self, path, payload):
        from src.tfpd_lane.receipt import write_receipt_transactionally
        write_receipt_transactionally(Path(path), payload)


def test_atomic_lifecycle_publishes_attempt_before_materialization_and_xor_failure(tmp_path):
    producer = binding.PACDProducerBinding.synthetic()
    cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="temporary")
    observed = []

    def success():
        observed.append((tmp_path / "temporary" / "attempt.json").is_file())
        return _valid_score_payload()

    result = lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "temporary", binding=producer, capability=cap,
                                      receipt=_Receipt(), materialize_then_score=success, allow_synthetic=True)
    assert len(result["rows"]) == 756 and observed == [True]
    assert (tmp_path / "temporary" / "complete" / "terminal.json").is_file()
    assert not (tmp_path / "temporary" / "failure.json").exists()
    failure_cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="failure")
    with pytest.raises(RuntimeError, match="expected"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "failure", binding=producer, capability=failure_cap,
                                 receipt=_Receipt(), materialize_then_score=lambda: (_ for _ in ()).throw(RuntimeError("expected")),
                                 allow_synthetic=True)
    assert (tmp_path / "failure" / "attempt.json").is_file()
    assert (tmp_path / "failure" / "failure.json").is_file()
    assert not (tmp_path / "failure" / "terminal.json").exists()


def test_public_cli_is_inert_and_imports_no_torch():
    command = [sys.executable, "-S", str(ROOT / "scripts/run_pacd_matched_score_v1.py")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0 and "producer_literals_deferred" in result.stdout
    dry = subprocess.run(command + ["--dry-run"], capture_output=True, text=True)
    assert dry.returncode == 0 and dry.stdout == result.stdout
    result = subprocess.run(command + ["--execute"], capture_output=True, text=True)
    assert result.returncode == 2
    probe = subprocess.run([sys.executable, "-S", "-c",
                            "import runpy,sys; runpy.run_path(sys.argv[1],run_name='score_dry'); assert 'torch' not in sys.modules",
                            str(ROOT / "scripts/run_pacd_matched_score_v1.py")], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


def test_explicit_closure_is_repeatable_and_excludes_results_and_tests():
    first = lifecycle.source_closure(ROOT.parent)
    second = lifecycle.source_closure(ROOT.parent)
    assert first == second and plan.WORK_ORDER_RELATIVE in first["files"]
    assert all("results/" not in name and "/tests/" not in name for name in first["files"])
    required = {"tfpd_exploration/src/cal_aug_v1/plan.py", "tfpd_exploration/src/cal_aug_v1/mechanism.py",
                "tfpd_exploration/src/tfpd_lane/receipt.py", "sua_exploration/mc_maze/datamodule.py",
                "streaming_calibration_exp/src/models/components/spint.py",
                "streaming_calibration_exp/src/models/components/streaming_encoders.py",
                "streaming_calibration_exp/src/models/components/streaming_spint.py",
                "streaming_calibration_exp/src/models/components/rt_ld_gain.py"}
    assert required <= set(first["files"])
    command = [sys.executable, "-S", "-c", "from pathlib import Path; import sys; from src.paired_anchored_calibration_dropout_score_v1.lifecycle import source_closure; source_closure(Path('.')); assert 'torch' not in sys.modules"]
    result = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True,
                            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": f"{ROOT}:{ROOT.parent}", "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stderr


def test_forward_evidence_is_closed_and_rejects_unknown_or_missing_fields():
    rows, rosters = _rows_for_all_systems()
    assert len(score.system_summaries(rows, bootstrap=lambda values, seed: {"seed": seed})) == 72
    authority, _ = score.materialize_authority(_Runtime(), materialize_session=lambda r, s, n: _Inputs(s, n),
                                                tensor_digest=lambda value: repr(value))

    def malformed(system, inputs, record):
        row = next(row for row in rows if row["system"] == system)
        return {key: row[key] for key in score.ForwardEvidence.__dataclass_fields__} | {"forged": True}

    with pytest.raises(score.ScoreError, match="forward evidence schema"):
        score.score_session_outer(authority, rosters=rosters, system_forward=malformed)


def test_m30_selection_and_activity_isolation_are_literal_not_prefix_assumptions():
    good = _Inputs("within", "one")
    total = score.authority_record(good, budget=4, regime="honest_total", tensor_digest=lambda value: repr(value))
    isolation = score.authority_record(good, budget=4, regime="activity_isolation", tensor_digest=lambda value: repr(value))
    assert isolation.calibration_activity_sha256 == isolation.calibration_m30_sha256
    assert total.calibration_activity_sha256 != isolation.calibration_activity_sha256
    good.selected_by_budget[30] = tuple(range(1, 31))
    with pytest.raises(score.ScoreError, match="chronological M30"):
        score.authority_record(good, budget=4, regime="activity_isolation", tensor_digest=lambda value: repr(value))


def test_capability_reuse_wrong_root_and_parent_inode_swap_fail_closed(tmp_path):
    producer = binding.PACDProducerBinding.synthetic()
    (tmp_path / "parent").mkdir()
    cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path,
                                               root_relative="parent/result")
    with pytest.raises(lifecycle.LifecycleError, match="canonical score root identity"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "wrong", binding=producer, capability=cap,
                                 receipt=_Receipt(), materialize_then_score=lambda: {}, allow_synthetic=True)
    old = tmp_path / "parent-old"
    (tmp_path / "parent").rename(old)
    (tmp_path / "parent").mkdir()
    with pytest.raises(lifecycle.LifecycleError, match="result-parent identity drift"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "parent" / "result", binding=producer, capability=cap,
                                 receipt=_Receipt(), materialize_then_score=lambda: {}, allow_synthetic=True)


def test_lifecycle_success_artifacts_are_readonly_linked_and_capability_is_one_shot(tmp_path):
    producer = binding.PACDProducerBinding.synthetic()
    cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="result")
    payload = _valid_score_payload()
    lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "result", binding=producer, capability=cap,
                             receipt=_Receipt(), materialize_then_score=lambda: payload, allow_synthetic=True)
    assert {path.name for path in (tmp_path / "result").iterdir()} == {"attempt.json", "attempt.json.sha256", "complete"}
    assert stat.S_IMODE((tmp_path / "result" / "attempt.json").stat().st_mode) == 0o444
    assert {path.name for path in (tmp_path / "result" / "complete").iterdir()} == {
        name for body in ("input_authority.json", "score.json", "terminal.json") for name in (body, body + ".sha256")
    }
    for path in (tmp_path / "result" / "complete").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    terminal, _ = lifecycle._descriptor_payload(tmp_path / "result" / "complete" / "terminal.json")
    score_body, score_sha = lifecycle._descriptor_payload(tmp_path / "result" / "complete" / "score.json")
    assert terminal["score_sha256"] == score_sha and score_body["attempt_sha256"] == terminal["attempt_sha256"]
    with pytest.raises(lifecycle.LifecycleError, match="consumed"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "result", binding=producer, capability=cap,
                                 receipt=_Receipt(), materialize_then_score=lambda: payload, allow_synthetic=True)


def test_lifecycle_failure_injections_leave_only_attempt_and_failure(tmp_path):
    producer = binding.PACDProducerBinding.synthetic()
    payload = _valid_score_payload()

    class FailingReceipt(_Receipt):
        def __init__(self, fail_name): self.fail_name = fail_name
        def write_receipt_transactionally(self, path, body):
            super().write_receipt_transactionally(path, body)
            if Path(path).name == self.fail_name:
                raise RuntimeError(f"inject-{self.fail_name}")

    for index, name in enumerate(("input_authority.json", "score.json", "terminal.json")):
        root = f"failure-{index}"
        cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative=root)
        with pytest.raises(RuntimeError, match=f"inject-{name}"):
            lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / root, binding=producer, capability=cap,
                                     receipt=FailingReceipt(name), materialize_then_score=lambda: payload, allow_synthetic=True)
        leaves = {path.name for path in (tmp_path / root).iterdir()}
        assert leaves == {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}

    class CorruptTerminalReceipt(_Receipt):
        def write_receipt_transactionally(self, path, body):
            super().write_receipt_transactionally(path, body)
            if Path(path).name == "terminal.json":
                side = Path(path).with_suffix(".json.sha256")
                os.chmod(side, 0o644)

    cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="failure-corrupt-terminal")
    with pytest.raises(lifecycle.LifecycleError, match="receipt sidecar"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "failure-corrupt-terminal", binding=producer,
                                 capability=cap, receipt=CorruptTerminalReceipt(), materialize_then_score=lambda: payload, allow_synthetic=True)
    assert {path.name for path in (tmp_path / "failure-corrupt-terminal").iterdir()} == {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}

    committed_cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="after-rename")
    with pytest.raises(RuntimeError, match="after-rename"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "after-rename", binding=producer,
                                 capability=committed_cap, receipt=_Receipt(), materialize_then_score=lambda: payload,
                                 allow_synthetic=True, _synthetic_after_commit=lambda bundle: (_ for _ in ()).throw(RuntimeError("after-rename")))
    # The directory rename is the commit point: a later observer failure must
    # not synthesize an incompatible failure receipt or mutate `complete/`.
    assert {path.name for path in (tmp_path / "after-rename").iterdir()} == {"attempt.json", "attempt.json.sha256", "complete"}
    lifecycle._validate_result_topology(tmp_path / "after-rename", failure=False)


def test_named_root_replacement_and_runtime_attestation_drift_fail_closed(tmp_path):
    producer = binding.PACDProducerBinding.synthetic()
    profile = {"cuda_visible_devices": "synthetic", "selected_device": "cpu", "cuda_initialized": False}
    current = dict(profile)
    cap = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="replace",
                                               runtime_attestor=lambda: dict(current))
    target = tmp_path / "replace"
    with pytest.raises(lifecycle.LifecycleError, match="profile drift"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=target, binding=producer, capability=cap, receipt=_Receipt(),
                                 materialize_then_score=lambda: (current.__setitem__("selected_device", "drift") or _valid_score_payload()), allow_synthetic=True)
    assert {path.name for path in target.iterdir()} == {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    cap2 = lifecycle._mint_synthetic_capability(lifecycle._SECRET, producer, repository_root=tmp_path, root_relative="swap")
    with pytest.raises(lifecycle.LifecycleError, match="reserved root"):
        lifecycle.execute_atomic(repository_root=tmp_path, out_dir=tmp_path / "swap", binding=producer, capability=cap2, receipt=_Receipt(),
                                 materialize_then_score=lambda: ((tmp_path / "swap").rename(tmp_path / "swap-old"), (tmp_path / "swap").mkdir(), _valid_score_payload())[2], allow_synthetic=True)


def test_descriptor_mode_symlink_and_extra_artifact_fail_closed(tmp_path):
    body = b'{"x": 1}\n'
    digest = hashlib.sha256(body).hexdigest()
    file = tmp_path / "body.json"
    file.write_bytes(body)
    Path(str(file) + ".sha256").write_text(f"{digest}  body.json\n")
    os.chmod(file, 0o444)
    os.chmod(Path(str(file) + ".sha256"), 0o444)
    link = tmp_path / "linked.json"
    link.symlink_to(file)
    with pytest.raises(binding.BindingError):
        binding.descriptor_sha256(tmp_path, "linked.json", digest)
    root = tmp_path / "topology"
    root.mkdir()
    for name in ("attempt.json", "failure.json"):
        item = root / name
        item.write_bytes(body)
        item.with_suffix(item.suffix + ".sha256").write_text(f"{digest}  {name}\n")
        os.chmod(item, 0o444); os.chmod(item.with_suffix(item.suffix + ".sha256"), 0o444)
    (root / "extra.txt").write_text("x")
    with pytest.raises(lifecycle.LifecycleError, match="topology"):
        lifecycle._validate_result_topology(root, failure=True)


def _seal_json(path: Path, value: dict) -> str:
    body = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    os.chmod(path, 0o444); os.chmod(path.with_suffix(path.suffix + ".sha256"), 0o444)
    return digest


def _seal_bytes(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    os.chmod(path, 0o444); os.chmod(path.with_suffix(path.suffix + ".sha256"), 0o444)
    return digest


def test_v3_held_graph_requires_all_48_epoch_receipts_and_fixed_semantics(tmp_path):
    relative = "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42"
    base = tmp_path / relative
    digest = "a" * 64
    attempt_sha = _seal_json(base / "attempt.json", {"schema": "pacd_matched_full_training_v3_attempt", "status": "ATTEMPT_PUBLISHED", "arm": "p0", "target_access": False, "budget": {"epochs": 48, "steps_per_epoch": 33925, "total_steps": 1628400, "seed": 42, "batch_size": 32, "num_workers": 0}})
    authority_sha = _seal_json(base / "source_authority.json", {"attempt_sha256": attempt_sha, "target_access": False, "val": [], "test": [], "source_roster_n": 27, "sampler": {"class": "SessionBatchSampler", "batch_size": 32, "shuffle": True, "seed": 42, "num_workers": 0, "steps_per_epoch": 33925}})
    launch_sha = _seal_json(base / "launch.json", {"schema": "pacd_matched_full_training_v3_launch", "target_access": False, "attempt": {"sha256": attempt_sha}, "source_authority": {"sha256": authority_sha}})
    epochs = []
    for epoch in range(48):
        sha = _seal_json(base / f"epoch{epoch:03d}.json", {"epoch": epoch, "optimizer_steps": 33925, "cumulative_optimizer_steps": (epoch + 1) * 33925, "p0_prediction_mismatches": 0, "p0_identity_mismatches": 0, "rng_violations": 0, "prefix_mutations": 0, "parameter_finiteness": {"violations": 0}, "sentinels": [{}, {}, {}, {}]})
        epochs.append({"name": f"epoch{epoch:03d}.json", "sha256": sha, "sidecar": f"epoch{epoch:03d}.json.sha256"})
    cps = []
    for epoch in (44, 45, 46, 47):
        sha = _seal_bytes(base / f"epoch{epoch:03d}.pt", b"synthetic checkpoint")
        cps.append({"name": f"epoch{epoch:03d}.pt", "sha256": sha, "sidecar": f"epoch{epoch:03d}.pt.sha256", "epoch": epoch})
    swa_sha = _seal_bytes(base / "swa_final4.pt", b"synthetic swa")
    manifest_sha = _seal_json(base / "manifest.json", {"attempt": {"sha256": attempt_sha}, "swa": {"sha256": swa_sha}, "checkpoints": cps})
    terminal = {"epochs": epochs, "checkpoints": cps, "swa": {"proof": {"strict_load": True, "eval": True, "no_grad": True, "repeated_forward_bitwise_equal": True, "output_finite": True, "dynamic_dropout_calls": 0, "state_before_sha256": digest, "state_after_sha256": digest}}}
    terminal_sha = _seal_json(base / "terminal.json", terminal)
    arm = binding.ProducerArm("P0", relative, terminal_sha, swa_sha, manifest_sha, attempt_sha, launch_sha, authority_sha,
                              tuple(item["sha256"] for item in cps))
    binding._validate_v3_arm_graph(tmp_path, arm, terminal)
    os.chmod(base / "epoch017.json", 0o644)
    with pytest.raises(binding.BindingError, match="artifact mode"):
        binding._validate_v3_arm_graph(tmp_path, arm, terminal)


def test_historical_route_codecs_reject_cross_schema_and_missing_swa_proof():
    root = Path("/synthetic-root")
    swa_path = "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/swa_final4.pt"
    checkpoint_sha = [hashlib.sha256(f"checkpoint-{epoch}".encode()).hexdigest() for epoch in (44, 45, 46, 47)]
    checkpoints = [{"epoch": epoch, "file": f"/synthetic-root/tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/epoch{epoch:03d}.ckpt", "sha256": sha}
                   for epoch, sha in zip((44, 45, 46, 47), checkpoint_sha, strict=True)]
    swa_sha = hashlib.sha256(b"swa").hexdigest()
    cal = {"schema": "cal_aug_v1_cell", "status": "CAL_AUG_CELL_TERMINAL", "arm": "t0", "epochs_run": 48,
           "budget": {"steps_per_epoch": 33925, "total_optimizer_steps": 1628400,
                      "optimizer": {"amsgrad": False, "betas": [0.9, 0.999], "cls": "torch.optim.Adam", "eps": 1e-08, "lr": 0.0001, "weight_decay": 0.0},
                      "schedule": {"kind": "warmup_then_cosine", "steps_per_epoch": 33925, "total_steps": 1628400, "warmup_epochs": 2, "warmup_steps": 67850, "n_epochs": 48, "phase_local_steps": True}},
           "data_contract": {"roster_n": 27, "external_sub_m_opened": False, "within_dev_sessions_opened": False, "formal_or_organizer_held_data_opened": False, "val_and_test_empty": True, "val_paths_resolved": [], "test_paths_resolved": []},
           "disclosures": {"target_updates_gradients_or_optimizer_steps": 0}, "invariant_failures": [], "checkpoints": checkpoints,
           "swa": {"window_epochs": [44, 45, 46, 47], "sha256": swa_sha, "path": str((root / swa_path).resolve()), "strict_reload_finite_forward_smoke": True,
                   "manifest": {"strict_reload_verified": True, "finite_forward_tensors": True, "optimizer_state_included": False,
                                "components": [{"path": row["file"], "sha256": row["sha256"]} for row in checkpoints]}},
           "source_closure": {"launch": {"closure_sha256": "same"}, "final": {"closure_sha256": "same"}, "launch_final_closure_equal": True}}
    validator_value = {"terminal": "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/terminal.json", "swa": swa_path, "swa_sha256": swa_sha}
    binding._validate_historical_terminal("T0", cal, validator_value, root)
    with pytest.raises(binding.BindingError, match="SD terminal schema"):
        binding._validate_historical_terminal("SD", cal, validator_value, root)
    cal["swa"]["manifest"]["finite_forward_tensors"] = False
    with pytest.raises(binding.BindingError, match="SWA manifest"):
        binding._validate_historical_terminal("T0", cal, validator_value, root)


def test_real_historical_descriptor_integration_is_receipt_only_and_does_not_import_torch():
    code = (
        "from pathlib import Path; import sys; "
        "from src.paired_anchored_calibration_dropout_score_v1.binding import verify_historical_comparators; "
        f"r=verify_historical_comparators(Path({str(ROOT.parent)!r})); "
        "assert tuple(r)==('T0','C1','SD'); assert 'torch' not in sys.modules"
    )
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": f"{ROOT}:{ROOT.parent}"})
    completed = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, env=environment)
    assert completed.returncode == 0, completed.stderr


def test_real_house_metric_cpu_seam_is_last_bin_only_and_never_initializes_cuda():
    """Reviewed metric primitive, synthetic CPU arrays only—no loader/model/data."""
    import torch
    from src.tfpd_lane import matched_scorer

    class Inputs:
        last_targets = np.asarray([[1.0, 2.0], [3.0, 4.0], [7.0, 9.0]], dtype=np.float32)
        last_valid_mask = np.asarray([True, False, True])

    prediction = torch.zeros((3, 50, 2), dtype=torch.float32)
    prediction[:, 49, :] = torch.as_tensor(Inputs.last_targets)
    prediction[:, 0, :] = 99.0  # proves the governed metric ignores earlier bins
    r2, n_valid, digest = score.governed_last_bin_r2(prediction, Inputs(), session_r2=matched_scorer.session_r2)
    assert r2 == pytest.approx(1.0) and n_valid == 2 and len(digest) == 64
    assert torch.cuda.is_initialized() is False


def test_route_owned_live_cpu_attestation_requires_empty_cvd_and_no_cuda(monkeypatch):
    import torch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert lifecycle._live_cpu_attestation() == {"cuda_visible_devices": "", "selected_device": "cpu", "cuda_initialized": False}
    assert torch.cuda.is_initialized() is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(lifecycle.LifecycleError, match="empty CUDA_VISIBLE_DEVICES"):
        lifecycle._live_cpu_attestation()
