from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tfpd_exploration.src.m2_postfusion_operator_corrected_score_v1 import binding, driver, physical, plan
from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import variants


ROOT = Path(__file__).resolve().parents[2]


class _Native(nn.Module):
    variant = "B3S"
    trial_length = 100
    window_size = 50
    hidden_dim = 50
    side_dim = 4
    electrode_embed_dim = 0

    def __init__(self) -> None:
        super().__init__()
        self.pre_pool = nn.Linear(100, 50, bias=False)
        self.post_pool = nn.Sequential(nn.Linear(54, 50), nn.Tanh())
        with torch.no_grad():
            self.pre_pool.weight.zero_()
            self.pre_pool.weight[:, :50] = torch.eye(50)
            self.post_pool[0].weight.fill_(0.01)
            self.post_pool[0].bias.zero_()

    def forward_batch(self, calib_trials: torch.Tensor, *, side_features: torch.Tensor | None = None,
                      trial_lengths: torch.Tensor | None = None, electrode_ids: torch.Tensor | None = None) -> torch.Tensor:
        assert side_features is not None
        features = self.pre_pool(calib_trials.permute(0, 1, 3, 2)).mean(dim=1)
        return self.post_pool(torch.cat((features, side_features), dim=-1))


def _adapter(arm: str) -> variants.PostFusionIdentityAdapter:
    adapter = variants.PostFusionIdentityAdapter(_Native(), arm)
    if adapter.alpha is not None:
        with torch.no_grad():
            adapter.alpha.fill_(-0.7)
    adapter.eval()
    return adapter


def _activity(members: int) -> np.ndarray:
    grid = np.arange(members * 100 * 96, dtype=np.float32).reshape(members, 100, 96)
    return (grid % 17 - 8) / np.float32(7.0)


def test_static_workorder_and_actual_v2_ten_leaf_success_graph() -> None:
    assert plan.validate_static(ROOT)["workorder_sha256"] == plan.WORKORDER_SHA256
    witness = binding.validate_v2_success_graph(ROOT)
    assert witness["body_sha256"] == plan.V2_BODIES
    assert witness["closure_sha256"] == plan.V2_CLOSURE_SHA256
    assert len(witness["pfmean_control"]) == 26


def test_explicit_closure_covers_transitive_cdm_stage_and_streaming_import_chain() -> None:
    closed = driver.closure(ROOT)
    assert len(closed["files"]) == len(plan.CLOSURE_RELATIVES) >= 47
    for relative in (
        "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
        "tfpd_exploration/src/support_anchored_t4_stage_o_v1/replay.py",
        "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "streaming_calibration_exp/src/data/falcon_datamodule.py",
    ):
        assert relative in closed["files"]


def test_whole_stack_pfmean_equals_singleton_mean_but_residual_does_not() -> None:
    side = torch.zeros((1, 96, 4), dtype=torch.float32)
    values = _activity(4)
    mean = _adapter("PF-MEAN")
    assert torch.equal(physical.whole_pool_identity(adapter=mean, torch=torch, activities=values, side_tensor=side),
                       physical.singleton_mean_identity(adapter=mean, torch=torch, activities=values, side_tensor=side))
    for arm in ("PF-R1", "PF-R50"):
        adapter = _adapter(arm)
        whole = physical.whole_pool_identity(adapter=adapter, torch=torch, activities=values, side_tensor=side)
        singleton = physical.singleton_mean_identity(adapter=adapter, torch=torch, activities=values, side_tensor=side)
        assert not torch.equal(whole, singleton), arm


def test_ordered_activity_pool_preserves_support_and_evicts_oldest_completed_only() -> None:
    support = [np.full((100, 96), index, dtype=np.float32) for index in range(4)]
    pool = physical.OrderedActivityPool(support, capacity=30)
    for index in range(27):
        pool.commit(np.full((100, 96), 100 + index, dtype=np.float32))
    assert pool.pool_count == 30 and pool.evictions == 1
    values = pool.stack()
    assert np.array_equal(values[:4], np.stack(support))
    assert np.all(values[4] == 101)  # completed index 100 was oldest and was evicted.
    with pytest.raises(physical.PhysicalError, match="support activity"):
        pool._members[0] = np.zeros((100, 96), dtype=np.float32)
        pool.stack()


def test_pfmean_control_rejects_one_prediction_or_r2_drift() -> None:
    witness = binding.validate_v2_success_graph(ROOT)
    rows = [dict(row) for row in witness["score"]["rows"] if row["arm"] == "PF-MEAN"]
    physical.assert_pfmean_control(rows, witness["pfmean_control"])
    rows[0]["prediction_sha256"] = "0" * 64
    with pytest.raises(physical.PhysicalError, match="PF-MEAN exact"):
        physical.assert_pfmean_control(rows, witness["pfmean_control"])


def test_held_v2_validator_rejects_extra_leaf_without_touching_real_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / plan.V2_ROOT_RELATIVE
    base.mkdir(parents=True)
    # Exact SHA literals are deliberately patched to a tiny synthetic graph;
    # the production validator still enforces held-FD topology and sidecars.
    bodies: dict[str, str] = {}
    payloads = {
        "attempt.json": {"schema": "m2_postfusion_checkpoint_score_v2_attempt_v1", "status": "ATTEMPT_RESERVED", "cuda_initialized": False},
        "launch.json": {"schema": "m2_postfusion_checkpoint_score_v2_launch_v1", "cuda_visible_devices": "", "cuda_initialized": False},
        "input_authority.json": {"schema": "m2_postfusion_checkpoint_score_v2_input_authority_v1", "record_count": 13, "records": {str(i): {} for i in range(13)}},
        "score.json": {"schema": "m2_postfusion_checkpoint_score_v2_score_v1", "rows": []},
        "terminal.json": {},
    }
    rows = []
    for surface, count in (("external_post30_local", 6), ("within_post30", 7)):
        for session in range(count):
            for arm in plan.ARMS:
                for law in plan.LAWS:
                    rows.append({"surface": surface, "session": str(session), "arm": arm, "memory_law": law,
                                 "prediction_sha256": "a" * 64, "target_sha256": "b" * 64, "r2": 0.0, "window_count": 1})
    payloads["score.json"]["rows"] = rows
    def publish(name: str, payload: dict[str, object]) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(); (base / name).write_bytes(body); (base / name).chmod(0o444)
        digest = hashlib.sha256(body).hexdigest(); side = base / (name + ".sha256"); side.write_text(f"{digest}  {name}\n"); side.chmod(0o444); return digest
    for name in ("attempt.json", "launch.json", "input_authority.json", "score.json"):
        bodies[name] = publish(name, payloads[name])
    payloads["terminal.json"] = {"schema": "m2_postfusion_checkpoint_score_v2_terminal_v1", "status": "TERMINAL", "terminal_xor_failure": True,
        "attempt_sha256": bodies["attempt.json"], "launch_sha256": bodies["launch.json"], "input_authority_sha256": bodies["input_authority.json"],
        "score_sha256": bodies["score.json"], "current_closure_sha256": plan.V2_CLOSURE_SHA256}
    bodies["terminal.json"] = publish("terminal.json", payloads["terminal.json"])
    monkeypatch.setattr(plan, "V2_BODIES", bodies)
    monkeypatch.setattr(plan, "V2_ROOT_RELATIVE", str(base.relative_to(tmp_path)))
    binding.validate_v2_success_graph(tmp_path)
    (base / "failure.json").write_text("{}")
    with pytest.raises(binding.BindingError, match="missing/extra"):
        binding.validate_v2_success_graph(tmp_path)


def test_public_cli_is_inert_and_clean_import_has_no_torch() -> None:
    purelib = str(Path(sysconfig.get_paths()["purelib"]))
    env = {"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join((str(ROOT), purelib))}
    trace = subprocess.run([sys.executable, "-S", "-c", "import sys; import tfpd_exploration.src.m2_postfusion_operator_corrected_score_v1.driver; assert 'torch' not in sys.modules; print('clean')"],
                           cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    assert trace.stdout.strip() == "clean"
    out = subprocess.run([sys.executable, "-S", "tfpd_exploration/scripts/run_m2_postfusion_operator_corrected_score_v1.py", "--dry-run"],
                         cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    assert json.loads(out.stdout)["status"] == "INERT_READY_REQUIRES_OPAQUE_CAPABILITY"


def test_capability_is_opaque_and_issue_does_not_open_v2_graph(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; (repo / "tfpd_exploration/results").mkdir(parents=True)
    with pytest.raises(driver.DriverError, match="opaque"):
        driver.issue_live_capability(repo, token=object())


def test_streaming_runtime_requires_repo_root_not_tfpd_top_level_src(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", [str(ROOT / "tfpd_exploration")])
    with pytest.raises(driver.DriverError, match="(PYTHONPATH exposes|top-level src already imported)"):
        driver._require_streaming_namespace_clean(ROOT)


def test_canonical_successor_root_rejects_even_a_broken_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; parent = repo / "tfpd_exploration/results"; parent.mkdir(parents=True)
    target = repo / plan.RESULT_ROOT_RELATIVE
    target.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(driver.DriverError, match="not fresh"):
        driver._target(repo)


def _synthetic_cap(repo: Path, monkeypatch: pytest.MonkeyPatch) -> driver._Capability:
    (repo / "tfpd_exploration/results").mkdir(parents=True)
    target, parent = driver._target(repo)
    monkeypatch.setattr(driver, "closure", lambda _root: {"files": {}, "sha256": "c" * 64})
    cap = driver._Capability(root=repo, result_root=target, parent_identity=parent, closure_sha256="c" * 64,
                             _token=driver._TOKEN)
    driver._CAPS.add(id(cap))
    return cap


def test_driver_lifecycle_reaches_terminal_or_failure_xor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ok_repo = tmp_path / "ok"; ok_cap = _synthetic_cap(ok_repo, monkeypatch)
    terminal, failure = driver._execute_synthetic_lifecycle_for_test(ok_cap)
    assert terminal is not None and failure is None
    root = ok_repo / plan.RESULT_ROOT_RELATIVE
    assert (root / "terminal.json").is_file() and not (root / "failure.json").exists()
    failed_repo = tmp_path / "failed"; failed_cap = _synthetic_cap(failed_repo, monkeypatch)
    terminal, failure = driver._execute_synthetic_lifecycle_for_test(failed_cap, fail_after="score")
    assert terminal is None and failure is not None
    root = failed_repo / plan.RESULT_ROOT_RELATIVE
    assert (root / "failure.json").is_file() and not (root / "terminal.json").exists()
    payload = json.loads((root / "failure.json").read_text())
    assert payload["published_prefix"].keys() == {"attempt.json", "launch.json", "input_authority.json", "score.json"}
