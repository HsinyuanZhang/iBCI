"""No-CUDA AJPF V1 science/controller/prehook contract tests."""
from __future__ import annotations

import subprocess
import sys
import json
import hashlib
import builtins
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import controller, plan


def _lineage():
    return {"selected_indices": [0, 1, 2, 3], "selected_order_sha256": "a" * 64,
            "raw_float64_sha256": "b" * 64, "counts_per_bin_float32_sha256": "c" * 64,
            "normalized_side_sha256": "d" * 64, "side_normalizer_sha256": "e" * 64,
            "query_activity_sha256": "f" * 64, "raw_m30_audit_sha256": "0" * 64}


def _frozen_env(monkeypatch, *, score: bool):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production
    values={**production._FROZEN_COMMON_ENV, "CUDA_VISIBLE_DEVICES":"" if score else "0", "PYTHONPATH":str(ROOT)}
    for key,value in values.items(): monkeypatch.setenv(key,value)


def _coordinate(trial: int = 56, start: int = 10) -> controller.M30Coordinate:
    return controller.M30Coordinate("2020-10-19-Run1", trial, start, tuple(range(30, trial)), "a" * 64)


def test_final_design_workorder_and_inert_cli_bind_exact_authority():
    plan.validate_static(ROOT)
    completed = subprocess.run([sys.executable, "-S", str(ROOT / "tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v1.py"), "--dry-run"],
                               check=True, capture_output=True, text=True)
    assert "READY_REQUIRES_OPAQUE_CAPABILITY" in completed.stdout


def test_m30_prefilter_controller_never_clamps_or_skips_short_state():
    row = _coordinate()
    assert controller.requested_pool(1, 0) == 4
    assert controller.requested_pool(1, 1) == 10
    assert controller.requested_pool(1, 2) == 30
    support = (0, 1, 2, 3)
    assert controller.causal_members(row, support, 4) == support
    assert controller.causal_members(row, support, 10) == support + tuple(range(50, 56))
    assert controller.causal_members(row, support, 30) == support + tuple(range(30, 56))
    with pytest.raises(Exception, match="not M30-ready"):
        controller.canonical_batches([_coordinate(trial=55)])
    with pytest.raises(Exception, match="runtime M30 shortage"):
        controller.causal_members(controller.M30Coordinate("s", 56, 1, tuple(range(31, 56)), "b" * 64), support, 30)


def test_update_proofs_are_predeclared_sparse_sentinels_not_hot_path_snapshots():
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import joint_training
    assert joint_training.UPDATE_SENTINEL_ORDINALS == (0, 11, plan.SOURCE_GROUPS_PER_EPOCH - 1)
    assert joint_training.is_update_sentinel(epoch=1, ordinal=0)
    assert joint_training.is_update_sentinel(epoch=12, ordinal=11)
    assert joint_training.is_update_sentinel(epoch=4, ordinal=plan.SOURCE_GROUPS_PER_EPOCH - 1)
    assert not joint_training.is_update_sentinel(epoch=4, ordinal=12)


def test_ordinary_mask_and_loss_paths_do_not_host_materialize_evidence():
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import dropout, runtime
    class Condition:
        def all(self): return self
        def __bool__(self): raise AssertionError("ordinary path attempted host finite bool")
    class Mask:
        is_cuda=True
        def __ge__(self, _): return self
        def to(self, **_): return self
        def __truediv__(self, _): return self
        def detach(self): return self
        def contiguous(self): return self
        def cpu(self): raise AssertionError("ordinary mask attempted cpu")
        def numpy(self): raise AssertionError("ordinary mask attempted numpy")
    class FakeMaskTorch:
        float32=object()
        def __init__(self): self.async_checks=0
        def rand(self, *_args, **_kwargs): return Mask()
        def isfinite(self, _): return Condition()
        def _assert_async(self, *_): self.async_checks += 1
    fake_mask=FakeMaskTorch()
    _p, _mask, digest=dropout.governing_probability_and_mask(torch=fake_mask,batch_size=2,units=3,
        dynamic=False,dropout_rate=0.2,dynamic_low=0.0,dynamic_high=1.0,device="cuda:0",audit=False)
    assert digest is None and fake_mask.async_checks == 1

    class Tensor:
        is_cuda=True; ndim=3; shape=(2,4,1)
        def __getitem__(self, _): return self
        def __truediv__(self, _): return self
        def __sub__(self, _): return self
        def __pow__(self, _): return self
    class Loss(Tensor):
        def cpu(self): raise AssertionError("ordinary loss attempted cpu")
        def item(self): raise AssertionError("ordinary loss attempted item")
    class FakeLossTorch:
        def __init__(self): self.async_checks=0
        def mean(self, _): return Loss()
        def isfinite(self, _): return Condition()
        def _assert_async(self, *_): self.async_checks += 1
    fake_loss=FakeLossTorch()
    assert isinstance(runtime.task_only_last_bin_mse(torch=fake_loss,prediction=Tensor(),target=Tensor(),audit=False),Loss)
    assert fake_loss.async_checks == 1


def test_sparse_receipt_keeps_rng_state_optimizer_and_alpha_evidence():
    torch=pytest.importorskip("torch")
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production
    row={"forward_calls":1,"backward_calls":1,"adam_steps":1,"teacher_forward_calls":0,
         "internal_unit_dropout_draws":0,"loss_tensor":torch.tensor(1.0),"requested_m":4,
         "hook_first_calls":1,"hook_rep_calls":1,"uses_shared_resident":True,"resident_object_id":3,
         "update_sentinel":True,"governing_dropout_probability":0.2,"governing_mask_sha256":"a"*64,
         "rng_pre_mask_sha256":"b"*64,"rng_post_mask_sha256":"c"*64,"rng_arm_end_sha256":"d"*64,
         "sparse_evidence":{"student_state_sha256":"e"*64,"optimizer_state_sha256":"f"*64,
                            "alpha_value":0.0,"alpha_gradient":0.125},
         "group_gradient":[{"materialized":1,"finite_tensor":torch.tensor(True),"nonzero_tensor":torch.tensor(True)}]*2,
         "group_update":[{"norm_tensor":torch.tensor(0.1),"nonzero_tensor":torch.tensor(True),"finite_tensor":torch.tensor(True)}]*2}
    receipt=production._receipt_step({"J-R1":row})["J-R1"]
    assert receipt["rng_post_mask_sha256"] == "c"*64
    assert receipt["sparse_evidence"]["optimizer_state_sha256"] == "f"*64
    assert receipt["sparse_evidence"]["alpha_gradient"] == 0.125
    assert receipt["group_update"][0]["finite"] is True
    assert production._smoke_rates(2.0) == {"steps_per_second":6.0,"arm_forwards":36,"arm_forwards_per_second":18.0}


def test_packed_resident_h2d_issues_exactly_one_transfer_and_preserves_views(monkeypatch):
    torch = pytest.importorskip("torch")
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import joint_training
    calls=[]
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value: value)
    real_to=torch.Tensor.to
    def counted_to(value, *args, **kwargs):
        calls.append((args, kwargs))
        return value
    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    source=(("calibration",torch.arange(6,dtype=torch.float32).reshape(1,2,3)),
            ("side",torch.arange(4,dtype=torch.float32).reshape(1,2,2)),
            ("windows",torch.arange(8,dtype=torch.float32).reshape(2,4)),
            ("target",torch.arange(2,dtype=torch.float32).reshape(2,1)))
    values, receipt=joint_training._packed_single_h2d(torch=torch,tensors=source)
    assert len(calls) == 1 and calls[0][0] == ("cuda:0",) and calls[0][1]["non_blocking"] is True
    assert receipt["h2d_transfers"] == 1 and receipt["packed_h2d"] is True
    assert [field["byte_offset"] for field in receipt["packed_fields"]] == [0,24,40,72]
    assert all(torch.equal(values[name], value) for name,value in source)


def test_canonical_batches_are_lexical_homogeneous_and_bounded():
    rows = [_coordinate(56, 8), _coordinate(56, 2), _coordinate(57, 1)]
    batches = controller.canonical_batches(rows, max_batch=32)
    assert [[x.window_start for x in batch] for batch in batches] == [[2, 8], [1]]
    assert len(controller.stream_digest(batches)) == 64
    split = controller.canonical_batches([_coordinate(56, 1), controller.M30Coordinate("other", 56, 2, tuple(range(30, 56)), "c" * 64)])
    assert len(split) == 2


def test_route_owned_adapter_plus_zero_parity_and_negative_zero_rejection_cpu_only():
    torch = pytest.importorskip("torch")
    from torch import nn
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import adapter

    class Post(nn.Module):
        def forward(self, value): return value[..., :4]
    class Native(nn.Module):
        variant = "B3S"; trial_length = 4; window_size = 4; hidden_dim = 4; side_dim = 1; electrode_embed_dim = 0
        def __init__(self): super().__init__(); self.pre_pool = nn.Identity(); self.post_pool = Post()
        def forward_batch(self, calib, trial_lengths=None, side_features=None, electrode_ids=None):
            return calib.mean(dim=1).permute(0, 2, 1)
    class Student(nn.Module):
        def __init__(self): super().__init__(); self.id_encoder = Native()
    student = Student()
    calib = torch.randn(2, 4, 4, 4)
    side = torch.randn(2, 4, 1)
    native = student.id_encoder.forward_batch(calib, side_features=side)
    gated = adapter.install_after_strict_load(student, "J-R1")
    adapter.require_positive_zero(gated.alpha)
    assert torch.equal(gated.forward_batch(calib, side_features=side), native)
    with torch.no_grad(): gated.alpha.copy_(torch.tensor(-0.0))
    with pytest.raises(Exception, match=r"IEEE \+0"):
        adapter.require_positive_zero(gated.alpha)


def test_first_fc_in_hook_masks_once_and_preserves_rep_call_cpu_only():
    torch = pytest.importorskip("torch")
    from torch import nn
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1.dropout import FirstCallMaskHook
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.eye(2))
    hook = FirstCallMaskHook(torch.tensor([[2.0, 0.0, 1.0]])).install(layer)
    value = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [11.0, 13.0]]])
    first = layer(value)
    second = layer(value)
    hook.remove(); hook.validate(require_rep_call=True)
    assert torch.equal(first, torch.tensor([[[2.0, 6.0], [0.0, 0.0], [11.0, 13.0]]]))
    assert torch.equal(second, value)


def test_task_only_last_bin_loss_and_two_group_adam_no_teacher_cpu_only():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sympy", reason="local Torch optimizer build requires sympy")
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import runtime
    first, second = torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([{"params": [first], "lr": plan.ENCODER_ALPHA_LR}, {"params": [second], "lr": plan.DECODER_LR}],
                                 betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS, weight_decay=0.0)
    runtime.validate_two_group_adam(optimizer=optimizer, encoder_alpha=[first], decoder=[second])
    prediction = torch.full((2, 4, 1), 10.0); target = torch.ones((2, 4, 1))
    assert float(runtime.task_only_last_bin_mse(torch=torch, prediction=prediction, target=target)) == 1.0


def test_production_shaped_lifecycle_success_failure_and_capability_drift(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle
    mapping = lifecycle.closure_map(ROOT)
    cap = lifecycle._issue_for_test(repo_root=ROOT, root=tmp_path / "success", reviewed=mapping, digest=lifecycle.closure_sha256(mapping))
    value = lifecycle.execute_synthetic(cap=cap, repo_root=ROOT, source={"activity_authority": plan.ACTIVITY_AUTHORITY},
                                        smoke={"groups": 12}, epochs=tuple({"epoch": n} for n in range(1, 13)), manifest={"arms": list(plan.ARMS)})
    terminal = Path(value["root"]) / "terminal.json"
    assert terminal.is_file() and not (terminal.parent / "failure.json").exists()
    with pytest.raises(Exception, match="consumed"):
        lifecycle.consume(cap, ROOT)
    cap2 = lifecycle._issue_for_test(repo_root=ROOT, root=tmp_path / "failure")
    with pytest.raises(Exception, match="AJPF failed"):
        lifecycle.execute_synthetic(cap=cap2, repo_root=ROOT, source={}, smoke={}, epochs=tuple(), manifest={})
    failure = json.loads((tmp_path / "failure" / "failure.json").read_text())
    assert failure["published_prefix"] == ["attempt.json", "launch.json", "source_authority.json", "smoke.json"]


def test_scorer_exact_78_order_same_input_and_mutation_guards():
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import scorer
    sessions = tuple(f"s{index:02d}" for index in range(13))
    rows=[]
    for session in sessions:
        for arm in plan.ARMS:
            for law in ("FIXED30", "UNCAPPED"):
                rows.append({"session":session,"arm":arm,"law":law,"prediction_sha256":"a"*64,"target_sha256":"b"*64,"starts_sha256":"c"*64,"state_before_sha256":"d"*64,"state_after_sha256":"d"*64,"parameter_updates":0,"target_updates":0,"r2":0.1,"selected_support4_carrier_lineage":_lineage()})
    scorer.validate_rows(rows,sessions)
    rows[1],rows[2]=rows[2],rows[1]
    with pytest.raises(Exception,match="canonical order"):
        scorer.validate_rows(rows,sessions)


def test_typed_training_production_shape_seals_only_epoch12_checkpoints(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle, production
    sessions=tuple(f"s{index:02d}" for index in range(13))
    rows=[{"session":session,"arm":arm,"law":law,"prediction_sha256":"a"*64,"target_sha256":"b"*64,"starts_sha256":"c"*64,"state_before_sha256":"d"*64,"state_after_sha256":"d"*64,"parameter_updates":0,"target_updates":0,"r2":0.1,"selected_support4_carrier_lineage":_lineage()} for session in sessions for arm in plan.ARMS for law in ("FIXED30","UNCAPPED")]
    class FakeRuntime:
        def source_authority(self): return {"activity_authority":plan.ACTIVITY_AUTHORITY}
        def smoke(self): return {"groups":12}
        def epochs(self): return tuple({"epoch":index} for index in range(1,13))
        def checkpoint_bytes(self,arm): return arm.encode()
    cap=lifecycle._issue_for_test(repo_root=ROOT,root=tmp_path/"production")
    result=production._execute_synthetic_for_test(cap=cap,repo_root=ROOT,runtime=FakeRuntime())
    root=Path(result["root"])
    assert (root/"terminal.json").is_file() and not (root/"score.json").exists()
    assert all((root/"checkpoints"/f"{arm}_epoch12.pt").is_file() for arm in plan.ARMS)
    witness=lifecycle.validate_held_training_success_graph(root=root,repo_root=ROOT)
    assert witness["manifest_sha256"] == json.loads((root/"terminal.json").read_text())["manifest_sha256"]


def test_held_training_validator_rejects_rehashed_terminal_link_and_sidecar_mode(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle, production
    class Runtime:
        def source_authority(self): return {"activity_authority":plan.ACTIVITY_AUTHORITY}
        def smoke(self): return {"groups":12}
        def epochs(self): return tuple({"epoch":index} for index in range(1,13))
        def checkpoint_bytes(self, arm): return arm.encode()
    cap=lifecycle._issue_for_test(repo_root=ROOT,root=tmp_path/"training")
    root=Path(production._execute_synthetic_for_test(cap=cap,repo_root=ROOT,runtime=Runtime())["root"])
    terminal=root/"terminal.json"; side=root/"terminal.json.sha256"
    body=json.loads(terminal.read_text()); body["launch_sha256"]="0"*64
    raw=(json.dumps(body,sort_keys=True,indent=2)+"\n").encode()
    os.chmod(terminal,0o644); terminal.write_bytes(raw); os.chmod(terminal,0o444)
    os.chmod(side,0o644); side.write_text(f"{hashlib.sha256(raw).hexdigest()}  terminal.json\n"); os.chmod(side,0o444)
    with pytest.raises(Exception,match="terminal/manifest link"):
        lifecycle.validate_held_training_success_graph(root=root,repo_root=ROOT)
    # Restore the valid body/sidecar then make the sidecar itself mutable;
    # content is still correct, so this isolates the held-fd mode/link gate.
    body["launch_sha256"]=json.loads((root/"launch.json").read_text()) and json.loads((root/"terminal.json").read_text()).get("launch_sha256", "")
    # Rebuild from the original launch's immutable digest rather than relying
    # on textual content: this is deliberately a valid/rehashed leaf.
    body["launch_sha256"]=hashlib.sha256((root/"launch.json").read_bytes()).hexdigest()
    raw=(json.dumps(body,sort_keys=True,indent=2)+"\n").encode()
    os.chmod(terminal,0o644); terminal.write_bytes(raw); os.chmod(terminal,0o444)
    os.chmod(side,0o644); side.write_text(f"{hashlib.sha256(raw).hexdigest()}  terminal.json\n")
    with pytest.raises(Exception,match="sidecar mode/link"):
        lifecycle.validate_held_training_success_graph(root=root,repo_root=ROOT)


def test_checkpoint_proof_clone_uses_matching_arm_schema_for_all_three_arms():
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production
    class Arm:
        def __init__(self, name): self.name=name
    prepared=type("Prepared",(),{})()
    prepared.modules={arm:Arm(arm) for arm in plan.ARMS}
    clones={arm:production._arm_proof_clone(prepared=prepared,arm=arm) for arm in plan.ARMS}
    assert all(clones[arm].name == arm for arm in plan.ARMS)
    assert all(clones[arm] is not prepared.modules[arm] for arm in plan.ARMS)


def test_cpu_score_attempt_is_published_before_torch_import_failure(tmp_path, monkeypatch):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle, production
    _frozen_env(monkeypatch,score=True)
    cap=lifecycle._issue_for_test(repo_root=ROOT,root=tmp_path/"score")
    original=builtins.__import__
    def deny_torch(name, *args, **kwargs):
        if name == "torch": raise ImportError("test score admission boundary")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins,"__import__",deny_torch)
    with pytest.raises(ImportError,match="test score admission boundary"):
        production.execute_cpu_score(cap=cap,repo_root=ROOT,training_root=tmp_path/"not-opened")
    attempt=json.loads((tmp_path/"score"/"attempt.json").read_text())
    assert attempt["torch_imported"] is False
    assert (tmp_path/"score"/"failure.json").is_file()


def test_created_result_root_identity_is_pinned_and_symlinks_fail_closed(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production
    root=tmp_path/"root"; root.mkdir()
    identity=production._created_root_identity(root)
    assert production._created_root_identity(root) == identity
    link=tmp_path/"link"; link.symlink_to(root, target_is_directory=True)
    with pytest.raises(Exception,match="result root type"):
        production._created_root_identity(link)


def test_frozen_env_and_gpu_uuid_accept_only_the_canonical_optional_prefix(monkeypatch):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production
    _frozen_env(monkeypatch,score=False)
    class Cuda:
        def is_available(self): return True
        def device_count(self): return 1
        def current_device(self): return 0
        def get_device_properties(self, _): return type("P",(),{"uuid":"ac7388a5-2e98-300a-fdb3-0b67bfd494d9"})()
        def mem_get_info(self, _): return 5*1024**3, 8*1024**3
    receipt=production._cuda_launch_attestation(type("Torch",(),{"cuda":Cuda()})())
    assert receipt["raw_uuid"] == "ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    assert receipt["canonical_uuid"] == "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    monkeypatch.setenv("OMP_NUM_THREADS","2")
    with pytest.raises(Exception,match="frozen execution environment"):
        production.validate_frozen_environment(repo_root=ROOT,score=False)


def test_driver_cpu_child_keeps_site_packages_and_dev_child_can_import_runtime(monkeypatch):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import driver, production
    calls=[]
    real_run=subprocess.run
    _frozen_env(monkeypatch,score=False)
    monkeypatch.setattr(production,"execute_training",lambda **_:{"root":"/synthetic/training"})
    monkeypatch.setattr(driver.subprocess,"run",lambda args,**kwargs: calls.append((args,kwargs)))
    driver.execute_production(object())
    args, environment=calls[0]
    assert "-S" not in args and args[1] == "-c"
    assert environment["env"]["PYTHONNOUSERSITE"] == "1"
    assert environment["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # The developer interpreter can prove site-packages availability; the
    # route child itself receives the frozen environment asserted above.
    developer_env=dict(os.environ); developer_env.pop("PYTHONNOUSERSITE",None); developer_env.pop("PYTHONPATH",None)
    child=real_run([sys.executable,"-c","import torch, numpy; print('runtime-ok')"],check=True,capture_output=True,text=True,env=developer_env)
    assert child.stdout.strip() == "runtime-ok"


def test_clean_qualified_late_bound_import_trace_is_a_subset_of_explicit_closure():
    code="""
import importlib, json, sys
from pathlib import Path
from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import plan
for name in plan.LATE_BOUND_IMPORT_MODULES:
    importlib.import_module(name)
root=Path(sys.argv[1]).resolve()
loaded=set()
for module in tuple(sys.modules.values()):
    path=getattr(module, '__file__', None)
    if not isinstance(path, str) or not path.endswith('.py'):
        continue
    try:
        loaded.add(Path(path).resolve().relative_to(root).as_posix())
    except ValueError:
        pass
import torch
print(json.dumps({'loaded': sorted(loaded), 'missing': sorted(loaded-set(plan.STATIC_CLOSURE_RELATIVES)),
                  'cuda_initialized': bool(torch.cuda.is_initialized())}, sort_keys=True))
"""
    environment=dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
    child=subprocess.run([sys.executable,"-c",code,str(ROOT)],check=True,capture_output=True,text=True,env=environment)
    trace=json.loads(child.stdout)
    assert trace["missing"] == []
    assert trace["cuda_initialized"] is False


def test_first_checkpoint_proof_failure_leaves_no_empty_checkpoint_directory(tmp_path, monkeypatch):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle, production
    root=tmp_path/"training"; root.mkdir()
    published=[]
    # This is the exact completed training-receipt prefix immediately before
    # epoch-12 checkpoint proofing starts.
    for name in lifecycle.TRAINING_BODIES[:-2]:
        lifecycle._pair(root,name,{"name":name}); published.append(name)
    class Runtime:
        def checkpoint_bytes(self, arm): return arm.encode("ascii")
    monkeypatch.setattr(production,"_checkpoint_proof",lambda **_: (_ for _ in ()).throw(production.ProductionError("proof fails first")))
    with pytest.raises(production.ProductionError,match="proof fails first"):
        production._seal_epoch12_checkpoints(root=root,published=published,checkpoint_arms=[],runtime=Runtime(),repo_root=ROOT)
    lifecycle._pair(root,"failure.json",{"published_prefix":published,"target_access":False})
    lifecycle.validate_training_failure_topology(root,published=tuple(published),checkpoint_arms=tuple())
    assert not (root/"checkpoints").exists()
