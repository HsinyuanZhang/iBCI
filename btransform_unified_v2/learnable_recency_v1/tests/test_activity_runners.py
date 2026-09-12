"""Contract tests for the pure activity train/score entry points."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def _load(name: str):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/f"{name}.py")
    mod=importlib.util.module_from_spec(spec); assert spec and spec.loader
    spec.loader.exec_module(mod); return mod

def test_train_defaults_preserve_all_task_formal_epochs_and_cpu():
    runner=_load("activity_train")
    p=runner.build_parser()
    a=p.parse_args(["--task","m2","--dest","/tmp/activity-run"])
    assert a.device=="cpu" and a.seed==42 and a.proj_dim==16 and a.identity_hidden is None
    assert runner.TASK_EPOCHS=={"m1":24,"m2":24,"h1":32}

def test_source_batch_selector_requires_exact_m2_manifest(monkeypatch):
    runner=_load("activity_train")
    # The real frozen manifest is intentionally used: a fake session roster
    # must fail before a batch can be yielded.
    with pytest.raises(RuntimeError,match="roster"):
        runner._m2_manifest_batches({"not-a-session":{}},1)

def test_score_surface_calibrates_once_per_session_and_never_with_grad(monkeypatch):
    runner=_load("activity_train")
    class Fake(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls=0
        def eval(self): return self
        def calibrate(self,a,tm=None):
            assert not torch.is_grad_enabled(); self.calls+=1; return torch.zeros(2,3)
        def forward(self,x,identity=None,input_valid_mask=None):
            assert identity.shape==(2,3); return torch.zeros(x.shape[0],2)
    model=Fake()
    item={"activity":torch.zeros(1,4,2).numpy(),"starts":torch.arange(3).numpy(),"X":torch.zeros(8,2).numpy(),"Y":torch.tensor([[1.,2.],[2.,3.],[3.,4.]]).numpy(),"pad":0,"support_provenance":{}}
    monkeypatch.setattr(runner.activity_data,"windows",lambda item,ids,context,device,scale:(torch.zeros(len(ids),context,2),torch.zeros(len(ids),2),torch.ones(len(ids),context,dtype=torch.bool)))
    out=runner.score_surface(model,{"s":item},context=2,device=torch.device("cpu"),behavior_scale=1.,max_batches=1)
    assert model.calls==1 and out["n_windows"]==3 and out["partial"]
