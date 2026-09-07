"""Lightweight CPU contracts for guarded H1 QueryAge EMA scoring."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_score as score


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float32));self.calls=0
    def forward_last(self,x,bank):
        self.calls+=1
        return x[:,-1,:2]*self.scale


class EMA:
    def __init__(self,value=2.):self.value=float(value)
    def score_with_ema(self,model,fn):
        raw=model.scale.detach().clone()
        try:
            with torch.no_grad():model.scale.fill_(self.value)
            return fn(model)
        finally:
            with torch.no_grad():model.scale.copy_(raw)


class BrokenEMA(EMA):
    def score_with_ema(self,model,fn):
        with torch.no_grad():model.scale.fill_(self.value)
        return fn(model)


def cache(per_session):
    rows={}
    for i in range(13):
        neural=np.zeros((702,2),np.float32)
        for t in range(702):neural[t]=((i+1)*10+t,(i+1)*20+t)
        starts=np.arange(per_session,dtype=np.int64)
        ends=starts+699
        mask=np.zeros(702,dtype=bool);mask[ends]=True
        # The EMA score is `neural[end] * 2 / 20` in native units.
        velocity=neural*2/20
        rows[f"source-{i:02d}"]={"neural":neural,"velocity":velocity.astype(np.float32),"query_starts":starts,
            "eval_mask":mask,"bank":{"E0":torch.zeros(1),"T":torch.zeros(1),"unit_mask":torch.ones(1,dtype=torch.bool)}}
    return {"minival":rows}


def test_guarded_selection_and_complete_ema_restore_native_fp64_and_ids(monkeypatch):
    monkeypatch.setattr(score,"SELECTION_BINS",13);monkeypatch.setattr(score,"COMPLETE_BINS",26)
    model=Toy();calls=[]
    selection=score.score_selection_cached_ema(model,EMA(),cache(1),torch.device("cpu"),lambda:calls.append("g"))
    assert selection["n_bins"]==13 and selection["finite"] and np.isfinite(selection["r2_concat_float64"])
    assert model.scale.item()==1. and model.calls==13 and len(calls)>=2*model.calls
    model.calls=0;calls.clear()
    complete=score.score_complete_cached_ema(model,EMA(),cache(2),torch.device("cpu"),lambda:calls.append("g"))
    assert complete["n_bins"]==26 and complete["finite"] and model.calls==13 and len(calls)>=2*model.calls
    assert set(complete["per_session_r2_float64"])==set(cache(2)["minival"])
    assert np.array_equal(complete["_end"],np.tile(np.array([699,700],np.int64),13))
    assert np.array_equal(complete["_session_id"],np.repeat(np.array(sorted(cache(2)["minival"]),dtype="U128"),2))
    assert complete["_prediction"].dtype==np.float64 and complete["_target"].dtype==np.float64


def test_guard_exception_restores_raw_even_broken_ema(monkeypatch):
    monkeypatch.setattr(score,"SELECTION_BINS",13);model=Toy();seen=[]
    def guard():
        seen.append(1)
        if len(seen)==4:raise TimeoutError("fixture deadline")
    with pytest.raises(TimeoutError,match="deadline"):
        score.score_selection_cached_ema(model,BrokenEMA(),cache(1),torch.device("cpu"),guard)
    assert model.scale.item()==1. and model.training


def test_source_roster_and_cardinality_are_not_softened(monkeypatch):
    monkeypatch.setattr(score,"SELECTION_BINS",13)
    bad=cache(1);bad["minival"].pop("source-12")
    with pytest.raises(RuntimeError,match="thirteen"):
        score.score_selection_cached_ema(Toy(),EMA(),bad,torch.device("cpu"),lambda:None)
    with pytest.raises(RuntimeError,match="cardinality"):
        score.score_selection_cached_ema(Toy(),EMA(),cache(2),torch.device("cpu"),lambda:None)
