"""Pure receipt/checkpoint refusal tests for the P1 post-run finalizer."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

import tfpd_exploration.src.m1_family_v1.finalize_p1 as finalizer
from tfpd_exploration.src.m1_family_v1.finalize_p1 import EPOCHS, load_completed, select_earliest_ema


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _history(root, recipe):
    rows=[]
    for epoch in range(1,EPOCHS+1):
        row={"epoch":epoch}
        for arm in ("flat","route"):
            p=root/arm/f"epoch_{epoch:03d}.pt"; p.parent.mkdir(exist_ok=True)
            state={"weight":torch.ones(2,2)*epoch}
            torch.save({"schema":"m1_family_v1_chron80_checkpoint_v2","epoch":epoch,"model":state,"ema":state,"recipe":recipe,"outer_query_opened":False},p)
            score=.5 if epoch in (3,5) else epoch/100.
            row[arm]={"epoch":epoch,"checkpoint":str(p),"checkpoint_sha256":_sha(p),"ema":{"equal_session_mean_r2":score,"complete_all_sessions":True,"pooled":{"r2":{"model":.25}},"per_session":{name:{"r2":{"model":score}} for name in ("ses-20120926","ses-20120927","ses-20120928")}}}
        rows.append(row)
    return rows


def _write_complete(root, *, status="COMPLETE"):
    recipe={"x":1,"epochs":24}; history=_history(root,recipe)
    meta={"status":status,"completed_epochs":24,"outer_query_opened":False,"recipe":recipe,"split":{"x":1},"provenance":{"x":1}}
    selected={arm:select_earliest_ema(history,arm) for arm in ("flat","route")}
    report={"status":"SOURCE_MINIVAL_ONLY","outer_query_opened":False,"history":history,"recipe":recipe,"split":{"x":1},"provenance":{"x":1},"selected_primary_ema":selected,"endpoint24":{arm:history[-1][arm] for arm in ("flat","route")}}
    (root/"run_meta.json").write_text(json.dumps(meta));(root/"epoch_metrics.json").write_text(json.dumps(history));(root/"report.json").write_text(json.dumps(report))
    return history


def test_ema_selection_is_earliest_exact_tie():
    rows=[]
    for epoch, score in enumerate((.1,.2,.5,.1,.5),start=1): rows.append({"epoch":epoch,"flat":{"epoch":epoch,"ema":{"equal_session_mean_r2":score}}})
    # The helper's formal contract requires all 24, so build that surface with
    # the same early tie and no later winner.
    rows += [{"epoch":i,"flat":{"epoch":i,"ema":{"equal_session_mean_r2":.1}}} for i in range(6,25)]
    assert select_earliest_ema(rows,"flat")["epoch"] == 3


def test_incomplete_run_refuses_before_any_scoring(tmp_path):
    _write_complete(tmp_path,status="RUNNING")
    with pytest.raises(RuntimeError,match="COMPLETE"):
        load_completed(tmp_path)


def test_checkpoint_hash_and_epoch_integrity_rejected(tmp_path):
    history=_write_complete(tmp_path)
    assert load_completed(tmp_path)[3]["selected"]["flat"]["epoch"] == 3
    path=tmp_path/"flat"/"epoch_003.pt"; path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError,match="hash/path"):
        load_completed(tmp_path)


def test_report_selection_tamper_rejected(tmp_path):
    _write_complete(tmp_path)
    report=json.loads((tmp_path/"report.json").read_text())
    report["selected_primary_ema"]["flat"]["epoch"] = 5
    (tmp_path/"report.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError,match="selection"):
        load_completed(tmp_path)


def test_fake_full_finalization_exports_both_selected_and_endpoint24(tmp_path, monkeypatch):
    _write_complete(tmp_path)
    monkeypatch.setattr(finalizer, "_verify_source_closure", lambda _meta: object())
    monkeypatch.setattr(finalizer, "_p1_models", lambda _device: (torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,2,bias=False)))
    def score(_model, _dev, *, device):
        equal=.5 if int(_model.weight[0,0].item()) in (3,5) else .24
        p=np.zeros((31252,16),np.float32); y=np.ones((31252,16),np.float32); s=np.asarray(["ses-20120926"]*31252); starts=np.arange(31252,dtype=np.int64)
        return p,y,s,starts,{"equal_session_mean_r2":equal,"pooled_r2":.25,"per_session_r2":{"ses-20120926":equal,"ses-20120927":equal,"ses-20120928":equal}}
    monkeypatch.setattr(finalizer,"_score_export",score)
    query_arrays={"target":finalizer.array_sha256(np.ones((31252,16),np.float32)),"session":finalizer.array_sha256(np.asarray(["ses-20120926"]*31252)),"start":finalizer.array_sha256(np.arange(31252,dtype=np.int64))}
    monkeypatch.setattr(finalizer,"_historical_same_surface",lambda current:{"references":{"formal12_current_query":{"query_array_sha256":query_arrays}},"bound":sorted(current)})
    out=finalizer.finalize(run_root=tmp_path,device="cpu")
    assert set(out["exports"]) == {"flat_selected","route_selected","flat_endpoint24","route_endpoint24"}
    assert (tmp_path/"finalized_p1"/"receipt.json").is_file()


def test_actual_historical_metadata_parsers_have_no_null_scores():
    current={arm:{"metrics":{"equal_session_mean_r2":.5,"pooled_r2":.4}} for arm in ("flat","route")}
    parsed=finalizer._historical_same_surface(current)
    for row in parsed["references"].values():
        assert row["n"] == 31252
        assert row["equal_session_mean_r2"] is not None and row["pooled_r2"] is not None
