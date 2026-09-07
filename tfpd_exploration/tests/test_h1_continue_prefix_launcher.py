from __future__ import annotations
import os
from pathlib import Path
import pytest
from tfpd_exploration.src.h1_queryage_family_v1 import continue_prefix_launcher as m

def row(epoch, score):
    return {"epoch":epoch,"selection":{"r2_concat_float64":score},"checkpoint":"/tmp/x","checkpoint_sha256":"x"}

def test_earliest_tie_is_earliest_and_raw_is_not_a_selector():
    rows=[row(i, .1) for i in range(1,25)]
    rows[4]["selection"]["r2_concat_float64"]=.9
    rows[17]["selection"]["r2_concat_float64"]=.9
    rows[17]["selection"]["raw_r2_concat_float64"]=99.
    assert m.earliest_ema(rows)["epoch"]==5

def test_ready_requires_parent_restore_hash_and_exact_13_24(tmp_path):
    parent={"arms":{"flat":{"epoch12":{"checkpoint_sha256":"a"}},"route":{"epoch12":{"checkpoint_sha256":"b"}}}}
    formal=(tmp_path/"formal").resolve();(formal/"barrier").mkdir(parents=True)
    parent_init="parent-shared"
    for arm in m.ARMS:
        (formal/"barrier"/(arm+".ready.json")).write_text('{"shared_init_sha256":"'+parent_init+'"}')
    ids={str(i):{} for i in range(13,25)}
    def one(arm,checkpoint):
        return {"arm":arm,"physical_gpu":m.ARMS[arm],"parent_epoch12_checkpoint_sha256":checkpoint,
                "parent_ready_sha256":m.sha(formal/"barrier"/(arm+".ready.json")),"parent_shared_init_sha256":parent_init,
                "shared_init_sha256":parent_init,"identities":ids,
                "strict_restore_evidence":{"raw_state_sha256":"r","ema_state_sha256":"e","optimizer_state_sha256":"o","rng_state_sha256":"n","global_step":8772,"ema_updates":8772,"lr":1e-4,"exact_restore":True}}
    ready={"flat":one("flat","a"),"route":one("route","b")}
    m.validate_ready(ready,parent,formal)
    ready["route"]["identities"]={str(i):{} for i in range(13,24)}
    with pytest.raises(RuntimeError):m.validate_ready(ready,parent,formal)

def test_run_fails_before_authority_gpu_or_model_without_explicit_go(tmp_path,monkeypatch):
    parent=(tmp_path/"parent").resolve(); parent.mkdir()
    output=(tmp_path/"output").resolve(); auth=(tmp_path/"auth.json").resolve();auth.write_text("{}")
    monkeypatch.delenv(m.GO,raising=False)
    monkeypatch.setattr(m.subprocess,"check_output",lambda *a,**k: pytest.fail("GPU inspection must not occur before GO"))
    with pytest.raises(RuntimeError,match="explicit GO"):
        m.run(parent_formal=parent,output=output,authorization=auth,authorization_sha="x")
    assert not output.exists()

def test_parent_incomplete_is_rejected_before_continuation_import(tmp_path):
    parent=(tmp_path/"parent").resolve(); parent.mkdir()
    (parent/"receipt.json").write_text('{"status":"INCOMPLETE"}')
    (parent/"selection_freeze.json").write_text("{}")
    (parent/"input_authority.json").write_text("{}")
    with pytest.raises(RuntimeError):m.audit_parent(parent)

def test_launcher_small_fixture_completes_children_then_closes_before_receipt(tmp_path, monkeypatch):
    """Exercise the root lifecycle without a cache, model, or forward call."""
    parent=(tmp_path/"parent").resolve();parent.mkdir()
    output=(tmp_path/"out").resolve();auth=(tmp_path/"external_auth.json").resolve()
    fresh={"schema":m.SCHEMA,"status":"ROOT_REVIEW_GO","bindings":{"inputs":{}},"parent_audit":{"arms":{}},"output":str(output)}
    auth.write_text(__import__("json").dumps(fresh))
    monkeypatch.setenv(m.GO,"1")
    monkeypatch.setattr(m,"collect_authority",lambda *_:fresh)
    monkeypatch.setattr(m,"reject_overlap",lambda **_:None)
    monkeypatch.setattr(m,"validate_ready",lambda *args:None)
    monkeypatch.setattr(m.subprocess,"check_output",lambda *args,**kwargs:"")
    from tfpd_exploration.src.h1_queryage_family_v1 import continue_prefix_train as train
    authority_checks=[]
    monkeypatch.setattr(train,"require_authority",lambda out: authority_checks.append(out))
    class Done:
        returncode=0
        def poll(self): return 0
        def terminate(self): raise AssertionError("successful child must not be stopped")
        def wait(self,timeout=None): return 0
    def fake_spawn(fun,arm,out,extra,handles):
        if fun=="worker_run":
            (out/"barrier"/(arm+".ready.json")).write_text("{}")
        else:
            reports={}
            for label,epoch in (("selected",13),("epoch24",24)):
                archive=out/"exports"/(arm+"_"+label+"_complete_native_float64.npz");plain=out/"exports"/(arm+"_"+label+"_plain_ema.pt")
                archive.write_bytes(b"archive-"+arm.encode()+label.encode());plain.write_bytes(b"plain-"+arm.encode()+label.encode())
                reports[label]={"epoch":epoch,"checkpoint_sha256":label+"-ckpt","selection_reproduced":{"n_bins":m.SELECTION_BINS,"r2_concat_float64":.25},"complete":{"n_bins":m.COMPLETE_BINS},"complete_archive":str(archive),"complete_archive_sha256":m.sha(archive),"plain_ema_path":str(plain),"plain_ema_sha256":m.sha(plain)}
            (out/"workers"/(arm+"_final.json")).write_text(__import__("json").dumps({"status":"COMPLETE_POST_FREEZE","reports":reports}))
        return Done()
    def fake_freeze(_,out):
        z={"selected":{a:{"epoch":13,"checkpoint_sha256":"selected-ckpt","ema_r2_float64":.25} for a in m.ARMS},"epoch24":{a:{"epoch":24,"checkpoint_sha256":"epoch24-ckpt","ema_r2_float64":.25} for a in m.ARMS}}
        m.atomic(out/"continuation_selection_freeze.json",z);return z
    monkeypatch.setattr(m,"spawn",fake_spawn);monkeypatch.setattr(m,"freeze_selection",fake_freeze)
    result=m.run(parent_formal=parent,output=output,authorization=auth,authorization_sha=m.sha(auth))
    assert result["status"]=="COMPLETE_FIXED_CONTINUATION_NO_PROMOTION"
    assert (output/"receipt.json").is_file()
    assert not (output/"FAILED_OR_INCOMPLETE.json").exists()
    # Admission and the pre-START barrier both independently recheck it.
    assert authority_checks==[output,output]
