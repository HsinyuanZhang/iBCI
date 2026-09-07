"""Synthetic archive contracts for the H1 cold phase summary (no torch/cache)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from tfpd_exploration.src.h1_family_v1 import cold_phase_summary as s


def _json(value, path): path.write_text(json.dumps(value, sort_keys=True)); return path
def _digest(path): return s.sha(path)


def _arrays(*, source=False, offset=0.):
    n, per = (s.SOURCE208_COUNT, s.SOURCE208_PER_SESSION) if source else (s.COUNT, s.COUNT // s.SESSIONS)
    sid=np.repeat(np.array(["a","b"]),per); coordinate=np.tile(np.arange(per,dtype=np.int64),s.SESSIONS)
    target=np.arange(n*s.OUTPUTS,dtype=np.float64).reshape(n,s.OUTPUTS)/11
    return {"prediction":target+offset,"target":target,"session_id":sid,"start" if source else "end":coordinate}


def _npz(path, arrays): np.savez_compressed(path,**arrays); return {"path":str(path),"sha256":_digest(path)}


def _e2_score(arrays): return s.metric(arrays)
def _e1_score(arrays):
    m=s.metric(arrays)
    return {"windows":s.SOURCE208_COUNT,"pooled":{"r2_concat_float64":m["r2_concat_float64"]},"per_session":{key:{"r2_concat_float64":value} for key,value in m["per_session_r2_float64"].items()}}


def _rows():
    # Exactly 23,212 windows: 551 batches of 32 plus 180 batches of 31.
    return [{"epoch":epoch,"batch":batch,"rows":32 if batch<551 else 31,"cold_rows":0,"loss":{"CONTROL":1.,"PREFIX":1.}} for epoch in (1,2) for batch in range(731)]


def _fixture(tmp_path, monkeypatch):
    # Keep archive geometry tiny while leaving the production 2 x 731 / 23212
    # worker-completion assertion intact.
    monkeypatch.setattr(s,"COUNT",26); monkeypatch.setattr(s,"SESSIONS",2); monkeypatch.setattr(s,"COLD",12); monkeypatch.setattr(s,"FULL",14); monkeypatch.setattr(s,"COLD_END_EXCLUSIVE",6)
    monkeypatch.setattr(s,"SOURCE208_COUNT",4); monkeypatch.setattr(s,"SOURCE208_PER_SESSION",2)
    formal=tmp_path/"formal"; (formal/"exports").mkdir(parents=True); closure=tmp_path/"closure.py"; source_closure=tmp_path/"source_closure.py"; closure.write_text("# frozen\n"); source_closure.write_text("# source frozen\n")
    identities={str(epoch):{"sampler_sha256":"a"*64,"keep_sha256":"b"*64,"prefix_sha256":"c"*64} for epoch in (1,2)}
    for arm in s.ARMS: _npz(formal/"exports"/f"{arm}_selected_complete_native_float64.npz",_arrays())
    artifact={"files":{str(closure):_digest(closure),**{str(formal/"exports"/f"{arm}_selected_complete_native_float64.npz"):_digest(formal/"exports"/f"{arm}_selected_complete_native_float64.npz") for arm in s.ARMS}}}
    source={"files":{str(source_closure):_digest(source_closure)}}; protocol={"schema":"test","epochs":2}
    for arm in s.ARMS:
        directory=tmp_path/"phase"/arm; directory.mkdir(parents=True)
        checkpoints=[]
        for epoch in (1,2):
            checkpoint=directory/f"epoch{epoch:02d}.pt"; checkpoint.write_bytes(b"checkpoint"+bytes([epoch])); checkpoints.append({"epoch":epoch,"path":str(checkpoint),"sha256":_digest(checkpoint),"identities":identities[str(epoch)]})
        evaluations={}
        for endpoint,is_source in (("epoch1_source208",True),("epoch2_complete",False)):
            evaluations[endpoint]={}
            for cell in s.CELLS:
                evaluations[endpoint][cell]={}
                for mode in s.MODES:
                    arrays=_arrays(source=is_source,offset={"CONTROL":.2,"PREFIX":.1}[cell]+({"RAW":.02,"EMA":0}[mode]))
                    archive=_npz(directory/f"{endpoint}_{cell}_{mode}.npz",arrays)
                    evaluations[endpoint][cell][mode]=(_e1_score(arrays) if is_source else _e2_score(arrays))|{"archive":archive}
        formal_identities={epoch:{key:value for key,value in identity.items() if key in ("sampler_sha256","keep_sha256")} for epoch,identity in identities.items()}
        pre={"formal_artifact":artifact,"formal_source":source,"files":{str(closure):_digest(closure)},"formal_epoch_identities":formal_identities,"protocol":protocol,"protocol_sha256":s._json_digest(protocol),"formal":str(formal),"output":str(directory),"arm":arm,"physical_gpu":0 if arm=="flat" else 1}
        receipt={"schema":s.WORKER_SCHEMA,"status":"COMPLETE_NO_SELECTION_OR_PROMOTION","pre":pre,"post":pre,"authorization_sha256":None,"no_selection_or_promotion":True,"not_official_or_external":True,"ema_updates":{"CONTROL":1462,"PREFIX":1462},"updates":_rows(),"identities":identities,"checkpoints":checkpoints,"evaluations":evaluations}
        auth=tmp_path/f"h1_cold_phase_authorization_{arm}_v1.json"; _json({"schema":"h1_cold_phase_root_authorization_v2","bindings":pre},auth); receipt["authorization_sha256"]=_digest(auth)
        _json(receipt,directory/"receipt.json"); _json({"bindings":pre,"authorization_sha256":receipt["authorization_sha256"]},directory/"input_authority.json"); _json({"epochs":identities,"checkpoints":checkpoints,"evaluations":evaluations},directory/"progress.json")
    return (tmp_path/"phase").resolve()


def test_complete_two_worker_archive_summary_and_reported_effects(tmp_path,monkeypatch):
    root=_fixture(tmp_path,monkeypatch); output=(tmp_path/"summary.json").resolve(); monkeypatch.setenv("H1_COLD_PHASE_SUMMARY_GO","1")
    result=s.summarize(root,output)
    assert output.is_file() and result["status"]=="COMPLETE_ARCHIVE_ONLY_NO_SELECTION"
    assert set(result["workers"])=={"flat","route"}
    for arm in s.ARMS:
        assert set(result["prefix_minus_control"][arm]["EMA"])=={"all","cold_history_lt_699","full_w700_ge_699"}
        assert result["workers"][arm]["PREFIX"]["EMA"]["groups"]["all"]["n_bins"]==26
        assert result["inputs_sha256_pre"]==result["inputs_sha256_post"]


@pytest.mark.parametrize("mutation,match",[
    ("bad_archive","archive geometry"), ("incomplete","2x731"), ("missing","bound input file missing"),
])
def test_summary_rejects_mutated_incomplete_or_missing_archives(tmp_path,monkeypatch,mutation,match):
    root=_fixture(tmp_path,monkeypatch); monkeypatch.setenv("H1_COLD_PHASE_SUMMARY_GO","1")
    if mutation=="bad_archive":
        path=next((root/"flat").glob("epoch2_complete_CONTROL_RAW.npz")); bad=_arrays(); bad["prediction"]=bad["prediction"].astype(np.float32); np.savez_compressed(path,**bad)
        # Preserve its recorded hash, so this reaches geometry validation.
        receipt=json.loads((root/"flat"/"receipt.json").read_text()); receipt["evaluations"]["epoch2_complete"]["CONTROL"]["RAW"]["archive"]["sha256"]=_digest(path); _json(receipt,root/"flat"/"receipt.json")
        progress=json.loads((root/"flat"/"progress.json").read_text()); progress["evaluations"]=receipt["evaluations"]; _json(progress,root/"flat"/"progress.json")
    elif mutation=="incomplete":
        receipt=json.loads((root/"flat"/"receipt.json").read_text()); receipt["updates"].pop(); _json(receipt,root/"flat"/"receipt.json")
    else: (root/"flat"/"epoch01.pt").unlink()
    with pytest.raises(RuntimeError,match=match): s.summarize(root,(tmp_path/"out.json").resolve())


def test_gate_precedes_any_archive_or_receipt_read(tmp_path,monkeypatch):
    monkeypatch.delenv("H1_COLD_PHASE_SUMMARY_GO",raising=False)
    monkeypatch.setattr(s,"sha",lambda path: pytest.fail("read before summary gate"))
    with pytest.raises(RuntimeError,match="explicit summary"): s.summarize(tmp_path.resolve(),(tmp_path/"out.json").resolve())


@pytest.mark.parametrize("change,match",[
    ("nested_closure","bound input SHA drift"), ("missing_checkpoint_hash","checkpoint hash"), ("formal_identity","formal sampler/keep"),
])
def test_worker_rejects_nested_closure_missing_hash_and_formal_identity(tmp_path,monkeypatch,change,match):
    root=_fixture(tmp_path,monkeypatch); directory=root/"flat"; receipt=json.loads((directory/"receipt.json").read_text())
    if change=="nested_closure":
        path=Path(next(iter(receipt["pre"]["formal_source"]["files"]))); path.write_text("mutated source closure\n")
    elif change=="missing_checkpoint_hash":
        receipt["checkpoints"][0].pop("sha256"); _json(receipt,directory/"receipt.json")
        progress=json.loads((directory/"progress.json").read_text()); progress["checkpoints"]=receipt["checkpoints"]; _json(progress,directory/"progress.json")
    else:
        receipt["pre"]["formal_epoch_identities"]={"1":{},"2":{}}; receipt["post"]=receipt["pre"]; _json(receipt,directory/"receipt.json")
        authority=json.loads((directory/"input_authority.json").read_text()); authority["bindings"]=receipt["pre"]; _json(authority,directory/"input_authority.json")
    with pytest.raises(RuntimeError,match=match): s.verify_worker(root,"flat",{})


def test_cross_arm_source208_identity_and_postimage_mutation_are_rejected(tmp_path,monkeypatch):
    root=_fixture(tmp_path,monkeypatch); monkeypatch.setenv("H1_COLD_PHASE_SUMMARY_GO","1")
    # Keep the route receipt's own metric/hash internally valid, but make its
    # fixed source208 target differ from flat's source208 target.
    directory=root/"route"; receipt=json.loads((directory/"receipt.json").read_text()); item=receipt["evaluations"]["epoch1_source208"]["CONTROL"]["RAW"]
    path=Path(item["archive"]["path"]); arrays={key:value for key,value in np.load(path,allow_pickle=False).items()}; arrays["target"][0,0]+=1; np.savez_compressed(path,**arrays)
    item.update(_e1_score(arrays)); item["archive"]["sha256"]=_digest(path); _json(receipt,directory/"receipt.json")
    progress=json.loads((directory/"progress.json").read_text()); progress["evaluations"]=receipt["evaluations"]; _json(progress,directory/"progress.json")
    with pytest.raises(RuntimeError,match="epoch1 source208 target/session/start identity drift"): s.summarize(root,(tmp_path/"cross_arm.json").resolve())

    root=_fixture(tmp_path/"post",monkeypatch); closure=Path(next(iter(json.loads((root/"flat"/"receipt.json").read_text())["pre"]["formal_source"]["files"])))
    original=s.analysis; fired=False
    def mutate_after_analysis(arrays):
        nonlocal fired
        value=original(arrays)
        if not fired: closure.write_text("changed during archive-only report\n"); fired=True
        return value
    monkeypatch.setattr(s,"analysis",mutate_after_analysis)
    with pytest.raises(RuntimeError,match="bound input SHA drift"): s.summarize(root,(tmp_path/"postimage.json").resolve())
