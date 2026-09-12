"""Synthetic-only boundary tests for the Flat follow-up final gate/report."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest

from dandi688_bench_v2 import protocol
from dandi688_bench_v2.flat_extension.finalize import FLAT_SEAL_SCHEMA, FLAT_SEAL_STATUS, FlatFinalAccess, _canonical, _cells, _spec

CELLS6=["full_flat_sua","full_flat_pmua","activity_flat_sua","activity_flat_pmua","raw_set_flat_sua","raw_set_flat_pmua"]
CELLS10=CELLS6+["full_flat_sua_s43","full_flat_pmua_s43","full_flat_sua_s44","full_flat_pmua_s44"]

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _json(p,x): Path(p).write_text(json.dumps(x,sort_keys=True)); return p
def _seal(tmp,cells=CELLS6):
    plan=_json(tmp/"plan.json",{"flat_final_cells":cells,"base_results_already_observed":True})
    artifact=tmp/"artifact"; artifact.write_text("sealed")
    payload={"schema":FLAT_SEAL_SCHEMA,"status":FLAT_SEAL_STATUS,"protocol":protocol.protocol_dict(),"plan_path":str(plan),"required_cells":cells,"cell_selections":{c:{"status":"SELECTED","checkpoint_sha256":"x","artifact_paths":[str(artifact)]} for c in cells},"artifacts":{str(plan):_sha(plan),str(artifact):_sha(artifact)},"final_sessions_opened":0}
    payload["sha256"]=_canonical(payload); return _json(tmp/"selection_seal.json",payload),artifact

def test_exact_six_and_ten_rosters_and_specs():
    assert _cells({"flat_final_cells":CELLS6})==tuple(CELLS6)
    assert _cells({"flat_final_cells":CELLS10})==tuple(CELLS10)
    assert _spec("raw_set_flat_pmua")==("raw_set","pmua",42)
    assert _spec("full_flat_sua_s44")==("full","sua",44)
    with pytest.raises(ValueError,match="canonical"):
        _cells({"flat_final_cells":CELLS6[:-1]+["full_sua"]})

def test_final_access_rejects_unsealed_or_tampered_artifact(tmp_path):
    with pytest.raises(PermissionError): FlatFinalAccess.from_manifest(tmp_path/"missing.json")
    # A rehashed but bare plan is still rejected: an authorization cannot be
    # manufactured without run_campaign's self-hashed dependency contract.
    seal,artifact=_seal(tmp_path)
    with pytest.raises(PermissionError,match="plan/dependency"):
        FlatFinalAccess.from_manifest(seal)
    artifact.write_text("changed")
    with pytest.raises(PermissionError):
        FlatFinalAccess.from_manifest(seal)

def test_final_access_rejects_bad_plan_selfhash_and_missing_selection(tmp_path):
    seal,_=_seal(tmp_path); value=json.loads(seal.read_text()); value["required_cells"]=CELLS10; _json(seal,value)
    with pytest.raises(PermissionError): FlatFinalAccess.from_manifest(seal)
    seal,_=_seal(tmp_path); value=json.loads(seal.read_text()); value["cell_selections"].pop(CELLS6[0]); value["sha256"]=_canonical(value); _json(seal,value)
    with pytest.raises(PermissionError): FlatFinalAccess.from_manifest(seal)

def test_authorize_rejects_nonfinal_session(tmp_path):
    seal,_=_seal(tmp_path)
    with pytest.raises(PermissionError,match="plan/dependency"):
        FlatFinalAccess.from_manifest(seal).authorize("not-a-final-session")
