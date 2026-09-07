from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/"scripts/rt_ld_fold0_terminalize_v3.py"
def _m():
 s=importlib.util.spec_from_file_location("t3",SCRIPT);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def _tree(t,m,direct=True,delta=.03):
 import importlib.util
 source=ROOT/"tests/test_rt_ld_fold0_terminalize_v2.py";s=importlib.util.spec_from_file_location("t2test",source);q=importlib.util.module_from_spec(s);s.loader.exec_module(q);q._tree(t,m,delta)
 for dn in m.ARMS:
  d=t/dn;sel=d/"rt_nested_selection_receipt.json";x=json.loads(sel.read_text());ck=Path(x["best_model_path"])
  if direct:
   directck=d/"checkpoints/direct.ckpt";directck.write_bytes(ck.read_bytes());x["best_model_path"]=str(directck.resolve());x["best_model_sha256"]=m._sha(directck);ck=directck
  sp=d/"split_manifest.json";y=json.loads(sp.read_text());y["nested_selection"].update({"clean":True,"inner_validation_only_for_checkpoint_selection":True});y["rt_ld"]={"identity_carrier":"aligned_full_afc4","gain_carrier":"aligned_full_afc4" if dn!="03_g_xls" else "strong_xls_v2"};sp.write_text(json.dumps(y));x.update({"selection_receipt_path":str(sel.resolve()),"config_path":str((d/".hydra/config.yaml").resolve()),"split_manifest_path":str(sp.resolve()),"split_manifest_sha256":m._sha(sp),"run_dir":str(d.resolve()),"selected_by_metric":"val_heldin/r2_mean","selected_metric_scope":"inner_validation_session_only"});sel.write_text(json.dumps(x));ev=d/"rt_ld_outer_eval.json";z=json.loads(ev.read_text());z["rt_ld"]={"identity_carrier":"aligned_full_afc4","gain_carrier":"aligned_full_afc4" if dn!="03_g_xls" else "strong_xls_v2"};ev.write_text(json.dumps(z))
def test_direct_nested_exact_root_and_output(tmp_path,monkeypatch):
 m=_m();monkeypatch.setattr(m,"_check",lambda:None);_tree(tmp_path,m,True);assert m.terminalize(tmp_path)["status"]=="PASS_BOTH_GATES"; (tmp_path/"extra").mkdir()
 with pytest.raises(ValueError):m.terminalize(tmp_path)
 (tmp_path/"extra").rmdir();out=tmp_path/"o.json";m.write(tmp_path,out);assert(out.stat().st_mode&0o777)==0o444
 with pytest.raises(FileExistsError):m.write(tmp_path,out)
def test_nested_missing_join_wrong_rtld_stop_edge_nan_hash(tmp_path,monkeypatch):
 m=_m();monkeypatch.setattr(m,"_check",lambda:None);_tree(tmp_path,m,False,.029999);assert m.terminalize(tmp_path)["status"]=="STOP_GATE_FAILED";p=tmp_path/"01_a0/rt_ld_outer_eval.json";x=json.loads(p.read_text());x.pop("data_dir");p.write_text(json.dumps(x))
 with pytest.raises(ValueError):m.terminalize(tmp_path)
 _tree(tmp_path,m);p=tmp_path/"03_g_xls/split_manifest.json";x=json.loads(p.read_text());x["rt_ld"]["gain_carrier"]="wrong";p.write_text(json.dumps(x))
 with pytest.raises(ValueError):m.terminalize(tmp_path)
