from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
import pytest,yaml
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/"scripts/rt_ld_fold0_terminalize_v1.py"
def _m():
 s=importlib.util.spec_from_file_location("term",SCRIPT);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def _tree(tmp:Path,m,*,delta:float=.03):
 for directory,(rid,arm,gain) in m.ARMS.items():
  d=tmp/directory;(d/".hydra").mkdir(parents=True,exist_ok=True);(d/"checkpoints/best_ckpt").mkdir(parents=True,exist_ok=True);ck=d/"checkpoints/best_ckpt/x.ckpt";ck.write_bytes(b"x")
  cfg={"run_id":rid,"seed":42,"model":{"rt_ld_arm":arm},"data":{"rt_ld_gain_source":gain,"outer_loso_fold":0},"callbacks":{"rt_nested_selection_receipt":{"monitor":"val_heldin/r2_mean"}}};cp=d/".hydra/config.yaml";cp.write_text(yaml.safe_dump(cfg))
  split={"validation_protocol":"nested_loso","requested_side_feature_group":"afc4_vel","outer_loso_fold":0,"nested_selection":{"checkpoint_metric":"val_heldin/r2_mean"},"rt_ld":{"identity_carrier":"aligned_full_afc4","gain_carrier":"aligned_full_afc4" if gain=="full" else "strong_xls_v2","identity_never_receives_xls_v2":True}};sp=d/"split_manifest.json";sp.write_text(json.dumps(split))
  sel={"schema":"rt_clean_nested_loso_selection_receipt_v1","status":"PASS_FIT_INNER_SELECTION_ONLY","run_id":rid,"arm":"afc4_vel","outer_loso_fold":0,"seed":42,"best_model_path":str(ck.resolve()),"best_model_sha256":m._sha(ck),"config_sha256":m._sha(cp)};(d/"rt_nested_selection_receipt.json").write_text(json.dumps(sel))
  r={"01_a0":.1,"02_g_full":.1+delta,"03_g_xls":.1}[directory];ev={"schema":"rt_clean_nested_loso_outer_eval_v1","status":"PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP","run_id":rid,"arm":"afc4_vel","seed":42,"outer_loso_fold":0,"formal_heldout_opened":False,"target_backpropagation":False,"target_query_labels_used_for_calibration":False,"target_query_labels_used_for_checkpoint_selection":False,"model_state_unchanged":True,"model_state_sha256_before":"a","model_state_sha256_after":"a","outer_target_session":"ses","query_start_trial":24,"window_size":50,"query_windows_evaluated":10,"r2_variance_weighted":r,"per_session_r2":{"ses":{"r2":r,"samples":10}}};(d/"rt_ld_outer_eval.json").write_text(json.dumps(ev))
def test_missing_arm_wrong_arm_query_nan_threshold_and_hash_drift_fail_closed(tmp_path:Path,monkeypatch):
 m=_m();_tree(tmp_path,m);monkeypatch.setattr(m,"_check_prepared",lambda:None)
 assert m.terminalize(tmp_path)["status"]=="PASS_BOTH_GATES"
 _tree(tmp_path,m,delta=.029999);assert m.terminalize(tmp_path)["status"]=="STOP_GATE_FAILED"
 (tmp_path/"01_a0/rt_ld_outer_eval.json").unlink()
 with pytest.raises(FileNotFoundError):m.terminalize(tmp_path)
def test_contract_mismatches_and_repeat_write_rejected(tmp_path:Path,monkeypatch):
 m=_m();_tree(tmp_path,m);monkeypatch.setattr(m,"_check_prepared",lambda:None)
 p=tmp_path/"02_g_full/rt_ld_outer_eval.json";d=json.loads(p.read_text());d["per_session_r2"]["ses"]["r2"]=float("nan");p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match="non-finite"):m.terminalize(tmp_path)
 _tree(tmp_path,m);out=tmp_path/"receipt.json";m.write(tmp_path,out);assert (out.stat().st_mode&0o777)==0o444
 with pytest.raises(FileExistsError):m.write(tmp_path,out)
def test_wrong_arm_query_mismatch_and_selection_hash_drift_fail_closed(tmp_path:Path,monkeypatch):
 m=_m();_tree(tmp_path,m);monkeypatch.setattr(m,"_check_prepared",lambda:None)
 cfg=tmp_path/"02_g_full/.hydra/config.yaml";body=yaml.safe_load(cfg.read_text());body["model"]["rt_ld_arm"]="wrong";cfg.write_text(yaml.safe_dump(body))
 with pytest.raises(ValueError,match="config arm identity"):m.terminalize(tmp_path)
 _tree(tmp_path,m);ev=tmp_path/"03_g_xls/rt_ld_outer_eval.json";body=json.loads(ev.read_text());body["query_windows_evaluated"]=11;ev.write_text(json.dumps(body))
 with pytest.raises(ValueError,match="pairing contract"):m.terminalize(tmp_path)
 _tree(tmp_path,m);sel=tmp_path/"01_a0/rt_nested_selection_receipt.json";body=json.loads(sel.read_text());body["best_model_sha256"]="0"*64;sel.write_text(json.dumps(body))
 with pytest.raises(ValueError,match="selection hash drift"):m.terminalize(tmp_path)
