from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
import pytest,yaml
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/"scripts/rt_ld_fold0_terminalize_v2.py"
def _m():
 s=importlib.util.spec_from_file_location("t2",SCRIPT);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def _tree(t:Path,m,delta=.03):
 for dn,(rid,ma,gain) in m.ARMS.items():
  d=t/dn;(d/".hydra").mkdir(parents=True,exist_ok=True);(d/"checkpoints/best_ckpt").mkdir(parents=True,exist_ok=True);ck=d/"checkpoints/best_ckpt/x.ckpt";ck.write_bytes(b"x");cp=d/".hydra/config.yaml";cp.write_text(yaml.safe_dump({"run_id":rid,"seed":42,"model":{"rt_ld_arm":ma},"data":{"rt_ld_gain_source":gain,"outer_loso_fold":0}}));sel={"schema":"rt_clean_nested_loso_selection_receipt_v1","status":"PASS_FIT_INNER_SELECTION_ONLY","run_id":rid,"arm":"afc4_vel","outer_loso_fold":0,"seed":42,"best_model_path":str(ck.resolve()),"best_model_sha256":m._sha(ck),"config_sha256":m._sha(cp),"formal_heldout_opened":False,"outer_target_loaded_during_fit":False,"outer_target_query_labels_read_during_fit":False};(d/"rt_nested_selection_receipt.json").write_text(json.dumps(sel));(d/"split_manifest.json").write_text(json.dumps({"validation_protocol":"nested_loso","requested_side_feature_group":"afc4_vel","outer_loso_fold":0,"formal_heldout_opened":False,"nested_selection":{"outer_target_loaded_during_fit":False,"outer_target_query_labels_read_during_fit":False}}));r={"01_a0":.1,"02_g_full":.1+delta,"03_g_xls":.1}[dn];ev={"schema":"rt_clean_nested_loso_outer_eval_v1","status":"PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP","run_id":rid,"arm":"afc4_vel","seed":42,"outer_loso_fold":0,"target_backpropagation":False,"target_query_labels_used_for_calibration":False,"target_query_labels_used_for_normalization":False,"target_query_labels_used_for_checkpoint_selection":False,"optimizer_present":False,"model_training_mode":False,"target_query_labels_used_for_scoring_only":True,"model_state_unchanged":True,"model_state_sha256_before":"x","model_state_sha256_after":"x","r2_variance_weighted":r,"outer_target_session":"ses","outer_target_path":"/data/ses","query_start_trial":24,"window_size":50,"query_windows_evaluated":10,"data_dir":"/data","inner_train_sessions":["a"],"inner_validation_session":"b"};(d/"rt_ld_outer_eval.json").write_text(json.dumps(ev))
def test_pass_stop_edge_missing_invented_mismatch_nan_hash_and_duplicate(tmp_path:Path,monkeypatch):
 m=_m();monkeypatch.setattr(m,"_check",lambda:None);_tree(tmp_path,m);assert m.terminalize(tmp_path)["status"]=="PASS_BOTH_GATES";_tree(tmp_path,m,.029999);assert m.terminalize(tmp_path)["status"]=="STOP_GATE_FAILED";p=tmp_path/"01_a0/rt_ld_outer_eval.json";x=json.loads(p.read_text());x["formal_heldout_opened"]=False;p.write_text(json.dumps(x));
 with pytest.raises(ValueError):m.terminalize(tmp_path)
 _tree(tmp_path,m);p=tmp_path/"03_g_xls/rt_ld_outer_eval.json";x=json.loads(p.read_text());x["query_windows_evaluated"]=11;p.write_text(json.dumps(x));
 with pytest.raises(ValueError):m.terminalize(tmp_path)
 _tree(tmp_path,m);p=tmp_path/"02_g_full/rt_ld_outer_eval.json";x=json.loads(p.read_text());x["r2_variance_weighted"]=float("nan");p.write_text(json.dumps(x));
 with pytest.raises(ValueError):m.terminalize(tmp_path)
 _tree(tmp_path,m);p=tmp_path/"01_a0/rt_nested_selection_receipt.json";x=json.loads(p.read_text());x["best_model_sha256"]="0"*64;p.write_text(json.dumps(x));
 with pytest.raises(ValueError):m.terminalize(tmp_path)
 _tree(tmp_path,m);out=tmp_path/"out.json";m.write(tmp_path,out);assert(out.stat().st_mode&0o777)==0o444
 with pytest.raises(FileExistsError):m.write(tmp_path,out)
def test_actual_evaluator_source_has_no_formal_or_per_session_output():
 source=(ROOT/"../streaming_calibration_exp/src/rt_clean_nested_loso_eval.py").resolve().read_text();start=source.index('result: dict');section=source[start:source.index('if output_path is not None',start)]
 assert '"formal_heldout_opened"' not in section and '"per_session_r2"' not in section and '"r2_variance_weighted"' in section
