from __future__ import annotations
import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];RUNNER=ROOT/"scripts/rt_ld_device_handoff_v8.py"
def _m():
 s=importlib.util.spec_from_file_location("v8",RUNNER);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def _arm(tmp:Path,m,exp:str):
 spec=m.ARM_SPECS[exp];rid,gain=spec["run_id"],spec["data_rt_ld_gain_source"];d=tmp;(d/"rt_ld_outer_eval.json").write_text(json.dumps({"schema":"rt_clean_nested_loso_outer_eval_v1","status":"PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP","run_id":rid,"arm":"afc4_vel","seed":42,"outer_loso_fold":0,"target_backpropagation":False,"target_query_labels_used_for_calibration":False,"target_query_labels_used_for_normalization":False,"target_query_labels_used_for_checkpoint_selection":False,"optimizer_present":False,"model_training_mode":False,"target_query_labels_used_for_scoring_only":True,"model_state_unchanged":True,"model_state_sha256_before":"x","model_state_sha256_after":"x","rt_ld":{"identity_carrier":"aligned_full_afc4","gain_carrier":"aligned_full_afc4" if gain=="full" else "strong_xls_v2","identity_never_receives_xls_v2":True}}));(d/"rt_nested_selection_receipt.json").write_text(json.dumps({"formal_heldout_opened":False,"outer_target_loaded_during_fit":False,"outer_target_query_labels_read_during_fit":False}));(d/"split_manifest.json").write_text(json.dumps({"formal_heldout_opened":False,"nested_selection":{"outer_target_loaded_during_fit":False,"outer_target_query_labels_read_during_fit":False}}));return d
def test_real_outer_schema_scope_contract_and_reject_invented_formal_field(tmp_path:Path):
 m=_m();d=_arm(tmp_path,m,"rt_ld_a0_full_m24_fold0_seed42");m._outer_scope_v8(d,"rt_ld_a0_full_m24_fold0_seed42")
 p=d/"rt_ld_outer_eval.json";x=json.loads(p.read_text());x["formal_heldout_opened"]=False;p.write_text(json.dumps(x))
 with pytest.raises(ValueError,match="invented"):m._outer_scope_v8(d,"rt_ld_a0_full_m24_fold0_seed42")
def test_v7_retirement_and_sequential_three_arm_progression_contract():
 m=_m();clean=("exited",False,[]);assert not m._eligible_from_probes(False,True,clean,clean);assert m._eligible_from_probes(True,True,clean,clean)
 assert list(m.ARM_SPECS)==["rt_ld_a0_full_m24_fold0_seed42","rt_ld_g_full_m24_fold0_seed42","rt_ld_g_xls_m24_fold0_seed42"]
