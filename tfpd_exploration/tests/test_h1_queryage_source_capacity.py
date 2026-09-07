"""Actual CPU QueryAge coverage for the source-capacity runner."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from tfpd_exploration.src.h1_queryage_family_v1 import source_capacity as s
from tfpd_exploration.src.h1_queryage_family_v1.model import make_queryage_localbalanced_pair
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.src.h1_optimized_v2 import cache as cache_module
from tfpd_exploration.src.family_runtime_v1 import diagnose_h1_selected_source208 as source_diag

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

def _row(seed=1):
 g=np.random.default_rng(seed);n=720
 return {"neural":g.normal(0,.02,(n,176)).astype(np.float32),"velocity":g.normal(0,.01,(n,7)).astype(np.float32),"eval_mask":np.ones(n,np.bool_),"bank":{"E0":torch.randn(176,700,generator=torch.Generator().manual_seed(seed))*.02,"T":torch.randn(176,4,generator=torch.Generator().manual_seed(seed+90))*.02,"unit_mask":torch.ones(176,dtype=torch.bool)}}

def _actual():
 flat,route=make_queryage_localbalanced_pair(seed=42);models={"flat":flat,"route":route};opts={a:torch.optim.AdamW(groups(m),lr=2e-4,weight_decay=.01) for a,m in models.items()};return models,opts

def test_actual_queryage_micro_weight_and_cpu_checkpoint_resume_rng(tmp_path):
 row=_row();starts=np.arange(16,dtype=np.int64);x,y=s.batch(torch,row,starts,torch.device("cpu"));bank=H1Bank(**row["bank"]);models,opts=_actual()
 # The reported update loss is the exact mean of the four equal micro losses.
 with torch.no_grad(): expected={a:float(np.mean([torch.nn.functional.mse_loss(models[a].forward_last(x[i:i+4],bank,dropout_keep=None),y[i:i+4]).item() for i in range(0,16,4)])) for a in s.ARMS}
 got=s.update(torch,models,opts,x,y,bank,lambda:None,require_gate=True)
 assert got==pytest.approx(expected,abs=1e-6)
 binding={"fixture":"actual"};payload=s.checkpoint(torch,models,opts,1,binding);path=tmp_path/"actual.pt";s.atomic_torch(payload,path,torch);loaded=torch.load(path,map_location="cpu",weights_only=False)
 # Continuation after restore is byte-identical to uninterrupted continuation,
 # including the Python/NumPy/Torch RNG checkpoint state.
 s.update(torch,models,opts,x,y,bank,lambda:None)
 after={a:torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone() for a,m in models.items()}
 s.restore(torch,loaded,models,opts,1,binding);s.update(torch,models,opts,x,y,bank,lambda:None)
 for a in s.ARMS: torch.testing.assert_close(after[a],torch.nn.utils.parameters_to_vector(models[a].parameters()),atol=0,rtol=0)
 assert all(torch.isfinite(v).all() for v in after.values())
 s.restore(torch,loaded,models,opts,1,binding);draw1=(torch.rand(3),np.random.rand(3),random.random())
 s.restore(torch,loaded,models,opts,1,binding);draw2=(torch.rand(3),np.random.rand(3),random.random())
 torch.testing.assert_close(draw1[0],draw2[0],atol=0,rtol=0);np.testing.assert_array_equal(draw1[1],draw2[1]);assert draw1[2]==draw2[2]

def test_actual_bank_mask_none_ignores_inactive_unit():
 flat,_=make_queryage_localbalanced_pair(seed=42);flat.eval();row=_row(18);row["bank"]["unit_mask"][0]=False;bank=H1Bank(**row["bank"]);x=torch.from_numpy(row["neural"][:700][None]);changed=x.clone();changed[:,:,0]=1e6
 with torch.inference_mode(): a=flat.forward_last(x,bank,dropout_keep=None);b=flat.forward_last(changed,bank,dropout_keep=None)
 torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

def test_actual_smoke20_runner_source_only_no_score_or_checkpoint(tmp_path,monkeypatch):
 """Real factory/update/run; only authorization/cache-source boundaries are injected."""
 cache={"train":{f"s{i:02d}":_row(i) for i in range(13)}};fixed={name:np.arange(16,dtype=np.int64) for name in cache["train"]}
 test_config={**s.CONFIG,"modes":{**s.CONFIG["modes"],"smoke20":2},"limits":{**s.CONFIG["limits"],"smoke20":1200}}
 monkeypatch.setattr(s,"CONFIG",test_config)
 monkeypatch.setattr(s,"smoke_forecasts",lambda times:{"capacity260":1.,"total1040":1.})
 out=(tmp_path/"smoke").resolve();binding={"fixture":"smoke","output":str(out)}
 monkeypatch.setattr(s,"bindings",lambda output,**kw:{"fixture":"smoke","output":str(output)})
 monkeypatch.setattr(cache_module,"validate_authority",lambda *a:None);monkeypatch.setattr(source_diag,"_require_ids",lambda:fixed);monkeypatch.setattr(source_diag,"validate_manifest_against_cache",lambda *a:None)
 # Smoke must never reach score/checkpoint paths.
 monkeypatch.setattr(s,"checkpoint",lambda *a:pytest.fail("smoke checkpoint"));monkeypatch.setattr(s,"atomic_torch",lambda *a:pytest.fail("smoke disk checkpoint"))
 auth=(tmp_path/"auth.json").resolve();auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","bindings":binding}));monkeypatch.setenv("H1_QUERYAGE_CAPACITY_GO","1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
 result=s.run(out,auth,s.sha(auth),mode="smoke20",allow_cpu_for_test=True,source_loader=lambda:(cache,{"fixture":"authority"}))
 assert result["status"]=="PASS_SMOKE_NO_CAPACITY_CLAIM" and result["updates"]==2 and result["scores"]=={} and result["checkpoint"] is None
 assert np.isfinite([row[arm] for row in result["losses"] for arm in s.ARMS]).all()
 assert isinstance(result["source_order_target_mask_sha256"],str) and len(result["source_order_target_mask_sha256"])==64
 assert result["initial_actual_g0_parity_max_abs"]==0.0

def test_preflight_bad_source_smoke_and_path_gates_before_model(tmp_path,monkeypatch):
 out=(tmp_path/"out").resolve();auth=(tmp_path/"auth").resolve();auth.write_text("{}")
 monkeypatch.setenv("H1_QUERYAGE_CAPACITY_GO","1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
 monkeypatch.setattr(s,"bindings",lambda output,**kw:{"x":1})
 with pytest.raises(RuntimeError,match="external authorization"): s.preflight(out,auth,s.sha(auth),"smoke20")
 with pytest.raises(RuntimeError,match="capacity requires bound smoke"): s.preflight(out,auth,s.sha(auth),"capacity260")

def test_pure_forecast_and_capacity_any260_both1040_gates():
 assert s.smoke_forecasts([1.]*20)["capacity260"]>0
 scores={a:{"pooled":{"r2_concat_float64":.6,"prediction_std_float64":.6,"target_std_float64":1.}} for a in s.ARMS}
 loss=[{"flat":2.,"route":2.}]*32+[{"flat":1.,"route":1.}]*32
 assert s.capacity_eligible("capacity260",scores,loss) is True and s.capacity_eligible("extend1040",scores,loss) is True
 scores["route"]["pooled"]["r2_concat_float64"]=.01
 assert s.capacity_eligible("capacity260",scores,loss) is True and s.capacity_eligible("extend1040",scores,loss) is False
