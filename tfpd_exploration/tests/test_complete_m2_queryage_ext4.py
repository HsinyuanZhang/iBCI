"""CPU tests for the launch-gated complete QueryAge stream proof."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pytest,torch
from tfpd_exploration.src.family_runtime_v1 import complete_m2_queryage_ext4 as c
from tfpd_exploration.src.family_runtime_v1.m2_family_queryage import M2FamilyQueryAgeRuntime,M2RuntimeBank
from tfpd_exploration.src.m2_dual_track_v1.contracts import make_stub_bank
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

def test_actual_queryage_w50_startup_rollover_both_arms_route_gate():
 torch.set_num_threads(1);bank0=make_stub_bank(seed=7);bank=M2RuntimeBank(bank0.E0,bank0.T,bank0.unit_mask)
 raw=np.zeros((109,96),np.float32);raw[49:]=np.random.default_rng(4).normal(size=(60,96)).astype(np.float32)
 for arm,model in zip(("FLAT","ROUTE"),make_paired_queryage_decoders(42),strict=True):
  if arm=="ROUTE":model.frontend.attn.routing.g.data.fill_(.2)
  rt=M2FamilyQueryAgeRuntime(model.eval(),bank)
  for tick in range(49,109):
   got=rt.predict(raw[tick:tick+1].copy());window=raw[tick-49:tick+1]
   with torch.inference_mode():want=(model.forward_last(torch.from_numpy(window[None]),bank,bank.unit_mask)/5).numpy()
   assert got.dtype==np.float32 and got.flags.owndata and got.flags.c_contiguous
   np.testing.assert_allclose(got,want,atol=1e-5,rtol=1e-5);np.testing.assert_array_equal(rt.raw[0].numpy(),window)

def test_collect_requires_absolute_fresh_separate_output(tmp_path):
 with pytest.raises(RuntimeError,match="fresh absolute"):
  c.collect_bindings(tmp_path,tmp_path,"a"*64,tmp_path,"b"*64,tmp_path)

def test_timeout_guards_before_any_forward(monkeypatch):
 monkeypatch.setenv("CUDA_VISIBLE_DEVICES","-1");monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
 with pytest.raises(TimeoutError,match="3600"):c._guard(time.monotonic()-c.HARD_SECONDS-1)

@pytest.mark.parametrize("runtime_kind",("uncached","cached"))
def test_actual_stream_four_sessions_both_arms_real_w50(monkeypatch,tmp_path,runtime_kind):
 """Reduced-cardinality fixture still exercises real models, runtime, W50, and ROUTE g=.2."""
 torch.set_num_threads(1);sessions=("a","b","c","d");monkeypatch.setattr(c.plan,"EXT4_SESSIONS",sessions);monkeypatch.setattr(c.plan,"EXT4_EXPECTED_WINDOWS",{s:2 for s in sessions});monkeypatch.setattr(c.native,"EXT4_TOTAL",8);monkeypatch.setattr(c,"EXPECTED_ENDPOINTS",8);monkeypatch.setattr(c,"EXPECTED_PUBLIC_RAW_BINS",8)
 cache=tmp_path/"cache";(cache/"ext4").mkdir(parents=True);monkeypatch.setattr(c.data,"cache_root",lambda:cache)
 banks={};rows=[];rng=np.random.default_rng(9)
 for j,s in enumerate(sessions):
  d=cache/"ext4"/s;d.mkdir();raw=np.zeros((51,96),np.float32);raw[49:]=rng.normal(size=(2,96)).astype(np.float32);starts=np.array([0,1],np.int64);target=np.array([[0.,0.],[1.,1.]],np.float32)
  np.save(d/"X_store.npy",raw);np.save(d/"eligible_starts.npy",starts);np.save(d/"target_store.npy",target)
  for n in c.native.REQUIRED_CACHE:
   if not (d/n).exists():(d/n).write_bytes(b"fixture")
  b=make_stub_bank(seed=j);banks[s]=b;rows.append({"session":s,"window_count":2,"files":{n:c.sha(d/n) for n in c.native.REQUIRED_CACHE},"ordered_window_starts_sha256":c.core.array_sha256(starts),"target_sha256":c.core.array_sha256(target)})
 monkeypatch.setattr(c.data,"load_session_bank",lambda surface,s,device:banks[s]);monkeypatch.setenv("CUDA_VISIBLE_DEVICES","-1");monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
 artifacts={}
 for arm,model in zip(("FLAT","ROUTE"),make_paired_queryage_decoders(42),strict=True):
  if arm=="ROUTE":model.frontend.attn.routing.g.data.fill_(.2)
  model.eval();state=tmp_path/f"{arm}.pt";torch.save(model.state_dict(),state);parts={k:[] for k in ("prediction","target","start","session")}
  for row in rows:
   d=cache/"ext4"/row["session"];raw=np.load(d/"X_store.npy");starts=np.load(d/"eligible_starts.npy");bank=M2RuntimeBank(banks[row["session"]].E0,banks[row["session"]].T,banks[row["session"]].unit_mask)
   with torch.inference_mode():pred=(model.forward_last(torch.from_numpy(np.stack([raw[x:x+50] for x in starts])),bank,bank.unit_mask)/5).numpy().astype(np.float64)
   parts["prediction"].append(pred);parts["target"].append(np.load(d/"target_store.npy").astype(np.float64));parts["start"].append(starts);parts["session"].append(np.array([row["session"]]*2))
  gold={k:np.concatenate(v) for k,v in parts.items()};result,arrays=c._stream(arm,{"path":str(state),"sha256":c.sha(state)},gold,rows,time.monotonic(),lambda *x:None,runtime_kind)
  assert result["endpoint_count"]==result["public_raw_bins"]==8;np.testing.assert_allclose(arrays["prediction"],gold["prediction"],atol=1e-5)
  archive=tmp_path/f"{arm}.npz";np.savez_compressed(archive,**gold);artifacts[arm]={"selected":{"path":str(state),"sha256":c.sha(state)},"native_archive_path":str(archive),"native_archive_sha256":c.sha(archive),"equal_session_r2":result["equal_session_r2"],"pooled_r2":result["pooled_r2"],"rows":result["rows"]}
 # Full integration: only the fresh external evaluator authority is injected;
 # cache/session-bank and real W50 model/runtime paths remain actual.
 native_root=(tmp_path/"native").resolve();native_root.mkdir();fresh={"selected":{a:artifacts[a]["selected"] for a in artifacts},"rows":rows}
 receipt={"schema":c.native.SCHEMA,"status":"COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY","pre":{**fresh,"output":str(native_root)},"post":{**fresh,"output":str(native_root)},"arms":artifacts}
 (native_root/"receipt.json").write_text(json.dumps(receipt));monkeypatch.setattr(c.native,"authority",lambda *args:fresh)
 run_root=(tmp_path/"run").resolve();final_root=(tmp_path/"final").resolve();output=(tmp_path/"out").resolve();binding=c.collect_bindings(run_root,final_root,"a"*64,native_root,c.sha(native_root/"receipt.json"),output,runtime_kind)
 auth=(tmp_path/"auth.json").resolve();auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":binding}));monkeypatch.setenv(c.GO_ENV,"1")
 wrong="cached" if runtime_kind=="uncached" else "uncached"
 with pytest.raises(RuntimeError,match="external authorization"):
  c.run(run_root,final_root,"a"*64,native_root,c.sha(native_root/"receipt.json"),output,auth,c.sha(auth),wrong)
 assert not output.exists()
 done=c.run(run_root,final_root,"a"*64,native_root,c.sha(native_root/"receipt.json"),output,auth,c.sha(auth),runtime_kind)
 assert done["authority_pre"]==done["authority_post"]==binding and (output/"receipt.json").is_file() and done["runtime_kind"]==runtime_kind
 assert done["authority_pre_sha256"]==c.sha(output/"authority_pre.json")

def test_native_authority_rejects_mutated_selected_export_before_run(tmp_path,monkeypatch):
 root=(tmp_path/"native").resolve();root.mkdir();state=root/"s.pt";state.write_bytes(b"ok");arc=root/"a.npz";np.savez(arc,x=np.array([1]))
 fresh={"selected":{a:{"path":str(state),"sha256":c.sha(state)} for a in ("FLAT","ROUTE")}}
 receipt={"schema":c.native.SCHEMA,"status":"COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY","pre":{**fresh,"output":str(root)},"post":{**fresh,"output":str(root)},"arms":{a:{"selected":fresh["selected"][a],"native_archive_path":str(arc),"native_archive_sha256":c.sha(arc)} for a in ("FLAT","ROUTE")}}
 (root/"receipt.json").write_text(json.dumps(receipt));monkeypatch.setattr(c.native,"authority",lambda *x:{"selected":{a:{"path":str(state),"sha256":c.sha(state)} for a in ("FLAT","ROUTE")}})
 state.write_bytes(b"changed")
 with pytest.raises(RuntimeError,match="fresh pre/post authority"):
  c._native_authority((tmp_path/"run").resolve(),(tmp_path/"fin").resolve(),"a"*64,root,c.sha(root/"receipt.json"))
