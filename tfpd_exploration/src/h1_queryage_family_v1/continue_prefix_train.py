"""Additive, authorization-bound absolute-epoch 13..24 QueryAge continuation.

This module intentionally has no CLI and never changes the completed 12 epoch
formal implementation.  It provides the pure schedule/checkpoint contracts
used by the paired continuation launcher.
"""
from __future__ import annotations
import hashlib,json,os,random,tempfile,time
from pathlib import Path
from typing import Any
import numpy as np
from . import formal_prefix_train as parent

SEED,EPOCH_UPDATES,MICRO,EFFECTIVE,LR,EMA=42,731,8,32,1e-4,.9995
PARENT_EPOCHS,FINAL_EPOCH=12,24
ARMS,ARM_DEVICE=parent.ARMS,parent.ARM_DEVICE
torch=parent.torch
PROTOCOL_DOC=parent.ROOT/"docs/PROTOCOL_H1_QUERYAGE_CONTINUE_24_V1_20260906.md"

PROTOCOL={"schema":"h1_queryage_prefix_continue24_v1","parent_epochs":12,"continuation_epochs":[13,24],"seed":SEED,"updates_per_epoch":EPOCH_UPDATES,"microbatch":MICRO,"effective_batch":EFFECTIVE,"lr":LR,"ema":EMA,"resume":"strict parent epoch12 RAW+optimizer+EMA+RNG; no reset"}
def sha(path:Path)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value:Any)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def protocol_sha256()->str:return digest(PROTOCOL)
def warmup_lr(*,epoch:int,global_step:int)->float:
 if not (PARENT_EPOCHS<epoch<=FINAL_EPOCH and (epoch-1)*EPOCH_UPDATES+1<=global_step<=epoch*EPOCH_UPDATES):raise ValueError("continuation absolute step geometry")
 return LR
def earliest_argmax_24(scores):
 if len(scores)!=FINAL_EPOCH or [e for e,_ in scores]!=list(range(1,FINAL_EPOCH+1)) or not np.isfinite([v for _,v in scores]).all():raise RuntimeError("requires finite ordered epochs 1..24")
 return max(scores,key=lambda x:x[1])
def parent_checkpoint_contract(payload:dict[str,Any])->None:
 if (payload.get("schema")!="h1_queryage_formal_prefix_end_epoch_checkpoint_v1" or payload.get("epoch")!=PARENT_EPOCHS or payload.get("next_batch_index")!=EPOCH_UPDATES or payload.get("global_step")!=PARENT_EPOCHS*EPOCH_UPDATES or payload.get("ema",{}).get("n_updates")!=PARENT_EPOCHS*EPOCH_UPDATES):raise RuntimeError("parent epoch12 strict-resume contract drift")
def checkpoint_payload(*,model,optimizer,ema,epoch,identities,arm,shared_init_sha256,bindings,parent_checkpoint_sha256):
 if arm not in ARMS or not PARENT_EPOCHS<epoch<=FINAL_EPOCH or ema.n_updates!=epoch*EPOCH_UPDATES:raise RuntimeError("continuation checkpoint schedule/EMA drift")
 return {"schema":"h1_queryage_prefix_continue_end_epoch_checkpoint_v1","arm":arm,"epoch":epoch,"next_batch_index":EPOCH_UPDATES,"global_step":epoch*EPOCH_UPDATES,"models":parent.copy.deepcopy(model.state_dict()),"optimizer":parent.copy.deepcopy(optimizer.state_dict()),"ema":parent.copy.deepcopy(ema.checkpoint_state()),"identities":identities,"shared_init_sha256":shared_init_sha256,"bindings_sha256":digest(bindings),"parent_epoch12_checkpoint_sha256":parent_checkpoint_sha256,"raw_state_sha256":parent.state_digest(model.state_dict()),"ema_state_sha256":parent.state_digest(ema.shadow),"rng":{"torch":parent.torch.get_rng_state(),"cuda":parent.torch.cuda.get_rng_state_all(),"numpy":np.random.get_state(),"python":random.getstate()}}
def code_closure():
 from tfpd_exploration.src.family_runtime_v1 import complete_h1_queryage_source as proof
 return {**parent.code_closure(),"continuation_train":sha(Path(__file__)),"continuation_launcher":sha(Path(__file__).with_name("continue_prefix_launcher.py")),"parent_admission_audit":sha(Path(proof.__file__))}
def collect_bindings(output:Path,*,parent_formal:Path,parent_receipt:Path|None=None,parent_authorization:Path|None=None)->dict[str,Any]:
 output,parent_formal=Path(output),Path(parent_formal)
 if any(not p.is_absolute() or p.resolve()!=p for p in (output,parent_formal)):raise RuntimeError("canonical absolute paths required")
 if output==parent_formal or parent_formal in output.parents or output in parent_formal.parents:raise RuntimeError("disjoint continuation output required")
 if parent_receipt is not None and Path(parent_receipt)!=parent_formal/"receipt.json":raise RuntimeError("exact parent receipt required")
 if parent_authorization is not None and Path(parent_authorization)!=parent_formal/"input_authority.json":raise RuntimeError("exact parent input authority required")
 old=parent.require_frozen_authority(parent_formal)
 files={PROTOCOL_DOC,parent_formal/"receipt.json",parent_formal/"selection_freeze.json",parent_formal/"input_authority.json",Path(old["authorization_path"]),*[Path(p) for p in old["bindings"]["inputs"]]}
 for arm in ARMS:files.update((parent_formal/"workers"/f"{arm}_complete.json",parent_formal/"barrier"/f"{arm}.ready.json",parent_formal/"checkpoints"/f"{arm}_epoch_012.pt"))
 if any(not p.is_file() for p in files):raise FileNotFoundError("completed parent formal artifacts required")
 return {"protocol":PROTOCOL,"protocol_sha256":protocol_sha256(),"code_closure":code_closure(),"output":str(output),"parent_formal":str(parent_formal),"inputs":{str(p):sha(p) for p in sorted(files)},"arms":ARM_DEVICE,"parent_bindings_sha256":digest(old["bindings"])}
def require_authority(output:Path)->dict[str,Any]:
 frozen=json.loads((Path(output)/"input_authority.json").read_text());wrapper=frozen.get("authority",{});b=wrapper.get("bindings",{});auth=Path(frozen.get("authorization_path","") )
 from . import continue_prefix_launcher as launch
 if not auth.is_absolute() or auth.resolve()!=auth or sha(auth)!=frozen.get("authorization_sha256") or json.loads(auth.read_text())!=wrapper or wrapper.get("schema")!=launch.SCHEMA or wrapper.get("status")!="ROOT_REVIEW_GO":raise RuntimeError("continuation external authority drift")
 if b!=collect_bindings(Path(output),parent_formal=Path(b.get("parent_formal",""))) or wrapper.get("output")!=str(output) or wrapper.get("protocol")!={"path":str(PROTOCOL_DOC),"sha256":sha(PROTOCOL_DOC)} or wrapper.get("launcher_sha256")!=sha(Path(launch.__file__)) or wrapper.get("train_sha256")!=sha(Path(__file__)):raise RuntimeError("continuation binding/code/output drift")
 if wrapper.get("limits")!={"wall_seconds":21600,"peak_rss_bytes":22<<30,"torch_threads":1,"torch_interop_threads":1,"physical_gpu":ARM_DEVICE,"cpu_affinity":launch.CPU} or wrapper.get("rules")!={"epochs":list(range(13,25)),"selection":"EMA R2 on fixed 2908 selection across 1..24; earliest tie","endpoint":24,"disclosure":"authorized after inspecting epoch1..12 trajectory; not initially preregistered 24"}:raise RuntimeError("continuation resource/selection rules drift")
 elapsed=wrapper.get("parent_audit",{}).get("parent_elapsed_seconds");forecast=wrapper.get("resource_forecast",{})
 if not isinstance(elapsed,(int,float)) or not np.isfinite(elapsed) or elapsed<=0 or forecast!={"formula":"parent_actual_elapsed_seconds * 1.5 + 1800","parent_actual_elapsed_seconds":elapsed,"forecast_seconds":elapsed*1.5+1800.} or not 0<forecast["forecast_seconds"]<21600:raise RuntimeError("continuation resource evidence drift")
 return frozen
def recursive_same(left,right):
 return parent.same(left,right)
def rng_state():
 return {"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"numpy":np.random.get_state(),"python":random.getstate()}
def tree_digest(value):
 """Stable recursive state hash, independent of device and pickle container."""
 h=hashlib.sha256()
 def visit(x):
  h.update(type(x).__name__.encode()+b"|")
  if isinstance(x,torch.Tensor):
   a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
  elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):visit(k);visit(x[k])
  elif isinstance(x,(tuple,list)):
   h.update(str(len(x)).encode())
   for a in x:visit(a)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()
def _assert_restored(payload,model,optimizer,ema):
 if not recursive_same(optimizer.state_dict(),payload["optimizer"]) or not recursive_same(ema.checkpoint_state(),payload["ema"]) or not recursive_same(rng_state(),payload["rng"]) or parent.state_digest(model.state_dict())!=payload["raw_state_sha256"] or parent.state_digest(ema.shadow)!=payload["ema_state_sha256"]:raise RuntimeError("strict RAW/AdamW/EMA/RNG restore drift")
 if ema.decay!=EMA or ema.n_updates!=payload["global_step"] or any(g["lr"]!=LR for g in optimizer.param_groups):raise RuntimeError("restored EMA or LR reset")
 # The byte-bound parent is authoritative; check every saved moment/step and
 # param group by recursive equality, without inventing missing optimizer state.
 if not optimizer.state_dict()["state"]:raise RuntimeError("resume has no AdamW moments")
 for state in optimizer.state_dict()["state"].values():
  if not {"step","exp_avg","exp_avg_sq"}.issubset(state) or any(isinstance(v,torch.Tensor) and not bool(torch.isfinite(v).all()) for v in state.values()):raise RuntimeError("invalid restored AdamW moment/step")
def strict_parent_restore(*,payload,model,optimizer,ema,identities,arm,shared_init_sha256,bindings):
 """Restore epoch12 including AdamW state, EMA and all RNG streams."""
 parent_checkpoint_contract(payload);parent.strict_restore(payload=payload,model=model,optimizer=optimizer,ema=ema,epoch=12,identities=identities,arm=arm,shared_init_sha256=shared_init_sha256,bindings=bindings)
 _assert_restored(payload,model,optimizer,ema)
def strict_restore(*,payload,model,optimizer,ema,epoch,identities,arm,shared_init_sha256,bindings,parent_epoch12_sha256):
 if not 12<epoch<=24:raise RuntimeError("continuation restore epoch geometry")
 expected={"schema":"h1_queryage_prefix_continue_end_epoch_checkpoint_v1","arm":arm,"epoch":epoch,"next_batch_index":EPOCH_UPDATES,"global_step":epoch*EPOCH_UPDATES,"identities":identities,"shared_init_sha256":shared_init_sha256,"bindings_sha256":digest(bindings),"parent_epoch12_checkpoint_sha256":parent_epoch12_sha256}
 if any(payload.get(k)!=v for k,v in expected.items()) or payload.get("ema",{}).get("n_updates")!=epoch*EPOCH_UPDATES:raise RuntimeError("continuation checkpoint identity drift")
 if parent.state_digest(payload["models"])!=payload.get("raw_state_sha256") or parent.state_digest(payload["ema"]["shadow"])!=payload.get("ema_state_sha256"):raise RuntimeError("continuation checkpoint state drift")
 model.load_state_dict(payload["models"],strict=True);optimizer.load_state_dict(payload["optimizer"]);ema.load_checkpoint_state(payload["ema"])
 ema.shadow={k:v.to(next(model.parameters()).device) for k,v in ema.shadow.items()}
 torch.set_rng_state(payload["rng"]["torch"]);torch.cuda.set_rng_state_all(payload["rng"]["cuda"]);np.random.set_state(payload["rng"]["numpy"]);random.setstate(payload["rng"]["python"])
 _assert_restored(payload,model,optimizer,ema)

def dropout_keep(*,n,epoch,batch_index,bank_mask,device):
 if not (1<=epoch<=24 and 0<=batch_index<EPOCH_UPDATES and n>0):raise ValueError("dropout schedule geometry")
 seed=int.from_bytes(hashlib.sha256(f"{SEED}|keep|{epoch}|{batch_index}".encode()).digest()[:8],"little")
 keep=torch.rand((n,bank_mask.numel()),generator=torch.Generator(device="cpu").manual_seed(seed))>=parent.DROP_P
 keep &= bank_mask.detach().cpu().bool().view(1,-1)
 if not bool(keep.any(dim=1).all()):raise RuntimeError("dropout produced empty row")
 return keep.to(device)
def prefix_and_keep(x,*,epoch,batch_index,bank_mask,device):
 from tfpd_exploration.src.h1_family_v1.cold_history import apply_cold_history
 prefixed,lengths=apply_cold_history(x,seed=SEED,epoch=epoch,batch_id=batch_index,probability=.5)
 if not torch.equal(lengths.detach().cpu(),parent.prefix_lengths(len(x),epoch=epoch,batch_index=batch_index)):raise RuntimeError("actual prefix differs from precommitted law")
 keep=dropout_keep(n=len(x),epoch=epoch,batch_index=batch_index,bank_mask=bank_mask,device=device)
 if not torch.equal(prefixed[:,-1],x[:,-1]):raise RuntimeError("prefix changed current bin")
 return prefixed.to(device),lengths.detach().cpu(),keep
def identity_digest(ordered,*,cache,epoch):
 sampler=parent.sampler_identity_digest(ordered);keeps,prefixes=hashlib.sha256(),hashlib.sha256()
 for batch_index,(session,starts) in enumerate(ordered):
  row,a=cache["train"][session],np.asarray(starts,dtype=np.int64)
  lengths=parent.prefix_lengths(len(a),epoch=epoch,batch_index=batch_index)
  keep=dropout_keep(n=len(a),epoch=epoch,batch_index=batch_index,bank_mask=row["bank"]["unit_mask"],device=torch.device("cpu"))
  for h in (keeps,prefixes):h.update(session.encode());h.update(a.tobytes())
  keeps.update(keep.numpy().tobytes());prefixes.update(lengths.numpy().tobytes())
 return {"sampler_sha256":sampler,"keep_sha256":keeps.hexdigest(),"prefix_sha256":prefixes.hexdigest()}
def _all_epoch_identities(cache):
 from tfpd_exploration.src.h1_optimized_v4.paired_train import batches
 result={}
 for epoch in range(13,25):
  ordered=batches(cache,epoch)
  if sum(len(starts) for _,starts in ordered)!=23212:raise RuntimeError("source sampler cardinality drift")
  result[str(epoch)]=identity_digest(ordered,cache=cache,epoch=epoch)
 return result
def _train_update(model,optimizer,ema,x,y,bank,*,epoch,batch_index,device,guard=lambda:None):
 prefixed,lengths,keep=prefix_and_keep(x,epoch=epoch,batch_index=batch_index,bank_mask=bank.unit_mask,device=device)
 model.train();optimizer.zero_grad(set_to_none=True);total=0.
 for offset in range(0,len(prefixed),MICRO):
  guard();n=len(prefixed[offset:offset+MICRO]);loss=parent.F.mse_loss(model.forward_last(prefixed[offset:offset+MICRO],bank,dropout_keep=keep[offset:offset+MICRO]),y[offset:offset+MICRO]);guard()
  if not bool(torch.isfinite(loss)):raise RuntimeError("nonfinite continuation loss")
  (loss*(n/len(prefixed))).backward();total+=float(loss.detach())*(n/len(prefixed));guard()
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);guard();optimizer.step();ema.update_after_step(model);guard()
 return total,lengths

def _runtime_setup(arm,output,physical_gpu):
 if arm not in ARMS or physical_gpu!=ARM_DEVICE[arm]:raise RuntimeError("arm/device drift")
 frozen=require_authority(output)
 if os.environ.get("H1_QUERYAGE_CONTINUE_24_GO")!="1" or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(physical_gpu) or not torch.cuda.is_available() or torch.cuda.current_device()!=0:raise RuntimeError("explicit GO and exact visible GPU required")
 from .continue_prefix_launcher import CPU
 low,high=map(int,CPU[arm].split("-"))
 if os.sched_getaffinity(0)!=set(range(low,high+1)):raise RuntimeError("worker CPU affinity drift")
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:
  if torch.get_num_interop_threads()!=1:raise
 torch.cuda.reset_peak_memory_stats();return frozen,torch.device("cuda:0")
def _new_model(arm,device):
 from tfpd_exploration.src.h1_queryage_family_v1.model import make_queryage_localbalanced_pair
 from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
 from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
 pair=make_queryage_localbalanced_pair(seed=SEED);shared=parent.state_digest(pair[0].state_dict());model=(pair[0] if arm=="flat" else pair[1]).to(device);del pair
 return model,torch.optim.AdamW(groups(model),lr=LR,weight_decay=.01),DecoderEMA(model,decay=EMA),shared
def worker_run(*,arm,output,physical_gpu,start_marker):
 """Production-only worker. No new forward probe, fixture switch or reseeding."""
 started=time.monotonic();output=Path(output);frozen,device=_runtime_setup(arm,output,physical_gpu);bindings=frozen["authority"]["bindings"]
 guard=lambda:parent.resource_guard(started,device)
 from tfpd_exploration.src.h1_optimized_v4.paired_train import batches,collate
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
 from .formal_prefix_score import score_selection_cached_ema
 parent_root=Path(bindings["parent_formal"]);old=parent.require_frozen_authority(parent_root);ready_path=parent_root/"barrier"/f"{arm}.ready.json";old_ready=json.loads(ready_path.read_text())
 guard();cache=parent.load_readonly_cache();guard();identities=_all_epoch_identities(cache);guard()
 model,optimizer,ema,shared=_new_model(arm,device)
 if shared!=old_ready["shared_init_sha256"]:raise RuntimeError("same model factory initialization drift")
 checkpoint=parent_root/"checkpoints"/f"{arm}_epoch_012.pt";parent_sha=sha(checkpoint)
 if parent_sha!=frozen["authority"]["parent_audit"]["arms"][arm]["epoch12"]["checkpoint_sha256"]:raise RuntimeError("resume checkpoint SHA drift")
 guard();payload=torch.load(checkpoint,map_location="cpu",weights_only=False);guard()
 strict_parent_restore(payload=payload,model=model,optimizer=optimizer,ema=ema,identities=old_ready["identities"]["12"],arm=arm,shared_init_sha256=shared,bindings=old["bindings"])
 evidence={"raw_state_sha256":parent.state_digest(model.state_dict()),"ema_state_sha256":parent.state_digest(ema.shadow),"optimizer_state_sha256":tree_digest(optimizer.state_dict()),"rng_state_sha256":tree_digest(rng_state()),"global_step":payload["global_step"],"ema_updates":ema.n_updates,"lr":LR,"exact_restore":True}
 require_authority(output);guard()
 parent.atomic_json({"arm":arm,"physical_gpu":physical_gpu,"shared_init_sha256":shared,"parent_shared_init_sha256":old_ready["shared_init_sha256"],"parent_ready_sha256":sha(ready_path),"parent_epoch12_checkpoint_sha256":parent_sha,"identities":identities,"strict_restore_evidence":evidence},output/"barrier"/f"{arm}.ready.json")
 while not Path(start_marker).is_file():guard();time.sleep(.05)
 # Barrier bookkeeping cannot alter any resume stream. Check, without drawing.
 _assert_restored(payload,model,optimizer,ema);guard();del payload
 records=[]
 for epoch in range(13,25):
  losses=[];ordered=batches(cache,epoch)
  if identity_digest(ordered,cache=cache,epoch=epoch)!=identities[str(epoch)]:raise RuntimeError("continuation sampler/dropout/prefix schedule drift")
  for batch_index,(session,starts) in enumerate(ordered):
   guard();row=cache["train"][session];x,y=collate(row,starts,device);bank=H1Bank(*[row["bank"][k].to(device) for k in ("E0","T","unit_mask")])
   step=(epoch-1)*EPOCH_UPDATES+batch_index+1
   for group in optimizer.param_groups:group["lr"]=warmup_lr(epoch=epoch,global_step=step)
   loss,_=_train_update(model,optimizer,ema,x,y,bank,epoch=epoch,batch_index=batch_index,device=device,guard=guard);losses.append(loss)
  payload=checkpoint_payload(model=model,optimizer=optimizer,ema=ema,epoch=epoch,identities=identities[str(epoch)],arm=arm,shared_init_sha256=shared,bindings=bindings,parent_checkpoint_sha256=parent_sha)
  checkpoint=output/"checkpoints"/f"{arm}_epoch_{epoch:03d}.pt"
  if checkpoint.exists():raise FileExistsError(checkpoint)
  guard();parent.atomic_torch_save(payload,checkpoint);guard();loaded=torch.load(checkpoint,map_location="cpu",weights_only=False);guard()
  if not recursive_same(payload,loaded):raise RuntimeError("continuation recursive checkpoint round-trip drift")
  strict_restore(payload=loaded,model=model,optimizer=optimizer,ema=ema,epoch=epoch,identities=identities[str(epoch)],arm=arm,shared_init_sha256=shared,bindings=bindings,parent_epoch12_sha256=parent_sha)
  del payload,loaded
  selection=parent._guarded_score(model,lambda:score_selection_cached_ema(model=model,ema=ema,cache=cache,device=device,guard=guard),guard=guard)
  record={"epoch":epoch,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"selection":selection,"mean_loss":float(np.mean(losses)),"identities":identities[str(epoch)],"parent_checkpoint_sha256":parent_sha};records.append(record);parent.atomic_json(record,output/"workers"/f"{arm}_epoch_{epoch:03d}.json")
  require_authority(output)
 parent.atomic_json({"arm":arm,"identities":identities,"shared_init_sha256":shared,"parent_epoch12_checkpoint_sha256":parent_sha,"epochs":records},output/"workers"/f"{arm}_continue_complete.json")

def finalizer_run(*,arm,output,physical_gpu):
 """Export only the paired frozen selected and fixed endpoint24 EMA states."""
 started=time.monotonic();output=Path(output);frozen,device=_runtime_setup(arm,output,physical_gpu);bindings=frozen["authority"]["bindings"]
 guard=lambda:parent.resource_guard(started,device)
 from .formal_prefix_score import score_selection_cached_ema,score_complete_cached_ema
 from .continue_prefix_launcher import ent,earliest_ema
 parent_root=Path(bindings["parent_formal"]);old=parent.require_frozen_authority(parent_root);old_ready=json.loads((parent_root/"barrier"/f"{arm}.ready.json").read_text());ready=json.loads((output/"barrier"/f"{arm}.ready.json").read_text());parent_sha=ready["parent_epoch12_checkpoint_sha256"]
 freeze_path=output/"continuation_selection_freeze.json";freeze=json.loads(freeze_path.read_text());freeze_sha=sha(freeze_path)
 before=json.loads((parent_root/"workers"/f"{arm}_complete.json").read_text())["epochs"];after=json.loads((output/"workers"/f"{arm}_continue_complete.json").read_text())["epochs"]
 if freeze.get("schema")!="h1_queryage_continue_selection_freeze_v1" or freeze["selected"][arm]!=ent(earliest_ema(before+after)) or freeze["epoch24"][arm]!=ent(after[-1]):raise RuntimeError("paired freeze selection/endpoint drift")
 guard();cache=parent.load_readonly_cache();guard();reports={}
 for label in ("selected","epoch24"):
  row=freeze[label][arm];epoch=int(row["epoch"]);path=Path(row["checkpoint"]);root=parent_root if epoch<=12 else output
  if path!=root/"checkpoints"/f"{arm}_epoch_{epoch:03d}.pt" or sha(path)!=row["checkpoint_sha256"]:raise RuntimeError("frozen checkpoint path/SHA drift")
  model,optimizer,ema,shared=_new_model(arm,device)
  if shared!=ready["shared_init_sha256"]:raise RuntimeError("finalizer initialization identity drift")
  guard();payload=torch.load(path,map_location="cpu",weights_only=False);guard()
  if epoch<=12:
   parent.strict_restore(payload=payload,model=model,optimizer=optimizer,ema=ema,epoch=epoch,identities=old_ready["identities"][str(epoch)],arm=arm,shared_init_sha256=shared,bindings=old["bindings"]);_assert_restored(payload,model,optimizer,ema)
  else:strict_restore(payload=payload,model=model,optimizer=optimizer,ema=ema,epoch=epoch,identities=ready["identities"][str(epoch)],arm=arm,shared_init_sha256=shared,bindings=bindings,parent_epoch12_sha256=parent_sha)
  selection=parent._guarded_score(model,lambda:score_selection_cached_ema(model=model,ema=ema,cache=cache,device=device,guard=guard),guard=guard)
  if abs(selection["r2_concat_float64"]-row["ema_r2_float64"])>1e-5:raise RuntimeError("frozen minival reproduction drift")
  complete=parent._guarded_score(model,lambda:score_complete_cached_ema(model=model,ema=ema,cache=cache,device=device,guard=guard),guard=guard)
  arrays={key.removeprefix("_"):complete.pop(key) for key in ("_prediction","_target","_session_id","_end")}
  export=output/"exports"/f"{arm}_{label}_complete_native_float64.npz";plain=output/"exports"/f"{arm}_{label}_plain_ema.pt"
  if export.exists() or plain.exists():raise FileExistsError("continuation exports must be fresh")
  with tempfile.NamedTemporaryFile(dir=export.parent,suffix=".tmp",delete=False) as handle:temporary=Path(handle.name)
  try:
   with temporary.open("wb") as handle:np.savez_compressed(handle,**arrays)
   os.replace(temporary,export)
  finally:temporary.unlink(missing_ok=True)
  plain_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  for k,v in ema.shadow.items():plain_state[k]=v.detach().cpu().clone()
  if any(not bool(torch.isfinite(v).all()) for v in plain_state.values()):raise RuntimeError("nonfinite EMA export")
  guard();parent.atomic_torch_save(plain_state,plain);guard()
  if not recursive_same(plain_state,torch.load(plain,map_location="cpu",weights_only=True)):raise RuntimeError("EMA export round-trip drift")
  with np.load(export,allow_pickle=False) as check:
   if set(check.files)!=set(arrays) or any(not np.array_equal(check[k],arrays[k]) for k in arrays):raise RuntimeError("native complete archive round-trip drift")
  reports[label]={"epoch":epoch,"checkpoint_sha256":sha(path),"selection_reproduced":selection,"complete":complete,"complete_archive":str(export),"complete_archive_sha256":sha(export),"plain_ema_path":str(plain),"plain_ema_sha256":sha(plain)}
  del model,optimizer,ema,payload
 require_authority(output)
 if sha(freeze_path)!=freeze_sha:raise RuntimeError("freeze drift during finalization")
 parent.atomic_json({"status":"COMPLETE_POST_FREEZE","arm":arm,"reports":reports},output/"workers"/f"{arm}_final.json")
