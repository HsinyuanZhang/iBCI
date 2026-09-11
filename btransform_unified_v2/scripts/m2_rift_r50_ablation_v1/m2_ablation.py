#!/usr/bin/env python3
"""M2 RIFT-R50 fixed information-ablation trainer.

This is an isolated adapter over the frozen concat training loop.  It never
changes the global dual-track cache: ``--bank-cache-root`` is mandatory and
all M2 cache lookups are redirected in-process only.  ACTIVITY_ONLY and NONE
therefore retain the exact R50/D4 decoder recipe while substituting sealed
static input banks produced by build_banks.py.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
import numpy as np
import torch
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
for p in (ROOT,ROOT/'src',WS/'btransform_unified_v1'/'src',WS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from scripts.rift_v1 import m2_concat_train as base
from btransform_unified_v1 import adapters
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data, plan as old_plan

ARMS = ("ACTIVITY_ONLY", "NONE")
EPOCHS = 24
UPDATES = 3165

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p:Path,x:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def cell(arm:str)->str:return f'M2-RIFT-R50-D4-CONCAT-{arm}-V1'
def _verify_root(root: Path, arm: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing ablation bank manifest {manifest_path}")
    doc = json.loads(manifest_path.read_text())
    if doc.get("schema") != "m2_rift_r50_ablation_bank_v1" or doc.get("status") != "COMPLETED" or doc.get("arm") != arm:
        raise RuntimeError("bank manifest arm/schema/status mismatch")
    if doc.get("source_manifest_contract") != "24x3165=75960 seed42 R50D4 local13/12/12/12 batch32 AdamW3e-4 warmup1 EMA.9995 dropout.1":
        raise RuntimeError("main M2 recipe contract drift")
    if doc.get("official_test_used") is not False or doc.get("evalai_opened") is not False:
        raise RuntimeError("bank artifact must not use official test or EvalAI")
    rows = doc.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise RuntimeError("bank manifest has no source-bank bindings")
    for key, expected in rows.items():
        directory = root / key
        required = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "calib_activity.npy", "T.npy", "e0_u.pt")
        if not all((directory / name).is_file() for name in required):
            raise RuntimeError(f"{key}: required sealed bank asset missing")
        for filename, receipt_key in (("X_store.npy", "X_store_sha256"), ("target_store.npy", "target_store_sha256"), ("eligible_starts.npy", "eligible_starts_sha256"), ("calib_activity.npy", "calib_activity_sha256")):
            if sha(directory / filename) != expected.get(receipt_key):
                raise RuntimeError(f"{key}: {filename} binding drift")
        carrier = np.ascontiguousarray(np.load(directory / "T.npy"), np.float32)
        state = torch.load(directory / "e0_u.pt", map_location="cpu", weights_only=False)
        e0 = np.ascontiguousarray(state.get("E0").detach().cpu().numpy(), np.float32)
        u = np.ascontiguousarray(state.get("frozen_u").detach().cpu().numpy(), np.float32)
        if carrier.shape != (96, 4) or e0.shape != (96, 50) or u.ndim != 3 or u.shape[1:] != (96, 64):
            raise RuntimeError(f"{key}: static E0/T geometry drift")
        if hashlib.sha256(carrier.tobytes()).hexdigest() != expected.get("T_sha256"):
            raise RuntimeError(f"{key}: T array binding drift")
        if hashlib.sha256(e0.tobytes()).hexdigest() != expected.get("E0_sha256"):
            raise RuntimeError(f"{key}: E0 array binding drift")
        if hashlib.sha256(u.tobytes()).hexdigest() != expected.get("frozen_u_sha256") or list(u.shape) != expected.get("frozen_u_shape"):
            raise RuntimeError(f"{key}: frozen_u binding drift")
        if arm == "NONE" and not np.array_equal(u, np.zeros_like(u)):
            raise RuntimeError(f"{key}: NONE frozen_u is not literal zero")
        if not np.array_equal(carrier, np.zeros((96, 4), np.float32)):
            raise RuntimeError(f"{key}: direct T is not literal zero")
        if arm == "NONE" and not np.array_equal(e0, np.zeros((96, 50), np.float32)):
            raise RuntimeError(f"{key}: NONE E0 is not literal zero")
        if arm == "ACTIVITY_ONLY" and expected.get("encoder_called") is not True:
            raise RuntimeError(f"{key}: ACTIVITY_ONLY lacks frozen encoder route evidence")
    return doc

def patch(arm:str,root:Path)->None:
 manifest=_verify_root(root,arm)
 # Explicit, process-local routing: never mutate adapters.py or old_data.py.
 old_data.cache_root=lambda:root
 adapters._M2_CACHE_ROOT=root
 base.CELL=cell(arm)
 original_source=base._source_hashes
 def sources():
  out=original_source(); out[str(Path(__file__).resolve())]=sha(Path(__file__).resolve()); out[str(root/'manifest.json')]=sha(root/'manifest.json'); return out
 base._source_hashes=sources
 old_meta=base._run_meta
 def meta(*a,**kw):
  x=old_meta(*a,**kw); x.update({'schema':'m2_rift_r50_ablation_train_v1','cell':cell(arm),'arm':arm,'bank_cache_root':str(root),'bank_manifest_sha256':sha(root/'manifest.json'),'input_contract':manifest['input_contract'],'baseline':{'submission_id':582189,'family':'rift_concat_e9','official_ho':0.34654225938843214,'local_ext6_is_not_official':True}}); return x
 base._run_meta=meta
 # concat's checkpoint schema is intentionally retained: cell + frozen-cache
 # hashes distinguish the arm and source bytes.  The static banks contain no
 # trainable/live encoder.
def run(args:argparse.Namespace)->dict[str,Any]:
 if args.arm not in ARMS:raise ValueError('unknown arm')
 patch(args.arm,args.bank_cache_root.resolve())
 ns=argparse.Namespace(dest=args.dest,stage='train',device=args.device,epochs=args.epochs,resume=args.resume,max_updates_smoke=args.smoke_steps,cpu_threads=args.cpu_threads)
 out=base.run_train(ns)
 # Preserve formal receipt and add arm-specific receipt without changing base checkpoint semantics.
 d=args.dest.resolve()
 if out['status']=='TRAIN_COMPLETED':
  r=json.loads((d/'train_receipt.json').read_text());r.update({'schema':'m2_rift_r50_ablation_train_receipt_v1','arm':args.arm,'baseline_submission_id':582189,'required_selection_surface':'EXT6_ALL24_EMA_EARLIEST_MAX'});atom(d/'train_receipt.json',r)
 if out['status']=='SMOKE_COMPLETED':
  r=json.loads((d/'smoke_receipt.json').read_text());r.update({'schema':'m2_rift_r50_ablation_smoke_receipt_v1','arm':args.arm,'bank_cache_root':str(args.bank_cache_root.resolve()),'route':'static sealed E0/T TaskBank into unchanged RiftConcatDecoder; no live/trainable encoder','init_grad_route_executed':True});atom(d/'smoke_receipt.json',r)
 return out
def main()->int:
 p=argparse.ArgumentParser(description='train M2 fixed ACTIVITY_ONLY/NONE RIFT ablation')
 p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--bank-cache-root',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--epochs',type=int,default=24);p.add_argument('--smoke-steps',type=int)
 a=p.parse_args()
 a.resume=None  # fresh-only ablation runner; no formal resume control.
 if a.epochs<1 or a.epochs>24 or a.smoke_steps is not None and a.smoke_steps<1:p.error('invalid epoch/smoke count')
 if a.smoke_steps is None and a.epochs!=24:p.error('formal run requires exactly 24 epochs')
 print(json.dumps(run(a),indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
