"""Persist source-only raw neural and frozen bank tensors for parity/CPU benches."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
import numpy as np
from . import bank,plan
from .data import build_source_only_datamodule,materialize_source_banks
NPZ=plan.RESULT_ROOT/"m1_optimized_v2_source_runtime_cache.npz"; RECEIPT=plan.RESULT_ROOT/"m1_optimized_v2_source_runtime_cache.receipt.json"
def _sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if NPZ.exists() or RECEIPT.exists():
  if not NPZ.exists() or not RECEIPT.exists(): raise RuntimeError("partial runtime cache")
  x=json.loads(RECEIPT.read_text())
  if _sha(NPZ)!=x["npz_sha256"]: raise RuntimeError("runtime cache byte drift")
  return x
 loaded=bank.load(); dm=build_source_only_datamodule(loaded); banks=materialize_source_banks()
 if getattr(dm,"target_path",None) is not None or dm.val_heldin_dataset is not None: raise RuntimeError("runtime cache touched outer")
 base=dm.train_dataset.base; payload={}; arrays={}
 for name in plan.SOURCE_SESSIONS:
  raw=np.ascontiguousarray(base.neural_data[name],dtype=np.float32); b=banks[name]
  payload[f"raw_neural/{name}"]=raw; payload[f"trial_starts_padded/{name}"]=np.asarray(base.trial_start_indices[name],dtype=np.int64)
  payload[f"bank_e0/{name}"]=b.E0.detach().cpu().numpy(); payload[f"bank_t/{name}"]=b.T.detach().cpu().numpy(); payload[f"bank_unit_mask/{name}"]=b.unit_mask.detach().cpu().numpy()
  arrays[name]={"raw_neural_shape":list(raw.shape),"raw_neural_sha256":hashlib.sha256(raw.tobytes()).hexdigest(),"e0_shape":list(b.E0.shape),"e0_sha256":hashlib.sha256(b.E0.detach().cpu().numpy().tobytes()).hexdigest(),"carrier_sha256":hashlib.sha256(b.T.detach().cpu().numpy().tobytes()).hexdigest()}
 np.savez_compressed(NPZ,**payload)
 r={"schema":"m1_optimized_v2_source_runtime_cache_v1","carrier_npz_sha256":loaded["receipt"]["digests"]["npz"],"npz_sha256":_sha(NPZ),"source_sessions":list(plan.SOURCE_SESSIONS),"arrays":arrays,"outer_path_resolved":False,"outer_query_opened":False,"raw_neural":"FalconDataset raw source-only neural_data; no smoothing/interpolation beyond data-module contract","created":datetime.now(timezone.utc).isoformat()}; RECEIPT.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n"); return r
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
