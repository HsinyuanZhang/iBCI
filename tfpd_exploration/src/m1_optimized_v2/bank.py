"""Persist/load exactly one source-only rSyn3 refit carrier bank."""
from __future__ import annotations

import hashlib, json, os
from datetime import datetime, timezone
from typing import Any
import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify, syn3
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan
from . import plan

def _digest(a: np.ndarray) -> str: return syn3.array_digest(np.ascontiguousarray(a))
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _numeric_env():
    values={k:str(plan.OMP_THREADS) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}
    for k,v in values.items(): os.environ[k]=v
    return values

def _source_paths():
    root=parent_data.repo_root(); paths={}
    for session in plan.SOURCE_SESSIONS:
        path=parent_data.require_source_path(root / parent_plan.SOURCE_RELATIVE[session])
        if parent_data.file_sha256(path) != parent_plan.SOURCE_FILE_SHA256[session]: raise RuntimeError(f"source hash drift: {session}")
        paths[session]=path
    return paths

def seal_once() -> dict[str, Any]:
    """Fit NMF once from 26/27/28 and atomically refuse a different overwrite."""
    if plan.BANK_NPZ.exists() or plan.BANK_RECEIPT.exists(): return load()
    plan.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    env=_numeric_env(); rectify.assert_frozen_law(); paths=_source_paths()
    records={n:fold_data.load_fold_session(paths[n], role="source") for n in plan.SOURCE_SESSIONS}
    emg=np.concatenate([rectify.relu_nonnegative_projection(r.emg) for r in records.values()],axis=0)
    basis=syn3.fit_source_nmf(emg)
    raw={n:np.ascontiguousarray(_encode_session(r,basis),dtype=np.float64) for n,r in records.items()}
    mean,scale=syn3.source_normalizer(list(raw.values()))
    normalized={n:np.ascontiguousarray(syn3.normalize_carriers(x,mean,scale),dtype=np.float64) for n,x in raw.items()}
    payload={"schema":np.asarray([plan.SCHEMA]),"revision":np.asarray([plan.CARRIER_REVISION]),"d0":basis.dictionary,"scale":basis.scale,"activations":basis.activations,"normalizer_mean":mean,"normalizer_scale":scale,"source_sessions":np.asarray(plan.SOURCE_SESSIONS),"nmf_order":np.asarray(basis.order,dtype=np.int64),"nmf_n_iter":np.asarray([basis.extra["n_iter"]],dtype=np.int64),"reconstruction_digest":np.asarray([basis.reconstruction_digest])}
    for n,r in records.items():
        payload[f"raw/{n}"]=raw[n]; payload[f"normalized/{n}"]=normalized[n]
        payload[f"support_emg_trial_ids/{n}"]=np.asarray(r.emg_trial_ids); payload[f"support_rate_trial_ids/{n}"]=np.asarray(r.rate_trial_ids)
        payload[f"support_row_ids/{n}"]=np.arange(raw[n].shape[0],dtype=np.int64)
        payload[f"unit_roster/{n}"]=np.arange(raw[n].shape[0],dtype=np.int64)
        payload[f"channel_names/{n}"]=np.asarray(r.channel_names)
    tmp=plan.BANK_NPZ.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**payload); tmp.replace(plan.BANK_NPZ)
    digests={"npz":_sha(plan.BANK_NPZ),"d0":_digest(payload["d0"]),"scale":_digest(payload["scale"]),"raw":{n:_digest(raw[n]) for n in raw},"normalized":{n:_digest(normalized[n]) for n in normalized}}
    if digests["d0"] == plan.OLD_STAGE0_D0_SHA256: raise RuntimeError("new refit collides with sealed Stage-0 D0")
    receipt={"schema":plan.SCHEMA,"decoder_revision":plan.DECODER_REVISION,"carrier_revision":plan.CARRIER_REVISION,"not_old_stage0":True,"old_stage0_d0_sha256":plan.OLD_STAGE0_D0_SHA256,"old_target_m10_sha256":plan.OLD_TARGET_M10_SHA256,"source_sessions":list(plan.SOURCE_SESSIONS),"outer_session_excluded":plan.OUTER_SESSION,"target_path_resolved":False,"target_file_opened":False,"target_query_values_read":False,"fit_once":True,"normalizer_scope":"sources_26_27_28_only","unit_encoding":"same legal first-10 unit ridge encoding","support_trials":[0,10],"env":env,"seed":plan.SEED,"nmf":{"library":dict(basis.library),"n_iter":int(basis.extra["n_iter"]),"reconstruction_digest":basis.reconstruction_digest},"provenance":{"source_files":{n:{"path":str(p),"sha256":parent_data.file_sha256(p)} for n,p in paths.items()},"sfix_path":str(plan.S_FIX_PATH),"sfix_sha256":plan.S_FIX_SHA256,"sfix_decoder_weights_copied":False},"digests":digests,"created":datetime.now(timezone.utc).isoformat()}
    plan.BANK_RECEIPT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return load()

def load() -> dict[str, Any]:
    if not plan.BANK_NPZ.is_file() or not plan.BANK_RECEIPT.is_file(): raise FileNotFoundError("persisted rSyn3-refit-v1 carrier is absent")
    receipt=json.loads(plan.BANK_RECEIPT.read_text()); blob=np.load(plan.BANK_NPZ,allow_pickle=False)
    if receipt.get("schema") != plan.SCHEMA or receipt.get("carrier_revision") != plan.CARRIER_REVISION: raise RuntimeError("carrier schema/revision drift")
    if receipt.get("target_path_resolved") or receipt.get("target_query_values_read"): raise RuntimeError("carrier receipt reports target leakage")
    if _sha(plan.BANK_NPZ) != receipt["digests"]["npz"]: raise RuntimeError("persisted carrier bytes drift")
    if receipt["digests"]["d0"] == plan.OLD_STAGE0_D0_SHA256: raise RuntimeError("old Stage-0 carrier cannot be used")
    return {"receipt":receipt,"d0":np.asarray(blob["d0"]),"scale":np.asarray(blob["scale"]),"normalizer_mean":np.asarray(blob["normalizer_mean"]),"normalizer_scale":np.asarray(blob["normalizer_scale"]),"normalized":{n:np.asarray(blob[f"normalized/{n}"],dtype=np.float32) for n in plan.SOURCE_SESSIONS},"raw":{n:np.asarray(blob[f"raw/{n}"],dtype=np.float64) for n in plan.SOURCE_SESSIONS}}
