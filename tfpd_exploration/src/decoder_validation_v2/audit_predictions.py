"""Independent score audit for future H1/M1 native-unit prediction exports.

The module deliberately refuses to invent model results: it only reports R2
after an on-disk export and its source authority pass every structural check.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000")
H1_AUTHORITY = ROOT / "h1/source_cache_authority.json"
M1_SESSIONS = ("ses-20120926", "ses-20120927", "ses-20120928")
M1_DATA = Path("/home/xinyuan/Work_host/SPINT/SPINT-main/data/000941/sub-MonkeyL-held-in-calib")

def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    p,y=np.asarray(pred,np.float64),np.asarray(target,np.float64)
    denom=np.square(y-y.mean(axis=0,keepdims=True)).sum()
    if not np.isfinite(denom) or denom <= 0: raise ValueError("degenerate float64 target variance")
    return float(1.0-np.square(p-y).sum()/denom)

def _metrics(p: np.ndarray,y: np.ndarray, train_mean: np.ndarray) -> dict[str,float]:
    return {"model":_r2(p,y),"zero":_r2(np.zeros_like(y),y),"train_only_mean":_r2(np.broadcast_to(train_mean,y.shape),y)}

def _open(path: Path, required: set[str]) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        if set(z.files) != required: raise ValueError(f"NPZ keys {set(z.files)} != required {required}")
        a={k:np.asarray(z[k]) for k in required}
    return a

def _score_by_session(p,y,sessions,mean):
    names=sorted(np.unique(sessions).tolist()); per={}
    for name in names:
        take=sessions==name; per[str(name)]=_metrics(p[take],y[take],mean)
    equal={key:float(np.mean([v[key] for v in per.values()])) for key in ("model","zero","train_only_mean")}
    return per,equal

def audit_h1(npz_path: Path, *, surface: str, authority_path: Path=H1_AUTHORITY) -> dict[str,Any]:
    required={"prediction_native_velocity","target_native_velocity","session_id","bin_timestep"}
    a=_open(npz_path,required); p,y,s,t=a["prediction_native_velocity"],a["target_native_velocity"],a["session_id"],a["bin_timestep"]
    if p.shape != y.shape or p.ndim != 2: raise ValueError("H1 prediction/target must have identical [N,D] shape")
    if p.shape[1] != 7: raise ValueError(f"H1 expected native velocity D=7, got {p.shape[1]}")
    if len(s)!=len(p) or len(t)!=len(p): raise ValueError("H1 row identifiers/count mismatch")
    if not np.isfinite(p).all() or not np.isfinite(y).all(): raise ValueError("H1 export is non-finite")
    expected={"selection":2908,"complete":20325}[surface]
    if len(p)!=expected: raise ValueError(f"H1 {surface} count {len(p)} != {expected}")
    pairs=np.char.add(np.char.add(s.astype("U"),":"),t.astype(str))
    if len(np.unique(pairs))!=len(pairs) or len(np.unique(s))!=13: raise ValueError("H1 IDs duplicate or session roster drift")
    # Compare every target directly to the sealed source-cache native velocity.
    import torch
    cache=torch.load(ROOT/"h1/source_cache.pt",map_location="cpu",weights_only=False)
    from tfpd_exploration.src.h1_optimized_v2.cache import validate_authority
    recorded=json.loads(authority_path.read_text())
    validate_authority(cache,recorded)
    expected_y=[]; expected_pairs=[]
    for name,row in cache["minival"].items():
        ends=row["query_starts"]+699 if surface=="selection" else np.flatnonzero(row["eval_mask"])
        expected_y.append(np.asarray(row["velocity"])[ends]); expected_pairs += [f"{name}:{int(i)}" for i in ends]
    if pairs.tolist()!=expected_pairs or not np.array_equal(y,np.concatenate(expected_y)):
        raise ValueError("H1 export target/IDs do not exactly match source_cache raw velocity")
    mean=np.concatenate([np.asarray(r["velocity"]) for r in cache["train"].values()]).mean(0,keepdims=True)
    per,equal=_score_by_session(p,y,s,mean)
    return {"track":"h1","surface":surface,"n":len(p),"pooled":_metrics(p,y,mean),"per_session":per,"equal_session_r2":equal,"authority_schema":recorded["schema"],"target_sha256":_sha(y)}

def _m1_paths() -> dict[str,Path]:
    out={n:M1_DATA/f"sub-MonkeyL-held-in-calib_{n}_behavior+ecephys.nwb" for n in M1_SESSIONS}
    if any(not p.is_file() for p in out.values()): raise FileNotFoundError("M1 source-only 26/27/28 paths required")
    return out

def _m1_source_truth() -> tuple[dict[tuple[str,int],np.ndarray], set[tuple[str,int]], str, np.ndarray, str]:
    """Independent raw-NWB reconstruction of M1's fixed padded-window split."""
    paths=_m1_paths()
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    truth,train_ids,dev_ids={},[],[]
    for session,path in paths.items():
        # `load_nwb` is the official preprocessed_emg reader used before the
        # owner datamodule's padding/window construction; no owner split/helper
        # is imported here and outer-24 has no possible path.
        _neural, emg, trial_change, eval_mask=load_nwb(str(path),FalconTask.m1)
        emg=np.asarray(emg,dtype=np.float32)
        if emg.ndim!=2 or emg.shape[1]!=16: raise ValueError(f"{session}: expected 16 preprocessed_emg series")
        padded_emg=np.pad(emg,((99,0),(0,0)),constant_values=0.0)
        padded_trial=np.pad(np.asarray(trial_change,dtype=bool),(99,0),constant_values=False)
        padded_eval=np.pad(np.asarray(eval_mask,dtype=bool),(99,0),constant_values=False)
        trial_starts=np.flatnonzero(padded_trial)
        cut_trial=int(np.floor(.8*len(trial_starts)))
        if not 10<cut_trial<len(trial_starts): raise ValueError(f"{session}: invalid M10 chronological cut")
        after_m10,cut=int(trial_starts[10]),int(trial_starts[cut_trial])
        for start in range(len(padded_emg)-100+1):
            endpoint=start+99
            if not padded_eval[endpoint]: continue
            key=(session,start); truth[key]=padded_emg[endpoint]
            if start>=after_m10 and endpoint<cut: train_ids.append(key)
            elif start>=cut: dev_ids.append(key)
    if len(dev_ids)!=31252: raise ValueError(f"frozen M1 dev count {len(dev_ids)} != 31252")
    digest=hashlib.sha256()
    for name,start in dev_ids: digest.update(name.encode()); digest.update(np.asarray([start],dtype=np.int64).tobytes())
    train_digest=hashlib.sha256()
    for name,start in train_ids: train_digest.update(name.encode()); train_digest.update(np.asarray([start],dtype=np.int64).tobytes())
    return truth,set(dev_ids),digest.hexdigest(),np.stack([truth[k] for k in train_ids]).mean(0,keepdims=True),train_digest.hexdigest()

def _file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def audit_m1(npz_path: Path, *, checkpoint: Path|None=None) -> dict[str,Any]:
    required={"prediction","target","session","window_start","window_end_exclusive","bin_timestep","train_target_mean","split_train_sha256","split_dev_sha256","checkpoint_sha256","carrier_npz_sha256"}
    a=_open(npz_path,required); p,y,s=a["prediction"],a["target"],a["session"]
    start,end,bin_t=a["window_start"].astype(np.int64),a["window_end_exclusive"].astype(np.int64),a["bin_timestep"].astype(np.int64)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=16: raise ValueError("M1 prediction/target must be [N,16]")
    if any(len(v)!=len(p) for v in (s,start,end,bin_t)): raise ValueError("M1 row identifiers/count mismatch")
    if not np.isfinite(p).all() or not np.isfinite(y).all(): raise ValueError("M1 export non-finite")
    if not np.array_equal(end,start+100) or not np.array_equal(bin_t,start+99): raise ValueError("M1 padded window endpoint contract violated")
    if set(np.unique(s).tolist())!=set(M1_SESSIONS): raise ValueError("M1 export includes non-source session")
    ids=np.char.add(np.char.add(s.astype("U"),":"),start.astype(str))
    if len(np.unique(ids))!=len(ids): raise ValueError("M1 duplicate session/window identifiers")
    if len(p)!=31252: raise ValueError(f"M1 expected frozen 31252 dev windows, got {len(p)}")
    truth,expected,expected_hash,expected_train_mean,expected_train_hash=_m1_source_truth()
    actual={(str(name),int(ws)) for name,ws in zip(s,start,strict=True)}
    if actual != expected: raise ValueError("M1 exports do not match frozen chronological source-dev endpoints")
    native=np.stack([truth[(str(name),int(ws))] for name,ws in zip(s,start,strict=True)])
    if not np.array_equal(y,native): raise ValueError("M1 exported target differs from official source preprocessed_emg target")
    if str(a["split_dev_sha256"].reshape(-1)[0]) != expected_hash: raise ValueError("M1 split-dev hash drift")
    if str(a["split_train_sha256"].reshape(-1)[0]) != expected_train_hash: raise ValueError("M1 split-train hash drift")
    checkpoint_sha=str(a["checkpoint_sha256"].reshape(-1)[0]); carrier_sha=str(a["carrier_npz_sha256"].reshape(-1)[0])
    if len(checkpoint_sha)!=64 or any(c not in "0123456789abcdef" for c in checkpoint_sha): raise ValueError("M1 checkpoint SHA256 provenance malformed")
    if checkpoint is not None:
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        if _file_sha(checkpoint) != checkpoint_sha: raise ValueError("M1 supplied checkpoint bytes do not match export checkpoint SHA256")
    receipt=json.loads((ROOT/"m1/rSyn3-refit-v1.source-only.receipt.json").read_text())
    if carrier_sha != str(receipt["digests"]["npz"]): raise ValueError("M1 carrier SHA256 differs from frozen source-only carrier receipt")
    mean=np.asarray(a["train_target_mean"],np.float64).reshape(1,-1)
    if not np.array_equal(mean,expected_train_mean): raise ValueError("M1 train-only mean is not the fixed M10/cut-purged source training mean")
    per,equal=_score_by_session(p,y,s,mean)
    return {"track":"m1","n":len(p),"pooled":_metrics(p,y,mean),"per_session":per,"equal_session_r2":equal,"window_ids_sha256":_sha(np.stack([start,end,bin_t],1)),"target_sha256":_sha(y),"source_paths":{k:str(v) for k,v in _m1_paths().items()},"frozen_dev_window_ids_sha256":expected_hash,"frozen_train_window_ids_sha256":expected_train_hash,"checkpoint_sha256":checkpoint_sha,"carrier_npz_sha256":carrier_sha}

def main():
    q=argparse.ArgumentParser(); q.add_argument("track",choices=("h1","m1")); q.add_argument("npz",type=Path); q.add_argument("--surface",choices=("selection","complete")); q.add_argument("--checkpoint",type=Path); a=q.parse_args()
    if a.track=="h1" and not a.surface: q.error("--surface is required for h1")
    if a.track=="h1" and a.checkpoint: q.error("--checkpoint applies to m1 exports only")
    print(json.dumps(audit_h1(a.npz,surface=a.surface) if a.track=="h1" else audit_m1(a.npz,checkpoint=a.checkpoint),indent=2,sort_keys=True))
if __name__=="__main__": main()
