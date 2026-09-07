"""Post-run-only independent finalizer for the frozen P1 source-minival run.

This module deliberately refuses a running/incomplete run.  It neither trains
nor reads outer/official data; it independently reloads the completed P1
checkpoints, applies the recorded EMA state, and rescans the exact known-source
chron80 development surface before atomically sealing exports.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tfpd_exploration.src.m1_optimized_v2 import plan
from tfpd_exploration.src.m2_same_query_comparator_v1.core import array_sha256
from .family_train_v2 import _batches, _name, _p1_models, _p1_provenance, _source_sets, _trusted_train_batches

RUN_ROOT = plan.RESULT_ROOT / "family_v1" / "p1_queryage16_pair_chron80_v2"
ARMS = ("flat", "route")
EPOCHS = 24


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _atomic_torch(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def _atomic_npz(path: Path, **arrays) -> None:
    if path.exists(): raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=f".{path.name}.",delete=False) as handle: tmp=Path(handle.name)
    try:
        np.savez_compressed(tmp, **arrays)
        # numpy appends .npz for a suffix-less temporary path.
        produced=tmp.with_suffix(tmp.suffix+".npz")
        os.replace(produced,path)
    finally:
        tmp.unlink(missing_ok=True); tmp.with_suffix(tmp.suffix+".npz").unlink(missing_ok=True)

def _atomic_json(path: Path, body: dict) -> None:
    if path.exists(): raise FileExistsError(path)
    encoded=json.dumps(body,indent=2,sort_keys=True)+"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=f".{path.name}.",mode="w",delete=False) as handle:
        tmp=Path(handle.name);handle.write(encoded)
    try: os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def _live_closure() -> dict[str,str]:
    root=plan.REPO_ROOT/"tfpd_exploration/src"
    rel=("m1_family_v1/finalize_p1.py","m1_family_v1/family_train_v2.py","m1_optimized_v2/model.py","m1_optimized_v2/source_dev.py","m1_optimized_v2/data.py","m1_optimized_v2/bank.py","m1_optimized_v2/plan.py","two_mainlines_long_v1/current_query_v2/core.py","two_mainlines_long_v1/decoder/h1_temporal.py","two_mainlines_long_v1/decoder/h1_config.py","two_mainlines_long_v1/decoder/m1_config.py")
    extra=("m1_optimized_v2/calibration.py","m2_same_query_comparator_v1/core.py",
           "m1_emg_rsyn3_fold_local_v1/module.py","two_mainlines_long_v1/decoder/m1_temporal.py")
    result={name:_sha(root/name) for name in rel+extra}
    for name in ("falcon_datamodule.py","m1_version_b_source_loso_datamodule.py","falcon_emg_afc4_features.py"):
        path=plan.REPO_ROOT/"streaming_calibration_exp/src/data"/name
        result[str(path.relative_to(plan.REPO_ROOT))]=_sha(path)
    return result


def _finite_score(row: dict, arm: str) -> float:
    try: score = float(row[arm]["ema"]["equal_session_mean_r2"])
    except (KeyError, TypeError, ValueError) as exc: raise RuntimeError(f"{arm}: missing EMA equal-session score") from exc
    if not math.isfinite(score): raise RuntimeError(f"{arm}: nonfinite EMA score")
    return score


def select_earliest_ema(history: list[dict], arm: str) -> dict:
    """Exact primary rule: highest EMA equal-session score, earliest tie."""
    if not isinstance(history, list) or [row.get("epoch") for row in history] != list(range(1, EPOCHS + 1)):
        raise RuntimeError("requires contiguous epochs 1..24")
    scores = [_finite_score(row, arm) for row in history]
    best = max(scores)
    return next(row[arm] for row, score in zip(history, scores, strict=True) if score == best)


def _load_json(path: Path):
    if not path.is_file(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _verify_checkpoint(record: dict, meta_recipe: dict, *, run_root: Path, arm: str, expected_epoch: int) -> dict:
    if record.get("epoch") != expected_epoch: raise RuntimeError("nested checkpoint epoch drift")
    path = Path(record["checkpoint"])
    exact = run_root / arm / f"epoch_{expected_epoch:03d}.pt"
    if path != exact:
        raise RuntimeError(f"checkpoint path is not exact run-root path: {path}")
    if not path.is_file() or _sha(path) != record.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint hash/path mismatch: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if (blob.get("schema") != "m1_family_v1_chron80_checkpoint_v2" or blob.get("epoch") != expected_epoch
            or blob.get("recipe") != meta_recipe or blob.get("outer_query_opened") is not False):
        raise RuntimeError(f"checkpoint schema/recipe/epoch drift: {path}")
    model, ema = blob.get("model"), blob.get("ema")
    if not isinstance(model, dict) or not isinstance(ema, dict) or set(model) != set(ema) or not model:
        raise RuntimeError(f"checkpoint model/EMA key mismatch: {path}")
    for key in model:
        a, b = model[key], ema[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.shape != b.shape:
            raise RuntimeError(f"checkpoint tensor shape drift: {path}:{key}")
        if a.is_floating_point() and (not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
            raise RuntimeError(f"checkpoint nonfinite tensor: {path}:{key}")
    if _sha(path) != record.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint changed during load: {path}")
    return blob


def load_completed(run_root: Path = RUN_ROOT) -> tuple[dict, dict, list[dict], dict]:
    meta = _load_json(run_root / "run_meta.json")
    report = _load_json(run_root / "report.json")
    history = _load_json(run_root / "epoch_metrics.json")
    if meta.get("status") != "COMPLETE" or meta.get("completed_epochs") != EPOCHS or report.get("status") != "SOURCE_MINIVAL_ONLY":
        raise RuntimeError("requires formal P1 run_meta COMPLETE and source-minival report")
    if meta.get("outer_query_opened") is not False or report.get("outer_query_opened") is not False:
        raise RuntimeError("outer-query authority violation")
    if meta.get("recipe",{}).get("epochs") != EPOCHS or report.get("recipe") != meta.get("recipe") or report.get("split") != meta.get("split") or report.get("provenance") != meta.get("provenance"):
        raise RuntimeError("report/meta recipe/provenance/split drift")
    if report.get("history") != history or len(history) != EPOCHS:
        raise RuntimeError("report/history completeness drift")
    selected = {arm: select_earliest_ema(history, arm) for arm in ARMS}
    endpoint = {arm: history[-1][arm] for arm in ARMS}
    blobs = {}
    for arm in ARMS:
        if report.get("selected_primary_ema",{}).get(arm) != selected[arm] or report.get("endpoint24",{}).get(arm) != endpoint[arm]:
            raise RuntimeError("report selection/endpoint drift")
        blobs[(arm, "selected")] = _verify_checkpoint(selected[arm], meta["recipe"], run_root=run_root, arm=arm, expected_epoch=selected[arm]["epoch"])
        blobs[(arm, "endpoint24")] = _verify_checkpoint(endpoint[arm], meta["recipe"], run_root=run_root, arm=arm, expected_epoch=EPOCHS)
    return meta, report, history, {"selected": selected, "endpoint24": endpoint, "blobs": blobs}


def _verify_source_closure(meta: dict):
    loaded, train, dev, split = _source_sets()
    provenance = _p1_provenance(loaded, split, _trusted_train_batches(train), _batches(dev))
    if meta.get("split") != split or meta.get("provenance") != provenance or len(dev) != 31252:
        raise RuntimeError("frozen source split/sampler/bank/code closure drift")
    return dev

def _historical_same_surface(current: dict) -> dict:
    """Bind comparators; disclose overlap and never label them a new holdout."""
    paths={
        "original_teacher_overlap": RUN_ROOT.parent/"original_teacher_chron80_overlap.json",
        "sfix_epoch011_overlap": RUN_ROOT.parent/"sfix_epoch011_source_dev_replay.json",
        "formal12_current_query": plan.RESULT_ROOT/"formal_formal12_chron80_v2_current_query_postscore.json",
    }
    refs={}
    for name,path in paths.items():
        if not path.is_file(): raise FileNotFoundError(path)
        body=_load_json(path)
        if body.get("outer_query_opened") is not False: raise RuntimeError(f"{name}: outer authority")
        if name == "formal12_current_query":
            row=body.get("selected",{}); pred_path=Path(row.get("prediction_export",""))
            if body.get("carrier_npz_sha256") != "27016199c39d630b4a0455ddeae36aa90e669bc4e913e765d2b00c03637298ab" or not pred_path.is_file(): raise RuntimeError("formal T authority/schema drift")
            if _sha(pred_path) != row.get("prediction_export_sha256"):
                raise RuntimeError("formal T archive hash drift")
            z=np.load(pred_path); p,y,s=z["prediction"],z["target"],z["session"]
            if p.shape != (31252,16) or y.shape != p.shape or s.shape != (31252,): raise RuntimeError("formal T 31252 surface drift")
            def r(a,b):
                a,b=np.asarray(a,np.float64),np.asarray(b,np.float64); return float(1-np.square(a-b).sum()/np.square(b-b.mean(0,keepdims=True)).sum())
            per={n:r(p[s==n],y[s==n]) for n in plan.SOURCE_SESSIONS}; equal=float(np.mean(list(per.values()))); pooled=r(p,y)
            if abs(equal-float(row.get("equal_session_mean_r2")))>1e-5: raise RuntimeError("formal T selected score drift")
            refs[name]={"path":str(path),"sha256":_sha(path),"prediction_sha256":_sha(pred_path),"n":31252,"equal_session_mean_r2":equal,"pooled_r2":pooled,
                        "query_array_sha256":{"target":array_sha256(np.asarray(y,dtype=np.float32)),"session":array_sha256(np.asarray(s)),"start":array_sha256(np.asarray(z["window_start"],dtype=np.int64))}}
        else:
            metrics=body.get("metrics",{}); split=body.get("split",{})
            n=metrics.get("n",split.get("dev_windows")); pooled=metrics.get("pooled",{}).get("r2",{}).get("model")
            if body.get("bank_sha256",body.get("carrier",{}).get("source_only_refit_npz_sha256")) != "27016199c39d630b4a0455ddeae36aa90e669bc4e913e765d2b00c03637298ab" or n != 31252: raise RuntimeError(f"{name}: same-surface authority drift")
            equal=metrics.get("equal_session_mean_r2")
            if not all(math.isfinite(float(v)) for v in (equal,pooled)): raise RuntimeError(f"{name}: null/nonfinite comparator")
            refs[name]={"path":str(path),"sha256":_sha(path),"n":n,"equal_session_mean_r2":float(equal),"pooled_r2":float(pooled)}
    return {"caveat":"same frozen 31,252 source-minival surface; source-training overlap means comparator-only, not new holdout/generalization","references":refs,"selected_deltas":{arm:{name:{"equal_session_delta":(current[arm]["metrics"]["equal_session_mean_r2"]-ref["equal_session_mean_r2"] if ref["equal_session_mean_r2"] is not None else None),"pooled_delta":(current[arm]["metrics"]["pooled_r2"]-ref["pooled_r2"] if ref["pooled_r2"] is not None else None)} for name,ref in refs.items()} for arm in ARMS}}


@torch.no_grad()
def _score_export(model, dev, *, device: str):
    from tfpd_exploration.src.m1_optimized_v2.data import materialize_source_banks
    banks = {n:type(v)(E0=v.E0.to(device),T=v.T.to(device),unit_mask=v.unit_mask.to(device)) for n,v in materialize_source_banks().items()}
    pred=[]; target=[]; session=[]; starts=[]
    for ids in _batches(dev):
        x,y,_c,s,_carrier = next(iter(DataLoader(dev, batch_sampler=[ids])))
        name=_name(s[0])
        if any(_name(v) != name for v in s): raise RuntimeError("non-pure source dev batch")
        out=model.forward_last(x.to(device).float(),banks[name])
        if out.shape != (len(ids),16) or y.shape[-1] != 16: raise RuntimeError("native16 scoring shape drift")
        pred.append(out.cpu().numpy().astype(np.float32)); target.append(y[:,-1,:].numpy().astype(np.float32))
        session.extend([name]*len(ids)); starts.extend(int(dev.base.window_indices[i][1]) for i in ids)
    p,y=np.concatenate(pred),np.concatenate(target)
    if p.shape != (31252,16) or not np.isfinite(p).all() or not np.isfinite(y).all(): raise RuntimeError("source prediction export drift")
    def r2(a,b):
        # Exact source-dev _r2: scalar SSE/SST in float64; no variance-clipping
        # is present in that frozen scorer, and all checked target dimensions
        # have positive variance.
        a,b=np.asarray(a,np.float64),np.asarray(b,np.float64); denom=np.square(b-b.mean(0,keepdims=True)).sum()
        if not np.isfinite(denom) or denom <= 0: raise RuntimeError("invalid native source R2 variance")
        return float(1-np.square(a-b).sum()/denom)
    per={n:r2(p[np.asarray(session)==n],y[np.asarray(session)==n]) for n in plan.SOURCE_SESSIONS}
    return p,y,np.asarray(session),np.asarray(starts,dtype=np.int64),{"equal_session_mean_r2":float(np.mean(list(per.values()))),"pooled_r2":r2(p,y),"per_session_r2":per}


def finalize(*, run_root: Path = RUN_ROOT, device: str = "cpu") -> dict:
    meta, report, history, records = load_completed(run_root)
    dev = _verify_source_closure(meta)
    dummy={arm:{"metrics":{"equal_session_mean_r2":0.,"pooled_r2":0.}} for arm in ARMS}
    baseline_pre=_historical_same_surface(dummy)["references"]
    frozen={"meta_sha256":_sha(run_root/"run_meta.json"),"report_sha256":_sha(run_root/"report.json"),"history_sha256":_sha(run_root/"epoch_metrics.json"),"selected":records["selected"],"endpoint24":records["endpoint24"],"checkpoint_sha256":{f"{arm}_{label}":records[label][arm]["checkpoint_sha256"] for arm in ARMS for label in ("selected","endpoint24")},"source_closure":meta["provenance"],"live_code_closure":_live_closure(),"historical_baseline_pre":baseline_pre}
    _atomic_json(run_root/"selection_freeze.json",{"schema":"m1_family_v1_p1_selection_freeze_v1","status":"FROZEN_PRE_INFERENCE","freeze_pre_inference":frozen,"outer_query_opened":False})
    models = dict(zip(ARMS, _p1_models(torch.device(device)), strict=True))
    output = run_root / "finalized_p1"
    if output.exists(): raise FileExistsError(output)
    output.mkdir()
    items={}
    for arm in ARMS:
        for label in ("selected", "endpoint24"):
            record=records[label][arm]; blob=records["blobs"][(arm,label)]; model=models[arm]
            model.load_state_dict(blob["ema"],strict=True); model.eval()
            # Plain EMA export then strict independently constructed reload.
            state_path=output/f"{arm}_{label}_ema_state.pt"; _atomic_torch(state_path,blob["ema"])
            fresh=_p1_models(torch.device(device))[0 if arm=="flat" else 1]; fresh.load_state_dict(torch.load(state_path,map_location=device,weights_only=False),strict=True); fresh.eval()
            p,y,s,start,metrics=_score_export(fresh,dev,device=device)
            if abs(metrics["equal_session_mean_r2"]-float(record["ema"]["equal_session_mean_r2"])) > 1e-5: raise RuntimeError("recorded EMA score not reproduced")
            if abs(metrics["pooled_r2"]-float(record["ema"]["pooled"]["r2"]["model"])) > 1e-5:
                raise RuntimeError("recorded EMA pooled score not reproduced")
            for name,value in metrics["per_session_r2"].items():
                if abs(value-float(record["ema"]["per_session"][name]["r2"]["model"])) > 1e-5:
                    raise RuntimeError("recorded EMA per-session score not reproduced")
            npz=output/f"{arm}_{label}_native_source_dev.npz"; _atomic_npz(npz,prediction=p,target=y,session=s,start=start)
            arrays={"prediction":array_sha256(p),"target":array_sha256(y),"session":array_sha256(s),"start":array_sha256(start)}
            if {key:arrays[key] for key in ("target","session","start")} != baseline_pre["formal12_current_query"]["query_array_sha256"]:
                raise RuntimeError("M1 current/reference query arrays differ")
            items[f"{arm}_{label}"]={"record":record,"state_sha256":_sha(state_path),"prediction_sha256":_sha(npz),"array_sha256":arrays,"metrics":metrics}
    _verify_source_closure(meta)
    if _live_closure() != frozen["live_code_closure"]: raise RuntimeError("live finalizer/trainer/data/model code closure changed during finalization")
    if {"meta_sha256":_sha(run_root/"run_meta.json"),"report_sha256":_sha(run_root/"report.json"),"history_sha256":_sha(run_root/"epoch_metrics.json")} != {k:frozen[k] for k in ("meta_sha256","report_sha256","history_sha256")}:
        raise RuntimeError("run receipt changed during finalization")
    for arm in ARMS:
        for label in ("selected","endpoint24"):
            if _sha(Path(records[label][arm]["checkpoint"])) != frozen["checkpoint_sha256"][f"{arm}_{label}"]: raise RuntimeError("checkpoint changed during finalization")
    selected_items={arm:items[f"{arm}_selected"] for arm in ARMS}
    historical_post=_historical_same_surface(selected_items)
    if historical_post["references"] != baseline_pre:
        raise RuntimeError("M1 historical reference changed during finalization")
    body={"schema":"m1_family_v1_p1_finalizer_v1","status":"SOURCE_MINIVAL_ONLY","selection":"EMA equal-session highest; earliest exact tie","freeze_pre_inference":frozen,"exports":items,"historical_same_surface_only":historical_post,"outer_query_opened":False}
    _atomic_json(output/"receipt.json",body)
    return body


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,default=RUN_ROOT);p.add_argument("--device",default="cpu");p.add_argument("--finalize",action="store_true");a=p.parse_args()
    if not a.finalize: p.error("explicit --finalize required; no work is performed otherwise")
    print(json.dumps(finalize(run_root=a.run_root,device=a.device),indent=2,sort_keys=True))
