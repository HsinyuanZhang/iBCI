"""Label-separated activity calibration data planes for learned identities.

The query plane contains ``Y`` only for source supervision / local metrics.
Support builders never take a target array as an argument.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG, ROOT, WS = HERE.parent, HERE.parent.parent, HERE.parent.parent.parent
M2_CACHE = WS / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
M2_EXT6 = ROOT / "results/rift_v1/m2_joint_ext6_raw_m33_v1"
M2_QUERY = WS / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query"


def resolve_m2_ext6_root() -> Path:
    """Return the canonical EXT6 root, following RELOCATED pointers if needed.

    A moved subtree is represented by a ``<name>.RELOCATED`` pointer file next
    to the missing directory.  Walk ancestors and remap the canonical path
    under the relocation target; the redirect is a runtime module attribute
    only, no file is modified.  Shared by the ACT and B3S-full scorers so both
    locate the evaluation data identically.
    """
    global M2_EXT6
    canonical = Path(M2_EXT6)
    if canonical.is_dir():
        return canonical
    workspace = WS.resolve()
    for ancestor in canonical.parents:
        pointer = ancestor.with_name(ancestor.name + ".RELOCATED")
        if pointer.is_file():
            target = Path(pointer.read_text().strip().split("->")[-1].strip())
            relocated = (target / canonical.relative_to(ancestor)).resolve()
            if relocated.is_dir():
                M2_EXT6 = relocated
                return relocated
            raise FileNotFoundError(f"relocation target missing: {relocated}")
        if ancestor.resolve() == workspace:
            break
    raise FileNotFoundError(canonical)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def _ah(a: Any) -> str:
    x=np.ascontiguousarray(np.asarray(a)); h=hashlib.sha256(); h.update(x.dtype.str.encode()); h.update(str(x.shape).encode()); h.update(x.tobytes()); return h.hexdigest()

def _hash_item(item: dict[str, Any]) -> dict[str, Any]:
    item["hashes"]={**dict(item.get("hashes",{})), **{k:_ah(item[k]) for k in ("X","Y","starts","activity")}}
    return item


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def windows(item: Mapping[str, Any], indices: np.ndarray, context: int,
            device: str | torch.device, behavior_scale: float):
    """Return query inputs, scaled source targets, and an explicit pad mask."""
    ix = np.asarray(indices, np.int64).reshape(-1)
    starts = np.asarray(item["starts"], np.int64)[ix]
    raw = np.asarray(item["X"], np.float32)
    if raw.ndim != 2 or np.any(starts < 0) or np.any(starts + context > len(raw)):
        raise ValueError("invalid padded query window request")
    x = np.stack([raw[int(s): int(s) + context] for s in starts]).astype(np.float32, copy=False)
    y = np.asarray(item["Y"], np.float32)[ix] * np.float32(behavior_scale)
    valid = starts[:, None] + np.arange(context)[None, :] >= int(item["pad"])
    return (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device),
            torch.as_tensor(valid, device=device))


def post_support_indices(starts: np.ndarray, boundary_padded: int, *, context: int | None = None) -> np.ndarray:
    """Rows whose complete padded context begins after a calibration prefix.

    ``starts`` are padded-window starts, so testing the start (rather than only
    the decoded endpoint) rules out a history window reaching into support.
    ``context`` is accepted to make the caller document its window contract.
    """
    value = np.asarray(starts, np.int64).reshape(-1)
    if context is not None and int(context) < 1: raise ValueError("context must be positive")
    return np.flatnonzero(value >= int(boundary_padded)).astype(np.int64, copy=False)


def _m2_activity(path: Path) -> np.ndarray:
    # Only this file is opened for support.  Do not replace with e0_u.pt/T.npy.
    a = np.load(path, mmap_mode="r")
    if a.shape != (33, 100, 96) or a.dtype != np.float32: raise RuntimeError(f"M2 activity drift: {path}: {a.shape}/{a.dtype}")
    return a


def _m2_item(query: Mapping[str, Any], activity_path: Path, *, support: Mapping[str, Any]) -> dict[str, Any]:
    a = _m2_activity(activity_path)
    item = {k: query[k] for k in ("X", "Y", "starts", "pad", "hashes")}
    item.update({"activity": a, "support_provenance": dict(support)})
    item["hashes"] = {**dict(item["hashes"]), "calib_activity.npy": _sha(activity_path)}
    return _hash_item(item)


def _load_m2_query_modules():
    train = _module("_activity_m2_static_train", HERE / "m2_static_train.py")
    score = _module("_activity_m2_static_score", HERE / "m2_static_score.py")
    return train, score


def load_m2_data(*, include_eval: bool = False) -> dict[str, Any]:
    """M2 raw-M33 bank.  Support loading touches no query labels or T/E0 files."""
    train_reader, score_reader = _load_m2_query_modules()
    source_q = train_reader.load_static_surface("source_train", cache_root=M2_CACHE)
    val_q = train_reader.load_static_surface("source_minival", cache_root=M2_CACHE)
    train: dict[str, Any] = {}; validation: dict[str, Any] = {}
    for session, q in source_q.items():
        d = M2_CACHE / "source_train" / session
        mapping = json.loads((d / "mapping.json").read_text())
        boundary = int(mapping["support_boundary_padded_bin"])
        starts = np.asarray(q["starts"], np.int64)
        keep = post_support_indices(starts, boundary, context=50)
        if len(keep) != len(starts): raise RuntimeError(f"{session}: source query/support crossing")
        p = {"kind": "held-in-calib raw M33", "trial_ids": list(range(33)),
             "strict_query_disjoint": True, "query_boundary_padded": boundary,
             "activity_path": str(d / "calib_activity.npy")}
        train[session] = _m2_item(q, d / "calib_activity.npy", support=p)
        # minival is a distinct query recording; use the same held-in calibration activity.
        validation[session] = _m2_item(val_q[session], d / "calib_activity.npy", support={**p, "query_file": "held-in-minival", "support_reused_from": "source_train"})
    evaluation: dict[str, Any] = {}
    if include_eval:
        resolve_m2_ext6_root()  # follow the RELOCATED pointer when the canonical dir moved
        manifest = json.loads((M2_EXT6 / "manifest.json").read_text())
        eq = score_reader.query_surface(M2_QUERY)
        for session, q in eq.items():
            d = M2_EXT6 / session
            row = manifest["sessions"][session]
            ext4_mapping=M2_CACHE/"ext4"/session/"mapping.json"
            # Four shared sessions have an immutable, byte-aligned EXT4 mapping.
            # Nov-24 records expose exactly M33 and hence no strict post-support row.
            nov24={"ses-2020-11-24-Run1","ses-2020-11-24-Run2"}
            if not ext4_mapping.is_file() and session not in nov24: raise FileNotFoundError(ext4_mapping)
            boundary=(int(json.loads(ext4_mapping.read_text())["support_boundary_padded_bin"])
                      if ext4_mapping.is_file() else int(len(q["X"])))
            post = post_support_indices(q["starts"], boundary, context=50)
            if np.any(np.asarray(q["starts"])[post] + 50 > len(q["X"])): raise RuntimeError(f"{session}: post-M33 bounds")
            item = _m2_item(q, d / "calib_activity.npy", support={
                "kind": "public held-out-calib raw M33", "trial_ids": list(range(33)),
                "strict_query_disjoint": False, "historical_allqueries_overlap_possible": True,
                "strict_post33_boundary_padded": boundary,
                "raw_nwb": row["raw_nwb"], "raw_nwb_sha256": row["raw_nwb_sha256"],
                "activity_path": str(d / "calib_activity.npy")})
            item["post_support_indices"] = post
            if ext4_mapping.is_file():
                item["hashes"]["ext4_mapping.json"]=_sha(ext4_mapping)
            evaluation[session] = _hash_item(item)
    return {"train": train, "validation": validation, "evaluation": evaluation,
            "metadata": {"task": "m2", "context": 50, "units": 96, "outputs": 2,
                         "support_geometry": [33, 100, 96], "behavior_scale": 5.0,
                         "source_query_support": "strictly disjoint by padded start >= trial33 boundary",
                         "validation_support": "source_train held-in-calib activity reused; minival never supplies support",
                         "evaluation_support": "raw EXT6 M33; target encoder must be eval/no_grad"}}


def _m1_activity_item(session: str, query: Mapping[str, Any], path: Path, *, heldout: bool) -> dict[str, Any]:
    # FalconDataset owns the only established M1 cubic trialization. Its calibration
    # tensor is separated immediately; Y is never passed into this support branch.
    import h5py
    from scipy.interpolate import interp1d
    raw=np.asarray(query["X"],np.float32)[99:]
    with h5py.File(path,"r") as f:
        ts=np.asarray(f["acquisition/eval_mask/timestamps"],np.float64)
        valid=np.asarray(f["acquisition/eval_mask/data"],bool)
        starts_t=np.asarray(f["intervals/trials/start_time"],np.float64)
    if len(ts)!=len(raw) or len(valid)!=len(raw): raise RuntimeError(f"{session}: query neural/timestamp topology drift")
    changes=np.zeros(len(raw),bool); changes[np.searchsorted(ts,starts_t,side="left").clip(0,len(raw)-1)]=True
    # Retain this coordinate for query filtering.  ``starts`` below is a
    # compressed calibration coordinate after eval-mask removal and must never
    # be compared with padded query starts.
    raw_trial_starts=np.flatnonzero(changes).astype(np.int64)
    raw=raw[valid]; changes=changes[valid]; starts=np.flatnonzero(changes).astype(np.int64)
    pieces=[]
    for i,lo in enumerate(starts[:10]):
        hi=starts[i+1] if i+1<len(starts) else len(raw); z=raw[lo:hi]
        if len(z)<4: raise RuntimeError(f"{session}: short neural trial")
        pieces.append(interp1d(np.linspace(0,1,len(z)),z,axis=0,kind="cubic",fill_value="extrapolate")(np.linspace(0,1,1024)).astype(np.float32))
    a=np.ascontiguousarray(np.stack(pieces),np.float32)
    if a.shape != (10, 1024, 64): raise RuntimeError(f"{session}: M1 activity drift {a.shape}")
    out = dict(query); out["activity"] = a
    trial_starts = raw_trial_starts + 99
    if not heldout and trial_starts.size < 11: raise RuntimeError(f"{session}: M1 lacks post-M10 trial")
    out["support_provenance"] = {"trial_ids": list(range(10)), "activity_only": True,
        "trialization": "Falcon cubic/raw M10", "strict_query_disjoint": not heldout,
        "historical_allqueries_overlap": bool(heldout), "query_boundary_padded": int(trial_starts[10]) if not heldout else None,
        "calibration_trial_starts_compressed": starts[:10].tolist(),
        "raw_nwb": str(path), "raw_nwb_sha256": _sha(path)}
    return _hash_item(out)


def load_m1_data(*, include_eval: bool = False) -> dict[str, Any]:
    data = _module("_activity_m1_static", HERE / "m1_static_data.py")
    source = data.load_source(); train = {}
    for s, q in source["items"].items():
        # Query item is deliberately filtered after support boundary; the static sampler is not reused.
        item = _m1_activity_item(s, q, data._source_path(s), heldout=False)
        keep = post_support_indices(item["starts"], item["support_provenance"]["query_boundary_padded"], context=100)
        if not len(keep): raise RuntimeError(f"{s}: no post-M10 source query windows")
        for key in ("Y", "starts"):
            item[key] = np.ascontiguousarray(np.asarray(item[key])[keep])
        item["support_provenance"]["query_filter"] = "whole W100 padded start >= Falcon trial_start_indices[10]"
        train[s] = _hash_item(item)
    evaluation = {}
    if include_eval:
        for s,q in data.load_heldout()["items"].items(): evaluation[s] = _m1_activity_item(s, q, data._heldout_path(s), heldout=True)
    return {"train": train, "validation": {}, "evaluation": evaluation, "batches": None,
            "metadata": {"task":"m1", "context":100, "units":64, "outputs":16, "behavior_scale":1.0, "support_geometry":[10,1024,64],
                         "sampler":"new activity route must build a post-M10 sampler; static 6665 plan is not reused",
                         "evaluation_overlap":"HO local query historically overlaps M10 support"}}


def load_h1_data(*, include_eval: bool = False) -> dict[str, Any]:
    # Query labels use the established source cache. Support derives trial selection from
    # raw TrialNum/eval_mask; callers receive only neural support plus an explicit mask.
    sys.path.insert(0, str(WS / "btransform_unified_v1/src")); sys.path.insert(0, str(WS / "SPINT-main"))
    from btransform_unified_v1 import adapters, h1_config
    import h5py
    root=WS/"SPINT-main/data/000954/sub-HumanPitt-held-in-calib"
    paths={"ses-" + p.stem.split("_ses-",1)[1]:p for p in root.glob("*.nwb")}
    all_cache = adapters._h1_source_cache(); cache = all_cache["train"]; train = {}; validation = {}
    for s in h1_config.H1_ALL_SESSIONS:
        # Support does not invoke load_nwb: neural is the already-opened source
        # query plane, while HDF5 supplies only TrialNum/eval_mask metadata.
        with h5py.File(paths[s],"r") as f:
            tid=np.asarray(f["acquisition/TrialNum/data"],np.float64); ev=np.asarray(f["acquisition/eval_mask/data"],bool)
        vals=[]
        for z in tid[ev & np.isfinite(tid)]:
            if not vals or z != vals[-1]: vals.append(float(z))
        if len(vals)<4: raise RuntimeError(f"{s}: needs M3 plus query")
        first3=tuple(vals[:3]); neural=np.asarray(cache[s]["neural"],np.float32)
        parts = [neural[ev & (tid == v)] for v in first3]
        # C2 identity contract is three trialwise 1024-bin tensors.  Interpolation
        # consumes neural activity only; trial IDs and eval_mask select the bins.
        from scipy.interpolate import interp1d
        a=np.empty((3,1024,176),np.float32)
        for i,x in enumerate(parts):
            if len(x) < 2: raise RuntimeError(f"{s}: insufficient valid support activity")
            old=np.linspace(0.,1.,len(x),dtype=np.float64); new=np.linspace(0.,1.,1024,dtype=np.float64)
            a[i]=interp1d(old,x,axis=0,kind="cubic",fill_value="extrapolate")(new).astype(np.float32)
        row=cache[s]; ends=np.asarray(row["query_starts"],np.int64)+h1_config.FULL_WINDOW-1; ends=ends[np.asarray(row["eval_mask"],bool)[ends]]
        first3_end=int(np.flatnonzero(ev & np.isin(tid,np.asarray(first3)))[-1])
        provenance={"trial_ids":list(first3),"resample_bins":1024,"resample_kind":"cubic","support_activity_mask":"eval_mask & TrialNum == trial_id","strict_query_disjoint":True,"first3_end":first3_end,"raw_nwb":str(paths[s]),"raw_nwb_sha256":_sha(paths[s]),"behavior_metadata_read":"TrialNum/eval_mask only"}
        train[s]=_hash_item({"X":np.pad(np.asarray(row["neural"],np.float32),((299,0),(0,0))),"Y":np.asarray(row["velocity"],np.float32)[ends],"starts":ends,"pad":299,"activity":a,"support_provenance":provenance,"hashes":{"raw_nwb":_sha(paths[s])}})
        mv=all_cache["minival"][s]; me=np.asarray(mv["query_starts"],np.int64)+h1_config.FULL_WINDOW-1; me=me[np.asarray(mv["eval_mask"],bool)[me]]
        validation[s]=_hash_item({"X":np.pad(np.asarray(mv["neural"],np.float32),((299,0),(0,0))),"Y":np.asarray(mv["velocity"],np.float32)[me],"starts":me,"pad":299,"activity":a,"support_provenance":{**provenance,"query_file":"held-in-minival","support_reused_from":"held-in-calib"},"hashes":{"raw_nwb":_sha(paths[s])}})
    evaluation={}
    if include_eval:
        # Public M3 local face. Query supervision is isolated below; support
        # selection uses only neural, TrialNum and eval_mask.
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
        from pynwb import NWBHDF5IO
        ho=WS/"SPINT-main/data/000954/sub-HumanPitt-held-out-calib"
        for path in sorted(ho.glob("*.nwb")):
            session=path.stem.split("_ses-",1)[1]
            neural, velocity, _change, ev = load_nwb(path,FalconTask.h1)
            with NWBHDF5IO(str(path),'r',load_namespaces=True) as f: tid=np.asarray(f.read().acquisition['TrialNum'].data[:],np.float64)
            ev=np.asarray(ev,bool); values=[]
            for x in tid[ev & np.isfinite(tid)]:
                if not values or x != values[-1]: values.append(float(x))
            if len(values)!=3: raise RuntimeError(f"{session}: public H1 must be M3")
            from scipy.interpolate import interp1d
            a=np.empty((3,1024,176),np.float32)
            for i,v in enumerate(values):
                z=np.asarray(neural,np.float32)[ev & (tid==v)]; old=np.linspace(0,1,len(z)); new=np.linspace(0,1,1024)
                a[i]=interp1d(old,z,axis=0,kind="cubic",fill_value="extrapolate")(new).astype(np.float32)
            ends=np.flatnonzero(ev).astype(np.int64)
            evaluation[session]=_hash_item({"X":np.pad(np.asarray(neural,np.float32),((299,0),(0,0))),"Y":np.asarray(velocity,np.float32)[ends],"starts":ends,"pad":299,"activity":a,
              "support_provenance":{"trial_ids":values,"resample_bins":1024,"resample_kind":"cubic","historical_allqueries_overlap":True,"public_m3_local_only":True,"raw_nwb":str(path),"raw_nwb_sha256":_sha(path)},"hashes":{"raw_nwb":_sha(path)}})
    return {"train":train,"validation":validation,"evaluation":evaluation,"metadata":{"task":"h1","context":300,"units":176,"outputs":7,"behavior_scale":float(h1_config.TARGET_MULTIPLIER),"support_geometry":[3,1024,176],"support":"first three eval-valid trials, each neural-only cubic-resampled to 1024; minival reuses held-in-calib support"}}


def load_task_data(task: str, *, include_eval: bool = False) -> dict[str, Any]:
    task = task.lower()
    if task == "m2": return load_m2_data(include_eval=include_eval)
    if task == "m1": return load_m1_data(include_eval=include_eval)
    if task == "h1": return load_h1_data(include_eval=include_eval)
    raise ValueError(f"unknown task: {task}")
