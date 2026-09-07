"""Fail-closed, archive-only report for the completed H1 cold-history phase.

No model, cache, training, or scoring code is imported here.  The report only
deserializes prediction arrays after their recorded NPZ bytes have been hashed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

COUNT, SESSIONS, OUTPUTS = 20325, 13, 7
COLD, FULL, COLD_END_EXCLUSIVE = 8702, 11623, 699
SOURCE208_PER_SESSION, SOURCE208_COUNT = 16, 208
ARMS, CELLS, MODES = ("flat", "route"), ("CONTROL", "PREFIX"), ("RAW", "EMA")
WORKER_SCHEMA = "h1_cold_history_phase_worker_v2"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise RuntimeError("JSON authority must be an object")
    return value


def atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as h:
        temporary = Path(h.name); json.dump(value, h, sort_keys=True, indent=2, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
    os.replace(temporary, path)


def _require_hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise RuntimeError(label + " must be lowercase SHA-256")
    return value


def _same_number(actual, expected, label):
    if not isinstance(actual, (int, float)) or not np.isfinite(actual) or abs(float(actual) - expected) > 1e-12: raise RuntimeError(label + " metric receipt drift")


def _json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _bind_file_map(snapshot, files, label):
    if not isinstance(files,dict) or not files: raise RuntimeError(label+" file closure missing")
    for path,digest in files.items():
        if not isinstance(path,str): raise RuntimeError(label+" file closure path drift")
        _bind(snapshot,path,_require_hash(digest,label+" file hash"))


def metric(a):
    p, y, sid = a["prediction"].astype(np.float64, copy=False), a["target"].astype(np.float64, copy=False), a["session_id"]
    sessions = sorted(set(sid.tolist()))
    if not sessions: raise RuntimeError("empty metric session roster")
    def r2(x, z):
        d = float(np.square(z-z.mean(0, keepdims=True)).sum(dtype=np.float64))
        if not np.isfinite(d) or d <= 0: raise RuntimeError("nonpositive target variance")
        return float(1-float(np.square(x-z).sum(dtype=np.float64))/d)
    per = {session: r2(p[sid == session], y[sid == session]) for session in sessions}
    worst = min(per, key=per.get)
    return {"n_bins": int(len(p)), "r2_concat_float64": r2(p,y), "equal_session_mean_r2_float64": float(np.mean(list(per.values()), dtype=np.float64)), "worst_session_r2_float64": float(per[worst]), "worst_session_id": worst, "per_session_r2_float64": per}


def _validate_session_geometry(sid, coordinate, *, source208):
    sessions = sorted(set(sid.tolist()))
    if len(sessions) != SESSIONS: raise RuntimeError("exact 13-session archive required")
    if sid.tolist() != sorted(sid.tolist()): raise RuntimeError("session archive order must be sorted")
    for session in sessions:
        values = coordinate[sid == session]
        if ((source208 and len(values) != SOURCE208_PER_SESSION) or np.any(values < 0) or np.any(np.diff(values) <= 0)):
            raise RuntimeError("per-session coordinate cardinality/order drift")


def load(path, *, source208=False):
    with np.load(path, allow_pickle=False) as z: a = {key:z[key] for key in z.files}
    coordinate, n = ("start", SOURCE208_COUNT) if source208 else ("end", COUNT)
    if (set(a) != {"prediction","target","session_id",coordinate} or a["prediction"].shape != (n,OUTPUTS) or a["target"].shape != (n,OUTPUTS) or a["session_id"].shape != (n,) or a[coordinate].shape != (n,) or a["prediction"].dtype != np.float64 or a["target"].dtype != np.float64 or a["session_id"].dtype.kind != "U" or a[coordinate].dtype != np.int64 or not np.isfinite(a["prediction"]).all() or not np.isfinite(a["target"]).all()):
        raise RuntimeError("archive geometry/dtype/finite drift")
    _validate_session_geometry(a["session_id"], a[coordinate], source208=source208)
    return a


def analysis(a):
    end = a["end"]
    masks = {"all":np.ones(COUNT,bool), "cold_history_lt_699":end<COLD_END_EXCLUSIVE, "full_w700_ge_699":end>=COLD_END_EXCLUSIVE}
    if [int(x.sum()) for x in masks.values()] != [COUNT,COLD,FULL] or np.any(masks["cold_history_lt_699"] & masks["full_w700_ge_699"]): raise RuntimeError("fixed cold/full partition drift")
    groups = {name:metric({"prediction":a["prediction"][mask],"target":a["target"][mask],"session_id":a["session_id"][mask]}) for name,mask in masks.items()}
    keys = groups["all"]["per_session_r2_float64"].keys()
    if any(row["per_session_r2_float64"].keys() != keys for row in groups.values()): raise RuntimeError("each cold/full group must retain all sessions")
    sse = {name:float(np.square(a["prediction"][mask]-a["target"][mask]).sum(dtype=np.float64)) for name,mask in masks.items()}
    if not np.isclose(sse["all"],sse["cold_history_lt_699"]+sse["full_w700_ge_699"],rtol=1e-12,atol=1e-15): raise RuntimeError("cold/full SSE recombination drift")
    return {"groups":groups,"sse_float64":sse}


def _delta(left,right):
    result = {}
    for group,value in left["groups"].items():
        other = right["groups"].get(group)
        if other is None or value["per_session_r2_float64"].keys() != other["per_session_r2_float64"].keys(): raise RuntimeError("delta session/group alignment drift")
        result[group] = {"pooled_r2_delta":value["r2_concat_float64"]-other["r2_concat_float64"], "equal_session_r2_delta":value["equal_session_mean_r2_float64"]-other["equal_session_mean_r2_float64"], "worst_session_r2_delta":value["worst_session_r2_float64"]-other["worst_session_r2_float64"], "left_worst_session_id":value["worst_session_id"], "right_worst_session_id":other["worst_session_id"], "per_session_r2_delta":{s:value["per_session_r2_float64"][s]-other["per_session_r2_float64"][s] for s in value["per_session_r2_float64"]}}
    return result


def _bind(snapshot,path,expected=None):
    path = Path(path)
    if not path.is_file(): raise RuntimeError("bound input file missing: "+str(path))
    actual = sha(path)
    if expected is not None and actual != _require_hash(expected,"recorded input hash"): raise RuntimeError("bound input SHA drift: "+str(path))
    if snapshot.setdefault(str(path),actual) != actual: raise RuntimeError("same input path has conflicting bindings")
    return path


def _archive_item(snapshot,item,label):
    if not isinstance(item,dict) or not isinstance(item.get("path"),str): raise RuntimeError(label+" archive receipt topology drift")
    return _bind(snapshot,item["path"],_require_hash(item.get("sha256"),label+" archive hash"))


def verify_worker(root,arm,snapshot):
    directory=root/arm; receipt_path,authority_path,progress_path=(directory/name for name in ("receipt.json","input_authority.json","progress.json"))
    for path in (receipt_path,authority_path,progress_path): _bind(snapshot,path)
    receipt,authority,progress=read(receipt_path),read(authority_path),read(progress_path)
    if (receipt.get("schema") != WORKER_SCHEMA or receipt.get("status") != "COMPLETE_NO_SELECTION_OR_PROMOTION" or receipt.get("pre") != receipt.get("post") or receipt.get("no_selection_or_promotion") is not True or receipt.get("not_official_or_external") is not True or receipt.get("ema_updates") != {x:1462 for x in CELLS}): raise RuntimeError("worker receipt completion/authority drift")
    pre=receipt.get("pre")
    if not isinstance(pre,dict) or authority.get("bindings") != pre or authority.get("authorization_sha256") != receipt.get("authorization_sha256"): raise RuntimeError("worker input-authority binding drift")
    if progress.get("epochs") != receipt.get("identities") or progress.get("checkpoints") != receipt.get("checkpoints") or progress.get("evaluations") != receipt.get("evaluations"): raise RuntimeError("worker progress/receipt content drift")
    if pre.get("arm") != arm or pre.get("output") != str(directory) or not isinstance(pre.get("formal"),str) or not Path(pre["formal"]).is_absolute() or pre.get("physical_gpu") not in (0,1): raise RuntimeError("worker intended arm/output/formal/GPU identity drift")
    if not isinstance(pre.get("protocol"),dict) or pre.get("protocol_sha256") != _json_digest(pre["protocol"]): raise RuntimeError("worker protocol/hash identity drift")
    _bind_file_map(snapshot,pre.get("files"),"worker")
    formal_artifact,formal_source=pre.get("formal_artifact"),pre.get("formal_source")
    if not isinstance(formal_artifact,dict) or not isinstance(formal_source,dict): raise RuntimeError("formal artifact/source closure topology drift")
    _bind_file_map(snapshot,formal_artifact.get("files"),"formal artifact")
    _bind_file_map(snapshot,formal_source.get("files"),"formal source")
    rows=receipt.get("updates"); expected=[(e,b) for e in (1,2) for b in range(731)]
    if (not isinstance(rows,list) or [(x.get("epoch"),x.get("batch")) for x in rows] != expected or any(not isinstance(x.get("rows"),int) or not 1<=x["rows"]<=32 or not isinstance(x.get("cold_rows"),int) or not 0<=x["cold_rows"]<=x["rows"] or not isinstance(x.get("loss"),dict) or set(x["loss"])!=set(CELLS) or any(not isinstance(v,(int,float)) or not np.isfinite(v) for v in x["loss"].values()) for x in rows)): raise RuntimeError("exact finite 2x731 update/source-window topology drift")
    if sum(x["rows"] for x in rows[:731]) != 23212 or sum(x["rows"] for x in rows[731:]) != 23212: raise RuntimeError("exact 23212 source windows per epoch required")
    identities=receipt.get("identities")
    if not isinstance(identities,dict) or set(identities)!={"1","2"}: raise RuntimeError("worker epoch identity topology drift")
    for epoch in ("1","2"):
        if not isinstance(identities[epoch],dict) or set(identities[epoch]) != {"sampler_sha256","keep_sha256","prefix_sha256"}: raise RuntimeError("worker sampler/keep/prefix identity drift")
        for value in identities[epoch].values(): _require_hash(value,"epoch identity")
    formal_identities=pre.get("formal_epoch_identities")
    if not isinstance(formal_identities,dict) or set(formal_identities)!={"1","2"}: raise RuntimeError("formal sampler/keep identity topology drift")
    for epoch in ("1","2"):
        formal_identity=formal_identities[epoch]
        if (not isinstance(formal_identity,dict) or set(formal_identity)!={"sampler_sha256","keep_sha256"}
                or any(_require_hash(formal_identity[key],"formal epoch identity") != identities[epoch][key] for key in formal_identity)):
            raise RuntimeError("worker/formal sampler/keep identity drift")
    checkpoints=receipt.get("checkpoints")
    if not isinstance(checkpoints,list) or [x.get("epoch") for x in checkpoints] != [1,2]: raise RuntimeError("worker checkpoint topology drift")
    for item,epoch in zip(checkpoints,("1","2"),strict=True):
        if item.get("identities") != identities[epoch] or not isinstance(item.get("path"),str): raise RuntimeError("checkpoint identity/path drift")
        _bind(snapshot,item["path"],_require_hash(item.get("sha256"),"checkpoint hash"))
    evaluations=receipt.get("evaluations")
    if not isinstance(evaluations,dict) or set(evaluations)!={"epoch1_source208","epoch2_complete"}: raise RuntimeError("worker evaluation topology drift")
    for endpoint in evaluations.values():
        if not isinstance(endpoint,dict) or set(endpoint)!=set(CELLS): raise RuntimeError("worker cell evaluation topology drift")
        for cell in CELLS:
            if not isinstance(endpoint[cell],dict) or set(endpoint[cell])!=set(MODES): raise RuntimeError("worker RAW/EMA topology drift")
            for mode in MODES: _archive_item(snapshot,endpoint[cell][mode].get("archive"),"worker")
    return receipt


def _check_e2(item, measured):
    for name in ("n_bins","r2_concat_float64","equal_session_mean_r2_float64","worst_session_r2_float64"): _same_number(item.get(name),measured[name],"epoch2 "+name)
    per=item.get("per_session_r2_float64")
    if not isinstance(per,dict) or set(per)!=set(measured["per_session_r2_float64"]): raise RuntimeError("epoch2 per-session metric receipt topology drift")
    for session,score in measured["per_session_r2_float64"].items(): _same_number(per[session],score,"epoch2 per-session")


def _check_e1(item, measured):
    if item.get("windows") != SOURCE208_COUNT or not isinstance(item.get("pooled"),dict): raise RuntimeError("epoch1 source208 receipt topology drift")
    _same_number(item["pooled"].get("r2_concat_float64"),measured["r2_concat_float64"],"epoch1 pooled")
    per=item.get("per_session")
    if not isinstance(per,dict) or set(per)!=set(measured["per_session_r2_float64"]): raise RuntimeError("epoch1 per-session receipt topology drift")
    for session,score in measured["per_session_r2_float64"].items(): _same_number(per[session].get("r2_concat_float64"),score,"epoch1 per-session")


def summarize(root,output):
    # The full gate comes before every authority or archive read.
    if output.exists(): raise FileExistsError(output)
    if os.environ.get("H1_COLD_PHASE_SUMMARY_GO")!="1" or not root.is_absolute() or not output.is_absolute(): raise RuntimeError("explicit summary GO and absolute phase root/output required")
    snapshot={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    workers={arm:verify_worker(root,arm,snapshot) for arm in ARMS}; base=workers["flat"]["pre"]
    for arm,receipt in workers.items():
        pre=receipt["pre"]
        if pre.get("formal")!=base.get("formal") or pre.get("formal_artifact")!=base.get("formal_artifact") or pre.get("formal_source")!=base.get("formal_source") or pre.get("protocol")!=base.get("protocol") or pre.get("protocol_sha256")!=base.get("protocol_sha256"): raise RuntimeError("cross-arm formal/protocol authority drift")
        auth=root.parent/f"h1_cold_phase_authorization_{arm}_v1.json"; _bind(snapshot,auth,_require_hash(receipt.get("authorization_sha256"),"authorization hash"))
        external=read(auth)
        if external.get("schema")!="h1_cold_phase_root_authorization_v2" or external.get("bindings")!=pre: raise RuntimeError("external authorization binding/schema drift")
    for epoch in ("1","2"):
        if workers["flat"]["identities"][epoch]!=workers["route"]["identities"][epoch]: raise RuntimeError("cross-arm sampler/keep/prefix identity drift")
    if workers["flat"]["pre"]["physical_gpu"] == workers["route"]["pre"]["physical_gpu"]: raise RuntimeError("workers must retain distinct intended physical GPUs")
    result={"schema":"h1_cold_history_phase_archive_summary_v2","status":"COMPLETE_ARCHIVE_ONLY_NO_SELECTION","scope":"hash-bound existing receipts/checkpoints/FP64 NPZ archives only; no model/data import, inference, fitting, selection, or promotion","primary":"fixed epoch2 EMA ALL20325 pooled R2 PREFIX minus matched CONTROL","split":{"cold_history_lt_699":"endpoint end < 699","full_w700_ge_699":"endpoint end >= 699"},"workers":{},"prefix_minus_control":{},"primary_fixed_e2_ema_pooled_prefix_minus_control":{},"raw_minus_ema":{},"baseline_selected":{},"cell_minus_baseline_selected":{},"source208_epoch1":{},"inputs_sha256_pre":snapshot}
    reference=None; e1reference=None
    for arm,receipt in workers.items():
        endpoint=receipt["evaluations"]["epoch2_complete"]; result["workers"][arm]={}; analyses={}
        for cell in CELLS:
            result["workers"][arm][cell]={}
            for mode in MODES:
                item=endpoint[cell][mode]; arrays=load(_archive_item(snapshot,item.get("archive"),"epoch2"))
                if reference is None: reference={key:arrays[key].copy() for key in ("target","session_id","end")}
                if any(not np.array_equal(arrays[key],reference[key]) for key in reference): raise RuntimeError("epoch2 target/session/end identity drift")
                measured=metric(arrays); _check_e2(item,measured); key=cell+"_"+mode; analyses[key]=analysis(arrays); result["workers"][arm][cell][mode]={"archive":item["archive"],**analyses[key]}
        result["prefix_minus_control"][arm]={mode:_delta(analyses["PREFIX_"+mode],analyses["CONTROL_"+mode]) for mode in MODES}
        result["primary_fixed_e2_ema_pooled_prefix_minus_control"][arm]=result["prefix_minus_control"][arm]["EMA"]["all"]["pooled_r2_delta"]
        result["raw_minus_ema"][arm]={cell:_delta(analyses[cell+"_RAW"],analyses[cell+"_EMA"]) for cell in CELLS}
        baseline_path=Path(receipt["pre"].get("formal",""))/"exports"/f"{arm}_selected_complete_native_float64.npz"
        baseline_expected=receipt["pre"]["formal_artifact"]["files"].get(str(baseline_path))
        baseline=load(_bind(snapshot,baseline_path,_require_hash(baseline_expected,"selected formal baseline hash")))
        if any(not np.array_equal(baseline[key],reference[key]) for key in reference): raise RuntimeError("selected formal baseline target/session/end drift")
        base_analysis=analysis(baseline); result["baseline_selected"][arm]={"archive":{"path":str(baseline_path),"sha256":snapshot[str(baseline_path)]},**base_analysis}; result["cell_minus_baseline_selected"][arm]={key:_delta(value,base_analysis) for key,value in analyses.items()}
        result["source208_epoch1"][arm]={}
        for cell in CELLS:
            result["source208_epoch1"][arm][cell]={}
            for mode in MODES:
                item=receipt["evaluations"]["epoch1_source208"][cell][mode]; arrays=load(_archive_item(snapshot,item.get("archive"),"epoch1"),source208=True)
                if e1reference is None: e1reference={key:arrays[key].copy() for key in ("target","session_id","start")}
                if any(not np.array_equal(arrays[key],e1reference[key]) for key in e1reference): raise RuntimeError("epoch1 source208 target/session/start identity drift")
                measured=metric(arrays); _check_e1(item,measured); tstd=float(arrays["target"].std(dtype=np.float64))
                if tstd<=0: raise RuntimeError("source208 target standard deviation drift")
                result["source208_epoch1"][arm][cell][mode]={"archive":item["archive"],"metric":measured,"prediction_std_float64":float(arrays["prediction"].std(dtype=np.float64)),"target_std_float64":tstd,"prediction_to_target_std_ratio":float(arrays["prediction"].std(dtype=np.float64)/tstd)}
    # Freshly re-read every worker, then verify every input bound before any NPZ load.
    fresh={}; fresh_workers={arm:verify_worker(root,arm,fresh) for arm in ARMS}
    if fresh_workers!=workers or {path:sha(path) for path in snapshot}!=snapshot: raise RuntimeError("fresh post worker/archive authority mutation")
    result["inputs_sha256_post"]={path:sha(path) for path in snapshot}; atomic(result,output); return result


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--phase-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(argv); return summarize(args.phase_root,args.output)


if __name__=="__main__": main()
