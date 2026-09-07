"""Immutable fixed-screen-bound source snapshot for Context ``ser_context_q4``.

The snapshot is the *training* authority, not merely an evaluator convenience.
It binds the exact fold-0 map selected by the immutable CPU design screen and
then requires map-derived cache/normalizer/manifest reconstruction to agree.
"""
from __future__ import annotations

import hashlib, json, os, stat, tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1
from src.data.h1_context_event_carrier import ContextScalarNormalizer

SCHEMA, RECEIPT_SCHEMA, MODE = "h1_context_event_source_snapshot_v1", "h1_context_event_source_snapshot_receipt_v1", 0o444
MAP_FIELDS = ("active_mask", "feature_mean", "feature_scale", "projection", "latent_scale", "energy_ratio")
FIELDS = MAP_FIELDS + ("source_carriers",)
ROOT = Path(__file__).resolve().parents[3]
FIXED_SCREEN = ROOT / "sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"
FIXED_SCREEN_SHA256 = "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"
FIXED_MAP_SHA256 = "50c0c55969e6898e00846302f97a6637ac78b0af6715f33de428a9f3d845e525"
FIXED_ARRAY_SHA256 = {"active_mask": "1b109aa95e5bee6519b035de92dbdddad5b8660ab014f429105a934b4b101223", "feature_mean": "cd50c588dc3992f1c07ba65a56553261641d7dcb794d4a6b331e7c8fe2f1b521", "feature_scale": "def0b8f639903128e42d775fb635f440d1259aaf9e0be2911030b25f332de6dc", "projection": "5f9a3f485ee4be33d915f4db587c67b3e8b88eabca76be34ef41a0fbbc6625df", "latent_scale": "164472716c91002aa80cca18ecb79f44e53f5b384757d720596d524695ef8618"}

def _need(ok: bool, msg: str) -> None:
    if not ok: raise ValueError(msg)
def _bytes(v: Mapping[str, Any]) -> bytes: return event_v1.canonical_json_bytes(dict(v))
def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()
def _readonly(p: Path) -> None: _need(p.is_file() and not p.is_symlink() and stat.S_IMODE(p.stat().st_mode)==MODE, f"not immutable 0444 regular file: {p}")
def _write_once(p: Path, raw: bytes) -> None:
    p=p.resolve(); _need(not p.exists(),f"refusing overwrite: {p}");p.parent.mkdir(parents=True,exist_ok=True);fd,n=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".tmp",dir=p.parent);t=Path(n)
    try:
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        _need(not p.exists(),f"destination raced: {p}");os.replace(t,p);p.chmod(MODE);_readonly(p)
    finally:
        if t.exists():t.unlink()
def _fixed_map() -> dict[str, Any]:
    _need(sha256_file(FIXED_SCREEN)==FIXED_SCREEN_SHA256,"immutable fixed CPU screen SHA drift")
    x=json.loads(FIXED_SCREEN.read_text()); m=x["basis_by_candidate_and_outer_date"]["ser_context_q4"]["19250101"]
    _need(m["map_sha256"]==FIXED_MAP_SHA256 and m["array_sha256"]==FIXED_ARRAY_SHA256,"fixed fold0 context map binding drift")
    return m
def _map_body(m: Mapping[str, Any], a: Mapping[str,np.ndarray]) -> dict[str, Any]:
    return {"protocol":design.PROTOCOL,"candidate":"ser_context_q4","outer_date":m["outer_date"],"source_sessions":list(m["source_sessions"]),"source_event_count":int(m["source_event_count"]),"active_mask":event_v1.array_sha256(a["active_mask"]),"feature_mean":event_v1.array_sha256(a["feature_mean"]),"feature_scale":event_v1.array_sha256(a["feature_scale"]),"projection":event_v1.array_sha256(a["projection"]),"latent_scale":event_v1.array_sha256(a["latent_scale"])}
def _normalizer_sha(norm: Mapping[str,Any], shape: list[int]) -> str:
    return event_v1.canonical_sha256({"formula":"s_src=sqrt(mean(source_context_cache^2)); carrier_norm=carrier/max(s_src,1e-12)","s_src":float(norm["s_src"]),"source_cache_sha256":str(norm["source_cache_sha256"]),"shape":shape})
def _latent(m: Mapping[str,Any], a: Mapping[str,np.ndarray]) -> design.LatentMap:
    candidate=next(x for x in design.CANDIDATES if x.name=="ser_context_q4")
    return design.LatentMap(candidate=candidate,outer_date=str(m["outer_date"]),source_sessions=tuple(m["source_sessions"]),raw_dim=int(m["raw_dim"]),active_mask=np.asarray(a["active_mask"],bool),feature_mean=np.asarray(a["feature_mean"],np.float64),feature_scale=np.asarray(a["feature_scale"],np.float64),projection=np.asarray(a["projection"],np.float64),latent_scale=np.asarray(a["latent_scale"],np.float64),energy_ratio=np.asarray(a["energy_ratio"],np.float64),source_event_count=int(m["source_event_count"]),map_sha256=str(m["map_sha256"]))

def write_snapshot(*, snapshot_path: str|Path, receipt_path: str|Path, source_module:Any, expected_manifest_sha256:str,builder_path:str|Path)->dict[str,Any]:
    _need(getattr(source_module,"_setup_done",False),"source setup('fit') required")
    manifest=source_module.pilot_manifest();_need(event_v1.canonical_sha256(manifest)==expected_manifest_sha256,"expected source manifest mismatch")
    fixed=_fixed_map(); mapping,normalizer=source_module.latent_map,source_module.normalizer;arrays={f:np.asarray(getattr(mapping,f)) for f in MAP_FIELDS};arrays["source_carriers"]=np.stack([x.carrier for x in source_module.carrier_cache.entries])
    _need(mapping.manifest()==fixed,"live source map differs from fixed CPU receipt")
    for f,v in arrays.items(): _need(np.isfinite(v).all(),f"nonfinite source map {f}")
    metadata={"schema":SCHEMA,"manifest_sha256":expected_manifest_sha256,"map_manifest":mapping.manifest(),"cache_manifest":source_module.carrier_cache.manifest,"cache_entries":[{"session":x.session_name,"start_index":x.start_index,"trial_values":list(x.trial_values),"carrier_sha256":x.carrier_sha256} for x in source_module.carrier_cache.entries],"normalizer":normalizer.manifest,"array_sha256":{f:event_v1.array_sha256(v) for f,v in arrays.items()},"normalizer_shape":[116,176,5],"fixed_screen":{"path":str(FIXED_SCREEN),"sha256":FIXED_SCREEN_SHA256,"fold0_map_sha256":FIXED_MAP_SHA256,"array_sha256":FIXED_ARRAY_SHA256}}
    p=Path(snapshot_path).resolve();r=Path(receipt_path).resolve();_need(p.suffix==".npz" and r.suffix==".json" and not p.exists() and not r.exists(),"snapshot/receipt must be new .npz/.json");p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{p.name}.",suffix=".npz",dir=p.parent,delete=False) as h:t=Path(h.name)
    try:
        np.savez_compressed(t,**arrays,metadata_json_utf8=np.frombuffer(_bytes(metadata),dtype=np.uint8),manifest_json_utf8=np.frombuffer(_bytes(manifest),dtype=np.uint8));_write_once(p,t.read_bytes())
    finally:
        if t.exists():t.unlink()
    body={"schema":RECEIPT_SCHEMA,"snapshot_schema":SCHEMA,"snapshot":{"path":str(p),"sha256":sha256_file(p),"immutable_mode":"0444"},"expected_manifest_sha256":expected_manifest_sha256,"source_manifest_sha256":expected_manifest_sha256,"context_map_sha256":mapping.map_sha256,"normalizer_sha256":normalizer.normalizer_sha256,"fixed_screen":metadata["fixed_screen"],"builder":{"path":str(Path(builder_path).resolve()),"sha256":sha256_file(builder_path)},"snapshot_module":{"path":str(Path(__file__).resolve()),"sha256":sha256_file(__file__)},"scope":{"setup_calls":["fit"],"target_nwb_opened":False,"gpu_used":False}}
    _write_once(r,_bytes(body));return {"snapshot":str(p),"snapshot_sha256":sha256_file(p),"receipt":str(r),"receipt_sha256":sha256_file(r),"manifest_sha256":expected_manifest_sha256,"context_map_sha256":mapping.map_sha256,"normalizer_sha256":normalizer.normalizer_sha256}

def load_snapshot(receipt_path:str|Path)->dict[str,Any]:
    r=Path(receipt_path).resolve();_readonly(r);body=json.loads(r.read_text());_need(body.get("schema")==RECEIPT_SCHEMA and body.get("snapshot_schema")==SCHEMA,"receipt schema drift")
    p=Path(body["snapshot"]["path"]).resolve();_readonly(p);_need(body["snapshot"].get("immutable_mode")=="0444" and sha256_file(p)==body["snapshot"].get("sha256"),"snapshot SHA/mode drift")
    _need(sha256_file(body["builder"]["path"])==body["builder"]["sha256"] and sha256_file(body["snapshot_module"]["path"])==body["snapshot_module"]["sha256"],"builder/module SHA drift")
    fixed=_fixed_map();_need(body.get("fixed_screen",{}).get("sha256")==FIXED_SCREEN_SHA256 and body["fixed_screen"].get("fold0_map_sha256")==FIXED_MAP_SHA256,"receipt fixed-screen binding drift")
    with np.load(p,allow_pickle=False) as z:
        _need(set(z.files)==set(FIELDS)|{"metadata_json_utf8","manifest_json_utf8"},"snapshot member drift");meta=json.loads(np.asarray(z["metadata_json_utf8"],np.uint8).tobytes());manifest=json.loads(np.asarray(z["manifest_json_utf8"],np.uint8).tobytes());a={f:np.asarray(z[f]) for f in FIELDS}
    _need(event_v1.canonical_sha256(manifest)==meta["manifest_sha256"]==body["source_manifest_sha256"]==body["expected_manifest_sha256"],"manifest SHA drift")
    m=meta["map_manifest"];_need(m==fixed and m["map_sha256"]==body["context_map_sha256"],"fixed context map manifest drift")
    shapes={"active_mask":(22,),"feature_mean":(20,),"feature_scale":(20,),"projection":(20,4),"latent_scale":(4,),"source_carriers":(116,176,5)}
    for f,v in a.items(): _need(np.isfinite(v).all() and (f not in shapes or v.shape==shapes[f]) and event_v1.array_sha256(v)==meta["array_sha256"][f],f"context snapshot array drift: {f}")
    for f,x in FIXED_ARRAY_SHA256.items():_need(meta["array_sha256"][f]==x,"fixed array SHA drift")
    _need(event_v1.canonical_sha256(_map_body(m,a))==FIXED_MAP_SHA256,"recomputed context map SHA drift")
    norm=meta["normalizer"];_need(_normalizer_sha(norm,list(meta["normalizer_shape"]))==norm["normalizer_sha256"]==body["normalizer_sha256"],"normalizer SHA drift")
    _need(manifest["source_map"]==m and manifest["carrier_cache_sha256"]==norm["source_cache_sha256"] and manifest["normalizer"]==norm and manifest["normalizer_sha256"]==norm["normalizer_sha256"],"manifest/map/cache/normalizer cross-check drift")
    _need(meta["cache_manifest"]["cache_sha256"]==manifest["carrier_cache_sha256"] and meta["normalizer"]["source_cache_sha256"]==meta["cache_manifest"]["cache_sha256"],"snapshot cache/normalizer manifest drift")
    return {"receipt":body,"receipt_path":r,"receipt_sha256":sha256_file(r),"snapshot_path":p,"snapshot_sha256":sha256_file(p),"manifest":manifest,"metadata":meta,"arrays":a,"latent_map":_latent(m,a)}

def apply_snapshot_to_source_module(source_module:Any,snapshot:Mapping[str,Any])->None:
    """Compatibility helper; training uses the stronger rebuild route in its DataModule."""
    source_module.latent_map=snapshot["latent_map"];n=snapshot["metadata"]["normalizer"];source_module.normalizer=ContextScalarNormalizer(float(n["s_src"]),str(n["source_cache_sha256"]),str(n["normalizer_sha256"]));source_module._manifest=dict(snapshot["manifest"]);source_module._manifest_sha256=event_v1.canonical_sha256(snapshot["manifest"])
