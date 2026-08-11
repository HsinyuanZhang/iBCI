#!/usr/bin/env python3
"""CPU-only immutable provenance-and-parity receipt for production RT T4d."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEALED = ROOT / "sua_exploration/results/rt_simple_label_v1/stage1/RT_SPARSE_ENDPOINT_STAGE1_RECEIPT_v1.json"
STAGE0B = ROOT / "sua_exploration/results/rt_simple_label_v1/stage0b/RT_SPARSE_ENDPOINT_STAGE0B_RECEIPT_v1.json"
V1 = ROOT / "sua_exploration/results/rt_simple_label_v1/stage2_preflight/RT_SPARSE_ENDPOINT_PRODUCTION_CPU_PARITY_v1.json"
OUT = ROOT / "sua_exploration/results/rt_simple_label_v1/stage2_preflight/RT_SPARSE_ENDPOINT_PRODUCTION_CPU_PARITY_v2.json"
TOL = 1.0e-6

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CUDA_VISIBLE_DEVICES must be empty')
    if OUT.exists(): raise FileExistsError(f'immutable receipt exists: {OUT}')
    from src.data.rt_sparse_endpoint_loader import load_rt_sparse_endpoint_t4d_session
    sealed=json.loads(SEALED.read_text()); stage0b=json.loads(STAGE0B.read_text()); rows={}
    sealed_sessions = sealed.get("sessions", {})
    if not isinstance(sealed_sessions, dict) or len(sealed_sessions) != 15: raise RuntimeError("sealed Stage1 session allowlist invalid")
    # Provenance is checked for all 15 files before the production loader may
    # read any payload: exact canonical path, byte length, and content SHA.
    paths: dict[str, Path] = {}
    for name, old in sealed_sessions.items():
        nwb = old.get("nwb", {}); path = Path(str(nwb.get("path", ""))).resolve()
        if not path.is_file() or str(path) != str(Path(str(nwb.get("path", "")).strip()).resolve()): raise RuntimeError(f"{name}: sealed NWB path missing/drift")
        if int(path.stat().st_size) != int(nwb.get("bytes", -1)) or sha(path) != nwb.get("sha256"): raise RuntimeError(f"{name}: sealed NWB bytes/SHA drift")
        paths[name] = path
    if len(paths) != 15: raise RuntimeError("sealed provenance allowlist cardinality mismatch")
    for name, path in sorted(paths.items()):
        raw=load_rt_sparse_endpoint_t4d_session(path); old=sealed_sessions[name]
        if raw['session_name'] != name: raise RuntimeError(f"loader session mismatch {raw['session_name']} != {name}")
        expected_ac=np.asarray(old['full_coefficients_ac'],dtype=np.float64); actual=np.asarray(raw['t4d_raw_feature'][:,:2],dtype=np.float64)
        err=float(np.max(np.abs(expected_ac-actual)))
        audit=raw['t4d_audit']; expected_rows=int(old['reach_counts']['support'])
        if err > TOL or audit['eligible_reach_rows'] != expected_rows or audit['design_rank'] != old['full_fit']['rank'] or abs(float(audit['design_condition'])-float(old['full_fit']['condition'])) > 1e-12:
            raise RuntimeError(f'parity failed {name}: error={err}')
        rows[name]={"max_abs_ac_error":err,"eligible_reach_rows":audit['eligible_reach_rows'],"stage1_eligible_reach_rows":expected_rows,"rank":audit['design_rank'],"condition":audit['design_condition'],"unique_endpoint_coordinate_samples":audit['unique_endpoint_coordinate_samples'],"unique_endpoint_coordinate_scalars":audit['unique_endpoint_coordinate_scalars'],"carrier_sha256":audit['carrier_sha256_before_dense_target'],"access_log":audit['access_log']}
    if len(rows)!=15: raise RuntimeError(f'expected 15 RT sessions, got {len(rows)}')
    semantic_scalars=sum(int(row['m24']['endpoint_scalar_accounting']['raw_scalar_coordinates']) for row in stage0b['sessions'].values())
    dense_scalars=sum(int(row['m24']['endpoint_scalar_accounting']['dense_rt_target_scalars']) for row in stage0b['sessions'].values())
    actual_samples=sum(int(row['unique_endpoint_coordinate_samples']) for row in rows.values()); actual_scalars=sum(int(row['unique_endpoint_coordinate_scalars']) for row in rows.values())
    if (semantic_scalars,dense_scalars,actual_samples,actual_scalars)!=(2764,15710,2751,5502): raise RuntimeError('frozen scalar accounting drift')
    payload={"schema":"rt_sparse_endpoint_production_cpu_parity_v2","status":"PASS_PRODUCTION_T4D_STAGE1_PARITY_AND_NWB_PROVENANCE_NO_GPU","supersedes":{"v1_path":str(V1),"v1_sha256":sha(V1),"reason":"v1 lacked sealed per-NWB path/bytes/SHA verification"},"sealed_stage1":{"path":str(SEALED),"sha256":sha(SEALED)},"sealed_stage0b":{"path":str(STAGE0B),"sha256":sha(STAGE0B)},"sealed_nwb_provenance_verified_before_payload_read":True,"tolerance":{"ac_max_abs":TOL,"rank_exact":True,"condition_abs":1e-12},"coordinate_accounting":{"semantic_endpoint_coordinates":{"meaning":"interpolated unique endpoint coordinates; not raw coordinate IO","scalars":semantic_scalars,"dense_target_scalars":dense_scalars,"reduction_vs_dense":dense_scalars/semantic_scalars},"actual_raw_coordinate_io":{"meaning":"deduplicated endpoint/bracket coordinate samples actually read from NWB","samples":actual_samples,"scalars":actual_scalars,"dense_target_scalars":dense_scalars,"reduction_vs_dense":dense_scalars/actual_scalars}},"sessions":rows,"compute":{"cuda_visible_devices":"","gpu_context_created":False,"decoder_constructed":False,"trainer_constructed":False,"nwb_position_coordinate_policy":"M24 endpoint/bracket samples only"},"implementation":{"production_loader":{"path":str((ROOT/'streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py')),"sha256":sha(ROOT/'streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py')},"script":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())}}}
    OUT.parent.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(prefix='.t4d-parity-',dir=OUT.parent))
    try:
        target=tmp/OUT.name; target.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.chmod(target,0o444); os.replace(target,OUT); tmp.rmdir()
    except Exception:
        for x in tmp.iterdir(): x.unlink()
        tmp.rmdir(); raise
    print(OUT)
if __name__=='__main__': main()
