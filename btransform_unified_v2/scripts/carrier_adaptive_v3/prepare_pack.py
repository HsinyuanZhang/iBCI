#!/usr/bin/env python3
"""Serialize a frozen adaptive-v3 projection as a carrier-last1-v2 pack.

The v2 schema below is a loader interface only.  The payload explicitly records
that its carrier is the adaptive-v3 statistical projection, never a relabelled
legacy M1/H1 carrier.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORKSPACE=ROOT.parent
for p in (ROOT, ROOT/'scripts', WORKSPACE, WORKSPACE/'btransform_unified_v1/src', HERE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from carrier_adaptive_v3 import panels, statistics


def die(message): raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def typed_sha(value):
    a=np.ascontiguousarray(value); return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def atomic_npz(path, **items):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**items); tmp.replace(path)
def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    return x

def roster(dataset,stage):
    if dataset=='m1':
        sources=('ses-20120924','ses-20120926') if stage=='inner' else ('ses-20120924','ses-20120926','ses-20120927')
        targets=('ses-20120927',) if stage=='inner' else ('ses-20120928',)
    else:
        from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
        dates=('1925-01-01','1925-01-08','1925-01-13','1925-01-15') if stage=='inner' else ('1925-01-01','1925-01-08','1925-01-13','1925-01-15','1925-01-19')
        target_dates=('1925-01-19',) if stage=='inner' else ('1925-01-20',)
        sources=tuple(s for d in dates for s in H1_SESSIONS_BY_DATE[d]); targets=tuple(s for d in target_dates for s in H1_SESSIONS_BY_DATE[d])
    return sources,targets

def read(path):
    if not Path(path).is_file():die(f'missing {path}')
    x=json.loads(Path(path).read_text())
    if not isinstance(x,dict):die(f'JSON object required {path}')
    return x

def verify_gate(path,dataset,stage,candidate_pack_sha):
    gate=read(path)
    if gate.get('schema')!='carrier_last1_v2_paired_audit' or gate.get('status')!='PASSED' or gate.get('phase')!='source' or gate.get('dataset')!=dataset or gate.get('stage')!=stage:die('source gate identity/status mismatch')
    seals=gate.get('source_seals')
    if not isinstance(seals,dict) or not seals:die('source gate has no seals')
    # Verify every SHA which the gate claims, before a target loader can run.
    for arm,row in seals.items():
        if not isinstance(row,dict):die(f'invalid source gate seal {arm}')
        sp=Path(row.get('path','')); seal=read(sp)
        if row.get('sha256')!=sha(sp) or seal.get('schema')!='carrier_last1_v2_source_seal' or seal.get('status')!='SEALED' or seal.get('dataset')!=dataset or seal.get('stage')!=stage:die(f'source seal identity/binding drift {arm}')
        receipt=Path(seal.get('receipt',''))
        if not receipt.is_file() or seal.get('receipt_sha256')!=sha(receipt) or row.get('receipt_sha256')!=sha(receipt):die(f'source receipt binding drift {arm}')
        if seal.get('full_checkpoint_sha256')!=row.get('full_checkpoint_sha256') or not isinstance(row.get('full_checkpoint_sha256'),dict) or not row['full_checkpoint_sha256']:die(f'source checkpoint binding drift {arm}')
        for name,digest in row['full_checkpoint_sha256'].items():
            artifact=sp.parent/name
            if not artifact.is_file() and artifact.suffix=='': artifact=artifact.with_suffix('.pt')
            if not artifact.is_file() or sha(artifact)!=digest:die(f'source checkpoint artifact binding drift {arm}/{name}')
    pack_hashes=[(seal.get('carrier_pack') or {}).get('sha256') for seal in (read(Path(row['path'])) for row in seals.values())]
    if candidate_pack_sha not in pack_hashes:die('source gate does not bind this candidate source pack')
    return gate

def verify_inner_reference_gate(path,dataset,candidate_pack_sha,candidate_pack_meta):
    """Verify the immutable matched-reference proof used by inner gates."""
    gate=read(path)
    if gate.get('schema')!='carrier_last1_v2_paired_audit' or gate.get('status')!='PASSED' or gate.get('phase')!='source' or gate.get('stage')!='inner' or gate.get('dataset')!=dataset or gate.get('historical_inner_reference') is not True:die('inner historical source gate identity mismatch')
    candidate=gate.get('candidate_pack',{})
    if candidate.get('sha256')!=candidate_pack_sha or not Path(candidate.get('path','')).is_file() or sha(Path(candidate['path']))!=candidate_pack_sha:die('inner candidate pack SHA/path drift')
    if candidate.get('metadata',{}).get('dataset')!=dataset or candidate.get('metadata',{}).get('stage')!='inner' or candidate_pack_meta.get('dataset')!=dataset or candidate_pack_meta.get('stage')!='inner':die('inner candidate pack metadata drift')
    bases=gate.get('historical_original_bases'); mapping=gate.get('original_bases_sha256')
    if not isinstance(bases,list) or not bases or not isinstance(mapping,dict):die('inner original-base evidence missing')
    recovered={}
    for item in bases:
        if not isinstance(item,dict):die('inner original-base item invalid')
        base=Path(item.get('path',''))
        if not base.is_file() or item.get('sha256')!=sha(base):die('inner original-base SHA drift')
        recovered[str(base)]=item['sha256']
    if recovered!=mapping:die('inner original-base mapping drift')
    seals=gate.get('source_seals')
    if not isinstance(seals,dict) or set(seals)!={'B','old_D','candidate_D'}:die('inner three-arm source evidence missing')
    for arm,row in seals.items():
        if not isinstance(row,dict):die(f'inner source row invalid {arm}')
        run=Path(row.get('run','')); fixed=row.get('fixed_checkpoints')
        if not run.is_dir() or not isinstance(fixed,dict) or not fixed:die(f'inner run/checkpoint authority missing {arm}')
        for name,digest in fixed.items():
            artifact=run/name
            if not artifact.is_file() and artifact.suffix=='': artifact=artifact.with_suffix('.pt')
            if not artifact.is_file() or sha(artifact)!=digest:die(f'inner checkpoint SHA drift {arm}/{name}')
    return gate

def verify_source_pack(path,dataset,stage,sources,bundle_sha,projection_sha):
    path=Path(path); side=path.with_suffix('.json')
    if not path.is_file() or not side.is_file():die('source pack/sidecar missing')
    if read(side).get('npz_sha256')!=sha(path):die('source pack sidecar SHA drift')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata'].item()))
    expected={'schema':'carrier_last1_v2_pack','dataset':dataset,'stage':stage,'surface':'source','sources':list(sources),'targets':[]}
    if any(meta.get(k)!=v for k,v in expected.items()):die('source pack identity/roster drift')
    if meta.get('adaptive_schema')!='carrier_adaptive_v3_pack_v1' or meta.get('panel_bundle_sha256')!=bundle_sha or meta.get('projection_receipt_sha256')!=projection_sha or meta.get('profile_id')!=('muscle' if dataset=='m1' else 'state'):die('source pack adaptive ancestry drift')
    return meta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=('m1','h1'),required=True); ap.add_argument('--stage',choices=('inner','outer'),required=True); ap.add_argument('--surface',choices=('source','target'),required=True); ap.add_argument('--bundle',type=Path,required=True); ap.add_argument('--projection',type=Path,required=True); ap.add_argument('--dest',type=Path,required=True); ap.add_argument('--source-pack',type=Path); ap.add_argument('--source-gate',type=Path); args=ap.parse_args()
    dest=args.dest.resolve(); side=dest.with_suffix('.json')
    allowed_root=(ROOT/'results'/'carrier_adaptive_v3').resolve()
    if allowed_root not in dest.parents:die(f'destination must be beneath {allowed_root}')
    if dest.exists() or side.exists():raise FileExistsError('refuse overwrite pack or sidecar')
    bundle_path=args.bundle.resolve(); projection_path=args.projection.resolve()
    bundle,bundle_meta=panels.load_bundle(bundle_path); plan=statistics.load_projection(projection_path)
    sources,targets=roster(args.dataset,args.stage)
    if bundle.dataset!=args.dataset or bundle.sessions!=sources:die('bundle dataset/source chronological roster mismatch')
    if plan.metadata.get('schema')!='carrier_adaptive_v3_projection_v1' or tuple(plan.metadata.get('sessions',()))!=sources:die('projection source roster/schema mismatch')
    rawdim=16 if args.dataset=='m1' else 14
    if bundle.behavior_rms.shape!=(16 if args.dataset=='m1' else 7,) or plan.mean.shape!=(rawdim,) or plan.transform.shape!=(rawdim,4):die('adaptive geometry mismatch')
    bundle_sha,projection_sha=sha(bundle_path),sha(projection_path)
    seal=hashlib.sha256(f'{args.stage}|{args.dataset}|{bundle_sha}|{projection_sha}'.encode()).hexdigest()
    profile='muscle' if args.dataset=='m1' else 'state'; candidate='adaptive_'+str(plan.metadata.get('method',''))
    source_pack_sha=source_gate_sha=None
    arrays={}; detail={}
    if args.surface=='source':
        for session in sources:
            raw=np.asarray(bundle.calibration_raw[session],np.float64); carrier=statistics.project(plan,raw)
            if carrier.ndim!=2 or carrier.shape[1:]!=(4,) or not np.isfinite(carrier).all() or (args.dataset=='m1' and carrier.shape!=(64,4)):die(f'invalid source carrier {session}')
            arrays[f'carrier/{session}']=carrier; detail[session]={'raw_typed_sha256':typed_sha(raw),'carrier_typed_sha256':typed_sha(carrier),'support_group':list(bundle.trial_groups[session][0])}
    else:
        if args.source_pack is None or args.source_gate is None:die('target requires --source-pack and --source-gate')
        # These checks intentionally precede all target-support materialization.
        smeta=verify_source_pack(args.source_pack,args.dataset,args.stage,sources,bundle_sha,projection_sha)
        source_pack_sha=sha(args.source_pack)
        if args.stage=='inner': verify_inner_reference_gate(args.source_gate,args.dataset,source_pack_sha,smeta)
        else: verify_gate(args.source_gate,args.dataset,args.stage,source_pack_sha)
        frozen_npz=Path(plan.metadata['arrays']).resolve()
        if not frozen_npz.is_file() or sha(frozen_npz)!=plan.metadata.get('arrays_sha256'):die('projection frozen NPZ SHA drift before target IO')
        if smeta.get('source_seal')!=seal or smeta.get('frozen_fit_sha256')!=plan.metadata.get('arrays_sha256'):die('source pack frozen-plan/source seal drift')
        source_gate_sha=sha(args.source_gate)
        for session in targets:
            if args.dataset=='m1': path=WORKSPACE/'SPINT-main/data/000941/sub-MonkeyL-held-in-calib'/f'sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb'
            else: path=WORKSPACE/'SPINT-main/data/000954'
            raw=panels.deploy_raw(args.dataset,session,path,bundle.behavior_rms,seal); carrier=statistics.project(plan,raw)
            if carrier.ndim!=2 or carrier.shape[1:]!=(4,) or not np.isfinite(carrier).all() or (args.dataset=='m1' and carrier.shape!=(64,4)):die(f'invalid target carrier {session}')
            arrays[f'carrier/{session}']=carrier; detail[session]={'raw_typed_sha256':typed_sha(raw),'carrier_typed_sha256':typed_sha(carrier),'target_support_only':True}
    frozen=Path(plan.metadata['arrays']).resolve()
    if not frozen.is_file() or sha(frozen)!=plan.metadata.get('arrays_sha256'):die('projection frozen NPZ SHA drift')
    # ``frozen_fit`` is the projection array NPZ, while ``projection`` is its immutable receipt.
    meta={'schema':'carrier_last1_v2_pack','adaptive_schema':'carrier_adaptive_v3_pack_v1','adaptive_method':plan.metadata['method'],'serialization_note':'carrier_last1_v2_pack is a compatibility interface; payload is adaptive-v3 statistical projection, not a legacy carrier claim','dataset':args.dataset,'profile_id':profile,'candidate':candidate,'stage':args.stage,'surface':args.surface,'sources':list(sources),'targets':[] if args.surface=='source' else list(targets),'carrier_shape':[64,4] if args.dataset=='m1' else ['variable',4],'source_seal':seal,'frozen_fit':str(frozen),'frozen_fit_sha256':sha(frozen),'projection_receipt':str(projection_path),'projection_receipt_sha256':projection_sha,'panel_bundle':str(bundle_path),'panel_bundle_sha256':bundle_sha,'source_pack_sha256':source_pack_sha,'source_gate_sha256':source_gate_sha,'target_query_labels_used_for_carrier':False,'target_query_labels_used_for_selection':False,'target_record_materialized_for_support_projection':args.dataset=='h1' and args.surface=='target','behavior_rms':np.asarray(bundle.behavior_rms).tolist(),'arrays':detail,'projection_metadata':jsonable(dict(plan.metadata))}
    arrays['behavior_rms']=np.asarray(bundle.behavior_rms,dtype=np.float64); arrays['metadata']=np.array(json.dumps(jsonable(meta),sort_keys=True)); atomic_npz(dest,**arrays); side.write_text(json.dumps(jsonable({**meta,'npz_sha256':sha(dest)}),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
