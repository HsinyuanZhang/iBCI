"""Shared, score-free H1 HO-M3 carrier-reliance surface contract."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

PERMUTATION_SEEDS=(101,102,103)

def _sha(values: list[tuple[str,np.ndarray]]) -> str:
    h=hashlib.sha256()
    for key,a in values: h.update(key.encode()); h.update(np.asarray(a,dtype=np.int64).tobytes())
    return h.hexdigest()

def array_sha256(a: np.ndarray) -> str:
    a=np.ascontiguousarray(a); return hashlib.sha256(a.view(np.uint8)).hexdigest()

def array_identity(a: np.ndarray) -> dict[str,Any]:
    """A JSON-safe, byte-exact identity for a materialized source array."""
    a=np.asarray(a)
    return {"sha256":array_sha256(a),"shape":list(a.shape),"dtype":str(a.dtype),"nbytes":int(a.nbytes)}

def load_surface(*, max_endpoints_per_group: int=2048) -> dict[str,Any]:
    """Load raw HO records/M3 banks and freeze a group-wide endpoint inventory.

    This deliberately does not call ``endpoint_context`` or create [N,L,U].
    Consumers construct only their selected batch windows from ``neural`` and
    ``selected_endpoints``.
    """
    import sys
    root=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/'scripts'/'rift_v1'))
    import h1_train as train
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    import h1_c2_cal1_b2_l200_p16 as b2
    from btransform_unified_v1 import adapters
    records=[]
    for session,key in train.HELDOUT_SESSION_TO_FALCON_KEY:
        path=b2.HO_DIR/f"sub-HumanPitt-held-out-calib_{session}.nwb"
        neural,velocity,_change,mask=load_nwb(path,FalconTask.h1)
        ends=np.flatnonzero(np.asarray(mask,bool)).astype(np.int64)
        activity,carrier=adapters._h1_payload_arrays(session); e0,hc=b2._materialize_e0(activity,carrier)
        records.append({"session":session,"key":key,"group":key.split('_set_')[0],"neural":np.asarray(neural,np.float32),"targets":np.asarray(velocity,np.float32),"eval_mask":np.asarray(mask,bool),"endpoints":ends,"E0":e0,"carrier":hc,"unit_mask":np.ones(e0.shape[0],bool)})
    inventories=[]
    for group in sorted({r['group'] for r in records}):
        members=[r for r in records if r['group']==group]; total=sum(len(r['endpoints']) for r in members)
        choose=np.unique(np.linspace(0,total-1,min(max_endpoints_per_group,total),dtype=np.int64))
        cursor=0
        for r in members:
            local=choose[(choose>=cursor)&(choose<cursor+len(r['endpoints']))]-cursor
            r['selected_endpoints']=r['endpoints'][local]; cursor+=len(r['endpoints'])
        inventories.extend((r['key'],r['selected_endpoints']) for r in members)
    perms={}
    for group in sorted({r['group'] for r in records}):
        members=[r for r in records if r['group']==group]
        valid=np.flatnonzero(members[0]['unit_mask']).astype(np.int64)
        if any(not np.array_equal(valid,np.flatnonzero(r['unit_mask'])) for r in members[1:]):
            raise RuntimeError(f"{group} does not have a shared direct-carrier unit mask")
        perms[group]={str(seed):np.random.default_rng(seed).permutation(valid).tolist() for seed in PERMUTATION_SEEDS}
    return {"records":records,"groups":sorted({r['group'] for r in records}),"max_endpoints_per_group":max_endpoints_per_group,"query_inventory_sha256":_sha(inventories),"full_valid_inventory_sha256":_sha([(r['key'],r['endpoints']) for r in records]),"permutation_seeds":PERMUTATION_SEEDS,"permutations":perms,"carrier_zero_semantics":"H-C arrays are the C2 materializer output; zero is normalized-space zero. Payload derivation is C2 M3 SHA-pinned but contains no explicit centering field; raw-space mean/zero interpretation remains UNSPECIFIED, not source mean.","payload_derivation":adapters._h1_payload()['derivation']}

def clone_bank_for_arm(record:dict[str,Any], arm:str, seed:int|None=None)->dict[str,Any]:
    """Return copied bank arrays; only direct H-C carrier rows are changed."""
    e0=np.array(record['E0'],copy=True); c=np.array(record['carrier'],copy=True); mask=np.array(record['unit_mask'],copy=True)
    valid=np.flatnonzero(mask)
    if arm=='normal': pass
    elif arm=='zero': c[valid]=0
    elif arm=='shuffle':
        if seed not in PERMUTATION_SEEDS: raise ValueError('shuffle seed must be 101/102/103')
        c[valid]=c[np.random.default_rng(seed).permutation(valid)]
    else: raise ValueError('arm must be normal, zero, or shuffle')
    return {'E0':e0,'carrier':c,'unit_mask':mask}

def write_protocol_artifacts(surface:dict[str,Any], dest:Path) -> dict[str,Path]:
    """Freeze the observed public load surface without evaluating a decoder.

    The produced JSON is intentionally sufficient to reproduce the input
    selection and interventions, but has no predictions, scores, or test data.
    """
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in surface['records']:
        arrays={name:array_identity(row[name]) for name in
                ('neural','targets','eval_mask','endpoints','selected_endpoints','E0','carrier','unit_mask')}
        records.append({'session':row['session'],'key':row['key'],'group':row['group'],
                        'arrays':arrays,'selected_endpoint_count':int(len(row['selected_endpoints'])),
                        'valid_endpoint_count':int(len(row['endpoints']))})
    permutation_rows={}
    for group,by_seed in surface['permutations'].items():
        valid=np.asarray(by_seed[str(PERMUTATION_SEEDS[0])],np.int64)
        permutation_rows[group]={
            'unit_indices_domain': {'definition':'global direct-carrier row indices; each mapping lists source rows in destination-row order',
                                    'count':int(len(valid))},
            'seeds':{seed:array_identity(np.asarray(mapping,np.int64)) for seed,mapping in by_seed.items()},
        }
    manifest={'schema':'h1_carrier_reliance_load_surface_v1','status':'SCORE_FREE_PROTOCOL_FROZEN',
              'scope':'H1 C2 HO-M3 public held-out-calib only','official_test_opened':False,
              'score_free':True,'no_scores_yet':True,
              'checkpoint_arms':{'rift':'R300 EMA22','bt_eort':'H1 B2 EMA18 L200'},
              'intervention':{'carrier':'H-C direct only; E0 remains fixed and may contain H-C',
                              'arms':['normal','zero','shuffle_101','shuffle_102','shuffle_103'],
                              'zero_semantics':surface['carrier_zero_semantics']},
              'endpoint_selection':{'method':'uniform linspace on concatenated valid endpoints within each group',
                                    'max_per_group':surface['max_endpoints_per_group'],
                                    'selected_inventory_sha256':surface['query_inventory_sha256'],
                                    'full_valid_inventory_sha256':surface['full_valid_inventory_sha256']},
              'groups':surface['groups'],'record_count':len(records),'records':records,
              'permutation_seeds':list(surface['permutation_seeds']),'permutations':permutation_rows,
              'payload_derivation':surface['payload_derivation']}
    provenance={'schema':'h1_carrier_reliance_provenance_v1','load_surface':'common.load_surface(max_endpoints_per_group=2048)',
                'manifest_file':'manifest.json','query_inventory_sha256':surface['query_inventory_sha256'],
                'full_valid_inventory_sha256':surface['full_valid_inventory_sha256'],
                'public_data_only':True,'official_test_opened':False,'score_free':True,
                'normalizer_interpretation':'The payload provides no explicit centering field. Numeric carrier zero is only normalized-space zero; raw-space mean/zero is UNSPECIFIED.',
                'payload_derivation':surface['payload_derivation']}
    paths={'manifest':dest/'manifest.json','provenance':dest/'provenance.json'}
    paths['manifest'].write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    paths['provenance'].write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    return paths

def grouped_bootstrap(group_values: dict[str,float], *, draws: int=2000, seed: int=20260907) -> dict[str,float]:
    """Frozen paired bootstrap: resample the seven H1 groups with replacement."""
    keys=sorted(group_values)
    if len(keys)!=7: raise ValueError('H1 bootstrap requires exactly seven S6-S12 groups')
    values=np.asarray([group_values[k] for k in keys],np.float64); rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,7,size=(draws,7))].mean(1)
    return {'equal_group_mean':float(values.mean()),'bootstrap_p025':float(np.quantile(samples,.025)),'bootstrap_p975':float(np.quantile(samples,.975)),'draws':draws,'seed':seed}
