from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from tfpd_exploration.src.family_runtime_v1 import compare_m1_frozen_quality as c

def put(x,p): p.write_text(json.dumps(x,sort_keys=True));return p
def npz(p,a): np.savez_compressed(p,**a);return c.sha(p)
def arrays(offset=0):
    session=np.concatenate([np.full(n,s) for s,n in zip(c.SESSIONS,c.COUNTS)])
    start=np.concatenate([np.arange(n,dtype=np.int64) for n in c.COUNTS]); target=np.arange(c.COUNT*c.OUTPUTS,dtype=np.float32).reshape(c.COUNT,c.OUTPUTS)/13
    return {'prediction':target+np.float32(offset),'target':target,'session':session,'start':start}
def m(a):
    x=c.metric(a);return {'equal_session_mean_r2':x['equal_session_mean_r2_float64'],'pooled_r2':x['pooled_r2_float64'],'per_session_r2':x['per_session_r2_float64']}
def fixture(tmp_path):
    root=tmp_path/'p1';root.mkdir(parents=True); closure=tmp_path/'closure';closure.write_text('frozen')
    exports={}; all_arrays={}
    for key,off in zip(('flat_selected','route_selected','flat_endpoint24','route_endpoint24'),(.1,.2,.3,.4)):
        a=arrays(off); all_arrays[key]=a; exports[key]={'prediction_sha256':npz(root/(key+'_native_source_dev.npz'),a),'metrics':m(a)}
    p1={'schema':'m1_family_v1_p1_finalizer_v1','status':'SOURCE_MINIVAL_ONLY','outer_query_opened':False,'exports':exports,'historical_same_surface_only':{'references':{'oldteacher':{'n':c.COUNT}}}}
    p1path=put(p1,root/'receipt.json')
    source={'files':{str(closure):c.sha(closure)}}; artifact={'files':{str(closure):c.sha(closure)}}
    selected={a:{'checkpoint_sha256':a*64} for a in ('flat','route')}; artifact={'files':{str(closure):c.sha(closure),str(p1path):c.sha(p1path)},'final':p1,'selected':selected}
    proof={'schema':'m1_p1_family_selected_complete_source_stream_v1','status':'PASS_IMPLEMENTATION_EQUIVALENCE_ONLY','outer_query_opened':False,'parameter_updates':0,'pre_artifact':artifact,'post_artifact':artifact,'pre_source':source,'post_source':source,'arms':{a:{'scored_count':c.COUNT,'public_calls':112985,'initial_current_predictions':3,'max_abs_error':1e-6,'selected':selected[a]} for a in ('flat','route')}}
    proofpath=put(proof,tmp_path/'proof.json')
    original_a=arrays(.5); original_npz=tmp_path/'original.npz'; original_sha=npz(original_npz,original_a)
    original={'schema':'original_m1_as_shipped_frozen_selected_source_dev_v1','status':'PASS_AS_SHIPPED_ORIGINAL_M1_REFERENCE_ONLY','scored_count':c.COUNT,'public_calls':112985,'initial_native_predictions':3,'outer_query_opened':False,'image':c.IMAGE,'window':100,'units':64,'outputs':16,'batch':1,'parameter_updates':0,'no_selection_or_fit':True,'full_selected_proof_sha256':c.sha(proofpath),'pre_artifact':artifact,'post_artifact':artifact,'pre_source':source,'post_source':source,'image_pre':{'/src/image.py':'a'*64},'image_post':{'/src/image.py':'a'*64},'archive':{'path':str(original_npz),'sha256':original_sha},'metrics_float64':m(original_a)}
    originalpath=put(original,tmp_path/'original.json')
    return p1path,originalpath,proofpath
def args(tmp_path,paths):
    return type('Args',(),{'p1_receipt':paths[0].resolve(),'p1_receipt_sha256':c.sha(paths[0]),'original_receipt':paths[1].resolve(),'original_receipt_sha256':c.sha(paths[1]),'source_proof':paths[2].resolve(),'source_proof_sha256':c.sha(paths[2]),'output':(tmp_path/'out.json').resolve()})()
def test_complete_archive_only_comparison(tmp_path,monkeypatch):
    monkeypatch.setenv(c.GO,'1'); r=c.run(args(tmp_path,fixture(tmp_path)))
    assert r['status']=='PASS_ARCHIVE_ONLY_DESCRIPTIVE' and len(r['tables'])==5 and r['inputs_sha256_pre']==r['inputs_sha256_post']
@pytest.mark.parametrize('kind,match',[('dtype','complete source runtime proof'),('metadata','complete source runtime proof'),('proof','complete source runtime proof')])
def test_rejects_archive_metadata_and_proof_drift(tmp_path,monkeypatch,kind,match):
    monkeypatch.setenv(c.GO,'1'); paths=fixture(tmp_path)
    if kind=='dtype':
        p=paths[0].parent/'flat_selected_native_source_dev.npz'; a=arrays(.1);a['prediction']=a['prediction'].astype(np.float64);np.savez_compressed(p,**a); body=json.loads(paths[0].read_text());body['exports']['flat_selected']['prediction_sha256']=c.sha(p);put(body,paths[0])
    elif kind=='metadata':
        p=paths[0].parent/'route_selected_native_source_dev.npz';a=arrays(.2);a['target'][0,0]+=1;np.savez_compressed(p,**a);body=json.loads(paths[0].read_text());body['exports']['route_selected']['prediction_sha256']=c.sha(p);body['exports']['route_selected']['metrics']=m(a);put(body,paths[0])
    else:
        body=json.loads(paths[2].read_text());body['arms']['flat']['public_calls']=1;put(body,paths[2])
    with pytest.raises(RuntimeError,match=match):c.run(args(tmp_path,paths))
def test_gate_precedes_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'sha',lambda p:pytest.fail('read before gate'))
    p=(tmp_path/'x').resolve()
    with pytest.raises(RuntimeError,match='explicit GO'):c.run(type('A',(),{'p1_receipt':p,'p1_receipt_sha256':'a'*64,'original_receipt':p,'original_receipt_sha256':'a'*64,'source_proof':p,'source_proof_sha256':'a'*64,'output':(tmp_path/'o').resolve()})())

def test_requires_explicit_hashes_and_rejects_crossauthority_image_and_post_mutation(tmp_path,monkeypatch):
    monkeypatch.setenv(c.GO,'1'); paths=fixture(tmp_path); value=args(tmp_path,paths)
    value.p1_receipt_sha256=None
    with pytest.raises(RuntimeError,match='p1_receipt_sha256'):c.run(value)
    paths=fixture(tmp_path/'cross'); value=args(tmp_path/'cross',paths); body=json.loads(paths[1].read_text());body['pre_source']={'files':{}};body['post_source']=body['pre_source'];put(body,paths[1])
    with pytest.raises(RuntimeError,match='original M1 receipt'):c.run(args(tmp_path/'cross',paths))
    paths=fixture(tmp_path/'image'); body=json.loads(paths[1].read_text());body['image_post']={};put(body,paths[1])
    with pytest.raises(RuntimeError,match='original M1 receipt'):c.run(args(tmp_path/'image',paths))
    paths=fixture(tmp_path/'post'); original_metric=c.metric; fired=False
    def mutate(a):
        nonlocal fired
        value=original_metric(a)
        if not fired: paths[0].write_text(paths[0].read_text()+" "); fired=True
        return value
    monkeypatch.setattr(c,'metric',mutate)
    with pytest.raises(RuntimeError,match='fresh post input mutation'):c.run(args(tmp_path/'post',paths))
