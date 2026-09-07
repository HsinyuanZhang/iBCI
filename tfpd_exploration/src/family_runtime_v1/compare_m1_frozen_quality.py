"""Hash-gated, archive-only descriptive comparison of frozen M1 references."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

COUNT, OUTPUTS = 31252, 16
SESSIONS, COUNTS = ("ses-20120926","ses-20120927","ses-20120928"), (10567,9705,10980)
GO="M1_FROZEN_QUALITY_COMPARISON_GO"
IMAGE="sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    x=json.loads(Path(path).read_text())
    if not isinstance(x,dict): raise RuntimeError("receipt must be JSON object")
    return x
def _hash(x,label):
    if not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x): raise RuntimeError(label+" SHA-256 missing/drift")
    return x
def atomic(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as h:
        t=Path(h.name);json.dump(value,h,sort_keys=True,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
    os.replace(t,path)
def bind(snapshot,path,expected=None):
    if not isinstance(path,(str,Path)): raise RuntimeError("bound input path missing")
    path=Path(path)
    if not path.is_file(): raise RuntimeError("bound input missing: "+str(path))
    got=sha(path)
    if expected is not None and got!=_hash(expected,"recorded"): raise RuntimeError("bound input SHA drift: "+str(path))
    if snapshot.setdefault(str(path),got)!=got: raise RuntimeError("conflicting input binding")
    return path
def bind_files(snapshot,files,label):
    if not isinstance(files,dict) or not files: raise RuntimeError(label+" files closure missing")
    for name,digest in files.items():
        bind(snapshot,name,_hash(digest,label+" closure"))

def image_evidence(snapshot,value):
    if not isinstance(value,dict) or not value: raise RuntimeError('nonempty image closure required')
    for path,digest in value.items():
        digest=_hash(digest,'image evidence'); candidate=Path(path)
        # The completed receipt can name both host helpers and frozen-image
        # namespaces.  Only an absolute path inside this repository is host
        # evidence that this archive-only process can honestly re-read.
        if candidate.is_absolute() and str(candidate).startswith(str(Path(__file__).resolve().parents[2])):
            bind(snapshot,candidate,digest)

def original_archive_host(recorded, receipt):
    """Permit only the scorer's recorded container results mount translation."""
    recorded=Path(recorded); host=receipt.parent/recorded.name
    expected=Path('/results')/receipt.parent.name
    if recorded.parent == expected: return host, {'recorded_path':str(recorded),'host_path':str(host),'rule':'/results/<receipt-parent-name>/<archive-name> -> receipt.parent/<archive-name>'}
    return recorded, None

def metric(a):
    p,y,s=a['prediction'].astype(np.float64),a['target'].astype(np.float64),a['session']
    def r(x,z):
        d=float(np.square(z-z.mean(0,keepdims=True)).sum(dtype=np.float64))
        if not np.isfinite(d) or d<=0: raise RuntimeError('nonpositive target variance')
        return float(1-float(np.square(x-z).sum(dtype=np.float64))/d)
    per={name:r(p[s==name],y[s==name]) for name in SESSIONS}; worst=min(per,key=per.get)
    return {'n_bins':int(len(p)),'pooled_r2_float64':r(p,y),'equal_session_mean_r2_float64':float(np.mean(list(per.values()))),'worst_session_r2_float64':per[worst],'worst_session_id':worst,'per_session_r2_float64':per}
def load(path):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if set(a)!={'prediction','target','session','start'} or a['prediction'].shape!=(COUNT,OUTPUTS) or a['target'].shape!=(COUNT,OUTPUTS) or a['prediction'].dtype!=np.float32 or a['target'].dtype!=np.float32 or a['session'].shape!=(COUNT,) or a['session'].dtype.kind!='U' or a['start'].shape!=(COUNT,) or a['start'].dtype!=np.int64 or not np.isfinite(a['prediction']).all() or not np.isfinite(a['target']).all(): raise RuntimeError('FP32 archive geometry/dtype/finite drift')
    offset=0
    for name,count in zip(SESSIONS,COUNTS,strict=True):
        ids=np.flatnonzero(a['session']==name)
        if len(ids)!=count or not np.array_equal(ids,np.arange(offset,offset+count)) or np.any(a['start'][ids]<0) or np.any(np.diff(a['start'][ids])<=0): raise RuntimeError('sorted three-session count/start topology drift')
        offset+=count
    if offset!=COUNT: raise RuntimeError('31252 archive cardinality drift')
    return a
def same(actual,reported,label):
    if not isinstance(reported,dict): raise RuntimeError(label+' metric receipt missing')
    mapping={'equal_session_mean_r2':'equal_session_mean_r2_float64','pooled_r2':'pooled_r2_float64','per_session_r2':'per_session_r2_float64'}
    for old,new in mapping.items():
        value=reported.get(old)
        if old=='per_session_r2':
            if not isinstance(value,dict) or set(value)!=set(SESSIONS): raise RuntimeError(label+' per-session topology drift')
            for key in SESSIONS:
                if not isinstance(value[key],(float,int)) or not np.isfinite(value[key]) or abs(float(value[key])-actual[new][key])>1e-12: raise RuntimeError(label+' per-session metric drift')
        elif not isinstance(value,(float,int)) or not np.isfinite(value) or abs(float(value)-actual[new])>1e-12: raise RuntimeError(label+' metric drift: '+old)
def delta(name,a,b):
    return {'candidate':name,'pooled_delta':a['pooled_r2_float64']-b['pooled_r2_float64'],'equal_session_mean_delta':a['equal_session_mean_r2_float64']-b['equal_session_mean_r2_float64'],'worst_session_delta':a['worst_session_r2_float64']-b['worst_session_r2_float64'],'candidate_worst_session_id':a['worst_session_id'],'original_worst_session_id':b['worst_session_id'],'per_session_delta':{s:a['per_session_r2_float64'][s]-b['per_session_r2_float64'][s] for s in SESSIONS}}

def run(args):
    p1,original,out,proof=map(Path,(args.p1_receipt,args.original_receipt,args.output,args.source_proof))
    if out.exists(): raise FileExistsError(out)
    if os.environ.get(GO)!='1' or not out.is_absolute() or not p1.is_absolute() or not original.is_absolute() or not proof.is_absolute(): raise RuntimeError('explicit GO and absolute input/output paths required')
    snap={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    p1sha,originalsha,proofsha=(_hash(getattr(args,name,None),name) for name in ('p1_receipt_sha256','original_receipt_sha256','source_proof_sha256'))
    bind(snap,p1,p1sha);bind(snap,original,originalsha);bind(snap,proof,proofsha)
    p1body,obody,proofbody=read(p1),read(original),read(proof)
    if p1body.get('schema')!='m1_family_v1_p1_finalizer_v1' or p1body.get('status')!='SOURCE_MINIVAL_ONLY' or p1body.get('outer_query_opened') is not False: raise RuntimeError('P1 finalized receipt status/schema drift')
    if proofbody.get('schema')!='m1_p1_family_selected_complete_source_stream_v1' or proofbody.get('status')!='PASS_IMPLEMENTATION_EQUIVALENCE_ONLY' or proofbody.get('outer_query_opened') is not False or proofbody.get('parameter_updates')!=0 or proofbody.get('pre_artifact')!=proofbody.get('post_artifact') or proofbody.get('pre_source')!=proofbody.get('post_source') or proofbody.get('pre_artifact',{}).get('final')!=p1body or proofbody.get('pre_artifact',{}).get('files',{}).get(str(p1))!=p1sha or set(proofbody.get('arms',{}))!={'flat','route'} or any(proofbody['arms'][a].get('scored_count')!=COUNT or proofbody['arms'][a].get('public_calls')!=112985 or proofbody['arms'][a].get('initial_current_predictions')!=3 or not np.isfinite(proofbody['arms'][a].get('max_abs_error',np.nan)) or proofbody['arms'][a]['max_abs_error']>1e-5 or proofbody['arms'][a].get('selected')!=proofbody['pre_artifact'].get('selected',{}).get(a) for a in ('flat','route')): raise RuntimeError('complete source runtime proof drift')
    for closure in (proofbody['pre_artifact'],proofbody['pre_source']): bind_files(snap,closure.get('files'),'proof')
    if (obody.get('schema')!='original_m1_as_shipped_frozen_selected_source_dev_v1' or obody.get('status')!='PASS_AS_SHIPPED_ORIGINAL_M1_REFERENCE_ONLY' or obody.get('scored_count')!=COUNT or obody.get('public_calls')!=112985 or obody.get('initial_native_predictions')!=3 or obody.get('outer_query_opened') is not False or obody.get('image')!=IMAGE or obody.get('window')!=100 or obody.get('units')!=64 or obody.get('outputs')!=16 or obody.get('batch')!=1 or obody.get('parameter_updates')!=0 or obody.get('no_selection_or_fit') is not True or obody.get('full_selected_proof_sha256')!=proofsha or obody.get('pre_artifact')!=obody.get('post_artifact') or obody.get('pre_source')!=obody.get('post_source') or obody.get('pre_artifact')!=proofbody.get('pre_artifact') or obody.get('pre_source')!=proofbody.get('pre_source') or obody.get('image_pre')!=obody.get('image_post')): raise RuntimeError('original M1 receipt completion/status/count drift')
    image_evidence(snap,obody.get('image_pre'))
    for closure in (obody['pre_artifact'],obody['pre_source']): bind_files(snap,closure.get('files'),'original')
    recorded_archive=obody.get('archive',{}); archive_hash=_hash(recorded_archive.get('sha256'),'original archive'); original_path,translation=original_archive_host(recorded_archive.get('path'),original); original_npz=bind(snap,original_path,archive_hash)
    exports=p1body.get('exports');
    if not isinstance(exports,dict) or set(exports)!={'flat_selected','route_selected','flat_endpoint24','route_endpoint24'}: raise RuntimeError('P1 export topology drift')
    paths={}
    for key,item in exports.items():
        if not isinstance(item,dict): raise RuntimeError('P1 export receipt topology drift')
        paths[key]=bind(snap,p1.parent/(key+'_native_source_dev.npz'),_hash(item.get('prediction_sha256'),'P1 archive'))
    arrays={'ORIGINAL_as_shipped':load(original_npz)}
    for key,path in paths.items(): arrays[key.upper()]=load(path)
    reference=arrays['ORIGINAL_as_shipped']
    for name,a in arrays.items():
        if name!='ORIGINAL_as_shipped' and any(not np.array_equal(a[k],reference[k]) for k in ('target','session','start')): raise RuntimeError('all five archives must share target/session/start identity')
    tables={'ORIGINAL_as_shipped':metric(reference)}; same(tables['ORIGINAL_as_shipped'],obody.get('metrics_float64'),'original')
    for key,item in exports.items():
        name=key.upper(); tables[name]=metric(arrays[name]); same(tables[name],item.get('metrics'),'P1 '+key)
    if {path:sha(path) for path in snap}!=snap: raise RuntimeError('fresh post input mutation')
    result={'schema':'m1_frozen_same31252_descriptive_quality_table_v1','status':'PASS_ARCHIVE_ONLY_DESCRIPTIVE','scope':'existing hash-bound FP32 archives only; no source/model/scorer import, fitting, selection, or promotion','tables':tables,'deltas_vs_original':{k:delta(k,v,tables['ORIGINAL_as_shipped']) for k,v in tables.items() if k!='ORIGINAL_as_shipped'},'named_baselines':p1body.get('historical_same_surface_only'),'original_image_f5':'as-shipped original-M1 image payload is distinct from the local source-fold teacher','comparability':'historical training/calibration exposure is unequal; reference-only descriptive comparison','noninferiority':'no formal noninferiority claim','inputs_sha256_pre':snap,'inputs_sha256_post':{path:sha(path) for path in snap},'recorded_container_archive_translation':translation,'image_evidence':'original receipt image_pre equals image_post; image /src,payload,and wrapper hashes are immutable recorded evidence not re-read on this host'}
    atomic(result,out);return result
def main(argv=None):
    q=argparse.ArgumentParser();q.add_argument('--p1-receipt',type=Path,required=True);q.add_argument('--p1-receipt-sha256',required=True);q.add_argument('--original-receipt',type=Path,required=True);q.add_argument('--original-receipt-sha256',required=True);q.add_argument('--source-proof',type=Path,required=True);q.add_argument('--source-proof-sha256',required=True);q.add_argument('--output',type=Path,required=True);return run(q.parse_args(argv))
if __name__=='__main__':main()
