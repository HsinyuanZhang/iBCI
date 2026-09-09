#!/usr/bin/env python3
"""Immutable source-authority reuse for exact paired v3 carrier experiments."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ah(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def read(p):
 if not p.is_file():raise RuntimeError(f'missing {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict):raise RuntimeError(f'object required {p}')
 return x
def fresh(p):
 if p.exists():raise FileExistsError(p)
def pack(path,dataset,stage):
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(str(z['metadata'].item()))
  if not isinstance(meta,dict) or meta.get('dataset')!=dataset or meta.get('stage')!=stage or meta.get('surface')!='source' or meta.get('targets')!=[]:raise RuntimeError('source pack metadata drift')
  out={s:np.ascontiguousarray(z[f'carrier/{s}'],np.float32) for s in meta.get('sources',[])}
 if not out or any(x.ndim!=2 or x.shape[1]!=4 or not np.isfinite(x).all() for x in out.values()):raise RuntimeError('invalid 4D source pack')
 return meta,out
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
def m1(a,meta):
 ref=a.reference.resolve();dest=a.dest.resolve();fresh(dest);dest.mkdir(parents=True)
 files=sorted(ref.glob('rsyn3_*.npz'))
 if not files:raise RuntimeError('M1 reference needs rsyn3_*.npz cache')
 rows=[]
 for src in files:
  side=src.with_suffix('.json')
  if not side.is_file():raise RuntimeError(f'missing cache sidecar {side}')
  sidev=read(side)
  if sidev.get('split_id') != ('m1_inner_last1_20120927' if a.stage=='inner' else 'm1_outer_last1_20120928') or sidev.get('sources')!=meta['sources']:raise RuntimeError('M1 cache source roster drift')
  with np.load(src,allow_pickle=False) as z:
   if 'sources' not in z.files or str(z['sources'].item())!='|'.join(meta['sources']):raise RuntimeError('M1 cache NPZ source roster drift')
  for q in (src,side):
   out=dest/q.name;shutil.copyfile(q,out);rows.append({'reference':str(q),'reference_sha256':sha(q),'output':str(out),'output_sha256':sha(out),'match':sha(q)==sha(out)})
 if not all(r['match'] for r in rows):raise RuntimeError('M1 cache copy SHA drift')
 return {'mode':'m1_cache_copy','shared_files':rows,'carrier_pack_sha256':sha(a.pack)}
def h1(a,meta,carriers):
 ref=a.reference.resolve()/'source';destroot=a.dest.resolve();dest=destroot/'source';fresh(destroot);dest.mkdir(parents=True)
 man=read(ref/'manifest.json');records=man.get('records',{})
 if man.get('surface')!='source' or sorted(records)!=sorted(meta['sources']):raise RuntimeError('H1 reference manifest/source roster drift')
 # Only authority files and source train/validation NPZ are copied. Selection seals
 # and any target directory are intentionally excluded.
 copied=[]
 for name in ('source_hc_plan.json','source_hc_plan_arrays.npz'):
  q=ref/name
  if not q.is_file():raise RuntimeError(f'missing H1 authority {q}')
  out=dest/name;shutil.copyfile(q,out);require_same=sha(q)==sha(out)
  if not require_same:raise RuntimeError(f'H1 authority copy SHA drift {name}')
  copied.append({'reference':str(q),'reference_sha256':sha(q),'output':str(out),'output_sha256':sha(out)})
 newman=json.loads(json.dumps(man))
 checks=[]
 for session in meta['sources']:
  if session not in carriers:raise RuntimeError(f'pack missing H1 source {session}')
  for suffix,key in (('.npz',None),('.val.npz','validation')):
   src=ref/f'{session}{suffix}';out=dest/f'{session}{suffix}'
   if not src.is_file():raise RuntimeError(f'missing reference source array {src}')
   with np.load(src,allow_pickle=False) as z:
    payload={k:np.array(z[k],copy=True,order='C') for k in z.files}
   if 'carrier' not in payload:raise RuntimeError(f'reference lacks carrier {src}')
   old=payload['carrier'];new=np.ascontiguousarray(carriers[session],np.float32)
   if old.shape!=new.shape:raise RuntimeError(f'carrier shape drift {session}: {old.shape} vs {new.shape}')
   payload['carrier']=new;np.savez_compressed(out,**payload)
   comparisons=[]
   with np.load(src,allow_pickle=False) as original, np.load(out,allow_pickle=False) as z2:
    if set(original.files)!=set(z2.files):raise RuntimeError(f'NPZ keyset rewrite drift {out}')
    for k in original.files:
     if k=='carrier':continue
     before,after=original[k],z2[k]
     equal=same(before,after)
     comparisons.append({'key':k,'dtype':before.dtype.str,'shape':list(before.shape),'reference_typed_sha256':ah(before),'output_typed_sha256':ah(after),'equal':equal})
     if not equal:raise RuntimeError(f'noncarrier rewrite drift {out}:{k}')
   row=newman['records'][session] if key is None else newman['records'][session].get(key)
   if not isinstance(row,dict):raise RuntimeError(f'manifest validation row missing {session}')
   row['carrier_sha256']=ah(new)
   checks.append({'reference':str(src),'reference_sha256':sha(src),'output':str(out),'output_sha256':sha(out),'session':session,'surface':'train' if key is None else 'validation','carrier_reference_typed_sha256':ah(old),'carrier_output_typed_sha256':ah(new),'noncarrier_arrays':comparisons})
 (dest/'manifest.json').write_text(json.dumps(newman,indent=2,sort_keys=True)+'\n')
 # source_authority and source_authority_files are intentionally byte-for-byte
 # manifest values from the reference.  Their referenced plan files were copied.
 if newman['source_authority']!=man['source_authority'] or newman['source_authority_files']!=man['source_authority_files']:raise RuntimeError('authority changed during reuse')
 return {'mode':'h1_source_copy_carrier_replace','shared_authority_files':copied,'source_manifest_reference':str(ref/'manifest.json'),'source_manifest_reference_sha256':sha(ref/'manifest.json'),'source_manifest_output':str(dest/'manifest.json'),'source_manifest_output_sha256':sha(dest/'manifest.json'),'source_authority_preserved':True,'source_authority_files_preserved':True,'carrier_pack_sha256':sha(a.pack),'arrays':checks}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('m1','h1'),required=True);p.add_argument('--stage',choices=('inner','outer'),required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--pack',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();a.pack=a.pack.resolve();meta,carriers=pack(a.pack,a.dataset,a.stage)
 root=Path(__file__).resolve().parents[2]/'results/carrier_adaptive_v3';dest=a.dest.resolve()
 if root not in (dest,*dest.parents):raise RuntimeError(f'dest must be below {root}')
 body=m1(a,meta) if a.dataset=='m1' else h1(a,meta,carriers)
 receipt={'schema':'carrier_adaptive_v3_source_data_reuse_v1','status':'REUSED_SOURCE_ONLY','dataset':a.dataset,'stage':a.stage,'reference':str(a.reference.resolve()),'reference_sha256':None if not a.reference.is_file() else sha(a.reference),'pack':str(a.pack),'pack_sha256':sha(a.pack),'pack_metadata':meta,**body}
 out=(dest/'source_data_reuse_check.json');out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
