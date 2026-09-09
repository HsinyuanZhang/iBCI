#!/usr/bin/env python3
"""Write an immutable prose-only erratum for one outer H1 prepared authority."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_DATES=['1925-01-01','1925-01-08','1925-01-13','1925-01-15','1925-01-19']
TARGET_DATES=['1925-01-20']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):
 if not p.is_file():raise RuntimeError(f'missing {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict):raise RuntimeError(f'object required {p}')
 return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);a=p.parse_args();root=a.prepared.resolve();out=root/'outer_authority_clarification.json'
 if out.exists():raise FileExistsError(out)
 manp=root/'source/manifest.json';planp=root/'source/source_hc_plan.json';arrp=root/'source/source_hc_plan_arrays.npz';man=read(manp);authority=man.get('source_authority',{})
 if man.get('surface')!='source' or man.get('source_dates')!=SOURCE_DATES or man.get('target_dates')!=TARGET_DATES or man.get('source_count')!=11 or man.get('target_count')!=2 or len(man.get('source_sessions',[]))!=11 or len(man.get('target_sessions',[]))!=2 or authority.get('target_records_opened')!=0:raise RuntimeError('not an outer H1 source-only authority')
 if not all('19250119' in s for s in man['source_sessions'][-2:]) or not all('19250120' in s for s in man['target_sessions']):raise RuntimeError('outer H1 roster/date drift')
 for q in (planp,arrp):
  if not q.is_file():raise RuntimeError(f'missing source authority artifact {q}')
 payload={'schema':'carrier_last1_v2_outer_h1_authority_erratum_v1','status':'SEALED','stage':'outer','dataset':'h1','prepared_root':str(root),'source_manifest':str(manp),'source_manifest_sha256':sha(manp),'source_hc_plan':str(planp),'source_hc_plan_sha256':sha(planp),'source_hc_plan_arrays':str(arrp),'source_hc_plan_arrays_sha256':sha(arrp),'adapter':str(Path(__file__).resolve()),'adapter_sha256':sha(Path(__file__)),'source_dates':SOURCE_DATES,'target_dates':TARGET_DATES,'source_sessions':man['source_sessions'],'target_sessions':man['target_sessions'],'erratum':'The frozen preparer prose saying nine sources and no 1925-01-19/20 records is inner-only text. Outer source authority opens its 11 listed source sessions through 1925-01-19 and opens no 1925-01-20 target records.','historical_text_correction_only':True,'algorithm_changed':False,'source_arrays_rewritten':False}
 out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
