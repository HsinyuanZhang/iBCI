#!/usr/bin/env python3
"""Independent real-cache padding and stream semantics verification for DANDI 688 RIFT."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]
SPEC=importlib.util.spec_from_file_location('dandi688_rift_runner',HERE.with_name('dandi688_train.py'))
runner=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(runner)

def atomic_json(path:Path,row:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def digest(value:np.ndarray)->str:
    a=np.ascontiguousarray(value); return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--dest',type=Path,required=True); p.add_argument('--cpu-threads',type=int,default=2); a=p.parse_args()
    torch.set_num_threads(a.cpu_threads); torch.manual_seed(runner.SEED); np.random.seed(runner.SEED)
    rows,prepared=runner.prepared_rows(a.cache,torch.device('cpu')); n=next(iter(rows.values()))['neural'].shape[1]
    candidate=next((s for s,r in sorted(rows.items()) if r['split']=='train' and not r['mask'].all()),None)
    if candidate is None: raise RuntimeError('real cache has no padded-unit sample')
    r=rows[candidate]; start=int(r['starts'][0]); raw,_,_=runner.xy(r,np.asarray([start]))
    # Create ten explicit left-padding bins.  One genuine remaining bin is all
    # zero and valid: it must be retained, while changes under the false mask
    # and changes in padded units must not influence output.
    x=raw.copy(); x[:,:10,:]=0; x[:,17,:]=0
    valid=np.ones((1,runner.W),dtype=bool); valid[:,:10]=False
    model=runner.build_model(n,torch.device('cpu')).eval(); bank=runner.bank(candidate,r)
    def direct(value:np.ndarray)->np.ndarray:
        with torch.inference_mode(): return model(torch.from_numpy(value),bank,input_valid_mask=torch.from_numpy(valid)).numpy()
    base=direct(x)
    left_changed=x.copy(); left_changed[:,:10,:]=np.linspace(-999,999,n,dtype=np.float32)
    left=direct(left_changed)
    padded_changed=x.copy(); padded_changed[:,:,~r['mask']]=12345.0
    padded=direct(padded_changed)
    stream=runner.RiftStreamDecoder(model); online=[]
    with torch.inference_mode():
        for t in range(runner.W): online.append(stream.stream_step(torch.from_numpy(x[:,t,:]),bank,['padding-real'],valid_mask=torch.from_numpy(valid[:,t]))[0].numpy())
    stream_final=np.asarray(online[-1])[None,:]
    errors={'leftpad_input_max_abs':float(np.abs(base-left).max()),'padded_unit_max_abs':float(np.abs(base-padded).max()),'leftpad_stream_max_abs':float(np.abs(base-stream_final).max())}
    if any(not np.isfinite(v) or v>1e-5 for v in errors.values()): raise RuntimeError(f'padding invariance/parity failed: {errors}')
    # This records that a true valid all-zero raw bin was actually supplied to
    # the stream; it is not conflated with the ten masked padding bins.
    receipt={'schema':'dandi688_rift_real_cache_padding_verification_v1','status':'PASSED','cache_prepared_contract_sha256':runner.obj_hash(prepared),'session':candidate,'n_units':n,'real_units':int(r['mask'].sum()),'padded_units':int((~r['mask']).sum()),'query_start':start,'left_padding_bins':10,'true_zero_bin_index':17,'true_zero_bin_valid':bool(valid[0,17]),'left_padding_valid_count':int(valid[0,:10].sum()),'errors':errors,'base_prediction_sha256':digest(base),'stream_final_sha256':digest(stream_final)}
    atomic_json(a.dest/'padding_verification.json',receipt)
if __name__=='__main__': main()
