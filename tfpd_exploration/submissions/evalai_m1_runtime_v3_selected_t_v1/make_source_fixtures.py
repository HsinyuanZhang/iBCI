"""Make source-only native public-API parity fixtures; never opens outer data."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from falcon_challenge.config import FalconConfig,FalconTask
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1')
PACKAGE=ROOT/'submissions/evalai_m1_runtime_v3_selected_t_v1'
PAYLOAD=PACKAGE/'artifacts/m1_optimized_v2_t_ema_e6.pkl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=Path(__file__).parent/'fixtures';out.mkdir(exist_ok=True); target=out/'source_b4_public_parity.npz'
 if target.exists():raise FileExistsError(target)
 rec=json.loads((PACKAGE/'artifacts/payload.receipt.inherited.json').read_text()); names=['ses-20120926','ses-20120927','ses-20120928','ses-20120926'];stems=[Path(rec['banks']['records'][n]['file']).stem for n in names]
 cache=np.load(SOURCE/'m1_optimized_v2_source_runtime_cache.npz',allow_pickle=False); inputs=np.stack([cache[f'raw_neural/{n}'][99:99+256] for n in names],1).astype(np.float32);cache.close()
 sys.path.insert(0,str(PACKAGE)); spec=importlib.util.spec_from_file_location('_frozen_m1_pkg',PACKAGE/'m1_trf_falcon_decoder.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 dec=mod.M1TemporalFalconDecoder(FalconConfig(task=FalconTask.m1),str(PAYLOAD),batch_size=4);dec.reset(stems);expected=np.stack([dec.predict(np.ascontiguousarray(x)) for x in inputs])
 np.savez_compressed(target,inputs=inputs,expected=expected,stems=np.asarray(stems),payload_sha256=np.asarray([sha(PAYLOAD)]),source_cache_sha256=np.asarray([sha(SOURCE/'m1_optimized_v2_source_runtime_cache.npz')]))
 print(target)
if __name__=='__main__':main()
