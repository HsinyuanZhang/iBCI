"""Mounted local-only container parity check; no evaluator, labels, or timing."""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/')
from falcon_challenge.config import FalconConfig,FalconTask
from m1_trf_falcon_decoder import M1TemporalFalconDecoder
ROOT=Path('/work'); fixture=np.load(ROOT/'fixtures/source_b4_public_parity.npz',allow_pickle=False)
if hashlib.sha256(Path('/data/decoder.pkl').read_bytes()).hexdigest()!=str(fixture['payload_sha256'][0]):raise RuntimeError('payload digest')
stems=[str(x) for x in fixture['stems']]; x,expected=fixture['inputs'],fixture['expected']
cfg=FalconConfig(task=FalconTask.m1); dec=M1TemporalFalconDecoder(cfg,'/data/decoder.pkl',batch_size=4);dec.reset(stems);mx=0.
for got,ref in zip((dec.predict(np.ascontiguousarray(v)) for v in x),expected,strict=True):
    if got.shape!=(4,16) or got.dtype!=np.float32 or not got.flags.owndata:raise RuntimeError('public B4 contract')
    mx=max(mx,float(np.abs(got-ref).max()))
    if not np.all(np.abs(got-ref)<=1e-5+1e-5*np.abs(ref)):raise RuntimeError('long B4 parity')
# Different bank and final partial evaluator batch share no state with prior reset.
dec.reset([stems[1],stems[2]]);partial=dec.predict(np.ascontiguousarray(x[-1,:2]));
if partial.shape!=(2,16) or partial.dtype!=np.float32 or not np.isfinite(partial).all():raise RuntimeError('partial')
one=M1TemporalFalconDecoder(cfg,'/data/decoder.pkl',batch_size=1);one.reset([stems[0]]);b1=one.predict(np.ascontiguousarray(x[0,:1]));
if b1.shape!=(1,16) or not np.isfinite(b1).all():raise RuntimeError('B1')
print({'status':'PASS','long_b4_calls':len(x),'max_native_abs_error':mx,'partial_shape':list(partial.shape),'b1_shape':list(b1.shape)})
