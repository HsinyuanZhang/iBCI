"""One-time, owned H1 source-data cache; never writes legacy result paths."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .data import build_window_manifest, materialize_banks

ROOT = Path(__file__).resolve().parents[2] / "results/decoder_validation_v2/20260905_190000/h1"
CACHE = ROOT / "source_cache.pt"

def _array_hash(x) -> str:
    a=np.ascontiguousarray(x)
    h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(memoryview(a).cast('B')); return h.hexdigest()

def authority(payload: dict) -> dict:
    rows={}
    for split in ('train','minival'):
        rows[split]={name:{'source_nwb_sha256':row['sha256'],'neural_sha256':_array_hash(row['neural']),
          'velocity_sha256':_array_hash(row['velocity']),'mask_sha256':_array_hash(row['eval_mask']),
          'starts_sha256':_array_hash(row['query_starts']),'bank_e0_sha256':_array_hash(row['bank']['E0'].numpy()),
          'bank_hc_sha256':_array_hash(row['bank']['T'].numpy())} for name,row in payload[split].items()}
    code=Path(__file__).parent
    files=('cache.py','data.py','model.py','paired_train.py','score.py')
    return {'schema':'h1_optimized_v2_cache_authority_v1','arrays':rows,
      'code_sha256':{f:hashlib.sha256((code/f).read_bytes()).hexdigest() for f in files if (code/f).is_file()}}

def validate_authority(payload: dict, recorded: dict) -> None:
    """Reject stale/corrupt cached source arrays; code hashes are provenance, not equality gates."""
    actual=authority(payload)
    if actual['arrays'] != recorded['arrays']:
        raise RuntimeError('source-cache authority mismatch: raw input, arrays, starts, or bank changed')

def build_or_load() -> dict:
    if CACHE.is_file():
        payload=torch.load(CACHE, map_location="cpu", weights_only=False)
        receipt=ROOT/'source_cache_authority.json'
        if not receipt.is_file():
            receipt.write_text(json.dumps(authority(payload),indent=2,sort_keys=True)+'\n')
        return payload
    pack = build_window_manifest()  # local non-mutating implementation
    all_sessions = {**pack['train_sessions'], **pack['mini_sessions']}
    banks = materialize_banks(all_sessions)
    def encode(rows):
        return {name: {
            'neural': rec.neural, 'velocity': rec.velocity, 'eval_mask': rec.eval_mask,
            'query_starts': np.asarray(rec.query_starts, dtype=np.int64),
            'first3_end': rec.first3_end, 'first3': rec.first3, 'sha256': rec.sha256,
            'bank': {'E0': banks[name].E0, 'T': banks[name].T, 'unit_mask': banks[name].unit_mask},
        } for name, rec in rows.items()}
    payload = {'schema':'h1_optimized_v2_source_cache_v1','train':encode(pack['train_sessions']),'minival':encode(pack['mini_sessions'])}
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE)
    (ROOT/'source_cache_authority.json').write_text(json.dumps(authority(payload),indent=2,sort_keys=True)+'\n')
    receipt = {'schema':payload['schema'], 'cache':str(CACHE),
      'train_windows':int(sum(len(x['query_starts']) for x in payload['train'].values())),
      'frozen_minival_window_selection':int(sum(len(x['query_starts']) for x in payload['minival'].values())),
      'complete_stream_eval_bins':int(sum(x['eval_mask'].sum() for x in payload['minival'].values())),
      'disclosure':'known-source development only; selection uses frozen minival windows, terminal stream metric is separate'}
    (ROOT/'source_cache_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return payload
