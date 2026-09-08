"""Seed admission tests for the independent multi-seed ext6 picker."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

P=Path(__file__).parents[1]/'scripts/rift_v1/m2_ext6_epoch_pick_multiseed.py'
spec=importlib.util.spec_from_file_location('m2_ext6_epoch_pick_multiseed',P); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _run(tmp_path, seed, receipt_seed):
    source=tmp_path/'train.py';source.write_text('frozen')
    run=tmp_path/'run';run.mkdir()
    meta={'schema':'m2_rift_joint_train_v2','status':'FORMAL','cell':'M2-RIFT-R50-D4-JOINT-FILM-M33-V1','arm':'D_JOINT','seed':seed,'source_hashes':{str(source):_sha(source)},'cache_hashes':{'x':'y'}}
    receipt={'schema':'m2_rift_joint_train_receipt_v2','status':'COMPLETED','cell':meta['cell'],'arm':'D_JOINT','seed':receipt_seed,'sampler_seed':42,'epochs':24,'global_step':75960,'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes']}
    (run/'run_meta.json').write_text(json.dumps(meta));(run/'train_receipt.json').write_text(json.dumps(receipt))
    for e in mod.EPOCHS:(run/f'epoch_{e:03d}.pt').write_bytes(str(e).encode())
    return run

@pytest.mark.parametrize('seed',[42,43,44])
def test_joint_three_seed_receipts_are_accepted(tmp_path,monkeypatch,seed):
    run=_run(tmp_path,seed,seed)
    monkeypatch.setattr(mod,'_source_paths',lambda:())
    monkeypatch.setattr(mod,'query_asset_hashes',lambda cache:{'sessions':{}})
    monkeypatch.setattr(mod,'sealed_bt_oracle_binding',lambda cache,assets:{})
    manifest=mod.manifest_for(run,tmp_path,None)
    assert manifest['seed']==seed and manifest['schema']==mod.SCHEMA+'_manifest'

def test_joint_train_receipt_seed_mismatch_is_rejected(tmp_path,monkeypatch):
    run=_run(tmp_path,43,42)
    monkeypatch.setattr(mod,'_source_paths',lambda:())
    with pytest.raises(RuntimeError): mod.manifest_for(run,tmp_path,None)

def test_joint_wrong_seed_is_rejected(tmp_path,monkeypatch):
    run=_run(tmp_path,45,45)
    monkeypatch.setattr(mod,'_source_paths',lambda:())
    with pytest.raises(RuntimeError): mod.manifest_for(run,tmp_path,None)
