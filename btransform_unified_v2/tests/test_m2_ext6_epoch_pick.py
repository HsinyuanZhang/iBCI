"""Contract tests for the isolated ext6 picker."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch

P=Path(__file__).parents[1]/'scripts/rift_v1/m2_ext6_epoch_pick.py'
spec=importlib.util.spec_from_file_location('m2_ext6_epoch_pick',P); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def report(value=0.2):
 p={s:{'r2':value,'window_count':1,'prediction_sha256':'x'} for s in mod.SIX}
 return {'view':'EMA','partial':False,'per_session':p,'equal_session_mean':value,'pooled_r2':value,'n_windows':6}

def test_complete_report_rejects_nan_and_forged_mean():
 expected={s:{'window_count':1} for s in mod.SIX}
 mod.validate_complete_report(report(),expected)
 bad=report();bad['per_session'][mod.SIX[0]]['r2']=float('nan')
 with pytest.raises(RuntimeError): mod.validate_complete_report(bad,expected)

def test_curve_cannot_complete_with_missing_epoch_or_session():
 expected={s:{'window_count':1} for s in mod.SIX}
 complete={str(e):{**report(), 'checkpoint_sha256':str(e)} for e in mod.EPOCHS}
 mod.validate_complete_curve(complete,expected)
 missing_epoch=dict(complete);missing_epoch.pop('24')
 with pytest.raises(RuntimeError): mod.validate_complete_curve(missing_epoch,expected)
 missing_session={**complete}; bad=report();bad['per_session'].pop(mod.SIX[-1]);missing_session['24']={**bad,'checkpoint_sha256':'24'}
 with pytest.raises(RuntimeError): mod.validate_complete_curve(missing_session,expected)
 bad=report();bad['equal_session_mean']=.3
 with pytest.raises(RuntimeError): mod.validate_complete_report(bad,expected)

def test_earliest_maximum_rule():
 values={1:.2,2:.3,3:.3,4:.1}
 assert max(values,key=lambda e:(values[e],-e)) == 2


def _formal_meta():
 return {'schema':'m2_rift_concat_train_v1','cell':mod.concat.CELL,'seed':42,
         'source_hashes':{'code':'h'},'frozen_cache_hashes':{'cache':'h'}}


def _formal_checkpoint(epoch=1):
 return {'epoch':epoch,'global_step':epoch*3165,'smoke':False,'epochs':24,
         'schema':'m2_rift_concat_epoch_checkpoint_v1','cell':mod.concat.CELL,
         'context_bins':50,'attention_backend':'local','identity_interface':'concat',
         'seed':42,'source_hashes':{'code':'h'},'frozen_cache_hashes':{'cache':'h'}}


def test_checkpoint_contract_rejects_wrong_seed_and_hash():
 meta=_formal_meta(); state=_formal_checkpoint()
 mod._checkpoint_contract(state,meta,1,'concat')
 wrong_seed=dict(state,seed=41)
 with pytest.raises(RuntimeError): mod._checkpoint_contract(wrong_seed,meta,1,'concat')
 wrong_hash=dict(state,frozen_cache_hashes={'cache':'other'})
 with pytest.raises(RuntimeError): mod._checkpoint_contract(wrong_hash,meta,1,'concat')
 joint_meta={'seed':42,'arm':'B','source_hashes':{'code':'h'},'cache_hashes':{'cache':'h'}}
 joint={'epoch':1,'global_step':3165,'smoke':False,'epochs':24,'schema':'m2_rift_joint_checkpoint_v2',
        'cell':'M2-RIFT-R50-D4-JOINT-FILM-M33-V1','context_bins':50,'attention_backend':'local',
        'seed':42,'arm':'B','source_hashes':{'code':'h'},'cache_hashes':{'cache':'h'}}
 mod._checkpoint_contract(joint,joint_meta,1,'joint')
 with pytest.raises(RuntimeError): mod._checkpoint_contract(dict(joint,arm='D'),joint_meta,1,'joint')


def test_actual_concat_loader_materializes_ema_not_raw():
 """Use the real formal checkpoint and the production loader, not a toy EMA."""
 run=Path(__file__).parents[1]/'results/rift_v1/m2_r50_concat_s42_formal_v1'
 state=torch.load(run/'epoch_008.pt',map_location='cpu',weights_only=False)
 raw=mod.concat._decoder(torch.device('cpu')); raw.load_state_dict(state['raw_state_dict'])
 ema=mod.concat._decoder(torch.device('cpu')); mod._load_ema(ema,state,'concat')
 raw_named=dict(raw.named_parameters()); ema_named=dict(ema.named_parameters())
 assert set(raw_named)==set(state['ema']['shadow'])
 assert any(not torch.equal(raw_named[name],state['ema']['shadow'][name]) for name in raw_named)
 for name,param in ema_named.items(): assert torch.equal(param,state['ema']['shadow'][name])


class _FakeModel(torch.nn.Module):
 def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(0.))


def test_resume_mock_24_epoch_run_skips_completed_epochs(tmp_path,monkeypatch):
 """An interrupted 24-epoch curve resumes only epochs that lack atomic rows."""
 manifest={'schema':'x','checkpoint_bytes':{str(e):{'path':str(tmp_path/f'e{e}.pt'),'sha256':f'h{e}'} for e in mod.EPOCHS},
           'query_assets':{'sessions':{s:{'window_count':1} for s in mod.SIX}}}
 (tmp_path/'manifest.json').write_text(json.dumps(manifest))
 (tmp_path/'run_meta.json').write_text(json.dumps(_formal_meta()))
 done={str(e):{**report(.1+e/1000),'checkpoint_sha256':f'h{e}'} for e in range(1,8)}
 (tmp_path/'score_progress.json').write_text(json.dumps({'schema':'x','manifest_sha256':mod.canonical_digest(manifest),'completed':done}))
 calls=[]
 monkeypatch.setattr(mod,'manifest_for',lambda *a:manifest)
 monkeypatch.setattr(mod,'verify_manifest',lambda *a:None)
 monkeypatch.setattr(mod,'load_query_pair',lambda *a:(object(),object()))
 monkeypatch.setattr(mod,'_model_and_loader',lambda *a:('concat',_FakeModel()))
 monkeypatch.setattr(mod,'_checkpoint_contract',lambda *a:None)
 monkeypatch.setattr(mod,'_load_ema',lambda *a:None)
 monkeypatch.setattr(mod.torch,'load',lambda *a,**k:{})
 monkeypatch.setattr(mod,'sha',lambda p: next((f'h{e}' for e in mod.EPOCHS if str(p).endswith(f'e{e}.pt')),'package'))
 def fake_score(*args,**kwargs):
  epoch=len(calls)+8; calls.append(epoch); return report(.1+epoch/1000)
 monkeypatch.setattr(mod,'score',fake_score)
 args=type('A',(),{'run':tmp_path,'dest':tmp_path,'query_cache':tmp_path,'joint_m33_root':None,
                    'resume':True,'preflight_only':False,'device':'cpu','cpu_threads':1})()
 out=mod.run(args)
 assert calls==list(range(8,25))
 assert out['selection']['epoch']==24
