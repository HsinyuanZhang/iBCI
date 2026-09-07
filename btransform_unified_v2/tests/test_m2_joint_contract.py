from pathlib import Path
import torch
from btransform_unified_v1 import adapters
from btransform_unified_v2.joint_m2_model import JointM2RiftDecoder,ARM_B,ARM_D
CACHE=Path('tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train')
def test_joint_provider_grad_and_arm_contract():
 b=adapters.build_m2_bank('source_train','ses-2020-10-19-Run1');x=torch.from_numpy(b.X_store[:1]);
 m=JointM2RiftDecoder(ARM_B);m.install_session_memory({b.session_id:b},CACHE);out=m(x,b);out.sum().backward();assert out.shape==(1,2);assert all(p.grad is not None for p in m.encoder.parameters())
 d=JointM2RiftDecoder(ARM_D);d.install_session_memory({b.session_id:b},CACHE); eb,cb=m._identity([b.session_id],torch.device('cpu'));ed,cd=d._identity([b.session_id],torch.device('cpu'));assert torch.count_nonzero(cb)==0 and torch.count_nonzero(cd)>0 and eb.shape==ed.shape==(1,96,50)
def test_joint_mask_rejects_holes():
 b=adapters.build_m2_bank('source_train','ses-2020-10-19-Run1');m=JointM2RiftDecoder(ARM_B);m.install_session_memory({b.session_id:b},CACHE);x=torch.from_numpy(b.X_store[:1]);mask=torch.ones(1,50,dtype=torch.bool);mask[:,10]=False
 import pytest
 with pytest.raises(ValueError):m(x,b,input_valid_mask=mask)

def test_joint_score_report_and_checkpoint_contract_guards():
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/rift_v1/m2_joint_train.py'
    spec=importlib.util.spec_from_file_location('joint_runner',script);r=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(r)
    report={'partial':False,'n_windows':2069,'equal_session_mean':0.1,'pooled_r2':0.2,'per_session':{name:{'window_count':count,'r2':0.1} for name,count in r.base.old_plan.EXT4_EXPECTED_WINDOWS.items()}}
    r.validate_report(report)
    import pytest
    report['per_session'][next(iter(report['per_session']))]['r2']=float('nan')
    with pytest.raises(RuntimeError):r.validate_report(report)
    meta={'arm':ARM_B,'seed':42,'source_hashes':{},'cache_hashes':{}}
    state={'schema':'m2_rift_joint_checkpoint_v2','cell':r.CELL,'arm':ARM_B,'seed':42,'smoke':False,'epochs':24,'context_bins':50,'depth':4,'attention_backend':'local','epoch':1,'global_step':3165,'source_hashes':{},'cache_hashes':{}}
    r.required(state,meta)
    state['arm']=ARM_D
    with pytest.raises(RuntimeError):r.required(state,meta)

def test_joint_state_dict_registers_every_trainable_parameter_once():
    b=adapters.build_m2_bank('source_train','ses-2020-10-19-Run1')
    m=JointM2RiftDecoder(ARM_D);m.install_session_memory({b.session_id:b},CACHE)
    names=dict(m.named_parameters());state=m.state_dict()
    assert set(names).issubset(state)
    assert len(names)==len(set(names))
    assert all(parameter.requires_grad for parameter in names.values())

def test_joint_frontend_last_uses_live_encoder_not_bank_cache():
    b=adapters.build_m2_bank('source_train','ses-2020-10-19-Run1');m=JointM2RiftDecoder(ARM_D).eval();m.install_session_memory({b.session_id:b},CACHE);raw=torch.from_numpy(b.X_store[:1,-5:])
    with torch.no_grad():before=m.frontend_last(raw,b);m.encoder.post_pool[0].bias.add_(1.0);after=m.frontend_last(raw,b)
    assert not torch.allclose(before,after)

def test_joint_cli_rejects_invalid_epochs_smoke_and_score_resume(monkeypatch,tmp_path):
    import importlib.util,sys,pytest
    script=Path(__file__).resolve().parents[1]/'scripts/rift_v1/m2_joint_train.py';spec=importlib.util.spec_from_file_location('joint_cli',script);r=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(r)
    for extra in (['--epochs','0'],['--max-updates-smoke','0'],['--stage','score','--resume',str(tmp_path/'x')]):
        monkeypatch.setattr(sys,'argv',['m2_joint_train.py','--dest',str(tmp_path),'--arm',ARM_B,*extra])
        with pytest.raises(SystemExit) as e:r.main()
        assert e.value.code==2

def test_score_state_machine_skips_valid_24_progress_and_rejects_wrong_bindings(monkeypatch,tmp_path):
    import importlib.util,types,json,hashlib,pytest
    script=Path(__file__).resolve().parents[1]/'scripts/rift_v1/m2_joint_train.py';spec=importlib.util.spec_from_file_location('joint_score',script);r=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(r)
    report={'partial':False,'n_windows':2069,'equal_session_mean':.1,'pooled_r2':.2,'per_session':{n:{'window_count':c,'r2':.1} for n,c in r.base.old_plan.EXT4_EXPECTED_WINDOWS.items()}}
    meta={'status':'FORMAL','cell':r.CELL,'arm':ARM_B,'seed':42,'sampler_seed':42,'epochs':24,'source_hashes':{},'cache_hashes':{'source_train':{},'source_minival':{},'ext4':{}}}
    receipt={'status':'COMPLETED','cell':r.CELL,'arm':ARM_B,'seed':42,'sampler_seed':42,'epochs':24,'global_step':75960,'source_hashes':{},'cache_hashes':{'source_train':{},'source_minival':{},'ext4':{}}}
    (tmp_path/'run_meta.json').write_text(json.dumps(meta));(tmp_path/'train_receipt.json').write_text(json.dumps(receipt))
    completed={}
    for e in range(1,25):
        p=tmp_path/f'epoch_{e:03d}.pt';p.write_bytes(str(e).encode());completed[str(e)]={**report,'checkpoint_sha256':hashlib.sha256(str(e).encode()).hexdigest()}
    (tmp_path/'ext4_score_progress.json').write_text(json.dumps({'schema':'m2_rift_joint_ext4_progress_v1','cell':r.CELL,'arm':ARM_B,'seed':42,'source_hashes':{},'cache_hashes':{'source_train':{},'source_minival':{},'ext4':{}},'completed':completed}))
    monkeypatch.setattr(r,'hashes',lambda:{})
    monkeypatch.setattr(r,'load_all',lambda dev: ({},{},{},{},{},{}))
    monkeypatch.setattr(r,'chash',lambda surf,banks:{})
    monkeypatch.setattr(r,'model_for',lambda *x:object())
    monkeypatch.setattr(r.base,'_surface_padding',lambda *x:{})
    a=types.SimpleNamespace(dest=tmp_path,arm=ARM_B,seed=42,device='cpu',cpu_threads=1,max_updates_smoke=None)
    assert r.score(a)['selection']['epoch']==1
    meta['seed']=43;(tmp_path/'run_meta.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):r.score(a)
    meta['seed']=42;meta['arm']=ARM_D;(tmp_path/'run_meta.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):r.score(a)
    meta['arm']=ARM_B;meta['source_hashes']={'drift':'x'};(tmp_path/'run_meta.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):r.score(a)
