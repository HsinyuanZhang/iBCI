from pathlib import Path
import importlib.util, json, hashlib, types
import pytest
import torch
from btransform_unified_v1 import adapters
from btransform_unified_v2.m2_mechanism_model import (
    M2MechanismDecoder, CausalDepthwiseTemporal, ARM_B, ARM_C, ARM_D, ARM_D_SHUFFLE,
)

CACHE=Path(__file__).resolve().parents[2]/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train'

def _model(arm, **kw):
    b=adapters.build_m2_bank('source_train','ses-2020-10-19-Run1')
    m=M2MechanismDecoder(arm,**kw);m.install_session_memory({b.session_id:b},CACHE);return m,b

def test_information_arms_define_static_paths_and_shuffle_both_carrier_routes():
    c,b=_model(ARM_C); assert not hasattr(c,'encoder')
    e,carrier=c._identity([b.session_id],torch.device('cpu')); assert torch.count_nonzero(e)==0 and torch.count_nonzero(carrier)>0
    b_only,_=_model(ARM_B); _,bc=b_only._identity([b.session_id],torch.device('cpu')); assert torch.count_nonzero(bc)==0
    d,_=_model(ARM_D); ds,_=_model(ARM_D_SHUFFLE); _,dc=d._identity([b.session_id],torch.device('cpu')); _,sc=ds._identity([b.session_id],torch.device('cpu'))
    assert not torch.equal(dc,sc) and torch.equal(sc[0],dc[0,ds.carrier_perm_ses_2020_10_19_Run1])

def test_mean_control_has_no_slot_modules_and_returns_256_tokens():
    m,b=_model(ARM_D,aggregation='mean'); assert not any(hasattr(m.frontend,n) for n in ('slots','mha','slot_ffn','slot_ffn_norm','slot_proj'))
    out=m.frontend_tokens(torch.from_numpy(b.X_store[:1]),b);assert out.shape==(1,50,256)

def test_nonattention_control_is_causal_rf50_and_stream_equivalent():
    torch.manual_seed(2); m=CausalDepthwiseTemporal().eval()
    with torch.no_grad():
        for norm in m.norms: norm.bias.fill_(.37)
    x=torch.randn(2,55,256); mask=torch.ones(2,55,dtype=torch.bool);mask[:,:3]=False; y=m(x,mask);changed=x.clone();changed[:,50:]+=100
    assert m.receptive_field==50 and torch.allclose(y[:,:50],m(changed,mask)[:,:50],atol=2e-5,rtol=2e-5)
    state=m.init_state(2,'cpu',x.dtype); pieces=[]
    for t in range(x.shape[1]):
        o,state=m.step(x[:,t],state,mask[:,t]);pieces.append(o)
    assert torch.allclose(torch.stack(pieces,1),y,atol=2e-5,rtol=2e-5)

def test_shared_decoder_initialization_is_seed_stable_across_information_arms():
    b,_=_model(ARM_B);c,_=_model(ARM_C); d,_=_model(ARM_D)
    for name in ('readout.0.weight','readout.0.bias','frontend.local_conv.conv.weight'):
        assert torch.equal(dict(b.named_parameters())[name],dict(c.named_parameters())[name])
        assert torch.equal(dict(b.named_parameters())[name],dict(d.named_parameters())[name])

def test_mechanism_score_state_reuses_valid_24_epoch_progress(monkeypatch,tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/rift_v1/m2_mechanism_train.py';spec=importlib.util.spec_from_file_location('mechanism_runner',script);r=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(r)
    report={'partial':False,'n_windows':2069,'equal_session_mean':.1,'pooled_r2':.2,'per_session':{n:{'window_count':c,'r2':.1} for n,c in r.base.old_plan.EXT4_EXPECTED_WINDOWS.items()}}
    meta={'status':'FORMAL','cell':r.CELL,'arm':ARM_C,'aggregation':'slots','temporal':'attention','seed':42,'sampler_seed':42,'epochs':24,'source_hashes':{},'cache_hashes':{'source_train':{},'source_minival':{},'ext4':{}}}
    receipt={**meta,'status':'COMPLETED','global_step':75960};(tmp_path/'run_meta.json').write_text(json.dumps(meta));(tmp_path/'train_receipt.json').write_text(json.dumps(receipt));done={}
    for e in range(1,25):
        p=tmp_path/f'epoch_{e:03d}.pt';p.write_bytes(str(e).encode());done[str(e)]={**report,'checkpoint_sha256':hashlib.sha256(str(e).encode()).hexdigest()}
    (tmp_path/'ext4_score_progress.json').write_text(json.dumps({'schema':'m2_rift_mechanism_ext4_progress_v1',**{k:meta[k] for k in ('cell','arm','aggregation','temporal','seed','source_hashes','cache_hashes')},'completed':done}))
    monkeypatch.setattr(r,'hashes',lambda:{});monkeypatch.setattr(r,'load_all',lambda d: ({},{},{},{},{},{}));monkeypatch.setattr(r,'chash',lambda s,b:{});monkeypatch.setattr(r,'model_for',lambda *x:object());monkeypatch.setattr(r.base,'_surface_padding',lambda *x:{})
    a=types.SimpleNamespace(dest=tmp_path,arm=ARM_C,aggregation='slots',temporal='attention',seed=42,device='cpu',cpu_threads=1,max_updates_smoke=None)
    assert r.score(a)['selection']['epoch']==1
    meta['arm']=ARM_D;(tmp_path/'run_meta.json').write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):r.score(a)
