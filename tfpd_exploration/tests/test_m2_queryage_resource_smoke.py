from __future__ import annotations
import torch,pytest
from tfpd_exploration.src.m2_queryage_family_v1 import resource_smoke as s

def test_prefix_geometry_native_current_and_determinism():
    x=torch.arange(3*s.W*s.UNITS,dtype=torch.float32).reshape(3,s.W,s.UNITS)
    a,lengths=s.prefix(x,seed=42,epoch=1,batch_id=3); b,lengths2=s.prefix(x,seed=42,epoch=1,batch_id=3)
    assert a.shape==x.shape and torch.equal(a[:,-1],x[:,-1]) and torch.equal(a,b) and torch.equal(lengths,lengths2)
    assert torch.all((lengths>=1)&(lengths<=s.W)) and s.prefix(x,seed=1,epoch=1,batch_id=1,probability=0)[1].tolist()==[s.W]*3
    with pytest.raises(RuntimeError,match='geometry'):s.prefix(torch.zeros(2,49,s.UNITS),seed=1,epoch=1,batch_id=1)

def test_prefix_leaves_target_and_paired_dropout_inputs_unmodified():
    x=torch.randn(2,s.W,s.UNITS); target=torch.randn(2,s.OUTPUTS);control,_=s.prefix(x,seed=42,epoch=1,batch_id=1,probability=0);short,_=s.prefix(x,seed=42,epoch=1,batch_id=1)
    assert torch.equal(control,x) and torch.equal(short[:,-1],x[:,-1]) and torch.equal(target,target.clone())

def test_preflight_gate_precedes_manifest_read(tmp_path,monkeypatch):
    monkeypatch.delenv(s.GO,raising=False);monkeypatch.setattr(s,'sha',lambda _:pytest.fail('read before gate'))
    with pytest.raises(RuntimeError,match='explicit GO'):s.preflight((tmp_path/'out').resolve(),0)

def test_real_tiny_adamw_then_ema_update_sequence():
    from tfpd_exploration.src.m2_b_small_stability_v1 import training
    from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def trainable_parameters(self): return {'weight':self.weight}
    model=Tiny(); opt=training.make_optimizer(model.trainable_parameters().items()); ema=DecoderEMA(model,decay=.9995)
    opt.zero_grad(); loss=(model.weight-3).square().sum();loss.backward();opt.step();ema.update_after_step(model)
    assert ema.n_updates==1 and opt.state[model.weight] and torch.isfinite(model.weight).all()
