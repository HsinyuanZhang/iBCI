"""CPU-only contracts for the unlaunched CRST-B4 split formal harness."""
import numpy as np
import pytest
import random
import copy
import io
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
import torch
from tfpd_exploration.src.h1_family_v1 import familyformal_split_train as t

def test_stateless_pair_mask_intersects_bank_and_warmup_is_exact():
    bank=torch.tensor([True,False,True,False])
    a=t.dropout_keep(n=5,epoch=1,batch_index=7,bank_mask=bank,device=torch.device('cpu'))
    b=t.dropout_keep(n=5,epoch=1,batch_index=7,bank_mask=bank,device=torch.device('cpu'))
    assert torch.equal(a,b) and not bool((a & ~bank).any())
    assert t.warmup_lr(epoch=1,global_step=1)==t.LR/t.EPOCH_UPDATES
    assert t.warmup_lr(epoch=1,global_step=t.EPOCH_UPDATES)==pytest.approx(t.LR)
    assert t.warmup_lr(epoch=2,global_step=t.EPOCH_UPDATES+1)==pytest.approx(t.LR)

def test_full_order_digest_and_earliest_finite_max_contract():
    rows=[('s',np.arange(i*32,(i+1)*32,dtype=np.int64)) for i in range(t.EPOCH_UPDATES)]
    assert t.sampler_identity_digest(rows)==t.sampler_identity_digest(rows)
    scores=[(i,0.0 if i in (3,7) else -1.0) for i in range(1,13)]
    assert t.earliest_argmax(scores)==(3,0.0)
    bad=list(scores); bad[5]=(6,float('nan'))
    with pytest.raises(RuntimeError,match='nonfinite'):
        t.earliest_argmax(bad)

def test_tiny_checkpoint_strict_restore_preserves_future_rng_and_rejects_provenance(monkeypatch):
    # One-step toy epoch exercises the real serialization primitive; it is
    # not evidence of a completed H1 training epoch.
    monkeypatch.setattr(t, "EPOCH_UPDATES", 1)
    m=torch.nn.Linear(3,2); o=torch.optim.AdamW(m.parameters(),lr=.01); e=DecoderEMA(m,decay=.9)
    x=torch.ones(2,3); m(x).sum().backward(); o.step(); e.update_after_step(m)
    rows=[('s',np.arange(i*32,(i+1)*32,dtype=np.int64)) for i in range(t.EPOCH_UPDATES)]
    dig=t.sampler_identity_digest(rows); code={'x':'y'}
    p=t.checkpoint_payload(model=m,optimizer=o,ema=e,epoch=1,global_step=t.EPOCH_UPDATES,sampler_digest=dig,source_authority_sha256='a',code_sha256=code,arm='flat',shared_init_sha256='shared',dropout_digest='keep')
    buffer=io.BytesIO();torch.save(p,buffer);buffer.seek(0);p=torch.load(buffer,weights_only=False)
    before=torch.rand(4); m2=torch.nn.Linear(3,2);o2=torch.optim.AdamW(m2.parameters(),lr=.01);e2=DecoderEMA(m2,decay=.9)
    kwargs=dict(model=m2,optimizer=o2,ema=e2,expected_epoch=1,expected_sampler_digest=dig,expected_source_authority_sha256='a',expected_code_sha256=code,expected_arm='flat',expected_shared_init_sha256='shared',expected_dropout_digest='keep')
    assert t.strict_restore(payload=p,**kwargs)==2
    assert torch.equal(torch.rand(4),before)
    for name in m.state_dict(): torch.testing.assert_close(m.state_dict()[name],m2.state_dict()[name],rtol=0,atol=0)
    for model,opt,ema in ((m,o,e),(m2,o2,e2)):
        opt.zero_grad(set_to_none=True); model(x).square().mean().backward(); opt.step(); ema.update_after_step(model)
    for name in m.state_dict(): torch.testing.assert_close(m.state_dict()[name],m2.state_dict()[name],rtol=0,atol=0)
    for name in e.shadow: torch.testing.assert_close(e.shadow[name],e2.shadow[name],rtol=0,atol=0)
    for field in ('source_authority_sha256','arm','shared_init_sha256','dropout_identity_sha256'):
        q=dict(p);q[field]='bad'
        with pytest.raises(RuntimeError): t.strict_restore(payload=q,**kwargs)


def test_uneven_microbatch_loss_and_gradient_match_full_mean():
    class Tiny(torch.nn.Linear):
        def forward_last(self, x, bank, dropout_keep=None): return self(x)
    m=Tiny(3,2); other=copy.deepcopy(m)
    x=torch.randn(14,3); y=torch.randn(14,2); keep=torch.ones(14,3,dtype=torch.bool)
    actual=t._normalised_micro_loss(m,x,y,None,keep)
    expected=torch.nn.functional.mse_loss(other(x),y);expected.backward()
    assert actual==pytest.approx(float(expected),abs=1e-6)
    for p,q in zip(m.parameters(),other.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
