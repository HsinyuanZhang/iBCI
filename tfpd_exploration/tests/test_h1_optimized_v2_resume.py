import random
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.paired_train import restore, state
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA

def test_end_epoch_resume_restores_model_optimizer_ema_and_rng_exactly():
    torch.manual_seed(9); np.random.seed(9); random.seed(9)
    a=torch.nn.Linear(3,2); oa=torch.optim.AdamW(a.parameters(),lr=1e-3); ea=DecoderEMA(a)
    x=torch.randn(4,3); y=torch.randn(4,2); loss=((a(x)-y)**2).mean();loss.backward();oa.step();ea.update_after_step(a)
    payload=state(a,oa,ea,epoch=1,step=1)
    expected=(torch.rand(3),np.random.rand(3),random.random())
    b=torch.nn.Linear(3,2); ob=torch.optim.AdamW(b.parameters(),lr=1e-3); eb=DecoderEMA(b)
    epoch,step=restore(payload,b,ob,eb)
    actual=(torch.rand(3),np.random.rand(3),random.random())
    assert (epoch,step)==(1,1)
    torch.testing.assert_close(expected[0],actual[0]);np.testing.assert_array_equal(expected[1],actual[1]);assert expected[2]==actual[2]
    for left,right in zip(a.parameters(),b.parameters()):torch.testing.assert_close(left,right)
