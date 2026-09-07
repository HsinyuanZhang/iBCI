"""CPU contract tests for the single-variable V7 p=.30 experiment."""
import torch
from tfpd_exploration.src.h1_optimized_v4.paired_train import keep as v4_keep
from tfpd_exploration.src.h1_optimized_v7.capacity_gate import keep as gate_keep
from tfpd_exploration.src.h1_optimized_v7.paired_train import keep as formal_keep


def test_p01_is_bitwise_v4_keep_stream_and_p30_is_subset_except_forced_rows():
    device=torch.device('cpu')
    for epoch,batch,n in ((1,0,16),(3,8,32),(12,101,7)):
        old=v4_keep(n,epoch,batch,device)
        p01=formal_keep(n,epoch,batch,device,p=.1)
        p30=formal_keep(n,epoch,batch,device,p=.3)
        assert torch.equal(p01,old)
        # At 176 units, the forced nonempty repair is irrelevant for these
        # deterministic rows; retain the broader property as well.
        assert bool((p30<=p01).all())
        assert bool((p30.sum(-1)>0).all())


def test_gate_and_formal_share_p30_law_when_epoch_batch_match():
    device=torch.device('cpu')
    assert torch.equal(gate_keep(16,1,37,device,p=.3),formal_keep(16,1,37,device,p=.3))
