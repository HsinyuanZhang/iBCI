from pathlib import Path
import sys
import numpy as np
import torch
import random

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"rift_v1"
sys.path.insert(0,str(SCRIPT))
from h1_train import CONTEXT, LAYERS, endpoint_context, layer_bins, _checkpoint

def test_r300_geometry_and_endpoint_alignment():
    assert CONTEXT == 300
    assert 5 + sum(b-1 for b in LAYERS) == 300
    raw=np.arange(10*2,dtype=np.float32).reshape(10,2)
    x, valid=endpoint_context(raw,np.array([0,4,9],np.int64))
    assert x.shape == (3,300,2) and valid.shape == (3,300)
    assert valid.sum(1).tolist() == [1,5,10]
    assert np.array_equal(x[0,-1],raw[0])
    assert np.array_equal(x[1,-5:],raw[:5])
    assert np.array_equal(x[2,-10:],raw)
    assert not valid[0,:299].any()  # padding cannot become zero-valued evidence

def test_nonearly_endpoint_is_exactly_a_minus_299_through_a():
    raw=np.arange(350,dtype=np.float32)[:,None]
    x, valid=endpoint_context(raw,np.array([320],np.int64))
    assert valid.all()
    assert x[0,0,0] == 21 and x[0,-1,0] == 320

def test_r200_geometry_and_same_endpoint_coordinate():
    assert layer_bins(200) == (50,50,50,49)
    assert 5 + sum(b-1 for b in layer_bins(200)) == 200
    raw=np.arange(250,dtype=np.float32)[:,None]
    x, valid=endpoint_context(raw,np.array([220],np.int64),200)
    assert valid.all() and x.shape == (1,200,1)
    assert x[0,0,0] == 21 and x[0,-1,0] == 220

def test_cold_start_mask_is_accepted_and_does_not_change_valid_tail_shape():
    """The trainer must pass an explicit mask, never treat left zero fill as KV."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT.parent / "btransform_unified_v1" / "src"))
    from btransform_unified_v2.model import RiftDecoder
    from btransform_unified_v1.bank import TaskBank
    bank=TaskBank("synthetic",np.zeros((176,700),np.float32),np.zeros((176,4),np.float32),np.ones(176,bool),np.zeros((1,300,176),np.float32),np.zeros((1,7),np.float32),np.array([0],np.int64),{"shape":(176,700),"trial_count":3,"budget":3,"estimator":"test","array_sha256":"test"})
    model=RiftDecoder(task="h1",context_bins=300,bias_mode="recency",seed=42,proj_dim=16)
    x=torch.zeros((1,300,176)); valid=torch.zeros((1,300),dtype=torch.bool); valid[:,-3:]=True
    assert model(x,bank,input_valid_mask=valid).shape == (1,7)
    assert model.forward_scores(x,bank,input_valid_mask=valid).shape == (1,300,7)

def test_checkpoint_cpu_rng_restore_fixture(tmp_path):
    """Checkpoint CPU RNG stays restorable even if a loader mapped tensors elsewhere."""
    model=torch.nn.Linear(2,1); opt=torch.optim.AdamW(model.parameters());
    class TinyEMA:
        def state_dict(self): return {"ok":True}
    rng=np.random.default_rng(42); random.seed(42); torch.manual_seed(42)
    path=tmp_path/"epoch_001.pt"
    _checkpoint(path,model,opt,TinyEMA(),1,731,rng,variant="recency",context_bins=300,attention_backend="local",microbatch=8,smoke=False)
    state=torch.load(path,map_location="cpu",weights_only=False)
    torch.set_rng_state(state["torch_rng_cpu"].cpu())
    assert state["smoke"] is False and state["global_step"] == 731
    assert state["torch_rng_cuda"] is None or all(t.device.type=="cpu" for t in state["torch_rng_cuda"])
