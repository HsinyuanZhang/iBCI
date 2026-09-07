from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]; SCRIPT=ROOT/'scripts/rift_v1/m1_concat_train.py'
def runner():
 s=importlib.util.spec_from_file_location('m1_concat_runner',SCRIPT); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def test_distinct_full_concat_contract():
 m=runner(); assert m.CELL=='M1-RIFT-R100-D4-CONCAT-E100-RECENCY-V1'
 x=torch.nn.Linear(1,1); e=m.DecoderEMA(x,decay=.9); state=m._checkpoint_payload(x,torch.optim.AdamW(x.parameters()),e,epoch=1,step=1,smoke=True,meta={'source_hashes':{},'source_contract':{},'initialization_sha256':'x'},device=torch.device('cpu'))
 assert state['schema']=='m1_rift_concat_epoch_checkpoint_v1'; assert state['config']['proj_dim'] is None
def test_full_e0_fold_preserves_m1_baseline_and_owns_only_active_head():
 from btransform_unified_v2 import RiftDecoder
 from btransform_unified_v2.concat_model import RiftConcatDecoder
 torch.manual_seed(123); b=RiftDecoder('m1',context_bins=100,bias_mode='recency',seed=42,proj_dim=16).eval()
 torch.manual_seed(123); c=RiftConcatDecoder('m1',context_bins=100,bias_mode='recency',seed=42).eval()
 assert c.frontend.token_in==120 and c._frontend_owner.final_norm is c.final_norm and c._frontend_owner.readout is c.readout
 shared=dict(c.named_parameters()); pairs=[(n,p) for n,p in b.named_parameters() if n in shared and p.shape==shared[n].shape]
 assert pairs and all(torch.equal(p,shared[n]) for n,p in pairs)
 # No bank/data is needed to prove the folded first affine map exactly.
 local=torch.randn(2,3,64,16); e0=torch.randn(2,64,100); carrier=torch.randn(2,64,4)
 old=b.frontend.token_mlp[0]; projected=e0.unsqueeze(1).expand(-1,3,-1,-1) @ b.frontend.e0_proj.weight.T; ref=old(torch.cat((local + projected,carrier.unsqueeze(1).expand(-1,3,-1,-1)),dim=-1))
 got=c.frontend.token_mlp[0](torch.cat((local,e0.unsqueeze(1).expand(-1,3,-1,-1),carrier.unsqueeze(1).expand(-1,3,-1,-1)),dim=-1))
 assert float((ref-got).abs().max())<2e-6
def test_score_rejects_smoke_argument(monkeypatch,tmp_path):
 m=runner(); monkeypatch.setattr(sys,'argv',['m1_concat_train.py','--dest',str(tmp_path),'--stage','score','--max-updates-smoke','1'])
 with pytest.raises(SystemExit) as e:m.main()
 assert e.value.code==2
