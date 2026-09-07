import numpy as np, pytest, torch
from tfpd_exploration.src.family_runtime_v1.m2 import FiveTokenM2Decoder
from tfpd_exploration.src.family_runtime_v1.m2_lifted import LiftedFiveTokenM2Decoder, _LiftedFiveTokenExactE
from tfpd_exploration.tests.test_m2_runtime_v3 import TAGS
P='tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl'
@pytest.mark.parametrize('batch',[1,7])
def test_lifted_m2_boundary_reset_and_full_native_parity(batch):
 torch.set_num_threads(1); a=FiveTokenM2Decoder(P,batch_size=batch); b=LiftedFiveTokenM2Decoder(P,batch_size=batch); a.reset(TAGS[:batch]);b.reset(TAGS[:batch]); x=np.random.default_rng(8601).poisson(.3,(55,batch,96)).astype('float32')
 for i,v in enumerate(x):
  got=b.predict(v); ref=a.predict(v); np.testing.assert_allclose(got,ref,atol=1e-5,rtol=1e-5)
  if i in (0,3,4,49,50,54):
   full=b.model.forward_last(b._engine.raw,b._engine.bank,b._engine.bank.unit_mask).numpy()/5
   np.testing.assert_allclose(got,full,atol=1e-5,rtol=1e-5)
 b.reset(TAGS[:1]); assert b.predict(x[0,:1]).shape==(batch,2)
def _oracle(engine): return engine.model.forward_last(engine.raw,engine.bank,engine.bank.unit_mask).numpy()/5
def test_lifted_m2_model_bank_mask_mutations_rebuild_numeric_oracle():
 b=LiftedFiveTokenM2Decoder(P,batch_size=1); b.reset(TAGS[:1]); x=np.zeros((1,96),np.float32); b.predict(x)
 with torch.no_grad():
  b.model.frontend.mha.in_proj_weight[:512].add_(.001); b.model.frontend.slots.add_(.002); b.model.frontend.slot_norm.weight.mul_(.999)
 got=b.predict(x); np.testing.assert_allclose(got,_oracle(b._engine),atol=1e-5,rtol=1e-5)
 fresh=_LiftedFiveTokenExactE(b.model,b._engine.bank); fresh.rebuild(b._engine.raw.clone()); np.testing.assert_allclose(got,fresh._last(fresh.frontend).numpy()/5,atol=1e-5,rtol=1e-5)
 b._engine.bank.E0.add_(.01); b._engine.bank.unit_mask[0,0].logical_not_(); got=b.predict(x); np.testing.assert_allclose(got,_oracle(b._engine),atol=1e-5,rtol=1e-5)
 b._engine._lifted.qwk.add_(.01)
 with pytest.raises(Exception,match='derived'): b.predict(x)
 b.model.train()
 with pytest.raises(Exception,match='eval'): b.predict(x)

def test_lifted_m2_on_done_partial_batch_and_owned_state_bytes():
 a=FiveTokenM2Decoder(P,batch_size=7); b=LiftedFiveTokenM2Decoder(P,batch_size=7); a.reset(TAGS[:3]);b.reset(TAGS[:3]); x=np.zeros((3,96),np.float32)
 before=b._engine.raw.clone(); b.on_done(np.array([True,False,True])); assert torch.equal(before,b._engine.raw)
 np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-5,rtol=1e-5); assert b.predict(x).shape==(7,2)
 assert b.state_bytes().static_active_projection_bytes > a.state_bytes().static_active_projection_bytes
