"""Model-free production-authority refusal tests for the cached B7 comparator."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest
from tfpd_exploration.src.family_runtime_v1 import compare_m2_queryage_cached_spint as c

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _binding(tmp_path):
 out=(tmp_path/"out.json").resolve();proof=(tmp_path/"proof").resolve();proof.mkdir();native=(tmp_path/"native").resolve();native.mkdir();run=(tmp_path/"run").resolve();run.mkdir();final=(tmp_path/"final").resolve();final.mkdir()
 return {"schema":c.SCHEMA,"image":c.IMAGE,"calls":32,"warmup":128,"threads":1,"interop_threads":1,"output":str(out),"run_root":str(run),"finalizer_root":str(final),"finalizer_receipt_sha256":"a"*64,"selected":{"FLAT":{"path":"/x","sha256":"b"*64},"ROUTE":{"path":"/y","sha256":"c"*64}},"proofs":{k:{"path":str(proof/ (k+".json")),"native_authority":{"output":str(native),"finalizer_receipt_sha256":"a"*64}} for k in ("uncached","cached")},"code":{"x":"a"},"trainer_source_train_authority":{"x":"a"},"finalizer_source_minival_authority":{"x":"a"},"finalizer_audit":{"x":"a"},"source_raw_sha256":{"x":"a"},"original":{"x":"a"}}
def _run(binding,tmp_path,monkeypatch):
 auth=(tmp_path/"auth.json").resolve();auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":binding}));monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 return c.run(binding,authorization=auth,authorization_sha=_sha(auth))

@pytest.mark.parametrize("field,value,match",[("proofs",{},"exact selected"),("selected",{},"exact selected"),("code",{},"nonempty code"),("output","relative.json","fresh absolute")])
def test_production_rejects_incomplete_authority_before_model(tmp_path,monkeypatch,field,value,match):
 binding=_binding(tmp_path);binding[field]=value
 with pytest.raises(RuntimeError,match=match): _run(binding,tmp_path,monkeypatch)

def test_production_rejects_native_finalizer_sha_mismatch_before_model(tmp_path,monkeypatch):
 binding=_binding(tmp_path);binding["proofs"]["cached"]["native_authority"]["finalizer_receipt_sha256"]="z"*64
 with pytest.raises(RuntimeError,match="cached production finalizer"): _run(binding,tmp_path,monkeypatch)

def test_injected_wrapper_requires_explicit_fixture_flag(tmp_path,monkeypatch):
 binding=_binding(tmp_path);auth=(tmp_path/"auth.json").resolve();auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":binding}));monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 with pytest.raises(RuntimeError,match="CPU-fixture-only"):
  c.run(binding,authorization=auth,authorization_sha=_sha(auth),wrapper=object())
