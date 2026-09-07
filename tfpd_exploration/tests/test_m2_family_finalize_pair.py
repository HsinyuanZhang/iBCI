import hashlib
import json
from pathlib import Path

import pytest
import torch
import numpy as np

from tfpd_exploration.src.m2_family_v1 import finalize_pair as f


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "pair"; scores = {}
    for arm in f.MEMBERS:
        raw = {str(e): float(e) / 100 for e in range(1, 25)}
        ema = {str(e): float(e) / 100 for e in range(1, 25)}
        ema["7"] = .99  # prove selected epoch need not be endpoint epoch24
        scores[arm] = {"RAW": raw, "EMA": ema}
        d = root / arm; d.mkdir(parents=True)
        for epoch in range(1,25):
            receipt = {"arm": arm, "epoch": epoch, "raw_equal_session_r2": raw[str(epoch)], "ema_equal_session_r2": ema[str(epoch)]}
            p = d / f"epoch_{epoch:03d}_metrics.json"; p.write_text(json.dumps(receipt))
            p.with_suffix(".json.sha256").write_text(f"{_sha(p)}  {p.name}\n")
        (d / "epoch_024.pt").write_bytes(b"fixture")
        (d / "epoch_007.pt").write_bytes(b"fixture-selected")
    (root / "summary.json").write_text(json.dumps({"status": "SOURCE_ONLY_COMPLETE", "scores": scores}))
    return root


def test_summary_selects_independent_earliest_ema_argmax(tmp_path):
    audit = f.validate_summary(_fixture(tmp_path))
    assert audit["picks"] == {"FLAT": 7, "ROUTE": 7}


def test_summary_refuses_incomplete_history(tmp_path):
    root = _fixture(tmp_path); payload = json.loads((root / "summary.json").read_text()); del payload["scores"]["FLAT"]["EMA"]["23"]
    (root / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="contiguous"):
        f.validate_summary(root)


def test_summary_refuses_receipt_tamper(tmp_path):
    root = _fixture(tmp_path); p = root / "ROUTE" / "epoch_007_metrics.json"; p.write_text(p.read_text().replace("0.99", "0.98", 1))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        f.validate_summary(root)


def test_checkpoint_contract_allows_parameter_shadow_but_preserves_buffer():
    class Fake(torch.nn.Module):
        def __init__(self): super().__init__(); self.x=torch.nn.Parameter(torch.ones(2)); self.register_buffer("b",torch.zeros(1))
        def trainable_parameters(self): return {"x":self.x}
    m=Fake(); good = {"schema":"m2_b_small_stability_v1_ckpt","cell":"CRST_B4_FLAT","epoch":7,"global_step":22155,"seed":42,"manifest_digest":"m","ema": {"n_updates":22155, "decay": .9995, "shadow": {"x": torch.ones(2)}}, "raw_state_dict": m.state_dict()}
    assert f.checkpoint_contract(good,m,"FLAT",7,"m")["ema_n_updates"] == 22155
    good["ema"]["n_updates"] = 1
    with pytest.raises(RuntimeError, match="updates"):
        f.checkpoint_contract(good,m,"FLAT",7,"m")

def test_fake_full_run_four_artifacts_and_score_refusal(tmp_path, monkeypatch):
    """No query opens: mock every model/data helper while exercising run's order."""
    out=tmp_path/"out"; pair=tmp_path/"pair"; pair.mkdir(); (pair/"pretrain_manifest.json").write_text(json.dumps({"schema":f.config.SCHEMA,"updates_per_epoch":3165,"manifest_digest":"m"}))
    baseline=tmp_path/"baseline.json"; rows=[{"session":f"s{i}","window_count":1,"ordered_window_starts_sha256":str(i),"target_sha256":str(i)} for i in range(7)]
    baseline.write_text(json.dumps({"rows":rows,"spint_equal_session_r2":.8,"spint_pooled_r2":.8,"e8_equal_session_r2":.1,"e8_pooled_r2":.1}))
    summary={"scores":{a:{"EMA":{e:.2 for e in range(1,25)}} for a in f.MEMBERS}}
    for a in f.MEMBERS: summary["scores"][a]["EMA"][7]=.5; summary["scores"][a]["EMA"][24]=.5
    audit={"picks":{"FLAT":7,"ROUTE":7},"summary":summary,"summary_path":"x","summary_sha256":"x","selected_receipts":{}}
    monkeypatch.setattr(f,"PAIR_ROOT",pair); monkeypatch.setattr(f,"OUT_ROOT",out); monkeypatch.setattr(f,"BASELINE",baseline); monkeypatch.setattr(f,"validate_summary",lambda:audit)
    monkeypatch.setattr(f,"_code_and_source_authority",lambda:{"same":1}); monkeypatch.setattr(f,"sha",lambda p:"0646d63801230cdc467c9ac9a4a1441b7380d43ea3a9fdf9a37e4cdd63fbcb46" if Path(p)==baseline else "h")
    monkeypatch.setattr(f,"_verify_pretrain_recipe",lambda p:{"fixture":True})
    class Toy(torch.nn.Module):
        routed=False
        def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.ones(1)); self.register_buffer("b",torch.zeros(1))
        def trainable_parameters(self): return {"p":self.p}
    monkeypatch.setattr(f,"make_paired_decoders",lambda seed:(Toy(),Toy()))
    monkeypatch.setattr(f.torch,"load",lambda *a,**k:{})
    monkeypatch.setattr(f,"checkpoint_contract",lambda *a,**k:{"ok":True})
    monkeypatch.setattr(f,"export_ema_strict",lambda payload,model,path: (path.parent.mkdir(parents=True,exist_ok=True),path.write_bytes(b"x"),{"export_path":str(path),"export_sha256":"h","strict_reload":True})[-1])
    monkeypatch.setattr(Toy,"load_state_dict",lambda self,*a,**k: torch.nn.modules.module._IncompatibleKeys([],[]))
    seen=[]
    def scorer(model,dev):
        assert (out/"selection_freeze.json").is_file(); seen.append(1)
        return ({"rows":rows,"equal_session_r2":.5,"pooled_r2":.5},{"prediction":np.zeros((1,2)),"target":np.zeros((1,2)),"start":np.zeros(1),"session":np.array(["s"])})
    monkeypatch.setattr(f,"score_source_minival",scorer); monkeypatch.setenv(f.AUTH,"1")
    result=f.run(device="cpu",allow_cpu_for_test=True)
    assert len(seen)==4 and len(result["exports"])==4 and (out/"receipt.json").is_file()
    assert result["status"] == "SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION"
    assert result["exposure_qualification"]["same_training_recipe_comparison"] is False
    def bad(model,dev):
        got,archive=scorer(model,dev); got["equal_session_r2"]=.0; return got,archive
    monkeypatch.setattr(f,"score_source_minival",bad); monkeypatch.setattr(f,"OUT_ROOT",tmp_path/"bad")
    with pytest.raises(RuntimeError,match="reproduction"):
        f.run(device="cpu",allow_cpu_for_test=True)


def test_real_ema_export_overlays_parameters_and_preserves_raw_buffer(tmp_path, monkeypatch):
    class Toy(torch.nn.Module):
        def __init__(self, *, routed=False, seed=42):
            super().__init__(); self.routed=routed
            self.weight=torch.nn.Parameter(torch.full((2,), -3.))
            self.register_buffer("buffer", torch.full((3,), -4.))
        def trainable_parameters(self): return {"weight": self.weight}
    monkeypatch.setattr(f, "M2FamilyDecoder", Toy)
    payload={"raw_state_dict":{"weight":torch.full((2,), 2.), "buffer":torch.full((3,), 7.)},
             "ema":{"shadow":{"weight":torch.full((2,), 5.)}}}
    path=tmp_path/"plain.pt"
    record=f.export_ema_strict(payload, Toy(), path)
    got=torch.load(path, map_location="cpu", weights_only=True)
    assert record["strict_reload"]
    assert torch.equal(got["weight"], torch.full((2,), 5.))
    assert torch.equal(got["buffer"], torch.full((3,), 7.))


def test_pretrain_recipe_requires_the_original_bound_inputs(monkeypatch):
    monkeypatch.setattr(f, "sha", lambda path: "unchanged")
    monkeypatch.setattr(f, "_read_json", lambda path: {"e0_sha256":"e", "t4_sha256":"t"})
    names = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "T.npy", "e0_u.pt", "mapping.json", "provenance.json")
    rows = {s: {**{n:"unchanged" for n in names}, "provenance_e0_sha256":"e", "provenance_t4_sha256":"t"} for s in f.plan.HELDIN_SESSIONS}
    bound = {"code_sha256":{n:"unchanged" for n in ("config.py","decoder.py","routing.py","launch.py")},
             "sampler_sha256":"unchanged", "source_cache_sha256":rows}
    assert f._verify_pretrain_recipe({"recipe_hashes":bound}) == bound
    bound["sampler_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="pre-update-one"):
        f._verify_pretrain_recipe({"recipe_hashes":bound})
