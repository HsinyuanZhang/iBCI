"""CPU-only authority and freeze contracts for the H1 QueryAge supervisor."""
from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_launcher as launcher
from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_smoke as smoke
from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_train as train


def _identities():
    return {str(epoch): {"sampler_sha256": f"s{epoch}", "keep_sha256": f"k{epoch}", "prefix_sha256": f"p{epoch}"}
            for epoch in range(1, 13)}


def _ready():
    ids = _identities()
    return {arm: {"arm": arm, "physical_gpu": gpu, "shared_init_sha256": "shared", "identities": copy.deepcopy(ids),
                  "g0_parity": {"max_abs_diff": 0.0}, "route_gate_gradient_l1": 1.0}
            for arm, gpu in launcher.ARMS.items()}


def test_validate_ready_requires_exact_paired_twelve_identities_and_gpu_assignment():
    ready = _ready();launcher.validate_ready(ready)
    wrong = _ready();wrong["route"]["physical_gpu"] = 0
    with pytest.raises(RuntimeError, match="readiness"):
        launcher.validate_ready(wrong)
    zero = _ready();zero["flat"]["route_gate_gradient_l1"] = 0.0
    with pytest.raises(RuntimeError, match="readiness"):
        launcher.validate_ready(zero)
    missing = _ready();missing["flat"]["identities"].pop("12")
    with pytest.raises(RuntimeError, match="identity"):
        launcher.validate_ready(missing)
    mismatch = _ready();mismatch["route"]["identities"]["1"]["prefix_sha256"] = "other"
    with pytest.raises(RuntimeError, match="paired"):
        launcher.validate_ready(mismatch)


def _write_formal_worker_artifacts(output: Path, ready: dict):
    (output / "workers").mkdir(parents=True);(output / "checkpoints").mkdir()
    for arm in launcher.ARMS:
        records=[]
        for epoch in range(1, 13):
            checkpoint=output / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt";checkpoint.write_bytes(f"{arm}-{epoch}".encode())
            # Epochs 1 and 2 tie, so the formal stable selector must choose 1.
            value=.9 if epoch in (1,2) else .1 + epoch / 100
            row={"epoch":epoch,"checkpoint":str(checkpoint),"checkpoint_sha256":launcher.sha(checkpoint),
                 "selection":{"n_bins":2908,"r2_concat_float64":value},"identities":ready[arm]["identities"][str(epoch)]}
            (output / "workers" / f"{arm}_epoch_{epoch:03d}.json").write_text(json.dumps(row))
            records.append(row)
        complete={"arm":arm,"shared_init_sha256":ready[arm]["shared_init_sha256"],"identities":ready[arm]["identities"],
                  "epochs":records,"selected_epoch":1,"selected_ema_r2_float64":.9}
        (output / "workers" / f"{arm}_complete.json").write_text(json.dumps(complete))


def test_freeze_selection_validates_every_persisted_epoch_and_stable_tie(tmp_path):
    output=tmp_path/"formal";ready=_ready();_write_formal_worker_artifacts(output,ready)
    freeze=launcher.freeze_selection(output,ready)
    assert freeze["selected"]["flat"]["epoch"]==1 and freeze["epoch12"]["route"]["epoch"]==12
    assert json.loads((output/"selection_freeze.json").read_text())==freeze
    # A single persisted worker row cannot silently diverge from completion.
    changed=json.loads((output/"workers"/"route_epoch_007.json").read_text());changed["selection"]["n_bins"]=7
    (output/"workers"/"route_epoch_007.json").write_text(json.dumps(changed))
    with pytest.raises(RuntimeError,match="epoch checkpoint/selection"):
        launcher.freeze_selection(output,ready)


def test_freeze_selection_rejects_missing_epoch_artifact(tmp_path):
    output=tmp_path/"formal";ready=_ready();_write_formal_worker_artifacts(output,ready)
    (output/"workers"/"flat_epoch_009.json").unlink()
    with pytest.raises(FileNotFoundError):
        launcher.freeze_selection(output,ready)


def test_collect_authorities_are_preconstruction_only_with_mocked_stage_audit(tmp_path,monkeypatch):
    output=(tmp_path/"output").resolve();capacity=(tmp_path/"capacity.json").resolve();checkpoint=(tmp_path/"checkpoint.pt").resolve()
    capacity.write_text("{}");checkpoint.write_bytes(b"checkpoint")
    auths={}
    for arm in launcher.ARMS:
        path=(tmp_path/f"{arm}.auth.json").resolve();path.write_text("{}");auths[arm]={"path":str(path),"sha256":launcher.sha(path)}
    monkeypatch.setattr(launcher,"capacity_audit",lambda *a:{"status":"mocked-readonly-stage-audit"})
    monkeypatch.setattr(smoke,"bindings",lambda output,**kwargs:{"output":str(output),**kwargs})
    monkeypatch.setattr(smoke,"_authorize",lambda binding,path,digest:None)
    monkeypatch.setattr(train,"code_closure",lambda:{"closure":"fixture"})
    smoke_authority=launcher.collect_smoke_authority(output,capacity,checkpoint,auths)
    assert smoke_authority["mode"]=="smoke" and set(smoke_authority["arms"])==set(launcher.ARMS)
    monkeypatch.setattr(train,"collect_bindings",lambda output,**kwargs:{"output":str(output),"fixture":True})
    monkeypatch.setattr(launcher,"audit_smoke",lambda *a:{"receipt":"mocked"})
    formal=launcher.collect_formal_authority(output,capacity,checkpoint,(tmp_path/"smoke.json").resolve())
    assert formal["mode"]=="formal" and formal["bindings"]["fixture"] is True


def test_run_refuses_without_explicit_go_before_external_or_gpu_work(tmp_path,monkeypatch):
    output=(tmp_path/"fresh").resolve();authorization=(tmp_path/"authority.json").resolve();authorization.write_text("{}")
    monkeypatch.delenv(launcher.GO,raising=False)
    with pytest.raises(RuntimeError,match="explicit GO"):
        launcher.run("formal",output,authorization,launcher.sha(authorization),tmp_path/"c.json",tmp_path/"c.pt",tmp_path/"smoke.json")
