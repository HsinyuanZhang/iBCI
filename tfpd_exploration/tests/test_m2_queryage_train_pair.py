"""CPU fixture tests for the prospective QueryAge+prefix pair trainer guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1.contracts import make_stub_batch, make_stub_bank
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders
from tfpd_exploration.src.m2_queryage_family_v1.train_pair import (
    EPOCHS,
    SMOKE_STATUS,
    checkpoint_payload,
    paired_optimizer_step,
    select_earliest_ema,
    validate_smoke_closure_current,
    validate_smoke_receipt,
)


def _smoke(path, *, status=SMOKE_STATUS):
    value = {"schema": "m2_queryage_pair_resource_smoke_v1", "status": status,
             "no_minival_loaded_or_scored": True, "no_checkpoint_selection_or_promotion": True,
             "updates": [{} for _ in range(100)], "max_memory_bytes": 1,
             "forecast_24x3165_pair_updates_seconds_with_50pct_margin_plus_1800_eval_checkpoint": 1.0}
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_bound_passed_smoke_receipt(tmp_path):
    path = tmp_path / "smoke.json"; digest = _smoke(path)
    assert validate_smoke_receipt(path, digest)["status"] == SMOKE_STATUS
    with pytest.raises(RuntimeError, match="absent or changed"):
        validate_smoke_receipt(path, "0" * 64)
    bad = tmp_path / "bad.json"; bad_digest = _smoke(bad, status="RUNNING")
    with pytest.raises(RuntimeError, match="required passed"):
        validate_smoke_receipt(bad, bad_digest)


def test_smoke_closure_requires_identical_prepost_and_current_files(tmp_path):
    frozen = tmp_path / "frozen.py"; frozen.write_text("x=1\n", encoding="utf-8")
    closure = {str(frozen): hashlib.sha256(frozen.read_bytes()).hexdigest()}
    receipt = {"authority_pre": {"closure": closure}, "authority_post": {"closure": dict(closure)}}
    assert validate_smoke_closure_current(receipt) == closure
    frozen.write_text("x=2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        validate_smoke_closure_current(receipt)


def test_earliest_ema_selection_and_contiguity_guard():
    history = {epoch: 0.1 for epoch in range(1, EPOCHS + 1)}
    history[4] = history[9] = 0.8
    assert select_earliest_ema(history) == 4
    history.pop(24)
    with pytest.raises(RuntimeError, match="contiguous"):
        select_earliest_ema(history)


def test_real_tiny_paired_optimizer_ema_and_checkpoint_loop():
    torch.set_num_threads(1)
    flat, route = make_paired_queryage_decoders(42)
    models = {"FLAT": flat, "ROUTE": route}
    optimizers = {key: training.make_optimizer(model.trainable_parameters().items()) for key, model in models.items()}
    emas = {key: DecoderEMA(model, decay=.9995) for key, model in models.items()}
    bank = make_stub_bank(seed=17)
    # The production validator checks that each W50 start is in the backing
    # source-store range; make the synthetic fixture satisfy that real gate.
    bank = replace(bank, X_store=np.zeros((64, 50, 96), dtype=np.float32))
    batch = make_stub_batch(bank, batch_size=2, seed=18)
    row = paired_optimizer_step(batch, models, optimizers, emas, epoch=1, batch_id=0)
    paired_optimizer_step(batch, models, optimizers, emas, epoch=1, batch_id=1)
    assert set(row["loss"]) == {"FLAT", "ROUTE"}
    assert all(torch.isfinite(torch.tensor(value)) for value in row["loss"].values())
    assert all(ema.n_updates == 2 for ema in emas.values())
    payload = checkpoint_payload(flat, optimizers["FLAT"], emas["FLAT"], arm="FLAT", epoch=1,
                                 manifest_digest="fixture", authority={"fixture": True}, protocol_sha256="a" * 64,
                                 recipe={"fixture": True}, updates_per_epoch=2)
    assert payload["schema"] == "m2_queryage_prefix_pair_checkpoint_v1"
    assert payload["ema"]["n_updates"] == 2
    assert set(("raw_state_dict", "ema", "optimizer", "rng")) <= set(payload)
    raw_before_score = {name: value.detach().clone() for name, value in flat.named_parameters()}
    value = emas["FLAT"].score_with_ema(flat, lambda model: float(next(model.parameters()).sum()))
    assert isinstance(value, float)
    assert all(torch.equal(value, dict(flat.named_parameters())[name]) for name, value in raw_before_score.items())
    restored, _ = make_paired_queryage_decoders(42)
    restored_opt = training.make_optimizer(restored.trainable_parameters().items())
    restored_ema = DecoderEMA(restored, decay=.9995)
    restored_state = training.TrainState(cell="")
    training.load_checkpoint(payload, restored, restored_opt, restored_ema, restored_state)
    assert restored_state.global_step == 2 and restored_ema.n_updates == 2
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in flat.state_dict().items())


def test_injectable_two_epoch_cpu_fixture_loop_selects_ema_without_global_constants():
    """A complete tiny loop: real updates, EMA scoring, checkpoint and selection."""
    torch.set_num_threads(1)
    flat, route = make_paired_queryage_decoders(42)
    models = {"FLAT": flat, "ROUTE": route}
    optimizers = {key: training.make_optimizer(model.trainable_parameters().items()) for key, model in models.items()}
    emas = {key: DecoderEMA(model, decay=.9995) for key, model in models.items()}
    bank = replace(make_stub_bank(seed=20), X_store=np.zeros((64, 50, 96), dtype=np.float32))
    batches = [make_stub_batch(bank, batch_size=2, seed=21 + item) for item in range(2)]
    histories = {key: {} for key in models}
    for epoch, batch in enumerate(batches, start=1):
        paired_optimizer_step(batch, models, optimizers, emas, epoch=epoch, batch_id=0)
        for arm in models:
            histories[arm][epoch] = emas[arm].score_with_ema(
                models[arm], lambda current: float(sum(param.detach().abs().mean() for param in current.parameters()))
            )
    assert all(ema.n_updates == 2 for ema in emas.values())
    # Low-epoch injection proves selection does not secretly depend on the
    # production 24-epoch constant.
    assert select_earliest_ema(histories["FLAT"], epochs=2) in {1, 2}


def test_actual_run_cpu_fixture_writes_and_revalidates_complete_artifacts(tmp_path, monkeypatch):
    from tfpd_exploration.src.m2_queryage_family_v1 import train_pair as q
    torch.set_num_threads(1)
    frozen = tmp_path / "source.bin"; frozen.write_bytes(b"fixture source")
    closure = {str(frozen): q.sha(frozen)}
    smoke_path = tmp_path / "smoke.json"; _smoke(smoke_path)
    smoke = json.loads(smoke_path.read_text())
    smoke.update(authority_pre={"closure": closure}, authority_post={"closure": closure})
    smoke_path.write_text(json.dumps(smoke)); smoke_sha = q.sha(smoke_path)
    bank = replace(make_stub_bank(seed=30), X_store=np.zeros((100, 96), dtype=np.float32))
    batches = [make_stub_batch(bank, batch_size=2, seed=31+i) for i in range(2)]
    monkeypatch.setattr(q, "EPOCHS", 2); monkeypatch.setattr(q, "UPDATES_PER_EPOCH", 2)
    monkeypatch.setattr(q, "MANIFEST", frozen)
    monkeypatch.setattr(q, "_launch_gate", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(q, "_check_budget", lambda *args: None)
    monkeypatch.setattr(q, "_closure_paths", lambda: [frozen])
    monkeypatch.setattr(q, "preflight_only", lambda **kwargs: {"closure": dict(closure), "smoke_closure": dict(closure)})
    monkeypatch.setattr(q.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(q.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(q.plan, "HELDIN_SESSIONS", (bank.session_id,))
    monkeypatch.setattr(q.data, "load_session_bank", lambda *args, **kwargs: bank)
    monkeypatch.setattr(q.sampler, "load_manifest", lambda *args: {"digest": "fixture"})
    monkeypatch.setattr(q.source_training, "count_updates", lambda *args: 2)
    monkeypatch.setattr(q.source_training, "epoch_batches_shuffled", lambda *args, **kwargs: iter(batches))
    original_select = q.select_earliest_ema
    monkeypatch.setattr(q, "select_earliest_ema", lambda values: original_select(values, epochs=2))
    score_calls = []
    def fixture_score(model, banks, device, budget_check=None):
        if budget_check: budget_check()
        model.eval()
        with torch.inference_mode():
            value = float(model.forward_last(batches[0].X, bank, batches[0].unit_mask).square().mean())
        score_calls.append(value)
        return -value
    monkeypatch.setattr(q, "score_equal_session", fixture_score)
    out = tmp_path / "run"
    result = q.run(out, physical_gpu=1, smoke_receipt=smoke_path, smoke_sha256=smoke_sha)
    assert result["status"] == "SOURCE_MINIVAL_SELECTION_ONLY_COMPLETE" and len(score_calls) == 8
    assert result["authority_pre"] == result["authority_post"]
    complete = json.loads((out / "completion.json").read_text())
    assert complete["selection_summary_sha256"] == q.sha(out / "selection_summary.json")
    for arm in q.ARMS:
        endpoint = torch.load(out / arm / "epoch_002.pt", map_location="cpu", weights_only=False)
        assert endpoint["global_step"] == endpoint["ema"]["n_updates"] == 4
        assert endpoint["actual_epoch_batches"] == 2
    q.verify_epoch_artifacts(out, result["history"], q.sha(out / "protocol_preconstruction.json"))
    (out / "FLAT" / "epoch_001_receipt.json").write_text("{}")
    with pytest.raises(RuntimeError, match="sealed receipt"):
        q.verify_epoch_artifacts(out, result["history"], q.sha(out / "protocol_preconstruction.json"))
