from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

_BENCH = Path(__file__).resolve().parents[2]
_ROOT, _WORKSPACE = _BENCH.parent, _BENCH.parents[1]
for _path in (_WORKSPACE, _ROOT, _ROOT / "src", _ROOT / "learnable_recency_v1" / "src",
              _WORKSPACE / "btransform_unified_v1" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dandi688_bench_v2 import protocol
from dandi688_bench_v2.model import B3SIdentityEncoder, DandiRiftDecoder
from dandi688_bench_v2.flat_extension.contract import FLAT_RECIPE, FLAT_SCHEMA, flat_config_metadata
from dandi688_bench_v2.flat_extension.model import make_model, validate_flat_model
from dandi688_bench_v2.flat_extension.training import load_trained_model
from dandi688_bench_v2.flat_extension.contract import flat_training_source_hashes


def _payload(model: torch.nn.Module) -> dict:
    proof = validate_flat_model(model)
    return {"schema": FLAT_SCHEMA + "_checkpoint", "status": "FORMAL", "stage": "train",
            "arm": model.arm, "representation": "sua", "seed": 42, "global_step": 2,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "source_stats": {"representation": "sua"}, "protocol": protocol.protocol_dict(),
            "recipe": FLAT_RECIPE, "flat_recipe": FLAT_RECIPE, "flat_config": flat_config_metadata(),
            "zero_slope_proof": proof, "encoder_checkpoint_sha256": None, "base_encoder_binding": None}


def test_flat_has_exact_zero_bias_and_no_trainable_recency_parameters():
    model = make_model("raw_set", seed=42, n_pad=4)
    proof = validate_flat_model(model)
    assert proof["slopes_shape"] == [4, 8]
    assert proof["slopes_zero_count"] == 32
    assert proof["learnable_recency_parameters"] == []
    assert not any("slope_log" in name for name, _ in model.named_parameters())


def test_flat_and_learned_same_seed_share_non_temporal_parameters():
    flat = make_model("raw_set", seed=42, n_pad=4)
    learned = DandiRiftDecoder("raw_set", seed=42, n_pad=4)
    left, right = flat.state_dict(), learned.state_dict()
    names = sorted(set(left) & set(right) - {"temporal.recency_slopes"})
    assert names
    assert all(torch.equal(left[name], right[name]) for name in names)


def test_full_encoder_is_base_encoder_and_frozen_with_exact_parameter_count():
    encoder = B3SIdentityEncoder(side_dim=4, seed=42)
    model = make_model("full", seed=42, n_pad=4, encoder=encoder, freeze_encoder=True)
    proof = validate_flat_model(model, require_frozen_full=True)
    assert proof["full_encoder_parameters"] == 18290
    assert proof["full_encoder_frozen"] is True
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())


def test_flat_checkpoint_reload_has_identical_prediction_and_rejects_recipe_drift(tmp_path):
    # Formal checkpoints always use the fixed M33/100-pad geometry, which the
    # strict loader reconstructs rather than trusting caller-provided shape.
    model = make_model("raw_set", seed=42).eval()
    path = tmp_path / "segment_01.pt"
    torch.save(_payload(model), path)
    restored, _ = load_trained_model(path)
    x = torch.randn(2, 50, 100)
    mask = torch.ones(2, 50, dtype=torch.bool)
    units = torch.ones(2, 100, dtype=torch.bool)
    with torch.inference_mode():
        assert torch.equal(model(x, unit_mask=units, input_valid_mask=mask),
                           restored(x, unit_mask=units, input_valid_mask=mask))
    broken = _payload(model)
    broken["recipe"] = {**FLAT_RECIPE, "tier": "learned_slope"}
    bad = tmp_path / "bad.pt"
    torch.save(broken, bad)
    with pytest.raises(ValueError, match="recipe"):
        load_trained_model(bad)


def test_full_checkpoint_reloads_with_frozen_base_encoder(tmp_path):
    model = make_model("full", seed=42, encoder=B3SIdentityEncoder(side_dim=4, seed=42)).eval()
    path = tmp_path / "full.pt"
    torch.save(_payload(model), path)
    restored, _ = load_trained_model(path)
    assert restored.arm == "full"
    assert not any(parameter.requires_grad for parameter in restored.encoder.parameters())
    x, activity, carrier = torch.randn(1, 50, 100), torch.randn(33, 100, 100), torch.randn(100, 4)
    valid, units = torch.ones(1, 50, dtype=torch.bool), torch.ones(1, 100, dtype=torch.bool)
    with torch.inference_mode():
        assert torch.equal(model(x, activity=activity, carrier=carrier, unit_mask=units, input_valid_mask=valid),
                           restored(x, activity=activity, carrier=carrier, unit_mask=units, input_valid_mask=valid))


def test_training_source_hashes_are_workspace_relative_and_current():
    hashes = flat_training_source_hashes()
    root = protocol.WORKSPACE_ROOT
    assert len(hashes) == 74
    assert all((root / relative).is_file() for relative in hashes)
    assert all(__import__("hashlib").sha256((root / relative).read_bytes()).hexdigest() == digest
               for relative, digest in hashes.items())
