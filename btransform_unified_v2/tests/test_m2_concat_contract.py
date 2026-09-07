from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rift_v1/m2_concat_train.py'


def _runner():
    spec = importlib.util.spec_from_file_location('concat_runner', SCRIPT)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(runner)
    return runner


def test_concat_runner_declares_a_distinct_full_e0_cell():
    runner = _runner()
    assert runner.CELL == 'M2-RIFT-R50-D4-CONCAT-E50-RECENCY-V1'
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = runner.DecoderEMA(model, decay=0.9)
    state = runner._checkpoint(1, 1, model, optimizer, ema, torch.device('cpu'), 'manifest', smoke=True, cache_hashes={}, epochs=1)
    assert state['schema'] == 'm2_rift_concat_epoch_checkpoint_v1'
    assert state['identity_interface'] == 'concat'


def test_concat_frontend_full_e0_fold_preserves_baseline_prediction_with_mask():
    from btransform_unified_v1 import adapters
    from btransform_unified_v2 import RiftDecoder
    from btransform_unified_v2.concat_model import RiftConcatDecoder

    torch.manual_seed(123)
    baseline = RiftDecoder('m2', context_bins=50, bias_mode='recency', seed=42, proj_dim=16).eval()
    torch.manual_seed(123)
    concat = RiftConcatDecoder('m2', context_bins=50, bias_mode='recency', seed=42).eval()
    assert concat.frontend.token_in == 16 + 50 + 4
    assert concat._frontend_owner.final_norm is concat.final_norm
    assert concat._frontend_owner.readout is concat.readout
    assert sum(p.numel() for p in concat.parameters()) == sum(p.numel() for p in baseline.parameters()) + 12_000
    shared = [(name, value) for name, value in baseline.named_parameters()
              if name in dict(concat.named_parameters()) and value.shape == dict(concat.named_parameters())[name].shape]
    assert shared and all(torch.equal(value, dict(concat.named_parameters())[name]) for name, value in shared)
    bank = adapters.build_m2_bank('source_train', 'ses-2020-10-19-Run1')
    x = torch.from_numpy(bank.X_store[:2]); valid = torch.ones(2, 50, dtype=torch.bool); valid[1, 0] = False
    with torch.no_grad():
        expected = baseline(x, bank, input_valid_mask=valid)
        actual = concat(x, bank, input_valid_mask=valid)
    assert torch.allclose(expected, actual, rtol=2e-5, atol=2e-5)


def test_concat_score_rejects_smoke_argument(monkeypatch, tmp_path):
    runner = _runner()
    monkeypatch.setattr(sys, 'argv', ['m2_concat_train.py', '--dest', str(tmp_path), '--stage', 'score', '--max-updates-smoke', '1'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
