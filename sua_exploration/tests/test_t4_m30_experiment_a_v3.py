from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('agg',ROOT/'scripts/aggregate_t4_m30_experiment_a_v3.py')
agg=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(agg)

def test_synthetic_aggregate_statistics_preserve_session_outer_bootstrap() -> None:
    delta=np.full((3,8,6),.04,dtype=float)
    stats=agg.summarize(delta,np.random.default_rng(1))
    assert np.isclose(stats['mean_delta'], .04)
    assert stats['positive_seed_means']==3 and stats['positive_session_means']==6
    assert 'sessions outer' in stats['hierarchical_bootstrap']['resampling']
    assert stats['exact_paired_wilcoxon_sessions']['p_two_sided'] <= 1.0
    assert agg.decide(stats)=='effective'

def test_synthetic_wilcoxon_all_zero_is_defined_and_no_nan_json_contract() -> None:
    result=agg._wilcoxon_exact(np.zeros(6))
    assert result['p_two_sided']==1.0 and result['n_nonzero']==0

def test_v3_runner_and_aggregate_bind_source_cost_and_authorization_contracts() -> None:
    runner=(ROOT/'scripts/run_t4_m30_experiment_a_one_cell.sh').read_text()
    scheduler=(ROOT/'scripts/schedule_t4_m30_experiment_a_2gpu.sh').read_text()
    aggregate=(ROOT/'scripts/aggregate_t4_m30_experiment_a_v3.py').read_text()
    verifier=(ROOT/'scripts/verify_t4_m30_experiment_a_v3_authorization.py').read_text()
    train=(ROOT/'scripts/train_variant_dandi688.py').read_text()
    assert '--num_workers 4' in runner and '--cache_dir' in runner
    assert 'PRELAUNCH_RECEIPT' in runner and 'prelaunch_source_sha256' in verifier
    assert 'v3 source drift' in verifier and 'post_run_cost_receipt.json' in aggregate
    assert 'cuda_peak_memory_allocated_bytes' in train and 'fit_wall_clock_seconds' in train
    assert '--prelaunch "$PRELAUNCH_RECEIPT"' in scheduler

def test_v3_writer_has_exact_static_cost_and_frozen_hash_gates() -> None:
    writer=(ROOT/'scripts/write_t4_m30_experiment_a_prelaunch_v3.py').read_text()
    assert 'SideFeatureEarlyPoolEncoder' in writer and 'decoder_cost_comparison_receipt' in writer
    assert 'd78097203d5c46b26b0ad251164607dadb78886230aea23f9ba55230148b240e' in writer
    assert 'a7ee643d066bd3db9ce6e2b178527bb6655d90c485fb5a0151e5babb8fe255ad' in writer
    assert 'ordinary_t4_fit_operation_proxy_upper_bound_formula' in writer
    assert 'second shuffled-label fit' in writer
