from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_r5_authorization_is_versioned_and_fixed_path() -> None:
    source=(ROOT/'scripts/t4_m30_experiment_a_r5_authorization.py').read_text()
    assert 'v3_r5_20260802' in source and 'attribution_v5' in source
