from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_r8_fixed_paths_and_managed_foreground_scheduler_contract():
 s=(ROOT/'scripts/schedule_t4_m30_experiment_a_r8_2gpu.sh').read_text();r=(ROOT/'scripts/run_t4_m30_experiment_a_r8_one_cell.sh').read_text();a=(ROOT/'scripts/t4_m30_experiment_a_r8_authorization.py').read_text()
 assert 'bash "$RUNNER"' in s and 'managed foreground' in s and 'nohup' in s and 'cell_status' in s and 'aggregate_t4_m30_experiment_a_r8.py' in s
 assert 'verify_t4_m30_experiment_a_r8_authorization.py' in r and 'attribution_v8' in r and 't4_m30_experiment_a_v8' in r
 assert 'v3_r8_20260803' in a and 'attribution_v8' in a
