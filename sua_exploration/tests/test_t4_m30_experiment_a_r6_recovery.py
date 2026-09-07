from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_r6_scheduler_invokes_runner_via_bash_and_is_versioned():
 s=(ROOT/'scripts/schedule_t4_m30_experiment_a_r6_2gpu.sh').read_text();r=ROOT/'scripts/run_t4_m30_experiment_a_r6_one_cell.sh'
 assert 'bash "$RUNNER"' in s and 'attribution_v6' in s and r.is_file() and r.stat().st_mode & 0o444
def test_r6_auth_and_aggregate_are_r6_bound():
 assert 'v3_r6_20260803' in (ROOT/'scripts/t4_m30_experiment_a_r6_authorization.py').read_text()
 assert 't4_m30_experiment_a_r6_authorization' in (ROOT/'scripts/aggregate_t4_m30_experiment_a_r6.py').read_text()
