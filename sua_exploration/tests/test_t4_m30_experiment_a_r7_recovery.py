from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_r7_requires_managed_foreground_and_rejects_nohup_contractually():
 s=(ROOT/'scripts/schedule_t4_m30_experiment_a_r7_2gpu.sh').read_text();r=(ROOT/'scripts/run_t4_m30_experiment_a_r7_one_cell.sh').read_text()
 assert 'managed foreground exec session' in s and 'nohup' in s and 'attribution_v7' in r
