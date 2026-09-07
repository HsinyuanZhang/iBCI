import ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_r9_recursive_aggregate_dependency_closure_is_sealed():
 paths=['sua_exploration/scripts/aggregate_t4_m30_experiment_a_r9.py','sua_exploration/scripts/aggregate_t4_m30_experiment_a_r5.py','sua_exploration/scripts/aggregate_t4_m30_experiment_a_v3.py','sua_exploration/scripts/t4_m30_experiment_a_r9_authorization.py','sua_exploration/scripts/t4_m30_experiment_a_r4_authorization.py','sua_exploration/scripts/t4_m30_experiment_a_r5_authorization.py']
 r9=(ROOT/'scripts/aggregate_t4_m30_experiment_a_r9.py').read_text();assert 'aggregate_t4_m30_experiment_a_r5' in r9
 r5=(ROOT/'scripts/aggregate_t4_m30_experiment_a_r5.py').read_text();assert 'aggregate_t4_m30_experiment_a_v3' in r5
 assert all((ROOT.parent/p).is_file() for p in paths)
