from pathlib import Path
R=Path(__file__).resolve().parents[2]
def test_zero4_is_width_matched_and_target_free():
 d=(R/'streaming_calibration_exp/src/data/falcon_datamodule.py').read_text()
 assert "'zero4'" in d and 'np.zeros((calib_trialized_neural_features.shape[-1], 4)' in d
def test_three_joint_configs_and_source_runner_contract():
 for arm in ('zero4','t4','ts4'):
  x=(R/f'streaming_calibration_exp/configs/experiment/m2_joint_upperbound_{arm}_m24_loso.yaml').read_text()
  assert 'freeze_decoder: false' in x and 'calibration_n_trials: 24' in x and 'max_epochs: 12' in x
 x=(R/'sua_exploration/scripts/run_m2_joint_t4_upperbound_one_arm.sh').read_text()
 assert 'validate_m2_clean_teacher_receipt.py' in x and 'refusing existing artifact' in x
def test_source_audit_requires_b3s4_joint_and_same_teacher():
 x=(R/'sua_exploration/scripts/audit_m2_joint_t4_upperbound_source.py').read_text()
 assert "('B3S',4,False,True)" in x and 'teacher init differs across zero4/T4/TS4' in x
def test_heldout_runner_is_locked_by_independent_go_and_disjoint_window():
 x=(R/'sua_exploration/scripts/run_m2_joint_t4_upperbound_heldout_one_arm.sh').read_text()
 assert 'independent protocol GO is absent' in x and 'source_receipt_sha256' in x
 assert 'train=false test=true' in x and 'data.query_start_trial=24' in x
