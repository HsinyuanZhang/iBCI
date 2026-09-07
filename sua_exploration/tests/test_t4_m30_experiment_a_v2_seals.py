from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_runner_scheduler_and_evaluator_are_sealed_m30():
 r=(ROOT/'scripts/run_t4_m30_experiment_a_one_cell.sh').read_text(); s=(ROOT/'scripts/schedule_t4_m30_experiment_a_2gpu.sh').read_text(); e=(ROOT/'scripts/eval_t4_m30_experiment_a.py').read_text()
 assert '--t4_logit_residual_mode none' in r and '--pool_size' not in r
 assert 'authorization receipt hash mismatch' in r and 'refusing resume/overwrite collision' in r
 assert 'GPU=$((i % 2))' in s and 'gpu_launch_authorized' in s
 assert "'--calibration_n','30','--pool_size','30'" in e and "'--total_epochs','12','--burn_in','4'" in e
def test_plain_b3s_residual_none_uses_standard_module_path():
 source=(ROOT/'scripts/train_variant_dandi688.py').read_text()
 assert 'if args.t4_logit_residual_mode == "none":' in source
 assert 'model = StreamingCalibrationLitModule(' in source
