"""Hydra/static contract for RT-LD fit-end selection receipts."""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def test_a0_callback_is_inherited_by_all_three_arms_and_binds_output_artifacts() -> None:
  experiment = ROOT / "configs/experiment"
  a0 = yaml.safe_load((experiment / "rt_ld_a0_full_m24_fold0_seed42.yaml").read_text())
  callback = a0["callbacks"]["rt_nested_selection_receipt"]
  assert callback["_target_"] == "src.callbacks.rt_nested_selection_receipt.RtNestedSelectionReceipt"
  assert callback["output_path"] == "${paths.output_dir}/rt_nested_selection_receipt.json"
  assert callback["split_manifest_path"] == "${paths.output_dir}/split_manifest.json"
  assert callback["config_path"] == "${paths.output_dir}/.hydra/config.yaml"
  assert callback["monitor"] == "val_heldin/r2_mean"
  assert callback["outer_loso_fold"] == "${data.outer_loso_fold}" and callback["seed"] == "${seed}"
  for name, expected_arm, expected_gain in (
    ("rt_ld_a0_full_m24_fold0_seed42.yaml", "a0", "full"),
    ("rt_ld_g_full_m24_fold0_seed42.yaml", "g_full", "full"),
    ("rt_ld_g_xls_m24_fold0_seed42.yaml", "g_xls", "xls_v2"),
  ):
    text = (experiment / name).read_text()
    assert expected_arm in text and expected_gain in text
