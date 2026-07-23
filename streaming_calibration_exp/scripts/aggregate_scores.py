"""Aggregate results from all training runs into a single comparison table.

Reads metrics_summary.csv from each run directory under outputs/streaming_calibration/
and produces a unified scoring table with R², hardware cost, and Pareto analysis.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunResult:
  run_id: str
  variant: str
  r2_heldin: float
  r2_heldout: Optional[float]
  identity_mse: float
  prediction_distill_mse: float
  parameter_count: int
  mac_per_session: int
  peak_state_bytes: int
  trial_buffer_bytes: int
  requires_cubic_interp: bool
  requires_general_multiplier: bool
  multiplier_free_prepool: bool
  dropout_mode: str
  ckpt_path: str


def parse_run(run_dir: Path) -> Optional[RunResult]:
  """Parse a single run directory."""
  summary_path = run_dir / "metrics_summary.csv"
  hw_path = run_dir / "hardware_cost.json"
  cfg_path = run_dir / "resolved_config.yaml"
  ckpt_manifest_path = run_dir / "checkpoint_manifest.json"

  if not summary_path.exists():
    return None

  # Parse metrics
  r2_heldin = None
  identity_mse = float("nan")
  pred_distill_mse = float("nan")
  with summary_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      if row.get("split") == "test_heldin":
        try:
          r2_heldin = float(row["R2_variance_weighted"])
          identity_mse = float(row.get("identity_mse", "nan"))
          pred_distill_mse = float(row.get("prediction_distill_mse", "nan"))
        except (ValueError, KeyError):
          pass
        break

  if r2_heldin is None:
    return None

  # Parse hardware cost
  hw = {}
  if hw_path.exists():
    with hw_path.open() as f:
      hw = json.load(f)

  # Parse config for dropout info
  dropout_mode = "none"
  if cfg_path.exists():
    with cfg_path.open() as f:
      for line in f:
        if "neuron_dropout_mode" in line and ":" in line:
          dropout_mode = line.split(":", 1)[1].strip().strip('"').strip("'")
          break

  # Parse checkpoint path
  ckpt_path = ""
  if ckpt_manifest_path.exists():
    with ckpt_manifest_path.open() as f:
      cm = json.load(f)
      ckpt_path = cm.get("copied_checkpoint_path", cm.get("original_path", ""))

  return RunResult(
    run_id=run_dir.name,
    variant=hw.get("variant", run_dir.name.split("_")[0]),
    r2_heldin=float(r2_heldin),
    r2_heldout=None,
    identity_mse=float(identity_mse),
    prediction_distill_mse=float(pred_distill_mse),
    parameter_count=int(hw.get("parameter_count", 0)),
    mac_per_session=int(hw.get("mac_per_session", 0)),
    peak_state_bytes=int(hw.get("peak_live_state_bytes", 0)),
    trial_buffer_bytes=int(hw.get("trial_buffer_bytes", 0)),
    requires_cubic_interp=bool(hw.get("requires_cubic_interpolation", True)),
    requires_general_multiplier=bool(hw.get("requires_general_multiplier", True)),
    multiplier_free_prepool=bool(hw.get("multiplier_free_prepool", False)),
    dropout_mode=dropout_mode,
    ckpt_path=ckpt_path,
  )


def collect_all_runs(base_dir: Path) -> List[RunResult]:
  """Collect all runs from the output directory."""
  results = []
  if not base_dir.exists():
    return results
  for run_dir in sorted(base_dir.iterdir()):
    if not run_dir.is_dir():
      continue
    result = parse_run(run_dir)
    if result is not None:
      results.append(result)
  return results


def compute_scores(results: List[RunResult]) -> List[Dict]:
  """Compute composite scores for Pareto analysis.

  Scoring philosophy:
  - Accuracy (R² heldin) is primary — must be within 5% of best
  - Hardware cost is secondary — prefer lower MAC, params, state
  - Hardware friendliness flags add bonus
  """
  if not results:
    return []

  best_r2 = max(r.r2_heldin for r in results)

  scored = []
  for r in results:
    # Accuracy gap (lower is better)
    acc_gap = best_r2 - r.r2_heldin
    # Normalized accuracy score [0, 1]
    acc_score = max(0.0, 1.0 - acc_gap / 0.10)  # 10% gap -> 0 score

    # Hardware efficiency (log-scale, lower MAC = higher score)
    import math
    mac_score = 1.0 - (math.log10(max(r.mac_per_session, 1)) - 3) / 7  # 1K MAC=1, 10M MAC=0
    mac_score = max(0.0, min(1.0, mac_score))

    # Param efficiency
    param_score = 1.0 - (math.log10(max(r.parameter_count, 1)) - 3) / 5
    param_score = max(0.0, min(1.0, param_score))

    # State efficiency
    state_kb = r.peak_state_bytes / 1024
    state_score = max(0.0, 1.0 - state_kb / 64)  # 64 KiB cap

    # Hardware friendliness bonus
    hw_bonus = 0.0
    if not r.requires_cubic_interp:
      hw_bonus += 0.1
    if not r.requires_general_multiplier:
      hw_bonus += 0.15
    if r.multiplier_free_prepool:
      hw_bonus += 0.05
    if r.trial_buffer_bytes == 0:
      hw_bonus += 0.1

    # Composite (accuracy-weighted)
    composite = 0.50 * acc_score + 0.20 * mac_score + 0.15 * param_score + 0.15 * state_score + hw_bonus

    scored.append({
      **asdict(r),
      "acc_gap_vs_best": round(acc_gap, 4),
      "acc_score": round(acc_score, 3),
      "mac_score": round(mac_score, 3),
      "param_score": round(param_score, 3),
      "state_score": round(state_score, 3),
      "hw_bonus": round(hw_bonus, 3),
      "composite_score": round(composite, 3),
    })

  scored.sort(key=lambda x: x["composite_score"], reverse=True)
  return scored


def format_table(scored: List[Dict]) -> str:
  if not scored:
    return "No results."

  header = (
    f"{'rank':<5} {'variant':<14} {'drop':<10} {'R²':>7} {'gap':>7} "
    f"{'params':>8} {'MAC/sess':>11} {'state_KB':>9} "
    f"{'acc':>5} {'mac':>5} {'par':>5} {'st':>5} {'hw+':>5} {'COMP':>6}"
  )
  lines = [header, "-" * len(header)]
  for i, s in enumerate(scored, 1):
    lines.append(
      f"{i:<5} {s['variant']:<14} {s['dropout_mode']:<10} "
      f"{s['r2_heldin']:>7.4f} {s['acc_gap_vs_best']:>+7.4f} "
      f"{s['parameter_count']:>8,} {s['mac_per_session']:>11,} "
      f"{s['peak_state_bytes']/1024:>9.1f} "
      f"{s['acc_score']:>5.2f} {s['mac_score']:>5.2f} {s['param_score']:>5.2f} "
      f"{s['state_score']:>5.2f} {s['hw_bonus']:>5.2f} {s['composite_score']:>6.3f}"
    )
  return "\n".join(lines)


def main():
  base_dir = Path("outputs/streaming_calibration")
  results = collect_all_runs(base_dir)
  if not results:
    print("No results found.")
    return 1

  print(f"Collected {len(results)} runs from {base_dir}")
  print()

  scored = compute_scores(results)
  print(format_table(scored))

  # Write CSV
  out_path = base_dir / "scoring_summary.csv"
  if scored:
    fieldnames = list(scored[0].keys())
    with out_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(scored)
    print(f"\nWrote {out_path}")

  # Write JSON
  out_json = base_dir / "scoring_summary.json"
  with out_json.open("w") as f:
    json.dump(scored, f, indent=2, default=str)
  print(f"Wrote {out_json}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
