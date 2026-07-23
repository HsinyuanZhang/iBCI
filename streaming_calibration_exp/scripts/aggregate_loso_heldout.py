"""Aggregate LOSO + held-out results into a credible scoring table.

Reads metrics_per_session.csv from each run and computes:
- R² held-in (LOSO held-out session, the proper within-distribution test)
- R² held-out mean (6 fully unseen sessions — the true cross-session generalization)
- Hardware cost
- Composite score
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunResult:
  run_id: str
  variant: str
  dropout_mode: str
  r2_heldin: float
  r2_heldout_sessions: List[float]
  r2_heldout_mean: float
  r2_heldout_std: float
  identity_mse: float
  parameter_count: int
  mac_per_session: int
  peak_state_bytes: int
  trial_buffer_bytes: int
  requires_cubic_interp: bool
  requires_general_multiplier: bool


def parse_run(run_dir: Path) -> Optional[RunResult]:
  metrics_path = run_dir / "metrics_per_session.csv"
  hw_path = run_dir / "hardware_cost.json"
  cfg_path = run_dir / "resolved_config.yaml"

  if not metrics_path.exists():
    return None

  r2_heldin = None
  r2_heldout_sessions = []
  identity_mse = float("nan")
  with metrics_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
      split = row.get("split", "")
      try:
        r2 = float(row["R2_variance_weighted"])
      except (ValueError, KeyError):
        continue
      if split == "test_heldin":
        r2_heldin = r2
        try:
          identity_mse = float(row.get("identity_mse", "nan"))
        except ValueError:
          pass
      elif split == "test_heldout":
        r2_heldout_sessions.append(r2)

  if r2_heldin is None:
    return None

  # Parse hardware cost
  hw = {}
  if hw_path.exists():
    with hw_path.open() as f:
      hw = json.load(f)

  # Parse dropout mode
  dropout_mode = "none"
  if cfg_path.exists():
    with cfg_path.open() as f:
      for line in f:
        if "neuron_dropout_mode" in line and ":" in line:
          dropout_mode = line.split(":", 1)[1].strip().strip('"').strip("'")
          break

  r2_mean = statistics.mean(r2_heldout_sessions) if r2_heldout_sessions else 0.0
  r2_std = statistics.stdev(r2_heldout_sessions) if len(r2_heldout_sessions) > 1 else 0.0

  return RunResult(
    run_id=run_dir.name,
    variant=hw.get("variant", run_dir.name.split("_")[0]),
    dropout_mode=dropout_mode,
    r2_heldin=float(r2_heldin),
    r2_heldout_sessions=[round(r, 4) for r in r2_heldout_sessions],
    r2_heldout_mean=round(r2_mean, 4),
    r2_heldout_std=round(r2_std, 4),
    identity_mse=float(identity_mse),
    parameter_count=int(hw.get("parameter_count", 0)),
    mac_per_session=int(hw.get("mac_per_session", 0)),
    peak_state_bytes=int(hw.get("peak_live_state_bytes", 0)),
    trial_buffer_bytes=int(hw.get("trial_buffer_bytes", 0)),
    requires_cubic_interp=bool(hw.get("requires_cubic_interpolation", True)),
    requires_general_multiplier=bool(hw.get("requires_general_multiplier", True)),
  )


def collect_loso_runs(base_dir: Path) -> List[RunResult]:
  """Only collect runs with validation_protocol=loso and heldout evaluated."""
  results = []
  if not base_dir.exists():
    return results
  for run_dir in sorted(base_dir.iterdir()):
    if not run_dir.is_dir():
      continue
    split_path = run_dir / "split_manifest.json"
    if not split_path.exists():
      continue
    with split_path.open() as f:
      manifest = json.load(f)
    if manifest.get("validation_protocol") != "loso":
      continue
    if not manifest.get("heldout_evaluated_in_test"):
      continue
    result = parse_run(run_dir)
    if result is not None and result.r2_heldout_sessions:
      results.append(result)
  return results


def compute_scores(results: List[RunResult]) -> List[Dict]:
  if not results:
    return []

  best_heldout = max(r.r2_heldout_mean for r in results)
  best_heldin = max(r.r2_heldin for r in results)

  scored = []
  for r in results:
    # Accuracy gaps
    gap_heldout = best_heldout - r.r2_heldout_mean
    gap_heldin = best_heldin - r.r2_heldin

    # Normalize to [0, 1] where 1 = best
    # Held-out is the primary metric (true generalization)
    acc_score = max(0.0, 1.0 - gap_heldout / 0.15)  # 15% gap -> 0

    # Hardware efficiency
    mac_score = 1.0 - (math.log10(max(r.mac_per_session, 1)) - 3) / 7
    mac_score = max(0.0, min(1.0, mac_score))
    param_score = 1.0 - (math.log10(max(r.parameter_count, 1)) - 3) / 5
    param_score = max(0.0, min(1.0, param_score))
    state_kb = r.peak_state_bytes / 1024
    state_score = max(0.0, 1.0 - state_kb / 64)

    hw_bonus = 0.0
    if not r.requires_cubic_interp:
      hw_bonus += 0.1
    if not r.requires_general_multiplier:
      hw_bonus += 0.15
    if r.trial_buffer_bytes == 0:
      hw_bonus += 0.1

    # Composite — held-out weighted heavily
    composite = (
      0.55 * acc_score +
      0.15 * mac_score +
      0.10 * param_score +
      0.10 * state_score +
      hw_bonus
    )

    scored.append({
      **asdict(r),
      "gap_heldout_vs_best": round(gap_heldout, 4),
      "gap_heldin_vs_best": round(gap_heldin, 4),
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
    f"{'rank':<5} {'variant':<14} {'drop':<10} "
    f"{'R²_heldin':>10} {'R²_heldout':>11} {'±std':>7} "
    f"{'params':>8} {'MAC/sess':>11} "
    f"{'acc':>5} {'mac':>5} {'hw+':>5} {'COMP':>6}"
  )
  lines = [header, "-" * len(header)]
  for i, s in enumerate(scored, 1):
    lines.append(
      f"{i:<5} {s['variant']:<14} {s['dropout_mode']:<10} "
      f"{s['r2_heldin']:>10.4f} {s['r2_heldout_mean']:>11.4f} {s['r2_heldout_std']:>7.4f} "
      f"{s['parameter_count']:>8,} {s['mac_per_session']:>11,} "
      f"{s['acc_score']:>5.2f} {s['mac_score']:>5.2f} {s['hw_bonus']:>5.2f} {s['composite_score']:>6.3f}"
    )
  return "\n".join(lines)


def main():
  base_dir = Path("outputs/streaming_calibration")
  results = collect_loso_runs(base_dir)
  if not results:
    print("No LOSO+heldout runs found.")
    return 1

  print(f"Collected {len(results)} LOSO+heldout runs from {base_dir}")
  print()

  scored = compute_scores(results)
  print(format_table(scored))

  # Write CSV
  out_path = base_dir / "loso_heldout_scoring.csv"
  if scored:
    fieldnames = list(scored[0].keys())
    with out_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(scored)
    print(f"\nWrote {out_path}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
