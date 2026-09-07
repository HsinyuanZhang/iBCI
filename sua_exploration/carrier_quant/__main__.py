"""CLI entry point for source-only Q0 audit (user-run later on real data)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script without installing the package.
_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parents[1]
if str(_PKG_ROOT.parent) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT.parent))
_SPINT_SRC = _REPO_ROOT / "SPINT-main" / "src"
if str(_SPINT_SRC) not in sys.path:
  sys.path.insert(0, str(_SPINT_SRC))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Source-only carrier quantization audit (Q0)")
  parser.add_argument("--data-dir", type=Path, required=True, help="H1 NWB data directory")
  parser.add_argument("--raw-receipt", type=Path, required=True)
  parser.add_argument("--eb-receipt", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)

  from carrier_quant.audit import AuditRecording, run_audit
  from data.h1_m4_eb_pilot import H1_M4_FOLD0_SOURCE, load_source_records, reconstruct_frozen_plan

  records = load_source_records(args.data_dir)
  plan = reconstruct_frozen_plan(records, args.raw_receipt, args.eb_receipt)

  audit_records = []
  for name in H1_M4_FOLD0_SOURCE:
    record = records[name]
    rates_list = []
    label_list = []
    for trial in record.trials[:4]:
      rates_list.append(trial.rates)
      label_list.append(trial.velocity)
    rates = __import__("numpy").concatenate(rates_list, axis=0)
    labels = __import__("numpy").concatenate(label_list, axis=0)
    audit_records.append(
      AuditRecording(
        session_name=record.session_name,
        date=record.date,
        rates=rates,
        labels=labels,
        input_sha256=record.input_sha256,
      )
    )

  run_audit(audit_records, plan, output_path=args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
