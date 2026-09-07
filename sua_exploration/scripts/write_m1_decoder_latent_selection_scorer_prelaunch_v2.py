#!/usr/bin/env python3
"""Re-seal scorer prelaunch after D4-label-loading correction; no data/scoring action."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2"
RUNNER = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_runner.py"
STAGE = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_stage.py"
TEST = ROOT / "sua_exploration" / "tests" / "test_m1_decoder_latent_selection_stage.py"
PREVIOUS = OUT / "selection_scorer_prelaunch.json"

def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""): d.update(b)
    return d.hexdigest()

def main() -> None:
    out = OUT / "selection_scorer_prelaunch_v2.json"
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    source = RUNNER.read_text(encoding="utf-8")
    required = ['side_feature_group="d4"', "if name == OUTER_LEFT_OUT", "_teacher_delta_for_source_only", "query_end_trial=210", "M1_DLA_ROOT_REVIEWED_EXECUTION"]
    if any(x not in source for x in required): raise ValueError("runner does not have current frozen D4/support or split guard")
    payload = {"schema_version": "m1_dla_selection_scorer_prelaunch_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "scorer_implemented_not_executed_pending_root_launch", "correction": "runner now loads M1 support obj_id through local d4 metadata only; query decoder receives no D4 side feature",
               "scope": {"heldin_sources": ["ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"], "outer_leftout": "ses-20120926", "support": [0,10], "query": [10,210], "report_or_later": "forbidden", "heldout": False, "EvalAI": False, "GPU": False},
               "hashes": {name: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for name,p in {"v1_prelaunch": PREVIOUS, "runner": RUNNER, "stage": STAGE, "tests": TEST}.items()},
               "not_executed": ["NWB_or_query_loading", "E0", "outer_train_Delta", "inner_candidates", "outer_gate", "bootstrap", "report", "heldout", "EvalAI", "GPU"]}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "selection_scorer_prelaunch_v2.sha256").write_text(f"{sha256(out)}  selection_scorer_prelaunch_v2.json\n", encoding="utf-8")
    print(out)
if __name__ == "__main__": main()
