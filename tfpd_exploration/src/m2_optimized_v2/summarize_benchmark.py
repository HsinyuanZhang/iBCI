"""Aggregate the three fresh-process M2 actual-adapter timing receipts."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from .benchmark import OUT

paths = sorted(glob.glob(str(OUT / "cpu_actualapi2048_threads2_process*.json")))
if len(paths) != 3:
    raise RuntimeError(f"need exactly 3 actual API process receipts, found {len(paths)}")
receipts = [json.loads(Path(p).read_text()) for p in paths]
rows = [row for receipt in receipts for row in receipt["records"]]
summary = []
for kind in ("small", "large"):
    for batch in (1, 7):
        for variant in ("reference", "optimized"):
            selected = [r for r in rows if (r["kind"],r["batch"],r["variant"]) == (kind,batch,variant)]
            wall = {k: float(np.mean([r["step_wall"][k] for r in selected])) for k in ("p50_ms","p95_ms","p99_ms","max_ms","mean_ms")}
            summary.append({"kind":kind,"batch":batch,"variant":variant,"fresh_processes":len(selected),"mean_of_process_quantiles_ms":wall})
out = {"schema":"m2_optimized_v2_cpu_actual_api_summary_v1","receipts":paths,
       "hardware":"AMD Ryzen 9 7950X; taskset CPUs 8-11; torch intraop=2, interop=1",
       "measurement":"full adapter step including buffer update, torch conversion, model, CPU NumPy conversion, /5, finite check, return; B7 wall is not divided by 7",
       "disclosure":"process0 predates automatic mutation guards but used frozen read-only state; process1/2 include guarded implementation revision","summary":summary}
(OUT / "cpu_actualapi2048_threads2_summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
