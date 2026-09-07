"""Post-score prediction-dispersion addendum for a completed RIFT carrier run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


ARMS=("REAL","C_ZERO","C_SHUF101","C_SHUF102","C_SHUF103")


def stats(values: np.ndarray) -> dict[str, object]:
    flat=np.asarray(values,np.float64)
    return {"shape":list(flat.shape),"finite":bool(np.isfinite(flat).all()),"global_mean":float(flat.mean()),"global_std":float(flat.std()),"global_min":float(flat.min()),"global_max":float(flat.max()),"per_output_mean":flat.mean(0).tolist(),"per_output_std":flat.std(0).tolist()}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--result-dir",type=Path,required=True); args=p.parse_args(); root=args.result_dir.resolve()
    result_path=root/"result.json"; result=json.loads(result_path.read_text())
    if result.get("schema")!="rift_h1_carrier_reliance_v1" or result.get("status")!="COMPLETED": raise ValueError("requires completed RIFT carrier result")
    arrays={arm:np.load(root/f"predictions_{arm}.npz") for arm in ARMS}
    keys=sorted(arrays["REAL"].files); real=np.concatenate([arrays["REAL"][key] for key in keys],axis=0)
    report={}
    for arm in ARMS:
        value=np.concatenate([arrays[arm][key] for key in keys],axis=0)
        row={"prediction":stats(value)}
        delta=np.asarray(value-real,np.float64); row["vs_real"]={"mean_abs":float(np.abs(delta).mean()),"rms":float(np.sqrt(np.mean(delta**2))),"max_abs":float(np.abs(delta).max()),"per_output_rms":np.sqrt(np.mean(delta**2,axis=0)).tolist()}
        report[arm]=row
    payload={"schema":"rift_h1_carrier_reliance_prediction_dispersion_v1","record_keys":keys,"total_selected_endpoints":int(real.shape[0]),"arms":report}
    (root/"prediction_dispersion.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    result["prediction_dispersion_file"]="prediction_dispersion.json"; result["prediction_dispersion_schema"]=payload["schema"]
    result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
