"""Independent-audit-bound complete M1 source-dev candidate stream replay."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m1_runtime_v3.complete_validator import ROOT, POST, array_sha, r2
from tfpd_exploration.src.m1_runtime_v3 import BankBatch
from .benchmark import sha
from .m1 import FiveTokenCurrentQueryStream
from .m1_lifted import LiftedFiveTokenCurrentQueryStream


def run(args):
    from tfpd_exploration.src.decoder_validation_v2.audit_predictions import audit_m1
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.m1_optimized_v2.source_dev import _model
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    begun = time.perf_counter()
    row = json.loads(POST.read_text())["selected"]
    state, reference = Path(row["plain_ema_model_state"]), Path(row["prediction_export"])
    if sha(state) != row["plain_ema_model_state_sha256"] or sha(reference) != row["prediction_export_sha256"]:
        raise RuntimeError("sealed selected authority drift")
    reference_audit = audit_m1(reference, checkpoint=Path(row["checkpoint"]))
    with np.load(reference, allow_pickle=False) as archive:
        export = {key: archive[key].copy() for key in archive.files}
    cache_path = ROOT / "m1_optimized_v2_source_runtime_cache.npz"
    receipt = json.loads(cache_path.with_suffix(".receipt.json").read_text())
    provenance_path = ROOT / "rSyn3-refit-v1.source-only.provenance-supplement.npz"
    provenance_receipt = json.loads(provenance_path.with_suffix(".receipt.json").read_text())
    if sha(cache_path) != receipt["npz_sha256"] or sha(provenance_path) != provenance_receipt["supplement_npz_sha256"]:
        raise RuntimeError("cache or physical unit authority drift")
    model = _model("current_query").eval()
    model.load_state_dict(torch.load(state, map_location="cpu", weights_only=False), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stream_type = FiveTokenCurrentQueryStream if args.variant == "five_token" else LiftedFiveTokenCurrentQueryStream
    offline = export["prediction"].copy()
    prediction = np.empty_like(offline)
    checks = {}
    with np.load(cache_path, allow_pickle=False) as cache, np.load(provenance_path, allow_pickle=False) as provenance:
        for name in plan.SOURCE_SESSIONS:
            ids = np.flatnonzero(export["session"] == name)
            ids = ids[np.argsort(export["window_start"][ids], kind="stable")]
            starts = export["window_start"][ids].astype(np.int64)
            raw = np.asarray(cache[f"raw_neural/{name}"], dtype=np.float32)
            if not np.all(raw[:99] == 0):
                raise RuntimeError("expected 99-bin source startup prefix")
            e0, t, mask = (np.asarray(cache[f"bank_{key}/{name}"]) for key in ("e0", "t", "unit_mask"))
            roster = np.asarray(provenance[f"nwb_unit_ids_in_rate_column_order/{name}"])
            if array_sha(e0) != receipt["arrays"][name]["e0_sha256"] or array_sha(roster) != provenance_receipt["rows"][name]["nwb_unit_ids_sha256"]:
                raise RuntimeError("bank/roster authority drift")
            bank = BankBatch(torch.from_numpy(e0)[None], torch.from_numpy(t)[None], torch.from_numpy(mask)[None], (name,), (tuple(roster.tolist()),))
            first = int(starts[0])
            stream = stream_type(model, bank)
            stream.refresh_state(history=torch.from_numpy(raw[None, first:first + 100]))
            cursor, consumed = first + 99, 0
            for index, start in zip(ids, starts, strict=True):
                endpoint = int(start) + 99
                if cursor == endpoint:
                    value = stream.current_prediction().detach().numpy()[0].copy()
                else:
                    while cursor < endpoint:
                        cursor += 1
                        value = stream.predict(np.ascontiguousarray(raw[cursor:cursor + 1]))
                        consumed += 1
                    value = value[0]
                error = np.abs(value - offline[index])
                if not np.all(error <= 1e-5 + 1e-5 * np.abs(offline[index])):
                    raise RuntimeError(f"native parity at {name}:{endpoint}: {error.max()}")
                prediction[index] = value
            delta = r2(prediction[ids], export["target"][ids]) - r2(offline[ids], export["target"][ids])
            if abs(delta) > 1e-5:
                raise RuntimeError("session R2 parity failure")
            checks[name] = {"n": len(ids), "max_abs_error": float(np.abs(prediction[ids] - offline[ids]).max()),
                            "r2_delta": delta, "public_calls_after_initial_window": consumed}
            print(json.dumps({"event": "session_complete", "session": name, **checks[name]}), flush=True)
    if len(prediction) != 31252:
        raise RuntimeError("complete 31,252 endpoint requirement")
    export["prediction"] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **export)
    streamed_audit = audit_m1(args.output.with_suffix(".npz"), checkpoint=Path(row["checkpoint"]))
    report = {"schema": "m1_family_runtime_complete_selected_stream_v1", "status": "PASS", "variant": args.variant,
              "device": "cpu", "n": len(prediction), "elapsed_seconds": time.perf_counter() - begun,
              "checks": checks, "max_abs_error": float(np.abs(prediction - offline).max()),
              "selected_state_sha256": row["plain_ema_model_state_sha256"], "reference_export_sha256": row["prediction_export_sha256"],
              "runtime_cache_sha256": receipt["npz_sha256"], "physical_unit_authority_sha256": provenance_receipt["supplement_npz_sha256"],
              "independent_reference_audit": reference_audit, "independent_stream_audit": streamed_audit,
              "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
              "stream_export_sha256": sha(args.output.with_suffix(".npz")),
              "startup_scope": "first dev W100 primed once; every subsequent gap consumed through public predict; cold zero startup is separately unit-tested",
              "not_latency_acceptance": True, "parameter_updates": 0}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("five_token", "lifted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
