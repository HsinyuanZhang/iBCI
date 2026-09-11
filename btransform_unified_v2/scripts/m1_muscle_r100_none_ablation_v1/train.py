#!/usr/bin/env python3
"""Static M1 R100/D4/P16 NONE ablation runner.

It preserves the frozen M1 source sampler and HO3/24-EMA score protocol while
making both information channels literal zero.  This program never accepts a
carrier artifact, warm-starts, or a non-NONE arm.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
AUDITED_TRAIN = SCRIPTS / "carrier_v4" / "m1" / "train.py"
sys.path.insert(0, str(HERE))
from model import M1InformationArmDecoder, model_source_sha256, none_contract

SOURCE = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
EPOCHS, UPDATES, SEED = 24, 6665, 42
CELL = "M1-RIFT-R100-D4-JOINT-B3S-CONCAT-P16-NONE-ABLATION-V1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _load_audited() -> Any:
    name = "_m1_none_audited_runner"
    if name in sys.modules:
        return sys.modules[name]
    # Force audited train.py's unqualified `model` import to this NONE facade.
    # Its carrier module is deliberately replaced before import: NONE has no
    # carrier artifact to load, fit, deserialize, or route into a TaskBank.
    carrier = ModuleType("carrier")
    def _forbidden_loader(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("M1/NONE attempted to load a carrier pack")
    def _replace(bank: Any, value: np.ndarray) -> Any:
        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.shape != (64, 4) or int(np.count_nonzero(array)):
            raise RuntimeError("M1/NONE carrier replacement must be literal zero [64,4]")
        metadata = dict(bank.calibration_meta)
        metadata.update({"carrier_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                         "carrier_method": "literal_zero_none", "carrier_schema": "m1_none_v1"})
        return dataclasses.replace(bank, carrier=array, calibration_meta=metadata)
    carrier.load_carrier_pack = _forbidden_loader
    carrier.replace_bank_carrier = _replace
    sys.modules["carrier"] = carrier
    spec = importlib.util.spec_from_file_location(name, AUDITED_TRAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audited M1 runner: {AUDITED_TRAIN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _zero_binding(module: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {session: np.zeros((64, 4), dtype=np.float32) for session in (*SOURCE, *HO)}
    binding = {
        "schema": "m1_none_literal_zero_carrier_binding_v1",
        "carrier_input": "none; no pack accepted or read",
        "sessions": list((*SOURCE, *HO)),
        "shape": [64, 4], "dtype": "float32", "all_nonzero_counts": {s: 0 for s in arrays},
        "train_py_sha256": _sha(Path(__file__)), "model_facade_sha256": _sha(HERE / "model.py"),
        "audited_model_py": str((HERE.parent / "carrier_v4" / "m1" / "model.py").resolve()),
        "audited_model_py_sha256": model_source_sha256(),
        # Kept solely because the audited checkpoint schema has this field.
        "fit_sha256": "0" * 64,
    }
    binding["binding_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return arrays, binding


def _route_receipt(module: Any, *, phase: str, device: str) -> dict[str, Any]:
    model = M1InformationArmDecoder("none", fusion="concat", proj_dim=16, seed=SEED).to(device).eval()
    # NONE deliberately receives no calibration side dictionary.
    class Bank:
        session_id = "m1-none-route"
    model.install_session_memory({"m1-none-route": Bank()}, {})
    def forbidden(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("NONE route invoked frozen activity encoder")
    model.encoder.forward = forbidden  # type: ignore[method-assign]
    e0, direct_t = model._identity(["m1-none-route"], torch.device(device))
    if int(torch.count_nonzero(e0)) or int(torch.count_nonzero(direct_t)):
        raise RuntimeError("NONE route was not literal zero")
    frozen = [name for name, value in model.named_parameters() if name.startswith("encoder.") and not value.requires_grad]
    if not frozen:
        raise RuntimeError("NONE encoder parameters were not frozen")
    return {"schema": "m1_none_information_route_receipt_v1", "status": "PASSED", "phase": phase,
            "information_contract": none_contract(), "e0_shape": list(e0.shape), "e0_nonzero": 0,
            "direct_t_shape": list(direct_t.shape), "direct_t_nonzero": 0,
            "encoder_forward_replaced_with_forbidden": True, "encoder_parameter_count_frozen": len(frozen),
            "normal_neural_query_is_decoder_input": True,
            "source_m10_calibration_not_passed_to_install_session_memory": True,
            "carrier_pack_not_accepted": True}


def _validate_baseline(module: Any, baseline: Path) -> dict[str, Any]:
    meta = _json(baseline / "run_meta.json")
    model = module._decoder(torch.device("cpu"), "none", SEED, fusion="concat", proj_dim=16)
    actual = module._initialization_sha(model)
    expected = meta.get("initialization_sha256")
    if not isinstance(expected, str) or expected != actual:
        raise RuntimeError("NONE initialization checksum does not equal original seed42 baseline")
    if meta.get("seed") != SEED or meta.get("sampler_seed") != SEED:
        raise RuntimeError("baseline is not the frozen seed42 source/sampler identity")
    return {"baseline_run": str(baseline.resolve()), "baseline_run_meta_sha256": _sha(baseline / "run_meta.json"),
            "baseline_initialization_sha256": expected, "none_initialization_sha256": actual,
            "initialization_sha256_match": True, "source_sampler_seed42_match": True}


def _configure(module: Any, baseline: Path) -> None:
    arrays, binding = _zero_binding(module)
    module.M1InformationArmDecoder = M1InformationArmDecoder
    module.INFORMATION_ARMS = ("none",)
    module._cell = lambda fusion, proj: CELL if (fusion, proj) == ("concat", 16) else (_ for _ in ()).throw(RuntimeError("fixed P16 concat only"))
    module._carrier_binding = lambda _ignored: (arrays, binding)
    # The original loader still builds canonical TaskBank objects for normal
    # neural query/session IDs. This replacement ensures T is zero before any
    # model call; the NONE model never reads the bank E0/calibration field.
    original_replace = module._replace_carriers
    module._replace_carriers = lambda banks, _carriers, sessions: original_replace(banks, arrays, sessions)
    original_ho = module._ho_material
    module._ho_material = lambda _carriers: original_ho(arrays)
    original_hashes = module._source_hashes
    module._source_hashes = lambda: {**original_hashes(), "none_runner_py": _sha(Path(__file__)),
                                    "none_model_py": _sha(HERE / "model.py"), "audited_none_model_py": model_source_sha256()}
    module._baseline_run = lambda _args: baseline.resolve()


def _args(ns: argparse.Namespace) -> Any:
    return SimpleNamespace(dest=ns.dest, information_arm="none", fusion="concat", proj_dim=16, seed=SEED,
                           stage=ns.stage, carrier_pack=HERE / "NO_CARRIER_INPUT", baseline_run=ns.baseline_run,
                           device=ns.device, epochs=EPOCHS, resume=None,
                           max_updates_smoke=ns.smoke_steps, cpu_threads=ns.cpu_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True, help="frozen original seed42 M1 B3S baseline")
    parser.add_argument("--stage", choices=("train", "score"), required=True)
    parser.add_argument("--smoke-steps", type=int, help="train-only route/gradient/init smoke step count")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--cpu-threads", type=int, default=4)
    ns = parser.parse_args()
    if ns.smoke_steps is not None and (ns.smoke_steps < 1 or ns.stage != "train"):
        parser.error("--smoke-steps requires --stage train and a positive value")
    dest = ns.dest.resolve(); baseline = ns.baseline_run.resolve()
    if not baseline.is_dir(): parser.error("--baseline-run must name an existing frozen run directory")
    if ns.stage == "train" and ns.smoke_steps is None and dest.exists():
        parser.error("formal train refuses an existing --dest; use a fresh destination")
    if ns.stage == "train" and ns.smoke_steps is not None and dest.exists():
        parser.error("smoke also requires a fresh --dest to preserve its receipt")
    module = _load_audited(); _configure(module, baseline)
    baseline_receipt = _validate_baseline(module, baseline)
    route = _route_receipt(module, phase="train" if ns.stage == "train" else "score", device=ns.device)
    inner = _args(ns)
    result = module.run_train(inner) if ns.stage == "train" else module.run_score(inner)
    if ns.stage == "train":
        target = dest / ("smoke_route_receipt.json" if ns.smoke_steps is not None else "route_receipt.json")
    else:
        target = dest / "score_route_receipt.json"
    _atomic_json(target, {**route, "baseline_initialization": baseline_receipt,
                          "fixed_recipe": {"seed": SEED, "epochs": EPOCHS, "updates_per_epoch": UPDATES,
                                           "total_updates": EPOCHS * UPDATES, "windows": [25, 25, 25, 24], "batch": 32}})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
