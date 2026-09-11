"""Fixed M1/NONE decoder facade.

The audited implementation is loaded under a private module name so this
directory has a stable, explicit model entry point without changing v4 code.
Only the literal NONE information route is exported.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
AUDITED = HERE.parent / "carrier_v4" / "m1" / "model.py"


def _load() -> Any:
    name = "_m1_none_audited_information_model"
    if name in sys.modules:
        return sys.modules[name]
    # Its absolute script dependencies are intentional audited dependencies.
    for entry in (AUDITED.parent, AUDITED.parent.parent):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    spec = importlib.util.spec_from_file_location(name, AUDITED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audited NONE model: {AUDITED}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_AUDITED = _load()
M1InformationArmDecoder = _AUDITED.M1InformationArmDecoder
INFORMATION_ARMS = ("none",)


def none_contract() -> dict[str, Any]:
    """Contract recorded in both train and score receipts."""
    base = dict(_AUDITED.arm_contract("none"))
    if base != {
        "information_arm": "none", "legacy_arm": "B_ACTIVITY_ONLY",
        "e0": "literal_zero_64x100", "direct_t": "literal_zero_64x4",
        "encoder_called": False, "encoder_trainable": False,
    }:
        raise RuntimeError("audited NONE information contract drift")
    return {**base, "side_path": "no_static_information_retained_or_consumed",
            "normal_neural_query": "preserved"}


def arm_contract(information_arm: str) -> dict[str, Any]:
    if information_arm != "none":
        raise ValueError("this facade supports only the fixed NONE arm")
    return none_contract()


def model_source_sha256() -> str:
    return hashlib.sha256(AUDITED.read_bytes()).hexdigest()


def build_none_decoder(*, seed: int = 42, device: str = "cpu") -> Any:
    model = M1InformationArmDecoder("none", fusion="concat", proj_dim=16, seed=seed).to(device)
    model.temporal.set_attention_backend("local")
    return model


__all__ = ["INFORMATION_ARMS", "M1InformationArmDecoder", "arm_contract", "build_none_decoder", "model_source_sha256", "none_contract"]


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="NWB-free fixed M1/NONE route smoke")
    parser.add_argument("--route-smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.route_smoke:
        parser.error("--route-smoke is required")
    model = build_none_decoder(device=args.device).eval()
    class _Bank:
        session_id = "m1-none-model-smoke"
    model.install_session_memory({_Bank.session_id: _Bank()}, {})
    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("NONE route invoked encoder")
    model.encoder.forward = _forbidden  # type: ignore[method-assign]
    e0, direct_t = model._identity([_Bank.session_id], __import__("torch").device(args.device))
    if int(e0.count_nonzero()) or int(direct_t.count_nonzero()):
        raise RuntimeError("NONE route is not literal zero")
    print(json.dumps({"status": "PASSED", "contract": none_contract(), "e0_nonzero": 0,
                      "direct_t_nonzero": 0, "encoder_forbidden": True}, sort_keys=True))
