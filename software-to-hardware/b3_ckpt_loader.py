"""Load B3 EarlyPool weights from a Lightning checkpoint into numpy B3Weights."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from b3_hw_golden import B3Weights


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("b3_ckpt_loader requires torch: pip install torch") from exc
    return torch


def _get_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def load_b3_weights_from_ckpt(ckpt_path: str | Path) -> B3Weights:
    torch = _require_torch()
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = _get_state_dict(obj)

    def pick(*candidates: str) -> np.ndarray:
        for key in candidates:
            if key in sd:
                return sd[key].detach().cpu().numpy().astype(np.float32)
        raise KeyError(f"None of {candidates} found in checkpoint")

    prefixes = ("student.id_encoder.", "id_encoder.", "module.student.id_encoder.")
    def pick_layer(layer: str, param: str) -> np.ndarray:
        for prefix in prefixes:
            key = f"{prefix}{layer}"
            if key in sd:
                return sd[key].detach().cpu().numpy().astype(np.float32)
        raise KeyError(f"Missing {layer}{param} in checkpoint (tried prefixes {prefixes})")

    return B3Weights(
        pre_w=pick_layer("pre_pool.0.weight", "pre_w"),
        pre_b=pick_layer("pre_pool.0.bias", "pre_b"),
        post0_w=pick_layer("post_pool.0.weight", "post0_w"),
        post0_b=pick_layer("post_pool.0.bias", "post0_b"),
        post1_w=pick_layer("post_pool.2.weight", "post1_w"),
        post1_b=pick_layer("post_pool.2.bias", "post1_b"),
        post2_w=pick_layer("post_pool.4.weight", "post2_w"),
        post2_b=pick_layer("post_pool.4.bias", "post2_b"),
    )


def load_hyperparams_from_ckpt(ckpt_path: str | Path) -> Dict[str, Any]:
    torch = _require_torch()
    obj = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    hp = obj.get("hyper_parameters", {})
    if not hp and "hparams" in obj:
        hp = obj["hparams"]
    return dict(hp)
