"""Average one residual parameter across trusted local Lightning checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train import _allow_known_checkpoint_globals


def make_residual_checkpoint_soup(
    checkpoint_paths: list[Path],
    output_path: Path,
    parameter_suffix: str = "student.id_encoder.var_linear.weight",
) -> dict[str, object]:
    if len(checkpoint_paths) < 2:
        raise ValueError("Checkpoint soup requires at least two checkpoints")

    checkpoints = []
    parameter_values = []
    resolved_key = None
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        _allow_known_checkpoint_globals(str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint has no tensor state_dict: {checkpoint_path}")
        matching_keys = [key for key in state_dict if key.endswith(parameter_suffix)]
        if len(matching_keys) != 1:
            raise ValueError(
                f"Expected exactly one key ending in {parameter_suffix!r}; "
                f"found {matching_keys} in {checkpoint_path}"
            )
        key = matching_keys[0]
        if resolved_key is not None and key != resolved_key:
            raise ValueError(f"Residual key mismatch: {resolved_key!r} versus {key!r}")
        resolved_key = key
        value = state_dict[key]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"Residual parameter is not floating point: {key}")
        checkpoints.append(checkpoint)
        parameter_values.append(value)

    shapes = {tuple(value.shape) for value in parameter_values}
    if len(shapes) != 1:
        raise ValueError(f"Residual parameter shape mismatch: {sorted(shapes)}")

    averaged = torch.stack(parameter_values, dim=0).mean(dim=0)
    template = checkpoints[0]
    template["state_dict"][resolved_key] = averaged
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "output": str(output_path.resolve()),
        "sha256": digest,
        "parameter_key": resolved_key,
        "checkpoint_count": len(checkpoint_paths),
        "sources": [str(path.resolve()) for path in checkpoint_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--parameter-suffix",
        default="student.id_encoder.var_linear.weight",
    )
    args = parser.parse_args()
    result = make_residual_checkpoint_soup(
        args.checkpoints, args.output, args.parameter_suffix
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
