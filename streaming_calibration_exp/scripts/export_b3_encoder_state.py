"""Export a plain, weights-only B3 encoder state dict from a local Lightning checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path, PosixPath
from typing import Any

import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode


PREFIX = "student.id_encoder."
EXPECTED_KEYS = {
    "pre_pool.0.weight",
    "pre_pool.0.bias",
    "post_pool.0.weight",
    "post_pool.0.bias",
    "post_pool.2.weight",
    "post_pool.2.bias",
    "post_pool.4.weight",
    "post_pool.4.bias",
}
SAFE_GLOBALS = [dict, list, int, PosixPath, ContainerMetadata, Metadata, ListConfig, AnyNode, Any, defaultdict]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint))
    allowed = {f"{item.__module__}.{item.__qualname__}" for item in SAFE_GLOBALS}
    unexpected_globals = unsafe.difference(allowed)
    if unexpected_globals:
        raise RuntimeError(f"Refusing checkpoint with unexpected globals: {sorted(unexpected_globals)}")

    with torch.serialization.safe_globals(SAFE_GLOBALS):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no tensor state_dict")
    encoder_state = {
        key.removeprefix(PREFIX): value.detach().cpu()
        for key, value in state.items()
        if key.startswith(PREFIX) and isinstance(value, torch.Tensor)
    }
    if set(encoder_state) != EXPECTED_KEYS:
        raise ValueError(
            f"Expected a three-layer B3 encoder; missing={sorted(EXPECTED_KEYS - set(encoder_state))}, "
            f"unexpected={sorted(set(encoder_state) - EXPECTED_KEYS)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, args.output)
    print(f"Exported {len(encoder_state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
