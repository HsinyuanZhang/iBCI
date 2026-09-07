"""Successor receipts. The parent EMG-Syn3 FAIL root cannot be reserved."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import receipts as parent_receipts

from . import plan


class ReceiptError(RuntimeError):
    """Fail closed for successor receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def refuse_parent_root(path: Path | str) -> None:
    text = str(path)
    if "m1_emg_syn3_fcm_v1" in text:
        raise ReceiptError("refusing sealed parent syn3_fcm_v1 root")


def run_stage0(
    root: Path,
    *,
    attempt_payload: Mapping[str, object],
    launch_builder: Callable[[], Mapping[str, object]],
    body_publisher,
    terminal_builder: Callable[[Mapping[str, str]], Mapping[str, object]],
    relative: str = "stage0",
) -> tuple[dict[str, str], str | None, str | None]:
    refuse_parent_root(root)
    return parent_receipts.run_stage0(
        root,
        attempt_payload=attempt_payload,
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        relative=relative,
    )
