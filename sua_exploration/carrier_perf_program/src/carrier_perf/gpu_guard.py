"""Force CPU-only execution for carrier_perf_program entrypoints."""
from __future__ import annotations

import os
from typing import Optional


GUARD_ENV = "CARRIER_PERF_REVIEWED_CPU"
GUARD_VALUE = "YES"


def force_cpu(reason: str = "carrier_perf_program forbids GPU use") -> None:
    """Blank CUDA visibility before any torch import and refuse a visible device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Best-effort: if torch is already imported with CUDA, fail closed.
    torch_mod = None
    try:
        import sys

        torch_mod = sys.modules.get("torch")
    except Exception:  # pragma: no cover
        torch_mod = None
    if torch_mod is not None:
        try:
            if bool(torch_mod.cuda.is_available()):
                raise RuntimeError(
                    f"{reason}: torch.cuda.is_available() is True after blanking "
                    "CUDA_VISIBLE_DEVICES; aborting rather than risking a GPU cell."
                )
        except RuntimeError:
            raise
        except Exception:
            # If cuda query itself fails, treat as non-CUDA and continue.
            pass


def require_reviewed_cpu(flag_execute: bool) -> None:
    """Require both CLI execute intent and the reviewed env guard."""
    if not flag_execute:
        raise RuntimeError("execute path requires --execute")
    if os.environ.get(GUARD_ENV) != GUARD_VALUE:
        raise RuntimeError(
            f"execute path requires {GUARD_ENV}={GUARD_VALUE} "
            "(scaffold default is dry-run / synthetic only)"
        )


def reviewed_cpu_ok() -> bool:
    return os.environ.get(GUARD_ENV) == GUARD_VALUE


def assert_no_gpu_flag(gpu_flag: Optional[bool]) -> None:
    if gpu_flag:
        raise RuntimeError("GPU flags are refused by carrier_perf_program")
