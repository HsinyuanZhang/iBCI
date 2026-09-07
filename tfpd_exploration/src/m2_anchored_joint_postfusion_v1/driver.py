"""AJPF root-owned one-shot entry; the public CLI deliberately cannot call it."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import os
import subprocess
import sys

from . import lifecycle, plan


class DriverError(RuntimeError):
    pass


def issue_after_independent_review(*, reviewed_closure: Mapping[str, str], reviewed_closure_sha256: str,
                                   repo_root: Path = plan.REPO_ROOT) -> lifecycle.Capability:
    """Future root-only issuer; it performs no Torch/data/CUDA/result action."""
    from . import production
    production.validate_frozen_environment(repo_root=repo_root, score=False)
    return lifecycle.issue_production_capability(repo_root=repo_root, root=repo_root / plan.TRAINING_ROOT_RELATIVE,
                                                 reviewed=reviewed_closure, digest=reviewed_closure_sha256)


def execute_production(_capability: lifecycle.Capability) -> dict[str, str]:
    """Execute training then a fresh private CPU scoring process.

    There is intentionally no callable/materializer/scorer argument.  A CUDA
    process cannot honestly become a no-CUDA scorer after training, so the
    process boundary is part of the scientific contract rather than a
    convenience wrapper.
    """
    from . import production
    train = production.execute_training(cap=_capability, repo_root=plan.REPO_ROOT)
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
                        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                        "PYTHONPATH": str(plan.REPO_ROOT)})
    code = ("from tfpd_exploration.src.m2_anchored_joint_postfusion_v1.production "
            "import _run_cpu_score_child; _run_cpu_score_child(" + repr(str(plan.REPO_ROOT)) + ")")
    # Do not use ``-S`` here: it strips the installed Torch/NumPy runtime.
    # PYTHONNOUSERSITE remains the work-order authority and the child itself
    # verifies its import/runtime boundary before opening any artifact.
    subprocess.run([sys.executable, "-c", code], check=True, env=environment, cwd=str(plan.REPO_ROOT))
    return {"training_root": str(train["root"]), "score_root": str(plan.REPO_ROOT / plan.SCORE_ROOT_RELATIVE)}
