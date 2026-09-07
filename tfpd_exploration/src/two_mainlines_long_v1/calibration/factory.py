"""Fresh P-FIX / P-CA factory. No global pair cache."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import basis as sealed_basis
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import carrier_solver as sealed_solver
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan as sealed_plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.full_query import load_student
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.stage1 import ensure_streaming_path

from . import constants as C
from . import hashes
from . import optimizer as optmod

FORBIDDEN_OLD_FACTORY = C.FORBIDDEN_OLD_FACTORY


class FactoryError(RuntimeError):
    """Fail closed for fresh-arm construction."""


def _assert_not_old_factory_source() -> None:
    source = inspect.getsource(build_fresh_arm)
    if "build_p_pair(" in source or "global _PAIR" in source or "_PAIR =" in source:
        raise FactoryError("fresh factory source must not call the old global _PAIR path")


@dataclass
class FreshPArm:
    """One independently constructed trainable arm."""

    name: str
    repo_root: Path
    frozen: sealed_normalizer.FrozenSourceNormalizer
    lit: Any
    student: Any
    basis: sealed_basis.RowNormalizedNNMFBasis
    optimizer: torch.optim.AdamW
    device: torch.device
    train_basis: bool
    seed_receipt: dict[str, object]
    sampler_index: int = 0
    sampler_epoch: int = 0
    global_step: int = 0
    extra: dict[str, object] = field(default_factory=dict)

    def consumer_sha256(self) -> str:
        return hashes.state_sha256(hashes.consumer_state(self.student))

    def learned_sha256(self) -> str:
        return hashes.state_sha256(hashes.learned_state(self.student, self.basis))

    def enter_train(self) -> dict[str, object]:
        self.student.train(True)
        self.student.decoder.train(True)
        self.student.id_encoder.train(True)
        return optmod.record_train_mode(self.student, self.basis, self.optimizer)

    def enter_eval(self) -> None:
        self.student.eval()


def load_sfix_student_fresh(repo_root: Path):
    """Load S-Fix bytes into a new Lightning module. Never reuse a global student."""
    repo_root = Path(repo_root)
    ensure_streaming_path(repo_root)
    path = repo_root / C.S_FIX_RELATIVE
    if sealed_plan.sha256_file(path) != C.S_FIX_SHA256:
        raise FactoryError("S-Fix bytes drifted")
    lit = load_student(repo_root, path, torch)
    student = lit.student
    if not hasattr(student, "carrier_projection_weight"):
        raise FactoryError("S-Fix P missing")
    if tuple(student.carrier_projection_weight.shape) != (1024, 4):
        raise FactoryError("P shape")
    return lit, student


def named_arm_parameters(student, basis, *, train_basis: bool) -> list[tuple[str, torch.nn.Parameter]]:
    items = [
        ("decoder", student.decoder),
        ("id_encoder", student.id_encoder),
        ("carrier_projection_weight", student.carrier_projection_weight),
    ]
    if train_basis:
        items.append(("basis", basis))
    return optmod.named_trainable(items)


def build_fresh_arm(
    repo_root: Path,
    frozen: sealed_normalizer.FrozenSourceNormalizer,
    arm: str,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = C.FORMAL_SEED,
) -> FreshPArm:
    """Construct one arm. Shared read-only D0/support is allowed; student/basis/opt are new."""
    _assert_not_old_factory_source()
    if arm not in {"P-FIX", "P-CA"}:
        raise FactoryError(f"unknown arm {arm}")
    repo_root = Path(repo_root)
    device = torch.device(device)
    seed_receipt = optmod.seed_all(int(seed), device=device) if seed is not None else {
        "seed": None,
        "python": False,
        "numpy": False,
        "torch": False,
        "cuda": False,
        "skipped": True,
    }
    lit, student = load_sfix_student_fresh(repo_root)
    student.to(device)
    d0 = torch.as_tensor(frozen.d0, dtype=torch.float64)
    scale = torch.as_tensor(frozen.scale, dtype=torch.float64)
    train_basis = arm == "P-CA"
    basis = sealed_basis.RowNormalizedNNMFBasis(
        dictionary=d0.clone(),
        scale=scale,
        trainable=train_basis,
    ).to(device)
    named = named_arm_parameters(student, basis, train_basis=train_basis)
    optimizer = optmod.build_adamw(named, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    arm_obj = FreshPArm(
        name=arm,
        repo_root=repo_root,
        frozen=frozen,
        lit=lit,
        student=student,
        basis=basis,
        optimizer=optimizer,
        device=device,
        train_basis=train_basis,
        seed_receipt=seed_receipt,
    )
    arm_obj.enter_train()
    return arm_obj


def solve_carrier_float64(z: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
    device_type = "cuda" if z.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        return sealed_solver.solve_ridge(
            z.to(dtype=torch.float64),
            rates.to(dtype=torch.float64),
            ridge_lambda=C.RIDGE_LAMBDA,
        )


def normalize_carrier(raw: torch.Tensor, frozen: sealed_normalizer.FrozenSourceNormalizer) -> torch.Tensor:
    mu = torch.as_tensor(frozen.mu0, dtype=torch.float64, device=raw.device)
    sigma = torch.as_tensor(frozen.sigma0, dtype=torch.float64, device=raw.device)
    return ((raw - mu) / sigma).to(dtype=torch.float32)
