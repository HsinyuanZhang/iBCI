"""Fail-closed aggregator for the B1 carrier-arm loss-mode three-cell matrix."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.metrics.gate2_matrix import (
    REQUIRED_R1_LOSS_MODES,
    _finite_float,
    _loss_row_for_mode,
    choose_winning_loss,
    load_matrix,
)


B1_MATRIX_TAG = "b1_carrier_loss_mode"
REQUIRED_B1_LOSS_MODES = REQUIRED_R1_LOSS_MODES


class IncompleteB1MatrixError(ValueError):
  """Raised when the carrier-arm loss-mode matrix is missing required cells."""


@dataclass
class B1CarrierLossModeReadiness:
  ready: bool
  missing_requirements: List[str] = field(default_factory=list)
  loss_r2: Dict[str, float] = field(default_factory=dict)
  loss_delta: Dict[str, float] = field(default_factory=dict)


@dataclass
class B1CarrierLossModeDecision:
  ready: bool
  decision_state: str
  winning_loss: Optional[str]
  loss_r2: Dict[str, float]
  loss_delta: Dict[str, float]
  missing_requirements: List[str]
  notes: List[str]


def _b1_loss_rows(
  matrix_rows: Sequence[Mapping[str, Any]],
  *,
  fold_id: int,
  seed: int,
) -> List[Dict[str, Any]]:
  selected: List[Dict[str, Any]] = []
  for row in matrix_rows:
    tags = str(row.get("tags", ""))
    role = str(row.get("comparison_role", ""))
    if B1_MATRIX_TAG not in tags and role != "b1_carrier_loss_mode":
      continue
    if str(row.get("fold_id")) != str(fold_id):
      continue
    if str(row.get("seed")) != str(seed):
      continue
    selected.append(dict(row))
  return selected


def check_b1_carrier_loss_mode_readiness(
  matrix_rows: Sequence[Mapping[str, Any]],
  *,
  fold_id: int = 0,
  seed: int = 42,
) -> B1CarrierLossModeReadiness:
  missing: List[str] = []
  loss_rows = _b1_loss_rows(matrix_rows, fold_id=fold_id, seed=seed)
  loss_r2: Dict[str, float] = {}
  loss_delta: Dict[str, float] = {}
  for loss_mode in REQUIRED_B1_LOSS_MODES:
    row = _loss_row_for_mode(loss_rows, loss_mode)
    if row is None:
      missing.append(f"b1_carrier_loss_mode loss_mode={loss_mode} fold={fold_id} seed={seed}")
      continue
    r2 = _finite_float(row.get("R2"))
    delta = _finite_float(row.get("delta_fixed_B0"))
    if r2 is None:
      missing.append(f"finite R2 for b1 loss_mode={loss_mode}")
    else:
      loss_r2[loss_mode] = r2
    if delta is None:
      missing.append(f"finite delta_fixed_B0 for b1 loss_mode={loss_mode}")
    else:
      loss_delta[loss_mode] = delta
  return B1CarrierLossModeReadiness(
    ready=not missing,
    missing_requirements=missing,
    loss_r2=loss_r2,
    loss_delta=loss_delta,
  )


def aggregate_b1_carrier_loss_mode(
  matrix_rows: Sequence[Mapping[str, Any]],
  *,
  fold_id: int = 0,
  seed: int = 42,
) -> B1CarrierLossModeDecision:
  """Fail-closed: reject incomplete three-cell carrier-arm matrices."""
  readiness = check_b1_carrier_loss_mode_readiness(
    matrix_rows, fold_id=fold_id, seed=seed
  )
  if not readiness.ready:
    raise IncompleteB1MatrixError(
      "B1 carrier loss-mode matrix incomplete: "
      + "; ".join(readiness.missing_requirements)
    )

  loss_rows = _b1_loss_rows(matrix_rows, fold_id=fold_id, seed=seed)
  ranked_rows = [
    row
    for mode in REQUIRED_B1_LOSS_MODES
    if (row := _loss_row_for_mode(loss_rows, mode)) is not None
  ]
  if len(ranked_rows) != len(REQUIRED_B1_LOSS_MODES):
    raise IncompleteB1MatrixError(
      f"Expected {len(REQUIRED_B1_LOSS_MODES)} B1 cells, found {len(ranked_rows)}"
    )

  winning_loss, decision_state, notes = choose_winning_loss(ranked_rows)
  return B1CarrierLossModeDecision(
    ready=True,
    decision_state=decision_state,
    winning_loss=winning_loss,
    loss_r2=readiness.loss_r2,
    loss_delta=readiness.loss_delta,
    missing_requirements=[],
    notes=notes,
  )


def aggregate_b1_carrier_loss_mode_file(
  matrix_path: str,
  *,
  fold_id: int = 0,
  seed: int = 42,
) -> B1CarrierLossModeDecision:
  from pathlib import Path

  return aggregate_b1_carrier_loss_mode(
    load_matrix(Path(matrix_path)), fold_id=fold_id, seed=seed
  )
