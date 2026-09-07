"""Thin physical composition for the MATCHED_ERM held-in scorer.

No parser, M1 model, checkpoint loader, metric, or evaluator loop is copied:
the reviewed CS-WG backend receives only this successor's typed receipt codec.
"""
from __future__ import annotations

from pathlib import Path

from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import physical as shared_physical

from . import score


class MatchedERMHeldInScorePhysicalError(score.MatchedERMHeldInScoreError):
    """Fail closed for the narrow physical composition seam."""


def build_reviewed_matched_erm_heldin_score_backend(
    *, root: Path, source_root: Path, device: str,
) -> shared_physical.PhysicalHeldInScoreBackend:
    """Build a deferred target evaluator without opening data or importing Torch.

    The inherited factory retains its reviewed namespace bootstrap and direct
    `FalconDataModule.prepare_session_data`/`FalconDataset` reader.  Target
    descriptor resolution, parser construction, checkpoint tensor loading,
    CUDA, and forward evaluation remain behind the durable score attempt.
    """
    try:
        return shared_physical.build_reviewed_heldin_score_backend(
            root=Path(root), source_root=Path(source_root), device=device,
            route_profile=score.MatchedERMPhysicalCodec(),
        )
    except shared_physical.HeldInScorePhysicalError as error:
        raise MatchedERMHeldInScorePhysicalError(
            "CS-WG matched-ERM held-in score inherited physical factory drift"
        ) from error


__all__ = ("MatchedERMHeldInScorePhysicalError", "build_reviewed_matched_erm_heldin_score_backend")
