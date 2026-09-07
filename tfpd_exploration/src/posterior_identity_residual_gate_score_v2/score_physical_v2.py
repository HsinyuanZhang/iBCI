"""The sole physical V2 seam: scalar-safe PIRG wrapper state hashing."""
from __future__ import annotations

import hashlib
from typing import Any

from src.posterior_identity_residual_gate_v1 import score_physical as v1_physical


def scalar_safe_tensor_digest(*, core: Any, tensor: Any) -> str:
    """Match V1's dtype/shape/bytes hash while supporting a rank-zero tensor.

    Non-scalars delegate byte-for-byte to the frozen generic implementation.
    Scalars retain their original ``()`` shape in the digest, normalize signed
    zero exactly as V1 does, and reshape only for the uint8 byte view.
    """
    if getattr(tensor, "ndim", None) != 0:
        return core.tensor_digest(tensor)
    torch = core.torch
    detached = tensor.detach().cpu().contiguous()
    if detached.is_floating_point():
        detached = detached + 0
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(str(tuple(detached.shape)).encode("utf-8"))
    if detached.numel():
        digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PhysicalPIRGQuickScoreBackendV2(v1_physical.PhysicalPIRGQuickScoreBackend):
    """Exact V1 physical backend with only the 0-D alpha digest repaired."""

    @staticmethod
    def _wrapper_state_digest(*, arm_common: Any, core: Any, wrapper: Any) -> str:
        base = arm_common.state_sha256(wrapper.cell_d)
        alpha = scalar_safe_tensor_digest(core=core, tensor=wrapper.alpha.detach())
        return v1_physical._digest(v1_physical._json({
            "base_cell_d_state_sha256": base,
            "alpha_tensor_sha256": alpha,
        }))

