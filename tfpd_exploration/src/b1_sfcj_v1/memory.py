"""Growing activity memory: whole-stack training vs O(1) running-sum deployment."""
from __future__ import annotations

import hashlib
from typing import Literal, Optional

import numpy as np
import torch

from .constants import CARRIER_DIM, N_CHANNELS, N_MS_BINS


class GrowingMemory:
    def __init__(
        self,
        model,
        *,
        law: Literal["GROWING", "FIXED3"] = "GROWING",
        mode: Literal["whole_stack", "running_sum"] = "whole_stack",
        carrier: Optional[np.ndarray] = None,
        dtype=torch.float32,
    ):
        self.model = model
        self.law = law
        self.mode = mode
        self.dtype = dtype
        self.device = next(model.parameters()).device
        self.stack: list[torch.Tensor] = []
        self.K = 0
        self.sum_u = None
        self.sum_post = None
        self.carrier = None
        self._committed_ids: list[str] = []
        if carrier is not None:
            self.set_carrier(carrier)
        self.reset_state()

    def reset_state(self) -> None:
        self.stack = []
        self.K = 0
        self.sum_u = None
        self.sum_post = None
        self._committed_ids = []

    def set_carrier(self, carrier: np.ndarray) -> None:
        arr = np.asarray(carrier, dtype=np.float64)
        if arr.shape != (N_CHANNELS, CARRIER_DIM):
            raise ValueError(f"carrier must be [85,9], got {arr.shape}")
        self.carrier = torch.as_tensor(arr, dtype=self.dtype, device=self.device)

    def _as_trial(self, activity: np.ndarray | torch.Tensor) -> torch.Tensor:
        ten = torch.as_tensor(activity, dtype=self.dtype, device=self.device)
        if ten.shape != (N_MS_BINS, N_CHANNELS):
            raise ValueError(f"activity must be [900,85], got {tuple(ten.shape)}")
        return ten

    def seed_m3(self, activities, carrier: np.ndarray) -> None:
        self.reset_state()
        self.set_carrier(carrier)
        for i, act in enumerate(activities):
            self.commit(act, member_id=f"m3:{i}")

    def _pre_post(self, trial: torch.Tensor):
        # trial [900,85] -> [1,85,900]
        x = trial.permute(1, 0).unsqueeze(0)
        u = self.model.pre_pool(x)[0]  # [85, D]
        c = self.carrier
        post = self.model.post_pool(torch.cat([u, c], dim=-1))  # [85, 900]
        return u, post

    def commit(self, activity, member_id: Optional[str] = None) -> None:
        if self.law == "FIXED3" and self.K >= 3:
            return
        trial = self._as_trial(activity)
        token = member_id or f"k{self.K}"
        # The deployment path must not retain the raw historical stack.  The
        # running-sum state below is sufficient once model parameters are
        # frozen; keeping the stack here silently made the advertised O(1)
        # state grow with every completed trial.
        if self.mode == "whole_stack":
            self.stack.append(trial)
        if self.mode == "running_sum":
            with torch.no_grad():
                u, post = self._pre_post(trial)
            if self.sum_u is None:
                self.sum_u = u
                self.sum_post = post
            else:
                self.sum_u = self.sum_u + u
                self.sum_post = self.sum_post + post
        self.K += 1
        self._committed_ids.append(token)

    def identity(self, *, with_grad: bool = False):
        if self.K == 0:
            raise RuntimeError("empty memory")
        c = self.carrier.unsqueeze(0)  # [1,85,9]
        if self.mode == "whole_stack":
            stack = torch.stack(self.stack, dim=0).unsqueeze(0)  # [1,K,900,85]
            if not with_grad:
                with torch.no_grad():
                    h, u, h_n, h_p = self.model.identity_from_stack(stack, c)
            else:
                h, u, h_n, h_p = self.model.identity_from_stack(stack, c)
            return h[0], {"h_n": h_n[0], "h_p": h_p[0], "K": self.K}
        # running-sum O(1); valid only when params are frozen
        k = float(self.K)
        u_mean = self.sum_u / k
        h_n = self.model.post_pool(torch.cat([u_mean, self.carrier], dim=-1))
        h_p = self.sum_post / k
        if self.model.fusion == "jr1":
            h = h_n + torch.tanh(self.model.alpha) * (h_p - h_n)
        else:
            h = h_n
        if not with_grad:
            h = h.detach()
            h_n = h_n.detach()
            h_p = h_p.detach()
        return h, {"h_n": h_n, "h_p": h_p, "K": self.K}

    def pool_digest(self) -> str:
        body = f"{self.law}|{self.K}|" + ",".join(self._committed_ids)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def decode_then_commit(self, current_activity, member_id: Optional[str] = None):
        """Decode using memory of completed trials, then commit current. Current is not in digest before decode."""
        digest_before = self.pool_digest()
        k_before = self.K
        ident, parts = self.identity(with_grad=self.mode == "whole_stack")
        self.commit(current_activity, member_id=member_id)
        return ident, {
            **parts,
            "digest_before": digest_before,
            "digest_after": self.pool_digest(),
            "k_before": k_before,
            "k_after": self.K,
            "current_in_digest_before": False,
        }
