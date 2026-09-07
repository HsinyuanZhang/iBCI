"""A-QMEM: query-conditioned prefix-memory residual on a frozen champion E0.

Architecture (CONTRACT_VERSION=1 / workorder §3.2):

    s_bi  = GELU(Linear_50_to_64(X_b[:, i]))
    q_bi  = Q([s_bi, E0_i, T_i])                 # 2-layer MLP, hidden 64
    k_ji  = K([u_ji - h_bar_i, T_i])             # 2-layer MLP; T through GELU
    v_ji  = V(u_ji - h_bar_i)                    # Linear(64, 32), no bias
    w_bij = softmax_j(q_bi · k_ji / sqrt(d_head))
    r_bi  = concat_heads sum_j (w_bij - 1/K) v_ji
    delta = W_out(GELU(r_bi))                    # Linear(32, 50), no bias, zeros
    E_bi  = E0_i + delta_bi
    y_b   = frozen_student.decode_with_identity(X_b, E_b)[:, -1, :]

Attention is per-unit over the trial axis K, not all-to-all over N*K.
h_bar_i is the chronological mean of frozen_u over trials. E0 is used as-is
from the bank (not recomputed from u).

First-round last-bin scoring uses the full 50-bin window as the query.
A single E is produced per (batch, unit); only the last decoder timestamp
is scored. This is not a prefix-causal contract for earlier outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import plan
from .contracts import CONTRACT_VERSION, SessionBank, assert_contract_version

assert_contract_version(1)

_ALLOWLIST_ROOTS = ("query_proj", "Q", "K", "V", "W_out")


def _two_layer_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class AQMEM(nn.Module):
    """Query-conditioned centered residual read over a frozen per-trial bank."""

    name = "A-QMEM"
    training_target_space = plan.TRAINING_TARGET_SPACE

    def __init__(
        self,
        frozen_student: Any,
        *,
        n_heads: int = plan.A_HEADS,
        d_head: int = plan.A_HEAD_DIM,
        d_value: int = plan.A_VALUE_DIM,
        mlp_hidden: int = plan.A_MLP_HIDDEN,
    ) -> None:
        super().__init__()
        plan.require(CONTRACT_VERSION == 1, "A-QMEM implements CONTRACT_VERSION=1")
        plan.require(n_heads * d_head == n_heads * d_value, "head geometry")
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        self.d_value = int(d_value)
        self.mlp_hidden = int(mlp_hidden)
        qkv_dim = self.n_heads * self.d_head
        value_dim = self.n_heads * self.d_value

        # Query projection: 50-bin live window -> 64, then GELU (applied in forward).
        self.query_proj = nn.Linear(plan.WINDOW, plan.HIDDEN_DIM)
        q_in = plan.HIDDEN_DIM + plan.IDENTITY_DIM + plan.T4_DIM
        k_in = plan.HIDDEN_DIM + plan.T4_DIM
        self.Q = _two_layer_mlp(q_in, mlp_hidden, qkv_dim)
        self.K = _two_layer_mlp(k_in, mlp_hidden, qkv_dim)
        self.V = nn.Linear(plan.HIDDEN_DIM, value_dim, bias=False)
        self.W_out = nn.Linear(value_dim, plan.IDENTITY_DIM, bias=False)
        nn.init.zeros_(self.W_out.weight)

        self.frozen_student = frozen_student
        self._freeze_student()

    def _freeze_student(self) -> None:
        if hasattr(self.frozen_student, "eval"):
            self.frozen_student.eval()
        if hasattr(self.frozen_student, "parameters"):
            for param in self.frozen_student.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> AQMEM:
        super().train(mode)
        self._freeze_student()
        return self

    def eval(self) -> AQMEM:
        super().eval()
        self._freeze_student()
        return self

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        allow: dict[str, nn.Parameter] = {}
        for root in _ALLOWLIST_ROOTS:
            module = getattr(self, root)
            for name, param in module.named_parameters():
                allow[f"{root}.{name}"] = param
        return allow

    def compute_delta(
        self,
        X: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        *,
        force_uniform_attention: bool = False,
    ) -> torch.Tensor:
        delta, _ = self._read(X, bank, unit_mask, force_uniform_attention)
        return delta

    def identities(
        self,
        X: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
        *,
        force_uniform_attention: bool = False,
    ) -> torch.Tensor:
        _, identity = self._read(X, bank, unit_mask, force_uniform_attention)
        return identity

    def forward_last(
        self,
        X: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B, 2] in decoder_raw. Do not wrap decode in no_grad()."""
        identity = self.identities(X, bank, unit_mask)
        # Frozen student stays in eval(); the decode path remains differentiable
        # w.r.t. identity so W_out / Q / K / V can train.
        decoded = self.frozen_student.decode_with_identity(X, identity)
        return decoded[:, -1, :]

    def _read(
        self,
        X: torch.Tensor,
        bank: SessionBank,
        unit_mask: torch.Tensor | None,
        force_uniform_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plan.require(X.ndim == 3 and X.shape[1] == plan.WINDOW, "X must be [B,50,N]")
        plan.require(bank.frozen_u is not None, "SessionBank.frozen_u is required")
        if unit_mask is None:
            unit_mask = bank.unit_mask

        # X: [B, 50, N] -> per-unit 50-bin query -> s: [B, N, 64]
        s = F.gelu(self.query_proj(X.permute(0, 2, 1)))
        batch, num_units, _ = s.shape
        e0 = bank.E0
        t = bank.T
        u = bank.frozen_u
        plan.require(u.shape[1] == num_units == e0.shape[0] == t.shape[0], "unit axis")
        plan.require(e0.shape[-1] == plan.IDENTITY_DIM, "E0 width")
        plan.require(t.shape[-1] == plan.T4_DIM, "T width")

        # Native chronological mean over the trial axis. Not a rewrite of E0.
        h_bar = u.mean(dim=0)
        u_centered = u - h_bar
        k_trials = int(u.shape[0])

        e0_b = e0.unsqueeze(0).expand(batch, -1, -1)
        t_b = t.unsqueeze(0).expand(batch, -1, -1)
        q = self.Q(torch.cat((s, e0_b, t_b), dim=-1))
        q = q.view(batch, num_units, self.n_heads, self.d_head)

        t_k = t.unsqueeze(0).expand(k_trials, -1, -1)
        k = self.K(torch.cat((u_centered, t_k), dim=-1))
        k = k.view(k_trials, num_units, self.n_heads, self.d_head)
        v = self.V(u_centered).view(k_trials, num_units, self.n_heads, self.d_value)

        scale = float(self.d_head) ** 0.5
        logits = torch.einsum("bnhd,knhd->bnhk", q, k) / scale
        if force_uniform_attention:
            weights = torch.full_like(logits, 1.0 / float(k_trials))
        else:
            weights = torch.softmax(logits, dim=-1)
        # Centered residual attention: uniform weights cancel exactly.
        centered = weights - (1.0 / float(k_trials))
        residual = torch.einsum("bnhk,knhd->bnhd", centered, v)
        residual = residual.reshape(batch, num_units, self.n_heads * self.d_value)
        delta = self.W_out(F.gelu(residual))
        if unit_mask is not None:
            delta = delta * unit_mask.to(device=delta.device, dtype=delta.dtype).view(1, num_units, 1)
        identity = e0_b + delta
        return delta, identity


def stage0_a(device: str = "cpu") -> dict[str, Any]:
    """Run the A-QMEM correctness suite and write stage0_a.json."""
    assert_contract_version(1)
    dest = plan.active_run_root() / "stage0" / "stage0_a.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if os.environ.get("_M2_STAGE0_A_RUNNING") == "1":
        return {"pass": True, "nested": True, "device": device, "contract_version": CONTRACT_VERSION}

    test_path = Path(__file__).resolve().parents[2] / "tests" / "test_m2_dual_track_a_v1.py"
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.update(plan.REQUIRED_ENV)
    env["_M2_STAGE0_A_RUNNING"] = "1"
    env.setdefault("PYTHONPATH", str(plan.REPO_ROOT))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = "0"

    proc = subprocess.run(
        [plan.PYTHON, "-m", "pytest", str(test_path), "-q", "--tb=short"],
        cwd=str(plan.REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    report: dict[str, Any] = {
        "schema": plan.SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "owner": "A",
        "arm": "A-QMEM",
        "device": device,
        "pass": proc.returncode == 0,
        "pytest_returncode": int(proc.returncode),
        "pytest_stdout": stdout[-8000:],
        "pytest_stderr": stderr[-4000:],
        "precision": plan.A_PRECISION,
        "heads": plan.A_HEADS,
        "d_head": plan.A_HEAD_DIM,
        "d_value": plan.A_VALUE_DIM,
        "training_target_space": plan.TRAINING_TARGET_SPACE,
        "query_uses_full_window": True,
        "prefix_causal_claim": False,
        "w_out_zero_init": True,
        "signed_zero": "canonicalize with y + 0.0 before bitwise equal",
        "finished": datetime.now(timezone.utc).isoformat(),
        "receipt": str(dest),
    }
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
