"""TKD (tuning-keyed decoder) model, M2 instance, CPU float32 deterministic.

Implements SPEC_FABLE_TKD_M2_V1_IMPL sections 3.1-3.4:

* psi: per-unit shared causal Conv1d(1->16, k=10) -> Linear(16->64).
* key k = Phi_k(t) (+ eps * P u content term for the learnable arm).
* value v = rho * (u - mu(t)) / sigma(t); mu/sigma are LINEAR maps of t
  (deviation D1, spec-approved: sigma uses clamp-min instead of softplus so
  the PV anchor can recover the raw b/m columns exactly by affine inverse).
* one latent cross-attention layer with L=8 learnable queries.
* time model: 2 diagonal-linear SSM layers (S4D/LRU type) with GLU and a
  zero-initialized learnable residual gain gamma, or a GRU(128).
* readout Linear(128->2), prediction divided by BEHAVIOR_SCALE.

PV-anchor initialization (init="pv") makes the whole network exactly equal a
static PV soft-binning function of the T4 descriptors at step 0 (spec
section 3.3; Phi_k hidden-layer bypass is deviation D2, spec-approved).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import plan

_ATTENTION_TEMP = 8.0  # sqrt(D_K) per spec section 3.1 item 4.
#: D8 (planner-directed): sigma floor = held-in pooled raw-m median; see
#: plan.SIGMA_FLOOR / plan.SIGMA_FLOOR_SOURCE.
#: Positive ReLU shift making hidden dims 0..3 an exact affine carrier of t.
_PHI_BYPASS_SHIFT = 50.0
#: Small positive bias keeping hidden dims 4..63 alive for training while
#: contributing exactly zero at init (their W2 columns start at zero).
_PHI_TAIL_BIAS = 0.01
#: D9/D15 (planner-directed): PV-init decay target a = 0.9 under the D15
#: parametrization a = 0.98 * sigmoid(a_log_raw) (bounded away from 1,
#: gradients alive).  gamma = 0 keeps the layer an exact bitwise
#: pass-through at init regardless of a.
_SSM_PV_DECAY = 0.9
#: D15: decay cap (a = 0.98 * sigmoid(a_log_raw)).
_SSM_DECAY_CAP = 0.98
#: D15: hard state clamp on the recurrence h (applied identically in the
#: parallel scan and the step).
_SSM_H_CLAMP = 10.0
#: D9: GLU gate bias +2.0 (sigmoid'(2) ~ 0.10, healthier gradients than +4).
_SSM_PV_GATE_BIAS = 2.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class DiagSSM(nn.Module):
    """One diagonal-linear SSM layer (S4D/LRU type) with GLU and residual.

    h_t = diag(a) h_{t-1} + B z_t          (a = exp(-softplus(a_log)) in (0,1))
    o_t = GLU(C h_t)                        (Linear(d->2d), gated half)
    out_t = z_t + gamma * o_t               (gamma learnable scalar)

    ``forward`` is the parallel (training) form; ``step`` is the O(d)
    recurrence used for streaming; stage0 A2 asserts parity <= 1e-6.
    """

    def __init__(self, d: int, *, pv_init: bool) -> None:
        super().__init__()
        self.d = int(d)
        # D15: a = 0.98 * sigmoid(a_log_raw), bounded away from 1.
        self.a_log_raw = nn.Parameter(torch.empty(d))
        self.B = nn.Parameter(torch.empty(d, d))
        self.C = nn.Parameter(torch.empty(d, d))
        self.glu = nn.Linear(d, 2 * d)
        # D13(c): gamma = tanh(gamma_raw) keeps the PV-init exact pass-through
        # (gamma_raw = 0) while bounding the residual loop gain in (-1, 1).
        self.gamma_raw = nn.Parameter(torch.empty(()))
        self.pv_init = bool(pv_init)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d = self.d
        if self.pv_init:
            with torch.no_grad():
                # D9/D15: a = 0.9 via a = 0.98 * sigmoid(a_log_raw)
                # (trainable timescale; gamma=0 below still makes the layer
                # an exact pass-through at init).
                self.a_log_raw.fill_(_logit(_SSM_PV_DECAY / _SSM_DECAY_CAP))
                self.B.copy_(torch.eye(d))
                self.C.copy_(torch.eye(d))
                # GLU: value half copies input, gate half pushed open; the
                # gamma=0 residual makes the layer an exact pass-through
                # regardless of the gate value (belt and suspenders).
                self.glu.weight.zero_()
                self.glu.weight[:d, :].copy_(torch.eye(d))
                self.glu.bias.zero_()
                self.glu.bias[d:].fill_(_SSM_PV_GATE_BIAS)
                self.gamma_raw.zero_()
        else:
            with torch.no_grad():
                self.a_log_raw.fill_(_logit(0.9 / _SSM_DECAY_CAP))
                self.B.normal_(0.0, 0.05)
                self.C.normal_(0.0, 0.05)
                self.gamma_raw.fill_(math.atanh(0.9))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.tanh(self.gamma_raw)

    def decay(self) -> torch.Tensor:
        return _SSM_DECAY_CAP * torch.sigmoid(self.a_log_raw)

    def step(self, h: torch.Tensor, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step; h/z_t are [B, d]."""
        a = self.decay()
        h_new = torch.clamp(a * h + z_t @ self.B.T, -_SSM_H_CLAMP, _SSM_H_CLAMP)
        out = self._glu_output(h_new @ self.C.T)
        return h_new, z_t + self.gamma * out

    def _glu_output(self, projected: torch.Tensor) -> torch.Tensor:
        """GLU on the C-projected state: Linear(d->2d), gated half split."""
        d = self.d
        gate_in = self.glu(projected)
        value = gate_in[..., :d]
        gate = torch.sigmoid(gate_in[..., d:])
        return value * gate

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Parallel (kernel) form: z is [B, T, d], returns [B, T, d]."""
        bsz, steps, d = z.shape
        plan.require(d == self.d, "SSM input width drift")
        a = self.decay()
        u = z @ self.B.T  # [B, T, d]
        # kernel[n, c] = a_c^n : depthwise causal FIR per channel.  F.conv1d
        # cross-correlates (it looks forward), so the kernel is time-reversed
        # and padded by T-1 to obtain the causal convolution
        # h[t, c] = sum_n a_c^n * u[t-n, c].
        ones = torch.ones(1, d, dtype=a.dtype, device=a.device)
        powers = torch.cumprod(torch.cat([ones, a.repeat(steps - 1, 1)], dim=0), dim=0)
        weight = powers.T.reshape(d, 1, steps).flip(-1)  # [d, 1, T]
        h = F.conv1d(
            u.transpose(1, 2),  # [B, d, T]
            weight,
            padding=steps - 1,
            groups=d,
        )[..., :steps].transpose(1, 2)  # [B, T, d]
        # D15: identical hard state clamp as in step().
        h = torch.clamp(h, -_SSM_H_CLAMP, _SSM_H_CLAMP)
        out = self._glu_output(h @ self.C.T)
        return z + self.gamma * out


class TKD(nn.Module):
    """Tuning-keyed decoder with frozen-arm switches (spec section 3)."""

    _EPSILON_MODES = ("learnable", "zero")
    _KEY_MODES = ("identity", "shuffled", "pool")
    _TIME_MODELS = ("ssm", "gru")
    _INITS = ("pv", "default", "nmf")

    def __init__(
        self,
        *,
        epsilon: str = "learnable",
        key_mode: str = "identity",
        time_model: str = "ssm",
        init: str = "default",
        frozen_readin: bool = False,
        channels: int = plan.CHANNELS,
        window: int = plan.WINDOW,
        out_dim: int = plan.OUT_DIM,
        behavior_scale: float = plan.BEHAVIOR_SCALE,
        shuf_perm: Any = None,
        pv_beta: float | None = None,
        pv_bin_masses: Any = None,
        authority_mean: Any = None,
        authority_std: Any = None,
        pv_output_scale: float = 1.0,
        pv_readout_bias: Any = None,
        nmf_anchor: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        plan.require(epsilon in self._EPSILON_MODES, f"unknown epsilon mode {epsilon}")
        plan.require(key_mode in self._KEY_MODES, f"unknown key mode {key_mode}")
        plan.require(time_model in self._TIME_MODELS, f"unknown time model {time_model}")
        plan.require(init in self._INITS, f"unknown init {init}")
        self.epsilon_mode = epsilon
        self.key_mode = key_mode
        self.time_model_kind = time_model
        self.init_kind = init
        self.channels = int(channels)
        self.window = int(window)
        self.out_dim = int(out_dim)
        self.behavior_scale = float(behavior_scale)

        # psi: per-unit shared causal temporal filter.
        self.psi_conv = nn.Conv1d(1, plan.CONV_CHANNELS, plan.CONV_KERNEL, padding=9)
        self.psi_linear = nn.Linear(plan.CONV_CHANNELS, plan.D_V)
        # Phi_k: Linear(4->64) -> ReLU -> Linear(64->64); hidden bypass at pv init.
        self.phi_in = nn.Linear(4, plan.D_K)
        self.phi_out = nn.Linear(plan.D_K, plan.D_K)
        # content-gating term (used only when epsilon mode is learnable).
        # epsilon is zero-initialized by design; P must NOT be zero -- the
        # (eps=0, P=0) pair is a mutual saddle (dL/deps ~ P.u = 0 and
        # dL/dP ~ eps = 0) that permanently kills the A' content term.
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.P = nn.Parameter(torch.randn(plan.D_K, plan.D_V) * 0.05)
        # value normalization maps (linear in t; deviation D1).
        self.mu_map = nn.Linear(4, 1)
        self.sig_map = nn.Linear(4, 1)
        self.queries = nn.Parameter(torch.empty(plan.N_QUERIES, plan.D_K))
        self.merge = nn.Linear(plan.N_QUERIES * plan.D_V, plan.D_H)
        if time_model == "ssm":
            self.ssm_layers = nn.ModuleList(
                DiagSSM(plan.D_H, pv_init=init in ("pv", "nmf"))
                for _ in range(plan.SSM_LAYERS)
            )
            self.gru: nn.GRU | None = None
        else:
            self.ssm_layers = None
            self.gru = nn.GRU(plan.D_H, plan.D_H, num_layers=1, batch_first=True)
        self.readout = nn.Linear(plan.D_H, self.out_dim)
        perm_source = plan.SHUF_PERM if shuf_perm is None else shuf_perm
        perm = torch.as_tensor(perm_source, dtype=torch.long).reshape(-1)
        plan.require(perm.numel() == self.channels, "shuf perm length drift")
        self.register_buffer("shuf_perm", perm, persistent=False)

        if init == "nmf":
            # M1 generative-anchor init (WORKORDER_FABLE_TKD_M1_V1 section 2):
            # a fully-cooked closed-form construction dict is installed; the
            # numbers are computed by the fable_tkd_m1_v1 data plane.
            plan.require(nmf_anchor is not None, "init='nmf' requires nmf_anchor")
            plan.require(authority_mean is not None and authority_std is not None,
                         "init='nmf' requires the carrier normalizer")
            mean = torch.as_tensor(authority_mean, dtype=torch.float32).reshape(4)
            std = torch.as_tensor(authority_std, dtype=torch.float32).reshape(4)
            self._apply_nmf_init(nmf_anchor, mean, std)
        elif init == "pv":
            beta = float(plan.PV_BETA if pv_beta is None else pv_beta)
            if pv_bin_masses is None:
                masses = torch.full((plan.N_QUERIES,), self.channels / plan.N_QUERIES)
            else:
                masses = torch.as_tensor(pv_bin_masses, dtype=torch.float32).reshape(-1)
                plan.require(masses.numel() == plan.N_QUERIES, "bin mass count drift")
                plan.require(bool((masses > 0).all()), "bin masses must be positive")
            plan.require(authority_mean is not None and authority_std is not None,
                         "init='pv' requires the T4 authority mean/std")
            mean = torch.as_tensor(authority_mean, dtype=torch.float32).reshape(4)
            std = torch.as_tensor(authority_std, dtype=torch.float32).reshape(4)
            plan.require(bool(torch.isfinite(mean).all() and torch.isfinite(std).all())
                         and bool((std > 0).all()), "authority normalizer drift")
            bias = (None if pv_readout_bias is None
                    else torch.as_tensor(pv_readout_bias, dtype=torch.float32).reshape(plan.OUT_DIM))
            self._apply_pv_init(beta, masses, mean, std,
                                output_scale=float(pv_output_scale), readout_bias=bias)
        if frozen_readin:
            self.freeze_readin()

    # ------------------------------------------------------------------
    # PV-anchor initialization (spec section 3.3).
    # ------------------------------------------------------------------
    def _apply_pv_init(
        self,
        beta: float,
        masses: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        output_scale: float = 1.0,
        readout_bias: torch.Tensor | None = None,
    ) -> None:
        cos_psi = [math.cos(angle) for angle in plan.QUERY_DIRECTIONS]
        sin_psi = [math.sin(angle) for angle in plan.QUERY_DIRECTIONS]
        with torch.no_grad():
            # psi: channel 0 = causal 10-bin mean (all-1/10 kernel), others 0;
            # Linear row 0 copies channel 0 -> u[..., 0] = rate10.  Columns
            # 1..15 of the Linear get a tiny nonzero seed: conv channels 1..15
            # output exactly 0 at init (zero kernel + zero bias), so u and the
            # A1 anchor are unchanged bit-exactly, but the channels now
            # receive gradients (with zero columns they were dead forever).
            self.psi_conv.weight.zero_()
            self.psi_conv.weight[0, 0, :].fill_(1.0 / plan.CONV_KERNEL)
            self.psi_conv.bias.zero_()
            self.psi_linear.weight.zero_()
            self.psi_linear.weight[0, 0] = 1.0
            with torch.no_grad():
                self.psi_linear.weight[:, 1:].normal_(0.0, 0.01)
            self.psi_linear.bias.zero_()
            # mu recovers raw b per unit (column 3): raw_b = t3*std3 + mean3.
            # D12: sigma is a single GLOBAL constant S_POOLED (no t
            # dependence; clamp min 1e-3 kept for API stability, inactive).
            self.mu_map.weight.zero_()
            self.mu_map.weight[0, 3] = std[3]
            self.mu_map.bias.fill_(float(mean[3]))
            self.sig_map.weight.zero_()
            self.sig_map.bias.fill_(plan.S_POOLED)
            # Phi_k hidden bypass (deviation D2): hidden dims 0..3 carry
            # t + 50 through the positive ReLU region, tail dims carry a tiny
            # positive constant with zero W2 columns; Phi_k(t) = beta*[a, c, 0...].
            self.phi_in.weight.zero_()
            self.phi_in.weight[0:4, 0:4].copy_(torch.eye(4))
            self.phi_in.bias.zero_()
            self.phi_in.bias[0:4].fill_(_PHI_BYPASS_SHIFT)
            self.phi_in.bias[4:].fill_(_PHI_TAIL_BIAS)
            self.phi_out.weight.zero_()
            self.phi_out.weight[0, 0] = beta * std[0]
            self.phi_out.weight[1, 1] = beta * std[1]
            self.phi_out.bias.zero_()
            self.phi_out.bias[0] = beta * float(mean[0]) - _PHI_BYPASS_SHIFT * beta * float(std[0])
            self.phi_out.bias[1] = beta * float(mean[1]) - _PHI_BYPASS_SHIFT * beta * float(std[1])
            # queries: 8 equidistant unit direction embeddings.
            self.queries.zero_()
            for ell in range(plan.N_QUERIES):
                self.queries[ell, 0] = cos_psi[ell]
                self.queries[ell, 1] = sin_psi[ell]
            # merge: bin dim 0 -> [cos psi_l, sin psi_l] * M_l, others 0.
            self.merge.weight.zero_()
            self.merge.bias.zero_()
            for ell in range(plan.N_QUERIES):
                self.merge.weight[0, ell * plan.D_V] = cos_psi[ell] * masses[ell]
                self.merge.weight[1, ell * plan.D_V] = sin_psi[ell] * masses[ell]
            # readout: identity on the first two channels.  D13(b): the merge
            # rows carry the closed-form output-scale calibration (affine, so
            # the A1 correlation is unchanged up to float rounding) and the
            # readout bias starts at the held-in per-dim target mean (x5).
            self.merge.weight[0:2] *= float(output_scale)
            self.readout.weight.zero_()
            self.readout.weight[0, 0] = 1.0
            self.readout.weight[1, 1] = 1.0
            if readout_bias is None:
                self.readout.bias.zero_()
            else:
                self.readout.bias.copy_(readout_bias)
            # epsilon/P stay at their zero construction default.

    def _apply_nmf_init(
        self, anchor: dict[str, Any], mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """M1 generative-anchor init (identity-as-read-in in its purest form).

        The anchor dict (closed-form, computed by fable_tkd_m1_v1.data):
          merge_u       [N_QUERIES, out_dim] output directions u_l (already
                        carrying the tilt constant and sigma scaling; the
                        auxiliary queries carry -sum(u)/5 so the common-mode
                        term cancels exactly)
          sigma_pooled  global value scale (D12 law)
          mu_weight     [4] / mu_bias scalar recovering the carrier intercept
          readout_bias  [out_dim] held-in target mean (D13b law)
          queries       optional [N_QUERIES, D_K] override (default e_0..e_2
                        then zeros)
          phi_row_weight [3] / phi_row_bias [3] fully-cooked key scales: the
                        bypass installs k_ik = phi_row_weight[k] * t_ik +
                        phi_row_bias[k] on hidden dims 0..2 (the caller folds
                        temperature, centering and unit scaling)
        Keys are (gamma / pooled spread) * (w_ik - pooled mean) of the
        session carrier (Phi_k bypass, the D2 law), so the first-order
        attention tilt realizes the rank-3 map
        D (W^T W + lam I)^-1 W^T (r - b); queries 3..7 are zero (uniform
        alpha) and only serve the common-mode cancellation.
        """
        with torch.no_grad():
            self.psi_conv.weight.zero_()
            self.psi_conv.weight[0, 0, 0] = 1.0  # u[..., 0] = current-bin rate
            self.psi_conv.bias.zero_()
            self.psi_linear.weight.zero_()
            self.psi_linear.weight[0, 0] = 1.0
            self.psi_linear.weight[:, 1:].normal_(0.0, 0.01)  # D11 un-deadening
            self.psi_linear.bias.zero_()
            self.mu_map.weight.zero_()
            self.mu_map.weight[0] = torch.as_tensor(
                anchor["mu_weight"], dtype=torch.float32
            ).reshape(4)
            self.mu_map.bias.fill_(float(anchor["mu_bias"]))
            self.sig_map.weight.zero_()
            self.sig_map.bias.fill_(float(anchor["sigma_pooled"]))
            self.phi_in.weight.zero_()
            self.phi_in.weight[0:4, 0:4].copy_(torch.eye(4))
            self.phi_in.bias.zero_()
            self.phi_in.bias[0:4].fill_(_PHI_BYPASS_SHIFT)
            self.phi_in.bias[4:].fill_(_PHI_TAIL_BIAS)
            self.phi_out.weight.zero_()
            # NOTE: no seeding of phi_out tail columns -- hidden dim 3 carries
            # the raw intercept (large values); the 0.01 tail bias keeps the
            # hidden tail learnable (the D2 law) without key noise.
            row_weight = torch.as_tensor(
                anchor["phi_row_weight"], dtype=torch.float32
            ).reshape(3)
            row_bias = torch.as_tensor(
                anchor["phi_row_bias"], dtype=torch.float32
            ).reshape(3)
            for index in range(3):
                self.phi_out.weight[index, index] = row_weight[index]
                self.phi_out.bias[index] = (
                    row_bias[index] - _PHI_BYPASS_SHIFT * row_weight[index]
                )
            self.queries.zero_()
            override = anchor.get("queries")
            if override is not None:
                self.queries.copy_(
                    torch.as_tensor(override, dtype=torch.float32).reshape(
                        plan.N_QUERIES, plan.D_K
                    )
                )
            else:
                for index in range(3):
                    self.queries[index, index] = 1.0
            merge_u = torch.as_tensor(
                anchor["merge_u"], dtype=torch.float32
            ).reshape(plan.N_QUERIES, self.out_dim)
            self.merge.weight.zero_()
            self.merge.bias.zero_()
            for ell in range(plan.N_QUERIES):
                for j in range(self.out_dim):
                    self.merge.weight[j, ell * plan.D_V] = merge_u[ell, j]
            self.readout.weight.zero_()
            for j in range(min(self.out_dim, plan.D_H)):
                self.readout.weight[j, j] = 1.0
            bias = anchor.get("readout_bias")
            if bias is None:
                self.readout.bias.zero_()
            else:
                self.readout.bias.copy_(
                    torch.as_tensor(bias, dtype=torch.float32).reshape(self.out_dim)
                )

    def freeze_readin(self) -> None:
        """Arm B: freeze everything except the time model and readout."""
        frozen = [self.psi_conv, self.psi_linear, self.phi_in, self.phi_out,
                  self.mu_map, self.sig_map, self.merge]
        for module in frozen:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.queries.requires_grad_(False)
        self.P.requires_grad_(False)
        self.epsilon.requires_grad_(False)

    # ------------------------------------------------------------------
    # Identity plumbing.
    # ------------------------------------------------------------------
    def _consumed_t(self, t: torch.Tensor) -> torch.Tensor:
        if self.key_mode == "shuffled":
            return t[self.shuf_perm]
        if self.key_mode == "pool":
            pooled = t.mean(dim=0)
            return pooled.unsqueeze(0).expand(self.channels, t.shape[1])
        return t

    def compute_key(self, t: torch.Tensor) -> torch.Tensor:
        return self.phi_out(F.relu(self.phi_in(self._consumed_t(t))))

    def compute_mu_sig(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        used = self._consumed_t(t)
        mu = self.mu_map(used).squeeze(-1)
        # D12: sigma = clamp(W_sig @ t + b_sig, min 1e-3); at PV init this is
        # the global constant S_POOLED for every unit (clamp inactive).
        sig = torch.clamp(self.sig_map(used).squeeze(-1), min=1.0e-3)
        return mu, sig

    def precompute_static(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Static (epsilon=0) session cache: keys and attention weights."""
        plan.require(self.epsilon_mode == "zero",
                     "static cache requires the epsilon='zero' arm")
        key = self.compute_key(t)
        scores = torch.einsum("qd,nd->qn", self.queries, key) / _ATTENTION_TEMP
        alpha = torch.softmax(scores, dim=-1)
        return {"key": key, "alpha": alpha}

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def _psi(self, x: torch.Tensor) -> torch.Tensor:
        plan.require(x.dim() == 3 and x.shape[1] == self.window
                     and x.shape[2] == self.channels, "psi input shape drift")
        bsz = x.shape[0]
        units = x.transpose(1, 2).reshape(bsz * self.channels, 1, self.window)
        conv = self.psi_conv(units)  # [B*N, C, W + 9]
        # Cross-correlation indexing: out[i] covers x[i-9 .. i]; the causal
        # 10-tap means ending at bins 0..W-1 are outputs 0..W-1.
        causal = conv[:, :, : self.window]
        filtered = causal.reshape(bsz, self.channels, plan.CONV_CHANNELS,
                                  self.window).permute(0, 3, 1, 2)  # [B, T, N, C]
        return self.psi_linear(filtered)  # [B, T, N, D_V]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        rho: torch.Tensor,
        static: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        plan.require(t.shape == (self.channels, 4), "t shape drift")
        plan.require(rho.shape == (self.channels,), "rho shape drift")
        u = self._psi(x)
        mu, sig = self.compute_mu_sig(t)
        v = rho[None, None, :, None] * (u - mu[None, None, :, None]) / sig[None, None, :, None]
        if static is not None:
            plan.require(self.epsilon_mode == "zero", "static cache misuse")
            z = torch.einsum("qn,btnd->btqd", static["alpha"], v)
        elif self.epsilon_mode == "learnable":
            key_eff = self.compute_key(t)[None, None] + self.epsilon * (u @ self.P.T)
            scores = torch.einsum("qd,btnd->btqn", self.queries, key_eff) / _ATTENTION_TEMP
            alpha_full = torch.softmax(scores, dim=-1)
            z = torch.einsum("btqn,btnd->btqd", alpha_full, v)
        else:
            key = self.compute_key(t)
            scores = torch.einsum("qd,nd->qn", self.queries, key) / _ATTENTION_TEMP
            alpha = torch.softmax(scores, dim=-1)
            z = torch.einsum("qn,btnd->btqd", alpha, v)
        merged = self.merge(z.reshape(z.shape[0], z.shape[1], -1))  # [B, T, D_H]
        if self.ssm_layers is not None:
            hidden = merged
            for layer in self.ssm_layers:
                hidden = layer(hidden)
        else:
            assert self.gru is not None
            hidden, _ = self.gru(merged)
        prediction = self.readout(hidden[:, -1, :]) / self.behavior_scale
        plan.require(prediction.shape == (x.shape[0], self.out_dim), "output shape drift")
        return prediction


def gru_step(
    gru: nn.GRU, x_t: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manual single step of ``nn.GRU`` (stage0 A2 parity partner)."""
    d = gru.hidden_size
    plan.require(x_t.dim() == 2 and x_t.shape[1] == gru.input_size, "gru step input drift")
    plan.require(h.dim() == 3 and h.shape[0] == gru.num_layers
                 and h.shape[-1] == d, "gru step state drift")
    w_ih = gru.weight_ih_l0
    w_hh = gru.weight_hh_l0
    b_ih = gru.bias_ih_l0
    b_hh = gru.bias_hh_l0
    gi = x_t @ w_ih.T + b_ih
    gh = h[0] @ w_hh.T + b_hh
    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    h_new = (1.0 - z) * n + z * h[0]
    out = h_new
    return out.unsqueeze(0), h_new.unsqueeze(0)
