"""B1: depthwise temporal-convolution student (ASIC-friendly, quantization-ready).

Design goals (contrast with SPINT cross-attn and BrainDistill CWT+linear-attn):
  - per-channel LEARNED temporal FIR (depthwise Conv1d, groups=n_ch): shares smooth
    taps across time -> far fewer params than the flat 96*(N_HIST+1) ridge, less overfit.
  - pointwise channel mix (1x1 conv / linear) -> C -> hidden -> n_out.
  - ONLY ReLU + conv + linear: no softmax, no LayerNorm, fully static shapes ->
    strictly more tapeout-friendly than a (linear-)attention student, and better
    matched to binned spikes than a fixed CWT filter bank.
  - fixed n_ch within a deployed session (permutation-invariance is handled OFF-chip
    by the input-side alignment, cf. Plan B step B3).

Input  x: (B, n_ch, window)   window past bins of (aligned) neural activity
Output y: (B, n_out)          last-frame finger x/y velocity
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseTemporalConvStudent(nn.Module):
    def __init__(self, n_ch: int = 96, window: int = 50, kernel: int = 9,
                 hidden: int = 32, n_out: int = 2, depth: int = 1,
                 dilations: list[int] | None = None, residual: bool = False,
                 bias: bool = True):
        super().__init__()
        self.n_ch, self.window, self.kernel = n_ch, window, kernel
        if dilations is None:
            dilations = [1] * depth
        self.dilations = dilations
        self.residual = residual
        # --- depthwise temporal conv stack (each channel its own FIR) ---
        # Causal padding: pad (kernel-1)*dilation on the LEFT only, conv padding=0.
        # Symmetric padding would leave right-side taps on zero padding with zero
        # gradient (dead weights) since forward() reads only the last frame.
        # Dilation spreads taps over a longer history (Plan §3 Step 3): a stack
        # [1,2,4] at k=9 gives RF = 1 + 8*(1+2+4) = 57 frames.
        # Optional per-layer residual (standard TCN): a depthwise causal conv
        # preserves length, so layer_out + layer_in is shape-safe. Without it,
        # stacking depthwise+ReLU (no cross-channel mixing) underfits badly
        # (train R2 0.16 vs 0.52 for depth-1) due to cascading dead units.
        self.pads = nn.ModuleList(
            [nn.ConstantPad1d(((kernel - 1) * dl, 0), 0.0) for dl in dilations])
        self.convs = nn.ModuleList(
            [nn.Conv1d(n_ch, n_ch, kernel_size=kernel, groups=n_ch,
                       padding=0, dilation=dl, bias=bias) for dl in dilations])
        # --- collapse time: take last frame (causal, matches online decode) ---
        # --- pointwise channel mix -> hidden -> out ---
        self.mix = nn.Sequential(
            nn.Linear(n_ch, hidden, bias=bias), nn.ReLU(),
            nn.Linear(hidden, n_out, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, n_ch, window)
        h = x
        for pad, conv in zip(self.pads, self.convs):
            y = F.relu(conv(pad(h)))
            if self.residual:
                y = y + h
            h = y
        h = h[:, :, -1]                                       # (B, n_ch)  last frame
        return self.mix(h)                                    # (B, n_out)

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FilmTCNStudent(nn.Module):
    """Depthwise causal TCN + FiLM conditioning from per-session CORAL stats.

    Same quantization-friendly backbone as DepthwiseTemporalConvStudent, with one
    added session-conditioning path (Plan §4): the CORAL-diag calib vector
    [a, b] (2*n_ch numbers, computed ONCE per session off-model) is fed through a
    small generator that emits per-channel gamma, beta; these affine-modulate the
    conv output's last frame before the pointwise mix.

      h = ReLU(conv(x))[:,:, -1]        # (B, n_ch)
      g,b = film_gen(cond)              # cond=(B, 2*n_ch) -> gamma,beta (B, n_ch)
      h = gamma * h + beta
      y = mix(h)

    The generator runs once per session at deploy (cond is constant across
    frames); per-frame cost is only 2*n_ch elementwise MACs. film_gen's last
    layer is zero-init with bias=[1...1, 0...0] so the model starts as the plain
    TCN and only learns conditioning deviations.
    """

    def __init__(self, n_ch: int = 96, window: int = 50, kernel: int = 9,
                 hidden: int = 32, n_out: int = 2, cond_dim: int | None = None,
                 gen_hidden: int = 32, bias: bool = True, use_beta: bool = True):
        super().__init__()
        self.n_ch, self.window, self.kernel = n_ch, window, kernel
        self.use_beta = use_beta
        cond_dim = cond_dim if cond_dim is not None else 2 * n_ch
        self.temporal = nn.Sequential(
            nn.ConstantPad1d((kernel - 1, 0), 0.0),
            nn.Conv1d(n_ch, n_ch, kernel_size=kernel, groups=n_ch,
                      padding=0, bias=bias),
            nn.ReLU(),
        )
        out_dim = 2 * n_ch if use_beta else n_ch   # gamma (+beta) per channel
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, gen_hidden, bias=bias), nn.ReLU(),
            nn.Linear(gen_hidden, out_dim, bias=bias),
        )
        # identity init: gamma=1, beta=0 -> starts as plain TCN
        with torch.no_grad():
            self.film_gen[-1].weight.zero_()
            bias_init = torch.zeros(out_dim)
            bias_init[:n_ch] = 1.0
            self.film_gen[-1].bias.copy_(bias_init)
        self.mix = nn.Sequential(
            nn.Linear(n_ch, hidden, bias=bias), nn.ReLU(),
            nn.Linear(hidden, n_out, bias=bias),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)[:, :, -1]                  # (B, n_ch)
        g = self.film_gen(cond)                         # (B, n_ch) or (B, 2*n_ch)
        if self.use_beta:
            gamma, beta = g[:, :self.n_ch], g[:, self.n_ch:]
            h = gamma * h + beta
        else:
            h = g * h
        return self.mix(h)

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SETCNStudent(nn.Module):
    """Depthwise causal TCN + Squeeze-Excitation channel attention.

    The cheapest content-dependent channel reweighting — the attention analogue
    without softmax. After the temporal conv, the per-channel temporal summary
    (global avg pool over time) drives a bottleneck (C -> C/r -> C) that emits a
    sigmoid gate per channel, scaling the conv output before last-frame readout.

      h = ReLU(conv(x))                     # (B, C, T)
      s = mean(h, dim=T)                     # squeeze: (B, C)
      s = sigmoid(W2(relu(W1(s))))           # excitation: (B, C)
      h = h * s                              # per-channel scale
      y = mix(h[:,:,-1])

    The gate s changes every frame (content-dependent), unlike FiLM's per-session
    (gamma, beta). Only Linear + sigmoid — no softmax, no LayerNorm. ~2.4k params.
    """

    def __init__(self, n_ch: int = 96, window: int = 50, kernel: int = 9,
                 hidden: int = 32, n_out: int = 2, se_r: int = 8, bias: bool = True):
        super().__init__()
        self.n_ch, self.window, self.kernel = n_ch, window, kernel
        self.temporal = nn.Sequential(
            nn.ConstantPad1d((kernel - 1, 0), 0.0),
            nn.Conv1d(n_ch, n_ch, kernel_size=kernel, groups=n_ch, padding=0, bias=bias),
            nn.ReLU(),
        )
        self.se = nn.Sequential(
            nn.Linear(n_ch, max(n_ch // se_r, 4), bias=bias),
            nn.ReLU(),
            nn.Linear(max(n_ch // se_r, 4), n_ch, bias=bias),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Linear(n_ch, hidden, bias=bias), nn.ReLU(),
            nn.Linear(hidden, n_out, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)                               # (B, C, T)
        s = self.se(h.mean(dim=2))                         # (B, C)
        h = h * s.unsqueeze(2)                             # per-channel scale
        h = h[:, :, -1]                                    # (B, C)
        return self.mix(h)

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GLUTCNStudent(nn.Module):
    """Depthwise causal TCN + Gated Linear Unit in the pointwise mix.

    Replaces the fixed ReLU activation in the channel mix with a learned,
    content-dependent gate:

      h = ReLU(conv(x))[:,:, -1]           # (B, C)
      a, g = Linear(C, 2*hidden)(h)        # split into value + gate
      h' = a * sigmoid(g)                  # gated activation (GLU)
      y = Linear(hidden, n_out)(h')

    The gate sigmoid(g) changes every frame based on neural activity, unlike a
    fixed ReLU threshold. ~3.1k extra params over plain TCN. Quantization-friendly
    (Linear + sigmoid only).
    """

    def __init__(self, n_ch: int = 96, window: int = 50, kernel: int = 9,
                 hidden: int = 32, n_out: int = 2, bias: bool = True):
        super().__init__()
        self.n_ch, self.window, self.kernel = n_ch, window, kernel
        self.hidden = hidden
        self.temporal = nn.Sequential(
            nn.ConstantPad1d((kernel - 1, 0), 0.0),
            nn.Conv1d(n_ch, n_ch, kernel_size=kernel, groups=n_ch, padding=0, bias=bias),
            nn.ReLU(),
        )
        self.proj = nn.Linear(n_ch, 2 * hidden, bias=bias)
        self.out = nn.Linear(hidden, n_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)[:, :, -1]                     # (B, C)
        ag = self.proj(h)                                  # (B, 2*hidden)
        a, g = ag[:, :self.hidden], ag[:, self.hidden:]    # value, gate
        h2 = a * torch.sigmoid(g)                          # GLU
        return self.out(h2)

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GroupedSETCNStudent(nn.Module):
    """Depthwise causal TCN + per-output Squeeze-Excitation (grouped SE).

    The shared-gate SETCN scales all outputs by the SAME per-channel gate s_n:
    out_c = Σ_n W_cn·(s_n·h_n)  — s is output-agnostic. But cross-attention's
    softmax_n(Q_c·K_n) is PER-OUTPUT: finger-x and finger-y can read DIFFERENT
    neuron subpopulations. Neuroscientifically, x/y direction-tuned neurons are
    distinct groups.

    Grouped SE fixes this: the generator emits n_out × C gates, one per channel
    per output. Squeeze (mean over T) stays shared; only the gate is per-output.
    The readout weights are shared across outputs; only the gating differs, so
    each output reads a different neuron-weighted summary.

      h = ReLU(conv(x))                       # (B, C, T)
      z = mean(h, T)                           # shared squeeze (B, C)
      s = sigmoid(W2(relu(W1(z))))             # (B, n_out*C) -> (B, n_out, C)
      h_gated = h[:,:,-1].unsqueeze(1) * s     # (B, n_out, C)
      y_o = head(readout(h_gated[:,o,:]))       # per-output scalar, shared weights
    """

    def __init__(self, n_ch: int = 96, window: int = 50, kernel: int = 9,
                 hidden: int = 32, n_out: int = 2, se_r: int = 8, bias: bool = True):
        super().__init__()
        self.n_ch, self.window, self.kernel = n_ch, window, kernel
        self.n_out, self.hidden = n_out, hidden
        self.temporal = nn.Sequential(
            nn.ConstantPad1d((kernel - 1, 0), 0.0),
            nn.Conv1d(n_ch, n_ch, kernel_size=kernel, groups=n_ch, padding=0, bias=bias),
            nn.ReLU(),
        )
        mid = max(n_ch // se_r, 4)
        self.se = nn.Sequential(
            nn.Linear(n_ch, mid, bias=bias),
            nn.ReLU(),
            nn.Linear(mid, n_out * n_ch, bias=bias),
            nn.Sigmoid(),
        )
        # shared readout + per-output scalar head
        self.readout = nn.Sequential(
            nn.Linear(n_ch, hidden, bias=bias), nn.ReLU(),
            nn.Linear(hidden, 1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)                                # (B, C, T)
        z = h.mean(dim=2)                                   # (B, C) shared squeeze
        s = self.se(z).view(-1, self.n_out, self.n_ch)      # (B, n_out, C)
        h_last = h[:, :, -1]                                # (B, C)
        h_gated = h_last.unsqueeze(1) * s                   # (B, n_out, C)
        y = self.readout(h_gated).squeeze(-1)               # (B, n_out)
        return y

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ChannelGateNoSqueezeStudent(nn.Module):
    """Depthwise causal TCN + channel gate WITHOUT temporal squeeze (deconfound).

    Isolates WHY SE helps: location (channel axis, 96) vs mechanism (global
    temporal context via mean-pool). This model gates at the channel axis like
    SE but computes the gate from the LAST FRAME only (no squeeze over T):

      h = ReLU(conv(x))[:,:, -1]           # (B, C)
      s = sigmoid(Linear(C, C)(h))         # channel gate, no temporal context
      h' = h * s
      y = mix(h')

    If this matches SETCN, location is what matters. If it's worse, the squeeze
    (full-window temporal summary) contributes. Quantization-friendly.
    """

    def __init__(self, n_ch: int = 96, window: int = 50, kernel: int = 9,
                 hidden: int = 32, n_out: int = 2, se_r: int = 8, bias: bool = True):
        super().__init__()
        self.n_ch, self.window, self.kernel = n_ch, window, kernel
        self.temporal = nn.Sequential(
            nn.ConstantPad1d((kernel - 1, 0), 0.0),
            nn.Conv1d(n_ch, n_ch, kernel_size=kernel, groups=n_ch, padding=0, bias=bias),
            nn.ReLU(),
        )
        mid = max(n_ch // se_r, 4)
        self.gate = nn.Sequential(
            nn.Linear(n_ch, mid, bias=bias),
            nn.ReLU(),
            nn.Linear(mid, n_ch, bias=bias),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Linear(n_ch, hidden, bias=bias), nn.ReLU(),
            nn.Linear(hidden, n_out, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)[:, :, -1]                     # (B, C)
        s = self.gate(h)                                   # (B, C) no squeeze
        h = h * s
        return self.mix(h)

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LinearWienerStudent(nn.Module):
    """Baseline = flat temporal FIR == the ridge student (for apples-to-apples).
    input x: (B, n_ch, window) -> flatten -> Linear -> n_out."""
    def __init__(self, n_ch: int = 96, window: int = 50, n_out: int = 2):
        super().__init__()
        self.fc = nn.Linear(n_ch * window, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


if __name__ == "__main__":
    for K in (5, 9, 15):
        m = DepthwiseTemporalConvStudent(kernel=K, hidden=32)
        x = torch.randn(4, 96, 50)
        y = m(x)
        print(f"kernel={K:2d}  out={tuple(y.shape)}  params={m.n_params():,}")
    lw = LinearWienerStudent()
    print(f"linear-wiener  params={sum(p.numel() for p in lw.parameters()):,}")
    # dilated stacks: RF = 1 + (k-1)*sum(dilations)
    for dil in ([1], [1, 1, 1], [1, 2, 4]):
        m = DepthwiseTemporalConvStudent(dilations=dil, hidden=32)
        rf = 1 + (9 - 1) * sum(dil)
        print(f"dilations={dil}  RF={rf:2d}  params={m.n_params():,}  out={tuple(m(torch.randn(4,96,50)).shape)}")
    # FiLM student
    fm = FilmTCNStudent(hidden=32)
    cond = torch.randn(4, 2 * 96)
    print(f"FilmTCN  params={fm.n_params():,}  out={tuple(fm(torch.randn(4,96,50), cond).shape)}")
    # SE-TCN (content-dependent channel attention)
    se = SETCNStudent(hidden=32)
    print(f"SETCN    params={se.n_params():,}  out={tuple(se(torch.randn(4,96,50)).shape)}")
    # GLU-TCN (gated activation in mix)
    gl = GLUTCNStudent(hidden=32)
    print(f"GLUTCN   params={gl.n_params():,}  out={tuple(gl(torch.randn(4,96,50)).shape)}")
    # grouped SE (per-output gates)
    gse = GroupedSETCNStudent(hidden=32, n_out=2)
    print(f"GroupSE  params={gse.n_params():,}  out={tuple(gse(torch.randn(4,96,50)).shape)}")
    # channel gate without squeeze (deconfound)
    cns = ChannelGateNoSqueezeStudent(hidden=32)
    print(f"ChNoSqz  params={cns.n_params():,}  out={tuple(cns(torch.randn(4,96,50)).shape)}")

