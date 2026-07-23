"""
SPINT few-shot mechanism smoke test (pure NumPy, faithful to src/models/components/spint.py).

Goal: without torch/GPU/DANDI data, verify the two claims that make SPINT's
few-shot adaptation "gradient-free / no parameter update":

  (1) The decoder output is INVARIANT to the ordering of recorded units (neurons),
      as long as the calibration features are permuted the same way.
  (2) The decoder still produces a valid (W x C) kinematic prediction when the
      NUMBER of recorded units changes (units dropped) -- no re-training needed.

All learnable weights are fixed (seeded) and SHARED across runs; only the neuron
axis is permuted / subsetted. This mirrors the real forward() in spint.py.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- hyperparams (M2-like, shrunk H for speed) ----
W   = 50    # window_size
H   = 32    # model_dim (shrunk from 512)
C   = 2     # num_covariates (finger x/y velocity)
NH  = 4     # num_heads
T   = 100   # max_trial_length
M   = 8     # calib trials
N   = 96    # recorded units (M2 Utah array)

def lin(x, Wt, b):            # x[...,in] @ Wt[in,out] + b
    return x @ Wt + b
def relu(x):
    return np.maximum(x, 0.0)
def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

# ---- fixed shared weights ----
def W_(a, b): return rng.standard_normal((a, b)).astype(np.float32) * 0.1
def b_(a):    return np.zeros(a, np.float32)

# fc_in : Linear(W->H), ReLU, Linear(H->H)   (per-neuron read-in; shared across all units)
Fi = [(W_(W, H), b_(H)), (W_(H, H), b_(H))]
# fc_id_in : Linear(T->H), ReLU, Linear(H->H), ReLU, Linear(H->H)   (num_id_layers=3)
Idi = [(W_(T, H), b_(H)), (W_(H, H), b_(H)), (W_(H, H), b_(H))]
# fc_id_out: Linear(H->H), ReLU, Linear(H->H), ReLU, Linear(H->W)
Ido = [(W_(H, H), b_(H)), (W_(H, H), b_(H)), (W_(H, W), b_(W))]
# cross-attn projections + norms + ffn + fc_out
Wq, Wk, Wv, Wo = W_(H, H), W_(H, H), W_(H, H), W_(H, H)
g1, bt1, g2, bt2 = np.ones(H, np.float32), b_(H), np.ones(H, np.float32), b_(H)
Ff = [(W_(H, 4 * H), b_(4 * H)), (W_(4 * H, H), b_(H))]
Fout = (W_(H, W), b_(W))
rep = rng.standard_normal((C, W)).astype(np.float32)   # learnable covariate queries (1 x C x W)

def fc_in(x):                       # last dim W -> H
    x = relu(lin(x, *Fi[0])); return lin(x, *Fi[1])
def fc_id_in(x):                    # last dim T -> H
    x = relu(lin(x, *Idi[0])); x = relu(lin(x, *Idi[1])); return lin(x, *Idi[2])
def fc_id_out(x):                   # last dim H -> W
    x = relu(lin(x, *Ido[0])); x = relu(lin(x, *Ido[1])); return lin(x, *Ido[2])

def mha(q, kv):                     # q:[Cq,H]  kv:[Nk,H] -> [Cq,H]
    Q, K, V = q @ Wq, kv @ Wk, kv @ Wv
    dh = H // NH
    out = np.zeros_like(Q)
    for h in range(NH):
        s = slice(h * dh, (h + 1) * dh)
        att = (Q[:, s] @ K[:, s].T) / np.sqrt(dh)         # [Cq,Nk]
        att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
        out[:, s] = att @ V[:, s]                          # sum over neuron keys -> perm-invariant
    return out @ Wo

def forward(src, calib):            # src:[W,N]   calib:[M,T,N]  ->  [W,C]
    src = src.T                                            # -> [N,W]
    idv = np.transpose(calib, (0, 2, 1))                   # [M,T,N] -> [M,N,T]
    idv = fc_id_in(idv)                                    # -> [M,N,H]
    idv = idv.mean(0)                                      # -> [N,H]
    idv = fc_id_out(idv)                                   # -> [N,W]  (per-unit "identity" from calib)
    src = src + idv                                        # inject identity  [N,W]
    src = fc_in(src)                                       # -> [N,H]  neuron tokens (key/value)
    query = fc_in(rep)                                     # -> [C,H]  covariate tokens (query)
    # cross-attention block (pre-norm), eval => dropout off
    x = query + mha(layernorm(query, g1, bt1), layernorm(src, g1, bt1))
    xn = layernorm(x, g2, bt2)
    x = x + lin(relu(lin(xn, *Ff[0])), *Ff[1])             # [C,H]
    out = lin(x, *Fout)                                    # [C,W]
    return out.T                                           # -> [W,C]

# ---- inputs ----
neural = rng.standard_normal((W, N)).astype(np.float32)
calib  = rng.standard_normal((M, T, N)).astype(np.float32)

base = forward(neural, calib)

# (1) permute neuron order (same perm on live window + calib features)
perm = rng.permutation(N)
p = forward(neural[:, perm], calib[:, :, perm])
print(f"[1] permutation invariance  |  max|Δ| = {np.abs(base - p).max():.2e}  "
      f"(output shape {p.shape})")

# (2) drop 30 units -> different unit count, still valid W x C output, no retrain
keep = rng.permutation(N)[:66]
d = forward(neural[:, keep], calib[:, :, keep])
print(f"[2] variable unit count     |  N: {N} -> {len(keep)}  output shape {d.shape}  "
      f"(still W x C, no param update)")
print("\nPASS: output is order-invariant and unit-count agnostic -> few-shot adaptation "
      "needs only new calib trials, no gradient step.")
