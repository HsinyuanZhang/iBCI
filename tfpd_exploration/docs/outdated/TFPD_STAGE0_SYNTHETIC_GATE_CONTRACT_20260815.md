# TFPD Stage-0 synthetic gate contract

**Status:** frozen implementation authority for Stage 0. CPU only; no real neural data,
no NWB access, no GPU, no target scoring.
**Amendment history:** three pre-receipt amendments (math freeze, gate fairness and
non-vacuousness, transactional receipt runner) consolidated below. No Stage-0 receipt
existed when any amendment was made, so no receipt is invalidated.
**Parent direction:**
[`HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md`](../../sua_exploration/docs/HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md) §8.1.

## 1. Frozen model mathematics

### 1.1 Priority-1 `BilinearTaskFrameDecoder`

```
z_t   = (1/N) * sum_i (U f(x_i[t-W:t])) (.) (V g(c_i))
y_hat = D_psi(z)      D_psi = causal GRU + linear head
```

- `f`: Linear(W→H) → ReLU → Linear(H→r) → Tanh. Shared across units, causal by
  trailing-window construction.
- `g`: affine **with bias** (Linear 4→k). `U`, `V`: **bias-free** linears.
- **Mean normalization `1/N`**, replacing the handoff's illustrative `1/sqrt(N)`:
  shared `f` makes unit features positively correlated, so the sum's variance grows
  ~N and only the mean keeps the latent scale N-stable (frozen by gate G2).
- **Zero-carrier fallback semantics (frozen):** with `c = 0`, `g(0) = bias_g` is a
  constant per-unit embedding, so `V g(0)` reduces the read-in to a carrier-content-
  free population mean of activity features. A `V` bias would add a
  carrier-independent constant latent and is excluded.

### 1.2 Priority-2 `LearnedPopulationVectorDecoder`

```
act_i(t)   = softplus(f(x_i[t-W:t]))         non-negative by freeze
conf_i     = sigmoid(MLP(c_i))               in (0, 1)
dir_i      = [a_i, c_i] / max(m_i, eps)
mass_t     = sum_i conf_i * act_i(t)         >= 0 by construction
pv_t       = sum_i conf_i * act_i(t) * dir_i / (mass_t + eps)
pop_conf_t = log(1 + mass_t)
```

- **Single denominator:** `pv` and `pop_conf` share the one mass statistic and the
  one `eps = 1e-6`; there is no second normalization constant in the model.
- **Zero-carrier fallback semantics (frozen):** `dir = 0`, `conf = sigmoid(bias)`
  constant, so the fallback channel is the population rate alone (`pv = 0`,
  `pop_conf = log(1 + const * sum act)`).

Arms are inputs, never flags: aligned carrier / zero carrier (`zeros_like`) /
wrong-pair carrier (row-permuted, one frozen permutation per session seed, shared
between that arm's training and evaluation). All reuse identical trained weights.

## 2. Synthetic data generator

Sessions of variable unit count `N ~ U[24, 192]` and length `T = 256`. A smooth 2-D
trajectory generates per-unit Poisson rates
`lambda_i(t) = softplus(b_i + a_i cos(phi_t - theta_i)) + 0.05`, with a hard
positivity assertion at generation time. The carrier is a noisy closed-form T4-style
estimate `[a_hat, c_hat, m_hat, b_hat]` fitted from **support bins only** (first 30%),
then noise drawn from the session rng. A `query_perturb` generator parameter perturbs
query counts **before** the carrier stage without changing the rng draw order, so the
emitted noisy carrier must remain bit-identical (gate G4b). Carrier-free inference is
possible only through activity statistics, which carry no session-stable direction
information by construction.

## 3. Frozen gates (all five required, both models)

| # | gate | pass criterion |
|---|---|---|
| G1 | unit-permutation invariance | `max |y(x,c) - y(x[:,:,perm], c[:,perm])| < 1e-5` over 100 random perms |
| G2 | variable-N support | finite output and finite input-gradients at `N in {24,96,192}`; output-variance ratio across N `< 10` |
| G3 | finite, live, genuinely branched gradients | every parameter has a finite grad; the **branch's own parameters** (`activity_encoder`+`readin_activity` / `carrier_map`+`readin_carrier`; PV: `activity` / `confidence`) have positive grad norm; the **input tensors** `x` and `carrier` both have positive finite grad norm |
| G4 | causality + support-only carrier (actual noisy pipeline) | (a) perturbing every bin after `q = T/2` leaves outputs at `t <= q` bit-identical (max diff exactly 0); (b) generating the same session with `query_perturb = 3` reproduces the identical noisy carrier (max diff exactly 0) while query counts verifiably changed. *(The original older-bin-invariance wording was unsatisfiable for any recurrent temporal decoder and is withdrawn.)* |
| G5 | attainable carrier effect, fair and non-vacuous | three arms (aligned / zero / wrong-pair) trained from **one deep-copied initial state**, 400 Adam steps on 8 sessions; **query-region R2 is the only headline** (support-region R2 reported separately, per session); pass requires `aligned - zero >= +0.10` AND `aligned - wrong_pair >= +0.10` on the query headline |

## 4. Receipt runner properties (frozen)

`scripts/run_stage0_synth.py`:

1. **CPU/env hard gate:** refuses to start unless `sys.flags.no_user_site` is true,
   `CUDA_VISIBLE_DEVICES` is the empty string, and `torch.cuda.is_available()` is
   false (exit 3).
2. **Launch/final closure equality:** SHA-256 closure over `src/tfpd/*.py`, the
   runner, and the test file, hashed before the gates and re-hashed before writing;
   drift writes a `FAIL_SOURCE_CLOSURE_DRIFT` receipt and exits nonzero.
3. **Transactional write:** receipt body written to a private temp file, fsynced,
   hard-linked into place with `O_EXCL` (an existing receipt can never be
   overwritten); sidecar written by the same procedure.
4. **Failure semantics:** gate failure or execution error still writes a failure
   receipt transactionally and exits nonzero; no partial pass exists.
5. **Independent loader:** the runner imports `src.tfpd` directly from disk paths
   and never depends on pytest session state.

## 5. Single authoritative run discipline

Once the bound bytes are frozen: run the pytest suite exactly once (no edits to any
bound file while it runs); if and only if it passes, run the authoritative receipt
runner under the same closure. The runner re-executes the gates itself; its receipt
is the only Stage-0 evidence.

## 6. Kill meaning

Failing any gate kills the corresponding Stage-1 promotion until the implementation
is fixed and the whole suite is rerun from a fresh receipt root. A fully green
Stage-0 receipt authorizes only the drafting of a Stage-1 seed-42 matched-pair
contract for separate review.
