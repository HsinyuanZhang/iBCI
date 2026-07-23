# TCN Student Scaling Plan — closing the gap to SPINT

> Analysis/design doc for Plan-B student work: *what to build and how to judge it*.
> §0–§9 are DONE (c1–c5, results consolidated below). §10 is the live plan.
> Per Plan-B split: this doc specifies; the implementer codes + runs.

## 0. Where we stand

| Model | Params | sat R² (ref_raw, held-out) |
|---|---:|---:|
| SPINT teacher (cross-attn) | 4,569,288 | 0.26 (≈ LOO oracle 0.268) |
| **SETCN student (BEST on-chip)** | **6,542** | **0.196** |
| plain TCN student (k9,W50,h32,d1,causal) | 4,130 | 0.163–0.168 ± 0.03 |
| ridge N_HIST=50 / =7 | 9.8k / 1.5k | 0.116 / 0.11 |
| ridge-CORAL | — | 0.114 |

Framing facts:
1. **The TCN edge is model-class, not window.** Ridge at a matched 50-frame window
   still gets only 0.116; after the causal-pad fix the TCN effectively uses ~9
   frames yet reaches 0.167 — the gain over ridge is *nonlinearity*, not more time.
2. **0.26 ≈ ceiling** (teacher sits on the LOO oracle 0.268). The deliverable is the
   **Pareto knee** (R² per param under the on-chip budget), not matching 0.26.
3. Plain-TCN param split: depthwise conv 960 (thin) + mix FC `96→32→2` 3,170 (77%, fat).

---

## 1–6. Original plan (executed — superseded by §7–§9 results)

The pre-execution plan hypothesized the gap was **session-conditioning, not FFN
width** (§2), and proposed a gated ladder: causal-pad fix (free) → width probe →
depthwise multiplier → dilated RF → deeper mix → FiLM-from-CORAL as the "smart
lever." **Causal-pad fix landed** (`models/tcn_student.py`; left-pad `kernel-1`,
read last frame — the old symmetric pad left 4 dead taps and clipped RF to 5
frames). It also exposed that the original **0.20 was a single un-seeded lucky
draw**; the seeded baseline is **0.167 ± 0.032**, the yardstick for everything
since. Every other lever in that plan was then tested and is reported below.

**Validation protocol (unchanged, applies to all probes):** ref_raw **saturated**
R², **3 seeds × 6 held-out sessions**, few-shot budgets 200/800/3200/∞, same
folds as B3. Report `n_params()`, sat R², R²@800 (16 s, realistic calib), and
**paired Δ vs plain** (paired = same seed/split; this is the correct noise floor —
between-seed marginal std ≈0.03 overstates it ~4×). **Hardware guardrail:** conv +
ReLU + linear + sigmoid only — no softmax / no LayerNorm / no dynamic shapes.

---

## 7–9. Results so far (c1–c5) — consolidated

All numbers seeded (3×6), paired Δ where available.

| Probe | Config | Params | sat R² | paired Δ vs plain | Verdict |
|---|---|---:|---:|---:|---|
| baseline | plain TCN | 4,130 | 0.163–0.168 ±0.03 | — | anchor |
| **c1** width | hidden 64 / 128 | 7.3k / 13.6k | 0.152 / 0.137 | −0.016 / −0.031 | capacity **NOT** binding (overfits) |
| **c2** depth | d3_res [1,2,4]+residual, RF 57 | 6,050 | 0.153 | −0.016 | temporal RF **NOT** binding |
| **c3** FiLM | FiLM-from-CORAL (γ,β/session) | 16,642 | 0.133 | −0.035 | session-level: **redundant w/ CORAL** |
| **c4/c5** SE | **SE shared, r=8** | **6,542** | **0.196** | **+0.033 (std 0.007, 4.7σ, 3/3)** | **BEST — window-level gain** |
| c5 | grouped SE (per-output x/y gates) | 7,757 | 0.179 | +0.016 (std 0.019) | per-output routing **unnecessary** |
| c4/c5 | GLU (gate at 32-d hidden) | 7,234 | 0.136 | −0.027 | gate too **downstream** |
| c5 | chnosqz (channel gate, no squeeze) | 6,542 | 0.165 | +0.002 (std 0.033) | isolates: **squeeze is the load** |

Key paired contrasts: SE→grouped −0.017 (std 0.014); SE→chnosqz −0.030. SE's
+0.033 is **stable across all budgets** (200/800/∞) and positive in **5/6
sessions** (hardest 11-24-Run2 goes 0.001→0.054); it is *not* exploiting more
calib data. c2 caveat: naive depth-3 depthwise+ReLU **collapses** (train R²
0.16 vs 0.52) from cascading dead units — needs per-layer residual just to match
d1, and even then RF 9→57 buys nothing.

### The reframe c5 forced — this is the load-bearing insight

Every gate answers the same question ("which neurons matter, how much"); what
differs is the **timescale of context** driving it. The completed probes map that
axis cleanly:

| gate timescale | probe | paired Δ | verdict |
|---|---|---:|---|
| per-frame | chnosqz | +0.002 | too fast / noisy — NULL |
| **per-window** | **SE squeeze** | **+0.033** | **the sweet spot — WIN** |
| per-session | FiLM | −0.035 | too slow / CORAL-redundant — FAIL |

So the missing capability is **window-level adaptive gain (≈ normalization)** —
*not* per-frame dynamic routing (chnosqz null), *not* per-output routing (grouped
worse), *not* per-session conditioning (FiLM redundant). c1/c2/c3 all failed
because none supplies it; SE succeeds because the squeeze does. (Guide:
`outputs/figures/network_diff_guide.pdf`.)

### Files (c1–c5)

| Probe | Script | CSV |
|---|---|---|
| c1 width | `scripts/c1_capacity_probe.py` | `outputs/results/c1_capacity_probe.csv` |
| c2 depth | `scripts/c2_dilated_probe.py` | `…/c2_dilated_probe.csv` |
| c3 FiLM | `scripts/c3_film_probe.py` | `…/c3_film_probe.csv` |
| c4 SE/GLU | `scripts/c4_se_glu_probe.py` | `…/c4_se_glu_probe.csv` |
| c5 grouped+deconf | `scripts/c5_grouped_se_deconfound.py` | `…/c5_grouped_se_deconfound.csv` |
| models | `models/tcn_student.py` — `DepthwiseTemporalConvStudent`, `FilmTCNStudent`, `SETCNStudent`, `GLUTCNStudent`, `GroupedSETCNStudent`, `ChannelGateNoSqueezeStudent` | |

---

## 10. Next round — probe the "window-level gain" reframe (live plan)

c5 relocated the mechanism from "per-frame routing" to **window-level adaptive
gain**. The next round tests that directly and pushes the two knobs it exposes:
the **content** of the squeeze (which statistic) and its **timescale**. All configs
stay inside the §1–§6 hardware guardrail; all judged by **paired Δ vs plain** on
the same 3×6 protocol. Current operating point to beat: **SETCN 6,542 / 0.196.**

Run in order; each is gated.

| # | Change | ~Params | Hypothesis → gate |
|---|---|---:|---|
| **n1** | **Parameter-free window norm** (replace SE gate with `h ← h/(mean_T\|h\|+ε)` or per-window z-score) | 0–96 | If the active ingredient is literally normalization, a *non-learned* norm recovers most of +0.033. **Gate:** recovers ≥~0.02 → capability = normalization, near-free in silicon (the SE MLP is optional icing) → new headline result. Recovers little → the learnable/nonlinear gate matters, keep SE. |
| **n2** | **Second moment in squeeze** (mean → mean+std into excitation MLP) | +~1k | Makes it literal divisive-norm / learned-LayerNorm-lite. Prefer **std over max** (max = outlier detection, wrong prior). Predict helps most in Run2 sessions where single-frame readout is unreliable. |
| **n3** | **Learnable-timescale EMA squeeze** (per-channel `s_t=(1−α)s_{t−1}+α·h_t`, α learnable) replaces fixed W-mean | +~96 | Lets the model pick its own point on the frame↔session axis. Predict α lands near window scale (away from chnosqz-null and FiLM-redundant). Bonus: one accumulator/channel → *cheaper* at deploy than window-mean, fully streaming. |
| **n4** | **SE + dilated-residual RF** (SE on c2 d3_res, RF 9→57) | ~8k | **Reverse-prediction check:** longer RF lengthens the squeeze's averaging window → drifts toward session scale → predict **plateau or reversal, not additive**. Run to *falsify* "more RF unlocks under SE." **Gate:** if ≤ SE alone, close the RF axis permanently. |
| **n5** | **SE r-sweep** r∈{4,8,16} | ±few 100 | Cheap capacity check on the gate; r=8 may be too tight. Low priority — only if n1–n3 leave headroom. |

**Decision rule.** n1 first — highest info-per-cost. If parameter-free norm
captures the gain, the project's answer is *"the missing capability is
window-level normalization, ~free on-chip"* (the Pareto-winning result); n2/n3
then only refine it. Otherwise n2/n3 probe the gate's content and timescale.
n4/n5 are gated ablations. **Stop** when marginal paired Δ per ~2k params < ~0.005.

### Diagnostics / hygiene (not student experiments, but do before trusting the gap)

- **d1 — Seed-average the teacher.** 0.26 is a single run; the whole "gap =
  teacher − student" numerator rests on it. Re-run SPINT on the same 3 seeds so
  the comparison is apples-to-apples. Cheap; removes a silent bias.
- **d2 — Attention-variance probe on SPINT.** Dump `attn_score` (C×N) on val;
  measure Var across windows and across the two covariates. Bounds how much any
  dynamic / per-output routing could *ever* buy — sanity-checks whether the
  residual gap is the "window-gain" story or something structural (the 512-D
  read-in, the ID path) that no on-chip gate reaches.

### Honest expected outcome

n1 likely captures a large fraction of SE's +0.033 (the deconfound says the gain
is the squeeze, and a fixed norm *is* a squeeze) → best case: a ~free on-chip
primitive at ≈0.19. n2/n3 add a sliver. The last stretch to 0.26 (= oracle) is
probably the teacher's 512-D read-in + ID encoder and **not** worth the silicon —
decide on the Pareto plot, not on chasing the teacher's number.
