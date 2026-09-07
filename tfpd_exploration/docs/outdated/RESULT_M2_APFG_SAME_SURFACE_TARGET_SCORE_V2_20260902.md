# Result — M2 APFG Same-Surface Target Score V2

Date: 2026-09-02  
Status: **completed immutable CPU score; positive point estimate, promotion gate not passed**  
Method: APFG (Anchored Post-Fusion Gate), with the selected-T4 native POOLED decoder as the exact `alpha=+0.0` anchor

## 1. Executive result

The same-surface V2 successor completed all 65 pre-registered rows and resolved
the V1 control failure. In one CPU process, native POOLED and APFG-ZERO were
exactly equal for every session in prediction bytes, R2, targets, governed
window starts, and window count. The V1 failure was therefore a cross-device
bitwise-control defect, not an APFG implementation defect.

The source-trained scalar gate transferred in the positive direction on the
six external sessions:

```text
APFG-LEARNED|UNCAPPED - NATIVE-POOLED|FIXED30
= +0.004612 R2, 4/6 positive sessions
95% session-bootstrap CI [-0.003402, +0.013194]
```

This raises the equal-session external mean from `0.299057` to `0.303670`, but
does **not** pass the frozen promotion gate of at least `+0.010` mean and at
least `4/6` positive sessions. The correct classification is a bounded
positive residual signal, not a promoted method or a second headline
innovation.

On the seven within sessions, the total deployed contrast was stronger:

```text
APFG-LEARNED|UNCAPPED - NATIVE-POOLED|FIXED30
= +0.014002 R2, 7/7 positive sessions
95% session-bootstrap CI [+0.010383, +0.017782]
```

However, essentially all of that within gain came from uncapped causal
activity memory (`+0.013933`, 7/7). The learned Post-Fusion gate added only
`+0.000069` on top of uncapped memory. The within result must therefore be
credited to continual activity memory, not to learned Post-Fusion gating.

## 2. Immutable authority

Result root:

```text
tfpd_exploration/results/m2_anchored_postfusion_gate_v2_same_surface_control
```

It contains exactly six immutable JSON body/sidecar pairs. Every leaf is a
regular file with mode `0444` and `nlink=1`; `failure.json` is absent.

| Body | SHA-256 |
|---|---|
| `attempt.json` | `febb976a8103322350efe712cbfb0d2648774f7c561f4c8210afd34db682a285` |
| `launch.json` | `31966d942fd592b9683f46fe89aab127660153ae25e6947848f510a5c0e09e47` |
| `predecessor_authority.json` | `6053d2b4c020f4a8f48cbb8fd24c1d2238b1fb02f5709332aa3843663f840fad` |
| `input_authority.json` | `9356827ea1a329e0e2a2ec491d5221da7a6fd7d5c7ce0deb116f08edbf453a21` |
| `score.json` | `5dd7c3e911ca00709e028590af93c065a38f5ea1bd1369dc27fabda87b595d6e` |
| `terminal.json` | `9ea88d3a4e9048c484a4e67575a3b1c9a1323fbf0558d739acb065217f2a7b63` |

Current explicit, no-glob, 96-leaf implementation closure:

```text
65d904509663d396e26f221471558f10e4b8042ae347696c3cfa70eee3a1394d
```

The terminal binds the exact V1 six-body failed predecessor, current closure,
all intermediate receipt digests, 65 rows, zero parameter updates, zero target
updates, `cuda_initialized=false`, and the frozen learned scalar:

```text
alpha = -0.20759029686450958
```

The attempt-to-terminal wall time was approximately 20 minutes 4 seconds. The
recorded first-session scoring smoke was 9.79 seconds and projected 127.21
seconds for the scoring portion; most of the total wall time was the single
PIT preparation, checkpoint reconstruction, target materialization, and native
same-process comparator construction.

## 3. Absolute scores

Equal-session mean R2:

| Surface | System | Memory law | Mean R2 | Median R2 |
|---|---|---|---:|---:|
| External | NATIVE-POOLED | FIXED30 | 0.299057 | 0.326079 |
| External | APFG-ZERO | FIXED30 | 0.299057 | 0.326079 |
| External | APFG-ZERO | UNCAPPED | 0.299057 | 0.326079 |
| External | APFG-LEARNED | FIXED30 | **0.303670** | **0.329803** |
| External | APFG-LEARNED | UNCAPPED | **0.303670** | **0.329803** |
| Within | NATIVE-POOLED | FIXED30 | 0.654086 | 0.648376 |
| Within | APFG-ZERO | FIXED30 | 0.654086 | 0.648376 |
| Within | APFG-ZERO | UNCAPPED | 0.668019 | 0.661026 |
| Within | APFG-LEARNED | FIXED30 | 0.654319 | 0.648509 |
| Within | APFG-LEARNED | UNCAPPED | **0.668088** | **0.663702** |

The external FIXED30 and UNCAPPED rows are identical because these short
external streams do not cross the memory-capacity boundary under this replay.
Consequently their memory-only contrast is exactly zero; the external total
contrast is entirely the learned gate effect.

## 4. Pre-registered contrasts

### 4.1 External, six sessions

| Contrast | Mean delta | Positive | 95% session-bootstrap CI |
|---|---:|---:|---:|
| ZERO UNCAPPED - ZERO FIXED30 (memory only) | +0.000000 | 0/6 | [0.000000, 0.000000] |
| LEARNED FIXED30 - ZERO FIXED30 (bounded gate) | **+0.004612** | **4/6** | [-0.003402, +0.013194] |
| LEARNED UNCAPPED - ZERO UNCAPPED (uncapped gate) | **+0.004612** | **4/6** | [-0.003402, +0.013194] |
| LEARNED UNCAPPED - NATIVE FIXED30 (total) | **+0.004612** | **4/6** | [-0.003402, +0.013194] |

External total deltas by session:

| Session | Delta R2 |
|---|---:|
| `ses-2020-10-30-Run1` | +0.005257 |
| `ses-2020-10-30-Run2` | -0.008970 |
| `ses-2020-11-18-Run1` | +0.022034 |
| `ses-2020-11-19-Run1` | -0.007033 |
| `ses-2020-11-24-Run1` | +0.010048 |
| `ses-2020-11-24-Run2` | +0.006337 |

The breadth condition is met exactly, but the mean-effect condition is not.
The confidence interval crosses zero and the response is heterogeneous. It is
not defensible to describe `+0.0046` as a robust external improvement.

### 4.2 Within, seven sessions

| Contrast | Mean delta | Positive | 95% session-bootstrap CI |
|---|---:|---:|---:|
| ZERO UNCAPPED - ZERO FIXED30 (memory only) | **+0.013933** | **7/7** | **[+0.011398, +0.016761]** |
| LEARNED FIXED30 - ZERO FIXED30 (bounded gate) | +0.000233 | 5/7 | [-0.002816, +0.002511] |
| LEARNED UNCAPPED - ZERO UNCAPPED (uncapped gate) | +0.000069 | 5/7 | [-0.002618, +0.002140] |
| LEARNED UNCAPPED - NATIVE FIXED30 (total) | **+0.014002** | **7/7** | **[+0.010383, +0.017782]** |

This factorization is important. The total looks publishably clean, but the
learned gate is effectively null on the within surface. The result supports
continual activity memory and says little in favor of learned Post-Fusion
gating there.

## 5. What the experiment resolves

### 5.1 The V1 target failure was not a method failure

Every same-process native-versus-zero control passed exactly. The CPU first
external row reproduced the already observed values:

```text
prediction_sha256 = 5e8378d58cac8f98b5ec5265323a9438d11fe4172d303b77297c27bd8e5cc886
R2 = 0.34367723372874626
```

Thus the V1 failure came from comparing CPU bytes with historical CUDA bytes.
It did not reveal a checkpoint, T4-unit, input-authority, wrapper, or scoring
bug.

### 5.2 Anchoring fixes the catastrophic Post-Fusion loss

The earlier corrected, jointly trained PF-R1 checkpoint had external mean R2
`0.2384`, approximately `-0.0607` below POOLED. APFG instead starts at the
exact POOLED solution and learns only a residual scalar. It obtains external
mean `0.3037`, eliminating the catastrophic loss and slightly exceeding the
strong comparator in point estimate.

This is a useful design lesson: if complementary Post-Fusion information is
used at all, it should be introduced as a zero-initialized, identity-preserving
residual around the native POOLED operator. Replacing or jointly retraining the
native pooling backbone is not supported by these experiments.

### 5.3 The scalar gate is not a second main contribution

The source split selected the negative scalar consistently, and the frozen
scalar transferred with a small positive external mean. Therefore the
Post-Fusion residual direction is not pure noise. But the magnitude is less
than half the pre-registered minimum effect, the interval crosses zero, and
two external sessions are harmed.

The defensible conclusion is:

- useful residual information exists;
- exact native anchoring is necessary;
- one global source-learned scalar does not extract enough transferable value
  to support a headline method claim.

## 6. Decision and research boundary

The frozen promotion gate did not pass. Under the pre-registered decision
rule, scalar APFG is **not promoted** and the current Post-Fusion architecture
axis is closed.

It would be methodologically weak to inspect these six external labels and
then tune a vector, per-layer, or session-dependent gate on the same surface.
Any future higher-capacity anchored gate must be proposed and selected on
source-grouped data alone and tested on a genuinely untouched dataset or fold.
It must retain exact `alpha=0` nesting and include the same native/zero
same-process sentinel. Such a future route is a new study, not a continuation
that can convert this result into a positive claim.

For the current paper, the stronger and cleaner result is continual causal
activity memory: within `+0.01393`, 7/7, with a confidence interval entirely
above zero. APFG may appear as a mechanistic ablation showing that anchoring
recovers the lost POOLED baseline and exposes only a small residual transfer
signal.

## 7. Paper-safe statement

> Anchoring a source-learned Post-Fusion residual at the exact native pooled
> decoder eliminated the large degradation of jointly trained Post-Fusion
> models and yielded a small positive external point estimate (+0.0046 R2,
> 4/6 sessions), but did not pass the pre-registered +0.010 promotion gate and
> had a bootstrap interval crossing zero. In contrast, uncapped causal
> activity memory improved the within-session surface by +0.0139 R2 in all
> seven sessions. We therefore treat Post-Fusion gating as a bounded
> mechanistic signal rather than a promoted contribution.

Do not report the within `+0.0140` total as a learned-gate effect, do not claim
external significance, and do not describe V1's cross-device SHA failure as a
negative target result.

## 8. EvalAI deployment compatibility audit

The current FALCON M2 EvalAI interface does **not** expose the completed-trial
event required by the local causal `UNCAPPED` memory law. The validated runtime
contract is:

- `reset(dataset_tags)` once per session;
- `predict(neural_observations)` once per neural bin;
- no trial metadata at prediction time;
- `on_done` is not called by the official continual-task evaluator.

This is established by the repository's official-loop replay in
`tfpd_exploration/submissions/evalai_m2_act30_full_v1/validate_local.py`: the
replay feeds every bin through `predict` (lines 205--214), records
`trial_metadata_available_at_predict_time=false` (lines 347--357), and observes
zero `on_done` calls from `FalconEvaluator.evaluate` (lines 389--427). The
existing EvalAI decoders consequently serve a session-specific identity cached
offline and implement `on_done` as a no-op; for example see
`sua_exploration/evalai_t4_m2/t4_spint_decoder.py` lines 73--96 and 109--145.
The registered metadata also declares `IsTestTimeAdaptive=false`.

Therefore the two APFG components have different submission status:

| Component | Retraining needed? | Direct EvalAI status |
|---|---:|---|
| Frozen APFG scalar residual, `alpha=-0.20759029686450958`, with a static cached identity | No | Packageable as a static candidate, subject to exact local container validation |
| `UNCAPPED` decode-before-commit activity memory | No model retraining, but requires online trial completion | **Not representable by the current interface**; must not be silently included |

The locally strongest within result (`+0.014002`) is therefore **not** an
EvalAI-ready configuration: `+0.013933` of it comes from the unavailable
trial-completion memory law. A static APFG EvalAI package can test only the
small learned residual seen on the external surface (approximately `+0.004612`
locally), not the continual-memory contribution.

If official continual-memory validation is still desired, it requires a
separately named trial-free successor, such as a fixed-bin chunk state machine.
That successor needs a causal, one-sided local non-inferiority test against the
true-trial implementation and must not inherit the `APFG-U` name or its local
number until equivalence is demonstrated. This is an implementation/evaluation
study, not a reason to retrain the current APFG checkpoint.
