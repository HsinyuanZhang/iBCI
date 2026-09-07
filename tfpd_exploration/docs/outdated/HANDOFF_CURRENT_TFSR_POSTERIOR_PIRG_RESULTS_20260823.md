# Current Results: TF-SR, Posterior Carrier, and PIRG

Date: 2026-08-23 HKT

## 0. Executive conclusion

Three conclusions are already supported.

1. **Cell D remains the strongest established teacher-free system in the current matched M30 zero-shot setting.** Its governing seed-42 score is `0.569685` within and `0.417936` external.
2. **TF-SR seed 42 is a clear transfer failure, not a failure to consume T4.** It slightly improves within performance but loses `0.163763` external R2 to Cell D. Zero and wrong-pair controls show that TF-SR uses the aligned carrier strongly; the learned task-frame/stateful consumer simply does not transfer well.
3. **The evaluated Posterior Carrier bundle and PIRG are both stopped.** The full Posterior bundle collapses even at M30. PIRG cleanly isolates a static posterior-credibility gate, learns an amplitude near zero, and changes R2 only at approximately `1e-4` scale. Do not spend another round on gate, bias, temperature, or shrinkage variants around these exact systems.

TF-SR seed 43 is still training. It should be treated as a replication and variance estimate, not as an expected rescue of seed 42.

The only posterior-related successor still scientifically plausible is **posterior-marginalized source training**: use posterior samples as source-side carrier augmentation while keeping the successful Cell-D inference graph unchanged. It should be attempted only if a cheap CPU calibration audit first shows that posterior precision predicts held-out carrier error. If M30 external performance is the primary objective, a genuinely different decoder architecture has higher priority.

## 1. Evaluation conventions

Do not mix the following two score surfaces.

| Evidence set | Within | External | Purpose |
|---|---:|---:|---|
| Governing TF-SR matched score | 6 sessions | 15 sessions | Full current decision surface |
| Posterior/PIRG quick screen | 3 sessions | 3 sessions | Non-governing engineering screen |

All reported model comparisons use:

- causal calibration inputs only;
- no target-session optimizer steps, backward calls, or parameter updates;
- the last valid bin of a trailing 50-bin window;
- variance-weighted two-coordinate R2 within each session;
- equal weighting across sessions;
- paired per-session deltas where a comparator exists.

The absolute Cell-D scores differ between the full 6/15 evaluation and the fixed 3+3 quick screen because the session rosters differ. Only paired deltas within the same surface are directly comparable.

## 2. Reference systems

### 2.1 Sealed Cell D

Cell D is the current teacher-free reference. It retains the fused calibration identity path:

```text
identity_i = B3S(calibration activity_i, T4_i)
unit token_i = activity_i + identity_i
```

Its active training intervention is per-step `U(0,1)` whole-unit dropout over the complete fused unit token.

Full matched governing score, seed 42:

| Surface | Cell-D R2 |
|---|---:|
| Within, 6 sessions | 0.569685 |
| External, 15 sessions | 0.417936 |

The earlier AM/IM decomposition established that complete fused-token removal is load-bearing:

- activity-only masking external: `0.2996`, delta versus D `-0.1183`;
- identity-only masking external: `0.1146`, delta versus D `-0.3033`;
- neither activity-only nor identity-only masking reproduces Cell D.

Therefore Cell D is not explained by simple activity thinning or simple session-fingerprint suppression. Joint removal of the complete fused unit token is the supported mechanism.

### 2.2 A2 contextual reference

The current matched scorer records the three-seed pooled A2 context as:

| Surface | A2 pooled R2 |
|---|---:|
| Within | 0.577619 |
| External | 0.346088 |

This is contextual and non-gating. Cell D seed 42 is below A2 by about `0.0079` within but above the pooled A2 external mean by about `0.0718`.

## 3. TF-SR result

### 3.1 Intended intervention

TF-SR replaces Cell D's window-as-feature decoder with a task-frame/stateful read-in family while retaining:

- causal, gradient-free activity and T4 encoders;
- fused B3S plus T4 identity information;
- Cell-D-style dynamic whole-unit dropout;
- no learned unit/session table;
- zero target-session updates.

The implementation has approximately `1.65M` initialized trainable parameters, but about `1.71B` analytic MACs for the audited `B=1, N=128` forward. It is therefore smaller in parameter count than Cell D but substantially more sequential and expensive in runtime.

### 3.2 Seed-42 governing result

| Surface | Cell D | TF-SR aligned | TF-SR minus D | Positive sessions | Paired 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Within, 6 | 0.569685 | 0.591893 | +0.022208 | 4/6 | [-0.013771, +0.052474] |
| External, 15 | 0.417936 | 0.254173 | **-0.163763** | **1/15** | **[-0.259280, -0.087096]** |

Against the pooled three-seed A2 context:

| Surface | TF-SR minus A2 | Positive sessions |
|---|---:|---:|
| Within | +0.014274 | 3/6 |
| External | -0.091915 | 1/15 |

The scientific verdict is **STOP for TF-SR as a performance route**. The external loss is large, broad, and its paired interval excludes zero.

### 3.3 T4-control result

The failure is not because TF-SR ignores T4.

| Surface | Aligned | Zero T4 | Wrong-pair T4 | Aligned minus zero | Aligned minus wrong |
|---|---:|---:|---:|---:|---:|
| Within | 0.591893 | -0.083401 | -0.017720 | +0.675294 | +0.609613 |
| External | 0.254173 | -0.634890 | -0.060217 | +0.889063 | +0.314390 |

Interpretation:

- TF-SR strongly and causally consumes the aligned carrier.
- Wrong carriers are actively misleading rather than harmless noise.
- The carrier pathway and fusion are functioning.
- The failure is in the learned representation/read-in's cross-subject transfer, not in carrier admission.
- More carrier-use diagnostics, fusion-strength tuning, or small TF-SR ablations are unlikely to recover the `0.164` external gap.

### 3.4 Seed-42 provenance

- Training terminal SHA: `80bc864fc78f2302b218e6b20ed980e90528827b2b93e94c0f656f829bce99d6`
- SWA artifact SHA: `3aaae2dff9de959c645b04ab3e0893edb515cc861772fb00da4a4a5607c7c5fb`
- SWA state SHA: `7d9261857d1fa96996103ce6c7039cfbc98be536f04cea5c79ce60ca06443be0`
- Matched score-v2 body SHA: `8551ddf2fe437c9bd37e4767991bec5f8d9328778ad4c883afcbac3fd185a9dd`
- Matched score-v2 terminal SHA: `3007383ef0e4d2b8cc78aea262729fca876261a8bf25c300212e6cc573a2b579`
- Matched score path: `tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_matched_score_v2`

The V1 scoring attempt failed before opening an evaluation file because its root environment pointed at the parent data directory. That failed attempt was preserved immutably. V2 corrected only the operational root binding and produced the result above.

### 3.5 Seed-43 live status

Read-only snapshot at `2026-08-23 09:34 HKT`:

- process alive on RTX 3090 GPU0, PID `13331`;
- latest immutable receipt: epoch 40, meaning `41/48` epochs complete;
- cumulative steps: `1,390,925 / 1,628,400`;
- epoch-40 mean loss: `0.342971`;
- epoch-40 throughput: `9.0038` steps/s;
- model and optimizer finite;
- all 11 critical gradient groups present;
- source-only; target/formal unopened;
- no checkpoint, SWA, terminal, or failure receipt yet.

Seed 43 should finish because only seven epochs remain and it provides a replication. It should not be treated as a likely rescue: seed 42's external gap is too large and too consistent for ordinary seed variance to reverse confidently.

## 4. Full Posterior Carrier result

### 4.1 Intended idea

The Posterior Carrier route upgraded point-estimated analytic T4 to a distributional identity representation using closed-form posterior information. The broader motivation was reasonable: short calibration prefixes produce unit-specific uncertainty that OLS point carriers ignore.

The evaluated full bundle changed multiple coupled components, including posterior carrier estimation/normalization and its learned consumer. Its quick screen therefore tests the **complete system**, not a single posterior factor.

### 4.2 Fixed 3+3 quick-screen result

| Surface | Budget | Sealed Cell-D point carrier | Posterior full SWA | Posterior minus sealed | Positive sessions |
|---|---:|---:|---:|---:|---:|
| Within | M30 | 0.53498 | 0.17634 | **-0.35864** | 0/3 |
| Within | M4 | 0.30490 | 0.10783 | **-0.19707** | 0/3 |
| External | M30 | 0.53449 | -0.03183 | **-0.56632** | 0/3 |
| External | M4 | 0.21304 | 0.01235 | **-0.20070** | 1/3 |

This is a clear performance NO-GO for the full Posterior system.

The key observation is the M30 collapse. A superficially smaller M30-to-M4 decline cannot be called robustness because the M30 starting point has already been destroyed. The negative result is not merely an M4 information-shortage effect.

Remote quick-screen receipt chain, independently audited read-only:

- attempt: `fd85e83bbc9180c9d7ca27720de3063c76be20cf6889797570234cfd82f413d2`
- input authority: `bad14dadec2c4e525c4cffbaf689b75ce1cebfa9bccf27bc022256574176386e`
- score: `07bb30a018ad33a63d89ed3cd4fcc0293d7150b59262f0db84ae9ae5ad6183b5`
- terminal: `51103a5703ff6933362f11913afd379c3b580c59abac62630194bf2a44626949`

The screen was engineering/non-governing and used three sessions per surface. It did not open formal or H1 surfaces and made zero target updates.

## 5. PIRG result

### 5.1 Why PIRG was run

PIRG was a deliberately narrow, performance-oriented successor intended to avoid another large Posterior bundle. It held the successful Cell-D system and ordinary OLS point T4 fixed. The sole trainable intervention was one scalar `alpha` controlling a bounded multiplicative gate on the existing fused B3S identity residual, using only posterior directional credibility.

At `alpha=0`, PIRG is exactly Cell D. The experiment therefore asked whether graded posterior credibility is useful as a static inference-time regulator of identity strength.

### 5.2 Training result

- Source training and receipt integrity completed successfully on the remote 5070Ti route.
- Final scalar: `alpha = 0.009198200888931751`.
- This value places the gate extremely close to its exact Cell-D identity setting.

The optimizer effectively chose to turn the new mechanism off.

### 5.3 Matched 3+3 score

| Surface | Budget | PIRG minus sealed Cell D |
|---|---:|---:|
| Within | M30 | -0.0000582 |
| Within | M10 | -0.0001029 |
| Within | M4 | -0.0001237 |
| External | M30 | +0.0000304 |
| External | M10 | +0.0001460 |
| External | M4 | -0.0000637 |

Pooled M4 result across all six quick-screen sessions:

- mean delta: `-0.0000937`;
- positive sessions: `1/6`;
- predeclared promise gate: at least `+0.03` and at least `4/6` positive;
- verdict: clear failure.

This is not an underpowered weak positive. The effect is approximately four orders of magnitude smaller than the desired `+0.03` improvement and is centered essentially at zero.

### 5.4 What PIRG rules out

PIRG provides a cleaner mechanistic result than the full Posterior bundle:

- posterior credibility is not useful as this static multiplicative gate on the complete fused identity residual;
- longer alpha training is unlikely to help because the learned optimum already sits near the exact no-op point;
- replacing the scalar with a small vector, MLP, attention-logit bias, temperature, or per-head gate would remain in the same low-priority neighborhood;
- Cell D's existing whole-unit dropout probably already captures much of the useful unit-reliability invariance;
- posterior precision estimates carrier uncertainty, but carrier uncertainty is not necessarily the same as a unit's utility to the decoder.

Therefore no more PIRG-neighborhood ablations are recommended.

## 6. Cross-experiment interpretation

The current evidence separates three questions.

### 6.1 Is analytic T4 useful?

**Yes.** TF-SR aligned-versus-zero/wrong controls show large positive effects. Earlier T4-vs-Z4 and Cell-D results point in the same direction. The carrier itself is not disproven.

### 6.2 Does a more expressive causal/stateful consumer automatically improve transfer?

**No.** TF-SR gains slightly within but loses strongly across subjects. More temporal/stateful capacity can specialize to source identity and task statistics rather than produce invariance.

### 6.3 Does exposing posterior precision to the decoder solve carrier uncertainty?

**Not in the tested forms.** The full Posterior consumer collapses, and the isolated PIRG gate is a learned no-op.

These results do not prove that posterior uncertainty is mathematically meaningless. They show that directly consuming it as part of the inference representation has not paid.

## 7. Remaining posterior-related hypothesis

Only one materially different posterior use remains defensible:

### Posterior-marginalized source training

Keep the Cell-D inference graph unchanged. During source training only:

1. choose a calibration budget from M4, M10, and M30 by a preregistered schedule;
2. fit the closed-form carrier posterior once per session and logical epoch;
3. sample a carrier from that posterior as training augmentation;
4. retain Cell D's existing whole-unit dropout;
5. use a deterministic ordinary point carrier or posterior mean at deployment;
6. add no credibility token, gate, session table, target gradient, or decoder parameter.

This would combine:

- existence marginalization from whole-unit dropout; and
- parameter-uncertainty marginalization from posterior carrier sampling.

It is a training-distribution hypothesis, not another inference-gate variant.

### Required CPU kill test before GPU training

Before authorizing a full run, test whether posterior uncertainty is calibrated:

- does posterior precision rank held-out carrier estimation error?
- does the posterior mean improve held-out tuning prediction over OLS at M4/M10?
- are credible intervals empirically calibrated?
- is the relation positive across source sessions rather than driven by a few sessions?

If precision does not predict held-out error, terminate the posterior route entirely. If calibration is strong, authorize at most one full distributional-training cell. Do not bundle it with EB shrinkage, a new normalizer, an inference gate, or a new decoder.

Expected claim boundary:

- plausible value: graceful degradation at M4/M10;
- expected M30 gain: small or null;
- therefore lower priority if the primary paper target is M30 zero-shot external performance.

## 8. Recommended decisions for an external reviewer

1. Accept Cell D as the current teacher-free reference.
2. Let TF-SR seed 43 finish, but use it only to measure reproducibility. Do not start seed 44 unless seed 43 materially contradicts seed 42.
3. Stop TF-SR architecture micro-ablation after the seed-43 verdict.
4. Stop the full Posterior bundle.
5. Stop PIRG and all nearby static credibility-gating variants.
6. Run the CPU posterior-calibration audit if short-prefix robustness remains strategically important.
7. If that audit passes, consider exactly one posterior-marginalized source-training cell.
8. If M30 external performance is the primary objective, prioritize a genuinely new transfer-oriented decoder factorization instead of further posterior/PIRG work.

## 9. Questions for the external reviewer

Please focus review on the following rather than proposing a broad sweep:

1. Does the TF-SR result identify a transfer-specific architectural failure, or is any alternative explanation still compatible with `1/15` positive external sessions and strong aligned-versus-control effects?
2. Does PIRG's learned `alpha=0.0092` adequately close static credibility gating, or is there a genuinely different inference mechanism with a precise, falsifiable prediction?
3. Is posterior-marginalized source training sufficiently distinct from the failed full Posterior consumer and PIRG gate?
4. What CPU calibration evidence should be required before spending a full GPU run on it?
5. If the primary goal is M30 zero-shot external performance, what single larger architectural change has a better prior than posterior uncertainty modeling?

## 10. Current execution state

- PIRG remote source training: complete.
- PIRG remote matched quick score: complete; performance near zero; stopped.
- Posterior full-system remote quick score: complete; strong negative result; stopped.
- TF-SR seed 42 training and governing matched score: complete; external STOP.
- TF-SR seed 43 training: active at 41/48 completed epochs as of the snapshot above.
- No existing process was stopped to prepare this handoff.
- The overall project goal is not complete because the seed-43 terminal and its matched replication score are still pending.
