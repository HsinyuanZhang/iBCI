# V4 proposal: carrier-conditioned signed activity mixing

## Status

**Design only.  No V4 weights, training run, selection, or formal evaluation
are authorized by this document.**  It follows the valid strict V3 diagnostic,
not the earlier incorrect V2-loader diagnostic.

## Motivation and bounded claim

On the fixed 208 source windows, V3 exactly reloads to its reported score but
changes little when neural histories are exchanged across examples (RMS native
velocity change: 0.000217 full, 0.000226 QueryTemporal; target standard
deviation: 0.005226).  A zero input does change the output substantially,
especially for QueryTemporal.  This is evidence that the current frontend has
poor example-specific signal transfer after limited source fitting; it does
not identify a unique mechanism.  V4 therefore tests one constrained
alternative to the nonlinear set-attention pooling path.  It makes no claim
that C2 can be represented, transferred, or used as a V4 temporal result.

## Proposed frontend

For every causal bin `t` and valid unit `i`, construct a learned carrier
feature from that session's immutable identity and carrier tables:

```
q_i = MLP_phi(concat(E0_i, T_i))              # 256 signed channels
a'_t,i = causal_depthwise_conv5(x_t,i)         # one shared k=5 filter/unit
z_t = LayerNorm((1 / sqrt(n_valid)) * sum_i(mask_i * a'_t,i * q_i))
```

`MLP_phi` is shared across units and has no unit index embedding.  The sum is
over the provided immutable unit mask.  The multiplication is deliberately
signed and happens before the permutation-invariant sum: neural activity can
therefore modulate carrier/identity-specific directions rather than first
being reduced by near-uniform nonlinear pooling.  The k=5 convolution is
left-padded and causal.  Its output has width 256 and feeds the existing
four-layer matched temporal arms unchanged:

```
same V4 frontend + final norm + readout
             |                         |
  CausalTransformerStack          QueryTemporalStack
          FULL                         T
```

The arms must be constructed as a matched pair: copy the complete frontend,
normalization, and readout state exactly from FULL to T, then map the temporal
initialization exactly as in V2/V3.  No C2 weight, prediction, calibration,
held-out result, or C2 architecture is permitted in V4 initialization.

## Invariants that implementation must prove before any fitting

1. Unit permutation invariance: jointly permuting `x` channels, `E0`, `T`,
   and mask leaves every output unchanged within deterministic tolerance.
2. Causality: changing input after bin `t` cannot affect `z_t` or the final
   causal reader output for any endpoint at or before `t`.
3. Zero-input behavior: the frontend token is exactly zero before its
   explicitly documented normalization/bias behavior.  Any nonzero decoder
   output on a zero input must be attributable to a recorded downstream bias,
   not an unrecorded frontend offset.
4. FULL/T matching: frontend, final norm, and readout state dictionaries are
   byte-identical at construction; only their temporal operators differ.
5. Checkpoint contract: persist `frontend_contract_version=4`; every evaluator
   must select V4 explicitly and load all model keys with `strict=True`.

## Predeclared diagnostic sequence if training authority is later granted

1. Run an initialization-only invariant test and a tiny fixed-window
   learnability check on source train.  It is a wiring gate, not a model
   selection event.
2. If it passes, run one frozen 208-window source-only capacity probe using
   the same IDs, 260 updates, 600-second deadline, optimizer grouping, target
   scaling, and no dropout as V2/V3.  Do not alter the recipe after observing
   V4 output.
3. Strictly reload the resulting V4 checkpoints and require diagnostic
   baseline equality with the frozen probe report before interpreting any
   perturbation or gradient measurement.
4. Report the existing perturbation suite plus a signed-mixing ablation that
   replaces `q_i` by a per-session carrier mean.  The ablation is source-only
   and diagnostic; it is not a route to held-out selection.
5. Advance to paired source/minival training only after explicit authority and
   a predeclared comparison plan.  C2 remains a separate fixed reference, not
   a fallback or a claimed V4 temporal result.
