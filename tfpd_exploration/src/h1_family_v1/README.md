# H1 CRST-B4 common-frontend staging area

This directory is deliberately **not an implementation**.  It exists so the
H1 common-frontend hypothesis and its prospective source-only gate have a
versioned home without modifying a legacy H1 decoder.

The only proposed change is recorded in the paired diagnosis artifact:
`activity_scale=1.0` at the raw neural input of the already frozen shared
8-slot frontend.  The proposed CRST-B4 implementation, if separately
authorized, must retain the exact frozen common operator:

```
raw x -> shared causal Conv(1,16,k=5) -> SiLU
      -> concat(local16, E0_700, T_4)
      -> Linear(720,256) -> GELU -> Linear(256,256) -> LayerNorm
      -> 8 learned slots / 8-head slot-to-unit attention
      -> residual FFN(256,1024,256) -> flatten/project(2048,256)
```

`CRST-B4-FLAT` has no calibration-logit bias.  `CRST-B4-ROUTE` may add only
the frozen-design additive calibration bias to those attention logits.  It
cannot change input scale, centering, local/token/slot normalization, PE,
temporal member, masks, initialization, or the target/output unit contract.

Nothing in this directory authorizes data access, model construction, fitting,
GPU use, a minival read, a formal train, checkpoint migration, or an export.
