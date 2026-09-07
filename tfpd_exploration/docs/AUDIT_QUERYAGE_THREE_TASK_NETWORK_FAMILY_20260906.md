# Audit: QueryAge across the M1, M2, and H1 task families

## Scope and conclusion

This is a read-only source audit, not a proposed architecture or a trained-outcome claim. “M1” below means the actually trained P1 QueryAge16 pair built by `m1_optimized_v2/model.py` and trained by `m1_family_v1/family_train_v2.py`; “M2” is the completed new 24-epoch QueryAge-plus-prefix pair; “H1” is the current formal QueryAge-plus-prefix pair.

All three use the same named *temporal operator template*: `QueryTemporalStack(width=256, heads=8, layers=4, ffn=512, age_buckets=16)`. It reads fixed frontend-token memory while updating only the current query, and returns `[B,1,256]`; it is not ordinary causal Transformer KV caching. The implementation is in [core.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/core.py:20), [core.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/core.py:74), and [core.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/core.py:101). The constructors explicitly bind this template for M1 [model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m1_optimized_v2/model.py:8), M2 [model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_queryage_family_v1/model.py:23), and H1 [model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/model.py:24).

This supports a stronger architectural unification than temporal core alone: all three are width-256, eight-slot, local-k=5 set-fronted QueryAge decoders with a width-256 final norm/readout path and FLAT/ROUTE spatial arms. Different task I/O dimensions, independently learned weights, initialization, and training recipe are normal task instantiations of that design; they do not by themselves refute a common architecture template. The material remaining operator-preset exception is H1’s explicit unscaled-dot/local-balanced spatial-attention preset, described below.

## Common architecture template versus necessary task adapters

| Property | M1 trained P1 | M2 completed 24-epoch pair | H1 formal pair |
| --- | --- | --- | --- |
| QueryAge temporal core | width 256, h=8, L=4, FFN=512, 16 age buckets | same | same |
| Input window | W=100 | W=50 | W=700 |
| Actual neural input | `[B,100,64]` | `[B,50,96]` | `[B,700,176]` |
| Spatial frontend skeleton | 16-channel local causal conv k=5 → token MLP → 8×256 slot attention/slot FFN → 2048-to-256 pool | same 16-channel/local-k=5/8×256/2048-to-256 skeleton, with explicit mapped MHA implementation | same skeleton, with H1 unscaled-dot/local-balanced attention preset |
| `E0` / functional / mask | E0=100, H-C=4, 64 units | E0=50, T4=4, 96 units | C2 fused E0=700, T=4, 176 units |
| Output / scale | 16 native outputs; divisor 1 | 2 outputs; decoder-raw target=native ×5 | 7 outputs; decoder-raw target=native ×20, inference /20 |
| Unit dropout | shared p=.1 `[B,64]` | shared deterministic p=.1 `[B,96]` | stateless p=.1 `[B,176]`, after prefix |
| p=.5 cold prefix | absent | left-history zeroing before dropout | left-history zeroing before dropout and immutable bank mask |

The M1 shape configuration is in [m1_config.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/m1_config.py:15); M2’s is in [config.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_b_small_stability_v1/config.py:76); H1’s is in [h1_config.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_config.py:17). M2 validates `[B,50,96]` and two-dimensional targets in [train_pair.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_queryage_family_v1/train_pair.py:197). H1’s source window, micro/effective batch, and training recipe are declared in [formal_prefix_train.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/formal_prefix_train.py:18).

### Material H1 operator-preset discrepancy (not merely task shape)

H1’s 176 vs 64/96 units, E0 width 700 vs 100/50, 7 vs 16/2 outputs, and divisor 20 vs 1/5 are necessary task adapters; they do not defeat the common-network-template claim. H1 does, however, select `make_v2_unscaled_dot_localbalanced_pair` [h1 model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/model.py:30). Its parent explicitly multiplies the shared `(qk/sqrt(d)+route_bonus)` logits by `sqrt(d)` [h1_family_v1/model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_family_v1/model.py:37), and applies a local-FC1 balancing factor to the first 16 local columns [h1_family_v1/model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_family_v1/model.py:127). Those are actual spatial-operator/initialization-preset differences and must stay explicit. M1’s compatible full-window temporal initialization [m1 model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m1_optimized_v2/model.py:11) is recipe provenance, not a reason to deny the shared architecture.

## Exact shared spatial skeleton and ROUTE dimensions

The shared spatial template is verified rather than inferred from naming. M1's decoder classes directly subclass the H1 FLAT/ROUTE classes [m1_temporal.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/m1_temporal.py:24). H1's common frontend has a 16-channel k=5 local causal convolution, token MLP, 8 slots of width 256, slot attention/FFN, and `8*256 -> 256` slot pooling [h1_temporal.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_temporal.py:233). Its output path is LayerNorm(256), `256 -> 128 -> out_dim` [h1_temporal.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_temporal.py:405). M2 preserves that same skeleton by mapping the original local conv, token MLP, slots, norms, slot FFN, and slot projection into its explicit attention frontend [m2_family_v1/decoder.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_family_v1/decoder.py:83); its immutable configuration fixes 16 local channels, k=5, 8 slots, width 256, `slot_proj_in=2048`, and readout `(256,128,2)` [m2 config.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_b_small_stability_v1/config.py:76).

## ROUTE is a spatial logit bonus, not a temporal variant

QueryAge replaces the temporal member for both FLAT and ROUTE. ROUTE adds calibration-conditioned routing terms to slot-to-unit *spatial attention logits* before softmax. M1 and H1 use 8 heads × 8 slots × route-key width 32; their projections consume `E0+H-C`, hence 104 inputs in M1 and 704 in H1 [h1_temporal.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_temporal.py:173). M2 has the same 8 heads, 8 slots, and 32-dimensional route key, but its `E0+T4` carrier is 54-wide [routing.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_family_v1/routing.py:19). The M2/H1 route implementations layer-normalize that carrier, take the per-head calibration dot product, multiply by `tanh(g)`, and add the result to pre-softmax logits [routing.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_family_v1/routing.py:30), [h1_temporal.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_temporal.py:184). Paired constructors copy all non-routing state and initialize the route gate to zero in M2 [model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_queryage_family_v1/model.py:60) and H1 [model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/model.py:30).

## Recipe and trained-state comparison

| Item | M1 P1 trained pair | M2 completed 24-epoch pair | H1 current formal pair |
| --- | --- | --- | --- |
| State/initialization | M1 shell + QueryAge initialized from compatible full-window initializer; P1 fresh pair training | fresh M2 frontend/readout + fresh QueryAge; no causal or M1 state loaded | fresh H1 unscaled/localbalanced shell + fresh QueryAge; no capacity-state warm-start |
| Horizon | 24 epochs; 122,688 train rows and 31,252 source-minival rows | 24 × 3,165 updates/arm | 12 × 731 updates/arm; 23,212 source windows/epoch |
| Optimizer | AdamW, lr 1e-4, wd .01 except bias/norm, one-epoch linear warm-up, clip 1 | existing AdamW groups/S1-SMALL-COS/clip 1 | AdamW trusted groups; epoch-1 linear ramp to 1e-4 then hold; wd .01; clip 1 |
| EMA/selection | EMA .9995; source-minival equal-session maximum, earliest tie | EMA .9995; fixed source-minival equal-session maximum, earliest tie | EMA .9995; frozen 2,908 endpoint pooled native-float64 R2 maximum, earliest tie |

M1’s actual recipe/counts and paired training loop are in [family_train_v2.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m1_family_v1/family_train_v2.py:254). M2’s fresh-state rule and prefix/paired update order are in [model.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_queryage_family_v1/model.py:52) and [train_pair.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_queryage_family_v1/train_pair.py:197). H1’s formal recipe and prefix-before-dropout ordering are in [formal_prefix_train.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/formal_prefix_train.py:31) and [formal_prefix_train.py](/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/h1_queryage_family_v1/formal_prefix_train.py:100).

## Safe wording boundary

Use: **“M1, M2, and H1 instantiate one QueryAge16 network-design family: a local-k=5, eight-slot, width-256 spatial frontend feeding the same four-layer, eight-head, FFN-512, 16-age-bucket current-query core and a width-256 norm/readout path. Task I/O adapters and independently trained weights differ; H1 additionally uses the explicit unscaled-dot/local-balanced spatial preset.”**

Add whenever results are compared: **“This is a compound cross-task comparison: task shape adapters, optimizer/initialization recipe, sampler, and—on M2/H1—p=.5 prefix treatment are not held constant; H1 also has the explicit unscaled-dot/local-balanced spatial preset.”**

Do not say “same trained model,” “identical weights,” “ROUTE temporal variant,” or attribute a result difference to QueryAge alone. Do not erase H1's unscaled-dot/local-balanced preset; conversely, do not treat necessary input/output dimensions or independent training as evidence against the common architecture template.
