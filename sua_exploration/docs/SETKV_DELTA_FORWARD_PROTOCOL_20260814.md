# SetKV-delta forward-only protocol

**Status:** additive implementation gate; no score or GPU claim is created by this file.

## Question

Does carrier content become more useful when it is available as a separate set of attention K/V members, while
the sealed ordinary `fc_in(x+E)` route remains present?  This is not the failed decoupled-K/V intervention: no
teacher read-in is removed or replaced.

## Frozen token construction

For the existing B3S post-pool `psi`, define

```text
E_carrier(T) = psi([0,T]) - psi([0,0])
Z_carrier(T) = fc_in(E_carrier(T)) - fc_in(0)
KV = concat(fc_in(x + E(T)), Z_carrier(T))
```

Both subtractions are mandatory.  They remove carrier-independent B3S and `fc_in` biases, and make an exact Z4
carrier token bitwise zero.  No token-type embedding, gate, projection, temperature, or new parameter is allowed
in Stage 0.

## Forward-only arms

The frozen A2 epoch-5--12 checkpoint bundles and query/normalizer/session authorities are reused unchanged.

1. ordinary A2 output, reused from its immutable receipt;
2. aligned SetKV-delta/T4;
3. SetKV-delta/Z4;
4. SetKV-delta/RS4, with one fixed row permutation per session;
5. duplicate-activity-token, which doubles set size without adding carrier-only content.

Stage 0 is diagnostic.  A positive aligned content/attachment contrast can promote a separately contracted
source-training screen; a flat frozen-consumer result does not prove a jointly trained SetKV consumer impossible.
No target optimizer, backward pass, normalizer refit, checkpoint selection, or formal sub-C test access is
permitted.

## Interpretation and cost

Appending N tokens changes the attention softmax even with zero content.  Therefore the ordinary, Z4, duplicate,
and RS4 controls are mandatory.  `parameter_delta=0` must never be reported as zero deployment cost: the current
uncached MHA path adds N hidden tokens, their state, K/V projections, and query-token attention MACs.  Every
receipt reports parameter count, token state, analytic MAC increment, and measured latency.

## Route after Stage 0

Only a reviewed source-training contract may create SetKV training cells.  Its minimum logical matrix is
`{ordinary,SetKV-delta} x {T4,Z4}`.  External absolute T4 lift, within-sub-C non-inferiority, and the carrier
interaction are separate gates; a collapsing Z4 arm cannot rescue an unchanged T4 result.
