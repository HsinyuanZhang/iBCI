# H1 Sparse-Event Endpoint Carrier V2 — Source-Shrinkage Addendum

**Status:** frozen after the V1 CPU stop and before any H-SE5 GPU training  
**Candidate:** `H-SE5 = [w1,w2,w3,w4,b]`  
**Scope:** development evidence on the 13 public held-in-calibration recordings

## Why V2 is a separate candidate

V1 (`q=3`, normalized ridge `lambda=0.1`) recovered correct endpoint attachment relative to a
within-trial endpoint-label shuffle in all 13 sessions at both M=3 and M=4, but it overfit absolute
event-rate modulation: correct fitting was below an intercept-only predictor in all 13 sessions.
Three components also retained only about 58% of standardized endpoint-displacement variance,
below the V1 frozen 65% median gate. The V1 receipt remains immutable and its GPU stop is not
relabelled.

V2 makes two prospective implementation changes motivated by those diagnostics:

1. increase normalized ridge shrinkage from `0.1` to `3.0`;
2. retain four displacement components and the rate intercept, yielding five carrier values.

The extra input changes the compact consumer by only 32 weights at `h=32`. Matched V2 Zero5,
label-shuffle, and row-shuffle controls use the same five-dimensional interface, so carrier content
is not confounded with width.

Because the V1 development outcomes motivated V2, V2 is not a pristine pre-registered
confirmation. It is a new development candidate whose GPU result must be expanded to independent
dates/seeds before a broad claim.

## Frozen V2 estimator

- Native event/parser/endpoint rules: exactly the V1 protocol.
- Dense `OpenLoopKinematicsVelocity`: never opened.
- Source displacement basis: fit from every valid native movement event in non-outer-date source
  recordings. Using all source events is offline source training, not target calibration.
- Basis: source mean/std, SVD, first `q=4` components, deterministic sign, projected-score scaling.
- Target support: first M=3 trials (primary) and M=4 (companion).
- Per-channel target estimator:

  \[
  \hat B=(X^TX+3n\,\mathrm{diag}(0,1,1,1,1))^{-1}X^TY.
  \]

- Carrier order: `[w1,w2,w3,w4,b]`.
- Target-session optimizer steps and backward steps: zero.

## V2 CPU gate

M=3 is GPU-ready only when all conditions hold across the six date-LODO folds:

1. all 13 sessions have at least eight support events and a rank-five design;
2. every source basis retains at least 65% variance and the median retains at least 70%;
3. correct-minus-label-shuffle forward-transfer mean/median and leave-largest-out mean are
   positive, with at least 10/13 positive sessions;
4. correct-minus-intercept forward-transfer mean/median and leave-largest-out mean are positive,
   with at least 10/13 positive sessions.

M=4 is diagnostic unless M=3 fails. Receipt and verifier must bind the V1 parser SHA, this
addendum, the V2 implementation, all 13 input SHAs, label counts, and the no-dense-velocity scope.

## First GPU cell

The first fold is date `19250101`, seed 42, 50 fixed epochs, with the exact existing H1 compact
consumer training windows and query boundary. The initial matched arms are:

- `H-SE5`: correct sparse endpoint carrier;
- `H-SE5-Z5`: separately trained literal-zero five-dimensional control.

After the pair is complete, score the correct checkpoint under same-checkpoint Zero5, row-shuffle,
and endpoint-label-shuffle inputs. Separately trained LS/RS arms are launched only if the correct
arm beats the separately trained Zero5 and both same-checkpoint content controls.

The matched `H-S` SPINT and existing dense `H-C` numbers are references, not retrained by this
candidate. The initial cell is positive only if:

- `H-SE5 - H-SE5-Z5 > 0`;
- `H-SE5 - H-S > 0` is reported separately;
- correct carrier beats same-checkpoint Zero5, LS, and RS;
- both target recordings do not show opposite large effects without being reported.

No formal, minival, organizer-held, or EvalAI endpoint is opened by this program.
