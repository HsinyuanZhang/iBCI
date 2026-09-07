# Independent official-runnability review — M1 V3 `95c33c68`

Reviewer: GPT-5.6 Terra. No EvalAI contact. Not a score pick.

**Verdict: CONDITIONAL-GO**

The packed image is `spint-m1:v3-t-ema-e6-t2-5e44fc37`
(`sha256:95c33c681e077474a99f8e99a92814b7a30e2c68471fbbbe8d53a74dce13491a`).
`register` remains false.

Condition: official per-call wall must stay under about 19.4 ms (~2.33× the
container B4/t2 mean of 8.3 ms), and hidden padded wave length must stay near
the known 160–178k-bin public-session bound. Then sequential HO+HI plus the
evaluator 300 s sleep is ~54 min vs the ~7200 s cap.

Versus the deleted naive wrap: ~80 ms/call → ~8.3 ms/call; bounded decode
from multi-hour to ~49 min before sleep.
