# H1-M3RC Source-Only Five-Fold Result V1

Date: 2026-09-03  
Status: `PASS_H1_M3RC_FOR_ALL_SOURCE_PACKAGE`

## Outcome

Exactly-M3 closed-form readout calibration passed every preregistered gate.
It uses no fourth calibration trial, no target gradient, and no target model
update.  Each of the official three calibration trials contributes its
eval-valid 20 ms prediction/target rows to a 7-output affine ridge fit.

| Outer date | Frozen C1 R2 | H1-M3RC R2 | Gain | TPL-M3 R2 |
|---|---:|---:|---:|---:|
| 19250108 | 0.402106 | 0.525993 | +0.123887 | -0.002215 |
| 19250113 | 0.295903 | 0.510252 | +0.214349 | -0.002992 |
| 19250115 | 0.584963 | 0.641807 | +0.056844 | -0.003921 |
| 19250119 | 0.347527 | 0.446814 | +0.099286 | -0.002783 |
| 19250120 | 0.399425 | 0.542334 | +0.142910 | -0.002677 |

Aggregate evidence:

- equal-date mean gain over frozen C1: `+0.127455`;
- nonnegative outer dates: `5/5`;
- worst outer-date gain: `+0.056844`;
- mean H1-M3RC minus neural-free `TPL-M3`: `+0.536358`;
- H1-M3RC beats `TPL-M3`: `5/5` dates.

The selected family was `MAT7` in every outer fold.  Source-only ridge
selection varied among `0`, `1e-6`, and `1`, which is why the all-source
package performs one final source-only selection before opening any held-out
calibration file.

## Interpretation

The result is not template recall: the M3 target-mean baseline is near zero
on every outer date, while calibrated neural predictions remain strongly
positive.  It is also not a late-pooling result.  The useful mechanism is
dense-label output-coordinate correction on top of the existing C1/M3
activity-plus-carrier decoder.

The official label budget remains exactly three trials.  The calibration map
has 56 fitted values for `MAT7` (49 slopes and 7 intercepts), estimated from
thousands of bin-level rows contained in those three trials.

## Immutable authority

- Attempt SHA-256: `4be50b70f1966f0f02ce58b049a6d030ae2f5e611c3168d79a11759aa6670ade`
- Score SHA-256: `7ac177c1f372d4f783fb841dfa320be9e6a67df429a0f42e70fc5693b0dd138f`
- Terminal SHA-256: `efde746176de8b6885d2eeb3962be4093d4aa1e69061ce208a98e1cfb34d6e27`

No formal held-out evaluation stream or EvalAI score was opened by this
screen.
