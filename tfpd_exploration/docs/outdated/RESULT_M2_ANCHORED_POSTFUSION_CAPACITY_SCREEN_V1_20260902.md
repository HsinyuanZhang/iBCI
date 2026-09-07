# Result — M2 Anchored Post-Fusion Capacity Screen V1

Date: 2026-09-02  
Status: **completed source-only screen; both capacity arms failed**  
Family: APFC (Anchored Post-Fusion Capacity)

## 1. Executive result

The matched three-arm, 12-epoch GPU0 screen completed normally. Increasing the
gate capacity did not improve source-grouped validation over the one-scalar
APFG control:

| Arm | Parameters | Selected epoch | Equal-session validation R2 | Delta vs A-S1 | Positive sessions |
|---|---:|---:|---:|---:|---:|
| A-S1 | 1 | 11 | **0.6789508340** | — | — |
| A-DC2 | 2 | 11 | 0.6782980815 | **-0.0006527525** | **0/2** |
| A-TB4 | 4 | 12 | 0.6774578863 | **-0.0014929477** | **0/2** |

Neither new arm passes the frozen requirement of at least `+0.003` over A-S1
with 2/2 positive validation sessions. Both lose on both validation sessions.
The APFC capacity hypothesis is therefore rejected and neither arm is eligible
for all-seven refit or target evaluation.

The important control succeeded: A-S1 selected a negative coefficient at epoch
11 and reproduced the earlier scalar APFG validation result to approximately
`4.3e-9` absolute R2:

```text
APFC A-S1             = 0.6789508340150332
sealed scalar APFG V1 = 0.6789508383276186
absolute difference   = 0.0000000043125854
```

This makes a runner or input-law drift an implausible explanation for the two
capacity-arm losses.

## 2. Immutable authority

Result root:

```text
tfpd_exploration/results/m2_anchored_postfusion_capacity_screen_v1/screen
```

The root contains exactly five immutable JSON body/sidecar pairs and no
`failure.json`:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `3b076d78381eef5c2513481e98aed258dc660641c2dc99c6cac7fa2c0c1146e4` |
| `launch.json` | `b0e3c4270f9d3533c02a3d7ececfc46f0a2b484d15cae1254d2d33ac2ba3d379` |
| `source_authority.json` | `471166d635b1711ff06b2e187a55c4a33d3bf98236a71c4edacd34b1c7331625` |
| `screen.json` | `8cba5b6dc95222cc935242cb3be1fc6edad17d42cc6fcf5c28c726027032e84c` |
| `terminal.json` | `a63fe5755238aafdcf117f7b70c74858d7288ccaf875aec2cdad299f0474b16a` |

Every inspected body and sidecar is mode `0444`, has `nlink=1`, and the
sidecar digest exactly matches the body. The screen records:

```text
optimizer_steps = 29,832
screen runtime   = 349.45 seconds
target access    = false
physical GPU     = GPU0 only
```

The source authority binds the seven source sessions, one Selected-T4/PIT
materialization, the frozen checkpoint and student state, exact zero
initialization of all three gates, the DCT basis, and the source-only DC2
standardizer:

```text
DC2 mean = -0.4516508425
DC2 std  =  0.0858880373
```

## 3. Per-session selected results

| Arm | `ses-2020-10-27-Run2` | Delta vs A-S1 | `ses-2020-10-28-Run1` | Delta vs A-S1 |
|---|---:|---:|---:|---:|
| A-S1 | 0.6637216292 | — | 0.6941800388 | — |
| A-DC2 | 0.6627811455 | -0.0009404837 | 0.6938150175 | -0.0003650213 |
| A-TB4 | 0.6618202907 | -0.0019013385 | 0.6930954818 | -0.0010845570 |

A-DC2 learned parameters `[-0.23483284, +0.11458734]`. A-TB4 learned
parameters `[-0.45878166, -0.05683602, -0.33106175, -0.19819778]`. The
non-scalar arms do learn nonzero structure, but that structure does not improve
grouped validation.

## 4. Interpretation

The more expressive arms achieve slightly lower fit-session task loss than
A-S1 while obtaining worse validation R2. This is the pattern expected from
source-specific fitting, not from a useful transferable residual:

- a smooth time-coordinate gate does not rescue the Post-Fusion residual;
- conditioning a scalar on within-pool disagreement does not rescue it;
- the completed one-scalar result was not limited by these two low-dimensional
  forms of gate capacity.

The screen therefore closes gate-capacity expansion. It does **not** justify a
larger temporal basis, a per-unit gate, a combined TB4+DC2 arm, more epochs, or
a seed sweep.

## 5. Receipt limitation and why it does not change the kill decision

The V1 runner omitted two planned audit computations: its capacity entries set
`delta_vs_zero=null`, and its scalar entry sets
`validation_gain_vs_v1_target=null`. It also did not serialize a selected gate
checkpoint or perform the conditional all-seven refit. These omissions mean
the V1 terminal is not, by itself, a deployable winner artifact.

They do not make either new arm eligible. Both arms already fail the primary
paired comparison against A-S1, with 0/2 positive sessions. The sealed scalar
APFG V1 zero reference was `0.6767482379`; using that value only as historical
context gives approximately `+0.001550` for A-DC2 and `+0.000710` for A-TB4,
also below the planned `+0.005` zero gate. No all-seven refit should be run for
an ineligible arm.

## 6. Decision and next PF question

**STOP APFC.** Do not score A-TB4 or A-DC2 on the already observed six-session
external surface and do not send either arm to EvalAI.

The next defensible PF diagnostic changes the *fusion location*, not the gate
capacity. A minimal candidate is zero-anchored output-space fusion:

```text
y = y_native + beta * (y_post - y_native)
```

with `beta=+0.0` exactly reproducing native POOLED, a single beta fitted on the
five source fit sessions, and the same two source validation sessions used only
once. This directly tests whether the nonlinear `fc_in`/decoder makes
identity-space interpolation the wrong operator. It should first be evaluated
as a frozen-branch, source-only closed-form screen; no new backbone training is
justified before that screen is positive.

