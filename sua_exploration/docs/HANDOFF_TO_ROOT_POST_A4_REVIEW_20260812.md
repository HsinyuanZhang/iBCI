# HANDOFF TO ROOT: post-A4 review findings

**Date:** 2026-08-12
**From:** independent review of the next-round program after the A4 three-seed result landed
**Scope:** five open items. Documentation and one correctness gate. No new experiment is proposed.
**Authorizes:** nothing. Items 1, 2 and 5 are doc/infrastructure work; item 3 is a gate that must close
before B9 produces any number; item 4 is a verification, not a claim.

---

## 0. Verified already done — do not redo

Section 1.3 of `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` has been correctly rewritten. It now states a
**phase-loss hypothesis** rather than a structural theorem, names the three reasons the theorem does not
follow (learned nonlinear per-trial projection before averaging, only approximately balanced direction set,
temporal structure can preserve phase-correlated statistics), labels its three motivating measurements as
cross-dataset motivation rather than one matched proof, and supersedes the H1 overlap residual from `0.862`
to `0.804819` with a named receipt. This is a better correction than the one I was going to request. The
status table row for A4 is also correctly scoped.

The A4 measurement itself is sound: three seeds, six-session LOSO, CPU forward-only, immutable receipts with
SHA-256 sidecars, `sealed_test_sessions_opened: false`, and a within-training-session pairing-permutation null
that correctly absorbs cross-unit phase clustering. The v1 defects it superseded (within-session unit folds,
M50 hard-code) were found and fixed rather than shipped.

---

## 1. Report the null-relative A4 advantages, not the raw cosines

**Finding.** The pairing-permutation null level is **`0.2236`**, not zero, because preferred directions cluster
across units. Section 1.3 and the status table currently quote only the raw pair `0.749669 / 0.319513`. A
reader who does not know the null level will read Z4's `0.319513` as "the activity path carries substantial
phase". Against its own matched null it carries much less:

| Arm | raw phase cosine | matched null | advantage over null |
|---|---:|---:|---:|
| AC4 | `0.740706` | `0.235635` | **`+0.505071`** |
| Z4 (activity-only) | `0.312414` | `0.223577` | **`+0.088837`** |

Ratio `5.68x`. Seed-42 aggregate SHA `063f98de27c9dff4d097f89d052942e04ab4e6fc92653c305047866b6395ba16`.

**Action.** Quote the null-relative advantages wherever the raw pair currently appears, or quote both with the
null level stated inline. The defensible sentence is that trial-averaged activity pooling is **phase-poor, not
phase-blind**: it clears its matched null by `+0.0888` while the carrier-trained encoder clears its own by
`+0.5051`.

**Why it matters beyond wording.** `z4_advantage_over_pairing_permutation = +0.088837` is positive. That is the
number which settles that the activity path is not phase-free, and it belongs in the record next to the claim
it constrains.

---

## 2. The A4 falsification rule cannot fire

**Finding.** `IDENTITY_TOKEN_CONTENT_PROBE_PROTOCOL_20260812.md` section 4.2 states that the phase-blindness
argument is falsified if Z4 has a positive advantage over its matched null **while** the mean `AC4 - Z4` delta
is `< 0.03`. Z4's advantage is positive (`+0.088837`), so the first conjunct holds; falsification then depends
entirely on the second. But if the carrier works at all, `AC4 - Z4` will exceed `0.03` — here it is `+0.430155`.
**The rule therefore cannot falsify the claim it names in any scenario where the carrier is useful.**

This is the same defect class as a gate that can only pass, and it should be treated the same way.

**Additional provenance point that a third-party reviewer will check.** The protocol file mtime is
`2026-08-12 22:10:26`; the seed-42 receipt is `22:01:43` and the multiseed summary is `22:07:11`. The file
postdates the numbers. The earlier drafted rule was an absolute one — activity-only phase cosine `<= 0.25`
counts as near chance, `> 0.25` falsifies — and at `0.312414` **that rule would have fired**. I cannot
determine from mtimes alone whether the rule was relaxed before or after the numbers were seen, and neither
can a reviewer.

**Action.** Replace 4.2 with a rule keyed on Z4 against its own null alone, since that is the quantity the
structural claim is about. Then record the rule-change history explicitly in the protocol: both versions, the
fact that the earlier version would have fired at `0.312414`, and why the conclusion currently in section 1.3
survives anyway. Stating this ourselves costs one paragraph; having it found costs the pre-registration
discipline's credibility across the whole program.

---

## 3. Correctness gate: `d_optimal_calibration_design.py` fits the carrier with a copied estimator

**Finding.** That module defines its own `CANONICAL_DIRECTIONS_RAD`, `_nearest_canonical_direction_index`, and
`_fit_cosine_tuning` rather than importing them from `mc_maze/unit_side_features.py`, and no test asserts the
copies agree with the canonical implementations. The motive was legitimate — importing `unit_side_features`
pulls in a Lightning dependency chain — but the consequence is that B9 would fit the carrier with a private
copy of the estimator, so its numbers are not guaranteed comparable to the mainline. This defect produces
plausible wrong numbers rather than an error, which is why it should close before B9 runs, not after.

`decoder_error_structure.py` carries the same duplication. It is lower priority because A13 has been dropped
on inferential-design grounds, but the copy should not be left where a future reader can reuse it.

**Action, either is sufficient.** Extract the direction constants and the cosine fit into a small
framework-free module that both sides import; or keep the copies and add a parameterized test asserting
bit-level agreement with `unit_side_features` on randomized inputs. Prefer the first.

---

## 4. Verify before it becomes a claim: B0 may be overtraining

**Finding.** The A11 provisional GPU parity receipt records B0 `epoch_000` at mean R-squared **`0.29514846`**
across the six validation sessions, with per-session values
`0.230127 / 0.210379 / 0.174095 / 0.353802 / 0.478143 / 0.324344`. The SUA lattice reports B0 at **`0.236417`**
under the epoch-5-to-12 averaging rule.

If those two scoring protocols are equivalent, **B0 gets worse with training** — it is overtrained, not
undertrained. That would mean a material part of `Z4 - B0 = +0.089591` is a training-schedule artifact rather
than the side-pathway effect the handoff currently attributes it to, and the derived framing
`T4 - Z4 = +0.248968` as "the actual carrier content" would need its baseline restated.

**This is a lead, not a result.** It is one seed, one epoch, and I have not verified that the A11 query
protocol, support budget, and averaging rule match the lattice's. The parity infrastructure itself is sound:
`cpu_mean_r2 = 0.29514842` versus `gpu_mean_r2 = 0.29514846`, max absolute difference `1.19e-07` against a
`1e-05` tolerance, identical query-window hashes, `torch_grad_enabled: false`, model-state hash unchanged
before and after, and no training or formal-test NWB access.

**Action.** Before the A11 epoch sweep is interpreted, confirm whether the A11 scoring protocol is identical
to the one that produced `0.236417`. If it is, the section 1.2 arithmetic needs restating and A11 rises in
priority, because it now bears on the headline rather than on baseline hygiene.

---

## 5. The test suite cannot be collected from the repository root

**Finding.** Running `pytest sua_exploration/tests/ streaming_calibration_exp/tests/...` from the repository
root produces **19 collection errors**. The mechanism is that `sua_exploration/src` and
`streaming_calibration_exp/src` are sibling packages with the same name, so whichever is imported first
shadows the other and `src.metrics` resolves against the wrong tree. This is a pre-existing repository hazard,
not something the new scaffolding introduced, but the new tests now sit on top of it, and pre-existing suites
such as `test_t4_paired_view_c1_encoder_int8.py` and `_qat.py` are among the failures.

Each subtree passes when invoked from its own directory. Verified independently: the seven new
`sua_exploration` suites give `47 passed, 1 skipped`; `test_training_method_scaffolding.py` gives `11 passed`
including the exact-null test; the D-optimal and pseudo-label suites give `32 passed, 2 skipped`.

**Action.** Either fix the collision with a rootdir or conftest arrangement, or document the exact per-tree
invocation that a reviewer must use and put it where a reviewer will look first. A third-party audit will very
likely begin with a root-level `pytest`, and it will fail.

---

## 6. Priority

| # | Item | Cost | Blocks |
|---|---|---|---|
| 3 | Estimator copy gate | small | any B9 number |
| 2 | Falsification rule and its change history | writing | credibility of every frozen gate in the program |
| 1 | Null-relative A4 reporting | writing | correct reading of the one result we have |
| 4 | B0 overtraining verification | small | section 1.2 arithmetic, A11 priority |
| 5 | Root-level test collection | small | third-party review workflow |

Items 1 and 2 are the same edit session. Item 3 should close before B9 is scheduled. Item 4 is a check, and it
may or may not turn into work.
