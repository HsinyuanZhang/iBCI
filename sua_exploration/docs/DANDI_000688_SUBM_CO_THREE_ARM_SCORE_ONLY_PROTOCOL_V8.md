# External sub-M three-arm score-only V8 quarantine protocol

## Disposition

**`BLOCKED_MISSING_ZERO4_TERMINALS`**.  V8 is not a V7 approval, not a formal
test opening, and not a score result.  The V7 independent audit found a P0:
its formal transition accepted a caller-supplied Python `TrustedRoots` object.
Even though V7 checked nominal private fields, an in-process caller could use
the private mint symbol plus `object.__setattr__` to forge the stated origin.
That control plane must therefore not receive a PASS.

V8 is append-only and leaves the V7 receipt intact.  It establishes a
quarantine boundary while the three shared-zero4 terminal checkpoints remain
missing.  It opens no external sub-M NWB, no real checkpoint, no policy, no
grant, and no score artifact; it imports no Torch for production execution,
does no forward pass/R² computation, and does not use a GPU.

The intended later endpoint remains `15 sessions × 2 views × 3 seeds × 3
arms = 270` cells for shared-T4, shared-zero4, and shared-TS4.  Nothing in V8
changes its scientific gates, data scope, or BP-free calibration hypothesis.

## What changed at the security boundary

There are now two deliberately separate layers.

1. The importable Python module
   [`subm_co_three_arm_score_only_v8.py`](../mc_maze/subm_co_three_arm_score_only_v8.py)
   has only zero-argument quarantine stubs.  It defines no `TrustedRoots`,
   `VerifiedPolicy`, `VerifiedGrant`, or `_ROOT_MINT` capability class.  No
   caller object, root path, public key, policy, contract, authorization,
   nonce, output path, or payload can be passed to `verify_complete_policy`,
   `construct_contract`, `verify_run_authorization`, or `open_score_ledger`.
   Every stub reads the captured blocked anchor and stops.

2. The only production-facing boundary is the separate process
   [`run_dandi688_subm_co_three_arm_score_only_v8_production.py`](../scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py),
   invoked only in isolated mode:

   ```bash
   /home/xinyuan/miniconda3/envs/spint/bin/python -I \
     /home/xinyuan/Work_host/SPINT/sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py \
     --mode status
   ```

   The CLI accepts only `--mode {status,formal}`.  It accepts no root object,
   path, key, payload, policy, contract, or authorization argument.  It does
   not import the V8 module or any project module before validating the fixed
   files.

The production CLI derives three canonical local paths in its own source:

```text
/home/xinyuan/Work_host/SPINT/
  sua_exploration/mc_maze/subm_co_three_arm_score_only_v8.py
  sua_exploration/configs/dandi_000688_subm_v8_pinned_production_anchor.json
  sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py
```

Before reporting even status, it requires isolated Python mode, the canonical
CLI path, regular files, owner UID `1002`, mode `0444`, exact byte counts,
SHA-256 values, stable `fstat` identity during read, no final-component
symlink, and exact canonical anchor JSON.  Core and anchor are opened via an
absolute dirfd/openat path chain with `O_NOFOLLOW`; the core is verified as
data, not imported as authority.

The pinned anchor's exact current status is
`BLOCKED_NO_ACTIVE_FORMAL_ROOT_CHAIN_V8`.  After all checks, both `status` and
`formal` therefore remain blocked.  There is intentionally no active-anchor
branch to exploit in V8.

## Threat model and non-claims

The meaningful boundary is a fresh `python -I` process executing a locally
owned, immutable reviewed CLI/source/anchor triplet.  A regular caller can
choose only the two modes; environment/module injection, caller-supplied
roots, paths, keys, payloads, copied objects, pickled objects, subclasses, and
proxies cannot carry authority into that process.

No pure Python module can defend against an adversary who already has arbitrary
code execution in the same interpreter and can replace the public function,
modify CPython internals, rewrite the CLI source, or change the ownership of
the trusted files.  Those capabilities are outside this library-level threat
model.  The CLI makes that assumption explicit through independent process
startup, isolation mode, fixed paths, owner/mode checks, and exact source and
anchor hashes; it does not claim that a mutable in-process object is a trust
root.

Consequently V8 does **not** claim an independent security PASS.  It records
the V7 P0 as unresolved for V7 and blocks all formal execution until a future,
separately reviewed active-root design replaces this quarantine anchor.

## Regression evidence

The V8 test suite includes exact attacks against the old failure mode:

- an attacker root-like object with `_ROOT_MINT`-style state and rewritten
  `_origin`/`_anchor_sha256` via `object.__setattr__`;
- copies, deep copies, pickle round trips, dict subclasses, and proxy objects;
- post-import monkeypatching of apparent anchor path/hash/status constants,
  a hypothetical loader, module `__file__`, and an injected `_ROOT_MINT`;
- direct construction / direct argument injection for a policy, contract,
  public key, root path, or payload;
- clean-process `python -I` CLI execution under environment/module injection
  attempts and unsupported CLI options;
- static proof that V8 defines no V7-style capability classes or Torch import;
- synthetic CPU-only parity with frozen `torchmetrics==1.5.1`
  `R2Score(multioutput="variance_weighted")`, including the `atol=1e-4`
  constant/near-constant branch.

The metric function is pure synthetic numerical code and is not attached to a
formal ledger or external output.  It does not give V8 a route to compute an
external score.

## Required future action, deliberately not performed here

Before any formal held-out endpoint can exist, a future reviewed successor
must do all of the following in a fresh design review:

1. Replace the blocked anchor with a source-pinned active root chain whose
   policy/checkpoint/run-auth public-key fingerprints are reviewed outside
   caller-controlled runtime inputs.
2. Re-establish an independently audited fresh-process verifier for the
   complete policy, all nine independent Ed25519 checkpoint closures, and live
   checkpoint path/SHA/bytes/mode checks **before** contract construction.
3. Bind short-lived run authorization and a canonical nonce claim without
   accepting output/external-data roots from the caller.
4. Re-audit safe NPZ/NPY parsing, frozen TorchMetrics R² semantics, and exact
   270-cell aggregate reconstruction in that activated design.
5. Obtain the three missing shared-zero4 epoch-011 terminals and refresh all
   nine V7-or-later independent closure receipts.

V8 performs none of these actions.  Its useful result is only an honest,
testable fail-closed quarantine while the MUA/SUA held-out scientific program
continues elsewhere.
