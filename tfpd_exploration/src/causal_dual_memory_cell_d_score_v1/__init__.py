"""Additive, fail-closed matched scorer for Causal Dual-Memory Cell D.

Importing this package is deliberately inert.  In particular it does not
import Torch, resolve an evaluation asset, inspect a result root, or create an
artifact directory.  The execution-only implementation is isolated in
``physical`` and is reachable only through an opaque in-process capability.
"""

