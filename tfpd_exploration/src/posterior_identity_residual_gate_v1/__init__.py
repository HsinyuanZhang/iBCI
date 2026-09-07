"""Additive PIRG route package.

The package initializer intentionally imports no Torch.  The public dry CLI
imports :mod:`plan` only; physical model code is deferred to :mod:`core` and
the reviewed training backend.
"""
