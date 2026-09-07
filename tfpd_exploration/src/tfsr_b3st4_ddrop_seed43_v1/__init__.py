"""Seed-43 matched-replication production route for TFSR_B3ST4_DDROP.

This package is a sibling of the frozen seed-42 route
``src/tfsr_b3st4_ddrop_v1``; it never modifies it.  The package ``__init__``
deliberately stays torch-free so a static zero-argument plan never imports a
CUDA-capable module.  Torch becomes reachable only inside the authorized
backend of :mod:`.train_43`.
"""

__all__ = ["accelerated_forward", "contract_43", "train_43"]
