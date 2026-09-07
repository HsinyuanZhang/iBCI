#!/usr/bin/env python3
"""r10 entrypoint for the fail-closed Experiment A aggregator."""
import aggregate_t4_m30_experiment_a_r5 as implementation

from t4_m30_experiment_a_r10_authorization import RECEIPT, require_claim

implementation.RECEIPT = RECEIPT
implementation.require_claim = require_claim
main = implementation.main


if __name__ == "__main__":
    main()

