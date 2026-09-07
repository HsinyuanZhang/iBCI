#!/usr/bin/env python3
"""Claim or verify the signed, single-use r10 GPU authorization."""
import argparse

from t4_m30_experiment_a_r10_authorization import claim, require_claim


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--claim", action="store_true")
    modes.add_argument("--require-claim", action="store_true")
    args = parser.parse_args()
    if args.claim:
        claim()
    else:
        require_claim()


if __name__ == "__main__":
    main()

