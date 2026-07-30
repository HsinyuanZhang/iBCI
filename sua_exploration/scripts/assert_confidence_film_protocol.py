"""Print and validate the no-test confidence-FiLM sweep protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.confidence_film_protocol import make_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t4-budget", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(make_protocol(args.t4_budget).__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
