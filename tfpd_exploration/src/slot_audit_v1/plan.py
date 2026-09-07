"""SLOT-AUDIT V1 plan — constants, sealed anchors, the pre-registered reading law.

Binding work order ``docs/WORKORDER_SLOT_AUDIT_V1_20260829.md``; purpose and
interpretation discipline from ``docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_
CONTINUITY_AUDIT_20260829.md`` sections 2.2 (continuity evidence) and 5
(output-slot audit, no training).

Nothing in this line trains, updates a checkpoint, or opens formal data:
the sealed Cell-D SWA is strict-loaded read-only on CPU, the decode is the
frozen static deployment recipe, and every audit statistic is numpy on a
sha256-bound cache.  The only ancestor numbers this package may cite are
LOADED from SHA-verified sealed receipts (never hardcoded).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # tfpd_exploration
REPO_ROOT = ROOT.parent                              # SPINT workspace root

SURFACES = ("external", "within")
BUDGETS = (4, 10, 30)
WINDOW_BINS = 50
GOVERNING_BIN = 49
SLOTS = tuple(range(WINDOW_BINS))
K_GRID = (2, 4, 8, 16)

# ---------------------------------------------------------------------------
# roots / anchors
# ---------------------------------------------------------------------------

CACHE_ROOT_RELATIVE = "cache/slot_audit_v1"
RESULTS_ROOT_RELATIVE = "results/slot_audit_v1"
ATTEMPT_NAME = "attempt.json"
MATERIALIZE_NAME = "materialize.json"
TERMINAL_NAME = "terminal.json"
FAILURE_NAME = "failure.json"

# The sealed continuity probe is THE anchor (work order section 2): baseline
# rows for per-(surface,budget,session) parity and trajalign arm rows for the
# re-measurement equality check.
SEALED_PROBE_REL = "results/continuity_probe_v1/continuity_probe_v1.json"
SEALED_PROBE_SHA256 = (
    "8afaa9109f2dbeb1fb68e1044c4e7ae275771113c4cb3daf5c5565ea77c588b1"
)

# CPU-to-CPU reproduction on bit-identical inputs: drift is a hard failure.
BASELINE_R2_TOLERANCE = 1.0e-12
TRAJALIGN_R2_TOLERANCE = 1.0e-9

# ---------------------------------------------------------------------------
# the audit grid (work order section 4)
# ---------------------------------------------------------------------------

# Slot-subset ensembles: fixed geometric subsets of the SAME window's slots,
# scored against that window's governing last-bin target (subset {49} IS the
# sealed baseline row).  ``source_selected_best_slot`` is filled per budget at
# audit time from the within-6 surface only (the single allowed selection).
FIXED_SUBSETS: dict[str, tuple[int, ...]] = {
    "slot49": (49,),
    "last5_s45_49": tuple(range(45, 50)),
    "last10_s40_49": tuple(range(40, 50)),
    "all50": tuple(range(50)),
}
AVERAGING_SUBSETS = ("last5_s45_49", "last10_s40_49", "all50")
SOURCE_SELECTED_SUBSET = "source_selected_best_slot"
SOURCE_SURFACE = "within"

# "near the tail" for the monotonicity leg of reading (a): the last 10 slots.
TAIL_LO = 40
# reading (b): a fixed subset must beat slot 49 on >= 80% of sessions in BOTH
# surfaces (evaluated per budget; see audit.readings for the full rule).
STABLE_SUBSET_SESSION_FRACTION = 0.8

# ---------------------------------------------------------------------------
# latency (work order section 4.5): the bin size is READ from the frozen
# loader call, never guessed.  Provenance is re-verified mechanically at
# materialization time by inspecting the frozen source text.
# ---------------------------------------------------------------------------

BIN_SIZE_MS = 20
BIN_SIZE_PROVENANCE = [
    {
        "file": "src/calibration_gap_v1/p4_stream_stats.py",
        "line": 400,
        "text": "bin_size_ms=20,",
        "role": "the frozen deployment session parse (materialize_session)",
    },
    {
        "file": "src/calibration_gap_v1/z1_oracle_cells.py",
        "line": 268,
        "text": "bin_size_ms=20,",
        "role": "the reviewed Z1 harness parse of the same loader",
    },
    {
        "file": "sua_exploration/mc_maze/multisession_datamodule.py",
        "line": 474,
        "text": "bin_size_s = bin_size_ms / 1000.0",
        "role": "the frozen DANDI 000688 loader bin-edge construction",
    },
]
ZERO_LAG_BAN = (
    "the phrase 'zero-lag' is banned: trajalign(K) pools windows ending at "
    "t..t+K-1 and therefore incurs K-1 bins of output delay"
)

# ---------------------------------------------------------------------------
# mandated caveats (guidance section 12; work order section 4.2)
# ---------------------------------------------------------------------------

COMMON_MODE_CAVEAT = (
    "residual correlations measured here are pairwise common-mode quantities; "
    "a correlation of rho (e.g. the prior rho_group=0.9171 measured on "
    "per-bin velocity errors of disjoint UNIT groups on 6 sub-C sessions at "
    "M4) must NEVER be phrased as '92% common-mode variance' — that claim "
    "requires a hierarchical variance decomposition across sessions, bins and "
    "groups which this audit does not perform"
)
NO_TRAINING_CAVEAT = (
    "this audit cannot authorize a training run by itself (guidance section "
    "5.2); any source-selected slot ensemble is a candidate for a future work "
    "order, not a deployment change"
)

# ---------------------------------------------------------------------------
# process boundaries
# ---------------------------------------------------------------------------

RUNTIME_BOUND_SECONDS = 1800.0  # work order section 6: <= 30 min total CPU
CLOSURE_PATTERNS = (
    "src/slot_audit_v1/*.py",
    "scripts/run_slot_audit_v1.py",
    "tests/test_slot_audit_v1.py",
    "src/continuity_probe_v1.py",
    "src/calibration_gap_v1/p4_stream_stats.py",
    "src/calibration_gap_v1/z1_oracle_cells.py",
    "src/calibration_gap_v1/ledger.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
)
