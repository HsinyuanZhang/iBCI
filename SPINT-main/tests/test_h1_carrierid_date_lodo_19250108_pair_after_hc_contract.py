"""Static no-target contract for the 19250108 H-C completion watcher.

The shell worker is intentionally not executed here: its only legal runtime is
the remote isolated stage after the pre-existing H-C tmux pane exits.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h1_carrierid_date_lodo_19250108_pair_after_hc.sh"
HC_HANDOFF = ROOT / "scripts/h1_carrierid_date_lodo_hc_after_hs_checked.sh"


def test_hc_handoff_binds_the_existing_dated_launch_receipt() -> None:
    source = HC_HANDOFF.read_text(encoding="utf-8")
    required = (
        'readonly LAUNCH_RECEIPT="$STAGE/pilot_artifacts/h1_carrierid_date_lodo_phase2/'
        'H1_CARRIERID_DATE_LODO_PHASE2_19250108_PAIRED_SOURCE_LAUNCH_RECEIPT_v2.json"'
    )
    assert required in source
    assert "H1_CARRIERID_DATE_LODO_PHASE2_PAIRED_SOURCE_LAUNCH_RECEIPT_v2.json" not in source


def test_watcher_is_exact_pane_wait_then_cpu_no_target_pair_checker() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in source
    assert "readonly STAGE='/home/xinyuan/Work_host/SPINT/h1_date_lodo_5070ti_stage/SPINT-main'" in source
    assert "readonly SESSION='h1_date_lodo_19250108_hc_after_hs_checked'" in source
    assert "tmux list-panes -t \"$SESSION\" -F '#{pane_dead}|#{pane_dead_status}'" in source
    assert "'0|'|'0|0'" in source
    assert "'1|0'" in source
    assert "watched H-C tmux session disappeared" in source
    assert "exited abnormally or reported malformed status" in source
    assert "readonly HS_CHECKPOINT=\"$ART/gpu_runs/hs_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt\"" in source
    assert "readonly HC_CHECKPOINT=\"$ART/gpu_runs/hc_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt\"" in source
    assert "readonly HS_RUNTIME_INIT_PROBE=\"$ART/hs_runtime_init_probe_v2/H1_CARRIERID_DATE_LODO_PHASE2_19250108_HS_RUNTIME_INIT_PROBE_v2.json\"" in source
    assert "[[ -f \"$HS_CHECKPOINT\" ]]" in source
    assert "[[ -f \"$HC_CHECKPOINT\" ]]" in source
    assert "[[ -f \"$HS_RUNTIME_INIT_PROBE\" ]]" in source
    assert source.count("[[ ! -e \"$PAIR_OUTPUT\" && ! -L \"$PAIR_OUTPUT\" ]]") == 2
    assert "CUDA_VISIBLE_DEVICES='' \"$PYTHON\" \"$STAGE/scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py\"" in source
    for required in ("--pair-preflight \"$PAIR_PREFLIGHT\"", "--launch-receipt \"$LAUNCH_RECEIPT\"",
                     "--output \"$PAIR_OUTPUT\"", "--pair", "--hs-checkpoint \"$HS_CHECKPOINT\"",
                     "--hc-checkpoint \"$HC_CHECKPOINT\"", "--hs-runtime-init-probe \"$HS_RUNTIME_INIT_PROBE\""):
        assert required in source


def test_watcher_cannot_start_or_advance_any_target_or_gpu_workflow() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "h1_carrierid_date_lodo_phase2_terminal_preflight.py",
        "h1_carrierid_date_lodo_phase2_terminal_evaluate.py",
        "--execute-target-evaluation",
        "load_outer_date_target_records",
        "data/000954",
        "nvidia-smi",
        "tmux new-session",
        "tmux send-keys",
        "tmux kill-session",
        "CUDA_VISIBLE_DEVICES=0",
    ):
        assert forbidden not in source
